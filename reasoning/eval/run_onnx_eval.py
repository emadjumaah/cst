"""
run_onnx_eval.py — local sanity test for trained CST-reasoning ONNX models.

Mirrors the Colab eval loop: loads vocab-reasoning.json + an ONNX model,
samples N held-out syllogism records from the tokenized JSONL, scores the
4 bilingual yes/no candidate continuations, and reports accuracy.

Usage:
    python reasoning/eval/run_onnx_eval.py \
        --onnx reasoning/runs/colab-cst-v0.2-small/model_logic_int8.onnx \
        --n 500
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "reasoning" / "tokenized" / "vocab-reasoning.json"
SYLL_JSONL = ROOT / "reasoning" / "tokenized" / "stage-2b-syllogisms.tokenized.jsonl"

# Verified Apr 2025: reasoning tokenizer emits ROOT:yes / ROOT:ن.ع.م, not LIT:*
CAND_TOKENS = {
    "en": {"yes": ["[BOS]", "ROOT:yes", "[EOS]"], "no": ["[BOS]", "REL:neg", "[EOS]"]},
    "ar": {"yes": ["[BOS]", "ROOT:ن.ع.م", "[EOS]"], "no": ["[BOS]", "STR:neg:general", "[EOS]"]},
}


def load_vocab(path: Path) -> tuple[dict[str, int], int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Colab script emitted {"token_to_id": {...}} or a flat dict.
    if isinstance(data, dict) and "token_to_id" in data:
        t2i = data["token_to_id"]
    else:
        t2i = data
    t2i = {k: int(v) for k, v in t2i.items()}
    pad = t2i.get("[PAD]", 0)
    return t2i, pad, len(t2i)


def to_ids(toks: list[str], t2i: dict[str, int]) -> list[int]:
    unk = t2i.get("[UNK]", 0)
    return [t2i.get(t, unk) for t in toks]


def infer_max_len(sess: ort.InferenceSession) -> int:
    shape = sess.get_inputs()[0].shape  # [batch, seq]
    seq = shape[1]
    return int(seq) if isinstance(seq, int) else 256


def score_candidate(
    sess: ort.InferenceSession,
    prefix_ids: list[int],
    cand_ids: list[int],
    pad: int,
    max_len: int,
) -> float:
    """Sum log-prob of each candidate token given its prefix (teacher forcing)."""
    seq = prefix_ids + cand_ids
    if len(seq) > max_len:
        seq = seq[-max_len:]
    n = min(len(cand_ids), len(seq))
    padded = np.full((1, max_len), pad, dtype=np.int64)
    padded[0, : len(seq)] = seq
    out = sess.run(None, {sess.get_inputs()[0].name: padded})[0]  # [1, max_len, V]
    # log-softmax over last axis
    logits = out[0]  # [max_len, V]
    # numeric-stable logsumexp
    m = logits.max(axis=-1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
    logp = logits - lse
    T = len(seq)
    total = 0.0
    for off in range(n):
        t = T - n + off
        prev = t - 1
        if prev < 0:
            continue
        total += float(logp[prev, seq[t]])
    return total


def prefix_from_record(rec: dict) -> list[str]:
    """Rebuild training-format prefix (without final answer segment).

    Training format: [BOS] q [EOS] [BOS] cot1 [EOS] ... [BOS] cotN [EOS] [BOS] ans [EOS]
    We stop before the final [BOS] ans [EOS].
    """
    q = (rec.get("question_tokens") or {}).get("reasoning") or []
    cot = rec.get("cot_tokens") or []
    toks: list[str] = []
    toks += ["[BOS]", *q, "[EOS]"]
    for step in cot:
        s = step.get("reasoning") or []
        if s:
            toks += ["[BOS]", *s, "[EOS]"]
    return toks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t2i, pad, vocab_size = load_vocab(VOCAB_PATH)
    print(f"vocab: {vocab_size} tokens (PAD={pad})")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    max_len = infer_max_len(sess)
    print(f"onnx:  {args.onnx.name}  max_len={max_len}")

    cand_ids = {
        lang: {k: to_ids(v, t2i) for k, v in lc.items()}
        for lang, lc in CAND_TOKENS.items()
    }

    # Sample records
    rng = random.Random(args.seed)
    all_recs: list[dict] = []
    with SYLL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            all_recs.append(json.loads(line))
    rng.shuffle(all_recs)
    recs = all_recs[: args.n]

    correct = total = 0
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_val: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    t0 = time.time()
    for i, rec in enumerate(recs):
        lang = rec.get("lang", rec.get("meta", {}).get("lang", "en"))
        gold_raw = (rec.get("answer") or rec.get("meta", {}).get("answer") or "").strip()
        gold = "yes" if gold_raw in {"yes", "نعم"} else "no" if gold_raw in {"no", "لا"} else None
        if gold is None or lang not in cand_ids:
            continue
        diff = (rec.get("meta") or {}).get("difficulty", "?")
        prefix_toks = prefix_from_record(rec)
        prefix_ids = to_ids(prefix_toks, t2i)
        scores = {
            k: score_candidate(sess, prefix_ids, v, pad, max_len)
            for k, v in cand_ids[lang].items()
        }
        pred = max(scores, key=scores.get)
        ok = int(pred == gold)
        correct += ok
        total += 1
        by_lang[lang][0] += ok
        by_lang[lang][1] += 1
        vkey = "valid" if gold == "yes" else "invalid"
        by_val[vkey][0] += ok
        by_val[vkey][1] += 1
        by_diff[diff][0] += ok
        by_diff[diff][1] += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(recs)}]  acc={correct/max(1,total):.3f}")

    elapsed = time.time() - t0
    summary = {
        "model_path": str(args.onnx),
        "model_name": args.onnx.stem,
        "size_bytes": args.onnx.stat().st_size,
        "size_mb": round(args.onnx.stat().st_size / 1e6, 3),
        "max_len": max_len,
        "n": total,
        "accuracy": round(correct / max(1, total), 4),
        "by_lang":     {k: {"acc": round(v[0]/max(1,v[1]), 4), "n": v[1]} for k, v in by_lang.items()},
        "by_validity": {k: {"acc": round(v[0]/max(1,v[1]), 4), "n": v[1]} for k, v in by_val.items()},
        "by_difficulty": {k: {"acc": round(v[0]/max(1,v[1]), 4), "n": v[1]} for k, v in by_diff.items()},
        "ms_per_example": round(elapsed / max(1, total) * 1000, 2),
        "elapsed_s": round(elapsed, 2),
        "seed": args.seed,
    }
    print("─" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out_path = args.onnx.with_suffix(".eval.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
