"""Evaluate reasoning-LM on syllogism yes/no via conditional likelihood.

For each held-out record we compute the log-likelihood the model
assigns to the two candidate answer continuations (``ROOT:yes``
vs ``REL:neg`` in English, ``ROOT:ن.ع.م`` vs ``STR:neg:general`` in Arabic) given
the question + CoT prefix. Higher = model's prediction. We then
compare to the gold answer and report accuracy overall, by language,
by validity, and by difficulty.

No sampling, no beam search — this is a clean classification signal
for a first sanity check.

Usage::

    python -m reasoning.train.eval \
        --ckpt  ./reasoning/train/runs/syllog-v0/ckpt.pt \
        --data  ./reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl \
        --vocab ./reasoning/tokenized/vocab-reasoning.json \
        --max   2000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import ids_from_tokens, load_vocab, split_indices
from .model import GPTConfig, TinyGPT


# Candidate answer tokens per language. These must match what the
# tokenizers actually emit for the bare answer strings — verified against
# the tokenized syllogism corpus (Apr 2025 re-audit):
#   en  "yes" -> [BOS, ROOT:yes, EOS]        "no"  -> [BOS, REL:neg, EOS]
#   ar  "نعم" -> [BOS, ROOT:ن.ع.م, EOS]      "لا"   -> [BOS, STR:neg:general, EOS]
# NOTE: LIT:yes and LIT:نعم are NOT emitted by the reasoning tokenizer.
ANSWER_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "en": {
        "yes": ["[BOS]", "ROOT:yes", "[EOS]"],
        "no":  ["[BOS]", "REL:neg",  "[EOS]"],
    },
    "ar": {
        "yes": ["[BOS]", "ROOT:ن.ع.م", "[EOS]"],
        "no":  ["[BOS]", "STR:neg:general",  "[EOS]"],
    },
}


def _prefix_tokens(rec: dict) -> list[str]:
    seq: list[str] = []
    seq.extend(rec["question_tokens"]["reasoning"])
    for step in rec["cot_tokens"]:
        seq.extend(step["reasoning"])
    return seq


def _score(model: TinyGPT, prefix_ids: list[int], cand_ids: list[int],
           max_len: int, device: torch.device) -> float:
    """Sum of log-probs of the candidate tokens given the prefix."""
    seq = prefix_ids + cand_ids
    if len(seq) > max_len:
        seq = seq[-max_len:]
        # After truncation, work out how many of the tail tokens are candidates.
        n_cand = min(len(cand_ids), len(seq))
    else:
        n_cand = len(cand_ids)
    ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids)               # (1, T, V)
        logp = F.log_softmax(logits, dim=-1)
    # Score tokens at positions [T-n_cand, T); they are predicted by positions
    # [T-n_cand-1, T-1).
    total = 0.0
    T = ids.size(1)
    for off in range(n_cand):
        t = T - n_cand + off
        prev = t - 1
        if prev < 0:
            continue
        total += logp[0, prev, ids[0, t]].item()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--max", type=int, default=2000,
                    help="Cap eval records (uses held-out split, then this cap).")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── Load model ────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device).eval()
    max_len = cfg.max_len

    vocab = load_vocab(args.vocab)
    unk = vocab["[UNK]"]

    # ── Pick held-out split ───────────────────────────────
    records = []
    with args.data.open("r", encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    _, val_idx = split_indices(len(records), val_frac=args.val_frac, seed=args.seed)
    val_records = [records[i] for i in val_idx[: args.max]]
    print(f"eval on {len(val_records)} held-out records (device={device})")

    # Precompute candidate id lists per language.
    cand_ids_by_lang: dict[str, dict[str, list[int]]] = {}
    for lang, cands in ANSWER_CANDIDATES.items():
        cand_ids_by_lang[lang] = {
            k: ids_from_tokens(v, vocab) for k, v in cands.items()
        }

    correct = 0
    totals: Counter = Counter()
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])       # [correct, total]
    by_valid: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for rec in val_records:
        lang = rec["lang"]
        if lang not in cand_ids_by_lang:
            continue
        prefix_ids = ids_from_tokens(_prefix_tokens(rec), vocab)

        scores = {
            label: _score(model, prefix_ids, cand_ids, max_len, device)
            for label, cand_ids in cand_ids_by_lang[lang].items()
        }
        pred_label = max(scores, key=lambda k: scores[k])
        # Gold
        gold_answer = str(rec["answer"]).strip()
        gold_label = (
            "yes" if gold_answer in {"yes", "نعم"}
            else "no" if gold_answer in {"no", "لا"}
            else None
        )
        if gold_label is None:
            continue

        is_valid_str = "valid" if gold_label == "yes" else "invalid"
        diff = rec.get("meta", {}).get("difficulty", "?")

        totals["total"] += 1
        ok = int(pred_label == gold_label)
        correct += ok
        by_lang[lang][0] += ok; by_lang[lang][1] += 1
        by_valid[is_valid_str][0] += ok; by_valid[is_valid_str][1] += 1
        by_diff[diff][0] += ok; by_diff[diff][1] += 1

    def _fmt(d: dict) -> dict:
        return {k: {"acc": round(v[0] / v[1], 4) if v[1] else 0.0,
                    "n": v[1]} for k, v in d.items()}

    summary = {
        "accuracy": round(correct / max(1, totals["total"]), 4),
        "n": totals["total"],
        "by_lang": _fmt(by_lang),
        "by_validity": _fmt(by_valid),
        "by_difficulty": _fmt(by_diff),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = args.ckpt.with_name("eval.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
