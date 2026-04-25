"""Paired OOD evaluation for CST vs BPE on the same yes/no split.

This script compares two checkpoints with the same conditional-likelihood
decision rule used by the existing eval harness:

- CST model: tokenized reasoning records (question_tokens/cot_tokens)
- BPE model: raw records (question/cot) + a saved tokenizer JSON

The goal is reproducible paired deltas on OOD without hand-editing notebook
cells for each run.

Usage:
    python -m reasoning.eval.paired_ood_eval \
        --cst-ckpt reasoning/train/runs/<run>/ckpt.pt \
        --cst-vocab reasoning/tokenized/vocab-reasoning.json \
        --cst-data reasoning/eval/holdout_tokenized/syllogisms_yesno.tokenized.jsonl \
        --bpe-ckpt reasoning/train/runs/<bpe_run>/bpe_ckpt.pt \
        --bpe-data reasoning/eval/holdout/syllogisms_yesno.jsonl \
        --out reasoning/eval/holdout/paired_ood_eval.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from reasoning.train.dataset import ids_from_tokens, load_vocab
from reasoning.train.model import GPTConfig, TinyGPT


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _score(
    model: TinyGPT,
    prefix_ids: list[int],
    cand_ids: list[int],
    max_len: int,
    device: torch.device,
) -> float:
    seq = prefix_ids + cand_ids
    if len(seq) > max_len:
        seq = seq[-max_len:]
    n_cand = min(len(cand_ids), len(seq))
    ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids)
        logp = F.log_softmax(logits, dim=-1)
    t_total = ids.size(1)
    total = 0.0
    for off in range(n_cand):
        t = t_total - n_cand + off
        prev = t - 1
        if prev < 0:
            continue
        total += logp[0, prev, ids[0, t]].item()
    return total


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gold_label(rec: dict[str, Any]) -> str | None:
    raw = str(rec.get("answer", rec.get("meta", {}).get("answer", ""))).strip().lower()
    if raw in {"yes", "نعم", "valid", "true"}:
        return "yes"
    if raw in {"no", "لا", "invalid", "false"}:
        return "no"
    return None


def _lang(rec: dict[str, Any]) -> str:
    return str(rec.get("lang", rec.get("meta", {}).get("lang", "en")))


def _summary(accumulator: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for k, (correct, total) in accumulator.items():
        out[k] = {
            "acc": round(correct / total, 4) if total else 0.0,
            "n": total,
        }
    return out


def _evaluate_cst(
    ckpt_path: Path,
    vocab_path: Path,
    data_path: Path,
    max_examples: int,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    vocab = load_vocab(vocab_path)
    cand_tokens = {
        "en": {
            "yes": ["[BOS]", "ROOT:yes", "[EOS]"],
            "no": ["[BOS]", "REL:neg", "[EOS]"],
        },
        "ar": {
            "yes": ["[BOS]", "ROOT:ن.ع.م", "[EOS]"],
            "no": ["[BOS]", "STR:neg:general", "[EOS]"],
        },
    }
    cand_ids = {
        lg: {k: ids_from_tokens(v, vocab) for k, v in cands.items()}
        for lg, cands in cand_tokens.items()
    }

    rows = _load_jsonl(data_path)
    if max_examples > 0:
        rows = rows[:max_examples]

    correct = 0
    total = 0
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_validity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_difficulty: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for rec in rows:
        lg = _lang(rec)
        gold = _gold_label(rec)
        if gold is None or lg not in cand_ids:
            continue

        if "q_ids" in rec:
            prefix = list(rec.get("q_ids") or [])
            for step in rec.get("cot_ids") or []:
                prefix.extend(step)
        else:
            prefix_tokens = list((rec.get("question_tokens") or {}).get("reasoning") or [])
            for step in rec.get("cot_tokens") or []:
                prefix_tokens.extend(step.get("reasoning") or [])
            prefix = ids_from_tokens(prefix_tokens, vocab)

        scores = {
            lbl: _score(model, prefix, cids, cfg.max_len, device)
            for lbl, cids in cand_ids[lg].items()
        }
        pred = max(scores, key=scores.get)
        ok = int(pred == gold)

        correct += ok
        total += 1
        by_lang[lg][0] += ok
        by_lang[lg][1] += 1
        vk = "valid" if gold == "yes" else "invalid"
        by_validity[vk][0] += ok
        by_validity[vk][1] += 1
        diff = str(rec.get("meta", {}).get("difficulty", "?"))
        by_difficulty[diff][0] += ok
        by_difficulty[diff][1] += 1

    return {
        "accuracy": round(correct / max(1, total), 4),
        "n": total,
        "by_lang": _summary(by_lang),
        "by_validity": _summary(by_validity),
        "by_difficulty": _summary(by_difficulty),
    }


def _resolve_bpe_tokenizer_path(ckpt_path: Path, ckpt_obj: dict[str, Any], cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    tok = ckpt_obj.get("tokenizer_json")
    if not tok:
        raise SystemExit("BPE ckpt missing tokenizer_json; pass --bpe-tokenizer")
    tok_path = Path(str(tok))
    if tok_path.exists():
        return tok_path
    # Allow relative path in ckpt metadata.
    candidate = ckpt_path.parent / tok_path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"BPE tokenizer path not found: {tok_path}")


def _evaluate_bpe(
    ckpt_path: Path,
    data_path: Path,
    max_examples: int,
    device: torch.device,
    bpe_tokenizer_override: str | None,
) -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - import guard for runtime
        raise SystemExit("Missing dependency: tokenizers. Install with `pip install tokenizers`") from exc

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    tok_path = _resolve_bpe_tokenizer_path(ckpt_path, ckpt, bpe_tokenizer_override)
    tok = Tokenizer.from_file(str(tok_path))

    pad_id = tok.token_to_id("[PAD]")
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    if any(v is None for v in (pad_id, bos_id, eos_id)):
        raise SystemExit("BPE tokenizer is missing required specials [PAD]/[BOS]/[EOS]")

    def enc(text: str) -> list[int]:
        return tok.encode(text).ids

    cand_text = {
        "en": {"yes": "yes", "no": "no"},
        "ar": {"yes": "نعم", "no": "لا"},
    }
    cand_ids = {
        lg: {k: [bos_id] + enc(v) + [eos_id] for k, v in mp.items()}
        for lg, mp in cand_text.items()
    }

    rows = _load_jsonl(data_path)
    if max_examples > 0:
        rows = rows[:max_examples]

    correct = 0
    total = 0
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_validity: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_difficulty: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for rec in rows:
        lg = _lang(rec)
        gold = _gold_label(rec)
        if gold is None or lg not in cand_ids:
            continue

        q = str(rec.get("question", ""))
        cot = [str(s) for s in (rec.get("cot") or [])]
        prefix = [bos_id] + enc(q) + [eos_id]
        for step in cot:
            prefix += [bos_id] + enc(step) + [eos_id]

        scores = {
            lbl: _score(model, prefix, cids, cfg.max_len, device)
            for lbl, cids in cand_ids[lg].items()
        }
        pred = max(scores, key=scores.get)
        ok = int(pred == gold)

        correct += ok
        total += 1
        by_lang[lg][0] += ok
        by_lang[lg][1] += 1
        vk = "valid" if gold == "yes" else "invalid"
        by_validity[vk][0] += ok
        by_validity[vk][1] += 1
        diff = str(rec.get("meta", {}).get("difficulty", "?"))
        by_difficulty[diff][0] += ok
        by_difficulty[diff][1] += 1

    return {
        "accuracy": round(correct / max(1, total), 4),
        "n": total,
        "by_lang": _summary(by_lang),
        "by_validity": _summary(by_validity),
        "by_difficulty": _summary(by_difficulty),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cst-ckpt", type=Path, required=True)
    ap.add_argument("--cst-vocab", type=Path, required=True)
    ap.add_argument("--cst-data", type=Path, required=True,
                    help="Tokenized JSONL with reasoning token fields.")
    ap.add_argument("--bpe-ckpt", type=Path, required=True)
    ap.add_argument("--bpe-data", type=Path, required=True,
                    help="Raw JSONL with question/cot/answer.")
    ap.add_argument("--bpe-tokenizer", type=str, default=None,
                    help="Optional tokenizer JSON path override for BPE eval.")
    ap.add_argument("--max", type=int, default=100000,
                    help="Max examples per side; <=0 means all")
    ap.add_argument("--out", type=Path, default=Path("reasoning/eval/paired_ood_eval.json"))
    args = ap.parse_args()

    device = _pick_device()

    cst = _evaluate_cst(
        ckpt_path=args.cst_ckpt,
        vocab_path=args.cst_vocab,
        data_path=args.cst_data,
        max_examples=args.max,
        device=device,
    )
    bpe = _evaluate_bpe(
        ckpt_path=args.bpe_ckpt,
        data_path=args.bpe_data,
        max_examples=args.max,
        device=device,
        bpe_tokenizer_override=args.bpe_tokenizer,
    )

    result = {
        "device": str(device),
        "cst": cst,
        "bpe": bpe,
        "comparison": {
            "delta_ood_accuracy_vs_bpe": round(cst["accuracy"] - bpe["accuracy"], 4),
            "paired_n_match": cst["n"] == bpe["n"],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
