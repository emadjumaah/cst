"""Evaluate reasoning signal for verifier with strict controls.

Controls:
- full: question + CoT + answer
- question_only: question + answer
- shuffled_cot: question + CoT(from another sample in same language) + answer

Proof gate passes when full beats both controls by required margins.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .common import (
    compose_reasoning_tokens,
    extract_token_segments,
    ids_from_tokens,
    load_jsonl,
    load_vocab_file,
    pick_device,
    record_gold_label,
    record_lang,
    record_schema,
)
from .model import load_checkpoint


class InferenceDataset(Dataset[list[int]]):
    def __init__(self, seqs: list[list[int]]) -> None:
        self.seqs = seqs

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> list[int]:
        return self.seqs[idx]


def truncate_ids(ids: list[int], max_len: int) -> list[int]:
    if len(ids) <= max_len:
        return ids
    head = max_len // 2
    tail = max_len - head
    return ids[:head] + ids[-tail:]


def make_collate_fn(pad_id: int):
    def collate(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(ids) for ids in batch)
        ids_tensor = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, ids in enumerate(batch):
            n = len(ids)
            ids_tensor[i, :n] = torch.tensor(ids, dtype=torch.long)
            mask[i, :n] = True
        return ids_tensor, mask

    return collate


def score_sequences(
    *,
    seqs: list[list[int]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    pad_id: int,
) -> list[float]:
    ds = InferenceDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate_fn(pad_id), num_workers=0)

    probs: list[float] = []
    model.eval()
    with torch.no_grad():
        for ids, mask in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            logits = model(ids, mask)
            p = torch.sigmoid(logits).detach().cpu().tolist()
            probs.extend(float(v) for v in p)
    return probs


def collect_source_examples(rows: list[dict[str, Any]], *, view: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in rows:
        if record_schema(rec) != "reasoning":
            continue
        answer_label = record_gold_label(rec)
        if answer_label not in {"yes", "no"}:
            continue

        q, cot, a = extract_token_segments(rec, view=view)
        if not q or not a:
            continue

        out.append(
            {
                "id": str(rec.get("id", f"src-{len(out)}")),
                "lang": record_lang(rec),
                "answer_label": answer_label,
                "q": q,
                "cot": cot,
                "a": a,
            }
        )
    return out


def maybe_balance_by_answer(examples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_label[str(ex["answer_label"])].append(ex)

    if len(by_label.get("yes", [])) == 0 or len(by_label.get("no", [])) == 0:
        return examples

    n = min(len(by_label["yes"]), len(by_label["no"]))
    rng.shuffle(by_label["yes"])
    rng.shuffle(by_label["no"])
    balanced = by_label["yes"][:n] + by_label["no"][:n]
    rng.shuffle(balanced)
    return balanced


def evaluate_signal(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    model, cfg, pad_id_ckpt, blob = load_checkpoint(args.checkpoint, device=device)

    vocab_path = args.vocab
    if vocab_path is None:
        if "vocab_path" not in blob:
            raise ValueError("--vocab is required because checkpoint does not contain vocab_path")
        vocab_path = Path(str(blob["vocab_path"]))

    vocab = load_vocab_file(vocab_path)
    pad_id = vocab.get("[PAD]", pad_id_ckpt)

    rows = load_jsonl(args.input)
    examples = collect_source_examples(rows, view=args.view)

    if args.balance_by_answer:
        examples = maybe_balance_by_answer(examples, seed=args.seed)

    if args.max_examples > 0 and len(examples) > args.max_examples:
        rnd = random.Random(args.seed)
        rnd.shuffle(examples)
        examples = examples[: args.max_examples]

    if len(examples) < 4:
        raise ValueError(f"Not enough eligible examples for evaluation: {len(examples)}")

    by_lang_indices: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_lang_indices[ex["lang"]].append(i)

    def sample_other(idx: int, lang: str, rng: random.Random) -> int:
        cands = by_lang_indices[lang]
        if not cands:
            return idx
        if len(cands) == 1:
            return cands[0]
        for _ in range(8):
            j = rng.choice(cands)
            if j != idx:
                return j
        return cands[0]

    full_tokens: list[list[str]] = []
    qonly_tokens: list[list[str]] = []
    shuf_tokens: list[list[str]] = []

    rng = random.Random(args.seed + 7)
    for i, ex in enumerate(examples):
        q = ex["q"]
        cot = ex["cot"]
        a = ex["a"]
        j = sample_other(i, ex["lang"], rng)
        cot_other = examples[j]["cot"]

        full_tokens.append(compose_reasoning_tokens(q, cot, a))
        qonly_tokens.append(q + a)
        shuf_tokens.append(compose_reasoning_tokens(q, cot_other, a))

    max_len = int(getattr(cfg, "max_len", 256))
    full_ids = [truncate_ids(ids_from_tokens(toks, vocab), max_len=max_len) for toks in full_tokens]
    qonly_ids = [truncate_ids(ids_from_tokens(toks, vocab), max_len=max_len) for toks in qonly_tokens]
    shuf_ids = [truncate_ids(ids_from_tokens(toks, vocab), max_len=max_len) for toks in shuf_tokens]

    p_full = score_sequences(seqs=full_ids, model=model, device=device, batch_size=args.batch_size, pad_id=pad_id)
    p_qonly = score_sequences(seqs=qonly_ids, model=model, device=device, batch_size=args.batch_size, pad_id=pad_id)
    p_shuf = score_sequences(seqs=shuf_ids, model=model, device=device, batch_size=args.batch_size, pad_id=pad_id)

    mean_full = float(mean(p_full))
    mean_qonly = float(mean(p_qonly))
    mean_shuf = float(mean(p_shuf))

    gap_q = mean_full - mean_qonly
    gap_s = mean_full - mean_shuf

    wins_vs_q = sum(1 for a, b in zip(p_full, p_qonly) if a > b) / len(p_full)
    wins_vs_s = sum(1 for a, b in zip(p_full, p_shuf) if a > b) / len(p_full)

    proof_pass = (
        mean_full >= args.min_full_mean
        and gap_q >= args.min_gap_qonly
        and gap_s >= args.min_gap_shuffled
        and wins_vs_q >= args.min_win_rate
        and wins_vs_s >= args.min_win_rate
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "token_view": args.view,
        "vocab": str(vocab_path),
        "device": str(device),
        "n_examples": len(examples),
        "by_lang": dict(Counter(ex["lang"] for ex in examples)),
        "by_answer_label": dict(Counter(ex["answer_label"] for ex in examples)),
        "means": {
            "full": round(mean_full, 4),
            "question_only": round(mean_qonly, 4),
            "shuffled_cot": round(mean_shuf, 4),
        },
        "gaps": {
            "full_minus_question_only": round(gap_q, 4),
            "full_minus_shuffled_cot": round(gap_s, 4),
        },
        "wins": {
            "full_gt_question_only": round(wins_vs_q, 4),
            "full_gt_shuffled_cot": round(wins_vs_s, 4),
        },
        "thresholds": {
            "min_full_mean": args.min_full_mean,
            "min_gap_qonly": args.min_gap_qonly,
            "min_gap_shuffled": args.min_gap_shuffled,
            "min_win_rate": args.min_win_rate,
        },
        "reasoning_proof_pass": bool(proof_pass),
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True, help="Original tokenized reasoning JSONL")
    ap.add_argument("--vocab", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--view", type=str, default="reasoning", choices=["reasoning", "default"])

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--balance-by-answer", action="store_true", default=False)

    ap.add_argument("--min-full-mean", type=float, default=0.60)
    ap.add_argument("--min-gap-qonly", type=float, default=0.03)
    ap.add_argument("--min-gap-shuffled", type=float, default=0.03)
    ap.add_argument("--min-win-rate", type=float, default=0.60)

    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    evaluate_signal(parse_args())
