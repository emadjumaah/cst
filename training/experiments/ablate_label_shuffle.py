"""
Ablation A.2: Label-shuffle on CST vocabulary.

Motivation
----------
Question raised by the critical review: how much of CST's benefit comes from
the *specific* semantic-field labels (ROOT:motion, ROOT:existence, …) versus
from merely having a *fine-grained, content-aware* vocabulary? This ablation
controls for the latter.

Procedure
---------
1. Load the trained CST vocabulary (e.g. cst-8k train-99963).
2. Build a permutation π over the set of structured token *identifiers*
   (IDs whose token string starts with any of CST's structural prefixes
   such as ROOT:, CMP:, STR:, REL:). Literal (LIT:) and special tokens
   are left untouched so character coverage is preserved.
3. Apply π to every token stream in the tokenized .jsonl — each ROOT/CMP/…
   ID is replaced with π(id). This scrambles *which* semantic label each
   surface form maps to, while preserving segmentation and vocabulary size.
4. Train a fresh GPT-2 with identical config and compare BPC to the
   un-shuffled CST baseline on the same validation set.

Interpretation
--------------
- If the shuffled BPC is close to the un-shuffled CST BPC, the win comes
  from segmentation granularity, not from label coherence. This would be
  a material negative result.
- If the shuffled BPC regresses sharply toward the BPE baseline, the
  specific label assignment is carrying the signal.

Usage
-----
    python training/experiments/ablate_label_shuffle.py \\
        --data /content/cst-8k-train-99963.jsonl \\
        --vocab /content/cst-8k-train-99963-vocab.json \\
        --out /content/results_label_shuffle.json \\
        --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments._core import (  # noqa: E402
    DEFAULTS,
    load_jsonl,
    split_train_val,
    train_and_eval,
)


STRUCTURED_PREFIXES = ("ROOT:", "CMP:", "STR:", "REL:", "NER:", "FIELD:")


def load_vocab_as_id_to_token(vocab_path: str) -> dict[int, str]:
    with open(vocab_path) as f:
        vocab = json.load(f)
    id_to_tok: dict[int, str] = {}
    if isinstance(vocab, dict):
        # Could be {token: id} or {id: token}
        for k, v in vocab.items():
            if isinstance(v, int):
                id_to_tok[v] = k
            else:
                id_to_tok[int(k)] = v if isinstance(v, str) else v["token"]
    elif isinstance(vocab, list):
        for entry in vocab:
            id_to_tok[entry["id"]] = entry["token"]
    else:
        raise ValueError("Unrecognized vocab format")
    return id_to_tok


def build_permutation(id_to_tok: dict[int, str], rng: random.Random) -> dict[int, int]:
    structured_ids = [
        tid for tid, tok in id_to_tok.items()
        if any(tok.startswith(p) for p in STRUCTURED_PREFIXES)
    ]
    shuffled = structured_ids[:]
    rng.shuffle(shuffled)
    perm = {a: b for a, b in zip(structured_ids, shuffled)}
    # Identity for everything else (LIT, specials).
    for tid in id_to_tok:
        perm.setdefault(tid, tid)
    return perm


def apply_permutation(ids_list, perm: dict[int, int]):
    return [[perm.get(t, t) for t in seq] for seq in ids_list]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CST .jsonl")
    p.add_argument("--vocab", required=True, help="Path to CST vocab JSON")
    p.add_argument("--out", required=True, help="Output results JSON")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--run-baseline", action="store_true",
                   help="Also train the un-shuffled CST baseline for direct comparison.")
    args = p.parse_args()

    ids_list, char_counts = load_jsonl(args.data, DEFAULTS["max_len"])
    id_to_tok = load_vocab_as_id_to_token(args.vocab)
    vocab_size = max(id_to_tok) + 1
    tr_ids, tr_ch, va_ids, va_ch = split_train_val(ids_list, char_counts, DEFAULTS["val_ratio"])

    n_structured = sum(
        1 for tok in id_to_tok.values()
        if any(tok.startswith(pfx) for pfx in STRUCTURED_PREFIXES)
    )
    print(f"  Vocab: {vocab_size:,} total, {n_structured:,} structured (will be shuffled)")

    results: list[dict] = []

    for seed in args.seeds:
        if args.run_baseline:
            base = train_and_eval(
                name="CST-baseline",
                train_ids=tr_ids, train_chars=tr_ch,
                val_ids=va_ids, val_chars=va_ch,
                vocab_size=vocab_size, seed=seed, epochs=args.epochs,
            )
            base["variant"] = "baseline"
            results.append(base)

        rng = random.Random(seed)
        perm = build_permutation(id_to_tok, rng)
        shuffled_tr = apply_permutation(tr_ids, perm)
        shuffled_va = apply_permutation(va_ids, perm)
        shuf = train_and_eval(
            name="CST-label-shuffled",
            train_ids=shuffled_tr, train_chars=tr_ch,
            val_ids=shuffled_va, val_chars=va_ch,
            vocab_size=vocab_size, seed=seed, epochs=args.epochs,
        )
        shuf["variant"] = "label_shuffled"
        results.append(shuf)

        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*70}\n  Label-shuffle ablation summary\n{'='*70}")
    from collections import defaultdict
    by_variant: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r["best_val_bpc"])
    for variant, bpcs in by_variant.items():
        mean = sum(bpcs) / len(bpcs)
        var = sum((x - mean) ** 2 for x in bpcs) / max(len(bpcs) - 1, 1)
        print(f"  {variant:<20} BPC = {mean:.4f} ± {var**0.5:.4f}  (n={len(bpcs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
