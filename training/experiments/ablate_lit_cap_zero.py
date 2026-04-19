"""
Ablation A.3: LIT-cap-zero.

Motivation
----------
Critical review flagged that a substantial fraction of CST tokens resolve to
literal fallbacks (LIT:word). If those LITs are what's driving the BPC win
(because LIT is effectively word-level), then the *structured* portion of
CST is not actually helping.

Procedure
---------
1. Re-tokenize the corpus (or post-process an existing .jsonl) with a
   modified vocabulary that drops every LIT token. Anything that would
   have hit LIT is split into SentencePiece BPE subpieces instead (the
   byproduct BPE model trained on the same corpus).
2. Character count per sentence is preserved; segmentation is ~CST for
   structured content, ~BPE for the residual.
3. Train and evaluate as usual.

This script assumes you have already produced:
  - the CST .jsonl + vocab (with LIT entries)
  - a SentencePiece BPE model over the same corpus (we fall back on its
    output .jsonl rather than rebuilding from raw text, to keep this script
    simple).

Usage
-----
    python training/experiments/ablate_lit_cap_zero.py \\
        --cst-data /content/cst-8k-train-99963.jsonl \\
        --cst-vocab /content/cst-8k-train-99963-vocab.json \\
        --spm-data /content/spm-train-99963-8k.jsonl \\
        --spm-vocab /content/spm-train-99963-8k-vocab.json \\
        --out /content/results_lit_cap_zero.json \\
        --seeds 0 1 2

Limitation
----------
Perfect LIT-cap-zero would re-tokenize from raw text; here we replace each
CST LIT *token* with the SPM ids that SPM used for the same sentence. This
is an approximation because sentence-level SPM ids don't give a token-level
alignment. For correctness we rebuild streams by concatenating per-sentence
structured CST tokens with a *suffix of SPM tokens for the same sentence*
whenever the CST sequence contained any LIT. A stricter variant requires
the raw corpus; see TODO at the bottom.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments._core import (  # noqa: E402
    DEFAULTS,
    load_jsonl,
    split_train_val,
    train_and_eval,
)
from experiments.ablate_label_shuffle import load_vocab_as_id_to_token  # noqa: E402


def build_lit_id_set(id_to_tok: dict[int, str]) -> set[int]:
    return {tid for tid, tok in id_to_tok.items() if tok.startswith("LIT:")}


def stream_without_lit(
    cst_stream: list[int],
    cst_lit_ids: set[int],
    spm_stream: list[int],
    spm_offset: int,
) -> list[int]:
    """Approximation: keep every non-LIT CST id; when any LIT id is present,
    append (offset-shifted) SPM ids for the same sentence. Offset shifts SPM
    ids into a disjoint range of the shared vocabulary.
    """
    kept = [t for t in cst_stream if t not in cst_lit_ids]
    had_lit = any(t in cst_lit_ids for t in cst_stream)
    if had_lit:
        kept.extend(s + spm_offset for s in spm_stream)
    return kept


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cst-data", required=True)
    p.add_argument("--cst-vocab", required=True)
    p.add_argument("--spm-data", required=True)
    p.add_argument("--spm-vocab", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    args = p.parse_args()

    cst_ids_all, cst_chars = load_jsonl(args.cst_data, DEFAULTS["max_len"])
    spm_ids_all, _spm_chars = load_jsonl(args.spm_data, DEFAULTS["max_len"])
    if len(cst_ids_all) != len(spm_ids_all):
        print(f"  WARN: cst stream length {len(cst_ids_all)} != spm {len(spm_ids_all)}; "
              f"truncating to min.")
        n = min(len(cst_ids_all), len(spm_ids_all))
        cst_ids_all, cst_chars = cst_ids_all[:n], cst_chars[:n]
        spm_ids_all = spm_ids_all[:n]

    cst_id2tok = load_vocab_as_id_to_token(args.cst_vocab)
    spm_id2tok = load_vocab_as_id_to_token(args.spm_vocab)
    lit_ids = build_lit_id_set(cst_id2tok)
    cst_vocab_size = max(cst_id2tok) + 1
    spm_vocab_size = max(spm_id2tok) + 1

    # SPM ids shift to live above CST id range so they don't collide.
    spm_offset = cst_vocab_size
    combined_vocab_size = cst_vocab_size + spm_vocab_size

    print(f"  CST vocab={cst_vocab_size:,}  LIT tokens={len(lit_ids):,}")
    print(f"  SPM vocab={spm_vocab_size:,}  → combined={combined_vocab_size:,}")

    new_streams = [
        stream_without_lit(c, lit_ids, s, spm_offset)
        for c, s in zip(cst_ids_all, spm_ids_all)
    ]
    # Drop anything that got too short after LIT removal.
    filtered = [(ids, ch) for ids, ch in zip(new_streams, cst_chars) if len(ids) >= 4]
    ids_list = [x[0] for x in filtered]
    char_counts = [x[1] for x in filtered]
    print(f"  kept {len(ids_list):,}/{len(new_streams):,} sentences after LIT cap")

    tr_ids, tr_ch, va_ids, va_ch = split_train_val(ids_list, char_counts, DEFAULTS["val_ratio"])

    results: list[dict] = []
    for seed in args.seeds:
        r = train_and_eval(
            name="CST-lit-cap-zero",
            train_ids=tr_ids, train_chars=tr_ch,
            val_ids=va_ids, val_chars=va_ch,
            vocab_size=combined_vocab_size, seed=seed, epochs=args.epochs,
        )
        r["variant"] = "lit_cap_zero"
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)

    # Summary
    bpcs = [r["best_val_bpc"] for r in results]
    mean = sum(bpcs) / len(bpcs)
    var = sum((x - mean) ** 2 for x in bpcs) / max(len(bpcs) - 1, 1)
    print(f"\n  LIT-cap-zero BPC = {mean:.4f} ± {var**0.5:.4f}  (n={len(bpcs)})")
    return 0


# TODO (stricter variant): take raw corpus as input, re-run the full CST
# pipeline with a flag that *forbids* emission of LIT tokens and forces BPE
# segmentation of OOV surface forms instead. That requires exposing such a
# flag in the TypeScript emitter and re-running the dump.

if __name__ == "__main__":
    raise SystemExit(main())
