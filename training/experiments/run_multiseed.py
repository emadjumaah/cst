"""
Multi-seed experiment runner.

Runs each (data, vocab) pair N times with different seeds and reports
mean ± std BPC. Produces a results JSON for ``aggregate_results.py``.

Usage (on Colab after uploading tokenized files to /content):

    python training/experiments/run_multiseed.py \\
        --out /content/results_multiseed.json \\
        --seeds 0 1 2 \\
        --runs 8k 32k \\
        --lang en

Arguments:
    --lang     {en, ar}       which language to run
    --runs     any of: 8k 32k (space-separated)
    --seeds    integer list (default: 0 1 2)
    --data-dir base directory with *.jsonl and *-vocab.json files (default /content)
    --out      output JSON path
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make ``training`` importable when this script is run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments._core import (  # noqa: E402
    DEFAULTS,
    load_jsonl,
    load_vocab_size,
    split_train_val,
    train_and_eval,
)


PAIRS = {
    "en": {
        "8k":  [("CST-8K",  "cst-8k/train-99963.jsonl",       "cst-8k/train-99963-vocab.json"),
                ("SPM-8K",  "spm/train-99963-8k.jsonl",        "spm/train-99963-8k-vocab.json")],
        "32k": [("CST-32K", "cst-32k/train-99963.jsonl",      "cst-32k/train-99963-vocab.json"),
                ("SPM-32K", "spm/train-99963-32k.jsonl",       "spm/train-99963-32k-vocab.json")],
    },
    "ar": {
        "8k":  [("AR-CST-8K",  "cst-ar-8k/train-100000.jsonl",  "cst-ar-8k/train-100000-vocab.json"),
                ("AR-SPM-8K",  "spm-ar/ar-bpe-8000.jsonl",      "spm-ar/ar-bpe-8000-vocab.json")],
        "32k": [("AR-CST-32K", "cst-ar-32k/train-100000.jsonl", "cst-ar-32k/train-100000-vocab.json"),
                ("AR-SPM-32K", "spm-ar/ar-bpe-32000.jsonl",     "spm-ar/ar-bpe-32000-vocab.json")],
    },
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["en", "ar"], required=True)
    p.add_argument("--runs", nargs="+", default=["8k", "32k"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data-dir", default="/content")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    args = p.parse_args()

    all_results: list[dict] = []
    for run in args.runs:
        for name, data_rel, vocab_rel in PAIRS[args.lang][run]:
            data_path = os.path.join(args.data_dir, data_rel)
            vocab_path = os.path.join(args.data_dir, vocab_rel)
            # Also accept flat-dir layouts by trying basename fallback.
            if not os.path.exists(data_path):
                flat = os.path.join(args.data_dir, os.path.basename(data_rel))
                if os.path.exists(flat):
                    data_path = flat
                    vocab_path = os.path.join(args.data_dir, os.path.basename(vocab_rel))
            if not os.path.exists(data_path):
                print(f"  MISSING: {data_path} (skipping {name})")
                continue

            ids_list, char_counts = load_jsonl(data_path, DEFAULTS["max_len"])
            vocab_size = load_vocab_size(vocab_path)
            tr_ids, tr_ch, va_ids, va_ch = split_train_val(ids_list, char_counts, DEFAULTS["val_ratio"])

            for seed in args.seeds:
                result = train_and_eval(
                    name=name,
                    train_ids=tr_ids, train_chars=tr_ch,
                    val_ids=va_ids, val_chars=va_ch,
                    vocab_size=vocab_size,
                    seed=seed,
                    epochs=args.epochs,
                )
                result["lang"] = args.lang
                result["run"] = run
                all_results.append(result)
                # Incrementally persist so Colab disconnects don't lose work.
                with open(args.out, "w") as f:
                    json.dump(all_results, f, indent=2, default=float)
                print(f"  saved {len(all_results)} runs → {args.out}")

    # Aggregate
    from collections import defaultdict
    by_name: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        by_name[r["name"]].append(r["best_val_bpc"])

    print(f"\n{'='*70}\n  Multi-seed aggregate (n={len(args.seeds)} seeds)\n{'='*70}")
    for name, bpcs in by_name.items():
        mean = sum(bpcs) / len(bpcs)
        var = sum((x - mean) ** 2 for x in bpcs) / max(len(bpcs) - 1, 1)
        std = var ** 0.5
        print(f"  {name:<14}  BPC = {mean:.4f} ± {std:.4f}   (n={len(bpcs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
