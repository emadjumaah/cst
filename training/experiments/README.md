# Experiment scripts

This directory contains the scripts required to turn the CST proof of concept
into a fully defensible publication. Every script writes its results as JSON
so `aggregate_results.py` can stitch them into a single Markdown report.

Shared code lives in `_core.py`: model builder, data loader, training loop,
seed setter. All experiment scripts import from it so the training recipe is
identical across runs.

## Scripts

| Script                    | Purpose                                                                                                                                       | Plan section       |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `run_multiseed.py`        | Run CST and SPM N times with different seeds; report mean ± std BPC.                                                                          | (main table rigor) |
| `ablate_label_shuffle.py` | Randomly permute CST structured token IDs; re-train; compare BPC. Tests whether the specific semantic labels matter, or only the granularity. | A.2                |
| `ablate_lit_cap_zero.py`  | Replace every `LIT:*` token with SPM subpieces of the same sentence; re-train. Tests whether LIT is carrying the BPC win.                     | A.3                |
| `wordlevel_baseline.py`   | Plain word-level tokenizer at matched vocab size. The missing "weak" control.                                                                 | A.1                |
| `aggregate_results.py`    | Merge all result JSONs into a Markdown table; run Welch's t-tests on matched pairs.                                                           | —                  |

## Typical Colab workflow

1. Upload the tokenized `.jsonl` and vocab files produced locally (from `data/tokenized/`) to `/content`.
2. Install deps: `pip install -r training/requirements.txt`.
3. Run the main multi-seed sweep:
   ```bash
   python training/experiments/run_multiseed.py --lang en --runs 8k 32k --seeds 0 1 2 --out /content/results_en.json
   python training/experiments/run_multiseed.py --lang ar --runs 8k 32k --seeds 0 1 2 --out /content/results_ar.json
   ```
4. Run the ablations on the 8K CST corpus:

   ```bash
   python training/experiments/ablate_label_shuffle.py \
       --data /content/cst-8k-train-99963.jsonl \
       --vocab /content/cst-8k-train-99963-vocab.json \
       --out /content/results_label_shuffle.json --seeds 0 1 2

   python training/experiments/ablate_lit_cap_zero.py \
       --cst-data /content/cst-8k-train-99963.jsonl \
       --cst-vocab /content/cst-8k-train-99963-vocab.json \
       --spm-data /content/spm-train-99963-8k.jsonl \
       --spm-vocab /content/spm-train-99963-8k-vocab.json \
       --out /content/results_lit_cap_zero.json --seeds 0 1 2

   python training/experiments/wordlevel_baseline.py \
       --jsonl /content/sentences-100k.jsonl \
       --vocab-size 8000 --lang en \
       --out /content/results_wordlevel_en_8k.json --seeds 0 1 2
   ```

5. Aggregate:
   ```bash
   python training/experiments/aggregate_results.py \
       /content/results_*.json \
       --out docs/results_tables.md
   ```

## What each ablation is answering

- **Multi-seed**: are the observed BPC gaps larger than random variation?
- **Label shuffle**: does _which_ field a word is assigned to matter, or
  only that CST has more, smaller, content-specific tokens?
- **LIT cap zero**: is the win coming from the structured part of CST, or
  from the literal fallback acting as a word-level tokenizer?
- **Word-level baseline**: how much of CST's improvement is "word-level
  beats BPE at this scale" that any word-level vocabulary would capture?

If any of these three ablations come back near the CST baseline, the claim
in the paper must be scoped accordingly.
