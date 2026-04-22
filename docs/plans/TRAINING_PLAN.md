# Training Plan — Phase 0, Scale Sweep, Reasoning

**Purpose:** One ordered checklist covering the three training tracks that
together constitute the "stand on its feet" milestone for the project. Each
track already has code; this doc is the unified operating manual so nothing
is missed.

**Scope & out of scope**
- In scope: run commands, smoke tests, result-interpretation rules, artifact paths.
- Out of scope: Phase 1+ (reasoning-mode tokenizer), verifier/scratchpad, edge demo.
- Languages: EN + AR for all three tracks. RU stays in [PHASE0_RUSSIAN.md](PHASE0_RUSSIAN.md).

**Global conventions**
- Training loop: `training/experiments/_core.py` (`PRESET` = n_embd/n_layer/n_head; `DEFAULTS` = epochs/lr/batch/max_len).
- Corpora: Python-tokenized (`edge/english_tokenizer.py`, `edge/arabic_tokenizer.py`). TS is rules source, not training.
- Scoring metric for LM runs: **best validation BPC** (bits-per-character). Lower = better. Compare *at matched character count*, not matched token count.
- Seeds: 5 (0..4) is the multi-seed standard. 2 seeds for smoke, 5 for reporting.
- Colab: T4 is the baseline; A100 for 50M/100M.

---

## Track 1 — Phase 0: multi-seed rigor @ 10M

**Goal:** Replace the single-seed numbers in the published paper with mean ± std
across 5 seeds at 8K and 32K vocabs, for EN + AR.

**Preconditions (done):**
- `data/tokenized/cst-py/train-99963.jsonl` — Python-tokenized EN, 99,963 rows.
- `training/colab-upload-en/` — cst-8k/cst-32k (Python) + spm baselines + bundled scripts.
- `training/colab-upload-en.tar.gz` (72 MB) — Colab upload blob.
- `training/phase0_en_colab.ipynb` — one-click notebook.
- AR artefacts on disk: `data/tokenized/cst-ar-8k/`, `cst-ar-32k/`, `spm-ar/` (already Python-tokenized).

**Preconditions (to do before AR runs):**
- Re-tokenize AR corpus through unified `python -m edge.tokenize --lang ar` to confirm parity with existing files (spot-check a 1k sample — type distribution within 1% is enough).
- Build `training/colab-upload-ar.tar.gz` using a `build_colab_upload_ar.sh` mirroring the EN script.

### Run (Colab)

```
# EN
!pip install -q torch transformers
!tar -xzf /content/colab-upload-en.tar.gz -C /content/
!cd /content/colab-upload-en && python scripts/experiments/colab_phase0.py \
    --data-dir /content/colab-upload-en \
    --out-dir  /content/phase0_out \
    --langs en --runs 8k 32k --seeds 0 1 2 3 4 --epochs 3 --skip-downstream

# AR — same, with colab-upload-ar.tar.gz
```

### Smoke test (before committing an 8-hour run)

```
# 2 seeds, 8k only, 1 epoch — should finish in ~15 min on T4 and produce sane BPC.
python scripts/experiments/colab_phase0.py ... --runs 8k --seeds 0 1 --epochs 1 --skip-downstream
```

**Pass:** CST-8K BPC < 2.0, SPM-8K BPC < 2.0, gap under 0.3. **Fail:** either tokenizer above 3.0 → corpus or vocab-cap is broken.

### Check results

- `phase0_out/summary.md` has a table with 4 rows (CST-8K, SPM-8K, CST-32K, SPM-32K), mean ± std, n=5.
- Expected direction: **CST ≤ SPM** at matched vocab. If CST is more than 0.05 BPC worse than SPM, investigate before publishing.
- Aggregate and push to paper: `training/experiments/aggregate_results.py` → `docs/cst-paper.md` "Robustness" section.

**Done when:** `docs/cst-paper.md` contains the updated table with 5-seed numbers for EN + AR at 8K + 32K, and OSF preprint is updated.

---

## Track 2 — Scale sweep: 1M / 10M / 50M / 100M

**Goal:** Show that the CST advantage is not a small-model artefact. One scaling curve per language plotting BPC vs parameter count for CST and SPM at matched 32K vocab.

**Shape of the run (6 model configs — 50M/100M optional if budget tight):**

| Tag  | n_embd | n_layer | n_head | approx params | T4 time / run | A100 time / run |
|------|-------:|--------:|-------:|--------------:|--------------:|----------------:|
| 1M   |     64 |       2 |      2 |   ~1.0M       | ~5 min        | <2 min          |
| 10M  |    256 |       6 |      4 |   ~10M (current PRESET) | ~20 min | ~5 min |
| 50M  |    512 |       8 |      8 |   ~50M        | ~2 h          | ~25 min         |
| 100M |    768 |      12 |     12 |   ~100M       | ~5 h          | ~1 h            |

### Required code change (one file)

`training/experiments/_core.py` currently hardcodes a single `PRESET`. Make it
parameter-swappable:

1. Rename `PRESET` → `PRESETS: dict[str, dict] = {"1M": {...}, "10M": {...}, ...}`.
2. Add `--preset` arg to `run_multiseed.py` and `colab_phase0.py`; default `"10M"` to preserve existing behaviour.
3. `train_and_eval(..., preset="10M")` picks the config, and the printed param-count matches the tag within ±10%.

This is a **2-hour refactor**, not a rewrite. Tests: run 1M preset for 1 epoch on 10k rows and confirm the model reports ~1M params and trains.

### Run

```
# One invocation per preset; 3 seeds is enough for a scaling curve.
for P in 1M 10M 50M 100M; do
  python scripts/experiments/colab_phase0.py \
      --data-dir /content/colab-upload-en \
      --out-dir  /content/scale_out/$P \
      --langs en --runs 32k --seeds 0 1 2 --preset $P --epochs 3 --skip-downstream
done
```

50M + 100M should be A100 runs. 1M + 10M can stay on T4.

### Smoke test

- 1M preset, 1 seed, 1 epoch → produces a result row. Param count printed matches `~1.0M`.
- 100M preset, 0 training steps (`--epochs 0` patch), just builds model → confirms VRAM fits on chosen GPU before committing a 5h run.

### Check results

- `scale_out/summary.md` has 4×2×2 = 16 rows (4 scales × CST/SPM × 2 seeds × ... averaged → 8 rows × 3 seeds).
- Plot script: `training/experiments/plot_scaling.py` (**to write**, ~50 lines) → matplotlib line chart, log-x on params, BPC on y, one line per tokenizer.
- Success: CST line stays **at or below** SPM line across all four scales. A crossover at 100M would be a real finding — either way it goes in the paper.

**Done when:** `docs/scaling_en.png` and `docs/scaling_ar.png` exist + `docs/cst-paper.md` has a "Scaling" section with the plot and interpretation.

---

## Track 3 — Reasoning model training (stages 2a/2b/2c)

**Goal:** Train a single model over all three reasoning corpora and measure whether the reasoning-level tokenizer actually lets a small model learn the underlying logic.

**Preconditions (done):**
- `MyDrive/cst-poc/reasoning/tokenized/stage-2a-prop_logic.tokenized.jsonl` (71 MB)
- `MyDrive/cst-poc/reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl` (50 MB)
- `MyDrive/cst-poc/reasoning/tokenized/stage-2c-algebra_engine.tokenized.jsonl` (229 MB)
- `MyDrive/cst-poc/reasoning/tokenized/vocab-reasoning.json`
- Trainer: [reasoning/train/train.py](../reasoning/train/train.py), [reasoning/train/dataset.py](../reasoning/train/dataset.py), [reasoning/train/model.py](../reasoning/train/model.py).
- Colab notebook stub: [reasoning/train/colab_train_reasoning.ipynb](../reasoning/train/colab_train_reasoning.ipynb).

### Required code changes

1. **Multi-corpus loader** — `ReasoningJsonlDataset` currently takes one file. Extend to accept a list and interleave proportionally to size (2a: 2b: 2c ≈ 71:50:229 → use length-weighted sampling).
2. **Staged curriculum option** — `--curriculum {mixed,sequential}`: `sequential` trains one epoch each on 2a → 2b → 2c; `mixed` is the default interleaved mode. Both produce the same final ckpt format.
3. **Eval harness** — wire `reasoning/eval/tokenizer_logic.py` + a new `reasoning/eval/lm_tasks.py` that scores the trained model on:
   - **2a prop-logic:** given premises + question, pick correct truth value. Metric: accuracy.
   - **2b syllogisms:** same, given two premises pick the valid conclusion form. Metric: accuracy.
   - **2c algebra:** given an expression, pick the correct canonical form from 4 distractors. Metric: accuracy.
   All three use the `lm_scoring` pattern from [training/experiments/downstream_eval.py](../training/experiments/downstream_eval.py) — lowest-NLL candidate wins. Reuse that code path; do not reimplement.

### Run (Colab, mounts Drive)

```
from google.colab import drive; drive.mount('/content/drive')
DATA=/content/drive/MyDrive/cst-poc/reasoning/tokenized
OUT=/content/drive/MyDrive/cst-poc/reasoning/runs/mixed-v1

!git clone <repo> /content/cst-poc && cd /content/cst-poc

!python -m reasoning.train.train \
    --data  $DATA/stage-2a-prop_logic.tokenized.jsonl \
            $DATA/stage-2b-syllogisms.tokenized.jsonl \
            $DATA/stage-2c-algebra_engine.tokenized.jsonl \
    --vocab $DATA/vocab-reasoning.json \
    --out   $OUT \
    --preset 10M --epochs 3 --seed 0 \
    --curriculum mixed
```

Emits `$OUT/ckpt.pt`, `$OUT/train_log.jsonl`, `$OUT/config.json`.

### Smoke test

```
# Subsample each stage to 2k rows, 1 epoch — must run < 10 min on T4 and loss must decrease monotonically.
!python -m reasoning.train.train ... --limit 2000 --epochs 1
```

**Pass:** final train loss < 90% of initial. **Fail:** loss flat → vocab / dataset mismatch (most likely cause: wrong `vocab-reasoning.json` path).

### Check results

Run eval on the best checkpoint:

```
!python -m reasoning.eval.lm_tasks \
    --ckpt  $OUT/ckpt.pt \
    --vocab $DATA/vocab-reasoning.json \
    --tasks prop_logic syllogisms algebra \
    --out   $OUT/eval.json
```

Expected ranges (10M params, 3 epochs, mixed curriculum):

| Task | Chance | Target | Stretch |
|---|---|---|---|
| prop_logic  | 50% | ≥ 75% | ≥ 90% |
| syllogisms  | 25% | ≥ 60% | ≥ 80% |
| algebra     | 25% | ≥ 50% | ≥ 70% |

**If below target for any task:** first suspect is curriculum. Switch to `--curriculum sequential` and rerun. Second suspect is scale — rerun at 50M preset once Track 2 unblocks the presets work.

**Done when:** `$OUT/eval.json` exists with all three task scores, and `docs/reasoning-experiments.md` has a results table and one example failure case per task.

---

## Execution order

Tracks 1 and 3 are independent and can run in parallel on two Colab instances.
Track 2 depends on the `PRESETS` refactor landing first, so the natural order is:

1. **Now:** submit Track 1 EN multi-seed to Colab (blocks on nothing).
2. **Today:** do the `PRESETS` refactor, commit, then queue Track 2 smoke runs (1M + 10M at 1 seed each) to confirm the refactor.
3. **In parallel:** start Track 3 smoke run on Drive-mounted Colab.
4. **Once Track 1 EN returns green:** build AR upload tarball, submit Track 1 AR.
5. **Once Track 2 smoke runs green:** submit full scale sweep (A100 for 50M/100M).
6. **Once Track 3 smoke returns green:** submit full mixed-curriculum training (3 epochs, all three stages).

Everything above lives in code already committed. Nothing new-to-write except:
- `training/build_colab_upload_ar.sh` (mirror of EN script).
- `training/experiments/plot_scaling.py` (~50 lines).
- `reasoning/eval/lm_tasks.py` (wraps existing `downstream_eval.lm_scoring`).
- `PRESETS` refactor in `training/experiments/_core.py`.
- Multi-corpus + curriculum flags in `reasoning/train/train.py`.

## Definition of "stands on its feet"

All three tracks have a results JSON committed to the repo, a summary section
in the paper, and a one-command reproduction recipe that another person with
a fresh Colab GPU could run end-to-end without asking questions.
