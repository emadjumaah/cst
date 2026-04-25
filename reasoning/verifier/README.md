# Reasoning Verifier Track

This track is a hard pivot away from autoregressive LM scoring. It trains a small
binary verifier that distinguishes coherent reasoning traces from controlled
counterfactuals.

v0.2 changes:

- Adds explicit `neg_question_only` shortcut negatives in dataset building.
- Uses combined objective in training: BCE classification + pairwise ranking.
- Pairwise constraints enforce `score(full) > score(question_only)` and
  `score(full) > score(shuffled_cot)` during training.

The key claim we care about is tokenizer advantage:

- CST logic tokenizer condition: `--view reasoning`
- baseline tokenizer condition: `--view default`

If you want to test the new Arabic morphology-aware tokenizer variant
(`ROOT+PAT` plus explicit `SPACE` boundaries), regenerate the tokenized
corpus first:

```bash
python -m reasoning.tokenize_corpus \
  --in reasoning/out \
  --out reasoning/tokenized-vnext \
  --ar-root-pattern \
  --ar-space-token
```

## Colab-First Quickstart

If local Mac is weak, run this in Colab GPU (`T4` is enough for this verifier).

```bash
# In Colab cell:
%cd /content
!git clone https://github.com/<your-org>/cst-poc.git
%cd /content/cst-poc

# Torch is usually preinstalled in Colab; install only if missing.
!python - <<'PY'
import importlib.util
print('torch:', bool(importlib.util.find_spec('torch')))
PY
```

Run all training/eval commands below with `--device cuda`.

## 1) Build verifier dataset

```bash
python -m reasoning.verifier.build_dataset \
  --in reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl \
  --out reasoning/verifier/data/verifier_train.jsonl \
  --view reasoning \
  --seed 42 \
  --neg-question-only 1 \
  --neg-shuffled-cot 1 \
  --neg-answer-flip 1 \
  --neg-shuffled-and-flip 1
```

## 2) Train tiny verifier

```bash
python -m reasoning.verifier.train_verifier \
  --data reasoning/verifier/data/verifier_train.jsonl \
  --vocab reasoning/tokenized/vocab-reasoning.json \
  --out-dir reasoning/verifier/runs/verifier-v0.2 \
  --epochs 8 \
  --batch-size 64 \
  --pairwise-weight 0.8 \
  --pairwise-margin 0.2 \
  --pairwise-variants neg_question_only,neg_shuffled_cot \
  --device auto
```

## 3) Evaluate reasoning signal with controls

```bash
python -m reasoning.verifier.eval_signal \
  --checkpoint reasoning/verifier/runs/verifier-v0.2/best.pt \
  --input reasoning/eval/holdout_ood_a1_tokenized/syllogisms.tokenized.jsonl \
  --vocab reasoning/tokenized/vocab-reasoning.json \
  --view reasoning \
  --out-json reasoning/verifier/runs/verifier-v0.2/eval_ood.json \
  --balance-by-answer \
  --min-full-mean 0.60 \
  --min-gap-qonly 0.03 \
  --min-gap-shuffled 0.03 \
  --min-win-rate 0.60
```

## Pass Gate

`reasoning_proof_pass = true` only if all are true:

- mean(full) >= min_full_mean
- mean(full) - mean(question_only) >= min_gap_qonly
- mean(full) - mean(shuffled_cot) >= min_gap_shuffled
- win_rate(full > question_only) >= min_win_rate
- win_rate(full > shuffled_cot) >= min_win_rate

## CST vs Baseline Tokenizer A/B (Prove/Disprove Advantage)

Run both conditions with the same architecture and schedule.

### A) Build datasets

```bash
python -m reasoning.verifier.build_dataset \
  --in reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl \
  --out reasoning/verifier/data/verifier_train.cst.jsonl \
  --view reasoning \
  --seed 42 \
  --neg-question-only 1 \
  --neg-shuffled-cot 1 \
  --neg-answer-flip 1 \
  --neg-shuffled-and-flip 1

python -m reasoning.verifier.build_dataset \
  --in reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl \
  --out reasoning/verifier/data/verifier_train.base.jsonl \
  --view default \
  --seed 42 \
  --neg-question-only 1 \
  --neg-shuffled-cot 1 \
  --neg-answer-flip 1 \
  --neg-shuffled-and-flip 1
```

### B) Train both conditions (matched budget)

```bash
python -m reasoning.verifier.train_verifier \
  --data reasoning/verifier/data/verifier_train.cst.jsonl \
  --vocab reasoning/tokenized/vocab-reasoning.json \
  --out-dir reasoning/verifier/runs/verifier-cst-v0.2 \
  --epochs 8 --batch-size 64 \
  --pairwise-weight 0.8 --pairwise-margin 0.2 \
  --pairwise-variants neg_question_only,neg_shuffled_cot \
  --device auto

python -m reasoning.verifier.train_verifier \
  --data reasoning/verifier/data/verifier_train.base.jsonl \
  --vocab reasoning/tokenized/vocab-default.json \
  --out-dir reasoning/verifier/runs/verifier-base-v0.2 \
  --epochs 8 --batch-size 64 \
  --pairwise-weight 0.8 --pairwise-margin 0.2 \
  --pairwise-variants neg_question_only,neg_shuffled_cot \
  --device auto
```

### C) Evaluate both conditions

```bash
python -m reasoning.verifier.eval_signal \
  --checkpoint reasoning/verifier/runs/verifier-cst-v0.2/best.pt \
  --input reasoning/eval/holdout_ood_a1_tokenized/syllogisms.tokenized.jsonl \
  --vocab reasoning/tokenized/vocab-reasoning.json \
  --view reasoning \
  --out-json reasoning/verifier/runs/verifier-cst-v0.2/eval_ood.json \
  --balance-by-answer --min-full-mean 0.60 --min-gap-qonly 0.03 --min-gap-shuffled 0.03 --min-win-rate 0.60

python -m reasoning.verifier.eval_signal \
  --checkpoint reasoning/verifier/runs/verifier-base-v0.2/best.pt \
  --input reasoning/eval/holdout_ood_a1_tokenized/syllogisms.tokenized.jsonl \
  --vocab reasoning/tokenized/vocab-default.json \
  --view default \
  --out-json reasoning/verifier/runs/verifier-base-v0.2/eval_ood.json \
  --balance-by-answer --min-full-mean 0.60 --min-gap-qonly 0.03 --min-gap-shuffled 0.03 --min-win-rate 0.60
```

### D) Emit final support/refute verdict

```bash
python -m reasoning.verifier.advantage_verdict \
  --cst-eval reasoning/verifier/runs/verifier-cst-v0.2/eval_ood.json \
  --baseline-eval reasoning/verifier/runs/verifier-base-v0.2/eval_ood.json \
  --cst-train-summary reasoning/verifier/runs/verifier-cst-v0.2/summary.json \
  --baseline-train-summary reasoning/verifier/runs/verifier-base-v0.2/summary.json \
  --out-json reasoning/verifier/runs/tokenizer_advantage_verdict.v0.2.json
```
