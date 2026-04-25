# BPE GPT-2 baseline — Phase 2b

The fair-comparison baseline for the Phase 2b head-to-head. Trains two
100M-parameter GPT-2 small variants on the same `reasoning/out/` 170k
records that the CST-logic model sees, so the four-way table in
`docs/reasoning-experiments.md` is apples-to-apples.

## The four models

| Model                     | Script                                                     | Vocab   | Init         |
| ------------------------- | ---------------------------------------------------------- | ------- | ------------ |
| A — CST-logic tiny (~2M)  | `training/colab_train_reasoning.py` (existing, ~2M config) | 151     | from-scratch |
| B — CST-logic small (~5M) | same, ~5M config                                           | 151     | from-scratch |
| C — GPT-2 from-scratch    | `train_gpt2_from_scratch.py` (here)                        | 50k BPE | from-scratch |
| D — GPT-2 fine-tuned      | `finetune_gpt2.py` (here)                                  | 50k BPE | HF `gpt2`    |

All four see the same 170k records (`reasoning/out/*.jsonl`), same
train/val split, same token budget, same wall-clock budget on Colab.

## Prompt format controls

BPE models need a prompt. We run two conditions per BPE model to be fair:

- **Raw** — natural-language question from the record, plain English/Arabic.
- **Symbolic** — question rewritten as `P=T, Q=F; P AND Q = ?` style.
  Tests whether BPE's disadvantage is about tokenization of symbols vs.
  about genuinely not learning logic.

Both conditions use the same model weights; only the eval prompt changes.

## Metrics reported

- Exact-match accuracy per task family (prop / syllogism / algebra).
- Sample-efficiency curves (acc vs. training steps).
- Inference latency (ms/example, CPU and GPU).
- Model size on disk (FP16 and INT8 where applicable).
- **Body-parameter count** (transformer blocks only, excluding
  embeddings) alongside total — avoids the "100M = 60M body + 40M
  embedding" asymmetry.
- Depth-generalization: held-out test set uses prop depth 5–7 while
  training is 1–4.

## Files

- `train_gpt2_from_scratch.py` — model C. Same hyperparameters as the
  CST-logic training, just different tokenizer.
- `finetune_gpt2.py` — model D. Loads HF `gpt2`, fine-tunes on the 170k.
- `eval_bpe.py` — runs a trained BPE model over `reasoning/eval/holdout/`
  in both raw and symbolic conditions, writes per-family accuracy JSON.
- `configs/gpt2_small.yaml` — shared hyperparameters.

## Status

Scaffold only. Run after:

1. Logic model training (Phase 2) completes.
2. Held-out eval set is regenerated via
   `python -m reasoning.eval.holdout_generator`.
