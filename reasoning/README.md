# Reasoning Pipeline

Parallel track to the raw-text pretraining pipeline in [`edge/`](../edge/).

This folder owns everything related to **reasoning data and the reasoning-level tokenizer**:

- The **reasoning tokenizer** `T_R` (Arabic + English), which projects default-level CST tokens to the coarser reasoning level defined in [`docs/two-level-tokenization.md`](../docs/two-level-tokenization.md).
- **Data builders** that pull, translate, and normalize reasoning datasets (entailment, CoT, instruction, formal logic).
- **Synthetic generators** for propositional logic and syllogisms.
- **Evaluation** for §6.2 of the spec — logic preservation under tokenization.

See [`plan/REASONING_DATA.md`](../plan/REASONING_DATA.md) for the full curriculum and deliverable list.

## Layout

```
reasoning/
  README.md                 ← this file
  requirements.txt
  tokenizer/
    __init__.py
    arabic.py               ← T_R^ar (wraps edge/arabic_tokenizer.py + applies π)
    english.py               ← T_R^en (stub — backend TBD)
    projection.py            ← projection rules shared by both languages
  data/
    __init__.py
    build.py                 ← master pipeline (all stages)
    sources/
      xnli.py                ← Category 1 — entailment
      gsm8k.py                ← Category 3 — CoT (translated)
      algebra_engine.py       ← Category 2 — pulls from arabic-algebra-engine
    generators/
      prop_logic.py           ← Category 2 — propositional logic
      syllogisms.py            ← Category 2 — syllogisms AR+EN
    translate.py              ← NLLB-200 + Claude QA
  eval/
    tokenizer_logic.py        ← §6.2 validator
```

## Running

```bash
# From repo root
cd reasoning
pip install -r requirements.txt

# Stage 2 validator — does the reasoning tokenizer preserve entailment?
python eval/tokenizer_logic.py --dataset xnli --n 1000

# Stage 3b — generate synthetic formal-logic corpus (no external data needed)
python data/build.py --stage 3b --out ./out

# Full build (downloads + translates + generates)
python data/build.py --stage all --out ./out

# Optional Arabic tokenizer v-next (root+pattern + SPACE boundaries)
python tokenize_corpus.py --in ./out --out ./tokenized \
  --ar-root-pattern --ar-space-token
```

## Contract

Every JSONL record produced by any builder in this folder conforms to:

```json
{
  "id": "<source>-<lang>-<idx>",
  "lang": "ar" | "en",
  "category": 1 | 2 | 3 | 4,
  "question": "...",
  "cot": ["step 1", "step 2", "..."],
  "answer": "...",
  "meta": {
    "source": "xnli | gsm8k | algebra-engine | prop-logic | ...",
    "license": "cc-by-4.0 | mit | apache-2.0 | nc | ...",
    "translated_from": "en" | null,
    "translation_quality": 4.5 | null,
    "difficulty": "easy" | "medium" | "hard"
  }
}
```

`cot` is an empty array for Category 1 and 4 items.
