# Reasoning Data Plan

**Status:** Draft v0.1
**Prerequisite:** 1M raw-text experiment complete (validates tokenizer + base LM setup).
**Parallel work:** Can start immediately after 1M result; does not block the 10M / 50M / 100M raw-text runs.
**Related:** [../docs/two-level-tokenization.md](../docs/two-level-tokenization.md), [../docs/cst-arabic-tokenizers.md](../docs/cst-arabic-tokenizers.md), [../docs/cst-english-tokenizers.md](../docs/cst-english-tokenizers.md)

---

## 1. Why This Plan Exists

Raw Wikipedia / CulturaX / OSCAR teaches the model what Arabic **looks like**, not how to **reason**. Scaling raw text from 10M to 100M improves the language prior but does not produce logical competence. Before we burn Colab hours on 50M / 100M, we need a parallel data pipeline for reasoning.

**Guiding principle (user, recorded in memory):**

> No fake tests. No tricks. No training data that does nothing. Build real, useful systems.

Every dataset below must earn its place by measurably improving a reasoning metric, not just inflating token counts.

## 2. The Four Data Categories

| #   | Category                      | Teaches                                       | Validates                                     | Priority |
| --- | ----------------------------- | --------------------------------------------- | --------------------------------------------- | -------- |
| 1   | Entailment / paraphrase pairs | logic preservation under rewording            | **reasoning tokenizer** itself (§6.2 of spec) | P0       |
| 2   | Structured / formal logic     | deterministic rules, ground-truth correctness | model logical competence                      | P0       |
| 3   | Chain-of-thought traces       | multi-step reasoning, intermediate steps      | model reasoning depth                         | P1       |
| 4   | Instruction / QA              | task shape, question → answer pattern         | model usability                               | P1       |

P0 items block any claim of a "reasoning model." P1 items make the model usable but are not proof of reasoning.

## 3. Category 1 — Entailment / Paraphrase (P0)

**Purpose.** Validate the reasoning tokenizer: does `T_R(premise)` and `T_R(hypothesis)` preserve the entailment/contradiction label?

### Arabic sources

| Dataset                   | HF id                                      | Size                         | Notes                                          |
| ------------------------- | ------------------------------------------ | ---------------------------- | ---------------------------------------------- |
| XNLI (ar)                 | `xnli` config `ar`                         | ~393k train, 5k dev, 5k test | Translated MultiNLI; standard benchmark.       |
| ArabicNLI                 | `arbml/arabic_nli` or re-derived from MNLI | ~433k                        | Translation quality varies; filter.            |
| ArabicSTS                 | `arbml/arsts`                              | ~10k pairs                   | Semantic similarity scores 0–5.                |
| PAWS-X (ar, if available) | `paws-x`                                   | ~50k                         | Paraphrase adversarial; may lack Arabic split. |

### English sources

| Dataset  | HF id                         | Size            |
| -------- | ----------------------------- | --------------- |
| SNLI     | `snli`                        | 570k            |
| MultiNLI | `multi_nli`                   | 433k            |
| ANLI     | `facebook/anli`               | 163k (r1+r2+r3) |
| PAWS     | `paws` config `labeled_final` | 65k             |

### Target subset for Phase 2

- Arabic: **10k curated pairs** (balanced across entail / contradict / neutral).
- English: **10k curated pairs** (same distribution).

Curation = label audit on a random 10% sample; reject translations with adequacy < 4/5.

### Use

1. Tokenize both sides with `T_R`.
2. Compute token-sequence relationships; check that paraphrases collapse to identical / near-identical sequences and contradictions do not.
3. Failures → candidates for drop-set or inventory revision in the language addendum.

## 4. Category 2 — Structured / Formal Logic (P0)

**Purpose.** Unlimited supply of ground-truth reasoning data. Unambiguous labels. Perfect for both tokenizer validation and model training.

### Sources we already own

**`arabic-algebra-engine`** ([../../arabic-algebra-engine/README.md](../../arabic-algebra-engine/README.md)) is literally a logic-data generator. We should treat it as such, not just as a runtime engine.

Generator emits triples:

```json
{
  "problem_ar": "بسّط: ٢س + ٣س - س",
  "problem_en": "Simplify: 2x + 3x - x",
  "steps_ar": ["اجمع الحدود المتشابهة", "٢س + ٣س = ٥س", "٥س - س = ٤س"],
  "steps_en": ["Combine like terms", "2x + 3x = 5x", "5x - x = 4x"],
  "answer_ar": "٤س",
  "answer_en": "4x",
  "meta": { "difficulty": 1, "topic": "linear_simplify", "seed": 42 }
}
```

Target: **100k–1M** items per language, controlled difficulty distribution.

### Synthetic generators (to write)

| Generator                       | What it produces                             | Target volume |
| ------------------------------- | -------------------------------------------- | ------------- |
| Propositional logic             | `(p ∧ q) → r` + truth-table reasoning        | 50k           |
| Syllogisms                      | "All X are Y. Some Y are Z. ∴ …" in AR + EN  | 20k           |
| Arithmetic word problems        | controlled-difficulty word problems with CoT | 100k          |
| Variable binding / substitution | symbolic manipulation traces                 | 20k           |
| Temporal / spatial reasoning    | "A before B, B before C" → ordering          | 20k           |

### External formal datasets (English; translate to Arabic)

- **GSM8K** — 8.5k grade-school math with CoT.
- **MATH** — 12.5k competition math.
- **MAWPS / SVAMP / ASDiv** — word problems.
- **LogiQA / ReClor** — logical reading comprehension.

## 5. Category 3 — Chain-of-Thought Traces (P1)

**Purpose.** Teach the model to emit intermediate steps, not jump to answers.

### Sources

| Dataset                | Language       | Size        |
| ---------------------- | -------------- | ----------- |
| GSM8K (+ translated)   | EN + AR        | 8.5k × 2    |
| StrategyQA             | EN (translate) | 2.7k        |
| CommonsenseQA          | EN (translate) | 12k         |
| ScienceQA              | EN (translate) | 21k         |
| ARC (easy + challenge) | EN (translate) | 7.7k        |
| AraMath                | AR native      | ~1k (small) |
| AraSciQA               | AR native      | limited     |

**Arabic-native CoT data is thin.** We will need to:

1. Translate English CoT corpora (MT + human review on a sample).
2. Generate synthetic CoT from `arabic-algebra-engine` (Category 2 overlap).
3. Tag every item with `source` and `translation_quality` so we can ablate later.

### Format

Unified JSONL across sources:

```json
{
  "id": "gsm8k-ar-0042",
  "lang": "ar",
  "question": "...",
  "cot": ["step 1", "step 2", "..."],
  "answer": "...",
  "source": "gsm8k",
  "translated_from": "en",
  "translation_quality": 4.5
}
```

## 6. Category 4 — Instruction / QA (P1)

**Purpose.** Task shape. Cheap to collect. Gets us to "usable" faster than CoT.

### Sources

| Dataset                             | Language     | HF id                              |
| ----------------------------------- | ------------ | ---------------------------------- |
| AraBench                            | AR           | per-task                           |
| ARCD (Arabic Reading Comprehension) | AR           | `arcd`                             |
| TyDi-QA (Arabic split)              | AR           | `tydiqa`                           |
| Alpaca-Arabic                       | AR           | `arbml/alpaca_arabic`              |
| CIDAR                               | AR           | `arbml/CIDAR`                      |
| Alpaca                              | EN           | `tatsu-lab/alpaca`                 |
| Dolly 15k                           | EN           | `databricks/databricks-dolly-15k`  |
| OpenAssistant (oasst1)              | EN + AR      | `OpenAssistant/oasst1`             |
| Natural Instructions                | EN + some AR | `Muennighoff/natural-instructions` |

Target: **20k AR + 20k EN** curated.

## 7. Training Curriculum (Order Matters)

```
Stage 1 — Raw text (10M → 100M)         ← language prior
Stage 2 — Instruction / QA (Cat. 4)      ← task shape
Stage 3 — CoT + formal logic (Cat. 2+3)  ← reasoning ability
Stage 4 — Entailment eval (Cat. 1)       ← validation, not training
Stage 5 — RL / preference (future)       ← reasoning quality
```

**Failure mode to avoid:** skipping Stage 1 → 3 directly. The model needs a language prior before it can be taught to reason in that language.

**Mixing note:** Stages 2 and 3 can be blended in a single fine-tune with careful ratios (e.g. 20% instruction / 60% CoT / 20% formal). Stage 1 stays pretraining-only.

## 8. Volume Targets per Stage

| Stage                  | Arabic                     | English             | Total |
| ---------------------- | -------------------------- | ------------------- | ----- |
| 1. Raw text            | 10M / 50M / 100M sentences | same corpus mix TBD | —     |
| 2. Instruction         | 20k items                  | 20k items           | 40k   |
| 3a. CoT (curated)      | 30k items                  | 30k items           | 60k   |
| 3b. Formal (synthetic) | 500k items                 | 500k items          | 1M    |
| 4. Entailment eval     | 10k pairs                  | 10k pairs           | 20k   |

Stages 2–4 fit comfortably on a single laptop / workstation. No Colab required to build them. Only Stage 3b's generator scripts may take a few hours to run end-to-end.

## 9. Deliverables

| #   | File                                                          | Purpose                                                                             |
| --- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| D1  | `edge/training/build_reasoning_data.py`                       | Master pipeline: pulls datasets, translates, normalizes, emits one JSONL per stage. |
| D2  | `edge/training/translate_cot.py`                              | MT + validation for English → Arabic CoT.                                           |
| D3  | `arabic-algebra-engine/src/training/generate_logic_corpus.ts` | Turns the engine into a CoT generator (both languages, controlled difficulty).      |
| D4  | `edge/training/generators/prop_logic.py`                      | Propositional-logic generator.                                                      |
| D5  | `edge/training/generators/syllogisms.py`                      | Syllogism generator (AR + EN).                                                      |
| D6  | `edge/training/eval/reasoning_tokenizer.py`                   | §6.2 validator — runs entailment pairs through `T_R` and checks logic preservation. |
| D7  | `plan/REASONING_DATA_RESULTS.md`                              | Live results doc; filled as each stage completes.                                   |

## 10. Execution Order (Post-1M)

1. **Week 1.** D3 (algebra-engine generator). Zero external dependencies, infinite data, ground-truth correct. This alone can deliver 200k–1M items for Stage 3b.
2. **Week 1.** D6 (reasoning tokenizer validator) on existing XNLI-ar sample. This validates §6.2 of the spec before we commit more data effort.
3. **Week 2.** D1 + D2 (pull + translate external CoT). Parallelized with D4/D5.
4. **Week 2.** D4 + D5 (symbolic generators).
5. **Week 3.** Integration: build the full Stage 2 + Stage 3 corpus. Begin Stage 2 fine-tune on top of the 10M pretraining checkpoint.
6. **Week 3+.** Scale raw text (Stages 1 → 50M / 100M) and reasoning fine-tune in parallel on separate machines.

## 11. Decisions (Locked)

1. **Translation provider.** NLLB-200 for bulk translation; **Gemini** (`gemini-2.0-flash`, via `GEMINI_API_KEY`) for a random 10% QA sample with adequacy scoring 1–5. Items scoring < 4 are re-translated or dropped.
2. **Difficulty distribution** for the algebra-engine generator: **skewed easy→hard at 70 / 20 / 10** (easy / medium / hard). Uniform for prop-logic and syllogism generators.
3. **Preserve English source** alongside Arabic translation — yes. Doubles storage but enables ablation and pair-level entailment checks between languages.
4. **License audit.** Every JSONL record carries a `license` field. Non-commercial items tagged `"nc"` and excluded from any future release build; permissive items tagged with SPDX id.
5. **Dialect policy.** **MSA-only** for all reasoning data in this phase. Dialectal reasoning is a separate future axis and is explicitly out of scope here.

## 12. Non-Goals

- Building a chatbot or assistant. This plan produces **training data**, not a product.
- Competing with GSM8K / MMLU leaderboard numbers. We want our **own** reasoning evaluation first; external benchmarks come later.
- Replacing raw-text pretraining. Stages 1 and 3 are complementary, not alternatives.
