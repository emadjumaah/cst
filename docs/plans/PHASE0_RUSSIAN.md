# Phase 0 — Russian track

Status: **not started**. Blocks the Phase 0 `ru` column in
[training/experiments/colab_phase0.py](../training/experiments/colab_phase0.py).
Stub entries for `ru` already exist in
[training/experiments/run_multiseed.py](../training/experiments/run_multiseed.py)
`PAIRS`, and they will be skipped by the runner until the tokenized files below
are produced.

Adding Russian to Phase 0 is a deliberate choice (ROADMAP Phase 0) to turn the
paper from bilingual into tri-lingual and to test CST on an inflectionally rich
Indo-European language whose morphology is different in kind from both English
(analytic) and Arabic (nonconcatenative / root-and-pattern).

This document is the checklist that has to be ticked before the multi-seed
sweep can include RU.

## 1. Corpus

- **Size target:** 100,000 sentences (matches EN and AR).
- **Source options:**
  - Wikipedia-ru dump → sentence-split → 100K random sample. Cheapest.
  - Leipzig Corpora Collection "rus_news" / "rus_wikipedia" 100K packages.
  - OSCAR-ru — larger than needed, but already cleaned.
- **Deliverable:**
  - `data/russian/sentences-100k.json` — one sentence per line / array entry,
    UTF-8, NFC-normalized, no duplicates, length filter 10–200 chars.
  - Mirror of the layout used by `data/arabic/sentences-100k.json`.

## 2. SPM baseline (trivial)

Uses the existing `training/train_bpe.py` path:

```bash
python training/train_bpe.py \
    --input data/russian/sentences-100k.json \
    --vocab-size 8000 --model-prefix data/tokenized/spm-ru/ru-bpe-8000
python training/train_bpe.py \
    --input data/russian/sentences-100k.json \
    --vocab-size 32000 --model-prefix data/tokenized/spm-ru/ru-bpe-32000
```

Outputs go to
`data/tokenized/spm-ru/ru-bpe-{8000,32000}.{jsonl,-vocab.json}` to match the
paths expected in `PAIRS["ru"]`.

## 3. CST Russian tokenizer (the real work)

Two parallel implementations, matching the existing EN (TS) / AR (Python)
pattern:

### 3.1 Python edge tokenizer

- File: `edge/russian_tokenizer.py` (new), modeled on
  [edge/arabic_tokenizer.py](../edge/arabic_tokenizer.py).
- Pipeline:
  1. Unicode NFC + case-fold + light punctuation split.
  2. Lemmatize with **pymorphy3** (MIT license, actively maintained, handles
     OOV via probabilistic morphology). Get `{lemma, POS, grammemes}` per
     token.
  3. Map grammemes to CST feature tokens:
     - `FEAT:case:<nom|gen|dat|acc|ins|loc>`
     - `FEAT:num:<sg|pl>`
     - `FEAT:gender:<m|f|n>`
     - `FEAT:aspect:<perf|impf>` (verbs)
     - `FEAT:tense:<pres|past|fut>`
     - `FEAT:person:<1|2|3>`
     - `FEAT:voice:<act|pass>`
     - `FEAT:mood:<ind|imp|cond>`
  4. Map lemma to the shared semantic field inventory in
     [src/tokenizer/cst-spec.ts](../src/tokenizer/cst-spec.ts) via a
     Russian→CONCEPT/REL/ROLE lookup table (manually curated for high-frequency
     closed-class words, lemma-as-LIT for open-class OOV).
  5. Preserve lossless round-trip via `FEAT:lemma:<ru_lemma>` alongside the
     CST concept token (mirrors D2 in ROADMAP).
- Tests:
  - `edge/tests/test_russian_tokenizer.py` — 200-sentence fixture with gold
    token sequences.
  - Round-trip test: `detokenize(tokenize(s)) == s` for the full fixture.

### 3.2 TypeScript spec + lookup

- Extend [src/tokenizer/cst-spec.ts](../src/tokenizer/cst-spec.ts) with a
  `RU_LEXICON` block: closed-class Russian words → CST tokens (mirrors the
  English `EN_LEXICON` and Arabic `AR_LEXICON` blocks already there, if
  present; otherwise extract the pattern from the Python mappings).
- No runtime TS Russian pipeline required for Phase 0 (training happens in
  Python). TS implementation can wait until Phase 1 reasoning-mode work
  unless we need it sooner for the browser demo.

### 3.3 Build lookups

- Extend [edge/build_lookups.py](../edge/build_lookups.py) (or add an RU
  sibling) to produce:
  - Top-N Russian lemmas with frequency counts.
  - Coverage report: % of corpus tokens resolved to CONCEPT vs LIT.
  - Target: ≥80% of non-function-word tokens resolve to a CONCEPT/REL/ROLE
    (Arabic coverage is the benchmark to match).

## 4. CST tokenized corpus

- Outputs at the paths declared in `PAIRS["ru"]`:
  - `data/tokenized/cst-ru-8k/train-100000.jsonl` (+ `-vocab.json`)
  - `data/tokenized/cst-ru-32k/train-100000.jsonl` (+ `-vocab.json`)
- Vocab cap script: clone `training/cap_cst_vocab_ar.py` → `cap_cst_vocab_ru.py`.
  The capping logic is language-agnostic; only the input path changes.

## 5. Downstream task — RuSentiment

- Dataset: **RuSentiment** (public, ~30K posts labeled
  positive/neutral/negative/skip/speech). Use the 3-class subset
  (pos/neu/neg) for a clean HARD-analog.
- Preparation script (new):
  `training/experiments/prepare_downstream_ru.py` — tokenizes each post with
  **the same** CST-ru / SPM-ru pipeline as the LM training corpus, emits:
  - `data/downstream/rusentiment-ru-cst-8k-{train,test}.jsonl`
  - `data/downstream/rusentiment-ru-cst-32k-{train,test}.jsonl`
  - `data/downstream/rusentiment-ru-spm-8k-{train,test}.jsonl`
  - `data/downstream/rusentiment-ru-spm-32k-{train,test}.jsonl`
- Each line: `{"ids": [...], "label": 0|1|2}`.
- Consumed by `downstream_eval.py --task classification --num-labels 3`.

## 6. Order of operations

1. Corpus → 2. SPM-ru baseline (can train day 1, validates data pipeline) →
2. pymorphy3 lemma pipeline → 4. concept lexicon (iterative, target 80%
   coverage on a 1K sample first) → 5. full CST-ru tokenized corpora →
3. RuSentiment prep → 7. add `--langs ru` to the Colab driver run.

## 7. Risks / decisions to lock before coding

- **Unknown lemma policy.** pymorphy3 returns the surface form for OOV; we
  must decide: emit `LIT:<surface>` (lossless, bigger vocab) or
  `CONCEPT:unknown` + `FEAT:lemma:<surface>` (cleaner, risks dropping
  information). **Default: `LIT:<surface>`, matches Arabic.**
- **Concept inventory sharing.** The same `CONCEPT:write`, `CONCEPT:walk`
  etc. must be used across EN / AR / RU. Any RU-specific concepts that
  don't exist in the shared inventory require a PR to `cst-spec.ts`, not a
  language-local addition. This is what makes the cross-lingual BPC claim
  land.
- **pymorphy3 license.** MIT, compatible with the repo's LICENSE. No action
  needed, noted here for the record.
- **POS-tag granularity.** Russian has 17 UniMorph cases in principle; 6
  standard cases cover >99% of corpus. Start with the 6, add "vocative"
  /"partitive" only if coverage metric demands.

## 8. Done criteria for the Russian track of Phase 0

- [ ] `data/russian/sentences-100k.json` committed.
- [ ] `edge/russian_tokenizer.py` + tests green.
- [ ] Coverage report ≥80% CONCEPT-resolved on a held-out 5K sample.
- [ ] `data/tokenized/cst-ru-{8k,32k}/` and `data/tokenized/spm-ru/` populated.
- [ ] `data/downstream/rusentiment-ru-*-{train,test}.jsonl` populated.
- [ ] `colab_phase0.py --langs en ar ru --seeds 0 1 2 3 4` runs end-to-end.
- [ ] Russian results present in `docs/cst-paper.md` Robustness section and
      regenerated PDF.
