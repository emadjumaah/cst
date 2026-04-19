# Contextual Semantic Tokenization (CST)

**A linguistically-grounded alternative to subword tokenization for language modeling.**

> Arabic morphology has an algebraic structure: **root × pattern = concept**.
> The root ك-ت-ب (k-t-b, write) combined with the pattern فَاعِل (agent) produces كاتب (writer).
> CST generalizes this principle — encoding semantic field and morphological role directly into every token — and applies it to any language.

---

## Results

Trained on 100K English sentences with GPT-2 architecture at **identical parameter counts**, evaluated using **bits-per-character (BPC)** — a metric that is directly comparable across tokenizers:

| Tokenizer         | Vocab | Params | Tokens/sent | BPC ↓    |
| ----------------- | ----- | ------ | ----------- | -------- |
| **CST**           | 8K    | 6.8M   | **22.1**    | **1.13** |
| SentencePiece BPE | 8K    | 6.8M   | 31.7        | 1.75     |
| **CST**           | 32K   | 13.0M  | **22.1**    | **1.23** |
| SentencePiece BPE | 32K   | 13.0M  | 26.6        | 1.65     |

**35.5% BPC reduction** at 8K vocabulary. **25.2%** at 32K.
CST also trains **1.56× faster** due to 30% shorter token sequences.

---

## How It Works

Instead of statistical subword fragments, CST maps every word to a typed semantic token:

| Type   | Format           | Example           | Meaning                                |
| ------ | ---------------- | ----------------- | -------------------------------------- |
| `CMP`  | `CMP:field:role` | `CMP:write:agent` | "writer" — write field, agent role     |
| `ROOT` | `ROOT:field`     | `ROOT:move`       | semantic field only, no derivation     |
| `REL`  | `REL:relation`   | `REL:causes`      | grammatical or logical relation        |
| `STR`  | `STR:marker`     | `STR:negation`    | sentence-level structural marker       |
| `LIT`  | `LIT:surface`    | `LIT:the`         | function words, proper nouns, fallback |

**Example:** `"The researchers discovered that rewriting the algorithm significantly improved computational efficiency."`

```
STR:past  LIT:the  CMP:science:agent  CMP:know:past  REL:that
CMP:write:repeat  LIT:the  ROOT:think  CMP:quality:manner
CMP:fix:past  CMP:think:quality  CMP:work:state
```

12 tokens vs. 17 fragments from BPE-8K — and every CST token carries an explicit semantic label.

The tokenizer runs in **7 stages**: normalize → detect structure → split words → identify named entities → lemmatize → morphological decomposition → emit token. The vocabulary contains ~846 semantic tokens derived from ~45 universal semantic fields and ~2,400 lemma-to-field mappings. These fields are language-agnostic: "write" in English, "écrire" in French, and "كتب" in Arabic all resolve to the same token space.

---

## Paper

The full study is available in this repository:

- **English:** [`docs/cst-paper.md`](docs/cst-paper.md)
- **Arabic:** [`docs/cst-paper-ar.md`](docs/cst-paper-ar.md)

**Title:** _Contextual Semantic Tokenization: A Linguistically-Grounded Alternative to Subword Segmentation for Language Modeling_
**Author:** Emad Jumaah

The paper covers the conceptual origin in Arabic triconsonantal morphology, the full 7-stage tokenizer design, a controlled comparison against SentencePiece BPE at matched vocabulary sizes and parameter counts, experimental results, mechanistic analysis, cross-lingual properties, limitations, and future directions.

---

## Quick Start

```bash
npm install
npm test                # run the test suite
npx tsx src/demo.ts     # tokenize example sentences
```

**Requirements:** Node.js 18+, TypeScript.

---

## Repository Structure

```
src/
  tokenizer/
    index.ts              ← CSTTokenizer (7-stage processing)
    types.ts              ← token types and interfaces
    normalizer.ts         ← stage 1: text normalization
    structureDetector.ts  ← stage 2: sentence-level STR detection
    ner.ts                ← stage 4: named entity recognition
    morphology.ts         ← stage 6: prefix/suffix decomposition (9+25 rules)
    semanticFields.ts     ← stage 6: lemma → semantic field (~2,400 mappings)
    emitter.ts            ← stage 7: token emission with priority resolution
    vocabulary.ts         ← token ↔ ID registry
    data.ts               ← relation map (~245 entries) + function words (~95)
  tests/
    examples.test.ts      ← end-to-end tokenization
    morphology.test.ts
    normalizer.test.ts
    structure.test.ts
training/
  colab_train_fair.py     ← 4-way comparison: CST vs BPE at 8K and 32K vocab
  colab_train.py          ← GPT-2 training script
  cap_cst_vocab.py        ← constrain CST vocabulary to a target size V
  train_bpe.py            ← train SentencePiece BPE baseline
  train_gpt2.py           ← GPT-2 model utilities
  requirements.txt
docs/
  cst-paper.md            ← full paper (English)
  cst-paper-ar.md         ← full paper (Arabic / النسخة العربية)
```

---

## Background

This project originated from the [Arabic Algebra Engine](https://emadjumaah.github.io/aae/), which organized 820+ Arabic roots into semantic domains based on the triconsonantal root system. The key observation: Arabic morphology is a formal algebra — root × pattern = concept — refined over fourteen centuries of linguistic scholarship. CST is the generalization of that algebra into a tokenization framework for neural language models.
