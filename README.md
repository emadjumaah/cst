# Contextual Semantic Tokenization (CST)

**A linguistically-grounded alternative to subword tokenization for language modeling.**

> Arabic morphology has an algebraic structure: **root × pattern = concept**.
> The root ك-ت-ب (write) combined with the pattern فَاعِل (agent) produces كاتب (writer).
> CST generalizes this principle — encoding semantic field and morphological role directly into every token — and applies it to any language.

---

## Results

Trained on 100K English sentences with GPT-2 architecture, **identical parameter counts**, using **bits-per-character (BPC)** as the cross-tokenizer metric:

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

Instead of statistical subword fragments, CST produces typed semantic tokens:

| Type   | Format           | Example           | Meaning                                |
| ------ | ---------------- | ----------------- | -------------------------------------- |
| `CMP`  | `CMP:field:role` | `CMP:write:agent` | "writer" — write field, agent role     |
| `ROOT` | `ROOT:field`     | `ROOT:move`       | bare semantic field, no derivation     |
| `REL`  | `REL:relation`   | `REL:causes`      | grammatical / logical relation         |
| `STR`  | `STR:marker`     | `STR:negation`    | sentence-level structure               |
| `LIT`  | `LIT:surface`    | `LIT:the`         | function words, proper nouns, fallback |

**Example:** `"The researchers discovered that rewriting the algorithm significantly improved computational efficiency."`

```
STR:past  LIT:the  CMP:science:agent  CMP:know:past  REL:that
CMP:write:repeat  LIT:the  ROOT:think  CMP:quality:manner
CMP:fix:past  CMP:think:quality  CMP:work:state
```

12 tokens. BPE-8K produces 17 fragments that carry no semantic structure.

The pipeline has **7 stages**: normalize → structure detect → word split → NER → lemmatize → morphological decompose → emit. The vocabulary contains ~846 semantic tokens (CMP, ROOT, REL, STR) derived from ~45 universal semantic fields and ~2,400 lemma-to-field mappings. These are language-agnostic: "write" in English, "écrire" in French, and "كتب" in Arabic all map to the same field.

---

## Paper

The full study is available in this repository:

- **English:** [`docs/cst-paper.md`](docs/cst-paper.md)
- **Arabic:** [`docs/cst-paper-ar.md`](docs/cst-paper-ar.md)

**Title:** _Contextual Semantic Tokenization: A Linguistically-Grounded Alternative to Subword Segmentation for Language Modeling_
**Author:** Emad Jumaah

The paper covers: conceptual origin in Arabic morphology → 7-stage pipeline description → controlled fair comparison (matched vocab sizes, matched parameter counts) → results → discussion of mechanisms → cross-lingual properties → limitations and future work.

---

## Quick Start

```bash
npm install
npm test                # 30 tests across 4 files
npx tsx src/demo.ts     # see the tokenizer in action
```

**Requirements:** Node.js 18+, TypeScript.

---

## Repository Structure

```
src/
  tokenizer/
    index.ts              ← CSTTokenizer class (7-stage pipeline)
    types.ts              ← Token types, interfaces
    normalizer.ts         ← Stage 1: text normalization
    structureDetector.ts  ← Stage 2: STR token detection
    ner.ts                ← Stage 4: named entity recognition
    morphology.ts         ← Stage 6: prefix/suffix decomposition (9+25 rules)
    semanticFields.ts     ← Stage 6: lemma → semantic field (~2,400 mappings)
    emitter.ts            ← Stage 7: token emission with priority cascade
    vocabulary.ts         ← token ↔ ID registry
    data.ts               ← relation map (~245) + function words (~95)
  tests/
    examples.test.ts      ← end-to-end tokenization cases
    morphology.test.ts
    normalizer.test.ts
    structure.test.ts
training/
  colab_train_fair.py     ← fair 4-way comparison (CST vs BPE, 8K vs 32K)
  colab_train.py          ← original training script
  cap_cst_vocab.py        ← cap CST vocabulary to target size V
  train_bpe.py            ← train SentencePiece BPE baseline
  train_gpt2.py           ← GPT-2 training utilities
  requirements.txt
docs/
  CST_PAPER_FINAL.md      ← full paper (English)
  CST_PAPER_FINAL_AR.md   ← full paper (Arabic / النسخة العربية)
```

---

## Background

This project began with the [Arabic Algebra Engine](https://emadjumaah.github.io/aae/), which organized 820+ Arabic roots into semantic domains based on the triconsonantal root system. The key observation: Arabic morphology is a formal algebra that has operated for 1,400 years. CST is the generalization of that algebra into a tokenization framework for neural language models.
