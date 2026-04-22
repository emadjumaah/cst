# Contextual Semantic Tokenization (CST)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending_Zenodo-grey.svg)](#citation)
[![Demo](https://img.shields.io/badge/Demo-Arabic_Algebra_Engine-green.svg)](https://emadjumaah.github.io/aae/)

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

### Arabic (100K Wikipedia sentences)

Trained on 100K Arabic Wikipedia sentences with the same GPT-2 architecture, **identical parameter counts**, and BPC evaluation:

| Tokenizer         | Vocab | Params | Tokens/sent | BPC ↓    |
| ----------------- | ----- | ------ | ----------- | -------- |
| **Arabic CST**    | 8K    | 6.8M   | **20.3**    | **1.15** |
| SentencePiece BPE | 8K    | 6.8M   | 30.1        | 2.12     |
| **Arabic CST**    | 32K   | 13.0M  | **20.3**    | **1.29** |
| SentencePiece BPE | 32K   | 13.0M  | 24.0        | 2.01     |

**46.0% BPC reduction** at 8K vocabulary. **35.8%** at 32K.
Arabic's triconsonantal root system is a natural fit for CST — the advantage over BPE is even larger than in English (46% vs 35.5%).

### Cross-lingual comparison

| Metric                | CST-8K   | SPM-8K   | CST-32K  | SPM-32K  |
| --------------------- | -------- | -------- | -------- | -------- |
| English BPC           | 1.13     | 1.75     | 1.23     | 1.65     |
| Arabic BPC            | 1.15     | 2.12     | 1.29     | 2.01     |
| **Cross-lingual gap** | **0.02** | **0.37** | **0.06** | **0.36** |

BPE creates a 21% performance penalty for Arabic (1.75 → 2.12). CST reduces the cross-lingual gap to 1.8% (1.13 → 1.15) — within measurement noise. A significant portion of Arabic's historically observed difficulty in language modeling is attributable to tokenization, not the language itself.

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

- **English:** [`docs/paper/cst-paper.md`](docs/paper/cst-paper.md)
- **Arabic:** [`docs/paper/cst-paper-ar.md`](docs/paper/cst-paper-ar.md) — published simultaneously in Arabic; to our knowledge the first NLP systems paper to do so.

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
.
├─ src/                      ← TypeScript tokenizer + pipeline + tests
│   ├─ tokenizer/            ← 7-stage core (CSTTokenizer)
│   ├─ pipeline/             ← download / process / stats / stream
│   └─ tests/                ← vitest suite
├─ edge/                     ← Python reference tokenizers + ONNX edge model
├─ reasoning/                ← reasoning-level tokenizer + data builders
├─ training/                 ← GPT-2 + BPE baseline training scripts
├─ scripts/                  ← utilities (pdf, data extract, parity check)
├─ data/                     ← corpora + tokenizer lookup tables (mostly gitignored)
├─ docs/
│   ├─ paper/                ← research paper (EN + AR)
│   ├─ spec/                 ← tokenizer specs (two-level, Arabic, English)
│   ├─ media/                ← media posts + press articles
│   └─ plans/                ← training + reasoning + research plans
├─ ROADMAP.md                ← single source of truth for what's next
├─ ARCHITECTURE.md           ← one-page system overview
├─ DATA.md                   ← data statement, provenance, licensing
├─ CONTRIBUTING.md
├─ CODE_OF_CONDUCT.md
├─ SECURITY.md
├─ CHANGELOG.md
└─ CITATION.cff
```

For more detail on any subproject, start from [`ARCHITECTURE.md`](ARCHITECTURE.md) and the [`docs/`](docs/README.md) index.

---

## Background

This project originated from the [Arabic Algebra Engine](https://emadjumaah.github.io/aae/), which organized 820+ Arabic roots into semantic domains based on the triconsonantal root system. The key observation: Arabic morphology is a formal algebra — root × pattern = concept — refined over fourteen centuries of linguistic scholarship. CST is the generalization of that algebra into a tokenization framework for neural language models.

---

## Regenerating Data

Training data (tokenized .jsonl files) is gitignored to keep the repo small. To regenerate:

```bash
# English: CST + SentencePiece BPE at 8K and 32K vocab
python training/cap_cst_vocab.py          # CST-8K and CST-32K
python training/train_bpe.py              # SPM-8K and SPM-32K

# Arabic: download 100K Wikipedia sentences + CST + SPM at 8K/32K
python training/arabic_experiment_v2.py --sentences 100000

# Coverage analysis (on cached 1K Arabic sentences)
python training/analyze_missed.py
```

**Requirements:** `pip install -r training/requirements.txt` + `camel_tools` with morphology DB (`camel_data -i morphology-db-msa-r13`).

---

## Documentation

| Topic | Link |
|-------|------|
| One-page architecture | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Roadmap and planned work | [`ROADMAP.md`](ROADMAP.md) |
| Research paper (EN / AR) | [`docs/paper/`](docs/paper) |
| Two-level tokenization spec | [`docs/spec/two-level-tokenization.md`](docs/spec/two-level-tokenization.md) |
| Arabic tokenizer spec | [`docs/spec/cst-arabic-tokenizers.md`](docs/spec/cst-arabic-tokenizers.md) |
| Training plan | [`docs/plans/TRAINING_PLAN.md`](docs/plans/TRAINING_PLAN.md) |
| Data statement | [`DATA.md`](DATA.md) |

See [`docs/README.md`](docs/README.md) for the full index.

---

## Contributing

Issues and pull requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). For security reports, see [`SECURITY.md`](SECURITY.md).

## Citation

If you use this work, please cite it via the [`CITATION.cff`](CITATION.cff) metadata (GitHub renders a "Cite this repository" button in the sidebar). A Zenodo DOI will be minted at first tagged release.

## License

Released under the [Apache License 2.0](LICENSE).
