# Data Statement

This document describes every data source used in the CST experiments, the licenses under which they are distributed, and the preprocessing applied. It is written to satisfy the data-statement requirements of ACL-family venues and artifact reviewers.

## 1. English corpus

- **Source**: `agentlans/high-quality-english-sentences` on Hugging Face Datasets.
- **Upstream origin**: English Wikipedia.
- **License**: CC BY-SA 3.0 / 4.0 (Wikipedia).
- **Access**: `https://huggingface.co/datasets/agentlans/high-quality-english-sentences` (Parquet format).
- **Volume used**: 99,963 sentences.
- **Domain coverage**: science, history, geography, technology, biography.
- **Language variety**: Standard written English.
- **Preprocessing**:
  - No additional filtering beyond what the upstream dataset applies.
  - Train/validation split: 90/10, deterministic slice (not shuffled). See §4 for limitations this imposes.

## 2. Arabic corpus

- **Source**: `wikimedia/wikipedia` on Hugging Face Datasets, configuration `20231101.ar`.
- **Upstream origin**: Arabic Wikipedia snapshot, 2023-11-01.
- **License**: CC BY-SA 4.0.
- **Access**: HuggingFace Datasets Server API (paged).
- **Volume used**: 100,000 sentences.
- **Language variety**: Modern Standard Arabic (MSA). No dialectal or Classical Arabic is included.
- **Preprocessing** (see `training/arabic_experiment_v2.py`):
  - Split article text on `.`, `!`, `؟`.
  - Filter by length: 20 ≤ chars ≤ 300.
  - Filter by Arabic-character ratio: ≥ 50% of characters in the Unicode Arabic block `U+0600–U+06FF`.
  - Deduplication: none beyond upstream.
  - Train/validation split: 90/10, deterministic slice.

## 3. Linguistic resources

### 3.1 English lemmatization and NER

- **Tool**: `compromise.js` v14.14.3.
- **License**: MIT.
- **Role**: lemmatization (stage 5), named-entity recognition (stage 4).

### 3.2 Arabic morphological analysis

- **Tool**: CAMeL Tools (Obeid et al., 2020).
- **License**: MIT.
- **Morphology DB**: `morphology-db-msa-r13` (MSA, release 13). Install via `camel_data -i morphology-db-msa-r13`.
- **Role**: root extraction for every Arabic token.

### 3.3 Tokenization baselines

- **SentencePiece** v0.2.1 (BPE mode).
- **License**: Apache-2.0.
- **Trained on**: the same corpus as each CST run (English 99,963 sentences; Arabic 100,000 sentences).

## 4. Known limitations and risks

### 4.1 Dictionary–data leakage

The English lemma→field dictionary (~2,400 mappings) and Arabic root→field dictionary (~560 mappings) were iteratively constructed by the author. During development, the author inspected coverage on the same Wikipedia corpora that are used for evaluation. This constitutes a form of implicit test-set exposure.

Mitigation (planned): evaluate CST on a second, held-out corpus the dictionary was never tuned against (e.g., CC-News for English, Arabic news / OSCAR for Arabic).

### 4.2 Deterministic splits

The train/validation split is a deterministic 90/10 slice rather than a shuffled split. This is reproducible but means the validation distribution can differ systematically from the training distribution if the upstream dataset is not randomized.

### 4.3 Single language variety

- English: written Wikipedia only.
- Arabic: MSA Wikipedia only. No dialect, Classical, or news text.

Results should be interpreted as "CST on MSA Wikipedia," not "CST on Arabic." Dialect robustness is explicitly not tested.

### 4.4 Weak-root handling

The Arabic tokenizer uses a `#` wildcard index to match weak-letter alternations returned by CAMeL Tools. This is heuristic; error rate is not measured.

### 4.5 Proper nouns

The Arabic pipeline does not have a dedicated NER step. Proper nouns are routed via the generic SURF fallback.

## 5. Ethical considerations

- No personally identifiable information (PII) is knowingly included. Wikipedia content is public, but may mention named individuals.
- No downstream applications of this work are designed for surveillance, profiling, or demographic inference.
- The authored dictionaries encode the researcher's intuitions about semantic fields and may reflect cultural or linguistic biases. These biases are not audited.

## 6. Reproducibility artifacts

Tokenized `.jsonl` files and SentencePiece models are regenerated deterministically (given the dataset snapshots above) by:

```bash
# English
python training/train_bpe.py
python training/cap_cst_vocab.py

# Arabic (all stages)
python training/arabic_experiment_v2.py --sentences 100000
python training/cap_cst_vocab_ar.py
```

See `training/requirements.txt` for the pinned dependency set.

## 7. Citation

If you use these datasets, please also cite:

- **Wikipedia**: Wikimedia Foundation, `wikimedia/wikipedia` on Hugging Face Datasets.
- **CAMeL Tools**: Obeid et al. (2020). _CAMeL Tools: An Open Source Python Toolkit for Arabic Natural Language Processing_. LREC.
- **SentencePiece**: Kudo & Richardson (2018). _SentencePiece: A simple and language independent subword tokenizer and detokenizer_. EMNLP System Demonstrations.
