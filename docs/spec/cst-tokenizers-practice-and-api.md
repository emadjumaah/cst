# CST Tokenizers in Practice: Standard vs Logic + Stable API

Status: working standardization guide
Last updated: 2026-04-26

This file is the practical reference for how we use CST tokenizers today, and the API/quality bar required to make CST a stable "tool" for any pretraining or reasoning pipeline.

Implementation prep status:

- Internal unified facade implemented at [edge/cst_api.py](../../edge/cst_api.py)
- Public exposure is intentionally deferred to a separate step

It complements (not replaces):

- [two-level-tokenization.md](two-level-tokenization.md)
- [cst-arabic-tokenizers.md](cst-arabic-tokenizers.md)
- [cst-english-tokenizers.md](cst-english-tokenizers.md)

---

## 1) The Two Tokenizer Standards

We keep two standards because we optimize for two different goals.

| Standard         | Goal                                                                    | Input/Output                                                                                                                                    | Where used                                            |
| ---------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **CST Standard** | Preserve maximum linguistic detail for language modeling and generation | text -> language-aware CST tokens (`ROOT`, `ROLE`/`CMP`, `REL`, `STR`, `FEAT`, `LIT`, etc.)                                                     | pretraining corpora, language modeling, data analysis |
| **CST Logic**    | Preserve truth-conditional reasoning structure in a small closed space  | CST-standard tokens (or formal text) -> closed ~151 logic tokens (`L:*`, `Q:*`, `R:*`, `T:*`, `M:*`, `RO:*`, `C:*`, `A:*`, `S:*`, `V:*`, `N:*`) | reasoning-model training/inference                    |

Core rule:

- Standard tokenizer: high-fidelity language representation.
- Logic tokenizer: compressed reasoning representation.
- They are separate models and separate vocabularies.

---

## 2) Current Architecture (As Implemented)

### 2.1 Canonical implementations

- Arabic standard tokenizer: [edge/arabic_tokenizer.py](../../edge/arabic_tokenizer.py)
- English standard tokenizer: [edge/english_tokenizer.py](../../edge/english_tokenizer.py)
- Unified CLI for standard tokenization: [edge/tokenize.py](../../edge/tokenize.py)
- Closed-vocab logic tokenizer: [edge/logic_tokenizer.py](../../edge/logic_tokenizer.py)

### 2.2 Reasoning pipeline path

The active reasoning corpus pipeline currently uses:

- [reasoning/tokenizer/arabic.py](../../reasoning/tokenizer/arabic.py)
- [reasoning/tokenizer/english.py](../../reasoning/tokenizer/english.py)
- [reasoning/tokenizer/projection.py](../../reasoning/tokenizer/projection.py)
- [reasoning/tokenize_corpus.py](../../reasoning/tokenize_corpus.py)

This means there are currently two logic-related paths in the repo:

1. closed-vocab logic path in `edge/logic_tokenizer.py`
2. reasoning projection path in `reasoning/tokenizer/*`

For long-term stability, these should converge to one canonical logic path.

---

## 3) Practical Usage Today

### 3.1 Standard tokenization (Arabic / English)

Use the unified CLI:

```bash
python -m edge.tokenize --lang ar --in input.txt --out ar.tokens.jsonl
python -m edge.tokenize --lang en --in input.txt --out en.tokens.jsonl
```

Useful Arabic flags:

- `--ar-root-pattern`
- `--ar-space-token`
- `--no-ar-atomic-composition`
- `--no-ar-critical-feat-only`

Useful English flag:

- `--no-en-atomic-composition`

### 3.2 Logic tokenization

Use the closed-vocab tokenizer directly when you want the 151-token reasoning space:

```python
from edge.logic_tokenizer import LogicTokenizer

tk = LogicTokenizer()

# project from CST standard token stream
logic_tokens = tk.from_standard(["REL:if", "ROOT:rain", "REL:then", "ROOT:earth"])
logic_ids = tk.to_ids(logic_tokens)

# formal text path
formal_logic_tokens = tk.from_formal("∀x. P(x) -> Q(x)")
```

### 3.3 Reasoning corpus build (current pipeline)

```bash
python -m reasoning.tokenize_corpus --in reasoning/out --out reasoning/tokenized
```

---

## 4) Arabic vs English/Other Languages: What Is Different

Arabic and English are aligned at the CST role level, but very different in analysis strategy.

| Dimension              | Arabic tokenizer                                                                        | English/other-language tokenizer                                                   |
| ---------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Morphology backend     | CAMeL morphology analyzer (root/pattern/clitic rich)                                    | spaCy or language-specific lemmatizer/POS/dependency stack                         |
| Word structure         | Root+pattern+clitics are first-class signals                                            | Lemma+syntax roles are primary signals                                             |
| Prefix/suffix behavior | Explicit proclitic/enclitic handling (`prc*`, `enc0`)                                   | Prefix/suffix heuristics + parser-driven roles                                     |
| Token richness         | Strong root identity (`ROOT:<field>` or `ROOT:<raw-root>`), pattern/role optional modes | Strong lemma identity (`ROOT:<lemma>`), optional atomic role split                 |
| Script handling        | Native Arabic script, Arabic punctuation normalization, Arabic numeral normalization    | Latin-centric by default; code-switching and numbers handled as literals/relations |
| Biggest quality risk   | Morphological ambiguity + dialect/OOV roots                                             | Parser/lemmatizer drift, entity tagging variance                                   |

Practical implication:

- Arabic gains more from morphology-aware tokenization because the language encodes meaning compositionally in roots/patterns and clitics.
- English and many other languages depend more on syntactic/lemma analysis quality than on root extraction.

---

## 5) Stable CST Tool API (Target Contract)

To make CST easy to plug into any pretraining stack, the API should be unified and explicit.

### 5.1 Required public methods

```text
tokenize(text, *, lang, level="standard"|"logic") -> TokenizeResult
tokenize_batch(texts, *, lang, level="standard"|"logic") -> list[TokenizeResult]

project(tokens, *, source="standard", target="logic", lang) -> list[str]

encode(tokens_or_text, *, vocab, lang, level) -> list[int]
decode(ids, *, vocab, lang, level) -> list[str]

untokenize(tokens, *, lang, level="standard") -> str
```

Notes:

- `untokenize` is **required** for standard level.
- `untokenize` is optional for logic level (logic is intentionally lossy).
- `project` must be deterministic and documented for every dropped token class.

### 5.2 Required output shape

```json
{
  "text": "...",
  "lang": "ar|en|...",
  "level": "standard|logic",
  "tokens": ["..."],
  "ids": [0, 1, 2],
  "meta": {
    "normalization": { "applied": true },
    "coverage": { "structured_ratio": 0.0 },
    "projection": { "dropped": 0, "mapped": 0 }
  }
}
```

### 5.3 Versioning rules

- Semantic token schema changes -> spec version bump.
- Projection-table changes -> rerun logic coverage tests.
- Normalization changes -> rerun round-trip tests.

---

## 6) "Perfect as Possible" Quality Bar

Use this as the release gate before calling CST a stable tool.

### 6.1 Architecture gates

1. One canonical standard tokenizer interface for all languages.
2. One canonical logic tokenizer path (no behavior drift across train/eval/demo).
3. Unified API surface and output schema.

### 6.2 Correctness gates

1. Standard round-trip: `untokenize(tokenize(normalize(s))) == normalize(s)` on held-out sets.
2. Logic projection completeness: every emitted standard token is mapped or explicitly dropped by policy.
3. Determinism: same input -> same output across runs.

### 6.3 Quality targets

1. Round-trip exact match >= 0.98 for standard tokenizer on held-out corpus.
2. Non-LIT logic `[UNK]` rate <= 1% on reasoning corpora.
3. Compression stability: logic/default ratio stays within declared bounds per language.

### 6.4 Minimum CI checks

```bash
# Arabic + logic unit tests
pytest edge/training/tests/test_tokenizer.py edge/training/tests/test_logic_tokenizer.py -q

# TypeScript tokenizer tests
npm test

# TS/Python parity sanity (English path)
python scripts/check_tokenizer_parity.py --sentences data/sentences-100.json --n 100
```

---

## 7) Current Status Snapshot (2026-04-26)

Validation run snapshot:

- Python tokenizer + API tests: 104 passed.
- TypeScript tests: 30 passed.

What is strong now:

- Clear canonical standard tokenizers in `edge/`.
- Closed-vocab logic tokenizer exists and is tested.
- Reasoning pipeline exists end-to-end.

What still blocks "stable tool" status:

1. Unified API exists internally but is not exposed as the official package/CLI entrypoint yet.
2. Standard-level `untokenize` remains a contract placeholder and is not implemented.
3. Logic path split remains (closed-vocab logic tokenizer vs separate reasoning projection pipeline).

---

## 8) Recommended Stabilization Order

1. Freeze and publish one canonical API contract (section 5).
2. Expose [edge/cst_api.py](../../edge/cst_api.py) through package/CLI entrypoints.
3. Implement `untokenize` for standard level and enforce round-trip tests.
4. Unify logic path so train/eval/demo all use one projection+vocab authority.
5. Add CI gates for projection coverage and unknown-rate budgets.

After these five are complete, CST can be used as a dependable tokenizer tool for any model pretraining pipeline.
