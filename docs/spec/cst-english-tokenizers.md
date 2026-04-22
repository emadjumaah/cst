# English Tokenizers — Addendum

**Parent spec:** [two-level-tokenization.md](two-level-tokenization.md)
**Status:** Draft v0.1
**Reference implementation:** [../src/tokenizer/](../src/tokenizer/) (shared CST spec) + to-be-added `src/tokenizer/en/`
**Morphology backend:** lemmatizer + POS tagger (spaCy / Stanza) or lightweight rule-based fallback

---

## 1. Normalization Pipeline

Applied before either tokenizer runs.

1. NFKC unicode normalization.
2. Smart quotes → straight quotes (`“ ” ‘ ’` → `" ' '`).
3. Dashes unified: `– —` → `-` with a length feature preserved in default.
4. Casing: **default** preserves case as a feature (`lower` / `upper` / `title` / `mixed`); **reasoning** drops case entirely.
5. Contractions expanded with alignment metadata (`don't` → `do`+`not` with offsets). Default stores both; reasoning keeps only the expansion.
6. Numbers: digit runs preserved; spelled-out numbers normalized to `LIT:<n>` at reasoning level only.
7. Whitespace collapsed; punctuation preserved in default, mapped to `STR:*` or dropped in reasoning.

## 2. Reasoning Tokenizer (`T_R^en`)

**Target vocab size:** ~2k roles + ~12k lemmas ≈ **14k**.
**Target compression:** `|T_R| / |T_D| ≤ 0.5`.

### Inventory

- **`ROOT:<field>`** — lemma mapped to a semantic field (shared field set with Arabic where possible: `write`, `know`, `speak`, `think`, `see`, `feel`, `move`, `give`, …). Unknown lemmas fall back to `ROOT:<lemma>`.
- **`REL:<type>`** — function words and discourse markers:
  - coordinators (`and → REL:and`, `or → REL:or`, `but → REL:contrast`)
  - prepositions (`in → REL:in`, `on → REL:on`, `with → REL:with`, `for → REL:for`, `by → REL:by`)
  - negation (`not`, `no`, `never` → `REL:neg`)
  - quantifiers (`all`, `some`, `every`, `any` → `REL:quant:*`)
  - discourse (`because → REL:cause`, `however → REL:contrast`, `therefore → REL:entail`)
- **`CMP:<role>`** — dependency-derived roles: `agent`, `patient`, `recipient`, `instrument`, `location`, `time`, `manner`. Pulled from parser labels (`nsubj → agent`, `dobj → patient`, `iobj → recipient`, `obl:instr → instrument`, `obl:loc → location`, `obl:tmp → time`).
- **`STR:<marker>`** — `STR:question` (`?` or wh-inversion), `STR:conditional` (`if`, `unless`), `STR:imperative` (bare-verb root at clause start), `STR:quote_open` / `STR:quote_close`, `STR:clause_end` (`.`, `;`).
- **`LIT:<value>`** — numerals, named entities (from `ner.ts`), URLs, code spans, out-of-script tokens.

### Drop set (maps to ∅ under π)

- Articles `a`, `an`, `the`.
- Tense-only auxiliaries that don't flip truth (`do`-support in questions keeps only `STR:question`; `will` / `shall` project to `STR:future`; `have` / `had` in perfect tenses drop when aspect isn't truth-conditional).
- Case (pronouns retain identity; `he` / `him` both project to `ROOT:he`, with gender/number as features in default only).
- Plural `-s`, 3sg `-s`, past `-ed`, progressive `-ing` — surface-only; lemma survives.
- Possessive `'s` — becomes `REL:of` relation on the head noun.
- Most punctuation except clause/sentence boundaries and quotes.

### Collapse rules

- Adjacent identical `REL` collapse (`REL:and` + `REL:and` → one).
- Multi-word prepositions (`in front of`, `because of`) collapse to a single `REL:*`.
- Numeral + unit (`5 km`) stays as `LIT:5` + `ROOT:length` (not `LIT:5km`).

## 3. Default Tokenizer (`T_D^en`)

**Target vocab size:** ~32k subword units via BPE / Unigram trained on target corpus.
**Target OOV:** < 0.1% (subword model guarantees fallback).

### Inventory (superset of reasoning)

Every reasoning token **plus**:

- **Subword pieces** for any `ROOT:<lemma>` whose surface form isn't in the lemma vocab — emitted as `SUB:<piece>` with boundary markers.
- **Inflection features on `ROOT` / `CMP`** — `[tense=past, aspect=perf, num=pl, person=3, case=nom, voice=passive, degree=comp]`.
- **Casing feature** on every token.
- **Auxiliaries** as real tokens: `AUX:do`, `AUX:have`, `AUX:be`, `AUX:will` with tense features.
- **Articles** as tokens: `DET:a`, `DET:the`.
- **Contractions** as paired tokens with `contracted=true` flag so detokenizer re-contracts.
- **Punctuation tokens** — `.`, `,`, `;`, `:`, `!`, `?`, quotes, parentheses, dashes.
- **Whitespace markers** where needed for exact reconstruction (e.g. around em-dash).

### Merge/split rules

- Lemmatizer splits inflected form into `LEMMA + [features]`.
- BPE runs _only_ on unknown lemmas; known lemmas stay whole.
- Compound words (`well-known`, `state-of-the-art`) keep hyphens as visible tokens.
- URLs / emails / code spans tokenized as single `LIT:*`.

## 4. Projection π (default → reasoning)

| Default token                           | π maps to                                         |
| --------------------------------------- | ------------------------------------------------- |
| `ROOT:<l>`                              | `ROOT:<field(l)>` (or `ROOT:<l>` if unknown)      |
| `SUB:<piece>`                           | ∅ (pieces reassemble to the lemma which projects) |
| `CMP:<role>[features]`                  | `CMP:<role>`                                      |
| `REL:<t>`                               | `REL:<t>`                                         |
| `STR:<m>`                               | `STR:<m>`                                         |
| `LIT:<v>`                               | `LIT:<v>`                                         |
| `DET:the` / `DET:a` / `DET:an`          | ∅                                                 |
| `AUX:do` (support)                      | ∅                                                 |
| `AUX:will` / `AUX:shall`                | `STR:future`                                      |
| `AUX:have` (perfect)                    | ∅                                                 |
| `AUX:be` (progressive)                  | ∅                                                 |
| `AUX:be` (passive)                      | `REL:passive`                                     |
| possessive `'s`                         | `REL:of`                                          |
| negation `not` / `n't`                  | `REL:neg`                                         |
| `.` / `;`                               | `STR:clause_end`                                  |
| `?`                                     | `STR:question`                                    |
| `"` / `"`                               | `STR:quote_open` / `STR:quote_close`              |
| other punctuation                       | ∅                                                 |
| casing / whitespace / contraction flags | ∅                                                 |

## 5. Failure Modes

- **Code-switching.** Non-Latin runs become `LIT:*`; if the other language has its own tokenizer, the caller decides which to invoke.
- **Informal text / typos.** Lemmatizer falls back to BPE; reasoning emits `ROOT:<surface>` and logs for vocabulary review.
- **Sarcasm / irony / negation scope.** Out of scope — tokenizer preserves surface; downstream model handles semantics.
- **Heavy punctuation (code, math).** Code spans detected by fencing / heuristics and emitted as `LIT:code`.
- **Named entities vs common nouns.** NER from `ner.ts`; on low confidence, default keeps `ROOT:<lemma>` and reasoning does the same (no `LIT` promotion).

## 6. Evaluation Datasets

- **Reconstruction (default):** held-out slice of `data/sentences-1k.json` English subset + Wikipedia sample.
- **Logic preservation (reasoning):** SNLI / MultiNLI mini-slice (≥ 1k pairs) — check that tokenization doesn't flip entailment; PAWS for paraphrase stability.
- **Inflection stability (reasoning):** lemma–inflection pairs from UniMorph English; expect identical reasoning sequences.
- **Compression ratio:** measured on `data/sentences-1k.json` English subset.

## 7. Open Items

- Choose morphology backend: spaCy (fast, decent) vs Stanza (better, slower) vs rule-based (fastest, brittle).
- Decide BPE corpus: same as CST training corpus, or a larger generic English corpus for default only.
- Cross-lingual `ROOT:<field>` alignment with Arabic — share a field registry in `src/tokenizer/semanticFields.ts`.
- Confirm whether `AUX:be` (passive) projects to `REL:passive` or absorbs into the `CMP:patient` role.
