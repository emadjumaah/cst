# Arabic Tokenizers — Addendum

**Parent spec:** [two-level-tokenization.md](two-level-tokenization.md)
**Status:** Draft v0.1
**Reference implementation:** [../edge/arabic_tokenizer.py](../edge/arabic_tokenizer.py)
**Morphology backend:** CAMeL Tools (`MorphologyDB` + `Analyzer`)

---

## 1. Normalization Pipeline

Applied before either tokenizer runs.

1. NFKC unicode normalization.
2. Alef unification: `{أ, إ, آ, ٱ} → ا` (default keeps a feature flag recording the original; reasoning discards).
3. Yeh / Alef-maqsura unification: `ى → ي` at word-final when analyzer confirms.
4. Teh-marbuta policy: kept as `ة` in default; folded to `ه` / dropped by role rule in reasoning.
5. Tatweel (`ـ`) removed.
6. Diacritics (`ً ٌ ٍ َ ُ ِ ّ ْ`): **default** keeps them as features; **reasoning** drops entirely.
7. Digit unification: Arabic-Indic `٠–٩` → ASCII `0–9` (both levels).
8. Whitespace collapsed; punctuation preserved in default, mapped to `STR:*` or dropped in reasoning.

## 2. Reasoning Tokenizer (`T_R^ar`)

**Target vocab size:** ~4k roles + ~6k roots ≈ **10k**.
**Target compression:** `|T_R| / |T_D| ≤ 0.55`.

### Inventory

- **`ROOT:<field>`** — derived from CAMeL root (e.g. `ك.ت.ب`) then mapped via `ARABIC_ROOT_TO_FIELD` to a semantic field (`write`, `know`, `speak`, …). Unknown roots fall back to `ROOT:<root>` verbatim.
- **`REL:<type>`** — function words via `ARABIC_REL_MAP` (و, أو, لكن, في, من, إلى, على, لا, ما, كل, بعض, …).
- **`CMP:<role>`** — derived from وزن (pattern) via `ARABIC_PATTERN_TO_ROLE` (e.g. `فاعل → agent`, `مفعول → patient`, `مِفعال → instrument`, `مَفعِل → place`).
- **`STR:<marker>`** — question (`هل`, `أ‍َ`), conditional (`إذا`, `إن`, `لو`), imperative mood, quote boundaries, clause `،` / `.` / `؛`.
- **`LIT:<value>`** — numerals, NEs from `ner.ts`, Latin-script code-switches, `ARABIC_LIT_WORDS`.

### Drop set (maps to ∅ under π)

- All diacritics.
- Definite article `ال` (captured as definiteness feature in default only).
- Clitics PRC0–PRC3 (و-, ف-, ل-, ب-, ك-, س-, pronominal suffixes) — their _semantic_ contribution is absorbed into the adjacent `REL` or `CMP` token when it alters meaning (e.g. future `س-` → `STR:future`); pure agreement clitics drop.
- Gender/number/case inflection on nouns and verbs.
- Teh-marbuta as a standalone feature.
- Tanwin.

### Collapse rules

- Adjacent `REL:and` + `REL:and` collapse to one.
- `STR:future` immediately before a `ROOT:*` stays adjacent (not merged).
- Numeral sequences collapse to a single `LIT:<number>`.

## 3. Default Tokenizer (`T_D^ar`)

**Target vocab size:** ~32k (roots × top patterns × clitic combos, bounded by `cap_cst_vocab_ar.py`).
**Target OOV:** < 0.5% on MSA Wikipedia; < 2% on dialectal.

### Inventory (superset of reasoning)

Every reasoning token **plus**:

- **Clitic tokens** — `PRC0:و`, `PRC1:ل`, `PRC2:ب`, `PRC3:ال`, `ENC0:ه`, etc., emitted in their surface order.
- **Inflection features attached to `ROOT` / `CMP`** — `[gender=m, num=sg, case=nom, state=def, voice=act, aspect=perf, person=3]`.
- **Diacritic tokens** — emitted as zero-width features on the host (not standalone).
- **Pattern token** — explicit `PAT:<wazn>` adjacent to the root.
- **Punctuation tokens** — `.`, `،`, `؟`, `؛`, quotes, parentheses.
- **Whitespace markers** where needed for exact reconstruction.

### Merge/split rules

- Analyzer segments word into `[PRC3][PRC2][PRC1][PRC0] stem [ENC0]`.
- Stem decomposed into `ROOT + PAT`.
- Each segment becomes its own token; detokenizer reassembles using the inverse of §1 plus diacritic restoration from features.

## 4. Projection π (default → reasoning)

| Default token                    | π maps to                                                              |
| -------------------------------- | ---------------------------------------------------------------------- |
| `ROOT:<r>`                       | `ROOT:<field(r)>` (or `ROOT:<r>` if unknown)                           |
| `PAT:<wazn>`                     | ∅ (its role contribution already encoded in `CMP:*` emitted alongside) |
| `CMP:<role>[features]`           | `CMP:<role>` (features dropped)                                        |
| `REL:<t>`                        | `REL:<t>`                                                              |
| `STR:<m>`                        | `STR:<m>`                                                              |
| `LIT:<v>`                        | `LIT:<v>`                                                              |
| `PRC0:و` (conjunction)           | `REL:and`                                                              |
| `PRC0:ف` (sequence)              | `REL:then`                                                             |
| `PRC1:ل` (purpose/dative)        | `REL:for`                                                              |
| `PRC2:ب` (instrument/comitative) | `REL:with`                                                             |
| `PRC2:س` / `سوف`                 | `STR:future`                                                           |
| `PRC3:ال` (definite article)     | ∅                                                                      |
| `ENC0:*` pronominal              | ∅ (role already in CMP)                                                |
| diacritic / tanwin / tatweel     | ∅                                                                      |
| punctuation `.` / `،` / `؛`      | `STR:clause_end`                                                       |
| punctuation `؟`                  | `STR:question`                                                         |

## 5. Failure Modes

- **Dialectal input.** Analyzer coverage drops; fallback: treat stem as literal `ROOT:<stem>` and skip pattern role.
- **Unknown root.** Emit `ROOT:<root>` verbatim in both levels; log for vocabulary review (`analyze_missed.py`).
- **Code-switching (Latin).** Whole run becomes `LIT:*` in both levels.
- **Ambiguous analysis.** CAMeL returns multiple analyses; pick highest-probability; record alternate as feature in default only.
- **Numerals with units.** `٣ كم` → `LIT:3` + `ROOT:length` (not `LIT:3km`).

## 6. Evaluation Datasets

- **Reconstruction (default):** held-out slice of `data/arabic/` (MSA Wikipedia + news).
- **Logic preservation (reasoning):** curated MSA paraphrase / entailment pairs under `edge/training/eval/ar_logic/` (to be built; target ≥ 1k pairs).
- **Inflection stability (reasoning):** lemma–inflection pairs generated from CAMeL paradigms; expect identical reasoning sequences.
- **Compression ratio:** measured on `data/sentences-1k.json` Arabic subset.

## 7. Open Items

- Port `edge/arabic_tokenizer.py` contract into `src/tokenizer/ar/` TypeScript once CAMeL-equivalent morphology is available (or wrap via subprocess for the POC).
- Decide whether `STR:future` should absorb `س-` _and_ `سوف` or only the clitic form.
- Grow `ARABIC_PATTERN_TO_ROLE` coverage; current map is partial.
