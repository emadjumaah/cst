# Two-Level Tokenization — Shared Specification

**Status:** Draft v0.1
**Scope:** Language-agnostic contract for the CST two-level tokenizer family.
**Addenda:** [cst-arabic-tokenizers.md](cst-arabic-tokenizers.md), [cst-english-tokenizers.md](cst-english-tokenizers.md)

---

## 1. Motivation

A single tokenizer cannot serve two very different downstream goals:

- **Reasoning** needs short, dense, semantically stable sequences. Surface noise (inflection, clitics, diacritics, casing, punctuation) dilutes attention and bloats context without changing the proposition.
- **Generation / reconstruction** needs every byte required to produce fluent, correct surface text. Dropping a clitic or a diacritic breaks output.

We therefore build **two tokenizers per language**, tied by a deterministic projection.

## 2. The Two Levels

| Level         | Optimizes for                                          | Granularity                                                     | Lossy on                                              | Lossless on                                  |
| ------------- | ------------------------------------------------------ | --------------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| **Reasoning** | semantic compression, logical stability, short context | **coarser** — one token ≈ one concept (ROOT, REL, CMP, STR)     | surface form, inflection, clitics, diacritics, casing | proposition, roles, relations, structure     |
| **Default**   | surface fidelity, generation, round-trip to text       | **finer** — morphemes, clitics, diacritics, punctuation, casing | nothing recoverable from normalized input             | full surface (modulo declared normalization) |

**Key rule.** The default tokenizer is _strictly_ a refinement of the reasoning tokenizer. It never emits less information. "As detailed as possible" applies only to **default**; **reasoning** must be as _coarse_ as possible without losing logic.

## 3. Shared Role Inventory

All four tokenizers emit tokens drawn from the same role set defined in
[../src/tokenizer/cst-spec.ts](../src/tokenizer/cst-spec.ts):

- `ROOT:<field>` — content concept (verb/noun root, lemma + semantic field)
- `REL:<type>` — relation / function word (conjunction, preposition, negation, quantifier, discourse marker)
- `CMP:<role>` — compositional role derived from morphological pattern (agent, patient, instrument, place, time, …)
- `STR:<marker>` — sentence-level structure (question, conditional, imperative, quote, clause boundary)
- `LIT:<value>` — literal preserved verbatim (numerals, named entities, code-switched tokens)

Languages populate these roles differently; the role names are shared so models can transfer across languages.

## 4. The Projection Contract

Let `T_D` be the default tokenizer and `T_R` the reasoning tokenizer for a language. Define a projection

$$\pi : \text{tokens}(T_D) \to \text{tokens}(T_R) \cup \{\varnothing\}$$

that maps each default token to exactly one reasoning token or drops it.

**Required properties.**

1. **Determinism.** `π` is a pure function of the default token and its local context window (≤ fixed _k_ tokens). No global state.
2. **Idempotence.** `π(π(x)) = π(x)` when the codomain is embedded back into the default vocabulary.
3. **Monotonicity.** For every input string _s_: `T_R(s) == collapse(π(T_D(s)))`, where `collapse` removes `∅` and merges adjacent duplicates per the role-specific rule.
4. **Role preservation.** `π` never changes the _role_ of a surviving token (a `ROOT:write` default token projects to a `ROOT:write` reasoning token, never to `REL:*`).
5. **Drop set is explicit.** Every token that `π` maps to `∅` must match a documented rule in the language addendum (e.g. "drop diacritics," "drop definite article," "drop tense-only auxiliaries").

## 5. Round-Trip Guarantees

| Guarantee                                    | Reasoning         | Default                  |
| -------------------------------------------- | ----------------- | ------------------------ |
| `detok(tok(s)) == normalize(s)`              | ❌ (not required) | ✅ (required)            |
| `tok(s₁) == tok(s₂)` for logical paraphrases | ✅ (goal)         | ❌ (not required)        |
| Stable under inflection of content words     | ✅                | ❌                       |
| Stable under clitic attachment               | ✅                | ❌                       |
| Stable under diacritization variation        | ✅                | ✅ (after normalization) |

`normalize` is the language-specific normalization step declared in the addendum. Anything the normalizer strips is _not_ required to round-trip.

## 6. Evaluation Protocol

Each tokenizer ships with a test suite proving its level's guarantee.

### 6.1 Default tokenizer

- **Reconstruction.** Char-level F1 ≥ 0.99 and exact-match ≥ 0.95 on a held-out corpus after `detok ∘ tok ∘ normalize`.
- **Coverage.** OOV rate < declared threshold (per addendum) on the target domain.
- **Invertibility.** For every token emitted, the detokenizer has a rule to produce surface bytes.

### 6.2 Reasoning tokenizer

- **Logic preservation.** On curated entailment / paraphrase pairs (≥ 1k per language), `T_R(premise)` and `T_R(hypothesis)` relationships match gold labels above a declared baseline. Tokenization must not flip entailment.
- **Compression ratio.** `|T_R(s)| / |T_D(s)| ≤ declared_ratio` on the target domain (target: ≤ 0.6).
- **Stability.** For a set of (base, inflected) pairs, `T_R(base) == T_R(inflected)` for the content subsequence.

### 6.3 Projection

- **Contract tests.** Properties 1–5 from §4 verified on a corpus sample ≥ 10k sentences.
- **No information leak.** No reasoning token carries information not derivable from its default preimage.

## 7. What Each Addendum Must Specify

For each language, the addendum fills in:

1. **Normalization pipeline** (NFKC, diacritics policy, letter unification, casing).
2. **Reasoning inventory** — concrete role populations and vocab size target.
3. **Default inventory** — subword / morpheme scheme and vocab size target.
4. **Drop set** for `π` with examples.
5. **Merge/split rules** for clitic attachment, compound words, numerals, named entities.
6. **Failure modes** — dialect, code-switching, OOV roots, rare patterns.
7. **Evaluation datasets** used for §6.

## 8. Non-Goals

- Unified cross-language vocabulary. Roles are shared; vocab tables are not.
- Replacing BPE universally. Default tokenizer _may_ wrap a BPE/Unigram model; it must still satisfy §5.
- Serving as a parser. CST roles are tokenizer-level approximations, not a syntactic tree.

## 9. File Layout

```
src/tokenizer/
  cst-spec.ts              # role definitions (source of truth)
  <lang>/
    reasoning.ts           # T_R
    default.ts             # T_D
    project.ts             # π
    normalize.ts
    tests/
      roundtrip.test.ts    # §6.1
      logic.test.ts        # §6.2
      projection.test.ts   # §6.3
```

Arabic currently lives in `edge/arabic_tokenizer.py` (Python, tied to CAMeL Tools) and will be mirrored/ported under `src/tokenizer/ar/` per the addendum.

## 10. Change Control

Changes to the role inventory in §3 require updating both addenda and bumping the `cst-spec.ts` version. Changes to a language's drop set require re-running §6 for that language only.
