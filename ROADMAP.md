# CST-POC — Consolidated Roadmap

**Status:** Active. Supersedes the scattered plans in this folder.
**Last updated:** 2026-04-21
**Owner:** Emad

This document is the single source of truth for where the project is going. It
reconciles the research-mode conversations (`conv-sonnet.md`,
`conv-sonnet-2.md`), the phased scaling plan (`ARABIC_REASONING_MODEL.md`), the
edge deployment roadmap (`OPUS_EDGE_ARABIC_MODEL.md`), the lossless-encode
design (`OPUS_INFERENCE_CACHE.md`), and the publication-rigor plan
(`research-assessment-plan.md`) into one ordered program of work.

Older plan files remain in the folder for historical reference but should be
read through this roadmap, not in isolation.

---

## 1. Thesis

CST is not "a tokenizer." It is a **vocabulary of meaning** that the transformer
is allowed to see directly. Every token has a human-defined semantic role, so:

- The model trains on structure, not surface statistics.
- Every attention weight corresponds to a readable semantic relationship.
- Errors can be traced to specific token interactions and corrected
  surgically, without retraining from scratch.
- The same vocabulary works across languages because meaning, not morphology,
  is what's tokenized.

The project's deliverable is not one model. It is a **neurosymbolic stack**
where every component — tokenizer, model, memory, verifier, router — speaks
CST tokens.

---

## 2. Current State (what actually exists on disk)

| Area                                    | Artifact                                                                            | Status                  |
| --------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------- |
| Tokenizer — English                     | `src/tokenizer/` (11 files, TS)                                                     | Shipped, tested         |
| Tokenizer — Arabic                      | `edge/arabic_tokenizer.py` (spec v1.0)                                              | Shipped                 |
| Paper                                   | `docs/cst-paper.md` + AR version, OSF preprint                                      | Published (single-seed) |
| Models                                  | `edge/artifacts/model.onnx`, `model_int8.onnx` (100K Arabic, 6.8M params, BPC 1.15) | Shipped                 |
| Data                                    | English 100K–1K corpora; Arabic 100K tokenized at 8K/32K vocab caps                 | Shipped                 |
| 1M scale                                | Colab scripts ready, **not run**                                                    | Pending                 |
| Edge demo                               | `edge/demo/`                                                                        | Not built               |
| Multi-seed validation                   | —                                                                                   | Not done                |
| Reasoning-mode tokenizer                | —                                                                                   | Not started             |
| Verifier / scratchpad / memory / router | —                                                                                   | Not started             |

Related but independent: `/Users/emad/projects/arabic-algebra/arabic-algebra-engine`
(symbolic engine, 820 roots, 74 rules). Candidate to become the verifier and
output router in later phases. **Not currently wired to the LLM.**

---

## 3. Architectural Decisions (locked)

These decisions come out of the Sonnet conversations and replace earlier
ambiguity. New work conforms to them.

- **D1 — Two tokenizer modes, one codebase.**
  - `mode: 'general'` — ~8K vocab, full surface/morphology detail. Used for
    generation, translation, fluent text.
  - `mode: 'reasoning'` — ~150 vocab, collapsed semantic classes
    (`NEG`, `QUERY`, `CAUSE`, `COND`, `CONCEPT:*`, `ROLE:*`, `TIME:*`, `REL:*`).
    Used for inference, interpretability, verification experiments.
  - Same `Vocabulary`, `Normalizer`, `Emitter` — mode switched by config.

- **D2 — Lossless round-trip is a first-class requirement.**
  - Adopt the `FEAT:root:<arabic_root>` token from `OPUS_INFERENCE_CACHE.md` in
    general mode. Reasoning mode is deliberately lossy.

- **D3 — The transformer stays vanilla.**
  - No custom attention, no new architectures. GPT-2 small / nanoGPT.
  - All novelty lives in the vocabulary and the scaffolding around the model.

- **D4 — Every component speaks CST tokens.**
  - Memory entries, retrieval results, scratchpad contents, verifier rules,
    and router inputs are all CST token sequences. No intermediate text form.

- **D5 — Interpretability is a testable hypothesis, not marketing.**
  - Every claim of the form "the model learns logic" must be backed by
    attention-pattern analysis with a published protocol.

---

## 4. Roadmap

Phases are ordered by dependency, not calendar. Each phase has a definition
of done that can be checked off without negotiation.

### Phase 0 — Publication rigor (prerequisite for everything else)

**Goal:** Turn the single-seed paper into a defensible result before building
on top of it.

- Re-run the English + Arabic fair comparisons from `training/colab_train*.py`
  with **5 seeds per setting**; report mean ± std and a paired significance
  test (BPC, sequence length).
- Add one downstream task per language (e.g., Arabic sentiment or NLI for AR;
  LAMBADA or a small QA set for EN) and report CST-vs-BPE head-to-head.
- Land results in `docs/cst-paper.md` as a "Robustness" section; regenerate
  PDF via `npm run pdf`.
- Update OSF preprint.

**Done when:** Paper has multi-seed numbers with significance, downstream
numbers, and the preprint version on OSF reflects them.

### Phase 1 — Reasoning-mode tokenizer

**Goal:** Implement the ~150-token reasoning vocabulary as a second config of
the existing tokenizer.

- Add `TokenizerMode = 'general' | 'reasoning'` to `src/tokenizer/types.ts`.
- In `src/tokenizer/cst-spec.ts`, define the reasoning vocabulary exactly as
  sketched in `conv-sonnet-2.md`: ~15 logical operators, ~5 time markers,
  ~30 `CONCEPT:*` fields, ~8 `ROLE:*`, ~10 `REL:*`, specials, plus a small
  `LIT:*` bucket.
- Implement `collapseToReasoning(token) -> Token | null` mapping from general
  tokens to reasoning tokens (e.g., all `STR:neg:*` → `NEG`, all
  `CMP:write:*` → `CONCEPT:write`).
- Mirror in `edge/arabic_tokenizer.py` so Arabic and English produce the
  **same reasoning token sequence** for equivalent meanings.
- Golden tests: a curated parallel AR/EN set where both languages must emit
  identical reasoning-mode token sequences.

**Done when:** `tokenize(sent, mode='reasoning')` works in both TS and Python,
parallel AR/EN test set passes, vocabulary frozen and versioned in
`cst-spec.ts` as `REASONING_SPEC_V1`.

### Phase 2 — Reasoning model (train & evaluate)

**Goal:** Train a small transformer on reasoning-mode tokens and measure what
it actually learns.

- Re-tokenize existing 100K AR + 100K EN corpora in reasoning mode.
- Train nanoGPT / GPT-2-small on reasoning tokens. Target: same parameter
  count as the current 6.8M Arabic model for direct comparison.
- Evaluation suite (build once, reuse):
  - **Negation probe:** N sentence pairs (`X happened` vs `X did not happen`)
    where the model must predict opposite continuations.
  - **Causal probe:** `CAUSE A B` vs `CAUSE (NEG A) B` completions.
  - **Conditional probe:** `COND A B` vs isolated `A`/`B`.
  - **Bilingual transfer:** train on AR-reasoning, eval on EN-reasoning without
    fine-tuning. Hypothesis: non-trivial transfer because vocab is shared.
- Log BPC and probe accuracies. Compare against general-mode model of the
  same size.

**Done when:** Reasoning-mode model exists, probe suite runs via one command,
results recorded in `docs/reasoning-experiments.md`.

### Phase 3 — Interpretability study

**Goal:** Produce the evidence for the "readable attention" claim from
`conv-sonnet-2.md`.

- Extract attention matrices from the Phase 2 model for a 1000-sentence probe
  set.
- For each layer, compute per-token-type attention statistics: e.g., does
  `NEG` attend to its scope? Does `QUERY` attend to the `CONCEPT:*` being
  questioned? Does `COND` attend to both antecedent and consequent?
- Publish a small supplementary study: "Attention patterns over a semantic
  vocabulary." Plots + statistics, not prose speculation.

**Done when:** Supplementary report in `docs/interpretability-v1.md` with
reproducible notebook under `edge/demo/interp/` (or similar) and concrete
hypotheses marked supported / not supported.

### Phase 4 — Verifier (first symbolic layer)

**Goal:** Turn `arabic-algebra-engine` into the logic checker that runs over
reasoning-mode outputs.

- Define a rule format over CST token sequences, e.g.:
  - `NEG CAUSE X Y` ⇒ Y is not entailed.
  - `COND A B` + `A` ⇒ `B` allowed; `COND A B` + `NEG A` ⇒ `B` not entailed.
- Implement a TS checker in `src/verifier/` that consumes reasoning-mode
  token sequences and returns `{ ok: boolean, violatedRules: Rule[] }`.
- Port the relevant subset of `arabic-algebra-engine`'s 74 rules that are
  expressible over the reasoning vocabulary. Rules that require surface
  detail stay in the engine and are invoked from general mode only.
- Integrate as a post-processor on model outputs in Phase 2 eval harness.
  Report: how often does the verifier catch wrong completions?

**Done when:** `verifier.check(tokens)` exists, covers ≥10 named rules, is
used in the Phase 2 eval loop, and "verifier-catch rate" is part of the
results table.

### Phase 5 — Scratchpad (multi-pass reasoning)

**Goal:** Let the model reason in explicit steps instead of one forward pass.

- Define the loop: input tokens → model → intermediate reasoning tokens →
  verifier → (if violated) re-prompt with correction marker → model → …
- Cap depth (e.g., 3 passes) and log the chain.
- Evaluate on the Phase 2 probe suite: does scratchpad + verifier improve
  negation / causal / conditional accuracy?

**Done when:** A `ReasoningRunner` class orchestrates input → loop → output,
all intermediate states are CST token sequences, and measurable lift over
single-pass is either demonstrated or ruled out in `docs/reasoning-experiments.md`.

### Phase 6 — Memory + retrieval

**Goal:** Give the system cross-turn memory in CST tokens.

- Simple store: `{ entityId, tokens: Token[], context: string }[]`.
- Retrieval: token-level match over the store, return up to K entries,
  inject into the scratchpad context.
- Start with an in-memory store; persistence is out of scope for this phase.

**Done when:** A two-turn demo works — e.g., user says "I am a doctor" (turn
1), asks about patients (turn 2), and the model's turn-2 input includes the
turn-1 tokens retrieved via `ROLE:causer + CONCEPT:health` match.

### Phase 7 — Output router

**Goal:** Decide, per output, whether to decode to Arabic text, decode to
English text, trigger an action, or store as a fact.

- Router is a small deterministic function over the leading reasoning tokens
  (`QUERY` → answer path; `COND` + action concept → action path; etc.).
- This is where `arabic-algebra-engine`'s routing maturity re-enters: its
  existing `run()` / `runVerbose()` / `runLLM()` pipeline becomes one of the
  router's backends.

**Done when:** Given a reasoning-token output, the router selects exactly one
of `{ decode-ar, decode-en, action, store }` with a documented policy.

### Phase 8 — Edge deployment (parallel track)

Runs in parallel with Phases 2–4, not blocking them. Absorbs
`OPUS_EDGE_ARABIC_MODEL.md` v2.

- Ship `edge/demo/` browser playground using existing int8 ONNX.
- Hit the v2 targets: ≤15 MB total, ≤100 ms CPU inference, 100% offline.
- Add reasoning-mode inference once Phase 2 has a checkpoint.

**Done when:** Public demo page loads both a general-mode and a reasoning-mode
model, runs entirely in-browser, and meets the size/latency targets.

### Phase 9 — Scale

Re-opens `ARABIC_REASONING_MODEL.md`'s 1M/10M/100M scaling once Phases 0–3
are solid. Do **not** scale before the reasoning-mode result is either
confirmed or falsified at small scale. Scaling a weak signal wastes
compute.

---

## 5. Open Decisions (need explicit yes/no)

These questions are in `conv-sonnet-2.md` but were never closed. They block
Phase 1 design.

- **Q1 — Reasoning vocabulary size.** 150, or 75? Finer `CONCEPT:*` (30) or
  coarser (15)? Default proposal: **start at 150**, ablate later.
- **Q2 — Keep `TIME:*` separate from `FEAT:*`?** Default: **yes, separate** —
  reasoning benefits from explicit time tokens.
- **Q3 — Reasoning mode monolingual first or bilingual from day one?**
  Default: **bilingual**. The parallel AR/EN golden set is what makes the
  interpretability claim land.
- **Q4 — Does the paper get updated in-place, or is there a second paper for
  reasoning mode?** Default: **second paper**. The first paper stays about
  compression; reasoning + interpretability is a separate contribution.

---

## 6. What Gets Retired

To stop the plan folder from drifting, these files are frozen as historical:

- `plan/OPUS_EDGE_ARABIC_MODEL.v1.md` — already superseded by v2.
- `plan/conv-sonnet.md`, `plan/conv-sonnet-2.md` — research notes; their
  decisions are lifted into this roadmap (section 3) and their open questions
  into section 5. No new work tracks these files directly.
- `plan/gcc-press-article-ar*.md` — marketing drafts; live or die on their
  own schedule, not part of this engineering roadmap.

Active plans replaced by sections of this document:

| Old file                      | Replaced by           |
| ----------------------------- | --------------------- |
| `ARABIC_REASONING_MODEL.md`   | Phase 9               |
| `OPUS_EDGE_ARABIC_MODEL.md`   | Phase 8               |
| `OPUS_INFERENCE_CACHE.md`     | Decision D2 + Phase 1 |
| `research-assessment-plan.md` | Phase 0               |

Those files should carry a one-line header pointing to this roadmap next time
they're touched, but need not be deleted.

---

## 7. Principle

Build the smallest version of each phase that can be tested. Prefer a
falsifiable experiment over a bigger model. The thesis — that a tiny
semantic vocabulary enables readable, controllable reasoning — is either
true at 6.8M params and 150 tokens or it isn't. Find out before scaling.
