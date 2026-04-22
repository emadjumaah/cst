# Arabic CST Reasoning Model — Plan

> **Core thesis:** A small Arabic model (20-50M params) trained on CST-tokenized data (10M-100M sentences) should show reasoning behavior. CST compresses semantics — the model sees structure, not characters. Less to learn → smaller model → same capability.

## Goal

Demonstrate that an Arabic CST model can **reason** — not just predict the next token, but show understanding of semantic relationships, cause-effect, and morphological patterns.

**Why Arabic first:**

- Arabic's وزن system gives CST its most precise tokens (CMP:field:role)
- Arabic naturally chains related derivations (كتب→كاتب→كتاب→مكتبة in one sentence)
- If reasoning appears in Arabic CST, it validates the algebra

---

## Current State

| Asset                              | Status                                                        |
| ---------------------------------- | ------------------------------------------------------------- |
| CST tokenizer with full وزن/CMP    | ✅ Done (tokenize_1m.py)                                      |
| 58 semantic fields, 500+ roots     | ✅ Done                                                       |
| Pattern→role mapping (30+ وزن)     | ✅ Done                                                       |
| All Arabic grammar categories      | ✅ Done (حروف جر/عطف، نفي، توكيد، شرط، إنّ وأخواتها، استثناء) |
| NER detection via camel-tools      | ✅ Done                                                       |
| 100K model (6.8M params, BPC 1.15) | ✅ Done (baseline)                                            |
| 1M training scripts                | ✅ Ready (not run yet)                                        |
| Browser demo (ONNX)                | ✅ Working (100K model)                                       |

---

## Phase 0 — Reasoning Tokenizer (deferred; resume later)

> **Scope:** Build the ultra-compact, language-independent **reasoning CST** vocabulary and the evaluation sets. This is a _separate_ tokenizer mode from the general CST used in Phases 1-4. We will **return to this after Phase 1**.

### Decisions (locked)

| #   | Decision                              | Answer                                                                                                                 |
| --- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1   | Reasoning vocabulary size             | **~150 tokens** (a little more is OK — reserve headroom by the end)                                                    |
| 2   | Keep `TIME:*` separate from `FEAT:*`? | **Yes** — `TIME:past/present/future/always/never` is its own dimension                                                 |
| 3   | Reasoning mode language scope         | **Bilingual / language-independent** — same token sequence for the same meaning in Arabic and English                  |
| 4   | Paper strategy                        | **Second paper.** The original CST paper remains the base; reasoning-mode tokenizer + results become a follow-up paper |

### Implications for Phase 0 work (later)

- Tokenizer gains a `mode='reasoning'` flag that collapses surface morphology (see `plan/conv-sonnet-2.md`).
- Vocabulary budget: ~15 logical operators + ~5 TIME + ~30 CONCEPT + ~8 ROLE + ~10 REL + ~5 special + LIT = target ≤ 150 structured tokens, leaving room for additions without breaching the budget.
- Build Level 1 + Level 4 eval sets as bilingual Arabic/English pairs — same reasoning-token sequence must decode/encode from either language.
- Paper split: keep `docs/cst-paper*.md` as the base CST paper; author a second paper focused on reasoning-mode + interpretability claims.

---

## Scaling Plan — 4 Phases

> **Status: jumping to Phase 1 now (general CST, existing tokenizer). Phase 0 resumes after Phase 1 BPC numbers land.**

### Phase 1: 1M sentences, 6.8M params (validate CMP)

> **Purpose:** Confirm the new وزن/CMP tokens improve BPC over the old ROOT-only tokenizer.

| Item           | Value                                                 |
| -------------- | ----------------------------------------------------- |
| Data           | 1M Arabic Wikipedia sentences                         |
| Model          | GPT-2: n_embd=256, n_layer=6, n_head=4 (6.8M params)  |
| Epochs         | 5                                                     |
| Script         | `tokenize_1m.py` → `colab_edge_1m.py`                 |
| Hardware       | Colab T4                                              |
| Time est.      | Tokenize: ~2-3h, Train: ~45min                        |
| Success metric | BPC < 1.10 (improvement over 1.15 ROOT-only baseline) |
| BPE baseline   | Train same 6.8M model on BPE-tokenized 1M data        |

**What to measure:**

- BPC: CST vs BPE (same model, same data, different tokenization)
- CMP% in vocab — what fraction of tokens are CMP vs ROOT?
- Vocab size — does CMP increase or decrease it?

### Phase 2: 10M sentences, 25M params (first reasoning test)

> **Purpose:** Scale data 10x and model 3.5x. Test if structured patterns emerge.

| Item           | Value                                                   |
| -------------- | ------------------------------------------------------- |
| Data           | 10M Arabic Wikipedia sentences                          |
| Model          | GPT-2: n_embd=384, n_layer=8, n_head=6 (~25M params)    |
| Epochs         | 3                                                       |
| Hardware       | Colab T4 or A100 (check memory)                         |
| Time est.      | Tokenize: ~24h (can parallelize), Train: ~4-6h on A100  |
| Success metric | BPC < 0.95 + passes Level 1 and Level 2 reasoning evals |
| BPE baseline   | Train same 25M model on BPE-tokenized 10M data          |

**⚠️ HARD GATE: Do not proceed to Phase 3 unless Phase 2 passes Level 1 and Level 2 evals. If morphological reasoning doesn't appear at 25M params and 10M sentences, scaling won't fix it — something architectural needs to change first.**

**What to measure:**

- BPC: CST vs BPE (mandatory comparison)
- Reasoning evals Level 1 + Level 2
- If CST doesn't beat BPE here, stop and investigate before scaling

**Tokenizer change needed:** Update `tokenize_1m.py` TARGET to 10M. No code changes — same tokenizer.

**Model config:**

```python
N_EMBD = 384
N_LAYER = 8
N_HEAD = 6
MAX_LEN = 128
# vocab_size: auto from tokenizer (~2000-4000 expected)
# params: ~25M
```

### Phase 3: 50M sentences, 50M params (reasoning at scale)

> **Purpose:** Push to the limit of what a "small" model can do. This is where reasoning should clearly appear if CST compression works.

| Item           | Value                                                 |
| -------------- | ----------------------------------------------------- |
| Data           | 50M Arabic sentences (Wikipedia + news + books)       |
| Model          | GPT-2: n_embd=512, n_layer=10, n_head=8 (~50M params) |
| Epochs         | 2-3                                                   |
| Hardware       | A100 (required)                                       |
| Time est.      | Tokenize: ~48h (parallelize), Train: ~12-20h          |
| Success metric | BPC < 0.85 + passes Level 1-3 reasoning evals         |
| BPE baseline   | Train same 50M model on BPE-tokenized 50M data        |

**Data sources (beyond Wikipedia):**

- Arabic Wikipedia: ~2M articles → ~50M+ sentences
- If not enough: OSCAR Arabic corpus, CC-100 Arabic, Arabic Gigaword

**Model config:**

```python
N_EMBD = 512
N_LAYER = 10
N_HEAD = 8
MAX_LEN = 128
# params: ~50M
```

### Phase 4: 100M sentences, 50M params (the research question)

> **Purpose:** Same model as Phase 3, 2x more data. Does more data with same model size still improve? If yes → CST hasn't saturated. If no → we've found the compression limit.

| Item         | Value                                                    |
| ------------ | -------------------------------------------------------- |
| Data         | 100M Arabic sentences                                    |
| Model        | Same 50M params                                          |
| Comparison   | 50M CST model vs 200M BPE model on same 100M data        |
| BPE baseline | Train same 50M model + 200M BPE model, both on 100M data |

**The key question:**

> Does a 50M param CST model trained on 100M sentences match a 200M param BPE model trained on the same data?

If yes — CST compresses learning, not just tokens. That's the publishable result.

### Phase 5: Distillation to edge (6.8M student)

> **Purpose:** If Phase 3/4 proves reasoning exists in the 50M model, distill it into an edge-deployable model.

| Item           | Value                                                                      |
| -------------- | -------------------------------------------------------------------------- |
| Teacher        | 50M CST model (from Phase 3 or 4)                                          |
| Student        | GPT-2: n_embd=256, n_layer=6, n_head=4 (6.8M params)                       |
| Method         | Knowledge distillation — student learns from teacher's output distribution |
| Hardware       | Colab T4/A100                                                              |
| Success metric | Student passes Level 1-2 evals + ONNX int8 ≤ 10MB                          |

**The path:**

```
Train 50M CST model (proves reasoning)
      ↓
Distill into 6.8M CST student model
      ↓
Student learns reasoning from teacher
      ↓
6.8M model with reasoning, runs in browser
```

> Knowledge distillation from a reasoning teacher into an edge student is a known technique. If Phase 3 proves reasoning exists in the 50M model, distillation gets it to the phone.

---

## Reasoning Evaluation

Next-token BPC measures fluency, not reasoning. We need specific tests.

### Level 1: Morphological Reasoning (Phase 1-2)

> Does the model understand root×pattern algebra?

**Test:** Give a partial CMP chain, check if model completes it correctly.

```
Input:  [CMP:write:agent] [CMP:write:instance] [REL:in] ___
Expect: [CMP:write:place]  (library follows writer+book+in)
```

```
Input:  [CMP:know:causer] [CMP:send:past] [REL:to] ___
Expect: [CMP:know:place] or [CMP:know:instance]  (teacher sent to school/knowledge)
```

**Metric:** Accuracy on 100 hand-crafted morphological chains.

> ⚠️ Build these BEFORE any training. Hand-craft them — do not generate. The eval set must exist before the model sees training data. That is the difference between science and demonstration.

### Level 2: Semantic Coherence (Phase 2-3)

> Does the model keep semantic field consistency?

**Test:** Generate 10-token continuations, measure field coherence.

```
Input: [BOS] [CMP:health:place]
Good:  [CMP:know:causer] [CMP:fix:agent]  (hospital → doctor → healer — health context)
Bad:   [CMP:sport:instance] [CMP:food:quality]  (random fields)
```

**Metric:** Field coherence score — what % of generated tokens stay in semantically related fields?

### Level 3: Structural Reasoning (Phase 3-4)

> Does the model understand STR markers?

**Test:**

```
Input: [BOS] [STR:negation] [CMP:move:agent] [REL:to] ___
Expect: model avoids [CMP:move:place] or adjusts prediction for negated context
```

```
Input: [BOS] [STR:question] ___
Expect: higher probability for question-typical patterns
```

### Level 4: Cross-root Reasoning (Phase 3-4)

> Does the model learn associations between different semantic fields?

**Test:**

```
Input: [CMP:health:place] [CMP:know:causer]
Does model predict: [CMP:fix:agent] more than [CMP:write:agent]?
(hospital + teacher → healer, not writer)
```

This is the real reasoning test — the model must learn that health:place + know:causer → fix context, not from a rule but from training data patterns.

> ⚠️ Build 50 cross-root test cases BEFORE training. These are the hardest and most important. If built after, you risk unconsciously making them easier.

---

## Execution Order

```
Week 0:  BUILD EVAL SETS FIRST (before any training)
         - Level 1: 100 morphological chain tests (hand-crafted)
         - Level 4: 50 cross-root association tests (hand-crafted)
         - Start Phase 1 tokenization (1M) while building evals

Week 1:  Phase 1 — train 6.8M CST + 6.8M BPE baseline
         Compare BPC, measure CMP%

Week 2:  Phase 2 — tokenize 10M, train 25M CST + 25M BPE baseline
         Run Level 1 + Level 2 evals
         ⚠️ HARD GATE: stop if Level 1+2 fail

Week 3:  Phase 3 — tokenize 50M, train 50M CST + 50M BPE baseline
         Run Level 1-3 evals

Week 4:  Phase 4 — 100M data, 50M CST vs 200M BPE
         Run all evals, write results

Week 5+: Phase 5 — distill 50M teacher → 6.8M student
         Validate student on Level 1-2 evals
         Deploy to browser demo
```

---

## Scripts Needed

| Script                  | Status    | Purpose                                      |
| ----------------------- | --------- | -------------------------------------------- |
| `tokenize_1m.py`        | ✅ Ready  | Tokenize any target size (change TARGET var) |
| `colab_edge_1m.py`      | ✅ Ready  | Train 6.8M model (Phase 1)                   |
| `colab_edge_10m.py`     | 🔲 Create | Train 25M model (Phase 2) — new config       |
| `colab_edge_50m.py`     | 🔲 Create | Train 50M model (Phase 3) — new config       |
| `eval_reasoning.py`     | 🔲 Create | Run reasoning evals on saved model           |
| `train_bpe_baseline.py` | 🔲 Create | Train matched BPE model at each phase        |
| `distill.py`            | 🔲 Create | Distill 50M teacher → 6.8M student (Phase 5) |
| `build_lookups.py`      | ✅ Exists | Build word↔token lookup tables for demo      |

---

## Model Size Reference

| Config                     | Params | ONNX (fp32) | ONNX (int8) | Fits edge?     |
| -------------------------- | ------ | ----------- | ----------- | -------------- |
| embd=256, layer=6, head=4  | ~6.8M  | ~27MB       | ~7.5MB      | ✅ Yes         |
| embd=384, layer=8, head=6  | ~25M   | ~100MB      | ~27MB       | ⚠️ Borderline  |
| embd=512, layer=10, head=8 | ~50M   | ~200MB      | ~55MB       | ❌ Server only |

> Phase 1-2 models can run in browser. Phase 3-4 are research models — edge deployment later with distillation if reasoning is proven.

---

## The One Sentence

> Train a 50M param Arabic CST model on 100M sentences.
> If it reasons like a 200M BPE model — CST compresses learning.
> Then distill into 6.8M for edge deployment.
> That is a complete research contribution from theory to phone.
