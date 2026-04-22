# CST Research Assessment and Improvement Plan

Date: 2026-04-19

## Executive summary

The project is in a strong state for research publication. The core idea is clear, the bilingual English and Arabic results are compelling, and the paper has a coherent theory-to-results story.

The main remaining work is not idea quality; it is publication rigor:

- statistical confidence over multiple seeds
- broader baseline coverage
- stronger reproducibility packaging
- at least one downstream task evaluation

## Current strengths

1. Strong conceptual framing from Arabic morphology to CST design.
2. Controlled setup with matched architecture and parameter counts.
3. Clear result advantage in both English and Arabic.
4. Important cross-lingual signal: CST reduces the English-Arabic gap compared with BPE.
5. Active codebase with tests and reproducible scripts.

## Main gaps to close before submission

### 1) Statistical rigor

Problem:

- Current reporting is effectively single-run per setting.
- Claims like "within measurement noise" need measured variance.

Action:

- Run each experiment with at least 5 random seeds.
- Report mean plus/minus standard deviation for BPC.
- Add significance testing for CST vs each baseline.

Success criteria:

- All key tables include mean plus/minus std.
- Statistical significance is reported for headline comparisons.

### 2) Baseline breadth

Problem:

- Main comparison focuses on SentencePiece BPE.

Action:

- Add at least two additional tokenization baselines:
  - SentencePiece Unigram
  - WordPiece or byte-level BPE baseline
- Keep model size, data, and optimization budget matched.

Success criteria:

- CST advantage remains robust against multiple baseline families.

### 3) External validity

Problem:

- Current evidence covers English and Arabic only, with language modeling objective only.

Action:

- Add one additional language family target (for example Turkish or Finnish).
- Add at least one downstream task per language track (classification or QA).

Success criteria:

- Show that gains are not limited to BPC and not limited to two languages.

### 4) Reproducibility hardening

Problem:

- Dependency versions and artifact workflow can be tightened for external reproduction.

Action:

- Pin Python requirements with exact versions.
- Add a single command entry point for end-to-end regeneration.
- Save results as machine-readable files (json/csv) for each run.
- Add an artifact checklist for expected outputs.

Success criteria:

- A new user can reproduce the reported core table from a clean environment.

### 5) Implementation hygiene for reviewer confidence

Problem:

- There are duplicate-key warnings in relation map data.
- Test threshold for structured coverage is permissive for the current maturity level.

Action:

- Remove duplicate keys and keep one canonical mapping.
- Raise coverage and consistency assertions to stricter levels after validation.
- Add Arabic tokenizer unit tests for root extraction and proclitic handling.

Success criteria:

- Clean test output and stronger guarantees around tokenizer behavior.

## Prioritized roadmap

## Week 1 (high impact)

- Run 5-seed experiments for current English and Arabic 8K/32K settings.
- Add variance and significance to paper tables and claims.
- Remove hardcoded cross-lingual baseline prints in scripts and load from saved results.

## Week 2 (submission strength)

- Add Unigram and WordPiece or byte-level baseline tracks.
- Add one downstream benchmark to verify transfer beyond language modeling.
- Pin dependencies and publish an explicit artifact reproduction workflow.

## Week 3 (positioning and polish)

- Add third language pilot.
- Complete ablations (UNK cap effect, field inventory size, analyzer dependency).
- Final pass for clean code/test output and paper consistency.

## Suggested paper updates after new runs

1. Add a dedicated subsection for statistical confidence and significance.
2. Add baseline expansion table with all tokenizers under matched budgets.
3. Add downstream task table.
4. Add reproducibility appendix with exact commands and expected artifacts.

## Publication positioning

Current status:

- Strong and novel idea with promising evidence.

After closing the gaps above:

- Competitive main-conference submission profile with stronger defensibility.

## Practical immediate next steps

1. Implement multi-seed runner and result aggregation.
2. Add baseline tokenizers and regenerate comparison datasets.
3. Re-run training grid and update tables automatically from result files.
4. Freeze dependencies and finalize reproducibility checklist.

---

# Critical Review (Addendum) — Publication Rigor Assessment

Date: 2026-04-19

The plan above covers operational hygiene. This addendum identifies methodological and scholarly gaps that reviewers at a main NLP conference (EMNLP/ACL) will raise. Addressing these is the difference between a workshop paper and a main-conference paper.

## A. Methodological concerns reviewers will raise

### A.1 BPC vs token count confound (highest risk)

Problem:

- CST produces ~22 tokens/sentence; BPE-8K produces ~32.
- Shorter sequences mechanically reduce per-character loss if per-token loss is held constant.
- BPC normalizes for this in principle, but reviewers will want direct evidence.

Action:

- Report per-token NLL alongside BPC in all tables.
- Plot BPC as a function of tokens-per-character across tokenizer configurations.
- Add a **word-level tokenizer baseline** with vocabulary capped to match CST-8K and CST-32K. If CST ≈ word-level, the semantic labels are decorative.

Success criteria:

- Reviewer cannot reduce the observed gain to segmentation granularity alone.

### A.2 Inductive bias ablation — does the _semantic structure_ actually help?

Problem:

- The paper claims tokens like `CMP:write:agent` carry useful structure. This has never been tested.

Action:

- **Label-shuffle ablation**: randomly permute field labels across the vocabulary (preserve sizes and token boundaries). Retrain. If BPC is unchanged, the semantic grouping is placebo.
- **Role-strip ablation**: collapse `CMP:write:agent` → `CMP:write`. Measure the BPC delta attributable to role information.
- **Field-strip ablation**: collapse all `CMP:*` to a single generic token. Measure delta.

Success criteria:

- Quantified contribution of (a) field identity, (b) role information, to the overall BPC advantage.
- Without this ablation set, the strongest claim in the paper is unsupported.

### A.3 LIT cache confound

Problem:

- CST-8K = 846 semantic + ~7,152 frequent LIT tokens. The LIT cache is effectively a word-level vocabulary for common words.
- SPM-8K has no such cache; its vocabulary is entirely subword fragments.
- This means part of CST's gain may come from the LIT cache alone, not the semantic layer.

Action:

- **LIT-cap-zero experiment**: CST with zero LIT cache (only semantic tokens + UNK fallback) vs SPM-8K.
- **LIT-only baseline**: vocabulary of top-N frequent whole words + UNK, no semantic tokens. This isolates the cache effect from semantic structure.

Success criteria:

- Decomposition of the BPC gain into (LIT cache contribution) + (semantic structure contribution).

### A.4 Dictionary–data leakage

Problem:

- The ~2,400 English lemma→field mappings and ~560 Arabic root→field mappings were constructed by the author.
- Dictionary was iteratively refined while inspecting coverage on the evaluation Wikipedia corpus (evidence: 5 rounds of Arabic root additions).
- This is a form of test-set peeking — the dictionary was fit to the corpus the model is evaluated on.

Action:

- Add a **data statement** documenting exactly when the dictionary was frozen relative to evaluation data.
- Evaluate BPC on a **second, held-out corpus the dictionary never saw**: for English, use non-Wikipedia news (e.g., CC-News sample); for Arabic, use Arabic news or OSCAR.
- Report BPC on both Wikipedia (in-dictionary) and the held-out source.

Success criteria:

- Demonstrated generalization of CST gains beyond the corpus used to construct the dictionary.

### A.5 "Algebra" framing is rhetorical, not formal

Problem:

- `root × pattern = concept` is suggestive but not an algebra. No identity, no inverse, no closure — it is a lookup.
- Main-conference reviewers (especially those with formal linguistics or mathematical backgrounds) will call this rhetorical overreach.

Action (choose one):

- **Soften**: replace "algebra" with "compositional structure" or "templatic composition" throughout.
- **Formalize**: define it properly as a group action on word-forms, or a semi-ring structure. This invites more scrutiny but could become a theoretical contribution.

Success criteria:

- Either the terminology is precise, or the claim is hedged to match the evidence.

### A.6 Cross-lingual gap claim requires a cross-lingual experiment

Problem:

- The claim "difficulty is in the tokenizer, not the language" is currently supported by two independent monolingual models with similar BPC.
- This is **not** cross-lingual transfer. A reviewer can argue CST just happens to be equally good at two languages independently.

Action:

- Train a **single model on interleaved EN+AR CST tokens**. Evaluate per-language BPC on held-out data in each language.
- Compare against a single joint model trained on interleaved EN+AR BPE tokens.
- Add a zero-shot transfer probe: train on EN only, evaluate on AR (and vice versa).

Success criteria:

- Evidence that CST's shared semantic vocabulary enables joint or transfer training, not just that two tokenizers produce similar numbers in isolation.

## B. Scholarly engagement — Related Work gaps

The current Section 2 is thin for a tokenization paper. Reviewers will expect engagement with at minimum:

**Morphology-aware tokenization:**

- Creutz & Lagus (2005) — Morfessor
- Park et al. — Korean morpheme-aware tokenization
- Banerjee & Bhattacharyya — morphology-aware NMT for Indic languages
- Klein & Tsarfaty — MRL parsing

**Tokenization evaluation:**

- Gallé (2019) — "Investigating the Effectiveness of BPE"
- Rust et al. (2021) — "How Good is Your Tokenizer?"
- Sajjad et al. (2017) — tokenization effects in NMT
- Bostrom & Durrett (2020) — already cited, good
- Mielke et al. (2021) — survey of tokenization

**Semantic / frame-based:**

- FrameNet (Baker et al., 1998) — already cited
- WordNet (Miller, 1995) — semantic fields precedent

**Arabic NLP specifically:**

- Habash & Sadat (2006) — Arabic preprocessing schemes
- MADAMIRA / MADA — Arabic morphological analysis
- CAMeL Tools (Obeid et al., 2020) — already cited

Action:

- Rewrite Section 2 as a comparative taxonomy (statistical subword vs morphology-aware vs semantic) with CST positioned explicitly.

## C. Repository professionalism (currently missing)

| Item                  | Status  | Action                                                                |
| --------------------- | ------- | --------------------------------------------------------------------- |
| LICENSE               | missing | Add Apache-2.0 or MIT — without it, the code is legally unusable      |
| CITATION.cff          | missing | Enables GitHub "Cite this repository" button                          |
| DATA.md               | missing | Document data sources, licenses (Wikipedia = CC-BY-SA), preprocessing |
| CONTRIBUTING.md       | missing | Standard for research artifacts                                       |
| CHANGELOG.md          | missing | Artifact reviewers look for this                                      |
| CI workflow           | missing | `.github/workflows/test.yml` running `npm test` on push               |
| Pre-commit hooks      | missing | Run tests + lint locally before commits                               |
| Pinned Python deps    | missing | `requirements.txt` uses bare names; needs `==` versions               |
| Docker / devcontainer | missing | One-command reproducible environment                                  |

## D. Paper production gaps

Problem:

- Paper is in Markdown, not LaTeX. EMNLP/ACL require the ACL Rolling Review LaTeX template.
- No bibtex file; references are inline markdown.
- No figures.

Action:

- Convert to `acl.sty` LaTeX template. Budget ≥1 week.
- Build proper `references.bib` with DOIs.
- Add 3–5 publication-quality figures:
  1. Training curves (BPC over epochs, all four configurations).
  2. BPC vs tokens-per-character scatter across tokenizers.
  3. Token length distribution histogram.
  4. Worked tokenization example as a figure (instead of prose).
  5. Cross-lingual comparison bar chart.
- Prepare separate arXiv bundle with figures inlined and abstract tuned for arXiv search.

## E. Scale and scope defense (reviewer objection 1)

Objection: "13M params on 100K sentences — does this matter at scale?"

Defenses, in order of effort:

1. **Weak**: Cite scaling laws (Kaplan, Hoffmann) and argue tokenizer effects are scale-independent.
2. **Medium**: Add one 50M-param / 1M-sentence run for at least one setting.
3. **Strong (recommended)**: **Reframe as a low-resource / efficient model contribution**. At ≤10M params, tokenizer choice matters _more_, not less. This turns the scale limitation into a feature and aligns with efficient NLP / green AI trends.

Action:

- Rewrite intro + conclusion to position CST as tokenizer for **low-resource and efficient settings** specifically.
- Add a compute budget table (params × tokens) showing CST wins per compute dollar.

## F. Arabic-specific rigor

Problem:

- CAMeL Tools DB version unpinned; different DB versions return different roots.
- Proclitic stripping is heuristic with no measured error rate.
- 79% coverage — composition of the 21% miss is undocumented.
- All 100K sentences are MSA Wikipedia. No evaluation on dialect, Classical, or news.

Action:

- Pin `camel-tools` version + DB release tag in requirements and DATA.md.
- Stratified error analysis on the 21% miss: proper nouns, loanwords, dialect, OOV.
- Evaluate on a held-out Arabic source (e.g., AraBench, Arabic news, or OSCAR-ar).
- Optional: dialect robustness probe on a small dialectal sample.

## G. Priority matrix — restructured by review risk

Tier 1 — Must have before submission (~2 weeks)

- [ ] 5-seed variance for all existing settings (from original plan)
- [ ] **Label-shuffle ablation** (A.2) — single highest-risk experiment
- [ ] **Word-level baseline** at matched vocab (A.1)
- [ ] **LIT-cap-zero experiment** (A.3)
- [ ] LICENSE + CITATION.cff + DATA.md
- [ ] LaTeX conversion with bibtex

Tier 2 — Strong submission form (~3 weeks)

- [ ] Cross-lingual joint model experiment (A.6)
- [ ] One downstream task per language (from original plan)
- [ ] Held-out corpus evaluation (A.4)
- [ ] Unigram + WordPiece baselines (from original plan)
- [ ] 5 publication figures + arXiv bundle

Tier 3 — Polish (~1 week)

- [ ] Third language pilot (from original plan)
- [ ] 50M-param scaling run (E)
- [ ] CI workflow + pre-commit
- [ ] Arabic stratified error analysis (F)
- [ ] Related work rewrite (B)

## H. Professional completion checklist

| Category                  | Current    | Target                                               |
| ------------------------- | ---------- | ---------------------------------------------------- |
| Code license              | missing    | Apache-2.0                                           |
| Citation metadata         | missing    | CITATION.cff                                         |
| Data statement            | missing    | DATA.md with sources/licenses/preprocessing          |
| CI                        | missing    | GitHub Actions                                       |
| Pinned deps               | partial    | `requirements.txt` with `==` pins                    |
| Seed variance             | single-run | 5 seeds, mean±std reported                           |
| Baselines                 | BPE only   | +Unigram, +WordPiece, +word-level                    |
| Ablations                 | none       | Label-shuffle, LIT-cap-zero, role-strip, field-strip |
| Paper format              | Markdown   | LaTeX (ACL template)                                 |
| Figures                   | none       | 3–5 publication-quality figures                      |
| arXiv bundle              | none       | Separate build                                       |
| Downstream tasks          | none       | 1 task per language                                  |
| Cross-lingual model       | none       | Joint EN+AR training                                 |
| Held-out eval             | none       | Non-Wikipedia corpus                                 |
| Arabic dialect robustness | none       | Stratified analysis                                  |

## Bottom line

The original plan would produce a competent workshop submission. To be main-conference competitive, the critical additions are:

1. The three ablations (A.2, A.3) — without them, the semantic-structure claim is unsupported.
2. A word-level baseline (A.1) — without it, the granularity confound is open.
3. The cross-lingual joint model (A.6) — without it, the "tokenizer not language" headline is not demonstrated.
4. LaTeX/bibtex paper form (D) — without it, no submission at all.
5. LICENSE + CITATION + DATA.md (C) — without them, the artifact is not professional.

The rest of the list is real work but can be phased in across versions 1, 2, and camera-ready.
