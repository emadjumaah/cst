# CST Documentation

Entry point to everything written about Contextual Semantic Tokenization.

## Papers ([`paper/`](paper))

| File | Language | Description |
|------|----------|-------------|
| [`cst-paper.md`](paper/cst-paper.md) | English | Full research paper — methodology, results, cross-lingual analysis |
| [`cst-paper-ar.md`](paper/cst-paper-ar.md) | Arabic | Arabic translation of the paper |

PDFs are generated from these sources with `npm run pdf` and are **not** committed.

## Specifications ([`spec/`](spec))

| File | Description |
|------|-------------|
| [`two-level-tokenization.md`](spec/two-level-tokenization.md) | Default vs. reasoning tokenizers; role inventory |
| [`cst-arabic-tokenizers.md`](spec/cst-arabic-tokenizers.md) | Shared spec + Arabic-specific rules |
| [`cst-arabic-coverage.md`](spec/cst-arabic-coverage.md) | Arabic surface decomposition (prefix / core / inflection) |
| [`cst-english-tokenizers.md`](spec/cst-english-tokenizers.md) | English normalization + tokenization rules |

## Media & press ([`media/`](media))

| File | Language | Description |
|------|----------|-------------|
| [`cst-media-post.md`](media/cst-media-post.md) | English | 2k-word general-audience article |
| [`cst-media-post-ar.md`](media/cst-media-post-ar.md) | Arabic | Arabic translation |
| [`area-press-article-ar-news.md`](media/area-press-article-ar-news.md) | Arabic | News-style press framing |

## Plans ([`plans/`](plans))

| File | Description |
|------|-------------|
| [`TRAINING_PLAN.md`](plans/TRAINING_PLAN.md) | Scaling sweep + ablations + multi-seed runs |
| [`RESEARCH_CHECKLIST.md`](plans/RESEARCH_CHECKLIST.md) | Publication rigor checklist |
| [`REASONING_DATA.md`](plans/REASONING_DATA.md) | Curriculum + data building for the reasoning track |
| [`ARABIC_REASONING_MODEL.md`](plans/ARABIC_REASONING_MODEL.md) | 20–50M Arabic CST reasoning model plan |
| [`PHASE0_RUSSIAN.md`](plans/PHASE0_RUSSIAN.md) | Russian Phase-0 extension |

The top-level [`ROADMAP.md`](../ROADMAP.md) is the single source of truth that links these plans together.

## Also at repo root

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — one-page system overview
- [`DATA.md`](../DATA.md) — data statement, provenance, licensing
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — setup + contribution checklist
- [`CHANGELOG.md`](../CHANGELOG.md) — release history
- [`CITATION.cff`](../CITATION.cff) — citation metadata
