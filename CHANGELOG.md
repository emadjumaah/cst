# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows semantic versioning where practical.

## [Unreleased]

### Added

- LICENSE (Apache-2.0).
- CITATION.cff for GitHub citation metadata.
- DATA.md describing all data sources, licenses, and known limitations.
- CONTRIBUTING.md with development setup and contribution guidelines.
- GitHub Actions CI workflow running the TypeScript test suite.
- Pinned Python dependency versions in `training/requirements.txt`.
- Multi-seed experiment runner script (`training/run_multiseed.py`).
- Label-shuffle ablation script (`training/ablate_label_shuffle.py`).
- LIT-cap-zero ablation script (`training/ablate_lit_cap_zero.py`).
- Word-level baseline tokenizer script (`training/wordlevel_baseline.py`).
- Result aggregation script (`training/aggregate_results.py`).

### Changed

- `training/colab_train_ar.py` now writes machine-readable JSON results and
  no longer hardcodes the English baseline numbers — it loads them from
  saved result files if available.
- Deduplicated overlapping entries in `src/tokenizer/data.ts` (removed
  Vite/esbuild duplicate-key warnings).

## [0.1.0] — 2026-04-XX

Initial public release of the proof of concept.

### Added

- CST tokenizer (7-stage pipeline) in TypeScript.
- English training pipeline and 4-way comparison (CST vs SentencePiece BPE
  at 8K and 32K vocabulary) on 100K sentences.
- Arabic training pipeline and 4-way comparison on 100K Arabic Wikipedia
  sentences, with root-based extraction via CAMeL Tools.
- Paper (English + Arabic) in `docs/`.
- PDF generation via Puppeteer.
