# Contributing to CST

Thank you for your interest in contributing.

## Development setup

### TypeScript side (tokenizer + tests)

```bash
npm install
npm test               # run full test suite
npm run test:watch     # watch mode
npx tsx src/demo.ts    # tokenize example sentences
```

### Python side (training + experiments)

```bash
cd training
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
camel_data -i morphology-db-msa-r13     # for Arabic experiments
```

## Project layout

See the "Repository Structure" section in `README.md`.

Source of truth for tokenizer behavior: `src/tokenizer/`. Behavior must be backed by a test in `src/tests/`.

Source of truth for Arabic tokenizer behavior: `edge/arabic_tokenizer.py` (`ArabicCSTTokenizer` class). Unit tests live in `edge/training/tests/test_tokenizer.py`. Experiment scripts (e.g. `edge/training/tokenize_1m.py`) must import from the library — never duplicate tokenizer logic.

## Before submitting a PR

1. `npm test` must pass.
2. New tokenizer rules must come with a unit test.
3. If you change the semantic field inventory or relation map, run `src/demo.ts` on the example sentences and confirm expected outputs.
4. If you change training scripts, note it in `CHANGELOG.md`.
5. No personal data, no large binaries, no generated artifacts in commits.

## Areas open for contribution

See `plan/research-assessment-plan.md` for the prioritized research roadmap (note: this plan file is gitignored; ask the maintainer for a copy).

High-impact areas:

- Additional language coverage (Hebrew, Turkish, Finnish, Korean).
- Additional baseline tokenizers (Unigram LM, WordPiece, byte-level, word-level).
- Downstream task evaluation (classification, QA, NLI).
- Held-out corpus evaluation (non-Wikipedia).
- Figures, visualizations, and analysis notebooks.

## Code style

- TypeScript: follow existing style; no bundler-specific imports; `.ts` extensions in relative imports.
- Python: PEP 8, docstrings on public functions, type hints where they help.
- No emojis in code or commit messages.

## Reporting issues

Include:

- OS and Node/Python versions.
- Exact command run.
- Expected vs actual output.
- Minimal reproducible example where possible.
