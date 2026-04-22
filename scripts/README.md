# Scripts

Small utilities used outside the core tokenizer / pipeline.

| File | Purpose | How to run |
|------|---------|-----------|
| [`generate-pdf.ts`](generate-pdf.ts) | Render the English + Arabic papers from `docs/paper/*.md` to PDF using system Chrome + MathJax. Output is **not** committed. | `npm run pdf` |
| [`extract_tokenizer_data.ts`](extract_tokenizer_data.ts) | Export the semantic-field tables from the TS tokenizer into JSON so the Python edge tokenizer can consume the same data. | `npx tsx scripts/extract_tokenizer_data.ts` |
| [`check_tokenizer_parity.py`](check_tokenizer_parity.py) | Verify that `src/tokenizer/` (TypeScript) and `edge/*_tokenizer.py` (Python) produce identical token sequences on a shared fixture set. | `python scripts/check_tokenizer_parity.py` |

When you change tokenizer behaviour in **either** language, run `extract_tokenizer_data.ts` (if data tables changed) and then `check_tokenizer_parity.py` before opening a PR.
