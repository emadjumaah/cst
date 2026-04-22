# Security Policy

## Supported versions

This repository is a research proof of concept. Only `main` is supported.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Instead, email **emadjumaah@gmail.com** with:

- A description of the issue and its impact
- Steps to reproduce (minimal example if possible)
- Affected file(s) / commit hash
- Your contact details for follow-up

You can expect an initial response within **7 days**. We will coordinate a fix and public disclosure with you.

## Scope

In scope:

- Tokenizer code in [`src/`](src), [`edge/`](edge), [`reasoning/`](reasoning)
- Build / training scripts in [`scripts/`](scripts), [`training/`](training)
- CI workflows under [`.github/workflows/`](.github/workflows)

Out of scope:

- Third-party services linked from the README or docs (Google Fonts, HF Datasets, CDN-hosted MathJax, etc.)
- Vulnerabilities in pinned dependencies — please report those upstream; we will update once a patched release is available.
