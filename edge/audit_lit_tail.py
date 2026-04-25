"""Audit the LIT tail of a v2 CST-Arabic vocab file.

Buckets every `LIT:<surface>` token by surface-form properties so we can
see what the 336K LIT tail is actually made of, and decide what the
Phase 3 tokenizer needs to catch.

Usage
-----

    python edge/audit_lit_tail.py <vocab.json> [--out edge/artifacts]

Outputs (under --out):

    lit_audit_summary.json      # bucket counts + cumulative frequency
    lit_audit_samples.json      # top-N per bucket for eyeballing
    lit_audit_handlabel.tsv     # top-1000 Arabic-only LITs, tab-separated,
                                # with empty columns for hand-classification

Bucket taxonomy (pure surface features, no CAMeL call):

    num              all digits (ASCII or Arabic-Indic) ± punctuation
    foreign          contains Latin/Cyrillic/CJK characters
    mixed_script     mix of Arabic + non-Arabic letters
    punct_only       only punctuation / symbols
    single_char      len == 1
    short_arabic     pure Arabic, len 2-3 (likely fragments or stop words)
    arabic_4plus     pure Arabic, len >= 4  ← the real mystery bucket
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF\u3040-\u30FF]")
DIGIT_RE = re.compile(r"[0-9\u0660-\u0669\u06F0-\u06F9]")
ARABIC_LETTER_RE = re.compile(r"^[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+$")


def classify(surface: str) -> str:
    if not surface:
        return "empty"

    has_arabic = bool(ARABIC_RE.search(surface))
    has_latin = bool(LATIN_RE.search(surface))
    has_cyr = bool(CYRILLIC_RE.search(surface))
    has_cjk = bool(CJK_RE.search(surface))
    has_digit = bool(DIGIT_RE.search(surface))
    only_digits_punct = all(
        DIGIT_RE.match(c) or not c.isalpha() for c in surface
    )

    if only_digits_punct and has_digit:
        return "num"

    if has_latin or has_cyr or has_cjk:
        if has_arabic:
            return "mixed_script"
        return "foreign"

    if not has_arabic:
        return "punct_only"

    # Pure Arabic from here on.
    # Strip combining marks (tashkeel) when measuring length.
    stripped = "".join(c for c in surface if not unicodedata.combining(c))
    n = len(stripped)
    if n == 1:
        return "single_char"
    if n <= 3:
        return "short_arabic"
    return "arabic_4plus"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("vocab", help="Path to cst-train-<N>-vocab.json")
    p.add_argument(
        "--out",
        default="edge/artifacts",
        help="Output directory for audit reports",
    )
    p.add_argument(
        "--handlabel-top",
        type=int,
        default=1000,
        help="How many arabic_4plus entries to dump for hand-labeling",
    )
    args = p.parse_args()

    vocab_path = Path(args.vocab)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
    print(f"Loaded vocab: {len(vocab):,} tokens", file=sys.stderr)

    # Tally by top-level token type first.
    type_counts: dict[str, int] = defaultdict(int)
    type_freq: dict[str, int] = defaultdict(int)
    for tok, freq in vocab.items():
        prefix = tok.split(":", 1)[0] if ":" in tok else tok
        type_counts[prefix] += 1
        type_freq[prefix] += freq

    # Now bucket only the LIT:* tokens.
    buckets_count: dict[str, int] = defaultdict(int)
    buckets_freq: dict[str, int] = defaultdict(int)
    buckets_examples: dict[str, list[tuple[str, int]]] = defaultdict(list)
    arabic_4plus: list[tuple[str, int]] = []

    lit_prefix = "LIT:"
    for tok, freq in vocab.items():
        if not tok.startswith(lit_prefix):
            continue
        surface = tok[len(lit_prefix):]
        bucket = classify(surface)
        buckets_count[bucket] += 1
        buckets_freq[bucket] += freq
        if len(buckets_examples[bucket]) < 200:
            buckets_examples[bucket].append((surface, freq))
        if bucket == "arabic_4plus":
            arabic_4plus.append((surface, freq))

    total_lit_count = sum(buckets_count.values())
    total_lit_freq = sum(buckets_freq.values())

    # Sort examples by frequency within each bucket.
    for b in buckets_examples:
        buckets_examples[b].sort(key=lambda x: -x[1])
        buckets_examples[b] = buckets_examples[b][:50]

    summary = {
        "vocab_path": str(vocab_path),
        "total_unique_tokens": len(vocab),
        "token_type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "token_type_total_occurrences": dict(
            sorted(type_freq.items(), key=lambda x: -x[1])
        ),
        "lit_bucket_counts": dict(
            sorted(buckets_count.items(), key=lambda x: -x[1])
        ),
        "lit_bucket_total_occurrences": dict(
            sorted(buckets_freq.items(), key=lambda x: -x[1])
        ),
        "lit_bucket_pct_of_unique_lit": {
            b: round(100 * n / max(total_lit_count, 1), 2)
            for b, n in sorted(buckets_count.items(), key=lambda x: -x[1])
        },
        "lit_bucket_pct_of_lit_occurrences": {
            b: round(100 * f / max(total_lit_freq, 1), 2)
            for b, f in sorted(buckets_freq.items(), key=lambda x: -x[1])
        },
    }

    samples = {
        b: [{"surface": s, "freq": f} for s, f in exs]
        for b, exs in buckets_examples.items()
    }

    (out_dir / "lit_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "lit_audit_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Hand-label TSV: top-N arabic_4plus entries with columns for category.
    arabic_4plus.sort(key=lambda x: -x[1])
    top = arabic_4plus[: args.handlabel_top]
    lines = [
        "surface\tfreq\tcategory\troot_if_known\tfield\tnotes",
        "# category vocab: PERSON | PLACE | ORG | FOREIGN | MODERN | DIALECT | RARE_MSA | JUNK",
    ]
    for surface, freq in top:
        lines.append(f"{surface}\t{freq}\t\t\t\t")
    (out_dir / "lit_audit_handlabel.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Console summary.
    print()
    print("=" * 60)
    print(f"Vocab: {vocab_path}")
    print(f"Total unique tokens: {len(vocab):,}")
    print()
    print("Top-level token types (unique count):")
    for t, n in summary["token_type_counts"].items():
        print(f"  {t:12s} {n:>10,}")
    print()
    print("LIT bucket breakdown (of unique LIT tokens):")
    print(f"  {'bucket':16s}  {'unique':>10s}  {'% uniq':>7s}  "
          f"{'total occ':>14s}  {'% occ':>7s}")
    for b in summary["lit_bucket_counts"]:
        print(
            f"  {b:16s}  {summary['lit_bucket_counts'][b]:>10,}  "
            f"{summary['lit_bucket_pct_of_unique_lit'][b]:>6.1f}%  "
            f"{summary['lit_bucket_total_occurrences'][b]:>14,}  "
            f"{summary['lit_bucket_pct_of_lit_occurrences'][b]:>6.1f}%"
        )
    print()
    print(f"Hand-label TSV: {out_dir / 'lit_audit_handlabel.tsv'}")
    print(f"  Top {len(top):,} of arabic_4plus bucket "
          f"({len(arabic_4plus):,} total). Fill the 'category' column.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
