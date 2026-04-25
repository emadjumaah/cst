"""Unified CST tokenizer CLI.

One command to tokenize text in any supported language, using the same
data tables (``data/tokenizer/*.json``) across Python and TypeScript.

Output shape (jsonl, one line per input line):

    {"text": "<original line>", "tokens": ["tok1", "tok2", ...]}

This shape feeds directly into ``training/experiments/prepare_downstream.py``
for classification data. For LM-scoring / LAMBADA data, a small wrapper
script assembles ``{context_tokens, candidate_tokens, gold}`` rows from
per-line outputs.

Usage
-----
    # Tokenize a plain-text file, one sentence per line:
    python -m edge.tokenize --lang en --in sentences.txt --out tokens.jsonl

    # Tokenize a JSON array file:
    python -m edge.tokenize --lang en --in data/sentences-100.json --out out.jsonl

    # Tokenize a jsonl file, reading the "text" field of each row:
    python -m edge.tokenize --lang ar --in in.jsonl --text-field text --out out.jsonl

    # Arabic v-next mode (ROOT+PAT + SPACE boundaries):
    python -m edge.tokenize --lang ar --in in.txt --out out.jsonl \
        --ar-root-pattern --ar-space-token

    # Read from stdin, write to stdout:
    echo "Hello world" | python -m edge.tokenize --lang en --in - --out -

Languages
---------
    en  \u2014 English. Requires spaCy + ``en_core_web_sm``.
    ar  \u2014 Arabic. Requires ``camel-tools`` + built-in morphology DB.
    ru  \u2014 Russian. Not yet implemented (see plan/PHASE0_RUSSIAN.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterator


def _iter_lines(path: str, text_field: str | None) -> Iterator[str]:
    """Yield text strings from a file or stdin.

    Accepts:
        - plain text (one sentence per line)
        - JSON array file (``[".." , "..."]``)
        - jsonl with ``--text-field``
    """
    if path == "-":
        stream = sys.stdin
        close = False
    else:
        stream = open(path, encoding="utf-8")
        close = True

    try:
        first = stream.read(1)
        if first == "":
            return
        if first in ("[",):
            # JSON array \u2014 re-read the whole thing.
            rest = stream.read()
            data = json.loads(first + rest)
            if not isinstance(data, list):
                raise ValueError("JSON array expected when input starts with '['")
            for item in data:
                if isinstance(item, str):
                    yield item
                elif isinstance(item, dict) and text_field and text_field in item:
                    yield item[text_field]
                elif isinstance(item, dict) and "text" in item:
                    yield item["text"]
            return

        # Line-oriented input.
        buf = first + stream.readline()
        while buf:
            line = buf.rstrip("\n")
            if line:
                if text_field:
                    try:
                        obj = json.loads(line)
                        if text_field in obj:
                            yield obj[text_field]
                        else:
                            # Skip silently \u2014 lets the user mix formats.
                            pass
                    except json.JSONDecodeError:
                        yield line
                else:
                    yield line
            buf = stream.readline()
    finally:
        if close:
            stream.close()


def _open_out(path: str):
    if path == "-":
        return sys.stdout, False
    return open(path, "w", encoding="utf-8"), True


# \u2500\u2500 Per-language token-string extraction \u2500\u2500

def _tokenize_en(
    lines: Iterator[str],
    *,
    emit_atomic_composition: bool = True,
) -> Iterator[tuple[str, list[str]]]:
    import spacy  # noqa: WPS433
    from edge.english_tokenizer import EnglishCSTTokenizer  # noqa: WPS433

    nlp = spacy.load("en_core_web_sm")
    tok = EnglishCSTTokenizer(
        nlp,
        emit_atomic_composition=emit_atomic_composition,
    )
    for text in lines:
        out = tok.tokenize(text)
        yield text, out["values"]


def _tokenize_ar(
    lines: Iterator[str],
    *,
    emit_root_pattern: bool = False,
    emit_space_token: bool = False,
    emit_atomic_composition: bool = True,
    critical_feat_only: bool = True,
) -> Iterator[tuple[str, list[str]]]:
    from camel_tools.morphology.database import MorphologyDB  # noqa: WPS433
    from camel_tools.morphology.analyzer import Analyzer  # noqa: WPS433
    from edge.arabic_tokenizer import ArabicCSTTokenizer  # noqa: WPS433

    analyzer = Analyzer(MorphologyDB.builtin_db())
    tok = ArabicCSTTokenizer(
        analyzer,
        emit_root_pattern=emit_root_pattern,
        emit_space_token=emit_space_token,
        emit_atomic_composition=emit_atomic_composition,
        critical_feat_only=critical_feat_only,
    )
    for text in lines:
        out = tok.tokenize(text)
        yield text, list(out["tokens"])


def _tokenize_ru(lines: Iterator[str]) -> Iterator[tuple[str, list[str]]]:
    raise NotImplementedError(
        "Russian tokenizer not implemented. "
        "See plan/PHASE0_RUSSIAN.md for the scoped work required."
    )


_LANGS = {
    "en": _tokenize_en,
    "ar": _tokenize_ar,
    "ru": _tokenize_ru,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", required=True, choices=list(_LANGS.keys()))
    p.add_argument("--in", dest="inp", required=True,
                   help="Input path (plain text, JSON array, or jsonl). Use '-' for stdin.")
    p.add_argument("--out", required=True,
                   help="Output jsonl path. Use '-' for stdout.")
    p.add_argument("--text-field", default=None,
                   help="For jsonl input, the field holding the text (default: 'text').")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of lines processed.")
    p.add_argument("--ar-root-pattern", action="store_true",
                   help="Arabic only: emit ROOT+PAT decomposition for content words.")
    p.add_argument("--ar-space-token", action="store_true",
                   help="Arabic only: emit explicit SPACE token after each tokenized word.")
    p.add_argument(
        "--ar-atomic-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Arabic only: split composition into atomic ROOT + ROLE tokens "
            "(default: enabled; use --no-ar-atomic-composition for legacy CMP output)."
        ),
    )
    p.add_argument(
        "--ar-critical-feat-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Arabic only: keep only critical FEAT markers "
            "(asp/pgn/enclitic pronoun); use --no-ar-critical-feat-only "
            "to include FEAT:def and noun f/p/d markers."
        ),
    )
    p.add_argument(
        "--en-atomic-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "English only: split composition into atomic ROOT + ROLE tokens "
            "(default: enabled; use --no-en-atomic-composition for legacy CMP output)."
        ),
    )
    args = p.parse_args()

    lines = _iter_lines(args.inp, args.text_field)
    if args.limit is not None:
        lines = (x for _, x in zip(range(args.limit), lines))

    fout, close = _open_out(args.out)
    n = 0
    if args.lang == "ar":
        rows = _tokenize_ar(
            lines,
            emit_root_pattern=bool(args.ar_root_pattern),
            emit_space_token=bool(args.ar_space_token),
            emit_atomic_composition=bool(args.ar_atomic_composition),
            critical_feat_only=bool(args.ar_critical_feat_only),
        )
    elif args.lang == "en":
        rows = _tokenize_en(
            lines,
            emit_atomic_composition=bool(args.en_atomic_composition),
        )
    else:
        rows = _LANGS[args.lang](lines)
    try:
        for text, tokens in rows:
            fout.write(json.dumps({"text": text, "tokens": tokens}, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  [{args.lang}] {n:,} lines", file=sys.stderr)
    finally:
        if close:
            fout.close()

    print(f"  [{args.lang}] wrote {n:,} lines -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
