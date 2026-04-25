"""Tokenize the reasoning corpus and build two-level vocabularies.

Reads every ``stage-*.jsonl`` under ``--in`` (raw records per
``reasoning/data/schema.py``) and produces, per stage:

    <stem>.tokenized.jsonl   — each record plus three fields:
        question_tokens  : {"default": [...], "reasoning": [...]}
        cot_tokens       : [ {"default": [...], "reasoning": [...]}, ... ]
        answer_tokens    : {"default": [...], "reasoning": [...]}

Plus two JSON vocabs at the output root:

    vocab-default.json    — full, per-language, capped at --default-cap
    vocab-reasoning.json  — shared (language-neutral), capped at --reasoning-cap

Both vocabs reserve ``[PAD] [UNK] [BOS] [EOS]`` at ids 0-3. A ``stats.json``
summarises compression ratios, OOV fractions, and per-stage token counts.

Usage::

    python -m reasoning.tokenize_corpus \
        --in ./reasoning/out --out ./reasoning/tokenized

    # Arabic v-next mode (ROOT+PAT + SPACE)
    python -m reasoning.tokenize_corpus \
        --in ./reasoning/out --out ./reasoning/tokenized-vnext \
        --ar-root-pattern --ar-space-token

The script is streaming: memory footprint is bounded by the vocab
counters, not corpus size.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# Lazy imports — tokenizers are heavy (spaCy, CAMeL).
_AR_TOK = None
_EN_TOK = None
_AR_TOKENIZER_KW: dict[str, Any] = {}
_EN_TOKENIZER_KW: dict[str, Any] = {}


SPECIAL_TOKENS: list[str] = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def _get_tokenizers() -> tuple[Any, Any]:
    """Lazily construct the AR+EN tokenizers exactly once."""
    global _AR_TOK, _EN_TOK
    if _AR_TOK is None:
        from reasoning.tokenizer.arabic import ArabicReasoningTokenizer
        _AR_TOK = ArabicReasoningTokenizer.default(**_AR_TOKENIZER_KW)
    if _EN_TOK is None:
        from reasoning.tokenizer.english import EnglishReasoningTokenizer
        _EN_TOK = EnglishReasoningTokenizer(**_EN_TOKENIZER_KW)
    return _AR_TOK, _EN_TOK


def _tokenize(text: str, lang: str) -> dict[str, list[str]]:
    """Return ``{default, reasoning}`` token lists for one string."""
    ar, en = _get_tokenizers()
    tok = ar if lang == "ar" else en
    r = tok.tokenize(text)
    return {
        "default": list(r["default_tokens"]),
        "reasoning": list(r["reasoning_tokens"]),
    }


def _tokenize_with_cst(
    text: str, lang: str, cst_override: list[str] | None,
) -> dict[str, list[str]]:
    """Like ``_tokenize`` but uses ``cst_override`` for the reasoning side
    when supplied (arabic-algebra-engine records carry precomputed CST
    sequences in ``meta.*_cst``)."""
    block = _tokenize(text, lang)
    if cst_override:
        block["reasoning"] = list(cst_override)
    return block


# ── Stage pass ────────────────────────────────────────────────

def _tokenize_record(rec: dict[str, Any]) -> dict[str, Any]:
    lang = rec["lang"]
    meta = rec.get("meta") or {}
    q_cst = meta.get("question_cst")
    a_cst = meta.get("answer_cst")
    cot_cst_list = meta.get("cot_cst") or []

    rec["question_tokens"] = _tokenize_with_cst(rec["question"], lang, q_cst)
    cot_steps = rec.get("cot") or []
    rec["cot_tokens"] = [
        _tokenize_with_cst(
            step,
            lang,
            cot_cst_list[i] if i < len(cot_cst_list) else None,
        )
        for i, step in enumerate(cot_steps)
    ]
    rec["answer_tokens"] = _tokenize_with_cst(str(rec["answer"]), lang, a_cst)
    return rec


def _iter_stage(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _process_stage(
    in_path: Path,
    out_path: Path,
    *,
    default_counter_per_lang: dict[str, Counter],
    reasoning_counter: Counter,
    stats: dict[str, Any],
) -> None:
    n_records = 0
    n_default_tokens = 0
    n_reasoning_tokens = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for rec in _iter_stage(in_path):
            rec = _tokenize_record(rec)
            lang = rec["lang"]

            # Accumulate vocab counts across question + cot + answer.
            for block_key in ("question_tokens", "answer_tokens"):
                block = rec[block_key]
                default_counter_per_lang[lang].update(block["default"])
                reasoning_counter.update(block["reasoning"])
                n_default_tokens += len(block["default"])
                n_reasoning_tokens += len(block["reasoning"])
            for step in rec["cot_tokens"]:
                default_counter_per_lang[lang].update(step["default"])
                reasoning_counter.update(step["reasoning"])
                n_default_tokens += len(step["default"])
                n_reasoning_tokens += len(step["reasoning"])

            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_records += 1

    stats[in_path.name] = {
        "records": n_records,
        "default_tokens": n_default_tokens,
        "reasoning_tokens": n_reasoning_tokens,
        "compression_ratio": (
            n_reasoning_tokens / n_default_tokens if n_default_tokens else 0.0
        ),
    }


# ── Vocab construction ───────────────────────────────────────

def _build_vocab(counter: Counter, cap: int) -> dict[str, int]:
    """Reserve specials at 0..N-1, then most-frequent tokens up to ``cap``."""
    vocab: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    remaining = cap - len(vocab)
    if remaining <= 0:
        return vocab
    for tok, _ in counter.most_common(remaining):
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


def _oov_fraction(counter: Counter, vocab: dict[str, int]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    covered = sum(c for t, c in counter.items() if t in vocab)
    return 1.0 - covered / total


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, required=True,
                    help="Directory containing stage-*.jsonl")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True,
                    help="Where to write tokenized stages + vocabs")
    ap.add_argument("--default-cap", type=int, default=32_000,
                    help="Max default-vocab size per language (incl. specials)")
    ap.add_argument("--reasoning-cap", type=int, default=10_000,
                    help="Max shared reasoning-vocab size (incl. specials)")
    ap.add_argument("--ar-root-pattern", action="store_true",
                    help="Arabic only: emit ROOT+PAT decomposition in default tokens")
    ap.add_argument("--ar-space-token", action="store_true",
                    help="Arabic only: emit explicit SPACE token after each tokenized word")
    ap.add_argument(
        "--ar-atomic-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Arabic only: split composition into ROOT + ROLE tokens "
            "(default: enabled; use --no-ar-atomic-composition for legacy CMP output)"
        ),
    )
    ap.add_argument(
        "--ar-critical-feat-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Arabic only: keep only critical FEAT markers in default tokens "
            "(asp/pgn/enclitic pronoun); use --no-ar-critical-feat-only "
            "to include FEAT:def and noun f/p/d markers."
        ),
    )
    ap.add_argument(
        "--en-atomic-composition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "English only: split composition into ROOT + ROLE tokens "
            "(default: enabled; use --no-en-atomic-composition for legacy CMP output)"
        ),
    )
    ap.add_argument("--pattern", default="stage-*.jsonl",
                    help="Glob for input stage files")
    args = ap.parse_args()

    in_files = sorted(args.in_dir.glob(args.pattern))
    if not in_files:
        raise SystemExit(f"No files matching {args.pattern} under {args.in_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    global _AR_TOKENIZER_KW, _EN_TOKENIZER_KW
    _AR_TOKENIZER_KW = {
        "emit_root_pattern": bool(args.ar_root_pattern),
        "emit_space_token": bool(args.ar_space_token),
        "emit_atomic_composition": bool(args.ar_atomic_composition),
        "critical_feat_only": bool(args.ar_critical_feat_only),
    }
    _EN_TOKENIZER_KW = {
        "emit_atomic_composition": bool(args.en_atomic_composition),
    }

    # Warm the tokenizers once so any import errors surface early.
    _get_tokenizers()
    print(f"Loaded tokenizers. Processing {len(in_files)} stage files …")

    default_counter: dict[str, Counter] = {"en": Counter(), "ar": Counter()}
    reasoning_counter: Counter = Counter()
    stats: dict[str, Any] = {}

    for f in in_files:
        out_path = args.out_dir / f.name.replace(".jsonl", ".tokenized.jsonl")
        print(f"  · {f.name} → {out_path.name}")
        _process_stage(
            f, out_path,
            default_counter_per_lang=default_counter,
            reasoning_counter=reasoning_counter,
            stats=stats,
        )

    # Build vocabs
    vocab_default = {
        lang: _build_vocab(default_counter[lang], args.default_cap)
        for lang in ("en", "ar")
    }
    vocab_reasoning = _build_vocab(reasoning_counter, args.reasoning_cap)

    (args.out_dir / "vocab-default.json").write_text(
        json.dumps(vocab_default, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "vocab-reasoning.json").write_text(
        json.dumps(vocab_reasoning, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Stats summary
    stats["_vocab"] = {
        "default": {
            lang: {
                "size": len(vocab_default[lang]),
                "unique_observed": len(default_counter[lang]),
                "oov_fraction": _oov_fraction(default_counter[lang], vocab_default[lang]),
            }
            for lang in ("en", "ar")
        },
        "reasoning": {
            "size": len(vocab_reasoning),
            "unique_observed": len(reasoning_counter),
            "oov_fraction": _oov_fraction(reasoning_counter, vocab_reasoning),
        },
    }
    total_def = sum(v["default_tokens"] for k, v in stats.items() if k != "_vocab")
    total_rea = sum(v["reasoning_tokens"] for k, v in stats.items() if k != "_vocab")
    stats["_totals"] = {
        "default_tokens": total_def,
        "reasoning_tokens": total_rea,
        "compression_ratio": total_rea / total_def if total_def else 0.0,
    }

    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n── Summary ──")
    print(json.dumps(stats["_vocab"], ensure_ascii=False, indent=2))
    print(json.dumps(stats["_totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
