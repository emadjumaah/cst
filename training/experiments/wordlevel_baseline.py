"""
Word-level baseline (ablation A.1 companion).

Motivation
----------
SentencePiece BPE is a subword baseline. A reviewer will ask: how does CST
compare to a plain *word-level* tokenizer with the same vocabulary budget?
Word-level is notoriously weak at OOV but it's the right control for the
"semantic granularity matters" claim.

Procedure
---------
1. Read raw sentences (one per line, or a .jsonl with a ``text`` field).
2. Lower-case (English) and tokenize on whitespace + simple punctuation.
3. Build a vocabulary of the top ``vocab_size - 1`` word types plus an
   ``<unk>`` id. OOVs at inference map to <unk>.
4. Emit ids per sentence; keep character counts (on the original text).
5. Train and evaluate with the same GPT-2 config as the other experiments.

Usage
-----
    python training/experiments/wordlevel_baseline.py \\
        --text /content/sentences-100k.txt \\
        --vocab-size 8000 \\
        --out /content/results_wordlevel_8k.json \\
        --seeds 0 1 2

Or with a .jsonl source (one ``{"text": ...}`` per line):
    python training/experiments/wordlevel_baseline.py \\
        --jsonl /content/sentences-100k.jsonl --text-field text ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from experiments._core import DEFAULTS, split_train_val, train_and_eval  # noqa: E402


# Simple tokenizer: keep Arabic + Latin word characters, split on the rest.
TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+|[^\s\w]", re.UNICODE)


def read_texts(args) -> list[str]:
    texts: list[str] = []
    if args.text:
        with open(args.text) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    else:
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get(args.text_field)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
    return texts


def tokenize(text: str, lower: bool) -> list[str]:
    if lower:
        text = text.lower()
    return TOKEN_RE.findall(text)


def build_vocab(texts: list[str], vocab_size: int, lower: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(tokenize(t, lower))
    # id 0 = <pad>, id 1 = <unk>. Everything else by frequency.
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(vocab_size - 2):
        vocab[tok] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int], lower: bool, max_len: int):
    unk = vocab["<unk>"]
    ids_list: list[list[int]] = []
    char_counts: list[int] = []
    for t in texts:
        toks = tokenize(t, lower)
        ids = [vocab.get(tok, unk) for tok in toks]
        if len(ids) < 4:
            continue
        if len(ids) > max_len:
            ratio = max_len / len(ids)
            ids = ids[:max_len]
            char_counts.append(int(len(t) * ratio))
        else:
            char_counts.append(len(t))
        ids_list.append(ids)
    return ids_list, char_counts


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Plain text file, one sentence per line.")
    src.add_argument("--jsonl", help="JSONL with a text field.")
    p.add_argument("--text-field", default="text")
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--lang", choices=["en", "ar"], default="en",
                   help="Used to decide lower-casing (ar → no lowercase).")
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    args = p.parse_args()

    lower = args.lang == "en"
    texts = read_texts(args)
    print(f"  Loaded {len(texts):,} sentences")
    vocab = build_vocab(texts, args.vocab_size, lower)
    print(f"  Word vocab size: {len(vocab):,} (requested {args.vocab_size:,})")

    ids_list, char_counts = encode(texts, vocab, lower, DEFAULTS["max_len"])
    print(f"  Encoded {len(ids_list):,} sentences")

    tr_ids, tr_ch, va_ids, va_ch = split_train_val(ids_list, char_counts, DEFAULTS["val_ratio"])

    results: list[dict] = []
    for seed in args.seeds:
        r = train_and_eval(
            name=f"WORD-{args.vocab_size//1000}K",
            train_ids=tr_ids, train_chars=tr_ch,
            val_ids=va_ids, val_chars=va_ch,
            vocab_size=len(vocab), seed=seed, epochs=args.epochs,
        )
        r["variant"] = "word_level"
        r["lang"] = args.lang
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=float)

    bpcs = [r["best_val_bpc"] for r in results]
    mean = sum(bpcs) / len(bpcs)
    var = sum((x - mean) ** 2 for x in bpcs) / max(len(bpcs) - 1, 1)
    print(f"\n  Word-level BPC = {mean:.4f} ± {var**0.5:.4f}  (n={len(bpcs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
