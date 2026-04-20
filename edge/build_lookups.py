"""Build word ↔ CST-token-sequence lookup tables for edge inference.

Inputs  : data/tokenized/cst-ar-8k/train-100000.jsonl (produced by
          an experiment script such as edge/training/tokenize_1m.py,
          which uses the canonical tokenizer in edge/arabic_tokenizer.py;
          the file must contain the `words`, `word_token_counts`, and
          `prefix_count` alignment fields).
Outputs : edge/demo/public/model/word2tok.json   Arabic word → token seq
          edge/demo/public/model/tok2word.json   primary token → Arabic word

Rationale
---------
Each Arabic surface word is now tokenized into a *sequence* of CST tokens
(proclitics, core content token, feature tags, enclitic pronoun). For
encoding we need the whole sequence; for fast decoding we use the first
content token (CMP / ROOT / LIT) as the "primary" and lookup the most
common surface form it corresponds to. FEAT tokens are not surface words
— the renderer composes them as morphological affixes at inference time.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path


INPUT = Path("data/tokenized/cst-ar-8k/train-100000.jsonl")
OUT_WORD2TOK = Path("edge/demo/public/model/word2tok.json")
OUT_TOK2WORD = Path("edge/demo/public/model/tok2word.json")

# Tokens that are attached *to* another content token and therefore do
# not carry a surface-word identity by themselves. The edge renderer
# treats them as affixes (ال / ه / ها / gender inflection…).
AFFIX_PREFIXES = ("FEAT:",)


def is_primary(tok: str) -> bool:
    """True for tokens that carry a standalone surface word.

    CMP / ROOT / LIT / REL / STR primaries correspond to real words.
    FEAT tokens are affixes and are skipped for lookup.
    """
    return tok[:5] not in ("FEAT:",) and not tok.startswith("[")


def main() -> None:
    w2t_seq: Counter[tuple[str, tuple[str, ...]]] = Counter()
    tok2word_cnt: Counter[tuple[str, str]] = Counter()
    skipped_no_align = 0
    total = 0

    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}. Run tokenize_1m.py first.")

    with INPUT.open() as f:
        for line in f:
            d = json.loads(line)
            total += 1
            words = d.get("words")
            counts = d.get("word_token_counts")
            prefix = d.get("prefix_count")
            tokens = d.get("tokens", [])
            if words is None or counts is None or prefix is None:
                skipped_no_align += 1
                continue

            # Locate word-token slices: skip BOS (index 0) and the
            # prefix STR markers, then consume counts[i] tokens per word.
            cursor = 1 + prefix
            for word, n in zip(words, counts):
                slice_ = tokens[cursor:cursor + n]
                cursor += n
                if not slice_:
                    continue
                clean = word.rstrip("،؛.؟!")
                w2t_seq[(clean, tuple(slice_))] += 1
                # Primary = first non-FEAT token in the slice, else first
                primary = next((t for t in slice_ if is_primary(t)),
                               slice_[0])
                tok2word_cnt[(primary, clean)] += 1

    # word → most-common full token sequence
    w2t: dict[str, list[str]] = {}
    for (word, seq), _ in w2t_seq.most_common():
        if word not in w2t:
            w2t[word] = list(seq)

    # primary token → most-common surface word
    t2w: dict[str, str] = {}
    for (tok, word), _ in tok2word_cnt.most_common():
        if tok not in t2w:
            t2w[tok] = word

    print(f"Sentences processed: {total}  (skipped {skipped_no_align} "
          f"with no alignment metadata)")
    print(f"word → token-sequence: {len(w2t)} entries")
    print(f"token → word:          {len(t2w)} entries")
    print()
    print("Examples (word → sequence):")
    for w in ("في", "من", "على", "الماء", "الرجل", "يعمل", "المدرسة",
              "وبكتابه", "سيكتبون"):
        seq = w2t.get(w)
        if seq:
            print(f"  {w}  →  {' '.join(seq)}")
        else:
            print(f"  {w}  →  (not seen)")
    print()
    print("Examples (primary token → word):")
    for t in ("ROOT:exist", "ROOT:speak", "ROOT:know", "ROOT:work",
              "CMP:write:instance", "CMP:know:instance", "FEAT:def"):
        print(f"  {t}  →  {t2w.get(t, '?')}")

    OUT_WORD2TOK.parent.mkdir(parents=True, exist_ok=True)
    with OUT_WORD2TOK.open("w") as f:
        json.dump(w2t, f, ensure_ascii=False)
    with OUT_TOK2WORD.open("w") as f:
        json.dump(t2w, f, ensure_ascii=False)
    print(f"\nSaved {OUT_WORD2TOK} and {OUT_TOK2WORD}")


if __name__ == "__main__":
    main()
