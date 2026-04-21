"""
Cap CST vocab to match SPM vocab sizes for fair comparison.

Reads full CST jsonl, counts actual token frequencies, keeps all semantic
tokens + top-N most frequent LIT tokens, maps the rest to UNK.

Usage:
  python3 training/cap_cst_vocab.py 8000
  python3 training/cap_cst_vocab.py 32000
  python3 training/cap_cst_vocab.py 8000 --src data/tokenized/cst-py/train-99963.jsonl --out-dir data/tokenized/cst-py-8k
"""

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_SRC = Path("data/tokenized/cst/train-100000.jsonl")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap", type=int, nargs="?", default=8000,
                    help="Target vocab size (default: 8000)")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help=f"Source jsonl with tokens field (default: {DEFAULT_SRC})")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: data/tokenized/cst-{cap//1000}k)")
    args = ap.parse_args()

    cap = args.cap
    SRC = args.src
    print(f"Capping CST vocab to {cap}  (src={SRC})")

    # Pass 1: count token frequencies
    freq = Counter()
    rows = []
    with open(SRC) as f:
        for line in f:
            obj = json.loads(line.strip())
            tokens = obj["tokens"]
            freq.update(tokens)
            rows.append(obj)

    print(f"Loaded {len(rows)} examples, {len(freq)} unique tokens")

    # Separate semantic vs LIT
    semantic = {tok for tok in freq if not tok.startswith("LIT:")}
    lit_tokens = {tok for tok in freq if tok.startswith("LIT:")}
    print(f"Semantic tokens: {len(semantic)}")
    print(f"LIT tokens: {len(lit_tokens)}")

    # Budget: cap - semantic - 2 (for PAD + UNK)
    lit_budget = cap - len(semantic) - 2  # -2 for PAD + UNK
    if lit_budget < 0:
        print(f"ERROR: {len(semantic)} semantic tokens already exceed cap {cap}")
        sys.exit(1)

    # Top LIT tokens by frequency
    lit_ranked = sorted(lit_tokens, key=lambda t: freq[t], reverse=True)
    keep_lit = set(lit_ranked[:lit_budget])
    drop_lit = set(lit_ranked[lit_budget:])

    print(f"Keeping {len(keep_lit)} LIT tokens, dropping {len(drop_lit)}")
    if keep_lit:
        min_freq_kept = freq[lit_ranked[min(lit_budget - 1, len(lit_ranked) - 1)]]
        print(f"Lowest kept LIT frequency: {min_freq_kept}")
    if drop_lit:
        total_drop_occurrences = sum(freq[t] for t in drop_lit)
        total_all_occurrences = sum(freq.values())
        print(f"Dropped tokens total occurrences: {total_drop_occurrences} ({total_drop_occurrences/total_all_occurrences*100:.1f}% of all token uses)")

    # Build new vocab: id 0 = PAD, 1 = UNK, then semantic, then LIT
    UNK = "<UNK>"
    new_vocab = {}
    new_vocab["<PAD>"] = {"token": "<PAD>", "id": 0, "type": "SPECIAL"}
    new_vocab[UNK] = {"token": UNK, "id": 1, "type": "SPECIAL"}

    next_id = 2
    # Add semantic tokens sorted by frequency
    for tok in sorted(semantic, key=lambda t: freq[t], reverse=True):
        new_vocab[tok] = {"token": tok, "id": next_id, "type": tok.split(":")[0] if ":" in tok else "SPECIAL"}
        next_id += 1

    # Add kept LIT tokens sorted by frequency
    for tok in lit_ranked[:lit_budget]:
        new_vocab[tok] = {"token": tok, "id": next_id, "type": "LIT"}
        next_id += 1

    print(f"Final vocab size: {len(new_vocab)}")
    assert len(new_vocab) <= cap, f"Vocab {len(new_vocab)} > cap {cap}"

    # Token-to-id mapping
    tok2id = {entry["token"]: entry["id"] for entry in new_vocab.values()}
    unk_id = tok2id[UNK]

    # Pass 2: remap IDs
    out_dir = args.out_dir if args.out_dir is not None else Path(f"data/tokenized/cst-{cap//1000}k")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"train-{len(rows)}.jsonl"
    out_vocab = out_dir / f"train-{len(rows)}-vocab.json"

    unk_count = 0
    total_count = 0
    with open(out_jsonl, "w") as f:
        for obj in rows:
            new_ids = []
            for tok in obj["tokens"]:
                tid = tok2id.get(tok, unk_id)
                if tid == unk_id and tok != UNK:
                    unk_count += 1
                new_ids.append(tid)
                total_count += 1
            f.write(json.dumps({"ids": new_ids, "tokens": obj["tokens"], "text": obj["text"]}) + "\n")

    # Save vocab as list (same format as original CST vocab)
    vocab_list = sorted(new_vocab.values(), key=lambda x: x["id"])
    with open(out_vocab, "w") as f:
        json.dump(vocab_list, f)

    print(f"\nWritten: {out_jsonl}")
    print(f"Written: {out_vocab}")
    print(f"UNK rate: {unk_count}/{total_count} = {unk_count/total_count*100:.1f}%")

    # Verify
    max_id = 0
    with open(out_jsonl) as f:
        for line in f:
            for i in json.loads(line.strip())["ids"]:
                if i > max_id:
                    max_id = i
    print(f"Max token ID: {max_id} (vocab: {len(new_vocab)})")


if __name__ == "__main__":
    main()
