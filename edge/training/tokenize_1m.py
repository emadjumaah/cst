"""Experiment script: tokenize 1M Arabic Wikipedia sentences into CST.

This is a **CLI driver** for one specific experiment (the 1M-sentence
Arabic Wikipedia run). It is NOT the tokenizer itself — all tokenizer
logic lives in :mod:`edge.arabic_tokenizer`. Write new experiment
scripts alongside this one; do not duplicate the tokenizer code.

Run (Colab):
  !pip install camel-tools
  !camel_data -i morphology-db-msa-r13
  !python tokenize_1m.py

Output: /content/cst_1m/train-1000000.jsonl + train-1000000-vocab.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Make the sibling library importable whether this script is run from
# the repo root (``python edge/training/tokenize_1m.py``) or from inside
# ``edge/training/`` directly.
_HERE = Path(__file__).resolve().parent
_EDGE = _HERE.parent
if str(_EDGE) not in sys.path:
    sys.path.insert(0, str(_EDGE))

from arabic_tokenizer import ArabicCSTTokenizer  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# Download — uses HuggingFace `datasets` library (no rate limits)
# ═══════════════════════════════════════════════════════════════

def download_sentences(target, output_path):
    if os.path.exists(output_path):
        print(f"  Loading cached: {output_path}")
        with open(output_path) as f:
            return json.load(f)

    from datasets import load_dataset

    print("  Downloading Arabic Wikipedia via `datasets` library...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.ar", split="train", streaming=True
    )

    sentences = []
    articles = 0
    t0 = time.time()

    for row in ds:
        text = row.get("text", "")
        for sent in re.split(r"[.؟!]\s*", text):
            sent = sent.strip()
            if len(sent) < 20 or len(sent) > 300:
                continue
            if sum(1 for c in sent if "\u0600" <= c <= "\u06FF") < len(sent) * 0.5:
                continue
            sentences.append(sent)
            if len(sentences) >= target:
                break
        if len(sentences) >= target:
            break
        articles += 1
        if articles % 10000 == 0:
            elapsed = time.time() - t0
            print(
                f"    {len(sentences):,} / {target:,} sentences "
                f"from {articles:,} articles ({elapsed:.0f}s)"
            )

    sentences = sentences[:target]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=0)
    elapsed = time.time() - t0
    print(f"  Saved {len(sentences):,} sentences ({elapsed:.0f}s)")
    return sentences


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    TARGET = 1_000_000
    DATA_DIR = "/content"
    GDRIVE_DIR = "/content/drive/MyDrive/cst-data"  # from download_data.py
    OUT_DIR = "/content/cst_1m"
    os.makedirs(OUT_DIR, exist_ok=True)

    sentences_path = f"{DATA_DIR}/sentences-{TARGET}.json"
    gdrive_path = os.path.join(GDRIVE_DIR, "sentences-1M.json")

    print("=" * 60)
    print(f"  Arabic CST Tokenization — {TARGET:,} sentences")
    print("=" * 60)

    # Step 1: Load sentences
    print("\n── Step 1: Load sentences ──")
    if os.path.exists(gdrive_path):
        print(f"  Loading from Google Drive: {gdrive_path}")
        with open(gdrive_path) as f:
            sentences = json.load(f)
        print(f"  Loaded {len(sentences):,} sentences")
    else:
        sentences = download_sentences(TARGET, sentences_path)
    print(f"  Total: {len(sentences):,} sentences")

    # Step 2: Tokenize
    print("\n── Step 2: CST Tokenize ──")
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer

    print("  Loading camel-tools...")
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    tokenizer = ArabicCSTTokenizer(analyzer)

    n = len(sentences)
    output_path = os.path.join(OUT_DIR, f"train-{n}.jsonl")
    lines = []
    t0 = time.time()

    for i, sent in enumerate(sentences):
        result = tokenizer.tokenize(sent)
        if len(result["ids"]) < 4:
            continue
        lines.append(json.dumps(result, ensure_ascii=False))
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60
            print(
                f"    {i+1:,} / {n:,} "
                f"({elapsed:.0f}s, {rate:.0f} sent/s, ETA {eta:.0f}min)"
            )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    vocab_path = output_path.replace(".jsonl", "-vocab.json")
    tokenizer.save_vocab(vocab_path)

    elapsed = time.time() - t0
    total_tok = sum(len(json.loads(line)["ids"]) for line in lines)
    stats = dict(tokenizer.stats)
    total_s = sum(stats.values()) or 1

    print("\n  ═══ Done ═══")
    print(f"  Sentences:  {len(lines):,}")
    print(f"  Vocab size: {tokenizer.next_id:,}")
    print(f"  Tokens:     {total_tok:,}")
    print(f"  Avg tok/s:  {total_tok/len(lines):.1f}")
    print(f"  Time:       {elapsed/60:.1f} min")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {k:12s} {v:10,} ({v/total_s*100:5.1f}%)")

    print(f"\n  Output: {output_path}")
    print(f"  Vocab:  {vocab_path}")
    print("\n  Next: upload to Colab and run colab_edge.py with:")
    print(f'    DATA_FILE = "train-{n}.jsonl"')
    print(f'    VOCAB_FILE = "train-{n}-vocab.json"')


if __name__ == "__main__":
    main()
