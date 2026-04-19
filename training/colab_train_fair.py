"""
FAIR CST vs SentencePiece BPE — Same vocab size, same params (Google Colab)

Upload these 8 files to Colab:
  - cst-8k-train-99963.jsonl  + cst-8k-train-99963-vocab.json
  - spm-8k-train-99963.jsonl  + spm-8k-train-99963-vocab.json
  - cst-32k-train-99963.jsonl + cst-32k-train-99963-vocab.json
  - spm-32k-train-99963.jsonl + spm-32k-train-99963-vocab.json

This is the FAIR test: same vocab budget → same embedding size → same total params.
The ONLY difference is tokenization strategy (semantic vs subword).
Primary metric: BPC (bits-per-character).

Runtime → Change runtime type → T4 GPU.
"""

# !pip install torch transformers -q

import json
import math
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

# ── Config ──

DATA_DIR = "/content"

# Fair pairs: same vocab size
EXPERIMENTS = [
    {
        "group": "8K vocab",
        "pairs": [
            {"name": "CST-8K",  "data": "cst-8k-train-99963.jsonl",  "vocab": "cst-8k-train-99963-vocab.json"},
            {"name": "SPM-8K",  "data": "spm-8k-train-99963.jsonl",  "vocab": "spm-8k-train-99963-vocab.json"},
        ],
    },
    {
        "group": "32K vocab",
        "pairs": [
            {"name": "CST-32K", "data": "cst-32k-train-99963.jsonl", "vocab": "cst-32k-train-99963-vocab.json"},
            {"name": "SPM-32K", "data": "spm-32k-train-99963.jsonl", "vocab": "spm-32k-train-99963-vocab.json"},
        ],
    },
]

PRESET = dict(n_embd=256, n_layer=6, n_head=4)  # 10M class
MODEL_TAG = "10m"
EPOCHS = 3
BATCH_SIZE = 32
LR = 3e-4
MAX_LEN = 128
VAL_RATIO = 0.1


# ── Dataset ──

class JsonlDataset(Dataset):
    def __init__(self, ids_list, char_counts):
        self.ids_list = ids_list
        self.char_counts = char_counts

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], self.char_counts[idx]


def load_jsonl(path, max_len=128):
    ids_list = []
    char_counts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids = obj["ids"]
            text = obj.get("text", "")
            if len(ids) < 4:
                continue
            if len(ids) > max_len:
                ratio = max_len / len(ids)
                ids = ids[:max_len]
                char_counts.append(int(len(text) * ratio))
            else:
                char_counts.append(len(text))
            ids_list.append(ids)
    return ids_list, char_counts


def collate_fn(batch, pad_id=0):
    ids_list = [item[0] for item in batch]
    char_counts = [item[1] for item in batch]
    max_len = max(len(seq) for seq in ids_list)
    padded = []
    masks = []
    for seq in ids_list:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "char_counts": char_counts,
    }


# ── Training ──

def train_model(name, train_ids, train_chars, val_ids, val_chars, vocab_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd = PRESET["n_embd"]
    embed_params = vocab_size * n_embd
    pos_params = MAX_LEN * n_embd

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Device: {device} | Vocab: {vocab_size:,}")
    print(f"{'='*60}\n")

    train_ds = JsonlDataset(train_ids, train_chars)
    val_ds = JsonlDataset(val_ids, val_chars)

    val_total_tokens = sum(len(ids) for ids in val_ids)
    val_total_chars = sum(val_chars)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"  Val tokens: {val_total_tokens:,} | Val chars: {val_total_chars:,}")
    print(f"  Avg tokens/sentence: {val_total_tokens/len(val_ids):.1f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_LEN,
        n_embd=n_embd,
        n_layer=PRESET["n_layer"],
        n_head=PRESET["n_head"],
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    transformer_params = total_params - embed_params - pos_params
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"    Embedding:   {embed_params:,} ({embed_params/1e6:.1f}M)")
    print(f"    Transformer: {transformer_params:,} ({transformer_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_loader)
    )

    best_val_bpc = float("inf")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "val_bpc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_tokens = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

            if (step + 1) % 200 == 0:
                avg = total_loss / total_tokens
                elapsed = time.time() - t0
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Loss: {avg:.4f} | {elapsed:.0f}s")

        train_loss = total_loss / total_tokens
        history["train_loss"].append(train_loss)

        model.eval()
        val_nll = 0
        val_toks = 0
        val_chars = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                bt = attention_mask.sum().item()
                val_nll += outputs.loss.item() * bt
                val_toks += bt
                val_chars += sum(batch["char_counts"])

        val_loss = val_nll / val_toks
        val_ppl = math.exp(min(val_loss, 20))
        val_bpc = val_nll / val_chars / math.log(2)

        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_bpc"].append(val_bpc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc

        elapsed = time.time() - t0
        print(f"\n  Epoch {epoch}/{EPOCHS} | Loss: {val_loss:.4f} | PPL: {val_ppl:.1f} | BPC: {val_bpc:.4f} | {elapsed:.0f}s\n")

    return {
        "name": name,
        "params": total_params,
        "embed_params": embed_params,
        "transformer_params": transformer_params,
        "vocab_size": vocab_size,
        "val_tokens": val_total_tokens,
        "val_chars": val_total_chars,
        "avg_toks": val_total_tokens / len(val_ids),
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "best_val_bpc": best_val_bpc,
        "history": history,
    }


# ── Run all experiments ──

all_results = []

for exp in EXPERIMENTS:
    print(f"\n{'#'*70}")
    print(f"  GROUP: {exp['group']}")
    print(f"{'#'*70}")

    for cfg in exp["pairs"]:
        data_path = os.path.join(DATA_DIR, cfg["data"])
        vocab_path = os.path.join(DATA_DIR, cfg["vocab"])

        if not os.path.exists(data_path):
            print(f"MISSING: {data_path}")
            continue

        print(f"\nLoading {cfg['name']}...")
        ids_list, char_counts = load_jsonl(data_path, MAX_LEN)
        split_idx = int(len(ids_list) * (1 - VAL_RATIO))

        with open(vocab_path) as f:
            vocab = json.load(f)
        if isinstance(vocab, dict):
            vocab_size = len(vocab)
        elif isinstance(vocab, list):
            vocab_size = max(entry["id"] for entry in vocab) + 1
        else:
            vocab_size = 50000

        result = train_model(
            name=cfg["name"],
            train_ids=ids_list[:split_idx],
            train_chars=char_counts[:split_idx],
            val_ids=ids_list[split_idx:],
            val_chars=char_counts[split_idx:],
            vocab_size=vocab_size,
        )
        result["group"] = exp["group"]
        all_results.append(result)


# ── Save machine-readable results ──

RESULTS_PATH = os.path.join(DATA_DIR, "results_english.json")
with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved machine-readable results to {RESULTS_PATH}")


# ── Final comparison ──

print("\n" + "=" * 80)
print("  FAIR COMPARISON: CST vs SentencePiece (same vocab budget)")
print("=" * 80)

header = f"  {'Metric':<22}"
for r in all_results:
    header += f" {r['name']:>12}"
print(header)
print(f"  {'-'*75}")

rows = [
    ("Vocab",           lambda r: f"{r['vocab_size']:>12,}"),
    ("Params",          lambda r: f"{r['params']/1e6:>11.1f}M"),
    ("  Embedding",     lambda r: f"{r['embed_params']/1e6:>11.1f}M"),
    ("  Transformer",   lambda r: f"{r['transformer_params']/1e6:>11.1f}M"),
    ("Avg tok/sent",    lambda r: f"{r['avg_toks']:>12.1f}"),
    ("Val PPL",         lambda r: f"{r['best_val_ppl']:>12.1f}"),
    ("*** BPC ***",     lambda r: f"{r['best_val_bpc']:>12.4f}"),
]
for label, fmt in rows:
    line = f"  {label:<22}"
    for r in all_results:
        line += fmt(r)
    print(line)

print(f"  {'-'*75}")

# Pairwise within each group
for exp in EXPERIMENTS:
    group_results = [r for r in all_results if r["group"] == exp["group"]]
    if len(group_results) == 2:
        a, b = group_results
        bpc_diff = ((b["best_val_bpc"] - a["best_val_bpc"]) / b["best_val_bpc"]) * 100
        tok_ratio = a["avg_toks"] / b["avg_toks"]
        print(f"\n  {exp['group']}:")
        print(f"    {a['name']}: BPC={a['best_val_bpc']:.4f}, params={a['params']/1e6:.1f}M, avg {a['avg_toks']:.1f} tok/sent")
        print(f"    {b['name']}: BPC={b['best_val_bpc']:.4f}, params={b['params']/1e6:.1f}M, avg {b['avg_toks']:.1f} tok/sent")
        print(f"    BPC delta: {bpc_diff:+.1f}% | Token ratio: {tok_ratio:.2f}x")
        if a["best_val_bpc"] < b["best_val_bpc"]:
            print(f"    → {a['name']} wins (lower BPC = better compression)")
        elif a["best_val_bpc"] > b["best_val_bpc"]:
            print(f"    → {b['name']} wins (lower BPC = better compression)")
        else:
            print(f"    → Tie")

print(f"\n  BPC = bits-per-character (normalized across tokenizers)")
print(f"  Same vocab → same embedding size → same total params")
print(f"  The ONLY variable is tokenization strategy")
