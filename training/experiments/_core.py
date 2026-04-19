"""
Shared training / evaluation core used by the multi-seed runner and the
ablation scripts in this directory.

The public surface is intentionally narrow:

    PRESET                      - model hyperparameter dict
    build_model(vocab_size)     - returns a fresh GPT2LMHeadModel
    load_jsonl(path, max_len)   - returns (ids_list, char_counts)
    train_and_eval(name, ...)   - returns a results dict (with best_val_bpc)

All scripts here reuse the same architecture, optimizer, schedule, and data
layout as ``colab_train_fair.py`` / ``colab_train_ar.py``. Keeping one copy
of the training loop eliminates silent divergence between experiments.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel


PRESET = dict(n_embd=256, n_layer=6, n_head=4)
DEFAULTS = dict(
    epochs=3,
    batch_size=32,
    lr=3e-4,
    max_len=128,
    val_ratio=0.1,
    weight_decay=0.01,
    grad_clip=1.0,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JsonlDataset(Dataset):
    def __init__(self, ids_list, char_counts):
        self.ids_list = ids_list
        self.char_counts = char_counts

    def __len__(self) -> int:
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], self.char_counts[idx]


def load_jsonl(path: str, max_len: int = 128):
    """Load a tokenized .jsonl file. Each line: {"ids": [...], "text": "..."}."""
    ids_list: list[list[int]] = []
    char_counts: list[int] = []
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


def load_vocab_size(vocab_path: str) -> int:
    with open(vocab_path) as f:
        vocab = json.load(f)
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return max(entry["id"] for entry in vocab) + 1
    raise ValueError(f"Unrecognized vocab format in {vocab_path}")


def collate_fn(batch, pad_id: int = 0):
    ids_list = [item[0] for item in batch]
    char_counts = [item[1] for item in batch]
    max_len = max(len(seq) for seq in ids_list)
    padded, masks = [], []
    for seq in ids_list:
        pad_len = max_len - len(seq)
        padded.append(seq + [pad_id] * pad_len)
        masks.append([1] * len(seq) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "char_counts": char_counts,
    }


def build_model(vocab_size: int, max_len: int = 128) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_len,
        n_embd=PRESET["n_embd"],
        n_layer=PRESET["n_layer"],
        n_head=PRESET["n_head"],
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config)


def train_and_eval(
    name: str,
    train_ids,
    train_chars,
    val_ids,
    val_chars,
    vocab_size: int,
    seed: int = 0,
    epochs: int = DEFAULTS["epochs"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    max_len: int = DEFAULTS["max_len"],
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a fresh GPT-2 on the given tokenized data, evaluate on val.

    Returns a dict with keys: name, seed, vocab_size, params, best_val_loss,
    best_val_ppl, best_val_bpc, history.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = JsonlDataset(train_ids, train_chars)
    val_ds = JsonlDataset(val_ids, val_chars)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    val_tokens = sum(len(ids) for ids in val_ids)
    val_chars_total = sum(val_chars)

    model = build_model(vocab_size, max_len=max_len).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = vocab_size * PRESET["n_embd"]
    pos_params = max_len * PRESET["n_embd"]
    transformer_params = total_params - embed_params - pos_params

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=DEFAULTS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    if verbose:
        print(f"\n  [{name}] seed={seed} vocab={vocab_size:,} device={device} "
              f"params={total_params/1e6:.1f}M (trans {transformer_params/1e6:.1f}M)")

    best_val_loss = float("inf")
    best_val_bpc = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "val_bpc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), DEFAULTS["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bt = attention_mask.sum().item()
            total_loss += outputs.loss.item() * bt
            total_tokens += bt
        train_loss = total_loss / total_tokens
        history["train_loss"].append(train_loss)

        model.eval()
        val_nll = 0.0
        val_toks = 0
        val_ch = 0
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
                val_ch += sum(batch["char_counts"])
        val_loss = val_nll / val_toks
        val_ppl = math.exp(min(val_loss, 20))
        val_bpc = val_nll / val_ch / math.log(2)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_bpc"].append(val_bpc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
        if verbose:
            print(f"    epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}  val_bpc={val_bpc:.4f}  "
                  f"{time.time()-t0:.0f}s")

    return {
        "name": name,
        "seed": seed,
        "vocab_size": vocab_size,
        "params": total_params,
        "embed_params": embed_params,
        "transformer_params": transformer_params,
        "val_tokens": val_tokens,
        "val_chars": val_chars_total,
        "avg_toks": val_tokens / len(val_ids),
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "best_val_bpc": best_val_bpc,
        "history": history,
    }


def split_train_val(ids_list, char_counts, val_ratio: float = 0.1):
    split_idx = int(len(ids_list) * (1 - val_ratio))
    return (
        ids_list[:split_idx],
        char_counts[:split_idx],
        ids_list[split_idx:],
        char_counts[split_idx:],
    )
