"""Train a tiny verifier model for reasoning traces.

Expected input JSONL format:
- input_tokens: list[str]
- label: int (1 positive, 0 negative)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import binary_metrics, ids_from_tokens, load_jsonl, load_vocab_file, pick_device
from .model import VerifierConfig, VerifierModel, load_checkpoint, save_checkpoint


class VerifierDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, rows: list[dict[str, Any]], vocab: dict[str, int], max_len: int) -> None:
        self.items: list[tuple[list[int], int]] = []
        for row in rows:
            toks = row.get("input_tokens") or []
            label = int(row.get("label", 0))
            if not toks:
                continue
            ids = ids_from_tokens(list(toks), vocab)
            ids = self._truncate(ids, max_len=max_len)
            self.items.append((ids, label))

    @staticmethod
    def _truncate(ids: list[int], *, max_len: int) -> list[int]:
        if len(ids) <= max_len:
            return ids
        # Keep both prefix and suffix so we do not drop question or answer entirely.
        head = max_len // 2
        tail = max_len - head
        return ids[:head] + ids[-tail:]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.items[idx]


class PairwiseDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(self, pairs: list[tuple[list[int], list[int]]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.pairs[idx]


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    pos = [r for r in rows if int(r.get("label", 0)) == 1]
    neg = [r for r in rows if int(r.get("label", 0)) == 0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(bucket: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        n = len(bucket)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test = bucket[:n_test]
        val = bucket[n_test : n_test + n_val]
        train = bucket[n_test + n_val :]
        return train, val, test

    tr_p, va_p, te_p = split_one(pos)
    tr_n, va_n, te_n = split_one(neg)

    train = tr_p + tr_n
    val = va_p + va_n
    test = te_p + te_n

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_pairwise_constraints(
    rows: list[dict[str, Any]],
    *,
    vocab: dict[str, int],
    max_len: int,
    pair_variants: list[str],
) -> list[tuple[list[int], list[int]]]:
    by_source: dict[str, dict[str, list[list[int]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        src = str(row.get("source_id", row.get("id", ""))).strip()
        if not src:
            continue
        variant = str(row.get("variant", ""))
        toks = list(row.get("input_tokens") or [])
        if not toks:
            continue
        ids = ids_from_tokens(toks, vocab)
        ids = VerifierDataset._truncate(ids, max_len=max_len)
        by_source[src][variant].append(ids)

    pairs: list[tuple[list[int], list[int]]] = []
    for groups in by_source.values():
        pos_list = groups.get("positive", [])
        if not pos_list:
            continue
        pos_ids = pos_list[0]
        for v in pair_variants:
            neg_list = groups.get(v, [])
            if neg_list:
                pairs.append((pos_ids, neg_list[0]))
    return pairs


def pad_sequence_batch(seq_batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in seq_batch)
    ids_tensor = torch.full((len(seq_batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seq_batch), max_len), dtype=torch.bool)

    for i, ids in enumerate(seq_batch):
        n = len(ids)
        ids_tensor[i, :n] = torch.tensor(ids, dtype=torch.long)
        mask[i, :n] = True
    return ids_tensor, mask


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seqs = [ids for ids, _ in batch]
        ids_tensor, mask = pad_sequence_batch(seqs, pad_id)
        labels = torch.zeros((len(batch),), dtype=torch.float32)

        for i, (_, label) in enumerate(batch):
            labels[i] = float(label)
        return ids_tensor, mask, labels

    return collate


def make_pair_collate_fn(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = [p for p, _ in batch]
        neg = [n for _, n in batch]
        pos_ids, pos_mask = pad_sequence_batch(pos, pad_id)
        neg_ids, neg_mask = pad_sequence_batch(neg, pad_id)
        return pos_ids, pos_mask, neg_ids, neg_mask

    return collate


def evaluate(
    model: VerifierModel,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    loss_total = 0.0
    n_batches = 0
    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for ids, mask, y in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            logits = model(ids, mask)
            loss = criterion(logits, y)
            loss_total += float(loss.item())
            n_batches += 1

            p = (logits >= 0.0).long().detach().cpu().tolist()
            yy = y.long().detach().cpu().tolist()
            preds.extend(int(v) for v in p)
            labels.extend(int(v) for v in yy)

    metrics = binary_metrics(preds, labels)
    avg_loss = loss_total / max(1, n_batches)
    return avg_loss, metrics


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    vocab = load_vocab_file(args.vocab)
    pad_id = vocab["[PAD]"]

    rows = load_jsonl(args.data)
    train_rows, val_rows, test_rows = stratified_split(
        rows,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_ds = VerifierDataset(train_rows, vocab=vocab, max_len=args.max_len)
    val_ds = VerifierDataset(val_rows, vocab=vocab, max_len=args.max_len)
    test_ds = VerifierDataset(test_rows, vocab=vocab, max_len=args.max_len)

    if len(train_ds) == 0:
        raise ValueError("Training split is empty after preprocessing. Check input JSONL and labels.")
    if len(val_ds) == 0:
        raise ValueError("Validation split is empty. Lower --val-ratio or provide more data.")
    if len(test_ds) == 0:
        raise ValueError("Test split is empty. Lower --test-ratio or provide more data.")

    collate = make_collate_fn(pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    pair_variants = [s.strip() for s in args.pairwise_variants.split(",") if s.strip()]
    pair_examples: list[tuple[list[int], list[int]]] = []
    pair_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    if not args.disable_pairwise and pair_variants:
        pair_examples = build_pairwise_constraints(
            train_rows,
            vocab=vocab,
            max_len=args.max_len,
            pair_variants=pair_variants,
        )
        if pair_examples:
            pair_ds = PairwiseDataset(pair_examples)
            pair_loader = DataLoader(
                pair_ds,
                batch_size=min(args.batch_size, max(1, len(pair_ds))),
                shuffle=True,
                collate_fn=make_pair_collate_fn(pad_id),
                num_workers=0,
            )

    cfg = VerifierConfig(
        vocab_size=max(vocab.values()) + 1,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )
    model = VerifierModel(cfg).to(device)

    y_train = [lbl for _, lbl in train_ds.items]
    pos = sum(1 for y in y_train if y == 1)
    neg = sum(1 for y in y_train if y == 0)
    pos_weight = torch.tensor(float(neg / max(1, pos)), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / "best.pt"
    final_ckpt = out_dir / "final.pt"

    history: list[dict[str, Any]] = []
    best_key = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_rank = 0.0
        steps = 0

        pair_iter = cycle(pair_loader) if pair_loader is not None else None

        for ids, mask, y in train_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(ids, mask)
            bce_loss = criterion(logits, y)

            rank_loss = torch.zeros((), device=device)
            if pair_iter is not None:
                p_ids, p_mask, n_ids, n_mask = next(pair_iter)
                p_ids = p_ids.to(device)
                p_mask = p_mask.to(device)
                n_ids = n_ids.to(device)
                n_mask = n_mask.to(device)

                pos_logits = model(p_ids, p_mask)
                neg_logits = model(n_ids, n_mask)
                rank_loss = torch.relu(args.pairwise_margin - (pos_logits - neg_logits)).mean()

            loss = bce_loss + args.pairwise_weight * rank_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running_loss += float(loss.item())
            running_bce += float(bce_loss.item())
            running_rank += float(rank_loss.item())
            steps += 1

        train_loss = running_loss / max(1, steps)
        train_bce = running_bce / max(1, steps)
        train_rank = running_rank / max(1, steps)
        val_loss, val_metrics = evaluate(model, val_loader, device=device, criterion=criterion)

        f1 = float(val_metrics["f1"])
        acc = float(val_metrics["accuracy"])
        key = f1 + 0.001 * acc
        is_best = key > best_key

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_bce": round(train_bce, 6),
            "train_pairwise": round(train_rank, 6),
            "val_loss": round(val_loss, 6),
            **val_metrics,
            "is_best": is_best,
        }
        history.append(row)
        print(row)

        if is_best:
            best_key = key
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(
                best_ckpt,
                model=model,
                cfg=cfg,
                pad_id=pad_id,
                extra={
                    "epoch": epoch,
                    "best_key": best_key,
                    "vocab_path": str(args.vocab),
                },
            )
        else:
            bad_epochs += 1

        if args.patience > 0 and bad_epochs >= args.patience:
            print({"early_stop": True, "epoch": epoch, "bad_epochs": bad_epochs})
            break

    save_checkpoint(
        final_ckpt,
        model=model,
        cfg=cfg,
        pad_id=pad_id,
        extra={"epoch": len(history), "vocab_path": str(args.vocab)},
    )

    best_model, _, _, _ = load_checkpoint(best_ckpt, device=device)
    test_loss, test_metrics = evaluate(best_model, test_loader, device=device, criterion=criterion)

    summary = {
        "device": str(device),
        "seed": args.seed,
        "config": asdict(cfg),
        "objective": {
            "bce": True,
            "pairwise_enabled": pair_loader is not None,
            "pairwise_weight": args.pairwise_weight,
            "pairwise_margin": args.pairwise_margin,
            "pairwise_variants": pair_variants,
            "pair_constraints": len(pair_examples),
        },
        "sizes": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "label_balance_train": {"pos": pos, "neg": neg},
        "best_epoch": best_epoch,
        "best_key": round(best_key, 6),
        "test_loss": round(test_loss, 6),
        "test_metrics": test_metrics,
        "checkpoints": {
            "best": str(best_ckpt),
            "final": str(final_ckpt),
        },
    }

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True, help="Verifier JSONL path produced by build_dataset.py")
    ap.add_argument("--vocab", type=Path, required=True, help="Tokenizer vocab JSON path")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--disable-pairwise", action="store_true", default=False)
    ap.add_argument("--pairwise-weight", type=float, default=0.8)
    ap.add_argument("--pairwise-margin", type=float, default=0.2)
    ap.add_argument(
        "--pairwise-variants",
        type=str,
        default="neg_question_only,neg_shuffled_cot",
        help="Comma-separated negative variants that full trace must outrank",
    )

    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
