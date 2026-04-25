from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch

TokenView = Literal["reasoning", "default"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_gold_label(raw: Any) -> str | None:
    a = str(raw).strip().lower()
    if a in {"yes", "نعم", "valid", "true"}:
        return "yes"
    if a in {"no", "لا", "invalid", "false"}:
        return "no"
    return None


def record_gold_label(rec: dict[str, Any]) -> str | None:
    return normalize_gold_label(rec.get("answer", rec.get("meta", {}).get("answer", "")))


def record_lang(rec: dict[str, Any]) -> str:
    return str(rec.get("lang", rec.get("meta", {}).get("lang", "en")))


def record_difficulty(rec: dict[str, Any]) -> str:
    return str(rec.get("meta", {}).get("difficulty", "unknown")).strip().lower()


def record_schema(rec: dict[str, Any]) -> str:
    if "q_ids" in rec and "a_ids" in rec:
        return "logic"
    if "question_tokens" in rec and "answer_tokens" in rec:
        return "reasoning"
    return "unknown"


def extract_token_segments(
    rec: dict[str, Any],
    *,
    view: TokenView,
) -> tuple[list[str], list[list[str]], list[str]]:
    q_blob = rec.get("question_tokens") or {}
    a_blob = rec.get("answer_tokens") or {}

    question = list(q_blob.get(view) or [])
    answer = list(a_blob.get(view) or [])

    if view != "reasoning":
        if not question:
            question = list(q_blob.get("reasoning") or [])
        if not answer:
            answer = list(a_blob.get("reasoning") or [])

    cot_steps: list[list[str]] = []
    for step in rec.get("cot_tokens") or []:
        if not isinstance(step, dict):
            continue
        toks = list(step.get(view) or [])
        if view != "reasoning" and not toks:
            toks = list(step.get("reasoning") or [])
        cot_steps.append(toks)

    return question, cot_steps, answer


def load_vocab_file(vocab_path: Path) -> dict[str, int]:
    blob = json.loads(vocab_path.read_text(encoding="utf-8"))

    if isinstance(blob, dict) and "token_to_id" in blob and isinstance(blob["token_to_id"], dict):
        vocab = {str(k): int(v) for k, v in blob["token_to_id"].items()}
    elif isinstance(blob, dict) and all(isinstance(v, (int, float)) for v in blob.values()):
        vocab = {str(k): int(v) for k, v in blob.items()}
    elif isinstance(blob, list):
        vocab = {str(tok): i for i, tok in enumerate(blob)}
    elif isinstance(blob, dict) and all(isinstance(v, dict) for v in blob.values()):
        # Some vocab files are language-scoped (e.g., {"en": {...}, "ar": {...}}).
        # Merge into one stable global map for multilingual training/eval.
        merged_tokens: set[str] = set()
        for part in blob.values():
            for tok in part.keys():
                merged_tokens.add(str(tok))

        vocab = {}
        specials = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[Q]"]
        for tok in specials:
            if tok in merged_tokens and tok not in vocab:
                vocab[tok] = len(vocab)
        for tok in sorted(merged_tokens):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    else:
        raise ValueError(f"Unsupported vocab format in {vocab_path}")

    if "[PAD]" not in vocab:
        vocab["[PAD]"] = max(vocab.values(), default=-1) + 1
    if "[UNK]" not in vocab:
        vocab["[UNK]"] = max(vocab.values(), default=-1) + 1
    return vocab


def compose_reasoning_tokens(
    question: list[str],
    cot_steps: list[list[str]],
    answer: list[str],
) -> list[str]:
    out = list(question)
    for step in cot_steps:
        out.extend(step)
    out.extend(answer)
    return out


def ids_from_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["[UNK]"]
    return [vocab.get(t, unk) for t in tokens]


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_metrics(preds: list[int], labels: list[int]) -> dict[str, float | int]:
    assert len(preds) == len(labels)
    tp = fp = tn = fn = 0
    for p, y in zip(preds, labels):
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 0:
            tn += 1
        else:
            fn += 1

    n = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / n

    return {
        "n": len(labels),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
