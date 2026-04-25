"""Proof-oriented hard-logic evaluation for CST checkpoints.

This script is intentionally independent from notebook gating logic. It measures
whether a trained CST tiny model behaves like a reasoning model on hard logic by
running three modes on the same split:

- full: question + CoT prefix
- question_only: question prefix only (no CoT)
- shuffled_cot: question + CoT from another sample in same language

If full stays high while control modes are near full, the model is likely not
using the reasoning chain. If full clearly exceeds controls and majority label,
that is evidence of reasoning behavior on this set.

Example:
    python -m reasoning.eval.hard_logic_proof \
      --ckpt reasoning/runs/colab-cst-v0.2/ckpt.pt \
      --vocab reasoning/tokenized/vocab-reasoning.json \
      --data reasoning/eval/holdout_ood_a1_tokenized/syllogisms.tokenized.jsonl \
      --difficulty hard \
      --out reasoning/runs/colab-cst-v0.2/proof_hard_logic.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_vocab(vocab_path: Path) -> dict[str, int]:
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def ids_from_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["[UNK]"]
    return [vocab.get(t, unk) for t in tokens]


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_len: int = 256
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, width = x.shape
        qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = ids.shape
        pos = torch.arange(seqlen, device=ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _gold_label(rec: dict[str, Any]) -> str | None:
    raw = str(rec.get("answer", rec.get("meta", {}).get("answer", ""))).strip().lower()
    if raw in {"yes", "نعم", "valid", "true"}:
        return "yes"
    if raw in {"no", "لا", "invalid", "false"}:
        return "no"
    return None


def _lang(rec: dict[str, Any]) -> str:
    return str(rec.get("lang", rec.get("meta", {}).get("lang", "en")))


def _difficulty(rec: dict[str, Any]) -> str:
    return str(rec.get("meta", {}).get("difficulty", "unknown")).strip().lower()


def _schema(rec: dict[str, Any]) -> str:
    if "q_ids" in rec and "a_ids" in rec:
        return "logic"
    if "question_tokens" in rec and "answer_tokens" in rec:
        return "reasoning"
    return "unknown"


def _score(
    model: TinyGPT,
    prefix_ids: list[int],
    cand_ids: list[int],
    max_len: int,
    device: torch.device,
) -> float:
    seq = prefix_ids + cand_ids
    if len(seq) > max_len:
        seq = seq[-max_len:]

    n_cand = min(len(cand_ids), len(seq))
    ids = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids)
        logp = F.log_softmax(logits, dim=-1)

    t_total = ids.size(1)
    total = 0.0
    for off in range(n_cand):
        t = t_total - n_cand + off
        prev = t - 1
        if prev < 0:
            continue
        total += logp[0, prev, ids[0, t]].item()
    return total


def _derive_reasoning_candidates(
    rows: list[dict[str, Any]],
    vocab: dict[str, int],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
    specials = {"[BOS]", "[EOS]", "[PAD]", "[UNK]", "[SEP]"}
    observed = {
        ("en", "yes"): Counter(),
        ("en", "no"): Counter(),
        ("ar", "yes"): Counter(),
        ("ar", "no"): Counter(),
    }
    fallback = {
        "en": {
            "yes": ["ROOT:yes", "ROOT:true"],
            "no": ["REL:neg", "STR:neg:general", "ROOT:false"],
        },
        "ar": {
            "yes": ["ROOT:ن.ع.م", "ROOT:true"],
            "no": ["STR:neg:general", "REL:neg", "ROOT:false"],
        },
    }

    for rec in rows:
        if _schema(rec) != "reasoning":
            continue
        lang = _lang(rec)
        gold = _gold_label(rec)
        if lang not in {"en", "ar"} or gold not in {"yes", "no"}:
            continue
        toks = list((rec.get("answer_tokens") or {}).get("reasoning") or [])
        for tok in toks:
            if tok not in specials:
                observed[(lang, gold)][tok] += 1

    cand_tokens: dict[str, dict[str, list[str]]] = {"en": {}, "ar": {}}
    observed_top: dict[str, list[str]] = {}

    for lang in ("en", "ar"):
        for lbl in ("yes", "no"):
            ordered = [tok for tok, _ in observed[(lang, lbl)].most_common(5)] + fallback[lang][lbl]
            chosen = next((tok for tok in ordered if tok in vocab and tok != "[UNK]"), ordered[0])
            cand_tokens[lang][lbl] = ["[BOS]", chosen, "[EOS]"]
            if observed[(lang, lbl)]:
                observed_top[f"{lang}_{lbl}"] = [tok for tok, _ in observed[(lang, lbl)].most_common(5)]

    return cand_tokens, observed_top


def _build_shuffled_cot(rows: list[dict[str, Any]]) -> list[Any]:
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        by_lang[_lang(rec)].append(i)

    shuffled: list[Any] = [None] * len(rows)
    for _, idxs in by_lang.items():
        if not idxs:
            continue
        if len(idxs) == 1:
            src_for = {idxs[0]: idxs[0]}
        else:
            rotated = idxs[1:] + idxs[:1]
            src_for = {dst: src for dst, src in zip(idxs, rotated)}

        for dst, src in src_for.items():
            src_rec = rows[src]
            if _schema(src_rec) == "logic":
                shuffled[dst] = [list(step) for step in (src_rec.get("cot_ids") or [])]
            else:
                shuffled[dst] = [list((step.get("reasoning") or [])) for step in (src_rec.get("cot_tokens") or [])]

    return shuffled


def _prefix_ids(
    rec: dict[str, Any],
    mode: str,
    cot_override: Any,
    vocab: dict[str, int],
) -> list[int]:
    schema = _schema(rec)
    if schema == "logic":
        q = list(rec.get("q_ids") or [])
        if mode == "question_only":
            return q
        steps = list(rec.get("cot_ids") or [])
        if mode == "shuffled_cot" and cot_override is not None:
            steps = cot_override
        out = list(q)
        for st in steps:
            out.extend(st)
        return out

    q_toks = list((rec.get("question_tokens") or {}).get("reasoning") or [])
    if mode == "question_only":
        return ids_from_tokens(q_toks, vocab)

    steps = list(rec.get("cot_tokens") or [])
    if mode == "shuffled_cot" and cot_override is not None:
        steps = [{"reasoning": s} for s in cot_override]

    out_toks = list(q_toks)
    for st in steps:
        out_toks.extend(st.get("reasoning") or [])
    return ids_from_tokens(out_toks, vocab)


def _summary(counter_map: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key, (correct, total) in counter_map.items():
        out[key] = {
            "acc": round(correct / total, 4) if total else 0.0,
            "n": total,
        }
    return out


def _evaluate_mode(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    model: TinyGPT,
    vocab: dict[str, int],
    cfg: GPTConfig,
    device: torch.device,
    cand_ids: dict[str, dict[str, dict[str, list[int]]]],
    shuffled_cot: list[Any],
) -> dict[str, Any]:
    correct = 0
    total = 0
    by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_val: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_diff: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for i, rec in enumerate(rows):
        gold = _gold_label(rec)
        lang = _lang(rec)
        schema = _schema(rec)
        if gold is None or lang not in {"en", "ar"} or schema not in {"logic", "reasoning"}:
            continue

        cands = cand_ids[schema][lang]
        pref = _prefix_ids(rec, mode, shuffled_cot[i] if mode == "shuffled_cot" else None, vocab)
        scores = {
            lbl: _score(model, pref, ids, cfg.max_len, device)
            for lbl, ids in cands.items()
        }
        pred = max(scores, key=scores.get)
        ok = int(pred == gold)

        correct += ok
        total += 1
        by_lang[lang][0] += ok
        by_lang[lang][1] += 1
        vk = "valid" if gold == "yes" else "invalid"
        by_val[vk][0] += ok
        by_val[vk][1] += 1
        diff = _difficulty(rec)
        by_diff[diff][0] += ok
        by_diff[diff][1] += 1

    return {
        "accuracy": round(correct / max(1, total), 4),
        "n": total,
        "by_lang": _summary(by_lang),
        "by_validity": _summary(by_val),
        "by_difficulty": _summary(by_diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help="CST checkpoint path (ckpt.pt)")
    ap.add_argument("--vocab", type=Path, required=True, help="CST vocab path (vocab-reasoning.json)")
    ap.add_argument("--data", type=Path, required=True, help="Tokenized JSONL to evaluate")
    ap.add_argument("--difficulty", type=str, default="all", help="all|easy|medium|hard")
    ap.add_argument("--max", type=int, default=0, help="Max examples; <=0 means all")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic shuffled-CoT control")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    ap.add_argument("--min-examples", type=int, default=300)
    ap.add_argument("--min-full-acc", type=float, default=0.70)
    ap.add_argument("--min-gap-majority", type=float, default=0.15)
    ap.add_argument("--min-gap-question", type=float, default=0.10)
    ap.add_argument("--min-gap-shuffled", type=float, default=0.10)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reasoning/eval/hard_logic_proof.json"),
        help="Where to save report JSON",
    )
    args = ap.parse_args()

    if args.difficulty not in {"all", "easy", "medium", "hard"}:
        raise SystemExit("--difficulty must be one of: all, easy, medium, hard")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _pick_device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    vocab = load_vocab(args.vocab)
    rows_all = _load_jsonl(args.data)

    rows_filtered: list[dict[str, Any]] = []
    for rec in rows_all:
        if _gold_label(rec) is None:
            continue
        if _lang(rec) not in {"en", "ar"}:
            continue
        if _schema(rec) not in {"logic", "reasoning"}:
            continue
        if args.difficulty != "all" and _difficulty(rec) != args.difficulty:
            continue
        rows_filtered.append(rec)

    if args.max > 0:
        rows_filtered = rows_filtered[: args.max]

    if not rows_filtered:
        raise SystemExit("No usable rows after filtering (label/lang/schema/difficulty).")

    reasoning_tokens, observed_top = _derive_reasoning_candidates(rows_filtered, vocab)
    logic_tokens = {
        "en": {"yes": ["[BOS]", "V:Y", "[EOS]"], "no": ["[BOS]", "V:N", "[EOS]" ]},
        "ar": {"yes": ["[BOS]", "V:Y", "[EOS]"], "no": ["[BOS]", "V:N", "[EOS]" ]},
    }

    cand_ids = {
        "reasoning": {
            lang: {lbl: ids_from_tokens(toks, vocab) for lbl, toks in mp.items()}
            for lang, mp in reasoning_tokens.items()
        },
        "logic": {
            lang: {lbl: ids_from_tokens(toks, vocab) for lbl, toks in mp.items()}
            for lang, mp in logic_tokens.items()
        },
    }

    unk_id = vocab["[UNK]"]
    unknown_candidates: dict[str, list[str]] = {}
    for schema_name in ("reasoning", "logic"):
        for lang in ("en", "ar"):
            for lbl in ("yes", "no"):
                toks = reasoning_tokens[lang][lbl] if schema_name == "reasoning" else logic_tokens[lang][lbl]
                ids = cand_ids[schema_name][lang][lbl]
                missing = [t for t, i in zip(toks, ids) if t != "[UNK]" and i == unk_id]
                if missing:
                    unknown_candidates[f"{schema_name}:{lang}:{lbl}"] = missing

    shuffled_cot = _build_shuffled_cot(rows_filtered)

    full = _evaluate_mode(
        rows_filtered,
        mode="full",
        model=model,
        vocab=vocab,
        cfg=cfg,
        device=device,
        cand_ids=cand_ids,
        shuffled_cot=shuffled_cot,
    )
    q_only = _evaluate_mode(
        rows_filtered,
        mode="question_only",
        model=model,
        vocab=vocab,
        cfg=cfg,
        device=device,
        cand_ids=cand_ids,
        shuffled_cot=shuffled_cot,
    )
    shuffled = _evaluate_mode(
        rows_filtered,
        mode="shuffled_cot",
        model=model,
        vocab=vocab,
        cfg=cfg,
        device=device,
        cand_ids=cand_ids,
        shuffled_cot=shuffled_cot,
    )

    golds = [_gold_label(r) for r in rows_filtered]
    maj = Counter(g for g in golds if g is not None).most_common(1)[0][0]
    maj_acc = sum(1 for g in golds if g == maj) / max(1, len(golds))

    full_acc = float(full["accuracy"])
    q_acc = float(q_only["accuracy"])
    s_acc = float(shuffled["accuracy"])

    gaps = {
        "full_minus_majority": round(full_acc - maj_acc, 4),
        "full_minus_question_only": round(full_acc - q_acc, 4),
        "full_minus_shuffled_cot": round(full_acc - s_acc, 4),
    }

    criteria = {
        "enough_examples": full["n"] >= args.min_examples,
        "full_acc_ok": full_acc >= args.min_full_acc,
        "gap_vs_majority_ok": gaps["full_minus_majority"] >= args.min_gap_majority,
        "gap_vs_question_only_ok": gaps["full_minus_question_only"] >= args.min_gap_question,
        "gap_vs_shuffled_cot_ok": gaps["full_minus_shuffled_cot"] >= args.min_gap_shuffled,
    }

    report = {
        "paths": {
            "ckpt": str(args.ckpt),
            "vocab": str(args.vocab),
            "data": str(args.data),
            "out": str(args.out),
        },
        "config": {
            "device": str(device),
            "difficulty": args.difficulty,
            "max_examples": args.max,
            "seed": args.seed,
        },
        "dataset": {
            "raw_rows": len(rows_all),
            "usable_rows": full["n"],
            "majority_label": maj,
            "majority_acc": round(maj_acc, 4),
        },
        "candidate_tokens": {
            "reasoning": reasoning_tokens,
            "logic": logic_tokens,
            "observed_reasoning_answer_tokens": observed_top,
            "unknown_candidates": unknown_candidates,
        },
        "modes": {
            "full": full,
            "question_only": q_only,
            "shuffled_cot": shuffled,
        },
        "gaps": gaps,
        "thresholds": {
            "min_examples": args.min_examples,
            "min_full_acc": args.min_full_acc,
            "min_gap_majority": args.min_gap_majority,
            "min_gap_question": args.min_gap_question,
            "min_gap_shuffled": args.min_gap_shuffled,
        },
        "criteria": criteria,
        "decision": {
            "reasoning_proof_pass": all(criteria.values()),
            "failed": [k for k, v in criteria.items() if not v],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    print(json.dumps({"full": full["accuracy"], "question_only": q_only["accuracy"], "shuffled": shuffled["accuracy"]}, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
