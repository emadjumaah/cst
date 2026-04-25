"""
build_demo_assets.py — copy ONNX + vocab + curated held-out examples into
edge/demo-reasoning/public/ so the browser demo is a single `npm run dev` away.

Samples ~25 examples per (lang × category) combination from the tokenized
syllogism JSONL, preserving the exact prefix format the model was trained on.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_VOCAB = ROOT / "reasoning" / "tokenized" / "vocab-reasoning.json"
SRC_JSONL = ROOT / "reasoning" / "tokenized" / "stage-2b-syllogisms.tokenized.jsonl"
SRC_SMALL = ROOT / "reasoning" / "runs" / "colab-cst-v0.2-small" / "model_logic_int8.onnx"
SRC_LARGE = ROOT / "reasoning" / "runs" / "colab-cst-v0.2" / "model_logic_int8.onnx"

DEST = ROOT / "edge" / "demo-reasoning" / "public"
DEST.mkdir(parents=True, exist_ok=True)

PER_BUCKET = 25
SEED = 42


def build_prefix_tokens(rec: dict) -> list[str]:
    q = (rec.get("question_tokens") or {}).get("reasoning") or []
    cot = rec.get("cot_tokens") or []
    toks: list[str] = ["[BOS]", *q, "[EOS]"]
    for step in cot:
        s = step.get("reasoning") or []
        if s:
            toks += ["[BOS]", *s, "[EOS]"]
    return toks


def main() -> None:
    rng = random.Random(SEED)

    # 1. Copy static assets
    shutil.copy2(SRC_VOCAB, DEST / "vocab-reasoning.json")
    shutil.copy2(SRC_SMALL, DEST / "model_logic_small.onnx")
    shutil.copy2(SRC_LARGE, DEST / "model_logic_large.onnx")
    print(f"copied vocab + 2 ONNX → {DEST}")
    for f in ("model_logic_small.onnx", "model_logic_large.onnx"):
        size_mb = (DEST / f).stat().st_size / 1e6
        print(f"  {f}: {size_mb:.2f} MB")

    # 2. Bucketed sample of held-out syllogisms
    bucket: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with SRC_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lang = r.get("lang", "en")
            diff = (r.get("meta") or {}).get("difficulty", "?")
            ans = str(r.get("answer", "")).strip()
            gold = "yes" if ans in {"yes", "نعم"} else "no" if ans in {"no", "لا"} else None
            if gold is None:
                continue
            bucket[(lang, diff, gold)].append(r)

    out: list[dict] = []
    for key, recs in bucket.items():
        rng.shuffle(recs)
        for r in recs[:PER_BUCKET]:
            diff = (r.get("meta") or {}).get("difficulty", "?")
            out.append({
                "id": r.get("id", ""),
                "lang": r.get("lang", "en"),
                "category": diff,
                "question": r.get("question", ""),
                "answer": "yes" if str(r.get("answer", "")).strip() in {"yes", "نعم"} else "no",
                "prefix_tokens": build_prefix_tokens(r),
            })
    rng.shuffle(out)
    (DEST / "examples.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8",
    )
    kb = (DEST / "examples.json").stat().st_size / 1024
    print(f"wrote {len(out)} examples → {DEST / 'examples.json'} ({kb:.1f} KB)")

    # distribution summary
    from collections import Counter
    c_lang = Counter(r["lang"] for r in out)
    c_cat = Counter(r["category"] for r in out)
    c_ans = Counter(r["answer"] for r in out)
    print(f"  by lang: {dict(c_lang)}")
    print(f"  by ans:  {dict(c_ans)}")
    print(f"  by cat:  {dict(c_cat)}")


if __name__ == "__main__":
    main()
