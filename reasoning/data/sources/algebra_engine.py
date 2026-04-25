"""Ingest reasoning traces from arabic-algebra-engine (Category 2).

The engine lives in a sibling workspace folder::

    /Users/emad/projects/arabic-algebra/arabic-algebra-engine/

It emits a bilingual CoT JSONL via its ``training/generate-reasoning-cot.ts``
script (to be added in that repo). This module **reads** that JSONL and
converts it to our shared ``Record`` schema.

Usage::

    # Step 1 (in the engine repo): produce the raw CoT JSONL
    cd /Users/emad/projects/arabic-algebra/arabic-algebra-engine
    npx tsx src/training/generate-reasoning-cot.ts --count 50000 \
        --out data/corpus/reasoning-cot.jsonl

    # Step 2 (here): ingest it
    python -m reasoning.data.sources.algebra_engine \
        --in /Users/emad/projects/arabic-algebra/arabic-algebra-engine/data/corpus/reasoning-cot.jsonl \
        --out out/algebra_engine.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


def _convert(path: Path) -> Iterable[Record]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Engine output shape (see generate-reasoning-cot.ts):
            #   { id, lang, question, cot[], answer, difficulty, meta }
            raw_meta = raw.get("meta") or {}
            yield Record(
                id=f"algebra-{raw['id']}",
                lang=raw["lang"],
                category=2,
                question=raw["question"],
                answer=raw["answer"],
                cot=list(raw.get("cot", [])),
                meta=Meta(
                    source="algebra-engine",
                    license="isc",  # engine's LICENSE
                    difficulty=raw.get("difficulty", "medium"),
                    question_cst=raw_meta.get("question_cst"),
                    cot_cst=raw_meta.get("cot_cst"),
                    answer_cst=raw_meta.get("answer_cst"),
                ),
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Path to engine's reasoning-cot.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = write_jsonl(args.out, _convert(args.inp))
    print(f"Wrote {n:,} algebra-engine records to {args.out}")


if __name__ == "__main__":
    main()
