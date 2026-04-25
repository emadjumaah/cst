"""Shared record schema for reasoning-data JSONL output.

All sources and generators produce records conforming to this contract
(see ``reasoning/README.md``)::

    {
      "id": "...",
      "lang": "ar" | "en",
      "category": 1 | 2 | 3 | 4,
      "question": "...",
      "cot": ["...", ...],
      "answer": "...",
      "meta": {
        "source": "...",
        "license": "...",
        "translated_from": "en" | null,
        "translation_quality": 4.5 | null,
        "difficulty": "easy" | "medium" | "hard"
      }
    }
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Literal


Lang = Literal["ar", "en"]
Category = Literal[1, 2, 3, 4]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class Meta:
    source: str
    license: str
    translated_from: str | None = None
    translation_quality: float | None = None
    difficulty: Difficulty = "medium"
    # Optional pre-computed CST-token views (arabic-algebra-engine only).
    # When present, the tokenizer uses them verbatim for the reasoning
    # side instead of re-deriving tokens from NL prose.
    question_cst: list[str] | None = None
    cot_cst: list[list[str]] | None = None
    answer_cst: list[str] | None = None


@dataclass
class Record:
    id: str
    lang: Lang
    category: Category
    question: str
    answer: str
    meta: Meta
    cot: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        # Drop None-valued optional CST fields so non-algebra records
        # stay lean.
        meta = d.get("meta") or {}
        for k in ("question_cst", "cot_cst", "answer_cst",
                  "translated_from", "translation_quality"):
            if meta.get(k) is None:
                meta.pop(k, None)
        return json.dumps(d, ensure_ascii=False)


def write_jsonl(path: Path, records: Iterable[Record]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json())
            f.write("\n")
            n += 1
    return n
