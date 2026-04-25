"""Build a reasoning-verifier training dataset with hard negatives.

Input is tokenized reasoning JSONL (question_tokens/cot_tokens/answer_tokens). Output
is JSONL of binary examples with `input_tokens` and `label`:

- positive: original question + CoT + answer
- neg_question_only: question + answer (explicit shortcut negative)
- neg_shuffled_cot: same question/answer, CoT from another sample in same language
- neg_answer_flip: same question/CoT, answer tokens from opposite label in same language
- neg_shuffled_and_flip: both shuffled CoT and flipped answer
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    compose_reasoning_tokens,
    extract_token_segments,
    load_jsonl,
    record_difficulty,
    record_gold_label,
    record_lang,
    record_schema,
    write_jsonl,
)


def _sample_other_index(
    candidates: list[int],
    current: int,
    rng: random.Random,
    fallback: list[int],
) -> int:
    primary = [i for i in candidates if i != current]
    if primary:
        return rng.choice(primary)

    secondary = [i for i in fallback if i != current]
    if secondary:
        return rng.choice(secondary)

    return current


def _build_examples(
    rows: list[dict[str, Any]],
    *,
    view: str,
    seed: int,
    neg_question_only: int,
    neg_shuffled_cot: int,
    neg_answer_flip: int,
    neg_shuffled_and_flip: int,
    max_source_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)

    eligible: list[dict[str, Any]] = []
    by_lang_indices: dict[str, list[int]] = defaultdict(list)
    answer_pool: dict[tuple[str, str], list[list[str]]] = defaultdict(list)

    for rec in rows:
        if record_schema(rec) != "reasoning":
            continue
        lbl = record_gold_label(rec)
        if lbl not in {"yes", "no"}:
            continue
        q, cot, ans = extract_token_segments(rec, view=view)
        if not q or not ans:
            continue
        lang = record_lang(rec)
        ex = {
            "source_id": str(rec.get("id", f"src-{len(eligible)}")),
            "lang": lang,
            "difficulty": record_difficulty(rec),
            "label_name": lbl,
            "question_tokens": q,
            "cot_tokens": cot,
            "answer_tokens": ans,
        }
        eligible.append(ex)
        idx = len(eligible) - 1
        by_lang_indices[lang].append(idx)
        answer_pool[(lang, lbl)].append(ans)

    if max_source_examples > 0 and len(eligible) > max_source_examples:
        rng.shuffle(eligible)
        eligible = eligible[:max_source_examples]

        # Rebuild language/answer pools for truncated set.
        by_lang_indices = defaultdict(list)
        answer_pool = defaultdict(list)
        for i, ex in enumerate(eligible):
            by_lang_indices[ex["lang"]].append(i)
            answer_pool[(ex["lang"], ex["label_name"])].append(ex["answer_tokens"])

    out: list[dict[str, Any]] = []
    all_indices = list(range(len(eligible)))
    variant_counts = Counter()
    skipped_flip = 0

    for i, ex in enumerate(eligible):
        source_id = ex["source_id"]
        lang = ex["lang"]
        lbl = ex["label_name"]
        opp = "no" if lbl == "yes" else "yes"

        q = ex["question_tokens"]
        cot = ex["cot_tokens"]
        ans = ex["answer_tokens"]

        out.append(
            {
                "id": f"{source_id}:pos",
                "source_id": source_id,
                "lang": lang,
                "difficulty": ex["difficulty"],
                "answer_label": lbl,
                "variant": "positive",
                "label": 1,
                "input_tokens": compose_reasoning_tokens(q, cot, ans),
            }
        )
        variant_counts["positive"] += 1

        # Negative: question + answer only (no CoT).
        for k in range(max(0, neg_question_only)):
            out.append(
                {
                    "id": f"{source_id}:neg_qonly:{k}",
                    "source_id": source_id,
                    "lang": lang,
                    "difficulty": ex["difficulty"],
                    "answer_label": lbl,
                    "variant": "neg_question_only",
                    "label": 0,
                    "input_tokens": list(q) + list(ans),
                }
            )
            variant_counts["neg_question_only"] += 1

        # Negative: shuffled CoT.
        for k in range(max(0, neg_shuffled_cot)):
            j = _sample_other_index(by_lang_indices[lang], i, rng, all_indices)
            cot_neg = eligible[j]["cot_tokens"]
            out.append(
                {
                    "id": f"{source_id}:neg_shuf:{k}",
                    "source_id": source_id,
                    "lang": lang,
                    "difficulty": ex["difficulty"],
                    "answer_label": lbl,
                    "variant": "neg_shuffled_cot",
                    "label": 0,
                    "input_tokens": compose_reasoning_tokens(q, cot_neg, ans),
                }
            )
            variant_counts["neg_shuffled_cot"] += 1

        # Negative: flipped answer.
        opp_answers = answer_pool.get((lang, opp), [])
        if opp_answers:
            for k in range(max(0, neg_answer_flip)):
                ans_neg = rng.choice(opp_answers)
                out.append(
                    {
                        "id": f"{source_id}:neg_flip:{k}",
                        "source_id": source_id,
                        "lang": lang,
                        "difficulty": ex["difficulty"],
                        "answer_label": lbl,
                        "variant": "neg_answer_flip",
                        "label": 0,
                        "input_tokens": compose_reasoning_tokens(q, cot, ans_neg),
                    }
                )
                variant_counts["neg_answer_flip"] += 1
        else:
            skipped_flip += 1

        # Negative: shuffled CoT + flipped answer.
        if opp_answers:
            for k in range(max(0, neg_shuffled_and_flip)):
                j = _sample_other_index(by_lang_indices[lang], i, rng, all_indices)
                cot_neg = eligible[j]["cot_tokens"]
                ans_neg = rng.choice(opp_answers)
                out.append(
                    {
                        "id": f"{source_id}:neg_both:{k}",
                        "source_id": source_id,
                        "lang": lang,
                        "difficulty": ex["difficulty"],
                        "answer_label": lbl,
                        "variant": "neg_shuffled_and_flip",
                        "label": 0,
                        "input_tokens": compose_reasoning_tokens(q, cot_neg, ans_neg),
                    }
                )
                variant_counts["neg_shuffled_and_flip"] += 1

    summary = {
        "token_view": view,
        "source_examples": len(eligible),
        "dataset_examples": len(out),
        "variant_counts": dict(variant_counts),
        "skipped_answer_flip_due_to_missing_opposite_pool": skipped_flip,
        "source_by_lang": dict(Counter(ex["lang"] for ex in eligible)),
        "source_by_label": dict(Counter(ex["label_name"] for ex in eligible)),
    }
    return out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Input tokenized JSONL path")
    ap.add_argument("--out", dest="out_path", type=Path, required=True, help="Output verifier JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--view", type=str, default="reasoning", choices=["reasoning", "default"])
    ap.add_argument("--max-source-examples", type=int, default=0, help="Cap source examples; <=0 means all")
    ap.add_argument("--neg-question-only", type=int, default=1)
    ap.add_argument("--neg-shuffled-cot", type=int, default=1)
    ap.add_argument("--neg-answer-flip", type=int, default=1)
    ap.add_argument("--neg-shuffled-and-flip", type=int, default=1)
    args = ap.parse_args()

    rows = load_jsonl(args.in_path)
    examples, summary = _build_examples(
        rows,
        view=args.view,
        seed=args.seed,
        neg_question_only=args.neg_question_only,
        neg_shuffled_cot=args.neg_shuffled_cot,
        neg_answer_flip=args.neg_answer_flip,
        neg_shuffled_and_flip=args.neg_shuffled_and_flip,
        max_source_examples=args.max_source_examples,
    )

    write_jsonl(args.out_path, examples)

    print(summary)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
