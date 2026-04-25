"""Propositional-logic corpus generator (Category 2).

Generates truth-preserving logical reasoning items in Arabic and English
with deterministic CoT traces. No external data required.

Format::

    question: "If p and q then r. Given p=T, q=T. Is r?"
    cot:      ["p ∧ q evaluates to T ∧ T", "T ∧ T = T", "From (p ∧ q → r) and T, r = T"]
    answer:   "yes"

Difficulty tiers
----------------
* **easy**   — 1 connective, 2 variables, no negation
* **medium** — 2 connectives, up to 3 variables, optional negation
* **hard**   — 3 connectives, 3–4 variables, nested implication

Run::

    python -m reasoning.data.generators.prop_logic --count 10000 --out out/prop_logic.jsonl
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..schema import Meta, Record, write_jsonl


# ── Language strings ──────────────────────────────────────────────

CONN_EN = {"and": "and", "or": "or", "not": "not", "implies": "implies"}
CONN_AR = {"and": "و", "or": "أو", "not": "ليس", "implies": "يستلزم"}


@dataclass
class Formula:
    """Minimal propositional formula AST."""

    op: str  # "var" | "not" | "and" | "or" | "implies"
    left: "Formula | str | None" = None
    right: "Formula | None" = None

    def eval(self, env: dict[str, bool]) -> bool:
        if self.op == "var":
            return env[self.left]  # type: ignore[index]
        if self.op == "not":
            return not self.left.eval(env)  # type: ignore[union-attr]
        l = self.left.eval(env)  # type: ignore[union-attr]
        r = self.right.eval(env)  # type: ignore[union-attr]
        if self.op == "and":
            return l and r
        if self.op == "or":
            return l or r
        if self.op == "implies":
            return (not l) or r
        raise ValueError(self.op)

    def render(self, conn: dict[str, str]) -> str:
        if self.op == "var":
            return str(self.left)
        if self.op == "not":
            return f"{conn['not']} {self.left.render(conn)}"  # type: ignore[union-attr]
        l = self.left.render(conn)  # type: ignore[union-attr]
        r = self.right.render(conn)  # type: ignore[union-attr]
        return f"({l} {conn[self.op]} {r})"


# ── Generation ───────────────────────────────────────────────────

def _var(name: str) -> Formula:
    return Formula("var", name)


def _make(rng: random.Random, depth: int, vars_: list[str]) -> Formula:
    if depth == 0 or rng.random() < 0.25:
        return _var(rng.choice(vars_))
    op = rng.choice(["and", "or", "implies", "not"])
    if op == "not":
        return Formula("not", _make(rng, depth - 1, vars_))
    return Formula(op, _make(rng, depth - 1, vars_), _make(rng, depth - 1, vars_))


def _make_top(rng: random.Random, depth: int, vars_: list[str]) -> Formula:
    """Top-level formula builder.

    Boosts the rate of an outer-``not`` head to ~25% so that the trained
    model actually sees long chains ending in ``not X = Y``. The
    unbiased sampler puts ``not`` at the root only ~6% of the time,
    which is not enough to reliably close nested negations.
    """
    if depth >= 2 and rng.random() < 0.25:
        return Formula("not", _make(rng, depth - 1, vars_))
    return _make(rng, depth, vars_)


def _difficulty_params(level: str) -> tuple[int, int]:
    """Return (num_vars, depth) for a difficulty tier."""
    return {"easy": (2, 1), "medium": (3, 2), "hard": (4, 3)}[level]


def _assignments(vars_: list[str]) -> list[dict[str, bool]]:
    n = len(vars_)
    out: list[dict[str, bool]] = []
    for mask in range(2 ** n):
        out.append({v: bool((mask >> i) & 1) for i, v in enumerate(vars_)})
    return out


def _cot_trace(f: Formula, env: dict[str, bool], conn: dict[str, str]) -> list[str]:
    """Produce a bottom-up evaluation trace."""
    steps: list[str] = []

    def walk(node: Formula) -> bool:
        if node.op == "var":
            v = env[node.left]  # type: ignore[index]
            steps.append(f"{node.left} = {_bool(v, conn)}")
            return v
        if node.op == "not":
            inner = walk(node.left)  # type: ignore[arg-type]
            result = not inner
            steps.append(
                f"{conn['not']} {_bool(inner, conn)} = {_bool(result, conn)}"
            )
            return result
        l = walk(node.left)  # type: ignore[arg-type]
        r = walk(node.right)  # type: ignore[arg-type]
        if node.op == "and":
            result = l and r
        elif node.op == "or":
            result = l or r
        else:
            result = (not l) or r
        steps.append(
            f"{_bool(l, conn)} {conn[node.op]} {_bool(r, conn)} = {_bool(result, conn)}"
        )
        return result

    walk(f)
    return steps


def _bool(b: bool, conn: dict[str, str]) -> str:
    # English "true/false" vs Arabic "صحيح/خاطئ".
    if conn is CONN_EN:
        return "true" if b else "false"
    return "صحيح" if b else "خاطئ"


# ── Record builders ──────────────────────────────────────────────

def _build_record(
    *,
    idx: int,
    lang: str,
    formula: Formula,
    env: dict[str, bool],
    difficulty: str,
) -> Record:
    conn = CONN_EN if lang == "en" else CONN_AR
    q_given = ", ".join(f"{k}={_bool(v, conn)}" for k, v in env.items())
    expr = formula.render(conn)
    if lang == "en":
        question = f"Given {q_given}, evaluate: {expr}"
    else:
        question = f"بمعطيات {q_given}، احسب قيمة: {expr}"
    cot = _cot_trace(formula, env, conn)
    answer = _bool(formula.eval(env), conn)
    return Record(
        id=f"prop-{lang}-{idx:06d}",
        lang=lang,  # type: ignore[arg-type]
        category=2,
        question=question,
        answer=answer,
        cot=cot,
        meta=Meta(
            source="prop-logic",
            license="cc0-1.0",
            difficulty=difficulty,  # type: ignore[arg-type]
        ),
    )


def generate(
    count: int,
    *,
    seed: int = 42,
    difficulty_mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> Iterable[Record]:
    """Yield ``count`` records per language (so ``2 * count`` total)."""
    rng = random.Random(seed)
    difficulties = ["easy", "medium", "hard"]
    idx = 0
    for _ in range(count):
        level = rng.choices(difficulties, weights=difficulty_mix, k=1)[0]
        n_vars, depth = _difficulty_params(level)
        vars_ = ["p", "q", "r", "s"][:n_vars]
        formula = _make_top(rng, depth, vars_)
        env = rng.choice(_assignments(vars_))
        for lang in ("en", "ar"):
            yield _build_record(
                idx=idx, lang=lang, formula=formula, env=env, difficulty=level
            )
        idx += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = write_jsonl(args.out, generate(args.count, seed=args.seed))
    print(f"Wrote {n:,} records to {args.out}")


if __name__ == "__main__":
    main()
