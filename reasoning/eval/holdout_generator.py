"""Held-out eval generator for Phase 2b head-to-head comparison.

Produces evaluation problems with DIFFERENT surface forms from the training
generator in `reasoning/out/`. Purpose: control for memorization so the
four-model comparison (CST-logic tiny/small vs. GPT-2 from-scratch/fine-tuned)
measures reasoning, not pattern matching.

Key differences from the training generator:
- Variable names: uses {a,b,c,x,y,z,m,n} instead of {p,q,r,s}.
- Prop-logic surface: uses "&", "|", "~", "->" instead of "and/or/not/implies".
- Syllogism surface: "every/no/some X is Y" instead of "all/none/some".
- Algebra surface: LaTeX-ish forms (e.g. `solve: 3\\cdot x = 9`).
- Depth range: 5-7 for prop (training is 1-4) to test generalization.

Output: one JSONL file per task family under `reasoning/eval/holdout/`.
Same schema as `reasoning/out/` so the eval harness is uniform.

Usage:
    python -m reasoning.eval.holdout_generator --n 1000 --out reasoning/eval/holdout
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Callable, Iterable

_VAR_POOL = ["a", "b", "c", "x", "y", "z", "m", "n"]
_OPS_SYM = {"and": "&", "or": "|", "not": "~", "implies": "->"}


# ─── prop-logic ─────────────────────────────────────────────────────────────
def _build_prop(depth: int, rng: random.Random) -> tuple[str, bool, list[str]]:
    """Return (expr_str, truth_value, cot_steps) for a formula of given depth."""
    nvars = rng.randint(2, 4)
    variables = rng.sample(_VAR_POOL, nvars)
    env = {v: rng.choice([True, False]) for v in variables}

    def gen(d: int) -> tuple[str, bool, list[str]]:
        if d == 0:
            v = rng.choice(variables)
            return v, env[v], [f"{v} = {env[v]}"]
        op = rng.choice(["and", "or", "implies", "not"])
        if op == "not":
            e, val, steps = gen(d - 1)
            return f"~{e}", not val, steps + [f"~{val} = {not val}"]
        l, lv, ls = gen(d - 1)
        r, rv, rs = gen(d - 1)
        if op == "and":
            v = lv and rv
        elif op == "or":
            v = lv or rv
        else:  # implies
            v = (not lv) or rv
        sym = _OPS_SYM[op]
        return f"({l} {sym} {r})", v, ls + rs + [f"{lv} {sym} {rv} = {v}"]

    expr, val, steps = gen(depth)
    assigns = ", ".join(f"{k}={'T' if v else 'F'}" for k, v in env.items())
    question = f"Given {assigns}; evaluate: {expr}"
    return question, val, steps


def gen_prop(n: int, rng: random.Random) -> Iterable[dict]:
    for i in range(n):
        depth = rng.randint(5, 7)  # harder than training (1-4)
        q, ans, cot = _build_prop(depth, rng)
        yield {
            "id": f"prop-holdout-{i:06d}",
            "lang": "en",
            "category": 2,
            "question": q,
            "answer": "true" if ans else "false",
            "meta": {"source": "holdout", "depth": depth},
            "cot": cot,
        }


# ─── syllogisms ─────────────────────────────────────────────────────────────
# Enumerate all 256 mood×figure and label by brute-force Venn over 7 regions.
_FORMS = ["A", "E", "I", "O"]  # all / no / some / some-not
_FORM_SURFACE = {
    "A": ("every", "is"),
    "E": ("no", "is"),
    "I": ("some", "is"),
    "O": ("some", "is not"),
}


def _eval_syllogism(maj: str, min_: str, conc: str) -> bool:
    """3-set Venn brute force. True iff valid."""
    for regions in itertools.product([0, 1], repeat=7):
        # regions: S-only, M-only, P-only, SM, SP, MP, SMP (Venn layout)
        S = {0, 3, 4, 6}
        M = {1, 3, 5, 6}
        P = {2, 4, 5, 6}

        def satisfies(form: str, A: set[int], B: set[int]) -> bool:
            A_pop = {r for r in A if regions[r]}
            B_pop = {r for r in B if regions[r]}
            if form == "A":
                return A_pop.issubset(B_pop)
            if form == "E":
                return A_pop.isdisjoint(B_pop)
            if form == "I":
                return bool(A_pop & B_pop)
            return bool(A_pop - B_pop)  # O

        prem1 = satisfies(maj, M, P)
        prem2 = satisfies(min_, S, M)
        if prem1 and prem2 and not satisfies(conc, S, P):
            return False
    return True


_VALID_CACHE: dict[tuple, bool] = {}


def _is_valid(maj: str, min_: str, conc: str) -> bool:
    k = (maj, min_, conc)
    if k not in _VALID_CACHE:
        _VALID_CACHE[k] = _eval_syllogism(maj, min_, conc)
    return _VALID_CACHE[k]


def gen_syllogism(n: int, rng: random.Random) -> Iterable[dict]:
    terms = [
        ("mammals", "animals", "dogs"),
        ("birds", "creatures", "sparrows"),
        ("metals", "elements", "iron"),
        ("triangles", "shapes", "right-triangles"),
        ("planets", "celestial bodies", "mars"),
    ]
    for i in range(n):
        maj, min_, conc = [rng.choice(_FORMS) for _ in range(3)]
        S, P, M = rng.choice(terms)  # S=minor, P=major, M=middle

        def surface(form: str, A: str, B: str) -> str:
            q, cop = _FORM_SURFACE[form]
            return f"{q} {A} {cop} {B}"

        p1 = surface(maj, M, P)
        p2 = surface(min_, S, M)
        c = surface(conc, S, P)
        q = f"Premise 1: {p1}. Premise 2: {p2}. Conclusion: {c}. Is the argument valid?"
        ans = "valid" if _is_valid(maj, min_, conc) else "invalid"
        yield {
            "id": f"syllogism-holdout-{i:06d}",
            "lang": "en",
            "category": 2,
            "question": q,
            "answer": ans,
            "meta": {"source": "holdout", "mood": maj + min_ + conc},
            "cot": [p1, p2, f"therefore: {c}", ans],
        }


# ─── algebra ────────────────────────────────────────────────────────────────
def gen_algebra(n: int, rng: random.Random) -> Iterable[dict]:
    tasks = ["solve", "simplify", "evaluate"]
    for i in range(n):
        kind = rng.choice(tasks)
        if kind == "solve":
            a = rng.randint(2, 9)
            b = rng.randint(-20, 20)
            x = rng.randint(-10, 10)
            c = a * x + b
            sign = "+" if b >= 0 else "-"
            q = f"solve: {a}\\cdot x {sign} {abs(b)} = {c}"
            ans = str(x)
            cot = [f"{a}x = {c - b}", f"x = {x}"]
        elif kind == "simplify":
            a = rng.randint(1, 5)
            b = rng.randint(1, 5)
            c = rng.randint(-10, 10)
            d = rng.randint(-10, 10)
            q = f"simplify: ({a}x {'+' if c >= 0 else '-'} {abs(c)}) + ({b}x {'+' if d >= 0 else '-'} {abs(d)})"
            xc = a + b
            k = c + d
            ans = f"{xc}x{'+' if k >= 0 else '-'}{abs(k)}" if k != 0 else f"{xc}x"
            cot = [f"x-coef: {a}+{b}={xc}", f"const: {c}+{d}={k}", ans]
        else:  # evaluate
            a = rng.randint(2, 20)
            b = rng.randint(2, 20)
            op = rng.choice(["+", "-", "*"])
            q = f"evaluate: {a} {op} {b}"
            if op == "+":
                v = a + b
            elif op == "-":
                v = a - b
            else:
                v = a * b
            ans = str(v)
            cot = [f"{a} {op} {b} = {v}"]
        yield {
            "id": f"algebra-holdout-{i:06d}",
            "lang": "en",
            "category": 2,
            "question": q,
            "answer": ans,
            "meta": {"source": "holdout", "kind": kind},
            "cot": cot,
        }


# ─── driver ─────────────────────────────────────────────────────────────────
_FAMILIES: dict[str, Callable[[int, random.Random], Iterable[dict]]] = {
    "prop_logic": gen_prop,
    "syllogisms": gen_syllogism,
    "algebra": gen_algebra,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000, help="problems per family")
    ap.add_argument("--out", type=Path, default=Path("reasoning/eval/holdout"))
    ap.add_argument("--seed", type=int, default=20260423)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name, fn in _FAMILIES.items():
        rng = random.Random(args.seed + hash(name) % 1000)
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in fn(args.n, rng):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {args.n} records to {path}")


if __name__ == "__main__":
    main()
