"""Phase 2b eval harness — runs all four models over the held-out set and
emits the headline results table.

Usage:
    python -m training.bpe_baseline.eval_harness \\
        --cst-tiny  edge/artifacts/model_logic_tiny.onnx \\
        --cst-small edge/artifacts/model_logic_small.onnx \\
        --gpt2-scratch training/bpe_baseline/out/gpt2_scratch \\
        --gpt2-ft      training/bpe_baseline/out/gpt2_ft \\
        --holdout reasoning/eval/holdout \\
        --out docs/reasoning-experiments.results.json

Writes JSON with the full results matrix; a companion script turns it into
the Markdown table that lands in `docs/reasoning-experiments.md`.

This file is a scaffold. The `run_*` functions are stubs that return the
shape the writer expects; wire them to real inference once the models exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASKS = ["prop_logic", "syllogisms", "algebra"]
CONDITIONS = ["raw", "symbolic"]  # symbolic only meaningful for BPE models


def load_holdout(path: Path, task: str) -> list[dict]:
    records = []
    with (path / f"{task}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def run_cst_logic(model_path: Path, task: str, records: list[dict]) -> dict[str, Any]:
    """Run a CST-logic ONNX model. TODO: wire once model is exported."""
    return {
        "accuracy": None,
        "latency_ms": None,
        "backend": "onnx-cst-logic",
        "note": f"stub — wire inference against {model_path}",
        "n": len(records),
    }


def run_gpt2(model_dir: Path, task: str, records: list[dict], condition: str) -> dict[str, Any]:
    """Run a HF GPT-2 model in either raw or symbolic prompt mode. TODO."""
    return {
        "accuracy": None,
        "latency_ms": None,
        "backend": f"hf-gpt2-{condition}",
        "note": f"stub — wire inference against {model_dir}",
        "n": len(records),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cst-tiny", type=Path, required=False)
    ap.add_argument("--cst-small", type=Path, required=False)
    ap.add_argument("--gpt2-scratch", type=Path, required=False)
    ap.add_argument("--gpt2-ft", type=Path, required=False)
    ap.add_argument("--holdout", type=Path, default=Path("reasoning/eval/holdout"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results: dict[str, Any] = {"tasks": {}}
    for task in TASKS:
        records = load_holdout(args.holdout, task)
        task_row: dict[str, Any] = {}
        if args.cst_tiny:
            task_row["A_cst_tiny"] = run_cst_logic(args.cst_tiny, task, records)
        if args.cst_small:
            task_row["B_cst_small"] = run_cst_logic(args.cst_small, task, records)
        if args.gpt2_scratch:
            for cond in CONDITIONS:
                task_row[f"C_gpt2_scratch_{cond}"] = run_gpt2(
                    args.gpt2_scratch, task, records, cond
                )
        if args.gpt2_ft:
            for cond in CONDITIONS:
                task_row[f"D_gpt2_ft_{cond}"] = run_gpt2(
                    args.gpt2_ft, task, records, cond
                )
        results["tasks"][task] = task_row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
