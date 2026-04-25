"""Decide whether CST logic tokenizer shows reasoning advantage over baseline.

This script compares two verifier eval JSONs produced by `eval_signal.py`:
- CST condition (`--view reasoning`)
- baseline condition (`--view default`)

It emits a structured verdict with deltas on the exact reasoning-signal metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def _bool_get(d: dict[str, Any], key: str) -> bool:
    return bool(d.get(key, False))


def _check_fairness(cst_train: dict[str, Any] | None, base_train: dict[str, Any] | None) -> dict[str, Any]:
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    if cst_train is None or base_train is None:
        warnings.append("Train summaries were not both provided; fairness checks are partial.")
        return {"ok": False, "checks": checks, "warnings": warnings}

    c_cfg = (cst_train.get("config") or {})
    b_cfg = (base_train.get("config") or {})
    arch_keys = ["max_len", "d_model", "n_heads", "n_layers", "d_ff", "dropout"]

    checks["same_architecture"] = all(c_cfg.get(k) == b_cfg.get(k) for k in arch_keys)
    if not checks["same_architecture"]:
        warnings.append("Architecture mismatch across CST/baseline runs.")

    c_sizes = (cst_train.get("sizes") or {})
    b_sizes = (base_train.get("sizes") or {})
    checks["same_train_size"] = c_sizes.get("train") == b_sizes.get("train")
    checks["same_val_size"] = c_sizes.get("val") == b_sizes.get("val")
    checks["same_test_size"] = c_sizes.get("test") == b_sizes.get("test")

    for k in ("same_train_size", "same_val_size", "same_test_size"):
        if not checks[k]:
            warnings.append(f"Split size mismatch: {k}.")

    ok = all(checks.values()) if checks else False
    return {"ok": ok, "checks": checks, "warnings": warnings}


def decide(args: argparse.Namespace) -> dict[str, Any]:
    cst_eval = _load_json(args.cst_eval)
    base_eval = _load_json(args.baseline_eval)

    cst_train = _load_json(args.cst_train_summary) if args.cst_train_summary is not None else None
    base_train = _load_json(args.baseline_train_summary) if args.baseline_train_summary is not None else None

    cst_pass = _bool_get(cst_eval, "reasoning_proof_pass")
    base_pass = _bool_get(base_eval, "reasoning_proof_pass")

    cst_full = _safe_get(cst_eval, "means", "full")
    cst_gap_q = _safe_get(cst_eval, "gaps", "full_minus_question_only")
    cst_gap_s = _safe_get(cst_eval, "gaps", "full_minus_shuffled_cot")

    base_full = _safe_get(base_eval, "means", "full")
    base_gap_q = _safe_get(base_eval, "gaps", "full_minus_question_only")
    base_gap_s = _safe_get(base_eval, "gaps", "full_minus_shuffled_cot")

    deltas = {
        "delta_full_mean": round(cst_full - base_full, 4),
        "delta_gap_qonly": round(cst_gap_q - base_gap_q, 4),
        "delta_gap_shuffled": round(cst_gap_s - base_gap_s, 4),
    }

    stronger_than_baseline = (
        deltas["delta_full_mean"] >= args.min_delta_full
        and deltas["delta_gap_qonly"] >= args.min_delta_gap_qonly
        and deltas["delta_gap_shuffled"] >= args.min_delta_gap_shuffled
    )

    if cst_pass and not base_pass:
        verdict = "cst_advantage_supported"
    elif cst_pass and base_pass:
        verdict = "cst_advantage_supported" if stronger_than_baseline else "inconclusive_both_pass"
    elif not cst_pass and base_pass:
        verdict = "cst_disadvantage_baseline_wins"
    else:
        verdict = "cst_better_but_not_reasoning_ready" if stronger_than_baseline else "no_evidence_of_cst_advantage"

    claim_supported = verdict == "cst_advantage_supported"
    claim_refuted = verdict == "cst_disadvantage_baseline_wins"

    fairness = _check_fairness(cst_train, base_train)

    report = {
        "inputs": {
            "cst_eval": str(args.cst_eval),
            "baseline_eval": str(args.baseline_eval),
            "cst_train_summary": str(args.cst_train_summary) if args.cst_train_summary else None,
            "baseline_train_summary": str(args.baseline_train_summary) if args.baseline_train_summary else None,
        },
        "thresholds": {
            "min_delta_full": args.min_delta_full,
            "min_delta_gap_qonly": args.min_delta_gap_qonly,
            "min_delta_gap_shuffled": args.min_delta_gap_shuffled,
        },
        "fairness": fairness,
        "cst": {
            "reasoning_proof_pass": cst_pass,
            "means": {
                "full": round(cst_full, 4),
            },
            "gaps": {
                "full_minus_question_only": round(cst_gap_q, 4),
                "full_minus_shuffled_cot": round(cst_gap_s, 4),
            },
        },
        "baseline": {
            "reasoning_proof_pass": base_pass,
            "means": {
                "full": round(base_full, 4),
            },
            "gaps": {
                "full_minus_question_only": round(base_gap_q, 4),
                "full_minus_shuffled_cot": round(base_gap_s, 4),
            },
        },
        "deltas": deltas,
        "decision": {
            "verdict": verdict,
            "claim_supported": claim_supported,
            "claim_refuted": claim_refuted,
        },
    }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cst-eval", type=Path, required=True)
    ap.add_argument("--baseline-eval", type=Path, required=True)
    ap.add_argument("--cst-train-summary", type=Path, default=None)
    ap.add_argument("--baseline-train-summary", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)

    ap.add_argument("--min-delta-full", type=float, default=0.01)
    ap.add_argument("--min-delta-gap-qonly", type=float, default=0.01)
    ap.add_argument("--min-delta-gap-shuffled", type=float, default=0.01)
    return ap.parse_args()


if __name__ == "__main__":
    decide(parse_args())
