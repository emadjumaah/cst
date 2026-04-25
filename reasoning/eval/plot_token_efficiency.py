"""
plot_token_efficiency.py — compare default (NL-ish) vs reasoning (CST) token
length per syllogism. Independent of any trained model.

Reads reasoning/tokenized/stage-2b-syllogisms.tokenized.jsonl and writes
docs/paper/figures/token_efficiency.png + summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reasoning" / "tokenized" / "stage-2b-syllogisms.tokenized.jsonl"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def record_length(rec: dict, view: str) -> int:
    q = (rec.get("question_tokens") or {}).get(view) or []
    cot_len = sum(len((step.get(view) or [])) for step in (rec.get("cot_tokens") or []))
    ans = (rec.get("answer_tokens") or {}).get(view) or []
    # count segment separators [BOS]/[EOS] the way the model sees them
    segs = 1 + len(rec.get("cot_tokens") or []) + 1
    return len(q) + cot_len + len(ans) + 2 * segs


def main() -> None:
    lens_default: list[int] = []
    lens_reason: list[int] = []
    with SRC.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lens_default.append(record_length(r, "default"))
            lens_reason.append(record_length(r, "reasoning"))

    d = np.array(lens_default)
    rz = np.array(lens_reason)
    summary = {
        "n": int(len(d)),
        "default": {
            "mean": float(d.mean()), "median": float(np.median(d)),
            "p90": float(np.percentile(d, 90)), "max": int(d.max()),
        },
        "reasoning": {
            "mean": float(rz.mean()), "median": float(np.median(rz)),
            "p90": float(np.percentile(rz, 90)), "max": int(rz.max()),
        },
        "compression_ratio_mean": float(d.mean() / rz.mean()),
    }
    (OUT_DIR / "token_efficiency.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    # Overlaid histogram
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    bins = np.arange(0, max(d.max(), rz.max()) + 5, 5)
    ax.hist(d, bins=bins, alpha=0.55, label=f"NL-tokenized (default): μ={d.mean():.1f}", color="#d29922")
    ax.hist(rz, bins=bins, alpha=0.7, label=f"CST-tokenized: μ={rz.mean():.1f}", color="#3fb950")
    ax.set_xlabel("tokens per syllogism example")
    ax.set_ylabel("examples")
    ax.set_title(f"CST tokenization uses {d.mean()/rz.mean():.2f}× fewer tokens")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_png = OUT_DIR / "token_efficiency.png"
    fig.savefig(out_png)
    print(f"wrote {out_png}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
