"""
plot_results.py — Pareto frontier + per-category bars from all *.eval.json
files under reasoning/runs/. Run `run_onnx_eval.py` first for each model.

Outputs docs/paper/figures/pareto.png and per_category.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "reasoning" / "runs"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_summaries() -> list[dict]:
    out: list[dict] = []
    for p in sorted(RUNS.rglob("*.eval.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        s["_run"] = p.parent.name
        s["_name"] = p.stem.replace(".eval", "")
        out.append(s)
    return out


def plot_pareto(summaries: list[dict]) -> None:
    if not summaries:
        print("no summaries yet")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    for s in summaries:
        label = f"{s['_run']} / {s['_name']}"
        ax.scatter(s["size_mb"], s["accuracy"] * 100, s=80,
                   color="#58a6ff" if "int8" in s["_name"] else "#d29922",
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(label, (s["size_mb"], s["accuracy"] * 100),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("model size (MB, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("CST reasoning: size × accuracy")
    ax.grid(True, alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "pareto.png"
    fig.savefig(out); print(f"wrote {out}")


def plot_per_category(summaries: list[dict]) -> None:
    if not summaries:
        return
    # only INT8 variants, one bar-group per model × (lang | validity | difficulty)
    int8 = [s for s in summaries if "int8" in s["_name"]]
    if not int8:
        int8 = summaries
    import numpy as np
    labels = []
    for k in ("by_lang", "by_validity", "by_difficulty"):
        for kk in sorted(set().union(*(s[k].keys() for s in int8))):
            labels.append((k, kk))
    ncols = len(labels)
    nmod = len(int8)
    width = 0.8 / nmod
    x = np.arange(ncols)
    fig, ax = plt.subplots(figsize=(max(6, ncols * 0.9), 4.5), dpi=140)
    colors = ["#3fb950", "#58a6ff", "#d29922", "#f85149"]
    for i, s in enumerate(int8):
        vals = []
        for k, kk in labels:
            d = s[k].get(kk, {})
            vals.append(d.get("acc", 0) * 100 if d else 0)
        ax.bar(x + i * width - (nmod - 1) * width / 2, vals, width=width,
               label=f"{s['_run']} ({s['size_mb']}MB)", color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{kk}" for _, kk in labels], rotation=30, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("CST reasoning: per-category accuracy (INT8)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "per_category.png"
    fig.savefig(out); print(f"wrote {out}")


def main() -> None:
    summaries = load_summaries()
    print(f"loaded {len(summaries)} summaries")
    for s in summaries:
        print(f"  {s['_run']}/{s['_name']}: {s['size_mb']} MB  acc={s['accuracy']*100:.2f}%  n={s['n']}")
    plot_pareto(summaries)
    plot_per_category(summaries)


if __name__ == "__main__":
    main()
