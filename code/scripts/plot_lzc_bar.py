#!/usr/bin/env python3
"""Generate LZc bar chart from cached results.

Usage: python plot_lzc_bar.py [FIGURES_DIR]
  FIGURES_DIR defaults to ../../figures/ relative to this script.
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) > 1:
        fig_dir = Path(sys.argv[1])
    else:
        fig_dir = Path(__file__).resolve().parent.parent.parent / "figures"

    cache_path = fig_dir / "lzc_results_cache.json"
    if not cache_path.exists():
        print(f"No lzc_results_cache.json found in {fig_dir}")
        sys.exit(1)

    with open(cache_path) as f:
        d = json.load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ["Awake", "Propofol", "Propofol+DOI", "DOI"]
    keys = ["awake", "propofol", "propofol_doi", "doi"]

    vals = [d.get(k, {}).get("mean_lzc", 0) for k in keys]
    stds = [d.get(k, {}).get("std_lzc", 0) for k in keys]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(conditions, vals, yerr=stds, color=colors, capsize=5, edgecolor="black")
    ax.set_ylabel("LZc")
    ax.set_title("Lempel-Ziv Complexity by Condition")
    ax.set_ylim(0, 1.05)
    if vals[0] > 0:
        ax.axhline(y=vals[0], color="gray", ls="--", alpha=0.5, label="Awake baseline")
        ax.legend()
    fig.tight_layout()

    out_path = fig_dir / "lzc_bar_chart.png"
    fig.savefig(out_path, dpi=150)
    print(f">>> Saved: {out_path}")


if __name__ == "__main__":
    main()
