"""All three scaling curves on one figure, from runs/scaling_{state,image,act}.json.

    python scripts/plot_scaling.py --out out/scaling_all.png

Mean +- std over training seeds, with every seed's own point drawn faintly,
because with three seeds the spread is the result as much as the mean is.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("state", "#9a9a9a"), ("image", "#4f7d93"), ("act", "#c0643a")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="out/scaling_all.png")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=130)
    for i, (m, c) in enumerate(MODELS):
        d = json.load(open(os.path.join(a.runs, f"scaling_{m}.json")))
        x = [s["size"] for s in d["summary"]]
        dx = 1 + 0.04 * (i - 1)
        ax.errorbar([v * dx for v in x], [s["mean"] for s in d["summary"]], yerr=[s["std"] for s in d["summary"]],
                    marker="o", capsize=3, color=c, label=f"{m} (n={d['summary'][0]['n']} seeds)")
        ax.scatter([r["size"] * dx for r in d["rows"]], [r["success"] for r in d["rows"]], s=10, color=c, alpha=0.35)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("demonstration episodes")
    ax.set_ylabel(f"success, lift:red ({d['args']['eval_n']} episodes)")
    ax.set_title("success vs demonstrations, 20 epochs")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
