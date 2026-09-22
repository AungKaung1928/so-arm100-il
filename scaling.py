"""Success against the number of demonstration episodes.

    python scaling.py --root data/lift_red --repo-id local/so_arm100_lift_red --task lift:red \
        --model image --sizes 10 25 50 100 200 --seeds 3 --eval-n 50

For each size and seed: train on the first `size` episodes (the same first
episodes for every seed, so seeds differ only in initialisation and
shuffling), evaluate with the bench protocol on nominal physics, record the
success rate. The figure is mean +- std over seeds against episodes on a
log x-axis. This is the result the project is for: how many demonstrations
each model needs before its success stops improving, and whether ACT's
chunking changes that slope.
"""
import argparse
import json
import os
import time

import numpy as np

from so_arm100_sim.evaluate import evaluate, provenance

from so_arm100_il.data import load_dataset
from so_arm100_il.policy import LearnedPolicy
from so_arm100_il.training import train


def run(episodes, kind, sizes, seeds, task, eval_n, eval_seeds, epochs, env_kwargs, threads, log=print):
    rows = []
    for size in sizes:
        subset = episodes[:size]
        if len(subset) < size:
            log(f"  only {len(subset)} episodes available; size {size} skipped")
            continue
        for seed in range(seeds):
            t0 = time.perf_counter()
            model, stats, hist = train(subset, kind, epochs=epochs, seed=seed, threads=threads,
                                       log=lambda *_: None)
            r = evaluate(lambda e: LearnedPolicy(model, stats), task, n_episodes=eval_n,
                         seeds=tuple(range(eval_seeds)), env_kwargs=env_kwargs)
            rows.append({"size": size, "seed": seed, "success": r["success_rate"],
                         "val_l1": hist[-1]["val_l1"], "train_l1": hist[-1]["train_l1"],
                         "wall_s": time.perf_counter() - t0})
            log(f"  size {size:4d} seed {seed}  success {r['success_rate']:.2f}  "
                f"val L1 {hist[-1]['val_l1']:.4f}  {rows[-1]['wall_s']:.0f} s")
    return rows


def summarise(rows):
    out = {}
    for r in rows:
        out.setdefault(r["size"], []).append(r["success"])
    return [{"size": s, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
             "n": len(v)} for s, v in sorted(out.items())]


def plot(summary, kind, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [s["size"] for s in summary]
    y = [s["mean"] for s in summary]
    e = [s["std"] for s in summary]
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=130)
    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, color="#4f7d93")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(0, 1)
    ax.set_xlabel("demonstration episodes")
    ax.set_ylabel("success rate")
    ax.set_title(f"{kind}: success vs demonstrations")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", default="lift:red")
    ap.add_argument("--model", choices=["state", "image", "act"], default="image")
    ap.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50, 100, 200])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--eval-seeds", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--image-h", type=int, default=96)
    ap.add_argument("--image-w", type=int, default=128)
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true", help="2 sizes, 1 seed, 1 epoch, 1 eval episode")
    a = ap.parse_args()
    if a.smoke:
        a.sizes, a.seeds, a.epochs, a.eval_n, a.eval_seeds = a.sizes[:2], 1, 1, 1, 1

    env_kwargs = {"obs_mode": "proprio", "action_mode": "absolute"}
    if a.model != "state":
        env_kwargs["image"] = {"camera": "front", "height": a.image_h, "width": a.image_w}
    eps = load_dataset(a.root, a.repo_id, n_episodes=max(a.sizes), with_images=a.model != "state")
    print(f"{len(eps)} episodes loaded; model {a.model}; sizes {a.sizes}; {a.seeds} seeds")
    rows = run(eps, a.model, a.sizes, a.seeds, a.task, a.eval_n, a.eval_seeds, a.epochs,
               env_kwargs, a.threads)
    summary = summarise(rows)
    out = a.out or f"runs/scaling_{a.model}{'_smoke' if a.smoke else ''}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(a), "rows": rows, "summary": summary, **provenance()}, f, indent=2)
    fig = plot(summary, a.model, f"out/scaling_{a.model}{'_smoke' if a.smoke else ''}.png")
    print("\n| episodes | success (mean +- std over seeds) |\n|---|---|")
    for s in summary:
        print(f"| {s['size']} | {s['mean']:.2f} +- {s['std']:.2f} (n={s['n']}) |")
    print(f"\nwrote {out} and {fig}")


if __name__ == "__main__":
    main()
