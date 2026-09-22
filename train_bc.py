"""Behaviour cloning on a recorded LeRobotDataset.

    python train_bc.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --model image --epochs 30 --out runs/bc_image_lift.pt

`--n-episodes N` trains on the first N episodes only (the scaling curve uses
it). The final epoch is saved; nothing is selected on validation.
"""
import argparse
import json
import os
import time

import torch

from so_arm100_il.data import load_dataset
from so_arm100_il.policy import save_checkpoint
from so_arm100_il.training import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--model", choices=["state", "image", "act"], default="image")
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=20, help="ACT action chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/bc.pt")
    a = ap.parse_args()

    t0 = time.perf_counter()
    eps = load_dataset(a.root, a.repo_id, n_episodes=a.n_episodes,
                       with_images=a.model != "state")
    print(f"loaded {len(eps)} episodes, {sum(len(e) for e in eps)} frames "
          f"in {time.perf_counter() - t0:.0f} s")
    model, stats, hist = train(eps, a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                               seed=a.seed, chunk=a.chunk, threads=a.threads)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cfg = {**vars(a), "n_episodes_used": len(eps), "history": hist,
           "wall_s": time.perf_counter() - t0, "torch": torch.__version__}
    save_checkpoint(a.out, model, stats, cfg)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved {a.out}  final train L1 {hist[-1]['train_l1']:.4f}  val L1 {hist[-1]['val_l1']:.4f}")


if __name__ == "__main__":
    main()
