"""DAgger with the scripted expert as the labelling oracle.

    python dagger.py --root data/lift_red --repo-id local/so_arm100_lift_red --task lift:red \
        --model image --iters 5 --rollouts 20 --beta0 0.5 --beta-decay 0.5

Iteration i:
  1. roll out `--rollouts` episodes executing the expert's action with
     probability beta_i = beta0 * decay^i and the current learner's otherwise
  2. at every visited state record the expert's action as the label
     (`shadow.shadow_rollout`), whatever was executed
  3. aggregate with everything so far and retrain -- from scratch by default,
     or `--warm` to continue from the current weights
  4. report the learner's success on a small evaluation (`--eval-n`)

The final learner is evaluated with the full bench protocol. The ordinary BC
policy trained on the same demonstrations is the comparison; DAgger's claim
is that visiting the learner's own states and getting labels there closes
the compounding-error gap BC has.
"""
import argparse
import json
import os
import time

import numpy as np

from so_arm100_sim.env import ArmEnv
from so_arm100_sim.evaluate import evaluate, save_json
from so_arm100_sim.expert import ScriptedExpert

from so_arm100_il.data import load_dataset
from so_arm100_il.policy import LearnedPolicy, save_checkpoint
from so_arm100_il.shadow import shadow_rollout
from so_arm100_il.training import train

IMAGE = {"camera": "front", "height": 96, "width": 128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", default="lift:red")
    ap.add_argument("--model", choices=["state", "image", "act"], default="image")
    ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--beta-decay", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--eval-n", type=int, default=20, help="episodes per iteration check")
    ap.add_argument("--final-n", type=int, default=100)
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/dagger.pt")
    ap.add_argument("--image-h", type=int, default=IMAGE["height"])
    ap.add_argument("--image-w", type=int, default=IMAGE["width"])
    a = ap.parse_args()

    image = {"camera": "front", "height": a.image_h, "width": a.image_w}
    env_kwargs = {"obs_mode": "proprio", "action_mode": "absolute"}
    if a.model != "state":
        env_kwargs["image"] = image
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()

    data = load_dataset(a.root, a.repo_id, n_episodes=a.n_episodes, with_images=a.model != "state")
    print(f"demos: {len(data)} episodes, {sum(len(e) for e in data)} frames")
    model, stats, _ = train(data, a.model, epochs=a.epochs, seed=a.seed, threads=a.threads)
    log = {"iters": [], "args": vars(a)}

    env = ArmEnv(a.task, seed=a.seed + 1, **env_kwargs)
    expert = ScriptedExpert(env)
    for i in range(a.iters):
        beta = a.beta0 * a.beta_decay ** i
        learner = LearnedPolicy(model, stats)
        new, succ = [], []
        for j in range(a.rollouts):
            ep, ok, _ = shadow_rollout(env, learner, beta, rng, seed=100_000 * (i + 1) + j,
                                       expert=expert)
            new.append(ep)
            succ.append(ok)
        data = data + new
        model, stats, _ = train(data, a.model, epochs=a.epochs, seed=a.seed + i + 1,
                                threads=a.threads, init_model=model if a.warm else None)
        r = evaluate(lambda e: LearnedPolicy(model, stats), a.task, n_episodes=a.eval_n,
                     seeds=(0,), env_kwargs=env_kwargs)
        row = {"iter": i, "beta": beta, "frames_total": sum(len(e) for e in data),
               "rollout_success_mixed": float(np.mean(succ)), "learner_success": r["success_rate"]}
        log["iters"].append(row)
        print(f"iter {i}: beta {beta:.2f}  data {row['frames_total']} frames  "
              f"mixed-rollout success {row['rollout_success_mixed']:.2f}  "
              f"learner success ({a.eval_n} eps) {r['success_rate']:.2f}  "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_checkpoint(a.out, model, stats, {"dagger": log})
    final = evaluate(lambda e: LearnedPolicy(model, stats), a.task, n_episodes=a.final_n,
                     seeds=tuple(range(a.final_seeds)), env_kwargs=env_kwargs)
    log["final"] = final
    log["wall_s"] = time.perf_counter() - t0
    save_json(os.path.splitext(a.out)[0] + ".json", log)
    print(f"final: success {final['success_rate']:.3f} +- {final['success_std']:.3f} "
          f"({a.final_n} x {a.final_seeds} seeds)  saved {a.out}")


if __name__ == "__main__":
    main()
