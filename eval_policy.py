"""Evaluate a checkpoint with the bench protocol on nominal and held-out physics.

    python eval_policy.py --ckpt runs/bc_image_lift.pt --task lift:red \
        --physics nominal heavy slippery weak laggy noisy small --out runs/eval_bc_image_lift.json

The work is split into (physics cell, seed) units of `--episodes` episodes
each. `--jobs 4` runs units in parallel processes; every unit is seeded on
its own (episode j of seed s is reset with `s * 10_007 + j`, as in the bench),
so the numbers are the same as a serial run. Each finished unit is written
to `<out>.parts/` straight away and skipped on a rerun, so a killed run
resumes and loses at most the units in flight. Units run seed-major (seed 0
of every cell first), so a partial run already covers every cell.

`--budget-min 110` stops starting new units after 110 minutes, lets the
running ones finish, and exits with status 75 without writing `--out`; run
the same command again to continue. That keeps every invocation under the
two-hour chunk rule on a laptop that needs to cool between chunks.
"""
INCOMPLETE = 75   # EX_TEMPFAIL: units left, rerun the same command
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np

from so_arm100_sim.dr import EVAL_PHYSICS
from so_arm100_sim.env import ArmEnv, TaskSpec
from so_arm100_sim.evaluate import format_gap_table, provenance, run_episode, save_json

from so_arm100_il.policy import policy_factory_from_checkpoint


def _one_thread():
    """Pool initializer: one thread per worker process, so --jobs N uses N
    cores. torch follows set_num_threads; Mesa's software renderer (llvmpipe,
    the only GL on this machine) starts one thread per core unless
    LP_NUM_THREADS is set before the first GL context, which is created later,
    inside the unit."""
    os.environ["LP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def env_kwargs_for(model, image_h, image_w, episode_steps=None):
    kw = {"obs_mode": "proprio", "action_mode": "absolute"}
    if model.needs_image:
        kw["image"] = {"camera": "front", "height": image_h, "width": image_w}
    if episode_steps:
        kw["episode_steps"] = episode_steps
    return kw


def run_unit(ckpt, ensemble_m, task, phys, seed, episodes, env_kwargs):
    """One (physics, seed) unit: the inner loop of `so_arm100_sim.evaluate.evaluate`
    for a single seed, returning the per-episode records so units can be merged
    exactly. Top-level so a worker process can run it."""
    t0 = time.perf_counter()
    factory, _ = policy_factory_from_checkpoint(ckpt, ensemble_m)
    env = ArmEnv(TaskSpec.parse(task), seed=int(seed), physics=phys, **env_kwargs)
    policy = factory(env)
    eps = [run_episode(env, policy, int(seed) * 10_007 + j) for j in range(episodes)]
    env.close()
    return {"physics": phys, "seed": int(seed), "episodes": eps, "wall_s": time.perf_counter() - t0}


def merge_units(task, phys, n_episodes, units):
    """Seed units of one cell -> the dict `evaluate()` returns for that cell."""
    units = sorted(units, key=lambda u: u["seed"])
    per_seed = [float(np.mean([e["success"] for e in u["episodes"]])) for u in units]
    all_eps = [e for u in units for e in u["episodes"]]
    firsts = [e["first_success"] for e in all_eps if e["first_success"] is not None]
    return {
        "task": str(TaskSpec.parse(task)), "physics": phys, "dr": None,
        "n_episodes": int(n_episodes), "seeds": [u["seed"] for u in units],
        "success_rate": float(np.mean(per_seed)),
        "success_std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed,
        "return_mean": float(np.mean([e["return"] for e in all_eps])),
        "steps_mean": float(np.mean([e["steps"] for e in all_eps])),
        "first_success_p50": float(np.percentile(firsts, 50)) if firsts else None,
        "first_success_p90": float(np.percentile(firsts, 90)) if firsts else None,
        "wall_s": float(sum(u["wall_s"] for u in units)),
        **provenance(),
    }


def _part_path(parts, phys, seed):
    return os.path.join(parts, f"{phys}_s{seed}.json")


def run_all(ckpt, task, physics, episodes, seeds, env_kwargs, ensemble_m=0.1, jobs=1,
            parts=None, log=print, budget_s=None):
    """Evaluate every (physics, seed) unit, reusing finished units in `parts`.
    Returns None when `budget_s` ran out before every unit was done."""
    t0 = time.perf_counter()
    in_budget = lambda: budget_s is None or time.perf_counter() - t0 < budget_s
    todo, done = [], {}
    for s in range(seeds):
        for phys in physics:
            p = parts and _part_path(parts, phys, s)
            if p and os.path.exists(p):
                with open(p) as f:
                    u = json.load(f)
                if len(u["episodes"]) == episodes:
                    done[(phys, s)] = u
                    continue
            todo.append((ckpt, ensemble_m, task, phys, s, episodes, env_kwargs))
    if done:
        log(f"resuming: {len(done)} units already in {parts}, {len(todo)} to run")
    if parts:
        os.makedirs(parts, exist_ok=True)

    def keep(u):
        done[(u["physics"], u["seed"])] = u
        if parts:
            tmp = _part_path(parts, u["physics"], u["seed"]) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(u, f)
            os.replace(tmp, _part_path(parts, u["physics"], u["seed"]))
        rate = np.mean([e["success"] for e in u["episodes"]])
        log(f"  {u['physics']:9s} seed {u['seed']}  success {rate:.3f}  {u['wall_s']:.0f} s  "
            f"[{len(done)}/{len(physics) * seeds}]")

    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(min(jobs, len(todo)), mp_context=mp.get_context("spawn"),
                                 initializer=_one_thread) as pool:
            queue, running = list(todo), set()
            while queue or running:
                while queue and len(running) < jobs and in_budget():
                    running.add(pool.submit(run_unit, *queue.pop(0)))
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    keep(f.result())
    else:
        for x in todo:
            if not in_budget():
                break
            keep(run_unit(*x))
    if len(done) < len(physics) * seeds:
        log(f"budget spent: {len(done)}/{len(physics) * seeds} units done; rerun to continue")
        return None
    return {phys: merge_units(task, phys, episodes, [done[(phys, s)] for s in range(seeds)])
            for phys in physics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="lift:red")
    ap.add_argument("--physics", nargs="*", default=list(EVAL_PHYSICS))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ensemble-m", type=float, default=0.1)
    ap.add_argument("--image-h", type=int, default=96)
    ap.add_argument("--image-w", type=int, default=128)
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=1, help="units evaluated in parallel processes")
    ap.add_argument("--budget-min", type=float, default=0,
                    help="stop starting units after this many minutes, exit 75 (0 = no limit)")
    a = ap.parse_args()

    _, model = policy_factory_from_checkpoint(a.ckpt, a.ensemble_m)
    parts = (os.path.splitext(a.out)[0] + ".parts") if a.out else None
    results = run_all(a.ckpt, a.task, a.physics, a.episodes, a.seeds,
                      env_kwargs_for(model, a.image_h, a.image_w), a.ensemble_m, a.jobs, parts,
                      budget_s=a.budget_min * 60 or None)
    if results is None:
        sys.exit(INCOMPLETE)
    print()
    for phys, r in results.items():
        print(f"  {phys:9s} success {r['success_rate']:.3f} +- {r['success_std']:.3f}", flush=True)
    print()
    print(format_gap_table(results))
    if a.out:
        save_json(a.out, {"ckpt": a.ckpt, "task": a.task, "model": model.kind,
                          "episodes": a.episodes, "seeds": a.seeds, "results": results})
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
