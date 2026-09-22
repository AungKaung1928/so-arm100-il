"""Evaluate a checkpoint with the bench protocol on nominal and held-out physics.

    python eval_policy.py --ckpt runs/bc_image_lift.pt --task lift:red \
        --physics nominal heavy slippery weak laggy noisy small --out runs/eval_bc_image_lift.json
"""
import argparse

from so_arm100_sim.dr import EVAL_PHYSICS
from so_arm100_sim.evaluate import evaluate, format_gap_table, save_json

from so_arm100_il.policy import policy_factory_from_checkpoint


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
    a = ap.parse_args()

    factory, model = policy_factory_from_checkpoint(a.ckpt, a.ensemble_m)
    env_kwargs = {"obs_mode": "proprio", "action_mode": "absolute"}
    if model.needs_image:
        env_kwargs["image"] = {"camera": "front", "height": a.image_h, "width": a.image_w}
    results = {}
    for phys in a.physics:
        r = evaluate(factory, a.task, n_episodes=a.episodes, seeds=tuple(range(a.seeds)),
                     physics=phys, env_kwargs=env_kwargs)
        results[phys] = r
        print(f"  {phys:9s} success {r['success_rate']:.3f} +- {r['success_std']:.3f}", flush=True)
    print()
    print(format_gap_table(results))
    if a.out:
        save_json(a.out, {"ckpt": a.ckpt, "task": a.task, "model": model.kind,
                          "episodes": a.episodes, "seeds": a.seeds, "results": results})
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
