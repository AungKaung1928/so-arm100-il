#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tier 1 is the test suite (a minute; the end-to-end smoke records three
# 8-step episodes, trains all three models for two epochs and evaluates
# them). Tiers 2-4 print the commands that produce the README's numbers;
# they take minutes to hours and are run on purpose, not by this script.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import so_arm100_sim, torch, so_arm100_il' 2>/dev/null; then
  echo "so_arm100_sim / torch / so_arm100_il are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "    pip install -r requirements.txt && pip install -e ." >&2
  exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1/4  tests -- data windows, models, ensembling, shadow expert, hand mapping, end-to-end smoke"
"$PY" -m pytest tests || exit 1

hr "2/4  demonstrations"
cat <<'MSG'
Records successful expert episodes only (bench script). ~0.4 s per episode
plus 20 ms per rendered frame on software GL:

    python scripts/record_demos.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --tasks lift:red --episodes 200
MSG

hr "3/4  behaviour cloning, evaluation, scaling curve"
cat <<'MSG'
    nice -n 10 python train_bc.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --model state --out runs/bc_state_lift.pt
    nice -n 10 python train_bc.py ... --model image --out runs/bc_image_lift.pt
    nice -n 10 python train_bc.py ... --model act   --out runs/bc_act_lift.pt
    python eval_policy.py --ckpt runs/bc_image_lift.pt --task lift:red --out runs/eval_bc_image_lift.json
    nice -n 10 python scaling.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --task lift:red --model image --sizes 10 25 50 100 200 --seeds 3
MSG

hr "4/4  DAgger"
cat <<'MSG'
    nice -n 10 python dagger.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --task lift:red --model image --iters 5 --rollouts 20 --out runs/dagger_image_lift.pt
MSG
for f in runs/eval_*.json; do
  [ -f "$f" ] && "$PY" - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"\n{sys.argv[1]}: {d['model']} on {d['task']}")
for k, r in d["results"].items():
    print(f"  {k:9s} {r['success_rate']:.3f} +- {r['success_std']:.3f}")
PY
done
exit 0
