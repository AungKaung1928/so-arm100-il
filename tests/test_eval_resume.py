"""eval_policy.py splits the protocol into (physics, seed) units. The merged
result must equal the bench's own evaluate(), serial or parallel, fresh or
resumed from saved units."""
import os

import numpy as np
import torch

import eval_policy as E
from so_arm100_sim.evaluate import evaluate

from so_arm100_il.data import Stats
from so_arm100_il.models import StateMLP
from so_arm100_il.policy import policy_factory_from_checkpoint, save_checkpoint

KEYS = ("success_rate", "success_std", "per_seed", "return_mean", "steps_mean",
        "first_success_p50", "first_success_p90", "seeds", "n_episodes")
PHYS = ("nominal", "heavy")


def _ckpt(tmp_path):
    torch.manual_seed(0)
    stats = Stats(np.zeros(15, np.float32), np.ones(15, np.float32),
                  np.zeros(6, np.float32), np.full(6, 0.3, np.float32))
    return save_checkpoint(os.path.join(tmp_path, "s.pt"), StateMLP(), stats, {})


def _same(a, b):
    for phys in PHYS:
        for k in KEYS:
            assert a[phys][k] == b[phys][k], (phys, k)


def test_units_merge_to_evaluate_and_resume(tmp_path):
    ckpt = _ckpt(tmp_path)
    factory, model = policy_factory_from_checkpoint(ckpt)
    kw = E.env_kwargs_for(model, 96, 128, episode_steps=6)
    ref = {p: evaluate(factory, "lift:red", n_episodes=2, seeds=(0, 1), physics=p, env_kwargs=kw)
           for p in PHYS}

    parts = os.path.join(tmp_path, "parts")
    serial = E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, parts=parts, log=lambda *_: None)
    _same(ref, serial)
    assert len(os.listdir(parts)) == 4

    os.remove(os.path.join(parts, "heavy_s1.json"))
    logs = []
    resumed = E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, parts=parts, log=logs.append)
    _same(ref, resumed)
    assert logs[0].startswith("resuming: 3 units")

    parallel = E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, jobs=2, log=lambda *_: None)
    _same(ref, parallel)


def test_budget_stops_then_rerun_completes(tmp_path):
    ckpt = _ckpt(tmp_path)
    _, model = policy_factory_from_checkpoint(ckpt)
    kw = E.env_kwargs_for(model, 96, 128, episode_steps=6)
    parts = os.path.join(tmp_path, "parts")
    for jobs in (1, 2):
        assert E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, jobs=jobs, parts=parts,
                         log=lambda *_: None, budget_s=1e-9) is None
    full = E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, parts=parts, log=lambda *_: None)
    fresh = E.run_all(ckpt, "lift:red", PHYS, 2, 2, kw, log=lambda *_: None)
    _same(fresh, full)
