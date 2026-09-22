"""Record -> train -> evaluate, tiny. Needs lerobot and a GL context."""
import numpy as np
import pytest

from so_arm100_sim import render
from so_arm100_sim.datasets import lerobot_available


@pytest.mark.skipif(not lerobot_available(), reason="lerobot not installed")
@pytest.mark.skipif(not render.available(), reason="no offscreen GL context")
def test_record_train_eval(tmp_path):
    from so_arm100_sim.datasets import LeRobotRecorder
    from so_arm100_sim.env import ArmEnv
    from so_arm100_sim.evaluate import evaluate
    from so_arm100_sim.expert import ScriptedExpert

    from so_arm100_il.data import load_dataset
    from so_arm100_il.policy import LearnedPolicy, load_checkpoint, save_checkpoint
    from so_arm100_il.training import train

    image = {"camera": "front", "height": 32, "width": 48}
    env = ArmEnv("reach:red", obs_mode="proprio", action_mode="absolute", seed=0, image=image,
                 episode_steps=8)
    ex = ScriptedExpert(env)
    rec = LeRobotRecorder(tmp_path / "ds", "test/il_smoke", fps=20, cameras=("front",),
                          height=32, width=48)
    for ep in range(3):
        obs = env.reset(seed=ep)
        ex.reset()
        frames = []
        for _ in range(8):
            a = ex.act()
            frames.append({"state": obs["state"], "action": a, "task": "go to the red cube",
                           "images": {"front": obs["image"]}})
            obs, _, done, _ = env.step(a)
            if done:
                break
        rec.add_episode(frames)
    rec.finalize()
    env.close()

    eps = load_dataset(tmp_path / "ds", "test/il_smoke")
    assert len(eps) == 3 and eps[0].image.shape[1:] == (3, 32, 48)
    eps2 = load_dataset(tmp_path / "ds", "test/il_smoke", n_episodes=2, with_images=False)
    assert len(eps2) == 2 and eps2[0].image is None

    logs = []
    for kind in ("state", "image", "act"):
        model, stats, hist = train(eps, kind, epochs=2, batch_size=8, seed=0, chunk=4,
                                   model_kwargs={"keypoints": 4} if kind != "state" else None,
                                   log=logs.append)
        assert len(hist) == 2 and np.isfinite(hist[-1]["train_l1"])
        p = save_checkpoint(str(tmp_path / f"{kind}.pt"), model, stats, {"smoke": True})
        m2, st2, cfg = load_checkpoint(p)
        assert m2.kind == kind and cfg["smoke"] and np.allclose(st2.action_mean, stats.action_mean)
        kw = {"obs_mode": "proprio", "action_mode": "absolute", "episode_steps": 5}
        if m2.needs_image:
            kw["image"] = image
        r = evaluate(lambda e: LearnedPolicy(m2, st2), "reach:red", n_episodes=1, seeds=(0,),
                     env_kwargs=kw)
        assert 0.0 <= r["success_rate"] <= 1.0
    assert any("params" in line for line in logs)
