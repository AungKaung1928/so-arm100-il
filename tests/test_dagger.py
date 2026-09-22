"""The shadow-expert oracle: labels equal the expert's own actions when the
expert drives, and the expert keeps tracking when the learner drives."""
import numpy as np

from so_arm100_sim.env import ArmEnv
from so_arm100_sim.expert import ScriptedExpert

from so_arm100_il.shadow import shadow_rollout


class Zero:
    """A learner that never moves the commanded target (holds the pose)."""
    def reset(self):
        pass

    def act(self, obs):
        return obs["state"][:6]        # absolute mode: current joint angles = hold


def test_beta_one_labels_equal_expert_alone():
    env = ArmEnv("lift:red", obs_mode="proprio", action_mode="absolute", seed=0, episode_steps=12)
    ex = ScriptedExpert(env)
    # reference: expert alone
    env.reset(seed=7)
    ex.reset()
    ref = []
    for _ in range(12):
        a = ex.act()
        ref.append(a.copy())
        _, _, done, _ = env.step(a)
        if done:
            break
    ep, _, n_exp = shadow_rollout(env, Zero(), beta=1.0, rng=np.random.default_rng(0), seed=7, expert=ex)
    assert n_exp == len(ref) == len(ep)
    assert np.allclose(ep.action, np.stack(ref), atol=1e-9)
    assert ep.image is None and ep.state.shape == (len(ref), 15)


def test_beta_zero_executes_learner_but_labels_from_expert():
    env = ArmEnv("lift:red", obs_mode="proprio", action_mode="absolute", seed=0, episode_steps=10)
    ex = ScriptedExpert(env)
    ep, _, n_exp = shadow_rollout(env, Zero(), beta=0.0, rng=np.random.default_rng(0), seed=3, expert=ex)
    assert n_exp == 0 and len(ep) == 10
    # the arm held still (learner), so joint angles barely move ...
    assert np.abs(ep.state[-1, :6] - ep.state[0, :6]).max() < 0.05
    # ... while the expert's labels move toward the cube: they differ from the hold
    assert np.abs(ep.action - ep.state[:, :6]).max() > 0.05
    # and change over time, i.e. the expert's phase machine kept running
    assert np.abs(ep.action[-1] - ep.action[0]).max() > 0.05


def test_images_are_recorded_when_the_env_has_them():
    from so_arm100_sim import render
    import pytest
    if not render.available():
        pytest.skip("no GL")
    env = ArmEnv("reach:red", obs_mode="proprio", action_mode="absolute", seed=0, episode_steps=3,
                 image={"camera": "front", "height": 24, "width": 32})
    ep, _, _ = shadow_rollout(env, None, beta=1.0, rng=np.random.default_rng(0), seed=1)
    assert ep.image.shape == (3, 3, 24, 32) and ep.image.dtype == np.uint8
    env.close()
