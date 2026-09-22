"""DAgger's oracle: the scripted expert running in the shadow of the learner.

The bench's `ScriptedExpert` is a phase machine over the simulator's true
state. It does not need to have produced the trajectory it is asked about:
call `expert.act()` at any state and it returns what it would do from there,
advancing its phases on what the env actually is. That is exactly the
labelling oracle DAgger assumes and a human demonstrator cannot be.

`shadow_rollout` executes a beta-mixture of expert and learner actions and
records, at every visited state, the expert's action as the label. With
beta = 1 the trajectory is the expert's own and the labels equal what the
expert alone would have done -- `tests/test_dagger.py` asserts that.
"""
import numpy as np

from so_arm100_sim.expert import ScriptedExpert

from .data import Episode


def shadow_rollout(env, learner, beta, rng, seed, expert=None, instruction=""):
    """One episode. Returns (Episode with expert labels, success, n_expert_steps)."""
    expert = expert or ScriptedExpert(env)
    obs = env.reset(seed=seed)
    expert.reset()
    if learner is not None and hasattr(learner, "reset"):
        learner.reset()
    states, images, labels = [], [], []
    n_expert = 0
    while True:
        label = expert.act()                        # the oracle's answer at this state
        if learner is None or rng.random() < beta:
            a = label
            n_expert += 1
        else:
            a = learner.act(obs)
        states.append(np.asarray(obs["state"], np.float32))
        if "image" in obs:
            images.append(np.ascontiguousarray(obs["image"].transpose(2, 0, 1)))
        labels.append(np.asarray(label, np.float32))
        obs, _, done, info = env.step(a)
        if done:
            ep = Episode(state=np.stack(states), action=np.stack(labels),
                         image=np.stack(images) if images else None, task=instruction, index=-1)
            return ep, bool(info["success_ever"]), n_expert
