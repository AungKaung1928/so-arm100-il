"""Checkpoints, and the wrapper that turns one into a bench policy.

`LearnedPolicy` is what `so_arm100_sim.evaluate.evaluate` receives: an
object with `reset()` and `act(obs)`. It applies the checkpoint's own
normalisation statistics, runs the model, de-normalises, and for chunked
models keeps a `TemporalEnsembler` across the episode.
"""
import numpy as np
import torch

from .data import Stats, normalize, denormalize
from .models import build, TemporalEnsembler


def save_checkpoint(path, model, stats, config):
    torch.save({"kind": model.kind, "state_dict": model.state_dict(),
                "stats": stats.to_dict(), "config": dict(config),
                "model_kwargs": getattr(model, "init_kwargs", {})}, path)
    return path


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["kind"], **ck.get("model_kwargs", {}))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Stats.from_dict(ck["stats"]), ck["config"]


class LearnedPolicy:
    def __init__(self, model, stats, ensemble_m=0.1):
        self.model, self.stats = model.eval(), stats
        self.ens = TemporalEnsembler(model.chunk, ensemble_m) if model.chunk > 1 else None

    def reset(self):
        if self.ens is not None:
            self.ens.reset()

    @torch.no_grad()
    def act(self, obs):
        s = torch.from_numpy(normalize(obs["state"], self.stats.state_mean, self.stats.state_std))[None]
        img = None
        if self.model.needs_image:
            if "image" not in obs:
                raise KeyError("this policy needs obs['image']; build the env with image=...")
            img = torch.from_numpy(np.ascontiguousarray(obs["image"].transpose(2, 0, 1)))[None]
        pred = self.model(s, img)[0]                     # (k, 6) normalised
        a = self.ens.step(pred) if self.ens is not None else pred[0]
        return denormalize(a.numpy(), self.stats.action_mean, self.stats.action_std)


def policy_factory_from_checkpoint(path, ensemble_m=0.1):
    model, stats, _ = load_checkpoint(path)

    def factory(env):
        return LearnedPolicy(model, stats, ensemble_m)
    return factory, model
