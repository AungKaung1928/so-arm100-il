"""Demonstrations in memory, normalisation statistics, and chunked windows.

A LeRobotDataset is read once into per-episode numpy arrays. The datasets
this project trains on are small (a few hundred episodes of 96x128 frames),
so decoding everything up front and indexing tensors is simpler and faster
than a DataLoader with workers, and it makes the episode-level split
explicit: `split_episodes` partitions by episode index, never by frame, so a
validation frame is never one step away from a training frame of the same
episode.

Normalisation statistics are computed from the TRAINING episodes only and
travel with every checkpoint (`Stats.to_dict`). A policy evaluated with the
wrong statistics is a different policy, so they are not recomputed at eval.

Chunked windows for ACT: frame t is paired with actions t .. t+k-1. Past the
end of the episode the window is padded with the last action, and a padding
mask marks those entries so the loss ignores them.
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import Dataset

STATE_DIM = 15
ACT_DIM = 6


@dataclass
class Stats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self):
        return {k: np.asarray(v, np.float32).tolist() for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: np.asarray(v, np.float32) for k, v in d.items()})


@dataclass
class Episode:
    state: np.ndarray            # (T, 15) float32
    action: np.ndarray           # (T, 6) float32, absolute joint targets in radians
    image: np.ndarray | None     # (T, 3, H, W) uint8, or None
    task: str
    index: int

    def __len__(self):
        return self.state.shape[0]


def _to_uint8_chw(img):
    """LeRobot hands back float CHW in [0, 1] by default; keep uint8 CHW."""
    t = torch.as_tensor(img)
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    if t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
        t = t.permute(2, 0, 1)
    return t.numpy()


def load_dataset(root, repo_id, n_episodes=None, camera="front", with_images=True):
    """Read a LeRobotDataset into `Episode`s, the first `n_episodes` of them."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=str(root))
    img_key = f"observation.images.{camera}"
    episodes = {}
    for i in range(len(ds)):
        item = ds[i]
        e = int(item["episode_index"])
        if n_episodes is not None and e >= n_episodes:
            continue
        ep = episodes.setdefault(e, {"state": [], "action": [], "image": [], "task": None})
        ep["state"].append(np.asarray(item["observation.state"], np.float32))
        ep["action"].append(np.asarray(item["action"], np.float32))
        if with_images and img_key in item:
            ep["image"].append(_to_uint8_chw(item[img_key]))
        if ep["task"] is None:
            ep["task"] = str(item.get("task", ""))
    out = []
    for e in sorted(episodes):
        ep = episodes[e]
        out.append(Episode(state=np.stack(ep["state"]), action=np.stack(ep["action"]),
                           image=np.stack(ep["image"]) if ep["image"] else None,
                           task=ep["task"], index=e))
    return out


def split_episodes(n, val_frac=0.2, seed=0):
    """Deterministic episode-level split. Returns (train_idx, val_idx)."""
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def compute_stats(episodes):
    s = np.concatenate([e.state for e in episodes])
    a = np.concatenate([e.action for e in episodes])
    return Stats(s.mean(0), s.std(0) + 1e-6, a.mean(0), a.std(0) + 1e-6)


def normalize(x, mean, std):
    return (np.asarray(x, np.float32) - mean) / std


def denormalize(x, mean, std):
    return np.asarray(x, np.float32) * std + mean


class FrameDataset(Dataset):
    """Flattened frames. With `chunk > 1`, targets are (chunk, 6) windows.

    Everything is pre-normalised and stacked once, so __getitem__ is a slice.
    Images stay uint8 and are converted to float inside the model so the
    dataset costs 1 byte per pixel in RAM.
    """

    def __init__(self, episodes, stats, chunk=1, with_images=True):
        self.chunk = int(chunk)
        states, actions, masks, images = [], [], [], []
        for ep in episodes:
            T = len(ep)
            s = normalize(ep.state, stats.state_mean, stats.state_std)
            a = normalize(ep.action, stats.action_mean, stats.action_std)
            states.append(s)
            if self.chunk == 1:
                actions.append(a[:, None, :])
                masks.append(np.zeros((T, 1), bool))
            else:
                # window t..t+k-1, padded with the last action past the end
                pad = np.repeat(a[-1:], self.chunk, axis=0)
                ext = np.concatenate([a, pad])
                win = np.stack([ext[t:t + self.chunk] for t in range(T)])
                m = np.stack([np.arange(t, t + self.chunk) >= T for t in range(T)])
                actions.append(win)
                masks.append(m)
            if with_images and ep.image is not None:
                images.append(ep.image)
        self.state = torch.from_numpy(np.concatenate(states).astype(np.float32))
        self.action = torch.from_numpy(np.concatenate(actions).astype(np.float32))
        self.mask = torch.from_numpy(np.concatenate(masks))
        self.image = torch.from_numpy(np.concatenate(images)) if images else None

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, i):
        img = self.image[i] if self.image is not None else torch.zeros(0, dtype=torch.uint8)
        return self.state[i], img, self.action[i], self.mask[i]
