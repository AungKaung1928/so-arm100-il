import numpy as np
import torch

from so_arm100_il.data import (Episode, FrameDataset, Stats, compute_stats, denormalize,
                               normalize, split_episodes)


def make_episode(T, idx=0, with_image=True, seed=0):
    rng = np.random.default_rng(seed + idx)
    return Episode(state=rng.normal(size=(T, 15)).astype(np.float32),
                   action=np.arange(T, dtype=np.float32)[:, None].repeat(6, 1) + idx * 100,
                   image=rng.integers(0, 255, (T, 3, 8, 12), dtype=np.uint8) if with_image else None,
                   task="t", index=idx)


def test_split_is_by_episode_deterministic_and_disjoint():
    a = split_episodes(20, 0.2, seed=3)
    b = split_episodes(20, 0.2, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert len(a[1]) == 4 and not set(a[0]) & set(a[1])
    assert set(a[0]) | set(a[1]) == set(range(20))
    tr, va = split_episodes(1, 0.2)
    assert len(tr) == 1 and len(va) == 0


def test_stats_round_trip():
    eps = [make_episode(10, i) for i in range(3)]
    st = compute_stats(eps)
    x = eps[1].action
    assert np.allclose(denormalize(normalize(x, st.action_mean, st.action_std),
                                   st.action_mean, st.action_std), x, atol=1e-4)
    st2 = Stats.from_dict(st.to_dict())
    assert np.allclose(st2.state_mean, st.state_mean) and np.allclose(st2.action_std, st.action_std)


def test_chunk_windows_pad_with_last_action_and_mask():
    ep = make_episode(5, 0)
    st = Stats(np.zeros(15, np.float32), np.ones(15, np.float32),
               np.zeros(6, np.float32), np.ones(6, np.float32))
    ds = FrameDataset([ep], st, chunk=3)
    assert len(ds) == 5
    s, img, a, m = ds[3]
    assert a.shape == (3, 6) and m.shape == (3,)
    # frame 3 of 5: actions 3, 4, then pad(4)
    assert torch.allclose(a[:, 0], torch.tensor([3.0, 4.0, 4.0]))
    assert m.tolist() == [False, False, True]
    s, img, a, m = ds[0]
    assert m.tolist() == [False, False, False]
    assert img.shape == (3, 8, 12) and img.dtype == torch.uint8


def test_chunk_one_has_no_padding_and_no_image_when_asked():
    ds = FrameDataset([make_episode(4, 0, with_image=False)],
                      compute_stats([make_episode(4, 0)]), chunk=1)
    s, img, a, m = ds[2]
    assert a.shape == (1, 6) and m.tolist() == [False] and img.numel() == 0
    assert ds.image is None
