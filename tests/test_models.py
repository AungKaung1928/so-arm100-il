import numpy as np
import torch

from so_arm100_il.models import (ACTLite, ImagePolicy, StateMLP, TemporalEnsembler, build,
                                 masked_l1)


def _batch(b=3, h=96, w=128):
    return torch.randn(b, 15), torch.randint(0, 255, (b, 3, h, w), dtype=torch.uint8)


def test_output_shapes():
    s, img = _batch()
    assert StateMLP()(s).shape == (3, 1, 6)
    assert ImagePolicy(keypoints=8)(s, img).shape == (3, 1, 6)
    m = ACTLite(chunk=7, keypoints=8, d_model=32, nhead=4, ff=64)
    assert m(s, img).shape == (3, 7, 6) and m.chunk == 7


def test_image_policy_handles_other_resolutions():
    s, img = _batch(2, 48, 64)
    assert ImagePolicy(keypoints=4)(s, img).shape == (2, 1, 6)


def test_build_and_flags():
    assert not build("state").needs_image and build("image").needs_image
    assert build("act", chunk=5).chunk == 5


def test_masked_l1_ignores_padding():
    pred = torch.zeros(2, 3, 6)
    tgt = torch.ones(2, 3, 6)
    tgt[:, 2] = 100.0                      # padded entries carry garbage
    mask = torch.tensor([[False, False, True], [False, False, True]])
    assert abs(masked_l1(pred, tgt, mask).item() - 1.0) < 1e-6
    assert masked_l1(pred, tgt, torch.zeros(2, 3, dtype=torch.bool)).item() > 1.0


def test_ensembler_weights_sum_to_one_and_single_prediction_passes_through():
    ens = TemporalEnsembler(chunk=4, m=0.3)
    for n in (1, 2, 4):
        assert abs(ens.weights(n).sum().item() - 1.0) < 1e-6
    c0 = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    a0 = ens.step(c0)
    assert torch.allclose(a0, c0[0])          # only one prediction: exactly it
    c1 = c0 + 10.0
    a1 = ens.step(c1)
    # step 1: newest chunk says c1[0], older chunk says c0[1]; weighted between them
    lo, hi = torch.minimum(c1[0], c0[1]), torch.maximum(c1[0], c0[1])
    assert torch.all(a1 >= lo - 1e-6) and torch.all(a1 <= hi + 1e-6)
    w = ens.weights(2)
    assert torch.allclose(a1, w[0] * c1[0] + w[1] * c0[1], atol=1e-5)


def test_ensembler_m_zero_is_plain_mean_and_buffer_expires():
    ens = TemporalEnsembler(chunk=2, m=0.0)
    ens.step(torch.zeros(2, 6))
    a = ens.step(torch.ones(2, 6) * 4)
    assert torch.allclose(a, torch.full((6,), 2.0))     # mean of 0 (old) and 4 (new)
    ens.step(torch.ones(2, 6) * 8)
    assert len(ens._preds) == 2                         # the first chunk has expired
