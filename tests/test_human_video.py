"""Pure-function tests on synthetic landmarks. No video, no model download."""
import numpy as np

from so_arm100_sim import scene

from so_arm100_il import human_video as HV


def hand(u=0.5, v=0.5, pinch=0.1, angle=0.0):
    lm = np.zeros((21, 3))
    lm[:, 0], lm[:, 1] = u, v
    d = np.array([np.cos(angle), -np.sin(angle)]) * pinch / 2
    lm[HV.THUMB_TIP, :2] = [u, v] - d
    lm[HV.INDEX_TIP, :2] = [u, v] + d
    return lm


def test_pinch_to_jaw_is_monotone_and_clipped():
    ds = np.linspace(0.0, 0.3, 31)
    jaws = [HV.pinch_to_jaw(d) for d in ds]
    assert all(b >= a - 1e-12 for a, b in zip(jaws, jaws[1:]))
    assert jaws[0] == scene.JAW_CLOSED and jaws[-1] == scene.JAW_OPEN


def test_wrist_maps_into_the_spawn_box():
    for u, v in [(0, 0), (1, 1), (0.5, 0.5), (0.2, 0.9)]:
        pos, jaw, yaw = HV.landmarks_to_target(hand(u, v))
        assert scene.SPAWN_X[0] - 1e-9 <= pos[0] <= scene.SPAWN_X[1] + 1e-9
        assert scene.SPAWN_Y[0] - 1e-9 <= pos[1] <= scene.SPAWN_Y[1] + 1e-9
    # v = 0 is the row nearest the robot
    assert HV.landmarks_to_target(hand(0.5, 0.0))[0][1] > HV.landmarks_to_target(hand(0.5, 1.0))[0][1]


def test_pinch_lowers_the_hand_and_yaw_is_folded():
    pos_o, jaw_o, _ = HV.landmarks_to_target(hand(pinch=0.25))
    pos_c, jaw_c, _ = HV.landmarks_to_target(hand(pinch=0.02))
    assert pos_o[2] == HV.Z_HIGH and pos_c[2] == HV.Z_LOW and jaw_c < jaw_o
    _, _, yaw = HV.landmarks_to_target(hand(angle=2.0))
    assert -np.pi / 2 - 1e-9 <= yaw <= np.pi / 2 + 1e-9


def test_targets_to_joints_is_solvable_and_rate_limited():
    model = scene.build_model(1)
    seq = [HV.landmarks_to_target(hand(u, 0.5, pinch=0.2)) for u in np.linspace(0.2, 0.8, 15)]
    q = HV.targets_to_joints(model, seq)
    lo, hi = scene.joint_limits(model)
    assert q.shape == (15, 6)
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
    assert np.all(np.abs(np.diff(q[:, :5], axis=0)) <= 0.05 + 1e-9)


def test_smoothing_preserves_constant_sequences():
    seq = np.repeat(hand()[None], 5, axis=0)
    assert np.allclose(HV.smooth_landmarks(seq), seq)
