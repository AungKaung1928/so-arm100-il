"""EXPERIMENTAL: demonstrations from a video of a human hand.

The chain is: MediaPipe Hand Landmarker on each frame -> 21 landmarks ->
a target for the arm's end effector and jaw -> IK (from the bench) -> joint
targets -> REPLAY in the simulator, recording the simulator's own images and
proprio as the dataset. The recorded observations are therefore sim
observations, not video frames: the human video supplies the motion, the
simulator supplies everything a policy sees. Said plainly because it is the
opposite of what "learning from human video" usually promises.

Mapping, deliberately minimal (see README for what it does not handle):

    wrist landmark (0) image (u, v) in [0, 1]  ->  world (x, y) on the table
        by a fixed scale and the spawn box: u across -> x, v down -> y toward
        the robot. The camera is assumed roughly overhead.
    thumb tip (4) -- index tip (8) distance   ->  jaw angle, linear between
        PINCH_CLOSED and PINCH_OPEN (normalised image units), clipped.
    height                                     ->  fixed hover height, or a
        two-level rule: pinched -> low (grasp height), open -> high.

Nothing here runs a model in the tests: `landmarks_to_target` and
`targets_to_joints` are pure functions over landmark arrays. The MediaPipe
model file is downloaded only by `scripts/hand_to_arm.py`.
"""
import numpy as np

from so_arm100_sim import scene
from so_arm100_sim.ik import IK, top_down_rotation

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
PINCH_CLOSED = 0.04      # normalised image distance at which the jaw is fully closed
PINCH_OPEN = 0.20        # ... and fully open
Z_HIGH = 0.08            # m, hover height while the hand is open
Z_LOW = 0.030            # m, grasp height while pinched (site height for a 25 mm cube)


def pinch_distance(lm):
    lm = np.asarray(lm, float)
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2]))


def pinch_to_jaw(d):
    """Linear map from pinch distance to the Jaw joint angle, clipped."""
    t = np.clip((d - PINCH_CLOSED) / (PINCH_OPEN - PINCH_CLOSED), 0.0, 1.0)
    return float(scene.JAW_CLOSED + t * (scene.JAW_OPEN - scene.JAW_CLOSED))


def landmarks_to_target(lm):
    """(21, 3) normalised landmarks -> (ee_pos (3,), jaw, yaw)."""
    lm = np.asarray(lm, float)
    u, v = float(lm[WRIST, 0]), float(lm[WRIST, 1])
    x = scene.SPAWN_X[0] + np.clip(u, 0, 1) * (scene.SPAWN_X[1] - scene.SPAWN_X[0])
    # v grows downward in the image; the robot sits at the top of an overhead
    # view, so v = 0 is nearest the robot (y = SPAWN_Y[1]) and v = 1 farthest.
    y = scene.SPAWN_Y[1] - np.clip(v, 0, 1) * (scene.SPAWN_Y[1] - scene.SPAWN_Y[0])
    d = pinch_distance(lm)
    jaw = pinch_to_jaw(d)
    z = Z_LOW if d < 0.5 * (PINCH_CLOSED + PINCH_OPEN) else Z_HIGH
    # closing direction = thumb -> index in the image plane
    dvec = lm[INDEX_TIP, :2] - lm[THUMB_TIP, :2]
    yaw = float(np.arctan2(-dvec[1], dvec[0])) if np.linalg.norm(dvec) > 1e-6 else 0.0
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2        # closing axis has 180 deg symmetry
    return np.array([x, y, z]), jaw, yaw


def targets_to_joints(model, targets, q0=None, max_step=0.05):
    """[(pos, jaw, yaw), ...] -> (T, 6) joint targets via IK, rate limited."""
    ik = IK(model)
    q = np.concatenate([scene.home_qpos(model)[:5], [scene.JAW_OPEN]]) if q0 is None else np.asarray(q0, float).copy()
    lo, hi = scene.joint_limits(model)
    out = []
    for pos, jaw, yaw in targets:
        q5, pe, re_, _ = ik.solve(q[:5], pos, top_down_rotation(yaw))
        if pe > 0.01:            # try the mirrored grasp before giving up
            R = top_down_rotation(yaw)
            R[:, 0] *= -1
            R[:, 2] *= -1
            q5b, peb, _, _ = ik.solve(q[:5], pos, R)
            if peb < pe:
                q5 = q5b
        q[:5] += np.clip(q5 - q[:5], -max_step, max_step)
        q[5] += np.clip(jaw - q[5], -2 * max_step, 2 * max_step)
        q = np.clip(q, lo, hi)
        out.append(q.copy())
    return np.stack(out)


def smooth_landmarks(seq, alpha=0.5):
    """Exponential smoothing over a (T, 21, 3) landmark sequence."""
    seq = np.asarray(seq, float)
    out = np.empty_like(seq)
    out[0] = seq[0]
    for t in range(1, len(seq)):
        out[t] = alpha * seq[t] + (1 - alpha) * out[t - 1]
    return out


def replay_in_sim(env, joint_targets, recorder=None, instruction="", cameras=("front",)):
    """Drive the env with the joint targets; optionally record a dataset episode."""
    obs = env.reset()
    frames = []
    success = False
    for q in joint_targets:
        frames.append({"state": obs["state"], "action": np.asarray(q, np.float32), "task": instruction,
                       "images": {c: (obs["image"] if c == env.image_cfg["camera"]
                                      else env.render(c, env.image_cfg["height"], env.image_cfg["width"]))
                                  for c in cameras} if env.image else {}})
        obs, _, done, info = env.step(q)
        success = success or bool(info["success_ever"])
        if done:
            break
    if recorder is not None:
        recorder.add_episode(frames)
    return frames, success


def detect_hands(video_path, model_path, max_frames=None):
    """Run MediaPipe Hand Landmarker over a video. Returns (T, 21, 3) or NaN rows."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, t = [], 0
    with vision.HandLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and t >= max_frames):
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(1000 * t / fps))
            if res.hand_landmarks:
                out.append([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
            else:
                out.append(np.full((21, 3), np.nan))
            t += 1
    cap.release()
    return np.asarray(out, float), fps
