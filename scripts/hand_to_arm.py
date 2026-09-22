"""EXPERIMENTAL: a human-hand video -> arm joint targets -> a sim-replayed
LeRobot dataset.

    python scripts/hand_to_arm.py --video my_hand.mp4 --root data/hand_lift \
        --repo-id local/so_arm100_hand_lift --task lift:red --instruction "pick up the red cube"

Needs the MediaPipe hand landmarker model file. It is not shipped here;
download it once (about 8 MB):

    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

and pass `--landmarker hand_landmarker.task`. Frames with no detected hand
are dropped. The result is a dataset whose observations are SIMULATOR
renders of the arm replaying the hand's motion, with the instruction given
here; success is whatever the replay achieved and is printed, not assumed.
"""
import argparse
import json
import os

import numpy as np

from so_arm100_sim.datasets import LeRobotRecorder
from so_arm100_sim.env import ArmEnv, CONTROL_HZ

from so_arm100_il import human_video as HV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--landmarker", default="hand_landmarker.task")
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", default="lift:red")
    ap.add_argument("--instruction", default="pick up the red cube")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th video frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.path.exists(a.landmarker):
        raise SystemExit(f"{a.landmarker} not found; see the docstring for the download URL")

    lm, fps = HV.detect_hands(a.video, a.landmarker, a.max_frames)
    valid = ~np.isnan(lm[:, 0, 0])
    print(f"{len(lm)} frames at {fps:.1f} fps, hand found in {valid.sum()}")
    seq = HV.smooth_landmarks(lm[valid])[::a.stride]
    targets = [HV.landmarks_to_target(x) for x in seq]

    env = ArmEnv(a.task, obs_mode="proprio", action_mode="absolute", seed=a.seed,
                 image={"camera": "front", "height": 96, "width": 128})
    joints = HV.targets_to_joints(env.model, targets)
    rec = LeRobotRecorder(a.root, a.repo_id, fps=CONTROL_HZ, cameras=("front",), height=96, width=128)
    frames, ok = HV.replay_in_sim(env, joints, rec, a.instruction)
    rec.finalize()
    with open(os.path.join(a.root, "hand_replay_log.json"), "w") as f:
        json.dump({"video": os.path.basename(a.video), "frames_video": int(len(lm)),
                   "frames_with_hand": int(valid.sum()), "frames_replayed": len(frames),
                   "task": a.task, "success": ok}, f, indent=2)
    print(f"replayed {len(frames)} steps in sim; task success = {ok}; dataset at {a.root}")


if __name__ == "__main__":
    main()
