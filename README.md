# so-arm100-il

Imitation learning on the [SO-ARM100 MuJoCo bench](https://github.com/AungKaung1928/so-arm100-sim):
behaviour cloning from scripted demonstrations, DAgger with the same scripted
expert as the labelling oracle, and an ACT-style chunked policy. Every policy
is measured with the bench's protocol on nominal and held-out physics, and
the main experiment is a dataset-size curve: how many demonstrations each
model needs. CPU only.

**Status: code complete, tests green, no number measured yet.** Every
`TODO(measure)` names the command that fills it in from a JSON; none is
typed by hand.

**Walkthrough:** https://aungkaung1928.github.io/projects/so-arm100.html — the bench and the three policy projects built on it, explained end to end.

## Why imitation after reinforcement learning on the same bench

The sibling project [`so-arm100-rl`](https://github.com/AungKaung1928/so-arm100-rl)
trains PPO on privileged 25-d state. Imitation is the other route to the
same tasks and the one real arms mostly take: no reward shaping, no
privileged state, a few hundred demonstrations and an image. Running both
on one bench with one success criterion is what makes the comparison mean
something. Here the policy sees what the real SO-ARM100 would see, the 15-d
proprio vector and a 96x128 front-camera image, and nothing else.

## The expert, twice

The bench's `ScriptedExpert` (IK waypoints on the simulator's true cube
pose) plays two roles:

- **Demonstrator.** `scripts/record_demos.py` keeps its successful episodes
  as a LeRobotDataset v3: proprio, front image, the expert's absolute joint
  targets as the action, an instruction from the bench's training templates.
- **Oracle.** DAgger needs a label at states the learner visits, which a
  human demonstrator cannot give after the fact. The scripted expert can:
  `expert.act()` at any state returns what it would do from there, its phase
  machine advancing on the env's true state. `so_arm100_il/shadow.py` runs
  it in the shadow of the learner, and `tests/test_dagger.py` checks that
  with beta = 1 the labels equal what the expert alone would have done.

Its own success rate, measured by the bench, is the ceiling every learned
policy is compared against; development readings were 1.00 reach, about
0.9 lift / pick_place and 0.8-0.95 push on nominal physics.

## Three models, one loss

| model | input | output | why it is here |
|---|---|---|---|
| `StateMLP` | 15-d proprio | 1 action | the floor: proprioception alone cannot locate the cube |
| `ImagePolicy` | image + proprio | 1 action | a small CNN ending in spatial soft-argmax keypoints, the visuomotor front end that won an earlier pose-regression project on this laptop; keypoints + proprio into an MLP |
| `ACTLite` | image + proprio | chunk of 20 actions | ACT without the CVAE: keypoint tokens + proprio token through a 2-layer encoder, 20 learned queries through a 2-layer decoder; overlapping chunks are temporally ensembled at run time |

All three train with **L1 on normalised actions**. MSE fits the conditional
mean, and when demonstrations are multimodal (two valid grasp yaws, two ways
round a cube) the mean is a third, invalid action. L1 fits the conditional
median, which is one of the modes; it is what ACT trains with. Padded chunk
entries are masked. Normalisation statistics come from the training
episodes only and travel with the checkpoint; the split is by episode, never
by frame. The final epoch is reported, nothing is selected on validation.

## The experiments

**Scaling curve** (`scaling.py`): for sizes 10, 25, 50, 100, 200 episodes
and 3 seeds each, train on the first `size` episodes and evaluate with the
bench protocol. Same first episodes for every seed, so seeds differ only in
initialisation and shuffling.

`nice -n 10 python scaling.py --root data/lift_red --repo-id local/so_arm100_lift_red --task lift:red --model image --sizes 10 25 50 100 200 --seeds 3` → `runs/scaling_image.json`, `out/scaling_image.png`

| episodes | `state` | `image` | `act` |
|---|---|---|---|
| 10 | TODO(measure) | TODO(measure) | TODO(measure) |
| 25 | | | |
| 50 | | | |
| 100 | | | |
| 200 | | | |

**DAgger** (`dagger.py`): beta_i = 0.5 · 0.5^i, 20 rollouts per iteration,
labels from the shadow expert, retrain from scratch on the aggregate (or
`--warm`), 5 iterations. The comparison is BC on the same 200 demonstrations.

`nice -n 10 python dagger.py --root data/lift_red --repo-id local/so_arm100_lift_red --task lift:red --model image --iters 5 --rollouts 20 --out runs/dagger_image_lift.pt`

| | BC, 200 demos | DAgger, 200 demos + 100 relabelled rollouts |
|---|---|---|
| lift, nominal | TODO(measure) | TODO(measure) |

**Held-out physics** (`eval_policy.py`): the best image policy on every cell
of the bench's `EVAL_PHYSICS`, 100 episodes x 5 seeds each. The
demonstrations were recorded on nominal physics, so this is a transfer
table, and the expert's own table from the bench is the reference.

`python eval_policy.py --ckpt runs/bc_image_lift.pt --task lift:red --out runs/eval_bc_image_lift.json`

| physics | expert (bench) | BC image | ACT |
|---|---|---|---|
| nominal | TODO(measure) | TODO(measure) | TODO(measure) |
| heavy | | | |
| slippery | | | |
| weak | | | |
| laggy | | | |
| noisy | | | |
| small | | | |

## Experimental: demonstrations from a hand video

`so_arm100_il/human_video.py` + `scripts/hand_to_arm.py`. MediaPipe Hand
Landmarker on an overhead phone video; wrist position → a point in the
spawn box, thumb–index pinch distance → jaw angle and a two-level height;
IK from the bench → joint targets; the targets are **replayed in the
simulator** and the simulator's own images and proprio are what gets
recorded. Said plainly: the video supplies the motion, the simulator
supplies everything the policy sees. It has run on synthetic landmarks
only (`tests/test_human_video.py`); it has not been validated on a real
recording and nothing here claims it works on one. The model file is not
shipped; the script's docstring has the download URL.

## Reproducing

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e ".[dev]"
./verify.sh
```

Python ≥ 3.12 (lerobot). `MUJOCO_GL=glfw` with a display or `egl`/`osmesa`
headless for anything that renders. `docker build -t so-arm100-il . && docker run --rm so-arm100-il`
runs the tests that need no GL context. Image built on 2026-09-23 and its default command passed inside it (17 tests passed, 2 skipped), image size 3.16 GB.

The test suite includes an end-to-end smoke: record three 8-step episodes
through the bench recorder, train all three models for two epochs, evaluate
each for one episode. About a minute.

## Limits, stated

- Simulation only, scripted demonstrations only. A policy that imitates an
  IK script inherits the script's failure modes and none of a human's.
- The `state` model is a floor, not a contender: proprioception cannot see
  the cube.
- Rendering is software GL at about 20 ms per frame here, which is what
  bounds image-policy evaluation, not the models.
- `ACTLite` has no CVAE; on near-deterministic scripted demonstrations
  there is little for a latent to encode, and that is an assumption until
  `--cvae` exists (see `docs/ISSUES.md`).
- The human-video path is unvalidated on real video.

## Licence

MIT. The bench and its vendored arm model carry their own licences.
