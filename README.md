# so-arm100-il

Imitation learning on the [SO-ARM100 MuJoCo bench](https://github.com/AungKaung1928/so-arm100-sim), CPU only: from the same 200 scripted demonstrations, an ACT-style chunked policy lifts the cube 0.77 of the time (one training seed) where single-step behaviour cloning reaches 0.05 and DAgger 0.14.

<img src="out/scaling_all.png" width="720" alt="Lift success against the number of scripted demonstrations for the state, image and ACT-style models, with single training seeds as faint points">

Lift on nominal physics, 100 episodes × 5 evaluation seeds, mean ± sd over seeds, one training seed each:

| policy | input | output | lift success |
|---|---|---|---|
| scripted expert (bench ceiling) | true cube pose | IK waypoints | 0.93 ± 0.02 |
| BC, `ImagePolicy` | image + proprio | 1 action | 0.05 ± 0.02 |
| DAgger, `ImagePolicy` | image + proprio, + 100 relabelled rollouts | 1 action | 0.14 ± 0.07 |
| BC, `ACTLite` | image + proprio | chunk of 20 actions | 0.77 ± 0.05 |

The scaling curve shows the limit: at 200 demonstrations `ACTLite` spans 0.12 to 0.92 across 6 training seeds. Held-out physics, the scaling curve and what failed: [docs/DETAILS.md](docs/DETAILS.md). Walkthrough: https://aungkaung1928.github.io/projects/so-arm100.html

## Quick start

Linux, Python 3.12+ (lerobot; Ubuntu: the `python3-venv` package), CPU only.

```bash
git clone <this repo URL>
cd so-arm100-il
python3 -m venv .venv && . .venv/bin/activate                                    # first time only
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu  # first time only
pip install -r requirements.txt && pip install -e ".[dev]"                       # first time only
python -m pytest -q tests          # 44 passed in ~20 s, including a record-train-evaluate smoke
```

Recording, training, evaluation and Docker: [docs/DETAILS.md#reproducing](docs/DETAILS.md#reproducing).

## How it works

1. `scripts/record_demos.py`: the bench's scripted expert records demonstrations in LeRobot format.
2. Three models, one L1 loss: `StateMLP` (proprio only, the floor), `ImagePolicy` (image + proprio, one action), `ACTLite` (chunk of 20 actions).
3. DAgger relabels the image policy's own rollouts with the scripted expert.

## License

MIT, see LICENSE. The bench and its vendored arm model carry their own licences.
