"""README held-out table, real_fit sentence and DAgger numbers match the saved eval JSONs."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md")).read()
FLAT = " ".join(README.split())


def _load(name):
    return json.load(open(os.path.join(ROOT, "runs", name)))


def _ms(r):
    return f"{r['success_rate']:.2f} ± {r['success_std']:.2f}"


def test_heldout_table_matches_evals():
    image, act = _load("eval_bc_image_lift.json")["results"], _load("eval_bc_act_lift.json")["results"]
    lines = README[README.index("| physics | expert (bench) | BC image | ACT |"):].splitlines()[2:]
    rows = {}
    for line in lines:
        if not line.startswith("|"):
            break
        c = [x.strip() for x in line.strip("|").split("|")]
        rows[c[0]] = c[2:]
    assert set(rows) == set(image) == set(act), sorted(rows)
    for k, (ci, ca) in rows.items():
        assert ci == _ms(image[k]), (k, ci, _ms(image[k]))
        assert ca == _ms(act[k]), (k, ca, _ms(act[k]))


def test_headline_rows_match_nominal():
    image, act = _load("eval_bc_image_lift.json")["results"], _load("eval_bc_act_lift.json")["results"]
    assert f"| BC, `ImagePolicy` | image + proprio | 1 action | {_ms(image['nominal'])} |" in README
    assert f"| BC, `ACTLite` | image + proprio | chunk of 20 actions | {_ms(act['nominal'])} |" in README


def test_realfit_sentence():
    r = _load("eval_bc_act_lift_realfit.json")["results"]["real_fit"]
    seeds = " / ".join(f"{s:.2f}" for s in r["per_seed"])
    assert f"ACT {_ms(r)} (seeds {seeds})" in FLAT


def test_dagger_numbers():
    d = _load("dagger_image_lift.json")
    learner = " / ".join(f"{i['learner_success']:.2f}" for i in d["iters"])
    assert f"| `ImagePolicy` + DAgger, 5 iterations | lift success, nominal | {_ms(d['final'])} |" in README
    assert f"{round(d['wall_s']):,} s wall" in README
    assert f"per-iteration learner success {learner}" in FLAT
