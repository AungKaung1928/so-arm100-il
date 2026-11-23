"""README held-out table, real_fit sentence, DAgger numbers and training table match the saved JSONs."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()
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


def test_training_table_matches_runs():
    for label, name in (("`ImagePolicy`", "bc_image_lift.json"), ("`ACTLite`", "bc_act_lift.json")):
        r = json.load(open(os.path.join(ROOT, "runs", name)))
        last = r["history"][-1]
        row = (f"| {label} | {r['n_episodes_used']} | {last['epoch']} | {last['train_l1']:.4f} | "
               f"{last['val_l1']:.4f} | {r['wall_s']:.0f} s | {r['threads']} |")
        assert row in README, row


def test_results_summary_rows_match_runs():
    act = _load("eval_bc_act_lift.json")["results"]
    assert f"| `ACTLite` | lift success, `small` cube | {_ms(act['small'])} |" in README
    assert f"| `ACTLite` | lift success, `laggy` / `weak` | {_ms(act['laggy'])} / {_ms(act['weak'])} |" in README
    img, a = _load("bc_image_lift.json"), _load("bc_act_lift.json")
    want = (f"| {img['wall_s']:.0f} s / {img['history'][-1]['val_l1']:.4f} (`ImagePolicy`), "
            f"{a['wall_s']:.0f} s / {a['history'][-1]['val_l1']:.4f} (`ACTLite`) |")
    assert want in README, want


def test_headline_summary_rows_match_evals():
    image, act = _load("eval_bc_image_lift.json")["results"], _load("eval_bc_act_lift.json")["results"]
    d = _load("dagger_image_lift.json")
    lines = README.splitlines()
    assert f"| `ACTLite`, 200 demos, 30 epochs | lift success, nominal | {_ms(act['nominal'])} |" in README
    assert f"| `ImagePolicy`, 200 demos, 30 epochs | lift success, nominal | {_ms(image['nominal'])} |" in README
    assert f"| lift, nominal | {_ms(image['nominal'])} | {_ms(d['final'])} |" in lines


def test_heldout_table_expert_column_matches_bench_copy():
    """Expert column = the bench's scripted-expert lift cells (runs/expert_lift_bench.json, copied with its source commit)."""
    bench = _load("expert_lift_bench.json")
    assert len(bench["source_commit"]) == 40 and len(bench["cells"]) == 7
    lines = README.splitlines()
    for physics, r in bench["cells"].items():
        assert any(l.startswith(f"| {physics} | {_ms(r)} |") for l in lines), physics


def test_record_demos_row_matches_recording_log():
    """runs/recording_log_lift_red.json is record_demos.py's own log (data/lift_red/recording_log.json, copied unchanged)."""
    r = _load("recording_log_lift_red.json")
    t = r["tasks"]["lift:red"]
    assert (f"| `scripts/record_demos.py` | demonstrations kept | {t['kept']} of {t['attempts']} ({t['kept'] / t['attempts']:.2f}) | "
            f"{t['frames']:,} frames at 20 Hz, {r['wall_s']:.0f} s wall |") in README
