"""The short README repeats a table and numbers of docs/DETAILS.md, which the other README tests check."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT = open(os.path.join(ROOT, "README.md")).read()
DETAILS = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()


def test_short_rows_are_copies():
    rows = [l for l in SHORT.splitlines() if l.startswith("| ") and not l.startswith("| policy |")]
    assert len(rows) == 4
    for l in rows:
        assert l in DETAILS.splitlines(), l


def test_short_seed_span_in_details():
    assert "spans 0.12 to 0.92 across 6 training seeds" in " ".join(SHORT.split())
    assert "seeds 0.92 / 0.82 / 0.76 vs 0.24 / 0.18 / 0.12" in DETAILS


def test_short_headline_and_protocol_match_runs():
    import json

    def load(name):
        return json.load(open(os.path.join(ROOT, "runs", name)))
    act, image = load("eval_bc_act_lift.json"), load("eval_bc_image_lift.json")
    dagger, train = load("dagger_image_lift.json"), load("bc_act_lift.json")
    (head,) = [l for l in SHORT.splitlines() if l.startswith("Imitation learning on")]
    assert (f"from the same {train['n_episodes_used']} scripted demonstrations, an ACT-style chunked policy lifts the cube "
            f"{act['results']['nominal']['success_rate']:.2f} of the time") in head
    assert (f"single-step behaviour cloning reaches {image['results']['nominal']['success_rate']:.2f} "
            f"and DAgger {dagger['final']['success_rate']:.2f}.") in head
    assert f"Lift on nominal physics, {act['episodes']} episodes × {act['seeds']} evaluation seeds" in SHORT
    (tail,) = [l for l in SHORT.splitlines() if l.startswith("The scaling curve")]
    assert f"at {train['n_episodes_used']} demonstrations `ACTLite` spans" in tail
