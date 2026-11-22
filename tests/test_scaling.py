"""scaling.py: the saved per-seed rows give the saved summary and the README
table, and a rerun skips every (size, seed) row already on disk."""
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("scaling", os.path.join(ROOT, "scaling.py"))
scaling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scaling)

KINDS = ("state", "image", "act")


def _saved(kind):
    return json.load(open(os.path.join(ROOT, "runs", f"scaling_{kind}.json")))


@pytest.mark.parametrize("kind", KINDS)
def test_summary_recomputes_from_rows(kind):
    d = _saved(kind)
    assert scaling.summarise(d["rows"]) == d["summary"]


def test_readme_table_matches_saved_runs():
    lines = open(os.path.join(ROOT, "README.md")).read().splitlines()
    head = lines.index("| episodes | `state` | `image` | `act` |")
    table = lines[head + 2:head + 7]
    sums = {k: {s["size"]: s for s in _saved(k)["summary"]} for k in KINDS}
    expect = [f"| {n} | " + " | ".join(f"{sums[k][n]['mean']:.2f} ± {sums[k][n]['std']:.2f}" for k in KINDS) + " |"
              for n in (10, 25, 50, 100, 200)]
    assert table == expect


def test_rerun_skips_rows_already_on_disk(tmp_path, monkeypatch):
    rows = [{"size": s, "seed": k, "success": 0.5} for s in (10, 25) for k in range(2)]
    path = tmp_path / "x.rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(scaling, "train", lambda *a, **k: pytest.fail("retrained a finished row"))
    out = scaling.run(list(range(30)), "state", [10, 25], 2, "lift", 1, 1, 1, {}, 1,
                      log=lambda *_: None, rows_path=str(path))
    assert out == rows
    # one missing row is the only one trained
    path.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    called = []
    monkeypatch.setattr(scaling, "train", lambda subset, kind, **k: called.append((len(subset), k["seed"])) or 1 / 0)
    with pytest.raises(ZeroDivisionError):
        scaling.run(list(range(30)), "state", [10, 25], 2, "lift", 1, 1, 1, {}, 1,
                    log=lambda *_: None, rows_path=str(path))
    assert called == [(25, 1)]


def _rows(kind):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", f"scaling_{kind}.rows.jsonl"))]
    return {(r["size"], r["seed"]): r["success"] for r in rows}


def test_readme_per_seed_table_matches_saved_rows():
    readme = open(os.path.join(ROOT, "README.md")).read()
    lines = readme.splitlines()
    head = lines.index("| episodes | `image` (0 / 1 / 2) | `act` (0 / 1 / 2 / 3 / 4 / 5) |")
    image, act = _rows("image"), _rows("act")
    expect = [f"| {n} | " + " / ".join(f"{image[n, s]:.2f}" for s in range(3)) + " | "
              + " / ".join(f"{act[n, s]:.2f}" for s in range(6)) + " |" for n in (10, 25, 50, 100, 200)]
    assert lines[head + 2:head + 7] == expect
    # summary row: the 200-episode ACT seeds split into a high and a low group of three
    top = sorted((act[200, s] for s in range(6)), reverse=True)
    split = " / ".join(f"{x:.2f}" for x in top[:3]) + " vs " + " / ".join(f"{x:.2f}" for x in top[3:])
    assert f"seeds {split} |" in readme
