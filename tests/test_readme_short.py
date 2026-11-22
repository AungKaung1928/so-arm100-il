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
