"""DETAILS 'ACT on three CPU backends' table and speedup sentence match runs/act_backends.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAILS = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()
RUN = json.load(open(os.path.join(ROOT, "runs", "act_backends.json")))
NAMES = {"pytorch": "PyTorch", "onnxruntime": "ONNX Runtime", "openvino": "OpenVINO"}


def _diff(x):
    return "0" if x == 0 else f"{x:.1e}".replace("e-0", "e-")


def test_backend_rows_match_json():
    lines = set(DETAILS.splitlines())
    for key, name in NAMES.items():
        b = RUN["backends"][key]
        row = f"| {name} | {b['p50_ms']:.3f} / {b['p99_ms']:.3f} ms | {_diff(b['max_abs_diff'])} |"
        assert row in lines, row


def test_speedup_and_setup_match_json():
    b = RUN["backends"]
    ratio = b["pytorch"]["p50_ms"] / b["onnxruntime"]["p50_ms"]
    flat = " ".join(DETAILS.split())
    assert f"ONNX Runtime is {ratio:.1f}x faster than PyTorch" in flat
    assert f"(n = {RUN['n']:,} each)" in flat
    v = RUN["versions"]
    assert f"torch {v['torch'].split('+')[0]}, onnxruntime {v['onnxruntime']}, openvino {v['openvino']}" in flat
