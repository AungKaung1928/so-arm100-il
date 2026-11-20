"""ACT lift policy (runs/bc_act_lift.pt) on three CPU backends: PyTorch, ONNX Runtime, OpenVINO.
1 thread, batch 1, image 96x128 as float 0-255 (the model casts and normalizes inside).
Parity over 100 random inputs, then the three backends interleaved call by call.

    python edge/act_backends.py runs/bc_act_lift.pt /tmp/act_edge 2000
Needs torch, onnx, onnxruntime, openvino (versions printed).
"""
import os, sys, time
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
import numpy as np, torch, onnxruntime as ort, openvino as ov
from so_arm100_il.policy import load_checkpoint
from so_arm100_il.data import STATE_DIM
torch.set_num_threads(1)
ckpt, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
model, _, _ = load_checkpoint(ckpt)
H, W = 96, 128
ex = (torch.zeros(1, STATE_DIM), torch.zeros(1, 3, H, W))
path = os.path.join(out, "act.onnx")
torch.onnx.export(model, ex, path, input_names=["state", "image"], output_names=["chunk"], dynamo=False, opset_version=17)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
cm = ov.Core().compile_model(path, "CPU", {"INFERENCE_NUM_THREADS": 1, "INFERENCE_PRECISION_HINT": "f32"})
def t(x):
    with torch.no_grad():
        return model(torch.from_numpy(x[0]), torch.from_numpy(x[1])).numpy()
def o(x): return s.run(None, {"state": x[0], "image": x[1]})[0]
def v(x): return cm([x[0], x[1]])[cm.outputs[0]]
runs = {"pytorch": t, "onnxruntime": o, "openvino": v}
rng = np.random.default_rng(0)
xs = [(rng.standard_normal((1, STATE_DIM)).astype(np.float32), rng.uniform(0, 255, (1, 3, H, W)).astype(np.float32)) for _ in range(100)]
for k in ("onnxruntime", "openvino"):
    print(f"{k}: max |out - pytorch| over 100 inputs {max(np.abs(runs[k](x) - t(x)).max() for x in xs):.2e}")
ts = {k: [] for k in runs}
for k in runs:
    for x in xs[:20]: runs[k](x)
for i in range(N):
    x = xs[i % 100]
    for k in (list(runs) if i % 2 == 0 else list(reversed(list(runs)))):
        t0 = time.perf_counter(); runs[k](x); ts[k].append(time.perf_counter() - t0)
print(f"torch {torch.__version__}, onnxruntime {ort.__version__}, openvino {ov.__version__.split('-')[0]}; "
      f"{sum(p.numel() for p in model.parameters())} params, chunk {model.chunk}")
for k in runs:
    print(f"{k:12s} 1 thread batch 1: p50 {np.median(ts[k])*1e3:.3f} ms p99 {np.percentile(ts[k], 99)*1e3:.3f} ms (n {N})")
