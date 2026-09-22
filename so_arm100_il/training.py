"""The training loop, shared by train_bc.py, dagger.py and scaling.py.

No early stopping and no model selection on validation: the final epoch is
what gets saved and reported. Validation L1 is logged so a reader can see
whether the model overfits, not so the run can pick its best moment.

Throughput (frames/s) is logged every epoch. On a machine that cannot
report CPU temperature, a sustained drop is the only visible sign of
thermal throttling -- or of another process on the box, which looks the
same, so the warning tells the reader to check both.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import FrameDataset, compute_stats, split_episodes
from .models import build, masked_l1, count_params


def train(episodes, kind, epochs=20, batch_size=64, lr=1e-3, seed=0, chunk=20,
          val_frac=0.2, model_kwargs=None, log=print, init_model=None, threads=None):
    """Train `kind` on `episodes`; returns (model, stats, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if threads:
        torch.set_num_threads(int(threads))
    tr_idx, va_idx = split_episodes(len(episodes), val_frac, seed)
    tr = [episodes[i] for i in tr_idx]
    va = [episodes[i] for i in va_idx]
    stats = compute_stats(tr)
    kw = dict(model_kwargs or {})
    if kind == "act":
        kw.setdefault("chunk", chunk)
    model = init_model if init_model is not None else build(kind, **kw)
    model.init_kwargs = kw
    k = model.chunk
    with_img = model.needs_image
    ds_tr = FrameDataset(tr, stats, chunk=k, with_images=with_img)
    ds_va = FrameDataset(va, stats, chunk=k, with_images=with_img) if va else None
    if with_img and ds_tr.image is None:
        raise ValueError(f"model {kind!r} needs images but the dataset has none")
    dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    log(f"  {kind}: {count_params(model):,} params, {len(ds_tr)} train frames from {len(tr)} "
        f"episodes, {len(ds_va) if ds_va else 0} val frames from {len(va)} episodes, chunk {k}")
    history, rate0 = [], None
    for ep in range(1, epochs + 1):
        model.train()
        t0, n, tot = time.perf_counter(), 0, 0.0
        for s, img, a, m in dl:
            pred = model(s, img if with_img else None)
            loss = masked_l1(pred, a, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * s.shape[0]
            n += s.shape[0]
        sched.step()
        rate = n / max(time.perf_counter() - t0, 1e-9)
        va_l1 = evaluate_l1(model, ds_va, with_img) if ds_va else float("nan")
        row = {"epoch": ep, "train_l1": tot / max(n, 1), "val_l1": va_l1, "frames_per_s": rate}
        history.append(row)
        warn = ""
        if rate0 is None and ep >= 1:
            rate0 = rate
        elif rate0 and rate < 0.8 * rate0:
            warn = "   <- throughput down >20%: throttling or another process on the box"
        log(f"  epoch {ep:3d}  train L1 {row['train_l1']:.4f}  val L1 {va_l1:.4f}  "
            f"{rate:7.0f} frames/s{warn}")
    model.eval()
    return model, stats, history


@torch.no_grad()
def evaluate_l1(model, ds, with_img, batch_size=256):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(ds), batch_size):
        s, img, a, m = ds[i:i + batch_size]
        pred = model(s, img if with_img else None)
        tot += masked_l1(pred, a, m).item() * s.shape[0]
        n += s.shape[0]
    return tot / max(n, 1)
