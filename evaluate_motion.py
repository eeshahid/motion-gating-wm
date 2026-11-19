"""Sequential evaluation of motion world models on full labeled scans (TrackRAD protocol: the first-frame
label is given; at time t only frames <= t are available).

Tracking = first-frame mask warped by the registration field U_t. Forecast for t + k:
  mwm:<run> : U_t + ensemble-mean predicted change over the ROI                              (ours)
  hold      : tracked mask at t (no latency compensation)
  lin       : per-pixel linear extrapolation of the field from the last two acquired frames
  nlms      : tracked mask shifted by an online NLMS AR(p) centroid predictor
  oracle    : tracked mask at t + k (what a perfect motion forecast would give)

Rows (one per scan, t, k, method) go to runs/p1/eval_motion/<split><tag>.pkl.

    python p1_trackrad/evaluate_motion.py --runs mwm_full mwm_fitonly mwm_all mwm_discrete --split val pretest test --tag _final
Each run uses its own context mode (window/seconds or last-K/steps) from its training args.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from common import RUNS, metas_for
from data import NPY, roi_from_mask
from flows import FLOW
from metrics import centroid, nlms_predict, row_metrics, shift, warp
from motion_wm import context_and_time, crop_field, full_field, load_ensemble, paste_roi

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--split", nargs="+", default=["pretest", "test"])
ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
ap.add_argument("--drop_p", type=float, default=0.0, help="frames not acquired at test time (irregular sampling)")
ap.add_argument("--margins", nargs="+", type=float, default=[3.0, 5.0])
ap.add_argument("--radii", nargs="+", type=float, default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
ap.add_argument("--nlms_p", type=int, default=4)
ap.add_argument("--nlms_mu", type=float, default=0.5)
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max_scans", type=int, default=0, help="smoke test: only the first N scans per split")
args = ap.parse_args()

dev = torch.device(args.device)
ens = {r: load_ensemble(r, dev) for r in args.runs}


@torch.no_grad()
def eval_scan(m, rng):
    L = np.load(NPY / m["split"] / m["sid"] / "labels.npy")
    U = np.load(FLOW / m["split"] / m["sid"] / "flow.npy")
    ref = np.load(FLOW / m["split"] / m["sid"] / "ref.npy")
    T, H, W = L.shape
    fps = m["fps"]
    if not L[0].any():
        return []
    cy, cx, side = roi_from_mask(L[0])
    # metric window: ROI + 20 mm, so all masks are compared on the same crop (1 mm pixels)
    s2 = int(round(side / 2 + 20))
    win = (slice(max(0, int(cy) - s2), int(cy) + s2), slice(max(0, int(cx) - s2), int(cx) + s2))
    boundaries = {mg: ndimage.binary_dilation(L[0], iterations=int(round(mg)))[win] for mg in args.margins}
    keep = np.ones(T, bool)
    if args.drop_p > 0:
        keep[1:] = rng.random(T - 1) > args.drop_p
    last_acq = np.maximum.accumulate(np.where(keep, np.arange(T), 0))
    Uf = [full_field(U[t], H, W) for t in range(T)]
    track = np.stack([warp(L[0], Uf[t]) for t in range(T)])[last_acq]
    c_track = np.stack([centroid(x) for x in track])
    rows = []
    for k in args.ks:
        c_nlms = nlms_predict(c_track, k, args.nlms_p, args.nlms_mu)
        ts = [t for t in range(1, T - k) if keep[t] and L[t + k].any()]
        # every method is scored on the same frames: those where all runs have a full context
        items = []
        for t in ts:
            per = {}
            for name, (_, a) in ens.items():
                c = context_and_time(t, k, fps, a["K"], a["ctx_window_s"], a.get("ctx", "window"),
                                     a.get("time_unit", "s"), avail=keep & (ref == ref[t]))
                if c[0] is None:
                    break
                per[name] = c
            else:
                items.append((t, per))
        if not items:
            continue
        preds = {}
        for name, (models, a) in ens.items():
            ctx = np.stack([crop_field(U[per[name][0]], cy, cx, side) - crop_field(U[t], cy, cx, side) for t, per in items])
            x = torch.from_numpy(ctx).float().to(dev)
            tc = torch.from_numpy(np.stack([per[name][1] for _, per in items])).float().to(dev)
            tt = torch.tensor([per[name][2] for _, per in items], dtype=torch.float32, device=dev)
            out = []
            for j in range(0, len(items), 256):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    out.append(torch.stack([mdl(x[j:j + 256], tc[j:j + 256], tt[j:j + 256])[:, 0].float() for mdl in models]))
            preds[name] = torch.cat(out, 1).cpu().numpy()  # (M, n, 2, G, G)
        for i, (t, _) in enumerate(items):
            gt = L[t + k][win]
            prev = np.nonzero(keep[:t])[0]
            lin_u = Uf[t] + (Uf[t] - Uf[prev[-1]]) * (k / (t - prev[-1])) if len(prev) else Uf[t]
            masks = {"hold": track[t], "lin": warp(L[0], lin_u),
                     "nlms": shift(track[t], c_nlms[t] - c_track[t]), "oracle": warp(L[0], Uf[t + k])}
            spreads = {}
            for name, P in preds.items():
                mean_mask = warp(L[0], Uf[t] + paste_roi(P[:, i].mean(0), cy, cx, side, H, W))
                cm = centroid(mean_mask[win])
                spreads[name] = float(np.nanmean([np.linalg.norm(centroid(warp(L[0], Uf[t] + paste_roi(P[j, i], cy, cx, side, H, W))[win]) - cm)
                                                  for j in range(len(P))]))
                masks[f"mwm:{name}"] = mean_mask
            for meth, pm in masks.items():
                r = dict(sid=m["sid"], centre=m["centre"], B=m["B"], region=m["region"], fps=fps, t=t, k=k,
                         dt_ms=1000 * k / fps, method=meth, px_mm=1.0, spread_mm=spreads.get(meth.split(":")[-1], np.nan)
                         if meth.startswith("mwm:") else np.nan)
                r.update(row_metrics(pm[win], gt, 1.0, boundaries, args.radii))
                rows.append(r)
    return rows


out_dir = RUNS / "eval_motion"
out_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for split in args.split:
    rows = []
    for m in metas_for(split)[:args.max_scans or None]:
        rows += eval_scan(m, rng)
        print(split, m["sid"], len(rows), flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(out_dir / f"{split}{args.tag}.pkl")
    print(df.groupby(["k", "method"])[["cerr_mm", "dice", "hd95_mm"]].mean().unstack("method").round(3).to_string())
