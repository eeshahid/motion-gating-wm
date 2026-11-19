"""Sequential evaluation on full scans (no ROI jitter, no frame drops unless --drop_p).

For every scan and every time t >= K-1 the model sees frames <= t (+ the first-frame label, as in the
TrackRAD protocol) and predicts the target mask at t + k for k in --ks (frames ahead).

Methods
  wm        : ensemble of heads on world-model tokens predicted for t + k        (ours)
  oracle_z  : heads on the TRUE tokens of frame t + k (upper bound for latent prediction)
  hold      : tracker mask at t, used as the prediction for t + k (no latency compensation)
  cv        : tracker mask at t shifted by constant-velocity centroid extrapolation
  nlms      : tracker mask at t shifted by an online NLMS AR(p) centroid predictor
Tracker = the same heads applied to the true tokens of frame t.

Outputs per-frame rows to <run>/eval_<split>.pkl: centroid error (mm), Dice, HD95 (mm),
ensemble spread, the true fraction of target outside each gating boundary (first-frame mask + margin),
and the same fraction for the predicted mask dilated by each radius in --radii (for conformal gating).

    python p1_trackrad/evaluate.py --run pilot_e2e --split pretest test --tag _final
    python p1_trackrad/analyze.py --run pilot_e2e --tag _final
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from common import RUNS, metas_for
from data import NPY, OUT_PX, context_indices, crop, roi_from_mask
from metrics import centroid, dice, frac_outside, hd95, nlms_predict, shift
from models import FrozenEncoder, MaskHead, Predictor

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--split", nargs="+", default=["pretest", "test"])
ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4])
ap.add_argument("--drop_p", type=float, default=0.0, help="drop context frames at test time (irregular sampling)")
ap.add_argument("--margins", nargs="+", type=float, default=[3.0, 5.0], help="gating boundary margins (mm)")
ap.add_argument("--radii", nargs="+", type=float, default=[0, 1, 2, 3, 4, 5, 6, 8, 10],
                help="dilation radii (mm) of the predicted mask, for conformal gating")
ap.add_argument("--nlms_p", type=int, default=4)
ap.add_argument("--nlms_mu", type=float, default=0.5)
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--max_scans", type=int, default=0, help="smoke test: only the first N scans per split")
args = ap.parse_args()
dev = torch.device(args.device)

run = RUNS / args.run
ck = torch.load(run / "head.pt", map_location="cpu")
wa = ck["wm_args"]
K = wa["K"]
enc = FrozenEncoder().to(dev)
pred = Predictor(d=wa["d"], depth=wa["depth"], heads=wa["d"] // 64).to(dev).eval()
pred.load_state_dict(ck["pred"])
heads = torch.nn.ModuleList([MaskHead() for _ in range(ck["args"]["M"])]).to(dev).eval()
heads.load_state_dict(ck["heads"])


@torch.no_grad()
def eval_scan(m, rng):
    d = NPY / m["split"] / m["sid"]
    F = np.load(d / "frames.npy")
    L = np.load(d / "labels.npy")
    T, fps = m["T"], m["fps"]
    if not L[0].any():
        print("no first-frame label, skipping", m["sid"], flush=True)
        return []
    cy, cx, side = roi_from_mask(L[0])
    px = side / OUT_PX
    Fr = crop(F, cy, cx, side)
    Lr = crop(L.astype(np.uint8), cy, cx, side, interp=0).astype(bool)
    z_all = torch.cat([enc(torch.from_numpy(Fr[i:i + 128]).to(dev)) for i in range(0, T, 128)])
    z_ref = z_all[:1]
    ref_mask = torch.from_numpy(Lr[0][None].astype(np.float32)).to(dev)
    keep = np.ones(T, bool)
    if args.drop_p > 0:
        keep[1:] = rng.random(T - 1) > args.drop_p
    boundaries = {mg: ndimage.binary_dilation(Lr[0], iterations=max(1, int(round(mg / px)))) for mg in args.margins}

    def run_heads(z):  # z (B,N,C) -> mean prob (B,P,P), per-head masks (M,B,P,P)
        ps = torch.stack([h(z, z_ref.expand(len(z), -1, -1), ref_mask.expand(len(z), -1, -1))[0].sigmoid() for h in heads])
        return ps.mean(0).cpu().numpy(), (ps > 0.5).cpu().numpy()

    # tracker on every acquired frame; positions forward-filled over dropped frames
    track_p, _ = zip(*[run_heads(z_all[i:i + 64]) for i in range(0, T, 64)])
    track = np.concatenate(track_p) > 0.5
    last_acq = np.maximum.accumulate(np.where(keep, np.arange(T), 0))
    track = track[last_acq]
    c_track = np.stack([centroid(t) for t in track])

    rows = []
    ts = [t for t in range(K - 1, T - 1) if keep[t]]
    for k in args.ks:
        tk = [t for t in ts if t + k < T and Lr[t + k].any()]
        if not tk:
            continue
        # world model context: K acquired frames spread over the last ctx_window_s seconds
        ctx_idx = [context_indices(t, fps, K, wa["ctx_window_s"], avail=keep) for t in tk]
        tk = [t for t, ci in zip(tk, ctx_idx) if ci is not None]
        ctx_idx = [ci for ci in ctx_idx if ci is not None]
        c_nlms = nlms_predict(c_track, k, args.nlms_p, args.nlms_mu)
        for j in range(0, len(tk), 32):
            tb, cb = tk[j:j + 32], np.stack(ctx_idx[j:j + 32])
            z_ctx = z_all[torch.from_numpy(cb).to(dev)]
            tau_ctx = torch.from_numpy((cb - np.array(tb)[:, None]) / fps).float().to(dev)
            tau_tgt = torch.full((len(tb),), k / fps, device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z_hat = pred(z_ctx, tau_ctx, tau_tgt)[:, 0].float()
            p_wm, m_wm = run_heads(z_hat)
            p_or, _ = run_heads(z_all[torch.tensor(tb).to(dev) + k])
            for i, t in enumerate(tb):
                gt = Lr[t + k]
                c_gt = centroid(gt)
                spread = float(np.nanmean([np.linalg.norm(centroid(mm) - centroid(p_wm[i] > 0.5)) for mm in m_wm[:, i]]) * px)
                # constant velocity from the last two acquired positions, scaled to the horizon
                prev = np.nonzero(keep[:t])[0]
                v = (c_track[t] - c_track[prev[-1]]) / (t - prev[-1]) if len(prev) else np.zeros(2)
                preds = {"wm": p_wm[i] > 0.5, "oracle_z": p_or[i] > 0.5, "hold": track[t],
                         "cv": shift(track[t], v * k), "nlms": shift(track[t], c_nlms[t] - c_track[t])}
                for meth, pm in preds.items():
                    r = dict(sid=m["sid"], centre=m["centre"], B=m["B"], region=m["region"], fps=fps, t=t, k=k,
                             dt_ms=1000 * k / fps, method=meth, px_mm=px,
                             cerr_mm=float(np.linalg.norm(centroid(pm) - c_gt) * px), dice=dice(pm, gt),
                             hd95_mm=hd95(pm, gt, px), spread_mm=spread if meth == "wm" else np.nan)
                    dist = ndimage.distance_transform_edt(~pm) * px if pm.any() else None
                    for mg, bd in boundaries.items():
                        r[f"out_true_{mg:g}"] = frac_outside(gt, bd)
                        for rad in args.radii:
                            r[f"out_pred_{mg:g}_r{rad:g}"] = frac_outside(dist <= rad, bd) if dist is not None else 1.0
                    rows.append(r)
    return rows


rng = np.random.default_rng(0)
for split in args.split:
    rows = []
    for m in metas_for(split)[:args.max_scans or None]:
        if m.get("labeled_frames", 0) < K + 2:
            continue
        rows += eval_scan(m, rng)
        print(split, m["sid"], len(rows), flush=True)
    df = pd.DataFrame(rows)
    out = run / f"eval_{split}{args.tag}.pkl"
    df.to_pickle(out)
    print(df.groupby(["k", "method"])[["cerr_mm", "dice", "hd95_mm"]].mean().unstack("method").round(3).to_string())
    print("saved", out)
