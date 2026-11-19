"""Motion world model: forecast the anatomy's deformation field from its recent (irregularly sampled) past.

State: registration field U_t (frame t -> reference, mm) over a square ROI, resampled to G x G.
Inputs are relative to the current frame (U_ti - U_t) and the target is U_{t+k} - U_t, so the model
only has to learn motion, not the arbitrary phase of the reference frame.

    python p1_trackrad/motion_wm.py --name mwm_full --sources fit unlabeled --scan_list p1_trackrad/mwm_full_scans.json
    python p1_trackrad/motion_wm.py --name mwm_fitonly --sources fit --steps 20000              # 40 labeled scans
    python p1_trackrad/motion_wm.py --name mwm_all --sources fit unlabeled                      # all 2,361 scans
    python p1_trackrad/motion_wm.py --name mwm_discrete --sources fit unlabeled --ctx last --time_unit step
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import RUNS, JsonLog, metas_for, seed_all
from data import NPY, context_indices, roi_from_mask
from flows import FLOW, POOL
from models import Predictor

# one memory-map per scan (2,300+ flow files): lift the soft open-file limit where supported
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except (ImportError, ValueError, OSError):
    pass

G, PATCH = 32, 4


def crop_field(U, cy, cx, side, g=G):
    """U (..., 2, Hp, Wp) on the 4 mm grid; ROI in 1 mm pixel units -> (..., 2, g, g), edge-padded."""
    Hp, Wp = U.shape[-2:]
    s = max(2, int(round(side / POOL)))
    y0, x0 = int(round(cy / POOL - s / 2)), int(round(cx / POOL - s / 2))
    ys = np.clip(np.arange(y0, y0 + s), 0, Hp - 1)
    xs = np.clip(np.arange(x0, x0 + s), 0, Wp - 1)
    a = np.asarray(U[..., ys[:, None], xs[None, :]], np.float32)
    flat = a.reshape(-1, s, s)
    return np.stack([cv2.resize(f, (g, g), interpolation=cv2.INTER_AREA) for f in flat]).reshape(*a.shape[:-2], g, g)


def context_and_time(t, k, fps, K, window_s, ctx="window", time_unit="s", rng=None, avail=None):
    """Context indices and (tau_ctx, tau_tgt) for a forecast of frame t + k made at frame t.

    ctx="window", time_unit="s"  : K frames spread over the last window_s seconds, times in seconds (ours)
    ctx="last",   time_unit="step": the last K acquired frames, times as sequence positions — a standard
                                    discrete-time model that cannot see frame rate or dropped frames (ablation)
    """
    if ctx == "window":
        idx = context_indices(t, fps, K, window_s, rng, avail=avail)
    else:
        a = np.arange(t + 1) if avail is None else np.nonzero(np.append(avail[:t], True))[0]
        idx = a[-K:] if len(a) >= K else None
    if idx is None:
        return None, None, None
    if time_unit == "s":
        return idx, (idx - t) / fps, k / fps
    return idx, np.arange(-K + 1, 1, dtype=np.float64), float(k)


def full_field(Ut, H, W):
    """(2, Hp, Wp) 4 mm field -> (H, W, 2) at 1 mm, edge-padded to the frame size."""
    u = cv2.resize(np.asarray(Ut, np.float32).transpose(1, 2, 0), (Ut.shape[2] * POOL, Ut.shape[1] * POOL))
    return np.pad(u, ((0, H - u.shape[0]), (0, W - u.shape[1]), (0, 0)), mode="edge")


def paste_roi(delta, cy, cx, side, H, W):
    """Predicted ROI change (2, G, G) -> full-frame (H, W, 2) field, zero outside the ROI."""
    s = int(round(side))
    d = cv2.resize(delta.transpose(1, 2, 0), (s, s))
    out = np.zeros((H, W, 2), np.float32)
    y0, x0 = int(round(cy - s / 2)), int(round(cx - s / 2))
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + s), min(W, x0 + s)
    out[ya:yb, xa:xb] = d[ya - y0:yb - y0, xa - x0:xb - x0]
    return out


class FieldDataset(Dataset):
    def __init__(self, metas, K=8, ctx_window_s=5.0, max_ahead_s=1.5, n=100000, seed=0, augment=True,
                 ctx="window", time_unit="s"):
        self.metas = [m for m in metas if (FLOW / m["split"] / m["sid"] / "flow.npy").exists() and m["T"] >= K + 2]
        self.K, self.W, self.max_ahead_s, self.n, self.augment = K, ctx_window_s, max_ahead_s, n, augment
        self.ctx, self.time_unit = ctx, time_unit
        w = np.array([m["T"] for m in self.metas], float)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def __len__(self):
        return self.n

    def load(self, m):
        if m["sid"] not in self._cache:
            d = FLOW / m["split"] / m["sid"]
            roi = None
            lp = NPY / m["split"] / m["sid"] / "labels.npy"
            if lp.exists():
                roi = roi_from_mask(np.load(lp, mmap_mode="r")[0])
            self._cache[m["sid"]] = (np.load(d / "flow.npy", mmap_mode="r"), np.load(d / "ref.npy"), roi)
        return self._cache[m["sid"]]

    def __getitem__(self, i):
        rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
        while True:
            m = self.metas[rng.choice(len(self.metas), p=self.p)]
            U, ref, roi = self.load(m)
            T, fps = m["T"], m["fps"]
            t = int(rng.integers(self.K - 1, T - 1))
            k = int(rng.integers(1, min(max(1, int(round(self.max_ahead_s * fps))), T - 1 - t) + 1))
            idx, tau_c, tau_t = context_and_time(t, k, fps, self.K, self.W, self.ctx, self.time_unit,
                                                 rng if self.ctx == "window" else None, avail=ref == ref[t])
            if idx is not None and ref[t + k] == ref[t] and ref[idx[0]] == ref[t]:
                break
        if roi is not None:
            cy, cx, side = roi[0] + rng.normal(0, 8), roi[1] + rng.normal(0, 8), roi[2] * rng.uniform(0.85, 1.15)
        else:
            side = rng.uniform(128, 256)
            cy, cx = m["H"] * rng.uniform(0.3, 0.7), m["W"] * rng.uniform(0.3, 0.7)
        f = crop_field(U[np.append(idx, t + k)], cy, cx, side)
        f = f - f[self.K - 1]                         # relative to the current frame t
        tau = np.append(tau_c, tau_t)
        if self.augment:
            if rng.random() < 0.5:                    # left-right flip: mirror grid, negate x-displacement
                f = f[..., ::-1].copy()
                f[:, 0] *= -1
            f = f * rng.uniform(0.8, 1.25)            # breathing amplitude
            if self.time_unit == "s":
                tau = tau * rng.uniform(0.8, 1.25)    # breathing rate
        return dict(ctx=torch.from_numpy(f[:self.K].copy()), tgt=torch.from_numpy(f[self.K].copy()),
                    tau_ctx=torch.tensor(tau[:self.K], dtype=torch.float32),
                    tau_tgt=torch.tensor(tau[self.K], dtype=torch.float32))


class MotionWM(nn.Module):
    """Fields (B,K,2,G,G) -> predicted field (B,Q,2,G,G) at the query times, via the shared Predictor."""

    def __init__(self, d=256, depth=6):
        super().__init__()
        self.n_side = G // PATCH
        self.pred = Predictor(in_dim=2 * PATCH * PATCH, d=d, depth=depth, heads=d // 64, n_tok=self.n_side ** 2)

    def tok(self, f):
        B, K = f.shape[:2]
        f = f.view(B, K, 2, self.n_side, PATCH, self.n_side, PATCH)
        return f.permute(0, 1, 3, 5, 2, 4, 6).reshape(B, K, self.n_side ** 2, -1)

    def untok(self, z):
        B, Q = z.shape[:2]
        z = z.view(B, Q, self.n_side, self.n_side, 2, PATCH, PATCH)
        return z.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, Q, 2, G, G)

    def forward(self, ctx, tau_ctx, tau_tgt):
        return self.untok(self.pred(self.tok(ctx), tau_ctx, tau_tgt))


def load_ensemble(name, device="cuda"):
    ck = torch.load(RUNS / name / "mwm.pt", map_location="cpu")
    a = ck["args"]
    models = nn.ModuleList([MotionWM(a["d"], a["depth"]) for _ in range(a["M"])])
    models.load_state_dict(ck["models"])
    return models.to(device).eval(), a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sources", nargs="+", default=["fit", "unlabeled"])
    ap.add_argument("--scan_list", default=None,
                    help="JSON list of scan ids to train on (e.g. p1_trackrad/mwm_full_scans.json for the main model)")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--ctx_window_s", type=float, default=5.0)
    ap.add_argument("--max_ahead_s", type=float, default=1.5)
    ap.add_argument("--ctx", default="window", choices=["window", "last"])
    ap.add_argument("--time_unit", default="s", choices=["s", "step"])
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = JsonLog(run / "log.jsonl")
    metas = [m for s in args.sources for m in metas_for(s)]
    if args.scan_list:
        keep = set(json.loads(open(args.scan_list).read()))
        missing = keep - {m["sid"] for m in metas}
        if missing:
            print(f"warning: {len(missing)} scans in {args.scan_list} not found (not downloaded/preprocessed?)", flush=True)
        metas = [m for m in metas if m["sid"] in keep]
    mode = dict(ctx=args.ctx, time_unit=args.time_unit)
    train_ds = FieldDataset(metas, args.K, args.ctx_window_s, args.max_ahead_s, args.steps * args.bs, args.seed, **mode)
    (run / "scans.json").write_text(json.dumps([m["sid"] for m in train_ds.metas]))
    print(f"{len(train_ds.metas)} scans, {sum(m['T'] for m in train_ds.metas)} frames from {args.sources}", flush=True)
    val_ds = FieldDataset(metas_for("val"), args.K, args.ctx_window_s, args.max_ahead_s, 2048, 123, augment=False, **mode)
    loader = DataLoader(train_ds, args.bs, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True, prefetch_factor=4)
    val_batches = list(DataLoader(val_ds, 128, num_workers=4))

    models = nn.ModuleList([MotionWM(args.d, args.depth) for _ in range(args.M)]).cuda()
    for j, mdl in enumerate(models):          # distinct inits per member
        torch.manual_seed(args.seed * 100 + j)
        for p in mdl.parameters():
            if p.dim() > 1 and p.abs().sum() > 0:
                nn.init.trunc_normal_(p, std=0.02)
    opt = torch.optim.AdamW(models.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"params/member {sum(p.numel() for p in models[0].parameters())/1e6:.2f}M", flush=True)

    @torch.no_grad()
    def validate():
        models.eval()
        err = {"wm": 0.0, "hold": 0.0, "lin": 0.0}
        n = 0
        for b in val_batches:
            ctx, tgt = b["ctx"].cuda(), b["tgt"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = torch.stack([mdl(ctx, b["tau_ctx"].cuda(), b["tau_tgt"].cuda())[:, 0].float() for mdl in models]).mean(0)
            # per-pixel linear extrapolation from the last two context fields (time-aware)
            dt = (b["tau_ctx"][:, -1] - b["tau_ctx"][:, -2]).cuda().clamp(min=1e-3)
            lin = -ctx[:, -2] / dt.view(-1, 1, 1, 1) * b["tau_tgt"].cuda().view(-1, 1, 1, 1)
            epe = lambda x: (x - tgt).norm(dim=1).mean((1, 2))   # endpoint error, mm
            err["wm"] += epe(pred).sum().item(); err["hold"] += epe(torch.zeros_like(tgt)).sum().item()
            err["lin"] += epe(lin).sum().item(); n += len(tgt)
        models.train()
        return {f"val_epe_{k}": v / n for k, v in err.items()}

    t0, step = time.time(), 0
    for b in loader:
        ctx, tgt = b["ctx"].cuda(non_blocking=True), b["tgt"].cuda(non_blocking=True)
        tau_c, tau_t = b["tau_ctx"].cuda(), b["tau_tgt"].cuda()
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for mdl in models:  # one member's graph at a time keeps memory flat in M
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.smooth_l1_loss(mdl(ctx, tau_c, tau_t)[:, 0].float(), tgt) / args.M
            l.backward()
            loss += l.item()
        torch.nn.utils.clip_grad_norm_(models.parameters(), 1.0)
        opt.step(), sched.step()
        step += 1
        if step % 200 == 0:
            log(step=step, loss=loss, lr=sched.get_last_lr()[0], it_s=200 / (time.time() - t0))
            t0 = time.time()
        if step % args.val_every == 0 or step == args.steps:
            log(step=step, **validate())
            torch.save({"models": models.state_dict(), "args": vars(args), "step": step}, run / "mwm.pt")
        if step >= args.steps:
            break
    print("done", flush=True)
