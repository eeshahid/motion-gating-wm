"""TrackRAD2025 preprocessing and datasets.

Raw .mha arrays are (H, W, T) at 1 mm in-plane. We store each scan as a (T, H, W) uint8
.npy (percentile-normalised) plus a meta.json, so training reads frames via memmap.

    python p1_trackrad/data.py preprocess            # convert everything downloaded so far
    python p1_trackrad/data.py preprocess --watch    # keep converting as the download progresses
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

PROJ = Path(__file__).resolve().parents[1]
RAW = PROJ / "data" / "TrackRAD2025"
NPY = PROJ / "data" / "trackrad_npy"
SPLITS = {"labeled_training_data": "train", "labeled_pre-testing_data": "pretest",
          "labeled_testing_data": "test", "unlabeled_training_data": "unlabeled"}
OUT_PX = 256  # ROI side in pixels fed to the encoder (DINOv3 /16 -> 16x16 tokens)


# --------------------------------------------------------------------------- preprocessing
def _read(p):
    return np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(str(p))), -1, 0)  # -> (T, H, W)


def _jload(p):
    return json.loads(p.read_text()) if p.exists() else None


def convert_scan(img_p, split):
    pdir = img_p.parent.parent
    suffix = img_p.stem.split("frames")[-1]            # "" or "2", "3", ...
    sid = pdir.name + (f"_{suffix}" if suffix else "")
    out = NPY / split / sid
    if (out / "meta.json").exists():
        return False
    out.mkdir(parents=True, exist_ok=True)
    x = _read(img_p).astype(np.float32)
    lo, hi = np.percentile(x, [0.5, 99.5])
    np.save(out / "frames.npy", np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))
    meta = dict(sid=sid, pid=pdir.name, centre=pdir.name[0], split=split, T=x.shape[0], H=x.shape[1],
                W=x.shape[2], fps=float(_jload(pdir / f"frame-rate{suffix}.json")),
                B=_jload(pdir / "b-field-strength.json"), region=_jload(pdir / "scanned-region.json"))
    lab_p = pdir / "targets" / f"{pdir.name}_labels{suffix}.mha"
    if lab_p.exists():
        lab = _read(lab_p) > 0
        np.save(out / "labels.npy", lab)
        meta["labeled_frames"] = int(lab.any((1, 2)).sum())
    (out / "meta.json").write_text(json.dumps(meta))
    return True


def preprocess(watch=False):
    while True:
        n = 0
        for raw_split, split in SPLITS.items():
            for img_p in sorted((RAW / f"trackrad2025_{raw_split}").glob("*/images/*.mha")):
                if (img_p.parent / ".incomplete").exists():
                    continue
                try:
                    n += convert_scan(img_p, split)
                except Exception as e:  # partially downloaded file: retry on the next pass
                    print("skip", img_p, e, flush=True)
        print(time.strftime("%H:%M:%S"), "converted", n, "new scans", flush=True)
        if not watch:
            return
        time.sleep(600)


def load_index(splits):
    metas = []
    for s in splits:
        metas += [json.loads(p.read_text()) for p in sorted((NPY / s).glob("*/meta.json"))]
    return metas


# --------------------------------------------------------------------------- ROI handling
def roi_from_mask(mask, motion_margin_mm=24.0):
    """Square ROI (cy, cx, side_mm) around a reference mask; side scales with target size."""
    ys, xs = np.nonzero(mask)
    ext = max(np.ptp(ys), np.ptp(xs)) + 1
    side = float(np.clip(1.5 * ext + 2 * motion_margin_mm, 128, 256))
    return float(ys.mean()), float(xs.mean()), side


def crop(arr, cy, cx, side, out_px=OUT_PX, interp=cv2.INTER_LINEAR):
    """Crop a (side x side) mm window (1 mm px) centred at (cy, cx) from (..., H, W); zero-pad; resize."""
    H, W = arr.shape[-2:]
    s = int(round(side))
    y0, x0 = int(round(cy - s / 2)), int(round(cx - s / 2))
    pad = ((0, 0),) * (arr.ndim - 2) + ((max(0, -y0), max(0, y0 + s - H)), (max(0, -x0), max(0, x0 + s - W)))
    a = np.pad(arr[..., max(0, y0):y0 + s, max(0, x0):x0 + s], pad)
    flat = a.reshape(-1, s, s)
    out = np.stack([cv2.resize(f, (out_px, out_px), interpolation=interp) for f in flat])
    return out.reshape(*arr.shape[:-2], out_px, out_px)


# --------------------------------------------------------------------------- datasets
def context_indices(t, fps, K, window_s, rng=None, avail=None):
    """K frame indices ending at t, spread over the last `window_s` seconds (so the context covers a
    breathing cycle whatever the frame rate). Random subset if rng is given (irregular sampling),
    evenly spaced otherwise. `avail` (bool per frame) restricts to frames actually acquired."""
    idx = np.arange(t) if avail is None else np.nonzero(avail[:t])[0]
    if len(idx) < K - 1:
        return None
    cand = idx[idx >= t - int(round(window_s * fps))]
    if len(cand) < K - 1:
        cand = idx[-(K - 1):]
    if rng is not None:
        sel = np.sort(rng.choice(cand, K - 1, replace=False))
    else:
        sel = cand[np.round(np.linspace(0, len(cand) - 1, K - 1)).astype(int)]
    return np.append(sel, t)


class ClipDataset(Dataset):
    """Context frames + one future frame, cropped to an ROI. Used for world-model and head training.

    Each item: ctx (K, P, P) uint8, tgt (P, P) uint8, tau_ctx (K,) seconds relative to the last
    context frame (<= 0), tau_tgt seconds ahead (> 0). Context frames are a random subset of the last
    `ctx_window_s` seconds, so the predictor always sees irregular sampling.
    """

    def __init__(self, metas, K=8, ctx_window_s=5.0, max_ahead_s=1.5, samples_per_epoch=20000, seed=0,
                 labels=False):
        self.metas = [m for m in metas if m["T"] >= K + 2]
        self.K, self.window, self.max_ahead_s, self.n, self.labels = K, ctx_window_s, max_ahead_s, samples_per_epoch, labels
        w = np.array([m["T"] for m in self.metas], float)
        self.p = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self._frames, self._labels, self._roi = {}, {}, {}

    def __len__(self):
        return self.n

    def frames(self, m):
        if m["sid"] not in self._frames:
            d = NPY / m["split"] / m["sid"]
            self._frames[m["sid"]] = np.load(d / "frames.npy", mmap_mode="r")
            if (d / "labels.npy").exists():
                lab = np.load(d / "labels.npy")
                self._labels[m["sid"]] = lab
                self._roi[m["sid"]] = roi_from_mask(lab[0])
        return self._frames[m["sid"]]

    def sample_roi(self, m, rng):
        if m["sid"] in self._roi:  # labeled: around the first-frame target, with jitter
            cy, cx, side = self._roi[m["sid"]]
            return cy + rng.normal(0, 8), cx + rng.normal(0, 8), side * rng.uniform(0.85, 1.15)
        side = rng.uniform(128, 256)
        return (m["H"] * rng.uniform(0.3, 0.7), m["W"] * rng.uniform(0.3, 0.7), side)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
        while True:  # for labeled data, resample until both t and t + k carry a label
            m = self.metas[rng.choice(len(self.metas), p=self.p)]
            F = self.frames(m)
            T, fps, K = m["T"], m["fps"], self.K
            max_k = max(1, int(round(self.max_ahead_s * fps)))
            t = int(rng.integers(K - 1, T - 1))
            k = int(rng.integers(1, min(max_k, T - 1 - t) + 1))
            if not self.labels or (self._labels[m["sid"]][t].any() and self._labels[m["sid"]][t + k].any()):
                break
        idx = context_indices(t, fps, K, self.window, rng)
        cy, cx, side = self.sample_roi(m, rng)
        clip = crop(np.asarray(F[np.concatenate([idx, [t + k]])]), cy, cx, side)
        flip = rng.random() < 0.5
        if flip:
            clip = clip[..., ::-1]
        out = dict(ctx=torch.from_numpy(clip[:K].copy()), tgt=torch.from_numpy(clip[K].copy()),
                   tau_ctx=torch.tensor((idx - t) / fps, dtype=torch.float32),
                   tau_tgt=torch.tensor(k / fps, dtype=torch.float32))
        if self.labels:
            L = self._labels[m["sid"]]
            lab = crop(L[np.concatenate([[0], [t], [t + k]])].astype(np.uint8), cy, cx, side, interp=cv2.INTER_NEAREST)
            ref = crop(np.asarray(F[[0]]), cy, cx, side)[0]
            if flip:
                lab, ref = lab[..., ::-1], ref[..., ::-1]
            out.update(ref_img=torch.from_numpy(ref.copy()), ref_mask=torch.from_numpy(lab[0].copy()),
                       cur_mask=torch.from_numpy(lab[1].copy()), tgt_mask=torch.from_numpy(lab[2].copy()),
                       px_mm=torch.tensor(side / OUT_PX, dtype=torch.float32))
        return out


if __name__ == "__main__":
    if sys.argv[1] == "preprocess":
        preprocess(watch="--watch" in sys.argv)
