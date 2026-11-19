"""Mask metrics, gating quantities and centroid-trajectory baselines shared by the evaluation scripts."""
import cv2
import numpy as np
from scipy import ndimage


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([ys.mean(), xs.mean()]) if len(ys) else np.array([np.nan, np.nan])


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else 1.0


def hd95(a, b, px):
    if not a.any() or not b.any():
        return np.nan
    ea, eb = a ^ ndimage.binary_erosion(a), b ^ ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~eb)[ea]
    db = ndimage.distance_transform_edt(~ea)[eb]
    return float(np.percentile(np.concatenate([da, db]), 95) * px)


def frac_outside(m, boundary):
    return float((m & ~boundary).sum() / max(m.sum(), 1))


def shift(m, d):
    return ndimage.shift(m.astype(np.float32), d, order=1) > 0.5


def warp(mask, u):
    """Backward-warp a reference mask by field u (H, W, 2) [x, y] in pixels: out(p) = mask(p + u(p))."""
    h, w = mask.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    u = u.astype(np.float32, copy=False)
    return cv2.remap(mask.astype(np.float32), gx + u[..., 0], gy + u[..., 1], cv2.INTER_LINEAR) > 0.5


def nlms_predict(c, k, p, mu):
    """Online NLMS AR(p) on centroid displacements; predicts c[t+k] from c[<=t]. Weights adapt as the
    k-step-delayed truth becomes available (the tracked centroids)."""
    c = np.asarray(c)
    out = np.full((len(c), 2), np.nan)
    w = np.zeros((2, p))
    for t in range(p, len(c)):
        s = t - k
        if s >= p:
            x = c[s - p + 1:s + 1] - c[s]
            y = c[t] - c[s]
            for a in range(2):
                e = y[a] - w[a] @ x[:, a]
                w[a] += mu * e * x[:, a] / (x[:, a] @ x[:, a] + 1e-3)
        x = c[t - p + 1:t + 1] - c[t]
        out[t] = c[t] + np.array([w[a] @ x[:, a] for a in range(2)])
    return out


def row_metrics(pm, gt, px, boundaries, radii):
    """Point metrics + gating fractions for one predicted mask (pm) vs ground truth (gt)."""
    r = dict(cerr_mm=float(np.linalg.norm(centroid(pm) - centroid(gt)) * px), dice=dice(pm, gt),
             hd95_mm=hd95(pm, gt, px))
    dist = ndimage.distance_transform_edt(~pm) * px if pm.any() else None
    for mg, bd in boundaries.items():
        r[f"out_true_{mg:g}"] = frac_outside(gt, bd)
        for rad in radii:
            r[f"out_pred_{mg:g}_r{rad:g}"] = frac_outside(dist <= rad, bd) if dist is not None else 1.0
    return r
