"""Label-free motion state: dense DIS optical flow from a reference frame to every frame.

For frame t the field u_t maps frame-t pixels to reference-frame pixels (backward warp), so the
reference target mask warped by u_t is the tracked mask at t. References are the first frame of each
`chunk_s`-second chunk (labeled scans are shorter than one chunk, so their reference is frame 0,
the frame with the given label). Flow is computed at 1 mm and stored 4x average-pooled (4 mm grid)
as float16 millimetres: <NPY>/../trackrad_flow/<split>/<sid>/flow.npy (T, 2, H/4, W/4), ref.npy (T,).

    python p1_trackrad/flows.py [--watch]
"""
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from data import NPY, load_index

FLOW = NPY.parent / "trackrad_flow"
POOL = 4
CHUNK_S = 60.0


def dis():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def ref_indices(T, fps, chunk_s=CHUNK_S):
    n = max(1, int(round(chunk_s * fps)))
    return (np.arange(T) // n) * n


def scan_flow(m):
    out = FLOW / m["split"] / m["sid"]
    if (out / "flow.npy").exists():
        return 0
    cv2.setNumThreads(1)
    F = np.load(NPY / m["split"] / m["sid"] / "frames.npy", mmap_mode="r")
    T, H, W = F.shape
    Hp, Wp = H // POOL, W // POOL
    ref = ref_indices(T, m["fps"])
    D = dis()
    flow = np.zeros((T, 2, Hp, Wp), np.float16)
    for t in range(T):
        if t == ref[t]:
            continue
        f = D.calc(np.ascontiguousarray(F[t]), np.ascontiguousarray(F[ref[t]]), None)  # (H, W, 2) px = mm
        flow[t] = f[:Hp * POOL, :Wp * POOL].reshape(Hp, POOL, Wp, POOL, 2).mean((1, 3)).transpose(2, 0, 1)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ref.npy", ref)
    np.save(out / "flow.tmp.npy", flow)
    (out / "flow.tmp.npy").rename(out / "flow.npy")
    return T


if __name__ == "__main__":
    while True:
        metas = [m for m in load_index(["train", "pretest", "test", "unlabeled"])
                 if not (FLOW / m["split"] / m["sid"] / "flow.npy").exists()]
        t0, n = time.time(), 0
        with Pool(12) as pool:
            for T in pool.imap_unordered(scan_flow, metas):
                n += T
        print(time.strftime("%H:%M:%S"), f"{len(metas)} scans, {n} frames, {n / max(time.time() - t0, 1):.0f} fr/s", flush=True)
        if "--watch" not in sys.argv:
            break
        time.sleep(600)
