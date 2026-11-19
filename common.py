import json
import random
from pathlib import Path

import numpy as np
import torch

from data import NPY, load_index

PROJ = Path(__file__).resolve().parents[1]
RUNS = PROJ / "runs" / "p1"
SPLIT_FILE = PROJ / "p1_trackrad" / "splits.json"


def labeled_splits():
    """Patient-level split of the official labeled training set: 40 fit / 10 val (every 5th per centre).
    Official pre-test (8) = conformal calibration; official test (30) = held-out evaluation."""
    if not SPLIT_FILE.exists():
        pids = sorted({m["pid"] for m in load_index(["train"])})
        val = [p for c in sorted({p[0] for p in pids}) for p in [q for q in pids if q[0] == c][2::5]]
        SPLIT_FILE.write_text(json.dumps({"fit": [p for p in pids if p not in val], "val": val}, indent=1))
    return json.loads(SPLIT_FILE.read_text())


def metas_for(name):
    s = labeled_splits()
    if name in ("fit", "val"):
        return [m for m in load_index(["train"]) if m["pid"] in s[name]]
    return load_index([name])


def seed_all(seed):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)


class JsonLog:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, **kw):
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)
