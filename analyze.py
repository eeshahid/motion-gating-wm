"""Point metrics, conformal calibration (official pre-test) and simulated beam gating (official test).

Gating rule at time t for delivery at t + k: beam ON iff the predicted target, dilated by the
conformal radius r_k (centroid-error quantile on the calibration split), has <= tau of its area
outside the boundary (first-frame target + margin). A miss = beam ON while the TRUE target has
> tau outside the boundary.

    python p1_trackrad/analyze.py --run head_full [--tag _drop30]          # DINOv3 latent-WM evaluation
    python p1_trackrad/analyze.py --run eval_motion [--tag _drop30]        # motion-WM evaluation
"""
import argparse

import numpy as np
import pandas as pd

from common import RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--margin", type=float, default=5.0)
args = ap.parse_args()

run = RUNS / args.run
pre = "" if args.run == "eval_motion" else "eval_"
cal = pd.read_pickle(run / f"{pre}pretest{args.tag}.pkl")
test = pd.read_pickle(run / f"{pre}test{args.tag}.pkl")
radii = sorted({float(c.split("_r")[-1]) for c in test.columns if c.startswith(f"out_pred_{args.margin:g}_r")})
pd.set_option("display.width", 200)

print("\n## Point metrics on test (mean over frames)")
pm = test.groupby(["k", "method"]).agg(dt_ms=("dt_ms", "median"), cerr=("cerr_mm", "mean"),
                                       dice=("dice", "mean"), hd95=("hd95_mm", "mean")).round(3)
print(pm.unstack("method").to_string())
print("\n## Centroid error (mm) by centre, k=2")
print(test[test.k == 2].pivot_table(index="centre", columns="method", values="cerr_mm", aggfunc="mean").round(2).to_string())


def conformal_q(scores, alpha):
    s = np.sort(scores[~np.isnan(scores)])
    n = len(s)
    return s[min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)]


rows = []
for k in sorted(test.k.unique()):
    tk = test[test.k == k]
    out_true = tk.groupby(["sid", "t"])[f"out_true_{args.margin:g}"].first()
    oracle_on = out_true <= args.tau
    rows.append(dict(k=k, method="oracle", r_mm=0, coverage=np.nan, duty=oracle_on.mean(), miss=0.0))
    for meth in tk.method.unique():
        if meth in ("oracle_z", "oracle"):
            continue
        c = cal[(cal.k == k) & (cal.method == meth)]
        t = tk[tk.method == meth]
        for conf in (False, True):
            q = conformal_q(c.cerr_mm.values, args.alpha) if conf else 0.0
            r = next((x for x in radii if x >= q), radii[-1])
            on = t[f"out_pred_{args.margin:g}_r{r:g}"] <= args.tau
            miss = (t[f"out_true_{args.margin:g}"] > args.tau)[on]
            rows.append(dict(k=k, method=meth + ("+conf" if conf else ""), r_mm=r,
                             coverage=(t.cerr_mm <= q).mean() if conf else np.nan,
                             duty=on.mean(), miss=miss.mean() if len(miss) else 0.0))
g = pd.DataFrame(rows)
print(f"\n## Gating on test: margin {args.margin} mm, tau {args.tau}, alpha {args.alpha}")
print("duty = beam-on fraction; miss = fraction of beam-on frames with true target outside boundary")
print(g.pivot_table(index="method", columns="k", values=["duty", "miss", "r_mm", "coverage"]).round(3).to_string())
g.to_csv(run / f"gating{args.tag}.csv", index=False)
pm.to_csv(run / f"point_metrics{args.tag}.csv")
