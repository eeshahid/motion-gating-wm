"""Statistics, conformal calibration and beam gating from the saved evaluation rows (CPU only).

Outputs (runs/p1/results/):
  main{tag}.csv         test metrics per horizon/method: scan-level mean with 95% bootstrap CI (resampling scans)
  paired{tag}.csv       per-scan paired differences (method A - method B), bootstrap CI, Wilcoxon p
  gap{tag}.csv          share of the forecastable gap closed: (hold - m) / (hold - oracle)
  centre{tag}.csv       centroid error per centre (X = centre unseen in training)
  conformal{tag}.csv    coverage / radius / gating for each calibration variant (alpha = --alpha)
  sweep{tag}.csv        gating duty vs miss over alpha (trade-off curve)
  results{tag}.md       everything above as markdown

Conformal variants: calibration set (pretest | pretest+val), score (abs = centroid error;
norm = error / (ensemble spread + c)), and Mondrian groups (none | field strength).

    python p1_trackrad/results.py --tag _final [--ours mwm:mwm_full]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--ours", default="mwm:mwm_full", help="method name of the main model")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--tau", type=float, default=0.05)
ap.add_argument("--margin", type=float, default=5.0)
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()

EV = RUNS / "eval_motion"
OUT = RUNS / "results"
OUT.mkdir(parents=True, exist_ok=True)
load = lambda s: pd.read_pickle(EV / f"{s}{args.tag}.pkl") if (EV / f"{s}{args.tag}.pkl").exists() else None
test, pre, val = load("test"), load("pretest"), load("val")
methods = list(test.method.unique())
ours = args.ours
rng = np.random.default_rng(0)
md = [f"# Results{args.tag}  (main model: {ours})\n"]


def boot_ci(per_scan):
    v = np.asarray(per_scan, float)
    b = rng.choice(v, (args.n_boot, len(v))).mean(1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(m, lo, hi, d=2):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ---------------------------------------------------------------- main table (scan-level CIs)
per_scan = test.groupby(["k", "method", "sid"])[["cerr_mm", "dice", "hd95_mm", "dt_ms"]].mean().reset_index()
rows = []
for (k, meth), g in per_scan.groupby(["k", "method"]):
    r = dict(k=k, dt_ms=round(g.dt_ms.median()), method=meth, n_scans=len(g))
    for col, d in (("cerr_mm", 2), ("dice", 3), ("hd95_mm", 2)):
        m, lo, hi = boot_ci(g[col])
        r[col] = m
        r[col + "_ci"] = fmt(m, lo, hi, d)
    rows.append(r)
main = pd.DataFrame(rows)
main.to_csv(OUT / f"main{args.tag}.csv", index=False)
md += ["## Test set (30 scans): scan-level mean [95% bootstrap CI]\n"]
for col in ("cerr_mm", "dice", "hd95_mm"):
    md += [f"### {col}\n", main.pivot(index=["k", "dt_ms"], columns="method", values=col + "_ci").to_markdown(), ""]

# ---------------------------------------------------------------- paired comparisons
comparisons = [(ours, "hold"), (ours, "lin"), (ours, "nlms")]
comparisons += [(ours, m) for m in methods if m.startswith("mwm:") and m != ours]
rows = []
for k in sorted(test.k.unique()):
    p = per_scan[per_scan.k == k].pivot(index="sid", columns="method", values="cerr_mm")
    for a, b in comparisons:
        d = (p[a] - p[b]).dropna()
        m, lo, hi = boot_ci(d)
        rows.append(dict(k=k, A=a, B=b, diff_cerr_mm=fmt(m, lo, hi), wilcoxon_p=wilcoxon(d).pvalue,
                         A_better_scans=f"{(d < 0).sum()}/{len(d)}"))
paired = pd.DataFrame(rows)
paired.to_csv(OUT / f"paired{args.tag}.csv", index=False)
md += ["## Paired per-scan differences in centroid error (A - B; negative = A better)\n",
       paired.to_markdown(index=False, floatfmt=".2g"), ""]

# ---------------------------------------------------------------- forecastable gap closed
rows = []
for k in sorted(test.k.unique()):
    p = per_scan[per_scan.k == k].pivot(index="sid", columns="method", values="cerr_mm").mean()
    for m in methods:
        if m.startswith("mwm:") or m in ("lin", "nlms"):
            rows.append(dict(k=k, method=m, gap_closed=(p["hold"] - p[m]) / (p["hold"] - p["oracle"])))
gap = pd.DataFrame(rows).pivot(index="k", columns="method", values="gap_closed")
gap.to_csv(OUT / f"gap{args.tag}.csv")
md += ["## Share of the forecastable gap closed, (hold - m) / (hold - oracle)\n", gap.to_markdown(floatfmt=".2f"), ""]

# ---------------------------------------------------------------- per centre
cen = test.groupby(["k", "centre", "method"]).cerr_mm.mean().unstack("method")
cen.to_csv(OUT / f"centre{args.tag}.csv")
md += ["## Centroid error (mm) per centre (X unseen in training)\n", cen.to_markdown(floatfmt=".2f"), ""]


# ---------------------------------------------------------------- conformal calibration + gating
def radii_cols(df):
    return sorted({float(c.split("_r")[-1]) for c in df.columns if c.startswith(f"out_pred_{args.margin:g}_r")})


RADII = radii_cols(test)


def conformal_q(s, alpha):
    s = np.sort(np.asarray(s)[~np.isnan(s)])
    n = len(s)
    j = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return np.inf if j >= n else s[j]


def gate(t, radius):
    """Beam ON iff the predicted mask dilated by the (per-row) radius has <= tau outside the boundary.
    Radii are rounded up to the evaluated grid; beyond the largest radius the beam stays off."""
    out = np.ones(len(t))
    rad = np.asarray(radius, float)
    for r in RADII[::-1]:
        sel = rad <= r
        out[sel] = t[f"out_pred_{args.margin:g}_r{r:g}"].values[sel]
    on = out <= args.tau
    miss = (t[f"out_true_{args.margin:g}"].values > args.tau) & on
    return on.mean(), miss.sum() / max(on.sum(), 1)


def calibrate(cal, t, score, mondrian, alpha):
    """Per-row radius on the test rows t from calibration rows cal (same method and horizon)."""
    c0 = np.nanmedian(cal.spread_mm) if score == "norm" else 0.0
    s = cal.cerr_mm / (cal.spread_mm + c0) if score == "norm" else cal.cerr_mm
    scale = (t.spread_mm + c0) if score == "norm" else pd.Series(1.0, index=t.index)
    radius = pd.Series(np.inf, index=t.index)
    groups = ["B"] if mondrian == "B" else []
    if not groups:
        radius[:] = conformal_q(s.values, alpha) * scale
    else:
        for gval, tg in t.groupby(groups):
            cg = cal[cal[groups[0]] == gval]
            q = conformal_q((s[cg.index]).values, alpha) if len(cg) else np.inf
            radius[tg.index] = q * scale[tg.index]
    return radius


def coverage_stats(t, radius):
    cov = t.cerr_mm <= radius
    per = cov.groupby(t.sid).mean()
    return cov.mean(), per.mean(), per.min()


variants = [("pretest", "abs", "none")]
if val is not None:
    variants += [("pretest+val", "abs", "none"), ("pretest+val", "abs", "B")]
variants += [(c, "norm", mo) for c, _, mo in list(variants)]
cal_sets = {"pretest": pre, "pretest+val": pd.concat([pre, val], ignore_index=True) if val is not None else None}

rows = []
for k in sorted(test.k.unique()):
    tk = test[test.k == k]
    oracle_on = (tk.groupby(["sid", "t"])[f"out_true_{args.margin:g}"].first() <= args.tau).mean()
    rows.append(dict(k=k, method="perfect knowledge", calib="-", score="-", mondrian="-", duty=oracle_on, miss=0.0))
    for meth in [m for m in methods if m != "oracle"]:
        t = tk[tk.method == meth].reset_index(drop=True)
        d, mi = gate(t, np.zeros(len(t)))
        rows.append(dict(k=k, method=meth, calib="none", score="-", mondrian="-", duty=d, miss=mi))
        for cal_name, score, mondrian in variants:
            if score == "norm" and not meth.startswith("mwm:"):
                continue
            cal = cal_sets[cal_name]
            c = cal[(cal.k == k) & (cal.method == meth)].reset_index(drop=True)
            rad = calibrate(c, t, score, mondrian, args.alpha)
            d, mi = gate(t, rad)
            cv, cv_scan, cv_min = coverage_stats(t, rad)
            rows.append(dict(k=k, method=meth, calib=cal_name, score=score, mondrian=mondrian, coverage=cv,
                             scan_cov_mean=cv_scan, scan_cov_min=cv_min, radius_mm=np.median(rad[np.isfinite(rad)]) if np.isfinite(rad).any() else np.inf,
                             duty=d, miss=mi))
conf = pd.DataFrame(rows)
conf.to_csv(OUT / f"conformal{args.tag}.csv", index=False)
md += [f"## Conformal calibration and gating (alpha {args.alpha}, margin {args.margin} mm, tau {args.tau})\n",
       "coverage = test frames with centroid error <= radius (target %.2f); duty = beam-on fraction; "
       "miss = beam-on frames with true target outside the boundary\n" % (1 - args.alpha),
       conf.round(3).to_markdown(index=False), ""]

# ---------------------------------------------------------------- alpha sweep (duty vs miss trade-off)
best = ("pretest+val" if val is not None else "pretest")
rows = []
for k in sorted(test.k.unique()):
    tk = test[test.k == k]
    cal = cal_sets[best]
    for meth in [m for m in methods if m != "oracle"]:
        t = tk[tk.method == meth].reset_index(drop=True)
        c = cal[(cal.k == k) & (cal.method == meth)].reset_index(drop=True)
        for alpha in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
            for score in (("abs", "norm") if meth.startswith("mwm:") else ("abs",)):
                d, mi = gate(t, calibrate(c, t, score, "none", alpha))
                rows.append(dict(k=k, method=meth, score=score, alpha=alpha, duty=d, miss=mi))
pd.DataFrame(rows).to_csv(OUT / f"sweep{args.tag}.csv", index=False)

(OUT / f"results{args.tag}.md").write_text("\n".join(md))
print("\n".join(md))
print("written to", OUT)
