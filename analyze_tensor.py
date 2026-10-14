"""Analyse results_tensor/: which tensor group drives damage at matched norm?

Unit of replication = parent (seed). Outputs tables to <out>/analysis_tensor/.

    python analyze_tensor.py --results results_tensor
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

rng = np.random.default_rng(0)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def boot_ci(x, n=4000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    v = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return tuple(np.quantile(v, [0.025, 0.975]))


def wilcoxon_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 6 or np.allclose(x, 0):
        return np.nan
    return float(stats.wilcoxon(x).pvalue)


def logistic_r50(logr, fail):
    """Radius at which P(fail)=0.5 from a 1-D logistic fit (MLE)."""
    x = (logr - logr.mean())
    def nll(b):
        z = np.clip(b[0] + b[1] * x, -30, 30)
        return np.sum(np.logaddexp(0, z) - fail * z)
    res = optimize.minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    b0, b1 = res.x
    if b1 <= 1e-6:
        return np.nan
    return 10 ** (logr.mean() - b0 / b1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_tensor")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(a.results, "*_seed*.csv"))])
    df["log10_mse_ratio"] = df.log10_mse_ratio.replace(np.inf, np.nan)
    out = os.path.join(a.results, "analysis_tensor")
    os.makedirs(out, exist_ok=True)

    # ---- per-seed aggregates (mean over draws; median for continuous)
    ps = (df.groupby(["system", "design", "group", "r", "seed"])
            .agg(fail=("fail", "mean"), div=("diverged", "mean"),
                 vpt_loss=("vpt_loss", "median"), lmse=("log10_mse_ratio", "median"),
                 r_W3=("r_W3", "median"), rel_norm=("rel_norm", "median"))
            .reset_index())

    # ---- 1. summary table over seeds
    summ = (ps.groupby(["system", "design", "group", "r"])
              .agg(parents=("seed", "nunique"), fail=("fail", "mean"), div=("div", "mean"),
                   vpt_loss=("vpt_loss", "median"), lmse=("lmse", "median"),
                   r_W3=("r_W3", "median")).reset_index())
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)

    # ---- 2. paired comparisons vs isotropic ("all") at the same total norm
    rows = []
    for sysn, d in ps[ps.design.isin(["total", "loo"])].groupby("system"):
        base = d[(d.design == "total") & (d.group == "all")].set_index(["r", "seed"])
        for (design, group), g in d[d.group != "all"].groupby(["design", "group"]):
            for r, gg in g.groupby("r"):
                gg = gg.set_index("seed")
                b = base.loc[r].reindex(gg.index)
                for metric, sign in (("vpt_loss", 1), ("lmse", 1), ("fail", 1)):
                    diff = (gg[metric] - b[metric]).dropna().values
                    lo, hi = boot_ci(diff)
                    rows.append(dict(system=sysn, design=design, group=group, r=r, metric=metric,
                                     n=len(diff), median_diff=float(np.median(diff)),
                                     ci_lo=lo, ci_hi=hi, p=wilcoxon_p(diff)))
    pc = pd.DataFrame(rows)
    pc["p_holm"] = pc.groupby(["system", "metric"])["p"].transform(lambda s: pd.Series(holm(s.values), index=s.index))
    pc.to_csv(os.path.join(out, "paired_vs_isotropic.csv"), index=False)

    # ---- 3. robustness radius (total-norm design) per group, bootstrap over parents
    rr = []
    for sysn, d in df[(df.design == "total")].groupby("system"):
        seeds = d.seed.unique()
        for grp, g in d.groupby("group"):
            def r50(sub):
                return logistic_r50(np.log10(sub.rel_norm.values), sub.fail.values.astype(float))
            est = r50(g)
            by = {s: g[g.seed == s] for s in seeds}
            bs = []
            for _ in range(200):
                pick = rng.choice(seeds, len(seeds))
                bs.append(r50(pd.concat([by[s] for s in pick])))
            bs = np.array(bs)
            rr.append(dict(system=sysn, group=grp, r50=est,
                           lo=np.nanquantile(bs, 0.025), hi=np.nanquantile(bs, 0.975)))
    pd.DataFrame(rr).to_csv(os.path.join(out, "radius_by_group.csv"), index=False)

    # ---- 4. W3 own-displacement needed: failure vs relative displacement of the tensor itself
    own = summ[summ.design == "own"].pivot_table(index=["system", "r"], columns="group", values="fail")
    own.to_csv(os.path.join(out, "own_norm_failure.csv"))

    pd.set_option("display.width", 200)
    print("\n=== RADIUS r50 (total relative norm) by noise group ===")
    print(pd.DataFrame(rr).round(4).to_string(index=False))
    print("\n=== FAILURE RATE, total-norm design ===")
    print(summ[summ.design == "total"].pivot_table(index=["system", "r"], columns="group", values="fail").round(2))
    print("\n=== VPT LOSS (median, Lyapunov times), total-norm design ===")
    print(summ[summ.design == "total"].pivot_table(index=["system", "r"], columns="group", values="vpt_loss").round(2))
    print("\n=== FAILURE RATE, leave-one-out (isotropic with group removed) ===")
    print(summ[summ.design == "loo"].pivot_table(index=["system", "r"], columns="group", values="fail").round(2))
    print("\n=== FAILURE RATE, own-norm design (same relative displacement of that tensor) ===")
    print(own.round(2))
    print("\n=== PAIRED vs isotropic, VPT loss, Holm-adjusted ===")
    q = pc[(pc.metric == "vpt_loss")].copy()
    q["sig"] = np.where(q.p_holm < 0.05, "*", "")
    print(q[["system", "design", "group", "r", "n", "median_diff", "ci_lo", "ci_hi", "p_holm", "sig"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
