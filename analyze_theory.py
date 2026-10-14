"""Test the first-order sensitivity theory (run_theory.py).

A. Accuracy: predicted vs measured one-step MSE increase.
B. Collapse: does failure depend on the tensor/design only through the predicted MSE increase?
   (AUC of predicted dMSE vs total relative norm vs own-relative displacement of the noised set)
C. Out-of-sample radii: a single critical error ratio c per system is fitted on the total-norm and
   leave-one-out designs ONLY; it then predicts the own-norm radius of every tensor
   (r_G = sqrt(c * MSE_parent / s_G)), compared with the measured own-norm radii.

    python analyze_theory.py --theory results_theory --out paper
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

from analyze_tensor import logistic_r50

rng = np.random.default_rng(3)
GROUPS = {"W1": ("W1",), "W2": ("W2",), "W3": ("W3",), "biases": ("b1", "b2", "b3")}
SYS = ["lorenz", "rossler", "lorenz96"]
NAME = {"lorenz": "Lorenz-63", "rossler": "Rössler", "lorenz96": "Lorenz-96"}


def auc(score, y):
    """Mann-Whitney AUC; ties handled by average ranks."""
    y = np.asarray(y, bool)
    r = stats.rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def logit_x50(x, y):
    """x at which a 1-D logistic fit crosses 0.5 (x already log-transformed)."""
    xm = x.mean()
    z = x - xm
    nll = lambda b: np.sum(np.logaddexp(0, np.clip(b[0] + b[1] * z, -30, 30)) - y * np.clip(b[0] + b[1] * z, -30, 30))
    b0, b1 = optimize.minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    return xm - b0 / b1, b1


def load(theory):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(theory, "*_seed*.csv")))], ignore_index=True)
    df["lpred"] = np.log10(df.dmse_pred / df.mse_parent)          # log10 predicted increase / parent MSE
    df["lmeas"] = np.log10(df.dmse_meas.where(df.dmse_meas > 0) / df.mse_parent)
    keys = ("W1", "b1", "W2", "b2", "W3", "b3")
    def setnorm(row_keys, d):
        return np.sqrt(sum(d[f"norm_{k}"] ** 2 for k in row_keys))
    # norm of the noised set (for the 'own-relative displacement of the noised set' baseline)
    ns = np.zeros(len(df))
    for g in df.group.unique():
        if g == "all":
            ks = keys
        elif g.startswith("all_but_"):
            ks = tuple(k for k in keys if k not in GROUPS[g[8:]])
        else:
            ks = GROUPS[g]
        m = (df.group == g).values
        ns[m] = setnorm(ks, df[m])
    df["own_rel"] = df.noise_norm / ns
    df["tot_rel"] = df.noise_norm / df.norm_total
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", default="results_theory")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df = load(a.theory)
    systems = [s for s in SYS if s in set(df.system)]
    out = {"parents": {s: int(df[df.system == s].seed.nunique()) for s in systems}}

    # ---------------- A. accuracy of the predicted MSE increase
    acc = {}
    for s in systems:
        d = df[(df.system == s) & np.isfinite(df.lmeas) & np.isfinite(df.lpred)]
        for name, sel in (("small", d.lpred < 0), ("moderate", (d.lpred >= 0) & (d.lpred < 1)), ("large", d.lpred >= 1)):
            q = d[sel]
            if len(q) < 30:
                continue
            slope = stats.linregress(q.lpred, q.lmeas)
            acc[f"{s}|{name}"] = dict(
                n=int(len(q)), med_log10_ratio=float(np.median(q.lmeas - q.lpred)),
                spearman=float(stats.spearmanr(q.lpred, q.lmeas)[0]), slope=float(slope.slope), r2=float(slope.rvalue ** 2))
        # pooled over all models with predicted increase below 10x parent MSE
        q = d[d.lpred < 1]
        acc[f"{s}|le10"] = dict(n=int(len(q)), med_factor=float(10 ** np.median(q.lmeas - q.lpred)),
                                p10=float(10 ** np.quantile(q.lmeas - q.lpred, .1)),
                                p90=float(10 ** np.quantile(q.lmeas - q.lpred, .9)),
                                r2=float(stats.linregress(q.lpred, q.lmeas).rvalue ** 2))
    out["accuracy"] = acc

    # ---------------- B. failure vs predicted increase: AUCs (pooled and within parent)
    aucs = {}
    for s in systems:
        d = df[df.system == s]
        y = d.fail.values.astype(bool)
        row = {}
        for name, col in (("predicted_dMSE", "lpred"), ("total_rel_norm", "tot_rel"), ("own_rel_noised_set", "own_rel")):
            sc = d[col].values if col == "lpred" else np.log10(d[col].values)
            row[name] = auc(sc, y)
            row[name + "_within"] = float(np.nanmean([auc((g[col].values if col == "lpred" else np.log10(g[col].values)), g.fail.values.astype(bool))
                                                      for _, g in d.groupby("seed")]))
        aucs[s] = row
    out["auc"] = aucs

    # ---------------- C. out-of-sample radii
    # critical error ratio fitted on total + loo designs only
    meas_r, pred_r, rows = {}, {}, []
    crit = {}
    for s in systems:
        d = df[(df.system == s)]
        fit = d[d.design.isin(["total", "loo"]) & np.isfinite(d.lpred)]
        x50, slope = logit_x50(fit.lpred.values, fit.fail.values.astype(float))
        crit[s] = dict(log10_ratio=float(x50), ratio=float(10 ** x50), slope=float(slope))
        own = d[d.design == "own"]
        for g, keys in GROUPS.items():
            meas = logistic_r50(np.log10(own[own.group == g].r.values), own[own.group == g].fail.values.astype(float))
            # per-parent predicted radius, then median over parents
            per = []
            for sd, p in d[(d.group == g) & (d.design == "own")].groupby("seed"):
                p0 = p.iloc[0]
                nk = sum(p0[f"n_{k}"] for k in keys)
                nrm = np.sqrt(sum(p0[f"norm_{k}"] ** 2 for k in keys))
                sk = nrm ** 2 / nk * sum(p0[f"S_{k}"] for k in keys)
                per.append(np.sqrt(10 ** x50 * p0.mse_parent / sk))
            pr = float(np.median(per))
            rows.append(dict(system=s, group=g, measured=float(meas), predicted=pr, ratio=pr / float(meas)))
    rad = pd.DataFrame(rows)
    for s in systems:
        crit[s]["parent_mse_median"] = float(df[df.system == s].groupby("seed").mse_parent.first().median())
        crit[s]["rms_critical"] = float(np.sqrt(crit[s]["ratio"] * crit[s]["parent_mse_median"]))
    out["critical"] = crit
    out["radii"] = rows
    lg = np.log10(rad[["measured", "predicted"]].values)
    out["radii_summary"] = dict(
        n=int(len(rad)), pearson_log=float(np.corrcoef(lg[:, 0], lg[:, 1])[0, 1]),
        spearman=float(stats.spearmanr(rad.measured, rad.predicted)[0]),
        median_abs_factor=float(10 ** np.median(np.abs(lg[:, 1] - lg[:, 0]))),
        within_1p5=int((np.abs(lg[:, 1] - lg[:, 0]) < np.log10(1.5)).sum()),
        within_2=int((np.abs(lg[:, 1] - lg[:, 0]) < np.log10(2)).sum()))
    # sensitivity of the radius prediction to the designs used to fit c
    sens = {}
    for fitname, designs in (("total", ["total"]), ("loo", ["loo"]), ("total+loo", ["total", "loo"]), ("all_incl_own", ["total", "loo", "own"])):
        within, cc = 0, {}
        for s in systems:
            d = df[df.system == s]
            fit = d[d.design.isin(designs) & np.isfinite(d.lpred)]
            x50, _ = logit_x50(fit.lpred.values, fit.fail.values.astype(float))
            cc[s] = float(10 ** x50)
            own = d[d.design == "own"]
            for g, keys in GROUPS.items():
                og = own[own.group == g]
                meas = logistic_r50(np.log10(og.r.values), og.fail.values.astype(float))
                per = []
                for sd, p in og.groupby("seed"):
                    p0 = p.iloc[0]
                    nk = sum(p0[f"n_{k}"] for k in keys)
                    nrm = np.sqrt(sum(p0[f"norm_{k}"] ** 2 for k in keys))
                    sk = nrm ** 2 / nk * sum(p0[f"S_{k}"] for k in keys)
                    per.append(np.sqrt(10 ** x50 * p0.mse_parent / sk))
                within += int(0.5 < np.median(per) / meas < 2.0)
        sens[fitname] = dict(within_2=within, n=4 * len(systems), c=cc)
    out["fit_sensitivity"] = sens
    rad.to_csv(os.path.join(a.out, "theory_radii.csv"), index=False)
    with open(os.path.join(a.out, "tables", "tab_theory_radii.tex"), "w") as f:
        lab = {"W1": "$W_1$", "W2": "$W_2$", "W3": "$W_3$", "biases": "biases"}
        for i, r in rad.iterrows():
            nm = NAME[r.system].replace("ö", "\\\"o")
            f.write(f"{nm if r.group == 'W1' else ''} & {lab[r.group]} & {r.measured:.3f} & {r.predicted:.3f} & {r.ratio:.2f}"
                    + ("\\\\\n" if i < len(rad) - 1 else "\n"))
    json.dump(out, open(os.path.join(a.out, "theory_numbers.json"), "w"), indent=1, default=float)

    # ---------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.1))
    col = {"lorenz": "tab:blue", "rossler": "tab:orange", "lorenz96": "tab:green"}
    for s in systems:
        d = df[(df.system == s) & np.isfinite(df.lmeas) & np.isfinite(df.lpred)]
        d = d.sample(min(len(d), 4000), random_state=0)
        ax[0].scatter(d.lpred, d.lmeas, s=2, alpha=0.25, color=col[s], label=NAME[s])
    lim = [-3, 4]
    ax[0].plot(lim, lim, "k-", lw=1)
    ax[0].set_xlim(-3, 3.2); ax[0].set_ylim(-3, 4.5)
    ax[0].set_xlabel("predicted $\\log_{10}(\\Delta\\mathrm{MSE}/\\mathrm{MSE}_{\\mathrm{parent}})$")
    ax[0].set_ylabel("measured")
    ax[0].legend(markerscale=4, fontsize=8)
    ax[0].set_title("(a) first-order prediction of one-step error", fontsize=9)
    # (b) radii
    for s in systems:
        r = rad[rad.system == s]
        ax[1].scatter(r.measured, r.predicted, color=col[s], label=NAME[s], s=40)
        for _, q in r.iterrows():
            ax[1].annotate(q.group.replace("biases", "b"), (q.measured, q.predicted), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo, hi = 0.03, 1.0
    xs = np.array([lo, hi])
    ax[1].plot(xs, xs, "k-", lw=1)
    ax[1].fill_between(xs, xs / 2, xs * 2, color="gray", alpha=0.15, lw=0)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("measured own-norm radius $r_{50}$"); ax[1].set_ylabel("predicted (error ratio fitted on other designs)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("(b) out-of-sample per-tensor radii (band: factor 2)", fontsize=9)
    # (c) collapse
    bins = np.arange(-3, 4.01, 0.5)
    for s in systems:
        d = df[(df.system == s) & np.isfinite(df.lpred)]
        dd = d.assign(b=pd.cut(d.lpred, bins))
        t = dd.groupby("b", observed=True).fail.mean()
        mid = [iv.mid for iv in t.index]
        ax[2].plot(mid, t.values, "-o", ms=3, color=col[s], label=NAME[s])
    ax[2].set_xlabel("predicted $\\log_{10}(\\Delta\\mathrm{MSE}/\\mathrm{MSE}_{\\mathrm{parent}})$")
    ax[2].set_ylabel("fraction of models that fail")
    ax[2].legend(fontsize=8)
    ax[2].set_title("(c) all tensors, designs and norms pooled", fontsize=9)
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_theory.pdf"))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
