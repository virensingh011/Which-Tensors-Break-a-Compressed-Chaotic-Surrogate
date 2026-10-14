"""Analyse results_mixed/: theory-guided vs uniform vs random bit allocations at equal bit budget.

Unit of replication = parent. Per (system, budget): failure rate and VPT loss per scheme, paired per-parent
differences (theory - uniform, theory - random-mean) with bootstrap CI, Wilcoxon p and Holm adjustment;
and AUC of the predicted error increase for predicting failure across allocations.

    python analyze_mixed.py --mixed results_mixed --out paper
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
from scipy import stats

from analyze_tensor import boot_ci, holm, wilcoxon_p

SYS = ["lorenz", "rossler", "lorenz96"]
NAME = {"lorenz": "Lorenz-63", "rossler": "Rössler", "lorenz96": "Lorenz-96"}


def auc(score, y):
    y = np.asarray(y, bool)
    r = stats.rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", default="results_mixed")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.mixed, "*_seed*.csv")))], ignore_index=True)
    df["fail"] = df.fail.astype(float)
    systems = [s for s in SYS if s in set(df.system)]
    num = {"parents": {s: int(df[df.system == s].seed.nunique()) for s in systems}}
    recs, summ = [], []
    for s in systems:
        for b in sorted(df.budget.unique()):
            d = df[(df.system == s) & (df.budget == b)]
            u = d[d.scheme == "uniform"].set_index("seed")
            t = d[d.scheme == "theory"].set_index("seed")
            r = d[d.scheme == "random"].groupby("seed")[["fail", "vpt_loss", "pred_dmse"]].mean()
            row = dict(system=s, budget=b, n=len(u),
                       fail_uniform=100 * u.fail.mean(), fail_random=100 * r.fail.mean(), fail_theory=100 * t.fail.mean(),
                       vpt_uniform=float(u.vpt_loss.median()), vpt_random=float(r.vpt_loss.median()), vpt_theory=float(t.vpt_loss.median()),
                       pred_ratio_theory_uniform=float(np.median(t.pred_dmse / u.pred_dmse)),
                       pred_ratio_random_uniform=float(np.median(r.pred_dmse / u.pred_dmse)),
                       theory_equals_uniform=float(100 * np.mean([np.allclose(t.loc[i, [f"bits_{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3")]].values.astype(float),
                                                                          u.loc[i, [f"bits_{k}" for k in ("W1", "b1", "W2", "b2", "W3", "b3")]].values.astype(float)) for i in u.index])))
            summ.append(row)
            for name, x, y in (("theory-uniform", t, u), ("theory-random", t, r)):
                for metric in ("fail", "vpt_loss"):
                    diff = (x[metric] - y[metric]).dropna().values
                    lo, hi = boot_ci(diff)
                    recs.append(dict(system=s, budget=b, cmp=name, metric=metric, n=len(diff), mean=float(diff.mean()),
                                     median=float(np.median(diff)), lo=lo, hi=hi,
                                     better=int((diff < 0).sum()), worse=int((diff > 0).sum()), tie=int((diff == 0).sum()),
                                     p=wilcoxon_p(diff)))
    S = pd.DataFrame(summ)
    pc = pd.DataFrame(recs)
    pc["p_holm"] = pc.groupby(["metric", "cmp"])["p"].transform(lambda s_: pd.Series(holm(s_.values), index=s_.index))
    S.to_csv(os.path.join(a.out, "mixed_summary.csv"), index=False)
    pc.to_csv(os.path.join(a.out, "mixed_paired.csv"), index=False)
    num["summary"] = S.to_dict("records")
    num["paired"] = pc.to_dict("records")

    # predicted error vs failure across allocations (within parent x budget groups containing both outcomes)
    aucs = {}
    for s in systems:
        d = df[df.system == s]
        pooled = auc(np.log10(d.pred_dmse.values), d.fail.values.astype(bool))
        within = []
        for _, g in d.groupby(["seed", "budget"]):
            if 0 < g.fail.sum() < len(g):
                within.append(auc(np.log10(g.pred_dmse.values), g.fail.values.astype(bool)))
        aucs[s] = dict(pooled=pooled, within=float(np.nanmean(within)) if within else np.nan, n_groups=len(within))
    num["auc"] = aucs

    # table
    with open(os.path.join(a.out, "tables", "tab_mixed.tex"), "w") as f:
        rows = []
        for _, r in S.iterrows():
            q = pc[(pc.system == r.system) & (pc.budget == r.budget) & (pc.metric == "vpt_loss") & (pc["cmp"] == "theory-uniform")].iloc[0]
            ph = "--" if not np.isfinite(q.p_holm) else ("$<0.001$" if q.p_holm < 1e-3 else f"{q.p_holm:.3f}")
            nm = NAME[r.system].replace("ö", "\\\"o")
            rows.append(f"{nm if r.budget == S.budget.min() else ''} & {int(r.budget)} & {r.fail_uniform:.0f} & {r.fail_random:.0f} & {r.fail_theory:.0f} & "
                        f"{r.vpt_uniform:.2f} & {r.vpt_theory:.2f} & {q['median']:+.3f} & {ph}")
        f.write("\\\\\n".join(rows) + "\n")

    # figure
    fig, ax = plt.subplots(1, len(systems), figsize=(4.2 * len(systems), 3.6), squeeze=False)
    w = 0.26
    for j, s in enumerate(systems):
        d = S[S.system == s].sort_values("budget")
        x = np.arange(len(d))
        ax[0, j].bar(x - w, d.fail_uniform, w, label="uniform", color="tab:gray")
        ax[0, j].bar(x, d.fail_random, w, label="random (mean of 20)", color="tab:orange")
        ax[0, j].bar(x + w, d.fail_theory, w, label="theory-guided", color="tab:green")
        ax[0, j].set_xticks(x)
        ax[0, j].set_xticklabels([f"{int(b)}" for b in d.budget])
        ax[0, j].set_xlabel("average bits per parameter")
        ax[0, j].set_title(NAME[s], fontsize=10)
        ax[0, j].set_ylim(0, 100)
        ax[0, j].grid(alpha=0.3, axis="y")
    ax[0, 0].set_ylabel("models that fail (%)")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figures", "fig_mixed.pdf"))

    json.dump(num, open(os.path.join(a.out, "mixed_numbers.json"), "w"), indent=1, default=float)
    pd.set_option("display.width", 220)
    print(S.round(2).to_string(index=False))
    print(pc[pc.metric == "fail"][["system", "budget", "cmp", "mean", "lo", "hi", "better", "worse", "tie", "p_holm"]].round(3).to_string(index=False))
    print(pc[pc.metric == "vpt_loss"][["system", "budget", "cmp", "median", "lo", "hi", "p_holm"]].round(3).to_string(index=False))
    print(aucs)


if __name__ == "__main__":
    main()
