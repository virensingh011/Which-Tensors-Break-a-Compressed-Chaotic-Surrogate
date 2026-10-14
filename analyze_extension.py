"""Tensor-isolated interventions on three systems + Lorenz-96 replication of the matched-control study.

Reads  results_tensor/{lorenz,rossler}_seed*.csv   (run_tensor_intervention.py)
       results_l96/{main,tensor}_seed*.csv          (run_l96.py)
Writes paper/figures/fig_tensor.pdf, paper/tables/tab_tensor*.tex, tab_l96_*.tex and
       paper/ext_numbers.json  (every number quoted in the new text).

    python analyze_extension.py --tensor results_tensor --l96 results_l96 --out paper
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

from analyze_tensor import boot_ci, holm, logistic_r50, wilcoxon_p

rng = np.random.default_rng(1)
SYS_LABEL = {"lorenz": "Lorenz-63", "rossler": "R\\\"ossler", "lorenz96": "Lorenz-96"}
SYS_PLAIN = {"lorenz": "Lorenz-63", "rossler": "Rössler", "lorenz96": "Lorenz-96 ($N=6$)"}
GROUP_LABEL = {"all": "isotropic (all tensors)", "W1": "$W_1$ only", "W2": "$W_2$ only",
               "W3": "$W_3$ only", "biases": "biases only", "all_but_W3": "isotropic without $W_3$"}


def load_tensor(tdir, ldir):
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(tdir, "*_seed*.csv")))]
    parts += [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ldir, "tensor_seed*.csv")))]
    df = pd.concat(parts, ignore_index=True)
    df["log10_mse_ratio"] = df.log10_mse_ratio.replace(np.inf, np.nan)
    return df


def per_seed(df):
    return (df.groupby(["system", "design", "group", "r", "seed"])
              .agg(fail=("fail", "mean"), div=("diverged", "mean"), vpt_loss=("vpt_loss", "median"),
                   lmse=("log10_mse_ratio", "median"), r_W3=("r_W3", "median")).reset_index())


# ----------------------------------------------------------------------------- tensor interventions
def tensor_assets(df, out, K):
    ps = per_seed(df)
    systems = [s for s in ("lorenz", "rossler", "lorenz96") if s in set(ps.system)]
    num = {}
    # ---- figure
    fig, ax = plt.subplots(3, len(systems), figsize=(4.0 * len(systems), 8.6), squeeze=False)
    style = {"all": ("k", "-", "o"), "W1": ("tab:blue", "-", "s"), "W2": ("tab:orange", "-", "^"),
             "W3": ("tab:red", "-", "D"), "biases": ("tab:purple", "-", "v"),
             "all_but_W3": ("tab:green", "--", "o")}
    for j, s in enumerate(systems):
        d = ps[(ps.system == s) & (ps.design.isin(["total", "loo"]))].copy()
        d.loc[d.group == "all_but_W3", "design"] = "total"
        for g, (c, ls, m) in style.items():
            x = d[(d.group == g) & (d.design == "total")]
            if x.empty:
                continue
            for i, metric in enumerate(["fail", "vpt_loss"]):
                agg = x.groupby("r")[metric].agg(["mean", "std", "count"]).reset_index()
                if metric == "vpt_loss":
                    agg["mean"] = x.groupby("r")[metric].median().values
                err = 1.96 * agg["std"] / np.sqrt(agg["count"])
                ax[i, j].errorbar(agg.r, agg["mean"], yerr=err if metric == "fail" else None,
                                  color=c, ls=ls, marker=m, ms=4, lw=1.4, capsize=2,
                                  label=GROUP_LABEL[g])
        # row 3: own-norm design -- failure vs relative displacement of the perturbed tensor itself
        o = ps[(ps.system == s) & (ps.design == "own")]
        for g in ("W1", "W2", "W3", "biases"):
            x = o[o.group == g].groupby("r").fail.agg(["mean", "std", "count"]).reset_index()
            c, ls, m = style[g]
            ax[2, j].errorbar(x.r, x["mean"], yerr=1.96 * x["std"] / np.sqrt(x["count"]), color=c, ls=ls,
                              marker=m, ms=4, lw=1.4, capsize=2, label=GROUP_LABEL[g])
        ax[2, j].set_xscale("log")
        ax[2, j].set_ylim(0, 1.02)
        ax[2, j].grid(alpha=0.3)
        ax[2, j].set_xlabel("tensor's own relative displacement  $\\|\\Delta\\theta_G\\|/\\|\\theta_G\\|$", fontsize=8)
        ax[0, j].set_title(SYS_PLAIN[s].replace("$", ""), fontsize=10)
        ax[0, j].set_ylim(0, 1.02)
        ax[1, j].set_xlabel("total relative displacement  $\\|\\Delta\\theta\\|/\\|\\theta\\|$", fontsize=8)
        for i in range(2):
            ax[i, j].set_xscale("log")
            ax[i, j].grid(alpha=0.3)
    ax[0, 0].set_ylabel("fraction of models that fail")
    ax[1, 0].set_ylabel("VPT lost (Lyapunov times, median)")
    ax[2, 0].set_ylabel("fraction of models that fail")
    ax[0, 0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig_tensor.pdf"))
    plt.close(fig)

    # ---- table: failure at three norms, six noise groups
    NS = {"lorenz": (0.01, 0.02, 0.05), "rossler": (0.01, 0.02, 0.05), "lorenz96": (0.05, 0.1, 0.2)}
    rows = []
    for s in systems:
        for r in NS[s]:
            vals = []
            for g in ["all", "W1", "W2", "W3", "biases", "all_but_W3"]:
                d = ps[(ps.system == s) & (ps.group == g) & (ps.r == r) & (ps.design.isin(["total", "loo"]))]
                vals.append(100 * d.fail.mean())
            rows.append((s, r, vals))
            num[f"fail_{s}_{r}"] = dict(zip(["all", "W1", "W2", "W3", "biases", "all_but_W3"], vals))
    with open(os.path.join(out, "tables", "tab_tensor.tex"), "w") as f:
        for s, r, v in rows:
            f.write(f"{SYS_LABEL[s]} & {r:g} & " + " & ".join(f"{x:.0f}" for x in v) + "\\\\\n")

    # ---- paired comparisons vs isotropic (per-seed differences, Wilcoxon, Holm within system x metric)
    recs = []
    for s in systems:
        d = ps[ps.system == s]
        base = d[(d.design == "total") & (d.group == "all")].set_index(["r", "seed"])
        for (design, g), gg in d[(d.group != "all") & d.design.isin(["total", "loo"])].groupby(["design", "group"]):
            for r, x in gg.groupby("r"):
                x = x.set_index("seed")
                b = base.loc[r].reindex(x.index)
                for metric in ("vpt_loss", "fail", "lmse"):
                    diff = (x[metric] - b[metric]).dropna().values
                    lo, hi = boot_ci(diff)
                    recs.append(dict(system=s, design=design, group=g, r=r, metric=metric, n=len(diff),
                                     med=float(np.median(diff)), lo=lo, hi=hi, p=wilcoxon_p(diff)))
    pc = pd.DataFrame(recs)
    pc["p_holm"] = pc.groupby(["system", "metric"])["p"].transform(lambda s_: pd.Series(holm(s_.values), index=s_.index))
    pc.to_csv(os.path.join(out, "tensor_paired.csv"), index=False)
    with open(os.path.join(out, "tables", "tab_tensor_paired.tex"), "w") as f:
        for s in systems:
            for g in ("W3", "all_but_W3", "W2"):
                for r in NS[s][:2]:
                    q = pc[(pc.system == s) & (pc.group == g) & (pc.r == r) & (pc.metric == "vpt_loss")]
                    q2 = pc[(pc.system == s) & (pc.group == g) & (pc.r == r) & (pc.metric == "fail")]
                    if q.empty:
                        continue
                    q, q2 = q.iloc[0], q2.iloc[0]
                    ph = "$<0.001$" if q.p_holm < 1e-3 else f"{q.p_holm:.3f}"
                    f.write(f"{SYS_LABEL[s]} & {GROUP_LABEL[g]} & {r:g} & {100 * q2.med:+.0f} & "
                            f"{q.med:+.2f} [{q.lo:+.2f}, {q.hi:+.2f}] & {ph}\\\\\n")
    # headline significance counts
    for s in systems:
        for g in ("W3", "W2", "all_but_W3", "W1", "biases"):
            q = pc[(pc.system == s) & (pc.group == g) & (pc.metric == "vpt_loss")]
            num[f"sig_{s}_{g}"] = dict(
                n=int(len(q)), worse=int(((q.p_holm < 0.05) & (q.med > 0)).sum()),
                better=int(((q.p_holm < 0.05) & (q.med < 0)).sum()))

    # ---- own-norm design: failure at equal relative displacement of the tensor itself
    own = ps[ps.design == "own"].groupby(["system", "group", "r"]).fail.mean().unstack("group") * 100
    for s in systems:
        for r in (0.05, 0.1, 0.2):
            num[f"own_{s}_{r}"] = own.loc[(s, r)].round(0).to_dict()
    with open(os.path.join(out, "tables", "tab_tensor_own.tex"), "w") as f:
        for s in systems:
            for r in (0.05, 0.1, 0.2):
                v = own.loc[(s, r)]
                f.write(f"{SYS_LABEL[s]} & {r:g} & " + " & ".join(f"{v[g]:.0f}" for g in ("W1", "W2", "W3", "biases")) + "\\\\\n")

    # ---- radii by noise group (total design); report '<0.01' when failure at lowest norm exceeds 50%
    rr = {}
    for s in systems:
        d = df[(df.system == s) & (df.design == "total")]
        seeds = d.seed.unique()
        for g, x in d.groupby("group"):
            low = x[x.r == x.r.min()].fail.mean()
            if low > 0.5:
                rr[f"{s}_{g}"] = dict(r50=None, note=f"failure {100 * low:.0f}% at r={x.r.min():g}")
                continue
            est = logistic_r50(np.log10(x.rel_norm.values), x.fail.values.astype(float))
            by = {sd: x[x.seed == sd] for sd in seeds}
            bs = np.array([logistic_r50(*(lambda c: (np.log10(c.rel_norm.values), c.fail.values.astype(float)))(
                pd.concat([by[k] for k in rng.choice(seeds, len(seeds))]))) for _ in range(150)])
            rr[f"{s}_{g}"] = dict(r50=float(est), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)))
    num["tensor_radius"] = rr
    # per-tensor radius in the own-norm design (relative displacement of the tensor itself)
    ro = {}
    for s in systems:
        d = df[(df.system == s) & (df.design == "own")]
        seeds = d.seed.unique()
        for g, x in d.groupby("group"):
            est = logistic_r50(np.log10(x.r.values), x.fail.values.astype(float))
            by = {sd: x[x.seed == sd] for sd in seeds}
            bs = np.array([logistic_r50(*(lambda c: (np.log10(c.r.values), c.fail.values.astype(float)))(
                pd.concat([by[k] for k in rng.choice(seeds, len(seeds))]))) for _ in range(150)])
            ro[f"{s}_{g}"] = dict(r50=float(est), lo=float(np.nanquantile(bs, .025)), hi=float(np.nanquantile(bs, .975)))
    num["own_radius"] = ro
    with open(os.path.join(out, "tables", "tab_tensor_radius.tex"), "w") as f:
        for s in systems:
            tot = [rr.get(f"{s}_{g}") for g in ("all", "W1", "W2", "W3", "biases")]
            def fmt(e):
                if e is None:
                    return "--"
                if e.get("r50") is None:
                    return "$<0.01$"
                return f"{e['r50']:.3f}"
            f.write(f"{SYS_LABEL[s]} & total & " + " & ".join(fmt(e) for e in tot) + "\\\\\n")
            f.write(" & own-norm & -- & " + " & ".join(
                f"{ro[f'{s}_{g}']['r50']:.2f}" for g in ("W1", "W2", "W3", "biases")) + "\\\\\n")
    num["tensor_parents"] = {s: int(ps[ps.system == s].seed.nunique()) for s in systems}
    num["tensor_models"] = int(len(df))
    return num


# ----------------------------------------------------------------------------- Lorenz-96 main design
def l96_assets(ldir, out):
    files = sorted(glob.glob(os.path.join(ldir, "main_seed*.csv")))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    infos = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ldir, "info_seed*.json")))]
    ref = json.load(open(os.path.join(ldir, "reference_lle.json")))["lorenz96"]
    df["fail"] = df.diverged | df.collapsed
    df["lmse"] = np.log10(np.maximum(df.mse, 1e-300) / df.mse_parent).replace(np.inf, np.nan)
    df.loc[~np.isfinite(df.mse), "lmse"] = np.nan
    num = dict(parents_total=len(infos), parents_valid=int(sum(i["valid"] for i in infos)),
               seeds_valid=int(df.seed.nunique()), models=int(len(df)), ref_lle=ref,
               parent_lle_mean=float(np.mean([i["parent_lle"] for i in infos if i["valid"]])),
               parent_lle_sd=float(np.std([i["parent_lle"] for i in infos if i["valid"]])),
               parent_vpt_mean=float(df.groupby("seed").parent_vpt_lt.first().mean()),
               parent_vpt_sd=float(df.groupby("seed").parent_vpt_lt.first().std()),
               parent_mse_median=float(df.groupby("seed").mse_parent.first().median()))
    num["fail_overall"] = float(100 * df.fail.mean())
    num["div_overall"] = float(100 * df.diverged.mean())
    num["col_overall"] = float(100 * df.collapsed.mean())

    # ---- table: failure of structured vs same-norm isotropic noise
    prim = df[df.role == "primary"]
    ctrl = df[df.role == "control"]
    rows = []
    for g, x in prim[prim.family != "prune_random"].groupby("group", sort=False):
        c = ctrl[(ctrl.control_of == g) & (ctrl.family == "noise_iso")]
        rows.append(dict(group=g, rel=float(x.rel_norm.median()), sdiv=100 * x.diverged.mean(),
                         scol=100 * x.collapsed.mean(), idiv=100 * c.diverged.mean(),
                         icol=100 * c.collapsed.mean()))
    nm = {"prune_global": "Global pruning", "prune_layer": "Layer-wise pruning", "quant": "quantization"}
    with open(os.path.join(out, "tables", "tab_l96_fail.tex"), "w") as f:
        for r in rows:
            fam, lvl = r["group"].split("@")
            lab = f"{lvl}-bit quantization" if fam == "quant" else f"{nm[fam]} {float(lvl) * 100:.0f}\\%"
            f.write(f"{lab} & {r['rel']:.3f} & {r['sdiv'] / 100:.2f} & {r['scol'] / 100:.2f} & {r['idiv'] / 100:.2f} & {r['icol'] / 100:.2f}\\\\\n")
    num["fail_table"] = rows

    # ---- robustness radius (isotropic noise: grid + controls pooled)
    iso = df[((df.role == "grid") | ((df.role == "control") & (df.family == "noise_iso")))]
    seeds = iso.seed.unique()
    lr = lambda x: logistic_r50(np.log10(x.rel_norm.values), x.fail.values.astype(float))
    by = {s: iso[iso.seed == s] for s in seeds}

    def radius(p):
        def r(x):
            lg = np.log10(x.rel_norm.values)
            y = x.fail.values.astype(float)
            from scipy import optimize
            xm = lg - lg.mean()
            nll = lambda b: np.sum(np.logaddexp(0, np.clip(b[0] + b[1] * xm, -30, 30)) -
                                   y * np.clip(b[0] + b[1] * xm, -30, 30))
            b0, b1 = optimize.minimize(nll, [0., 1.], method="Nelder-Mead").x
            return 10 ** (lg.mean() + (np.log(p / (1 - p)) - b0) / b1)
        est = r(iso)
        bs = np.array([r(pd.concat([by[k] for k in rng.choice(seeds, len(seeds))])) for _ in range(300)])
        return float(est), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
    num["radius"] = {f"r{int(p * 100)}": radius(p) for p in (0.1, 0.5, 0.9)}

    # ---- paired comparisons (survival-independent endpoints)
    out_pairs = {}
    for fam, name in (("noise_iso", "iso"), ("noise_layer", "layer")):
        recs = []
        for g, x in prim[prim.family != "prune_random"].groupby("group", sort=False):
            c = ctrl[(ctrl.control_of == g) & (ctrl.family == fam)]
            for metric, col in (("vpt", "vpt_lt"), ("mse", "lmse")):
                a = x.set_index("seed")[col]
                b = c.groupby("seed")[col].median()
                d = (a - b.reindex(a.index)).dropna().values
                recs.append(dict(group=g, metric=metric, n=len(d), med=float(np.median(d)),
                                 lo=boot_ci(d)[0], hi=boot_ci(d)[1], p=wilcoxon_p(d)))
        pr = pd.DataFrame(recs)
        pr["p_holm"] = pr.groupby("metric")["p"].transform(lambda s_: pd.Series(holm(s_.values), index=s_.index))
        for metric in ("vpt", "mse"):
            q = pr[pr.metric == metric]
            sig = q.p_holm < 0.05
            good = (q.med > 0) if metric == "vpt" else (q.med < 0)
            out_pairs[f"{name}_{metric}"] = dict(n=int(len(q)), sig_good=int((sig & good).sum()),
                                                 sig_bad=int((sig & ~good).sum()), n_sig=int(sig.sum()),
                                                 med_min=float(q.med.min()), med_max=float(q.med.max()))
        pr.to_csv(os.path.join(out, f"l96_paired_{name}.csv"), index=False)
    num["paired"] = out_pairs

    # ---- H1: failure vs parent-relative one-step error
    d = df[df.role != "parent"].copy()
    d["ratio"] = d.mse / d.mse_parent
    d.loc[~np.isfinite(d.ratio), "ratio"] = np.inf
    failed, ok = d[d.fail], d[~d.fail]
    num["h1"] = dict(
        failed_ge10=float(100 * (failed.ratio >= 10).mean()), failed_lt3=float(100 * (failed.ratio < 3).mean()),
        surv_ge10=float(100 * (ok.ratio >= 10).mean()),
        fail_rate_lt3=float(100 * d[d.ratio < 3].fail.mean()), n_lt3=int((d.ratio < 3).sum()),
        fail_rate_gt1000=float(100 * d[d.ratio > 1000].fail.mean()))

    # ---- VPT vs one-step error (window fit)
    w = d[np.isfinite(d.mse) & (np.log10(d.mse) >= -6) & (np.log10(d.mse) <= -3)]
    if len(w) > 20:
        lm = np.log10(w.mse.values)
        res = stats.linregress(lm, w.vpt_lt.values)
        num["law"] = dict(n=int(len(w)), r2=float(res.rvalue ** 2), slope=float(res.slope))
    # ---- LLE usable fraction cannot be computed (LLE only for parents); report survival
    num["survive_pct"] = float(100 * (1 - df.fail.mean()))
    return num


# ----------------------------------------------------------------------------- protect-the-output-layer
def protect_assets(pdir, out):
    files = sorted(glob.glob(os.path.join(pdir, "*_seed*.csv")))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    systems = [s for s in ("lorenz", "rossler", "lorenz96") if s in set(df.system)]
    num = {"parents": {s: int(df[df.system == s].seed.nunique()) for s in systems}}
    fam_name = {"prune_global": "Global pruning", "prune_layer": "Layer-wise pruning", "quant": "quantization"}
    keys = [("quant", 8), ("quant", 6), ("quant", 4), ("quant", 3),
            ("prune_layer", 0.25), ("prune_layer", 0.5), ("prune_global", 0.25), ("prune_global", 0.5)]
    fr = df.groupby(["system", "family", "level", "scope"]).fail.mean() * 100
    with open(os.path.join(out, "tables", "tab_protect.tex"), "w") as f:
        for fam, lvl in keys:
            lab = f"{int(lvl)}-bit quantization" if fam == "quant" else f"{fam_name[fam]} {lvl * 100:.0f}\\%"
            cells = []
            for s in systems:
                for sc in ("all", "protOut", "onlyOut"):
                    cells.append(f"{fr.loc[(s, fam, lvl, sc)]:.0f}")
            f.write(lab + " & " + " & ".join(cells) + "\\\\\n")
    num["fail"] = {f"{s}|{fam}|{lvl}|{sc}": float(fr.loc[(s, fam, lvl, sc)])
                   for s in systems for fam, lvl in keys for sc in ("all", "protOut", "onlyOut")}
    # paired: protOut - all and onlyOut - all, per family/level, per seed
    recs = []
    for s in systems:
        d = df[df.system == s]
        for (fam, lvl), g in d.groupby(["family", "level"]):
            base = g[g.scope == "all"].set_index("seed")
            for sc in ("protOut", "onlyOut"):
                x = g[g.scope == sc].set_index("seed")
                for metric in ("vpt_loss", "fail"):
                    diff = (x[metric].astype(float) - base[metric].astype(float).reindex(x.index)).dropna().values
                    recs.append(dict(system=s, family=fam, level=lvl, scope=sc, metric=metric, n=len(diff),
                                     med=float(np.median(diff)), mean=float(np.mean(diff)),
                                     p=wilcoxon_p(diff)))
    pc = pd.DataFrame(recs)
    pc["p_holm"] = pc.groupby(["system", "scope", "metric"])["p"].transform(
        lambda s_: pd.Series(holm(s_.values), index=s_.index))
    pc.to_csv(os.path.join(out, "protect_paired.csv"), index=False)
    for s in systems:
        for sc in ("protOut", "onlyOut"):
            q = pc[(pc.system == s) & (pc.scope == sc) & (pc.metric == "vpt_loss")]
            qf = pc[(pc.system == s) & (pc.scope == sc) & (pc.metric == "fail")]
            num[f"{s}|{sc}"] = dict(
                n=int(len(q)), vpt_better=int(((q.p_holm < .05) & (q["mean"] < 0)).sum()),
                vpt_worse=int(((q.p_holm < .05) & (q["mean"] > 0)).sum()),
                fail_better=int(((qf.p_holm < .05) & (qf["mean"] < 0)).sum()),
                fail_worse=int(((qf.p_holm < .05) & (qf["mean"] > 0)).sum()),
                mean_fail_change_pts=float(100 * qf["mean"].mean()),
                max_abs_fail_change_pts=float(100 * qf["mean"].abs().max()))
    num["rel_norm_ratio_protOut_vs_all"] = float(
        (df[df.scope == "protOut"].groupby(["system", "family", "level"]).rel_norm.median().values /
         df[df.scope == "all"].groupby(["system", "family", "level"]).rel_norm.median().values).mean())
    num["rel_norm_onlyOut_vs_all"] = float(
        (df[df.scope == "onlyOut"].groupby(["system", "family", "level"]).rel_norm.median().values /
         df[df.scope == "all"].groupby(["system", "family", "level"]).rel_norm.median().values).mean())
    return num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="results_tensor")
    ap.add_argument("--l96", default="results_l96")
    ap.add_argument("--protect", default="results_protect")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    df = load_tensor(a.tensor, a.l96)
    num = dict(tensor=tensor_assets(df, a.out, 10), l96=l96_assets(a.l96, a.out),
               protect=protect_assets(a.protect, a.out))
    json.dump(num, open(os.path.join(a.out, "ext_numbers.json"), "w"), indent=1, default=float)
    # LaTeX: the last row of an \\input table must not end with \\ (main.tex adds it after \\input)
    for name in ("tab_tensor", "tab_tensor_radius", "tab_tensor_paired", "tab_protect", "tab_l96_fail", "tab_tensor_own"):
        p = os.path.join(a.out, "tables", name + ".tex")
        if os.path.exists(p):
            t = open(p).read().rstrip("\n")
            if t.endswith("\\\\"):
                open(p, "w").write(t[:-2] + "\n")
    print(json.dumps(num, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
