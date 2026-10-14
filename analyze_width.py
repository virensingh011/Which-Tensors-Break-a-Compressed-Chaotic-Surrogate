"""Width generality: own-norm radii, theory accuracy and out-of-design radius prediction for H in {32, 64, 128}.

H = 64 comes from the main study (results_theory); H = 32, 128 from run_width.py (results_width/H*).
For comparability the critical ratio c is fitted on the TOTAL-norm design only, for every width.

    python analyze_width.py --out paper
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyze_tensor import logistic_r50
from analyze_theory import GROUPS, load, logit_x50

NAME = {"lorenz": "Lorenz-63", "rossler": "R\\\"ossler"}


def analyse(df):
    out = {}
    for s in ("lorenz", "rossler"):
        d = df[df.system == s]
        if d.empty:
            continue
        fit = d[(d.design == "total") & np.isfinite(d.lpred)]
        x50, _ = logit_x50(fit.lpred.values, fit.fail.values.astype(float))
        own = d[d.design == "own"]
        res = {"parents": int(d.seed.nunique()), "c": float(10 ** x50), "radii": {}}
        q = d[np.isfinite(d.lmeas) & np.isfinite(d.lpred) & (d.lpred >= 1)]
        if len(q) > 30:
            lr = stats.linregress(q.lpred, q.lmeas)
            res["acc_slope"], res["acc_r2"], res["acc_n"] = float(lr.slope), float(lr.rvalue ** 2), int(len(q))
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
            res["radii"][g] = dict(measured=float(meas), predicted=float(np.median(per)))
        out[s] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", default="results_theory")
    ap.add_argument("--width", default="results_width")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    res = {}
    base = load(a.theory)
    base = base[base.system.isin(["lorenz", "rossler"])]
    res[64] = analyse(base)
    for H in (32, 128):
        files = sorted(glob.glob(os.path.join(a.width, f"H{H}", "*_seed*.csv")))
        if not files:
            continue
        df = load(os.path.join(a.width, f"H{H}"))
        res[H] = analyse(df)
    json.dump(res, open(os.path.join(a.out, "width_numbers.json"), "w"), indent=1, default=float)
    lines = []
    for s in ("lorenz", "rossler"):
        for H in sorted(res):
            r = res[H].get(s)
            if not r:
                continue
            rd = r["radii"]
            cells = [f"{rd[g]['measured']:.3f}" for g in GROUPS] + [f"{rd[g]['predicted'] / rd[g]['measured']:.2f}" for g in GROUPS]
            lines.append(f"{NAME[s] if H == min(res) else ''} & {H} & {r['parents']} & {r['c']:.0f} & " + " & ".join(cells))
    open(os.path.join(a.out, "tables", "tab_width.tex"), "w").write("\\\\\n".join(lines) + "\n")
    for H in sorted(res):
        for s, r in res[H].items():
            print(H, s, r["parents"], "c=%.0f" % r["c"], {g: (round(v["measured"], 3), round(v["predicted"] / v["measured"], 2)) for g, v in r["radii"].items()},
                  "acc", r.get("acc_slope"), r.get("acc_r2"))


if __name__ == "__main__":
    main()
