"""First-order sensitivity theory for the tensor-isolated interventions.

For f(x) = x + W3 tanh(W2 tanh(W1 x + b1) + b2) + b3 and i.i.d. Gaussian noise of variance s^2 per entry in a
set S of parameter keys, a first-order expansion gives, at the parent,

    E[ ||f_theta+delta(x) - f_theta(x)||^2 ] = s^2 * sum_{k in S} ||J_k(x)||_F^2 ,

so the expected one-step MSE increase (MSE is a mean over the d state coordinates) is

    dMSE_pred = (target^2 / n_S) * sum_{k in S} S_k ,     S_k = E_x ||J_k(x)||_F^2 / d,

where target = ||delta||_2 is the total norm of the injected noise and n_S its number of entries.
S_k is computed analytically from the parent and the test inputs (no perturbed model is evaluated):
    b3: d | W3: d*||h2||^2 | b2: ||W3 D2||_F^2 | W2: ||h1||^2 ||W3 D2||_F^2 |
    b1: ||W3 D2 W2 D1||_F^2 | W1: ||x||^2 ||W3 D2 W2 D1||_F^2          (D = diag(1 - h^2)).

This script rebuilds every valid parent (verified against stored MSE), computes S_k, and attaches the
prediction to every tensor-intervention model of results_tensor/ and results_l96/.

    python run_theory.py --out results_theory
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.model import KEYS
from run_protect import get_parent

GROUPS = {"W1": ("W1",), "W2": ("W2",), "W3": ("W3",), "biases": ("b1", "b2", "b3")}


def sensitivities(p, X):
    """S_k = E_x ||dF/dtheta_k||_F^2 / d for each key (analytic, first order)."""
    d = X.shape[1]
    h1 = np.tanh(X @ p["W1"].T + p["b1"])
    h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
    D1, D2 = 1 - h1 ** 2, 1 - h2 ** 2
    w3sq = (p["W3"] ** 2).sum(0)                                   # (H,)
    a2 = (D2 ** 2) @ w3sq                                          # ||W3 D2||_F^2 per sample
    T = np.einsum("ij,nj,jk->nik", p["W3"], D2, p["W2"]) * D1[:, None, :]
    a1 = (T ** 2).sum(axis=(1, 2))                                 # ||W3 D2 W2 D1||_F^2 per sample
    S = {"b3": float(d) / d,
         "W3": float(np.mean(d * (h2 ** 2).sum(1))) / d,
         "b2": float(np.mean(a2)) / d,
         "W2": float(np.mean((h1 ** 2).sum(1) * a2)) / d,
         "b1": float(np.mean(a1)) / d,
         "W1": float(np.mean((X ** 2).sum(1) * a1)) / d}
    return S


def key_set(group):
    if group == "all":
        return KEYS
    if group.startswith("all_but_"):
        drop = GROUPS[group[len("all_but_"):]]
        return tuple(k for k in KEYS if k not in drop)
    return GROUPS[group]


def predict(row, p, S, tot_norm):
    keys = key_set(row["group"])
    nS = sum(p[k].size for k in keys)
    if row["design"] == "own":
        target = row["r"] * np.sqrt(sum(np.sum(p[k] ** 2) for k in keys))
    else:
        target = row["r"] * tot_norm
    return target ** 2 / nS * sum(S[k] for k in keys), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_theory")
    ap.add_argument("--main", default="results_full_30seeds")
    ap.add_argument("--l96", default="results_l96")
    ap.add_argument("--tensor", default="results_tensor")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = dict(main=a.main, l96=a.l96)
    for system in ("lorenz", "rossler", "lorenz96"):
        for seed in range(30):
            path = os.path.join(a.out, f"{system}_seed{seed}.csv")
            if os.path.exists(path):
                continue
            tpath = (os.path.join(a.l96, f"tensor_seed{seed}.csv") if system == "lorenz96"
                     else os.path.join(a.tensor, f"{system}_seed{seed}.csv"))
            if not os.path.exists(tpath):
                continue
            t0 = time.time()
            parent, Zte, dt, ref, pmse, valid = get_parent(system, seed, res)
            if not valid:
                continue
            Xte = Zte[:-1]
            S = sensitivities(parent, Xte)
            tot = float(np.sqrt(sum(np.sum(parent[k] ** 2) for k in KEYS)))
            df = pd.read_csv(tpath)
            preds = [predict(r, parent, S, tot) for r in df.to_dict("records")]
            df["dmse_pred"] = [x[0] for x in preds]
            df["noise_norm"] = [x[1] for x in preds]
            df["mse_parent"] = pmse
            lr = df.log10_mse_ratio.replace(np.inf, np.nan)
            df["dmse_meas"] = pmse * (10 ** lr - 1)
            for k in KEYS:
                df[f"S_{k}"] = S[k]
                df[f"norm_{k}"] = float(np.linalg.norm(parent[k]))
                df[f"n_{k}"] = parent[k].size
            df["norm_total"] = tot
            df.to_csv(path, index=False)
            print(f"[{system} seed {seed}] {len(df)} rows {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
