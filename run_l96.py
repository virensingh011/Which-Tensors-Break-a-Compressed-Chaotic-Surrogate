"""Lorenz-96 extension (N=6, F=8, dt=0.05): same protocol as run_experiment.py, plus tensor-isolated
interventions, on a 6-dimensional chaotic system.

Per seed writes
  <out>/main_seed<k>.csv    structured perturbations, matched controls, calibration grid, parent
  <out>/tensor_seed<k>.csv  tensor-isolated interventions (see run_tensor_intervention.py)
Metrics per model: one-step MSE, diverged / collapsed, VPT (Lyapunov times).  LLE only for the parent
(used for the same parent-validity rule as the paper).

    python run_l96.py --seeds 30 --K 10 --out results_l96
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.dynamics import lle_batch, vpt_batch
from jacsens.perturb import make_perturbations
from run_tensor_intervention import build_models, fail_batch

N, F, DT = 6, 8.0, 0.05


def f96(x):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def jvp96(x, u):
    return ((np.roll(u, -1) - np.roll(u, 2)) * np.roll(x, 1)
            + (np.roll(x, -1) - np.roll(x, 2)) * np.roll(u, 1) - u)


def rk4(x, h):
    k1 = f96(x)
    k2 = f96(x + 0.5 * h * k1)
    k3 = f96(x + 0.5 * h * k2)
    k4 = f96(x + h * k3)
    return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(x0, n_samples, dt=DT, h=0.005, transient=20.0):
    x = x0.copy()
    nsub = int(round(dt / h))
    for _ in range(int(round(transient / h))):
        x = rk4(x, h)
    out = np.empty((n_samples, N))
    for i in range(n_samples):
        out[i] = x
        for _ in range(nsub):
            x = rk4(x, h)
    return out


def reference_lle(T=1000.0, h=0.01, renorm_every=5, seed=0):
    rng = np.random.default_rng(seed)
    x = F * np.ones(N) + 0.01 * rng.normal(size=N)
    for _ in range(int(50 / h)):
        x = rk4(x, h)
    u = rng.normal(size=N)
    u /= np.linalg.norm(u)
    acc = 0.0
    n = int(round(T / h))
    for i in range(n):
        # joint RK4 on (x, u)
        k1x, k1u = f96(x), jvp96(x, u)
        x2, u2 = x + 0.5 * h * k1x, u + 0.5 * h * k1u
        k2x, k2u = f96(x2), jvp96(x2, u2)
        x3, u3 = x + 0.5 * h * k2x, u + 0.5 * h * k2u
        k3x, k3u = f96(x3), jvp96(x3, u3)
        x4, u4 = x + h * k3x, u + h * k3u
        k4x, k4u = f96(x4), jvp96(x4, u4)
        x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        u = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        if (i + 1) % renorm_every == 0:
            nr = np.linalg.norm(u)
            acc += math.log(nr)
            u /= nr
    return acc / T


def run_seed(seed, args, ref_lle):
    rng = np.random.default_rng(10_000 * seed + 5)
    x0 = F * np.ones(N) + rng.normal(size=N)
    data = integrate(x0, args.n_train + args.n_test + 2)
    tr = data[:args.n_train + 1]
    Z = (data - tr.mean(0)) / tr.std(0)
    Ztr, Zte = Z[:args.n_train + 1], Z[-(args.n_test + 1):]
    Xte, Yte = Zte[:-1], Zte[1:]
    parent = M.train(M.init_params(rng, d=N), Ztr[:-1], Ztr[1:], epochs=args.epochs, rng=rng)
    pmse = M.mse(parent, Xte, Yte)

    # ---- parent validity (same rule as the paper)
    P0 = M.stack_params([parent])
    r0 = lle_batch(P0, Xte[100], DT)
    spread = r0["samples"].std(axis=1).mean(axis=1)[0]
    valid = bool(r0["alive"][0] and spread >= 0.05 and np.isfinite(r0["lle"][0])
                 and abs(r0["lle_h1"][0] - r0["lle_h2"][0]) <= 0.25)
    info = dict(seed=seed, parent_mse=pmse, parent_lle=float(r0["lle"][0]), valid=valid)
    if not valid:
        return None, None, info

    H = args.vpt_horizon
    offsets = np.linspace(0, len(Zte) - H - 2, args.n_ic).astype(int)
    vpt0 = float(vpt_batch(P0, Zte, offsets, H, DT)[0] * ref_lle)
    info["parent_vpt_lt"] = vpt0

    def evaluate(plist):
        P = M.stack_params(plist)
        mses = np.array([M.mse(p, Xte, Yte) for p in plist])
        div, col = fail_batch(P, Xte[100])
        vpt = vpt_batch(P, Zte, offsets, H, DT) * ref_lle
        return mses, div, col, vpt

    # ---- main design (structured + matched controls + grid)
    perts = make_perturbations(parent, rng, K=args.K)
    mses, div, col, vpt = evaluate([m["params"] for m in perts])
    main = pd.DataFrame([dict(
        system="lorenz96", seed=seed, role=m["role"], family=m["family"], level=m["level"],
        group=m["group"], control_of=m["control_of"], draw=m["draw"], rel_norm=m["rel_norm"],
        mse=mses[i], mse_parent=pmse, diverged=bool(div[i]), collapsed=bool(col[i]),
        vpt_lt=float(vpt[i]), parent_vpt_lt=vpt0) for i, m in enumerate(perts)])

    # ---- tensor-isolated interventions
    models = build_models(parent, np.random.default_rng(777_000 + seed), args.K)
    mses, div, col, vpt = evaluate([m["params"] for m in models])
    ten = pd.DataFrame([dict(
        system="lorenz96", seed=seed, design=m["design"], group=m["group"], r=m["r"], draw=m["draw"],
        rel_norm=m["rel_norm"], r_W3=m["r_W3"],
        log10_mse_ratio=float(np.log10(max(mses[i], 1e-300) / pmse)) if np.isfinite(mses[i]) else np.inf,
        diverged=bool(div[i]), collapsed=bool(col[i]), fail=bool(div[i] | col[i]),
        vpt_lt=float(vpt[i]), vpt_loss=vpt0 - float(vpt[i]), parent_vpt_lt=vpt0)
        for i, m in enumerate(models)])
    return main, ten, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--vpt_horizon", type=int, default=500)
    ap.add_argument("--n_ic", type=int, default=20)
    ap.add_argument("--out", default="results_l96")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args) | dict(N=N, F=F, dt=DT), open(os.path.join(args.out, "config.json"), "w"), indent=2)
    refp = os.path.join(args.out, "reference_lle.json")
    if os.path.exists(refp):
        ref = json.load(open(refp))["lorenz96"]
    else:
        t0 = time.time()
        ref = reference_lle()
        json.dump({"lorenz96": ref}, open(refp, "w"))
        print(f"reference LLE (N={N}, F={F}) = {ref:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    infos = []
    for seed in range(args.seed0, args.seed0 + args.seeds):
        pm, pt = os.path.join(args.out, f"main_seed{seed}.csv"), os.path.join(args.out, f"tensor_seed{seed}.csv")
        ip = os.path.join(args.out, f"info_seed{seed}.json")
        if os.path.exists(ip):
            continue
        t0 = time.time()
        main_df, ten_df, info = run_seed(seed, args, ref)
        if main_df is not None:
            main_df.to_csv(pm, index=False)
            ten_df.to_csv(pt, index=False)
        json.dump(info, open(ip, "w"))
        print(f"[lorenz96 seed {seed}] valid={info['valid']} parent mse={info['parent_mse']:.2e} "
              f"lle={info['parent_lle']:.3f} (ref {ref:.3f}) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
