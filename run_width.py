"""Width generality of the tensor-intervention picture and of the first-order theory.

Trains NEW residual-MLP parents with hidden width H (same recipe as the paper otherwise: Adam 1e-3, batch 128,
50 epochs, 20k transitions) on Lorenz-63 / Rossler, applies the total-norm and own-norm designs of
run_tensor_intervention.py (leave-one-out omitted) with K draws, and attaches the analytic first-order
prediction of run_theory.py, writing files in the format analyze_theory.py reads.

Parent validity here: the parent's own rollout must neither diverge nor collapse (the half-run Lyapunov
agreement check of the main study is not applied).

    python run_width.py --system lorenz --H 128 --out results_width/H128
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.dynamics import vpt_batch
from jacsens.model import KEYS
from jacsens.systems import SYSTEMS, initial_condition, integrate
from run_tensor_intervention import build_models, fail_batch
from run_theory import predict, sensitivities


def make_parent(system, seed, H, cfg):
    dt = SYSTEMS[system]["dt"]
    rng = np.random.default_rng(10_000 * seed + sorted(SYSTEMS).index(system) + 7 * H)
    data = integrate(system, initial_condition(system, rng), cfg["n_train"] + cfg["n_test"] + 2, dt_sample=dt,
                     h=cfg["h"], transient=cfg["transient"])
    tr = data[:cfg["n_train"] + 1]
    Z = (data - tr.mean(0)) / tr.std(0)
    Ztr, Zte = Z[:cfg["n_train"] + 1], Z[-(cfg["n_test"] + 1):]
    parent = M.train(M.init_params(rng, H=H), Ztr[:-1], Ztr[1:], epochs=cfg["epochs"], lr=cfg["lr"],
                     batch=cfg["batch"], rng=rng)
    return parent, Zte, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["lorenz", "rossler"])
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--results", default="results_full_30seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.results, "config.json")))
    ref = json.load(open(os.path.join(a.results, "reference_lle.json")))[a.system]
    os.makedirs(a.out, exist_ok=True)
    for seed in a.seeds:
        path = os.path.join(a.out, f"{a.system}_seed{seed}.csv")
        if os.path.exists(path):
            continue
        t0 = time.time()
        parent, Zte, dt = make_parent(a.system, seed, a.H, cfg)
        Xte, Yte = Zte[:-1], Zte[1:]
        P0 = M.stack_params([parent])
        d0, c0 = fail_batch(P0, Xte[100])
        if d0[0] or c0[0]:
            print(f"[{a.system} H={a.H} seed {seed}] parent fails -> skipped", flush=True)
            continue
        pmse = M.mse(parent, Xte, Yte)
        offsets = np.linspace(0, len(Zte) - cfg["vpt_horizon"] - 2, cfg["n_ic"]).astype(int)
        vpt0 = float(vpt_batch(P0, Zte, offsets, cfg["vpt_horizon"], dt)[0] * ref)
        rng = np.random.default_rng(555_000 + 100 * seed + a.H)
        models = [m for m in build_models(parent, rng, a.K) if m["design"] != "loo"]
        P = M.stack_params([m["params"] for m in models])
        mses = np.array([M.mse(m["params"], Xte, Yte) for m in models])
        div, col = fail_batch(P, Xte[100], cfg["n_transient"], cfg["n_steps"])
        vpt = vpt_batch(P, Zte, offsets, cfg["vpt_horizon"], dt) * ref
        S = sensitivities(parent, Xte)
        tot = float(np.sqrt(sum(np.sum(parent[k] ** 2) for k in KEYS)))
        rows = []
        for i, m in enumerate(models):
            base = dict(design=m["design"], group=m["group"], r=m["r"])
            dpred, tgt = predict(base, parent, S, tot)
            lr = float(np.log10(max(mses[i], 1e-300) / pmse)) if np.isfinite(mses[i]) else np.nan
            row = dict(system=a.system, seed=seed, H=a.H, design=m["design"], group=m["group"], r=m["r"], draw=m["draw"],
                       rel_norm=m["rel_norm"], r_W3=m["r_W3"], log10_mse_ratio=lr, diverged=bool(div[i]),
                       collapsed=bool(col[i]), fail=bool(div[i] | col[i]), vpt_lt=float(vpt[i]),
                       vpt_loss=vpt0 - float(vpt[i]), parent_vpt_lt=vpt0, dmse_pred=dpred, noise_norm=tgt,
                       mse_parent=pmse, dmse_meas=pmse * (10 ** lr - 1) if np.isfinite(lr) else np.nan)
            for k in KEYS:
                row[f"S_{k}"] = S[k]
                row[f"norm_{k}"] = float(np.linalg.norm(parent[k]))
                row[f"n_{k}"] = parent[k].size
            row["norm_total"] = tot
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{a.system} H={a.H} seed {seed}] {len(rows)} models {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
