"""Tensor-isolated interventions: a causal test of Section 4.5 (which tensors carry the difference?).

Uses the repo's own parents (rebuilt deterministically; parent MSE verified against the stored CSV)
and the repo's own VPT estimator and failure criteria, so results are directly comparable with the paper.

For every valid parent, noise is injected into ONE tensor group at a time:

  design "total": noise confined to group g, rescaled so ||d theta|| / ||theta|| (all parameters) = r
                  -> the paper's matched-total-norm metric
  design "own"  : noise confined to group g, rescaled so ||d_g|| / ||theta_g|| = r
                  -> per-tensor sensitivity (same *relative* displacement of each tensor)
  design "loo"  : isotropic noise with group g REMOVED (leave-one-out), rescaled to total norm r
                  -> if W3 is the culprit, removing it should rescue isotropic noise

Groups: W1, W2, W3, biases (b1,b2,b3).  Control: isotropic noise over all parameters ("all").

Per model: failure (diverged / collapsed), one-step MSE ratio to parent, VPT loss (Lyapunov times).

    python run_tensor_intervention.py --system lorenz --out results_tensor
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.dynamics import vpt_batch
from jacsens.model import KEYS, step_batch_states
from recompute_extra import rebuild

GROUPS = {"W1": ("W1",), "W2": ("W2",), "W3": ("W3",), "biases": ("b1", "b2", "b3")}
NORMS_TOTAL = (0.01, 0.02, 0.05, 0.1, 0.2)
NORMS_OWN = (0.02, 0.05, 0.1, 0.2, 0.5)


def flat_norm(p, keys=KEYS):
    return float(np.sqrt(sum(np.sum(p[k] ** 2) for k in keys)))


def noisy(parent, rng, keys, target):
    z = {k: rng.normal(size=parent[k].shape) for k in keys}
    nz = np.sqrt(sum(np.sum(v ** 2) for v in z.values()))
    q = M.copy_params(parent)
    for k in keys:
        q[k] = q[k] + z[k] * (target / nz)
    return q


def build_models(parent, rng, K):
    tot = flat_norm(parent)
    out = []

    def rec(design, group, r, draw, q):
        d2 = {k: float(np.sum((q[k] - parent[k]) ** 2)) for k in KEYS}
        out.append(dict(design=design, group=group, r=r, draw=draw, params=q,
                        rel_norm=float(np.sqrt(sum(d2.values())) / tot),
                        r_W3=float(np.sqrt(d2["W3"]) / np.linalg.norm(parent["W3"]))))

    for r in NORMS_TOTAL:
        for k in range(K):
            rec("total", "all", r, k, noisy(parent, rng, KEYS, r * tot))
            for g, keys in GROUPS.items():
                rec("total", g, r, k, noisy(parent, rng, keys, r * tot))
                rest = tuple(x for x in KEYS if x not in keys)
                rec("loo", f"all_but_{g}", r, k, noisy(parent, rng, rest, r * tot))
    for r in NORMS_OWN:
        for k in range(K):
            for g, keys in GROUPS.items():
                rec("own", g, r, k, noisy(parent, rng, keys, r * flat_norm(parent, keys)))
    return out


def fail_batch(P, x0, n_transient=1000, n_steps=10000, bound=50.0, stride=5, collapse_std=0.05):
    """Same failure criteria as lle_batch (no Jacobians needed). Returns (diverged, collapsed)."""
    Mn, d = P["W1"].shape[0], x0.shape[0]
    S = np.tile(x0[None, None, :], (Mn, 1, 1))
    alive = np.ones(Mn, bool)
    keep = np.zeros((Mn, n_steps // stride, d))
    with np.errstate(all="ignore"):
        for t in range(n_transient + n_steps):
            S = step_batch_states(P, S)
            bad = ~np.isfinite(S).all(axis=(1, 2)) | (np.abs(S).max(axis=(1, 2)) > bound)
            alive &= ~bad
            S = np.where(alive[:, None, None], S, 0.0)
            j = t - n_transient
            if j >= 0 and j % stride == 0 and j // stride < keep.shape[1]:
                keep[:, j // stride] = S[:, 0]
    spread = keep.std(axis=1).mean(axis=1)
    return ~alive, alive & (spread < collapse_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["lorenz", "rossler"])
    ap.add_argument("--results", default="results_full_30seeds")
    ap.add_argument("--out", default="results_tensor")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--max_half_diff", type=float, default=0.25)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.results, "config.json")))
    ref_lle = json.load(open(os.path.join(a.results, "reference_lle.json")))[a.system]
    os.makedirs(a.out, exist_ok=True)
    seeds = a.seeds if a.seeds is not None else range(cfg["seed0"], cfg["seed0"] + cfg["seeds"])

    for seed in seeds:
        path = os.path.join(a.out, f"{a.system}_seed{seed}.csv")
        if os.path.exists(path):
            continue
        raw = pd.read_csv(os.path.join(a.results, "raw", f"{a.system}_seed{seed}.csv"))
        par = raw.iloc[0]
        valid = bool(par.alive and not par.collapsed and np.isfinite(par.lle)
                     and abs(par.lle_h1 - par.lle_h2) <= a.max_half_diff)
        if not valid:
            print(f"[{a.system} seed {seed}] parent not valid -> skipped", flush=True)
            continue
        t0 = time.time()
        parent, _perts, Zte, dt = rebuild(a.system, seed, cfg)   # perts unused; same RNG path for the parent
        Xte, Yte = Zte[:-1], Zte[1:]
        assert abs(M.mse(parent, Xte, Yte) - par.mse) < 1e-12 * max(1.0, par.mse) + 1e-14, "parent mismatch"

        rng = np.random.default_rng(777_000 + 100 * seed + (a.system == "rossler"))
        models = build_models(parent, rng, a.K)
        P = M.stack_params([m["params"] for m in models])
        mses = np.array([M.mse(m["params"], Xte, Yte) for m in models])
        div, col = fail_batch(P, Xte[100], cfg["n_transient"], cfg["n_steps"])
        H = cfg["vpt_horizon"]
        offsets = np.linspace(0, len(Zte) - H - 2, cfg["n_ic"]).astype(int)
        vpt = vpt_batch(P, Zte, offsets, H, dt) * ref_lle
        P0 = M.stack_params([parent])
        vpt0 = float(vpt_batch(P0, Zte, offsets, H, dt)[0] * ref_lle)

        rows = [dict(system=a.system, seed=seed, design=m["design"], group=m["group"], r=m["r"],
                     draw=m["draw"], rel_norm=m["rel_norm"], r_W3=m["r_W3"],
                     log10_mse_ratio=float(np.log10(max(mses[i], 1e-300) / par.mse)) if np.isfinite(mses[i]) else np.inf,
                     diverged=bool(div[i]), collapsed=bool(col[i]), fail=bool(div[i] | col[i]),
                     vpt_lt=float(vpt[i]), vpt_loss=vpt0 - float(vpt[i]), parent_vpt_lt=vpt0)
                for i, m in enumerate(models)]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{a.system} seed {seed}] {len(rows)} models, parent VPT {vpt0:.2f} LT, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
