"""Does protecting the output layer help real compression?  (practical test of the tensor-intervention result)

For every valid parent, pruning / quantization is applied with three scopes:
  all      : every tensor (as in the paper)
  protOut  : every tensor EXCEPT the output layer (W3, b3), which stays at full precision / unpruned
  onlyOut  : ONLY the output layer (W3, b3) is compressed
Families: global magnitude pruning, layer-wise magnitude pruning (25/50/75%), per-tensor
symmetric quantization (8/6/4/3 bit).  30 models per parent, no random draws (deterministic).

    python run_protect.py --system lorenz|rossler|lorenz96 --out results_protect
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
from jacsens.perturb import rel_norm
from run_tensor_intervention import fail_batch

OUT = ("W3", "b3")
SCOPES = {"all": KEYS, "protOut": tuple(k for k in KEYS if k not in OUT), "onlyOut": OUT}


def quant_keys(p, bits, keys):
    q, qmax = M.copy_params(p), 2 ** (bits - 1) - 1
    for k in keys:
        amax = np.abs(p[k]).max()
        scale = amax / qmax if amax > 0 else 1.0
        q[k] = np.clip(np.round(p[k] / scale), -qmax, qmax) * scale
    return q


def prune_global_keys(p, s, keys):
    wk = [k for k in keys if k.startswith("W")]
    q = M.copy_params(p)
    if not wk:
        return q
    allw = np.concatenate([np.abs(p[k]).ravel() for k in wk])
    kth = int(round(s * allw.size))
    mask = np.ones(allw.size, bool)
    mask[np.argsort(allw, kind="stable")[:kth]] = False
    i = 0
    for k in wk:
        n = p[k].size
        q[k] = (p[k].ravel() * mask[i:i + n]).reshape(p[k].shape)
        i += n
    return q


def prune_layer_keys(p, s, keys):
    q = M.copy_params(p)
    for k in (k for k in keys if k.startswith("W")):
        w = np.abs(p[k]).ravel()
        mask = np.ones(w.size, bool)
        mask[np.argsort(w, kind="stable")[:int(round(s * w.size))]] = False
        q[k] = (p[k].ravel() * mask).reshape(p[k].shape)
    return q


def variants(parent):
    out = []
    for scope, keys in SCOPES.items():
        for s in (0.25, 0.5, 0.75):
            out.append(("prune_global", s, scope, prune_global_keys(parent, s, keys)))
            out.append(("prune_layer", s, scope, prune_layer_keys(parent, s, keys)))
        for b in (8, 6, 4, 3):
            out.append(("quant", b, scope, quant_keys(parent, b, keys)))
    return out


def get_parent(system, seed, results):
    """Return (parent, Zte, dt, ref_lle, parent_mse, valid)."""
    if system == "lorenz96":
        import run_l96 as L
        cfg = json.load(open(os.path.join(results["l96"], "config.json")))
        info = json.load(open(os.path.join(results["l96"], f"info_seed{seed}.json")))
        rng = np.random.default_rng(10_000 * seed + 5)
        x0 = L.F * np.ones(L.N) + rng.normal(size=L.N)
        data = L.integrate(x0, cfg["n_train"] + cfg["n_test"] + 2)
        tr = data[:cfg["n_train"] + 1]
        Z = (data - tr.mean(0)) / tr.std(0)
        Ztr, Zte = Z[:cfg["n_train"] + 1], Z[-(cfg["n_test"] + 1):]
        parent = M.train(M.init_params(rng, d=L.N), Ztr[:-1], Ztr[1:], epochs=cfg["epochs"], rng=rng)
        ref = json.load(open(os.path.join(results["l96"], "reference_lle.json")))["lorenz96"]
        assert abs(M.mse(parent, Zte[:-1], Zte[1:]) - info["parent_mse"]) < 1e-12 * max(1, info["parent_mse"]) + 1e-14
        return parent, Zte, L.DT, ref, info["parent_mse"], bool(info["valid"])
    from recompute_extra import rebuild
    cfg = json.load(open(os.path.join(results["main"], "config.json")))
    ref = json.load(open(os.path.join(results["main"], "reference_lle.json")))[system]
    raw = pd.read_csv(os.path.join(results["main"], "raw", f"{system}_seed{seed}.csv"))
    par = raw.iloc[0]
    valid = bool(par.alive and not par.collapsed and np.isfinite(par.lle) and abs(par.lle_h1 - par.lle_h2) <= 0.25)
    if not valid:
        return None, None, None, ref, None, False
    parent, _, Zte, dt = rebuild(system, seed, cfg)
    assert abs(M.mse(parent, Zte[:-1], Zte[1:]) - par.mse) < 1e-12 * max(1, par.mse) + 1e-14
    return parent, Zte, dt, ref, float(par.mse), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["lorenz", "rossler", "lorenz96"])
    ap.add_argument("--main", default="results_full_30seeds")
    ap.add_argument("--l96", default="results_l96")
    ap.add_argument("--out", default="results_protect")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(30)))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = dict(main=a.main, l96=a.l96)
    for seed in a.seeds:
        path = os.path.join(a.out, f"{a.system}_seed{seed}.csv")
        if os.path.exists(path):
            continue
        if a.system == "lorenz96" and not os.path.exists(os.path.join(a.l96, f"info_seed{seed}.json")):
            continue
        t0 = time.time()
        parent, Zte, dt, ref, pmse, valid = get_parent(a.system, seed, res)
        if not valid:
            print(f"[{a.system} seed {seed}] parent not valid -> skipped", flush=True)
            continue
        Xte, Yte = Zte[:-1], Zte[1:]
        vs = variants(parent)
        P = M.stack_params([v[3] for v in vs])
        mses = np.array([M.mse(v[3], Xte, Yte) for v in vs])
        div, col = fail_batch(P, Xte[100])
        offsets = np.linspace(0, len(Zte) - 502, 20).astype(int)
        vpt = vpt_batch(P, Zte, offsets, 500, dt) * ref
        vpt0 = float(vpt_batch(M.stack_params([parent]), Zte, offsets, 500, dt)[0] * ref)
        rows = [dict(system=a.system, seed=seed, family=v[0], level=v[1], scope=v[2],
                     rel_norm=rel_norm(parent, v[3]),
                     log10_mse_ratio=float(np.log10(max(mses[i], 1e-300) / pmse)) if np.isfinite(mses[i]) else np.nan,
                     diverged=bool(div[i]), collapsed=bool(col[i]), fail=bool(div[i] | col[i]),
                     vpt_lt=float(vpt[i]), vpt_loss=vpt0 - float(vpt[i]), parent_vpt_lt=vpt0)
                for i, v in enumerate(vs)]
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{a.system} seed {seed}] {len(rows)} models {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
