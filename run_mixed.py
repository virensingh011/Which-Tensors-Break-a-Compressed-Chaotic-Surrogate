"""Does the first-order theory DESIGN better compression?  (theory-guided mixed-precision quantization)

For every valid parent and every average-bit budget b in {3, 4, 5} bits/parameter we compare, at the same total
number of bits (sum_k n_k b_k <= b * N):
  uniform : every tensor at b bits (the paper's quantization)
  theory  : bits per tensor chosen to minimise the PREDICTED one-step error increase
            dMSE = sum_k s_k * (amax_k / qmax(b_k))^2 / 12      (s_k: analytic sensitivities, run_theory.py;
            (amax/qmax)^2/12 is the variance of uniform rounding noise), by exact search over the 7^6 width combinations
  random  : 20 random per-tensor allocations with the same budget (matched control)
Bits per tensor are chosen among {2,...,8}; the six tensors W1,b1,W2,b2,W3,b3 get one width each.

    python run_mixed.py --system lorenz|rossler|lorenz96 --out results_mixed
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from jacsens import model as M
from jacsens.dynamics import vpt_batch
from jacsens.model import KEYS
from run_protect import get_parent
from run_tensor_intervention import fail_batch
from run_theory import sensitivities

BITS = range(2, 9)
BUDGETS = (3, 4, 5)
K_RANDOM = 20


def qmax(b):
    return 2 ** (b - 1) - 1


def quantize(p, bits):
    q = M.copy_params(p)
    for k in KEYS:
        amax = np.abs(p[k]).max()
        scale = amax / qmax(bits[k]) if amax > 0 else 1.0
        q[k] = np.clip(np.round(p[k] / scale), -qmax(bits[k]), qmax(bits[k])) * scale
    return q


def predicted_dmse(p, S, bits):
    return float(sum(S[k] * (np.abs(p[k]).max() / qmax(bits[k])) ** 2 / 12.0 for k in KEYS))


def total_bits(p, bits):
    return sum(p[k].size * bits[k] for k in KEYS)


def optimal_allocation(p, S, budget_bits):
    """Exact minimiser of the predicted error over all width combinations (7^6 = 117,649) within the bit budget."""
    import itertools
    grids = np.array(list(itertools.product(list(BITS), repeat=len(KEYS))))          # (C, 6)
    n = np.array([p[k].size for k in KEYS], float)
    a2 = np.array([S[k] * np.abs(p[k]).max() ** 2 / 12.0 for k in KEYS])
    q2 = (2.0 ** (grids - 1) - 1) ** 2
    cost = grids @ n
    pred = (a2[None, :] / q2).sum(1)
    pred[cost > budget_bits] = np.inf
    best = grids[int(np.argmin(pred))]
    return {k: int(b) for k, b in zip(KEYS, best)}


def random_allocations(p, budget_bits, K, rng):
    out, tries = [], 0
    while len(out) < K and tries < 200000:
        tries += 1
        b = {k: int(rng.integers(min(BITS), max(BITS) + 1)) for k in KEYS}
        tb = total_bits(p, b)
        if 0.96 * budget_bits <= tb <= budget_bits:
            out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["lorenz", "rossler", "lorenz96"])
    ap.add_argument("--main", default="results_full_30seeds")
    ap.add_argument("--l96", default="results_l96")
    ap.add_argument("--out", default="results_mixed")
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
        S = sensitivities(parent, Xte)
        N = sum(parent[k].size for k in KEYS)
        rng = np.random.default_rng(424_242 + 100 * seed)
        specs = []   # (budget, scheme, draw, bits)
        for bb in BUDGETS:
            B = bb * N
            specs.append((bb, "uniform", 0, {k: bb for k in KEYS}))
            specs.append((bb, "theory", 0, optimal_allocation(parent, S, B)))
            for j, b in enumerate(random_allocations(parent, B, K_RANDOM, rng)):
                specs.append((bb, "random", j, b))
        models = [quantize(parent, s[3]) for s in specs]
        P = M.stack_params(models)
        mses = np.array([M.mse(m, Xte, Yte) for m in models])
        div, col = fail_batch(P, Xte[100])
        offsets = np.linspace(0, len(Zte) - 502, 20).astype(int)
        vpt = vpt_batch(P, Zte, offsets, 500, dt) * ref
        vpt0 = float(vpt_batch(M.stack_params([parent]), Zte, offsets, 500, dt)[0] * ref)
        rows = []
        for i, (bb, scheme, j, b) in enumerate(specs):
            row = dict(system=a.system, seed=seed, budget=bb, scheme=scheme, draw=j,
                       avg_bits=total_bits(parent, b) / N, pred_dmse=predicted_dmse(parent, S, b),
                       mse_parent=pmse,
                       log10_mse_ratio=float(np.log10(max(mses[i], 1e-300) / pmse)) if np.isfinite(mses[i]) else np.nan,
                       diverged=bool(div[i]), collapsed=bool(col[i]), fail=bool(div[i] | col[i]),
                       vpt_lt=float(vpt[i]), vpt_loss=vpt0 - float(vpt[i]), parent_vpt_lt=vpt0)
            row.update({f"bits_{k}": b[k] for k in KEYS})
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{a.system} seed {seed}] {len(rows)} models {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
