"""Log-depth graph: frames are nodes, measurements of l(k) - l(a) are edges (Eq. 5 of the paper).

solve()           Huber-robust weighted least squares by iteratively reweighted least squares on the largest connected
                  component, residuals normalised by the measurement standard deviations (threshold 2), gauge l = 0 at
                  the first node.  This gives the projective-scale branch.
robust_average()  the same graph with equal weights and a Huber threshold in log units, for raw (a, k, m) triples
                  (used by the unweighted size-cue ablation).
"""
from collections import defaultdict

import numpy as np


def _largest_component(pairs):
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, k in pairs:
        par[find(a)] = find(k)
    comp = defaultdict(list)
    for x in list(par):
        comp[find(x)].append(x)
    return sorted(max(comp.values(), key=len))


def solve(meas, var=None, delta=2.0, iters=8, cov=False):
    """meas: dicts with a, k, m.  var: per-measurement variance (None = equal weights).

    Returns ({frame: l}, residuals) and, with cov=True, also (nodes, posterior covariance of l under the final
    weights; relative to the gauge)."""
    if not meas:
        return ({}, [], ([], np.zeros((0, 0)))) if cov else ({}, [])
    nodes = _largest_component([(m["a"], m["k"]) for m in meas])
    ix = {n: i for i, n in enumerate(nodes)}
    use = [i for i, m in enumerate(meas) if m["a"] in ix and m["k"] in ix]
    A = np.zeros((len(use) + 1, len(nodes)))
    b = np.zeros(len(use) + 1)
    sd = np.ones(len(use) + 1)
    for r, i in enumerate(use):
        A[r, ix[meas[i]["k"]]], A[r, ix[meas[i]["a"]]], b[r] = 1.0, -1.0, meas[i]["m"]
        sd[r] = np.sqrt(var[i]) if var is not None else 0.01
    A[-1, 0], sd[-1] = 1.0, 1e-3                      # gauge
    w = 1.0 / sd
    for _ in range(iters):
        s = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        z = np.abs(A @ s - b) / sd
        w = np.where(z <= delta, 1.0, np.sqrt(delta / np.maximum(z, 1e-12))) / sd
        w[-1] = 1.0 / sd[-1]
    res = (A @ s - b)[:-1]
    sol = {n: float(s[ix[n]]) for n in nodes}
    if cov:
        Aw = A * w[:, None]
        return sol, res, (nodes, np.linalg.pinv(Aw.T @ Aw))
    return sol, res


def robust_average(meas, delta=0.02, iters=8):
    """meas: (a, k, m) with m ~ x_k - x_a.  Equal-weight least squares with Huber weights (threshold delta in the
    units of m) on the largest connected component, gauge x = 0 at the first node.  Returns {frame: x}."""
    if not meas:
        return {}
    nodes = _largest_component([(a, k) for a, k, _ in meas])
    ix = {n: i for i, n in enumerate(nodes)}
    M = [(ix[a], ix[k], m) for a, k, m in meas if a in ix and k in ix]
    A = np.zeros((len(M) + 1, len(nodes)))
    b = np.zeros(len(M) + 1)
    for r, (i, j, m) in enumerate(M):
        A[r, j], A[r, i], b[r] = 1.0, -1.0, m
    A[-1, 0] = 1.0                                    # gauge
    w = np.ones(len(b))
    for _ in range(iters):
        s = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)[0]
        res = np.abs(A @ s - b)
        w = np.sqrt(np.where(res <= delta, 1.0, delta / np.maximum(res, 1e-12)))
        w[-1] = 1.0
    return {n: float(s[ix[n]]) for n in nodes}
