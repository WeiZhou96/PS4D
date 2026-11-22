import numpy as np

from ps4d import fusion, graph


def _chain(L, W=6, H=18, noise=0.0, rng=None):
    meas = []
    for a in range(0, len(L) - 1, W):
        for k in range(a + 1, min(a + H, len(L) - 1) + 1):
            m = L[k] - L[a] + (rng.normal() * noise if rng is not None else 0.0)
            meas.append({"a": a, "k": k, "m": m})
    return meas


def test_exact_measurements_recover_log_depth_up_to_gauge():
    L = 0.05 * np.sin(np.linspace(0, 4, 60))
    sol, res = graph.solve(_chain(L), np.full(len(_chain(L)), 1e-4))
    est = np.array([sol[k] for k in range(60)])
    assert np.allclose(est - est[0], L - L[0], atol=1e-6)
    assert abs(sol[0]) < 1e-6                                   # gauge l = 0 at the first node


def test_huber_downweights_outliers():
    rng = np.random.default_rng(0)
    L = 0.05 * np.sin(np.linspace(0, 4, 60))
    meas = _chain(L, noise=0.002, rng=rng)
    hit = set()
    for i in rng.permutation(len(meas)):                        # one gross error on frames seen by 3 anchors
        k = meas[i]["k"]
        if 18 <= k < 54 and k not in hit and len(hit) < 15:
            meas[i]["m"] += 0.5
            hit.add(k)
    var = np.full(len(meas), 0.002 ** 2)
    robust, _ = graph.solve(meas, var)
    plain, _ = graph.solve(meas, var, delta=1e9)
    err = lambda s: np.std([s[k] - L[k] - (s[0] - L[0]) for k in range(60)])  # noqa: E731
    assert err(robust) < 0.5 * err(plain)


def test_largest_component_only():
    meas = [{"a": 0, "k": 1, "m": 0.1}, {"a": 1, "k": 2, "m": 0.1}, {"a": 10, "k": 11, "m": 0.3}]
    sol, _ = graph.solve(meas)
    assert set(sol) == {0, 1, 2}
    assert np.isclose(sol[2] - sol[0], 0.2)


def test_posterior_covariance_shape():
    L = np.linspace(0, 0.1, 30)
    meas = _chain(L)
    sol, _, (nodes, S) = graph.solve(meas, np.full(len(meas), 1e-4), cov=True)
    assert S.shape == (len(nodes), len(nodes)) and np.all(np.diag(S) >= 0)


def test_robust_average():
    x = np.cumsum(np.random.default_rng(1).normal(size=40)) * 0.01
    meas = [(a, k, x[k] - x[a]) for a in range(0, 39, 6) for k in range(a + 1, min(a + 18, 39) + 1)]
    sol = graph.robust_average(meas)
    assert np.allclose([sol[k] - sol[0] for k in range(40)], x - x[0], atol=1e-9)


def test_blend_and_4d_branch():
    T, H, W = 5, 40, 50
    depth = np.full((T, H, W), np.nan, np.float32)
    masks = np.zeros((T, H, W), bool)
    masks[:, 10:30, 10:40] = True
    for t in range(T):
        depth[t] = 1.0 + 0.1 * t
    obs = fusion.depth4d_observations(depth, masks, range(T))
    b4 = fusion.branch_4d(obs)
    assert np.allclose([b4[t] for t in range(T)], np.log(1.0 + 0.1 * np.arange(T)), atol=1e-6)
    ps = {t: 0.02 * t for t in range(1, T)}
    bl = fusion.blend(ps, b4)
    assert set(bl) == set(range(1, T)) and np.isclose(bl[2], 0.5 * ps[2] + 0.5 * b4[2])
    small = np.zeros_like(masks)
    small[:, :3, :3] = True                                     # fewer than 20 pixels after erosion
    assert fusion.depth4d_observations(depth, small, range(T)) == []
