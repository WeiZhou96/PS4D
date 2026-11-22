import numpy as np

from ps4d.metrics import best_affine_ratio, blend_beats_better, blend_identity, bootstrap_medians, median_ci, scores


def test_no_motion_estimate_has_ratio_one():
    ref = np.sin(np.linspace(0, 3, 40)) * 0.1 + 0.7
    s = scores(np.full_like(ref, 2.0), ref)
    assert np.isclose(s["r"], 1.0) and np.isclose(s["beta"], 0.0)


def test_perfect_and_scaled_estimates():
    ref = np.cumsum(np.random.default_rng(0).normal(size=50)) * 0.01
    s = scores(ref + 3.0, ref)                                   # an additive offset does not matter
    assert np.isclose(s["r"], 0.0, atol=1e-12) and np.isclose(s["beta"], 1.0)
    s = scores(0.5 * ref, ref)                                   # compressed motion
    assert np.isclose(s["beta"], 0.5) and np.isclose(s["r"], 0.5)


def test_ratio_is_norm_ratio_of_centred_errors():
    rng = np.random.default_rng(1)
    ref, x = rng.normal(size=30), rng.normal(size=30)
    d, l = x - x.mean(), ref - ref.mean()
    assert np.isclose(scores(x, ref)["r"], np.linalg.norm(d - l) / np.linalg.norm(l))
    assert np.isclose(scores(x, ref)["beta"], d @ l / (l @ l))


def test_equal_blend_identities():
    """Proposition 2: r_1/2^2 = (r1^2 + r2^2 + 2 c r1 r2) / 4 and beta_1/2 = (beta1 + beta2) / 2, exactly."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        ref = np.cumsum(rng.normal(size=60)) * 0.02
        x1 = ref * rng.uniform(0.3, 1.5) + rng.normal(size=60) * 0.03
        x2 = ref * rng.uniform(0.3, 1.5) + rng.normal(size=60) * 0.03
        s1, s2, sb = scores(x1, ref), scores(x2, ref), scores(0.5 * (x1 + x2), ref)
        e1 = (x1 - x1.mean()) - (ref - ref.mean())
        e2 = (x2 - x2.mean()) - (ref - ref.mean())
        c = e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))
        assert np.isclose(sb["r"], blend_identity(s1["r"], s2["r"], c))
        assert np.isclose(sb["beta"], 0.5 * (s1["beta"] + s2["beta"]))
        lo, hi = min(s1["r"], s2["r"]), max(s1["r"], s2["r"])
        assert (sb["r"] < lo) == blend_beats_better(s1["r"], s2["r"], c)
        assert sb["r"] <= 0.5 * (s1["r"] + s2["r"]) + 1e-12   # never worse than the mean of the inputs


def test_best_affine_weight():
    rng = np.random.default_rng(3)
    ref = np.cumsum(rng.normal(size=80)) * 0.02
    x1, x2 = ref + rng.normal(size=80) * 0.03, ref + rng.normal(size=80) * 0.05
    r1, r2 = scores(x1, ref)["r"], scores(x2, ref)["r"]
    e1 = (x1 - x1.mean()) - (ref - ref.mean())
    e2 = (x2 - x2.mean()) - (ref - ref.mean())
    c = e1 @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))
    best = min(scores(w * x1 + (1 - w) * x2, ref)["r"] for w in np.linspace(-1, 2, 30001))
    assert np.isclose(best_affine_ratio(r1, r2, c), best, rtol=1e-5)
    assert np.isclose(best_affine_ratio(0.5, 0.5, c), blend_identity(0.5, 0.5, c))   # equal ratios: equal weight


def test_bootstrap_is_seeded_and_clustered():
    d = np.random.default_rng(4).normal(size=100)
    assert bootstrap_medians(d) == bootstrap_medians(d)
    clips = [{"x": v, "sequence": i // 10} for i, v in enumerate(d)]
    m, lo, hi = median_ci(clips, lambda c: c["x"])
    assert lo <= m <= hi
