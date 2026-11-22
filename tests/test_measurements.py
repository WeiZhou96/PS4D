"""Proposition 1 (projective scale) on synthetic rigid motion, and the measurement / feature layout."""
import cv2
import numpy as np

from ps4d.measurements import FEATURES, feature_matrix, measurements, pair_log_ratios


def _rot(axis, ang):
    return cv2.Rodrigues(np.asarray(axis, float) * ang)[0]


def _project(X, f, p):
    return f * X[:, :2] / X[:, 2:3] + p


def _object(n=40, seed=0):
    return np.random.default_rng(seed).uniform(-0.05, 0.05, size=(n, 3))


def test_size_ratio_independent_of_focal_principal_point_and_scale():
    """(i): the log distance ratio does not depend on f, p, or a common scaling of object and trajectory."""
    P = _object()
    Ta, Tk = np.array([0.02, -0.01, 0.60]), np.array([0.05, 0.03, 0.75])
    Rk = _rot([0, 0, 1], 0.3)
    vals = []
    for f, p, s in ((500.0, (320, 240), 1.0), (900.0, (100, 50), 1.0), (500.0, (320, 240), 2.5)):
        Xa, Xk = s * (P + Ta), s * (P @ Rk.T + Tk)
        vals.append(np.median(pair_log_ratios(_project(Xa, f, p), _project(Xk, f, p))))
    assert np.allclose(vals, vals[0], atol=1e-9)


def test_optical_axis_rotation_gives_identical_ratios_under_weak_perspective():
    """(ii): rotation about the optical axis -> every pairwise ratio equals -(l_k - l_a) (weak perspective)."""
    P = _object()
    Za, Zk = 0.6, 0.8
    Rk = _rot([0, 0, 1], 0.7)
    ua = 600.0 / Za * P[:, :2]                                        # weak perspective at the object depth
    uk = 600.0 / Zk * (P @ Rk.T)[:, :2]
    lr = pair_log_ratios(ua, uk)
    assert np.allclose(lr, -(np.log(Zk) - np.log(Za)), atol=1e-12)


def test_rigid_measurement_independent_of_shape_scale():
    """(iii): posing an up-to-scale shape gives a depth ratio that does not depend on the scale s."""
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    P = _object(60, 1)
    Xa = P + [0.0, 0.0, 0.6]
    Rk = _rot([1, 1, 0], 0.4)
    Xk = P @ Rk.T + [0.03, -0.02, 0.85]
    Uk = _project(Xk, 600.0, (320, 240))
    out = []
    for s in (1.0, 3.0, 0.2):
        Xs = s * Xa * 1000.0
        ok, rv, tv = cv2.solvePnP(Xs, Uk, K, None, flags=cv2.SOLVEPNP_SQPNP)
        zk = (Xs @ cv2.Rodrigues(rv)[0].T + tv.T)[:, 2]
        out.append(np.log(np.median(zk)) - np.log(np.median(Xs[:, 2])))
    assert np.allclose(out, out[0], atol=1e-6)
    assert np.isclose(out[0], np.log(np.median(Xk[:, 2])) - np.log(np.median(Xa[:, 2])), atol=1e-4)


def test_measurements_on_synthetic_tracks():
    """Both measurement types recover the log-depth change of a rigidly moving object."""
    T, K = 30, np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    P = _object(40, 2)
    Z = 0.6 + 0.004 * np.arange(T)
    X = np.stack([P @ _rot([0, 0, 1], 0.01 * t).T + [0.0, 0.0, Z[t]] for t in range(T)])
    U = np.stack([_project(X[t], 600.0, (320, 240)) for t in range(T)]).astype(np.float32)
    query_frame = np.repeat([0, 6, 12], 40)
    tracks = np.full((T, 120, 2), np.nan, np.float32)
    visible = np.zeros((T, 120), bool)
    for i, a in enumerate((0, 6, 12)):
        sl = slice(40 * i, 40 * (i + 1))
        tracks[a:a + 19, sl] = U[a:a + 19]
        visible[a:a + 19, sl] = True
    mono = np.zeros((T, 480, 640), np.float32)
    for a in (0, 6, 12):
        mono[a] = Z[a]                                                # a fronto-parallel plane at the object depth
    meas = measurements(tracks, visible, query_frame, mono, K, horizon=18)
    assert {m["branch"] for m in meas} == {"size", "rigid"}
    for m in meas:
        true = np.log(np.median(X[m["k"], :, 2])) - np.log(np.median(X[m["a"], :, 2]))
        assert abs(m["m"] - true) < (0.01 if m["branch"] == "size" else 0.03)
    F = feature_matrix(meas)
    assert F.shape == (len(meas), 12) and len(FEATURES) == 12
    assert set(F[:, FEATURES.index("is_pnp")]) == {0.0, 1.0}
