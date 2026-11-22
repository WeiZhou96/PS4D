"""Measurement reliability learned from robot kinematics (Sec. 3.4 / Eq. 6 of the paper).

A gradient-boosted regressor g maps the eleven geometric statistics of a measurement (twelve input
columns, one duplicated) to its log squared error; the variance
is kappa * exp(g), with kappa set so that the median ratio of squared error to predicted variance is 0.455 (the median
of a chi-square variable with one degree of freedom).  The released model (weights/reliability_model.pkl) was trained
once on 134,274 measurements from 963 DROID grasps in 15 buildings with kinematically certified labels.

The model file is a pickle {model: sklearn HistGradientBoostingRegressor, calib: kappa, tau, feats}; it was written with
scikit-learn 1.8.0 and numpy 2.0.1, and loading it with another scikit-learn version may fail or change predictions.
"""
import pickle

import numpy as np

from . import graph
from .measurements import FEATURES, feature_matrix
from .utils import RELIABILITY_SHA256, default_reliability_model, sha256_file

# Statistics of a 4D-model observation (joint-graph ablation only): log mask area, area relative to the clip median,
# share of valid depth, IQR of log depth in the mask, share of mask pixels at the image border, mask-centroid step per
# frame (in units of sqrt(area)), log depth relative to the clip median, position in the clip.
FEATURES_4D = ("log_area", "area_rel", "valid_frac", "z_iqr", "border_frac", "ctr_step", "z_rel", "pos")

NU = 1e-6           # small constant inside the log squared error
CHI2_1_MEDIAN = 0.455


class VarianceModel:
    """Predicts per-measurement variances calib * exp(model(features))."""

    def __init__(self, payload, features):
        self.model = payload["model"]
        self.calib = float(payload["calib"])
        self.tau = payload.get("tau")
        self.features = tuple(features)
        stored = tuple(payload.get("feats", features))
        if stored != self.features:
            raise ValueError(f"model features {stored} differ from {self.features}")

    @classmethod
    def load(cls, path=None, features=FEATURES, expected_sha256=None):
        path = path or default_reliability_model()
        if expected_sha256 is None and features == FEATURES and path == default_reliability_model():
            expected_sha256 = RELIABILITY_SHA256
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise ValueError(f"{path}: sha256 differs from the released model {expected_sha256[:8]}...")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(payload, features)

    def variance(self, meas):
        return self.calib * np.exp(self.model.predict(feature_matrix(meas, self.features)))


def fit_regressor(X, y):
    """The regressor used for both variance models (300 rounds, 31 leaves, >= 200 samples per leaf)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0).fit(X, y)


def _ranks(v):
    return np.argsort(np.argsort(v))


def train_measurement_model(X, m, t, seg, seg_group, is_pnp, log=print):
    """Fit the measurement variance model on a training table (see scripts/extract_reliability_features.py).

    X: (n, 12) input columns (eleven statistics, zoom_px duplicates spread_px); m: measurements; t: certified
    targets (NaN where a frame has no label); seg: clip index of
    each row; seg_group: building of each clip.  Reports leave-building-out ranking quality, refits on all buildings,
    calibrates kappa and returns the payload {model, calib, tau, feats}.  tau (90th percentile over training clips of
    median predicted sd / sd of the solved log depth) is stored for completeness; PS4D does not use it.
    """
    grp = seg_group[seg]
    ok = np.isfinite(t)
    y = np.log((m - t) ** 2 + NU)
    groups = np.unique(grp[ok])
    folds = np.array_split(np.random.default_rng(0).permutation(groups), 5)
    pred = np.full(len(y), np.nan)
    for f in folds:
        te = ok & np.isin(grp, f)
        pred[te] = fit_regressor(X[ok & ~te], y[ok & ~te]).predict(X[te])
    e2 = (m - t) ** 2
    log(f"training set: {len(np.unique(seg))} clips, {len(groups)} buildings, {ok.sum()} measurements with a label")
    for b, name in ((0, "size"), (1, "rigid"), (None, "all")):
        s = ok & (True if b is None else (is_pnp == b))
        log(f"  {name}: leave-building-out Spearman(pred, err^2) {np.corrcoef(_ranks(pred[s]), _ranks(e2[s]))[0, 1]:+.3f}")
    dec = np.percentile(pred[ok], np.arange(0, 101, 10))
    for i in range(10):
        s = ok & (pred >= dec[i]) & (pred <= dec[i + 1])
        log(f"  predicted-variance decile {i + 1}: median |err| {np.median(np.sqrt(e2[s])):.4f}, "
            f"p90 {np.percentile(np.sqrt(e2[s]), 90):.4f}")
    model = fit_regressor(X[ok], y[ok])
    calib = float(np.median(e2[ok] / np.exp(model.predict(X[ok]))) / CHI2_1_MEDIAN)
    return {"model": model, "calib": calib, "feats": list(FEATURES)}


def clip_noise_threshold(payload, X, m, seg, a, k):
    """tau of the payload: 90th percentile over training clips of median(predicted sd) / sd(solved log depth)."""
    r_seg = []
    for s in np.unique(seg):
        i = np.where(seg == s)[0]
        meas = [{"a": int(a[j]), "k": int(k[j]), "m": float(m[j])} for j in i]
        var = payload["calib"] * np.exp(payload["model"].predict(X[i]))
        sol, _ = graph.solve(meas, var)
        L = np.array(list(sol.values()))
        if len(L) >= 5 and L.std() > 0:
            r_seg.append(float(np.median(np.sqrt(var)) / L.std()))
    return float(np.percentile(r_seg, 90))


def train_depth4d_model(X, e, seg, seg_group, log=print):
    """Variance model of 4D-model observations (joint-graph ablation), fitted exactly like the measurement model on
    offset-free errors e of the observations.  Returns {model, calib, feats}."""
    grp = seg_group[seg]
    y = np.log(e ** 2 + NU)
    groups = np.unique(grp)
    folds = np.array_split(np.random.default_rng(0).permutation(groups), 5)
    pred = np.full(len(y), np.nan)
    for f in folds:
        te = np.isin(grp, f)
        pred[te] = fit_regressor(X[~te], y[~te]).predict(X[te])
    log(f"training set: {len(np.unique(seg))} clips, {len(groups)} buildings, {len(y)} observations; "
        f"median |e| {np.median(np.abs(e)):.4f}")
    log(f"  leave-building-out Spearman(pred, e^2) {np.corrcoef(_ranks(pred), _ranks(e ** 2))[0, 1]:+.3f}")
    model = fit_regressor(X, y)
    calib = float(np.median(e ** 2 / np.exp(model.predict(X))) / CHI2_1_MEDIAN)
    return {"model": model, "calib": calib, "feats": list(FEATURES_4D)}
