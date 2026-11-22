"""Depth-motion measures (Eq. 1 of the paper) and paired bootstrap comparisons.

For an estimate x and the reference l over the evaluation frames of a clip, both centred:
  r     = ||x - l|| / ||l||           residual ratio (1 for an estimate that predicts no motion)
  eps   = r * sd(l)                   absolute error in log depth (about a relative depth error)
  beta  = <x, l> / ||l||^2            amplitude (least-squares slope; < 1 compresses the motion)
Clips are summarized by medians; two methods are compared by the median of their per-clip differences with a 95%
bootstrap interval (2000 resamples, seed 0), resampling recording sequences as clusters when every clip carries one.
"""
from collections import defaultdict

import numpy as np


def scores(x, ref):
    """Residual ratio r, amplitude beta and correlation of an estimate x with the reference (both 1-D log depth)."""
    x, ref = np.asarray(x), np.asarray(ref)        # dtypes kept: float32 inputs are centred in float32
    x, ref = x - x.mean(), ref - ref.mean()
    return {"r": float((x - ref).std() / ref.std()),
            "beta": float((x * ref).sum() / (ref ** 2).sum()),
            "corr": float(np.corrcoef(x, ref)[0, 1]) if x.std() > 0 else 0.0}


def absolute_error(clip, method):
    """eps of a scored clip: r * sd(reference)."""
    return clip["methods"][method]["r"] * clip["sd_ref"]


def complete(clips, methods):
    """Clips on which every listed method has a score."""
    return [c for c in clips if all(c["methods"].get(m) is not None for m in methods)]


def median(clips, method, field="r"):
    if field == "eps":
        return float(np.median([absolute_error(c, method) for c in clips]))
    return float(np.median([c["methods"][method][field] for c in clips]))


def bootstrap_medians(d, groups=None, n=2000, seed=0):
    """Bootstrap distribution of the median of d; groups (one label per element) resample whole clusters."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if groups is None:
        return [np.median(rng.choice(d, len(d))) for _ in range(n)]
    g = defaultdict(list)
    for i, s in enumerate(groups):
        g[s].append(i)
    G = list(g.values())
    return [np.median(d[np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])]) for _ in range(n)]


def median_ci(clips, f, by=None):
    """Median of f(clip) over clips and its 95% bootstrap interval.  by: cluster labels; default = the recording
    sequence when every clip has one, else none."""
    d = np.array([f(c) for c in clips])
    if by is None:
        seq = [c.get("sequence") for c in clips]
        by = seq if all(s is not None for s in seq) else None
    b = bootstrap_medians(d, by)
    return float(np.median(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired(clips, x, y, by=None):
    """x minus y: (median diff in r, lo, hi), (median diff in eps, lo, hi), share of clips where x has the lower r."""
    r = median_ci(clips, lambda c: c["methods"][x]["r"] - c["methods"][y]["r"], by)
    e = median_ci(clips, lambda c: (c["methods"][x]["r"] - c["methods"][y]["r"]) * c["sd_ref"], by)
    win = float(np.mean([c["methods"][x]["r"] < c["methods"][y]["r"] for c in clips]))
    return r, e, win


def blend_identity(r1, r2, c):
    """Proposition 2: residual ratio of the equal average of two estimates with residual ratios r1, r2 and error
    correlation c."""
    return float(np.sqrt((r1 ** 2 + r2 ** 2 + 2 * c * r1 * r2) / 4))


def blend_beats_better(r1, r2, c):
    """Proposition 2: the average beats the better input iff max/min < sqrt(c^2 + 3) - c."""
    lo, hi = min(r1, r2), max(r1, r2)
    return hi / lo < np.sqrt(c ** 2 + 3) - c


def best_affine_ratio(r1, r2, c):
    """Proposition 2: residual ratio of the best affine weight, r*^2 = r1^2 r2^2 (1 - c^2) / (r1^2 + r2^2 - 2 c r1 r2)."""
    return float(np.sqrt(r1 ** 2 * r2 ** 2 * (1 - c ** 2) / (r1 ** 2 + r2 ** 2 - 2 * c * r1 * r2)))
