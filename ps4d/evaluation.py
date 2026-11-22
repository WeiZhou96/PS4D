"""Per-clip scoring against reference log depth (the evaluation protocol of the paper).

Inputs per clip (formats in README, "Data preparation"):
  record    the clip's json record: reference observations of up to 64 object points at the reference frames
            (X_ref: reference 3D point in the camera frame; X_mono: the MoGe-2 lift at the point's pixel)
  clip      the clip archive (K, mono_depth, ref_frames, ...)
  tracks    re-anchored tracks of PS4D (ps4d.tracks), or None
  masks     (T, H, W) bool SAM 2 masks of PS4D
  lifted    {name: (T, N, 3) camera-frame coordinates of the reference points} for the track-lifting baselines; the
            baselines in `baselines` decide the evaluation frames, any further entry is scored on those frames where
            it is valid on every point
  depth4d   {name: (T, H, W) depth} of 4D models; each gives a 4D branch and a PS4D variant

Evaluation frames: reference frames with at least 3 points that are visible, have a positive monocular lift and a
valid estimate from every baseline, and at least 3 point pairs more than 3 px apart in the frame with the most
points.  The reference log depth of a frame is the log median depth of those points; every baseline's log depth is
the log median of its depths of the same points.  A method is scored on the evaluation frames where it has a value
(at least 5), and a clip needs 5 evaluation frames.

Method names
  lift_moge2                CoTracker3 tracks lifted with MoGe-2 (the reference points' pixels lifted per frame)
  lift_<model>              the same points lifted with another depth / 4D model; spatracker2: SpatialTrackerV2
  size_unweighted           re-anchored size measurements, unweighted robust average (ablation)
  ps_equal                  size + rigid measurements, robust graph, equal weights (ablation)
  ps                        projective-scale branch (learned reliability)
  branch4d_<model>          4D branch: the model's median depth inside our masks
  ps4d_<model>              PS4D: average of ps and branch4d_<model>
  joint_graph               ablation: one graph with learned variances of the 4D observations (first 4D model)
  disagreement_weights      ablation: branches weighted by their observed disagreement (first 4D model)
"""
import itertools
from collections import defaultdict

import numpy as np

from . import fusion, graph
from .measurements import feature_matrix, measurements, size_measurements
from .metrics import scores
from .reliability import FEATURES_4D

MONO_BASELINE = "lift_moge2"


def _reference_points(record):
    """frame -> {point column: observation} for observations with a reference point, unoccluded, with a positive
    monocular lift."""
    F = defaultdict(dict)
    for q in record.get("points", []):
        for x in q.get("obs", []):
            if "X_ref" in x and x["occluded"] == 0 and "X_mono" in x and x["X_mono"][2] > 0:
                F[x["k"]][q["col"]] = x
    return F


def _disagreement_weight(an, x4, ok_frames, cov_ours):
    """4D weight s2_ours / var(x4 - an) of the disagreement-weighted average, clipped to [0.1, 0.9]; None if the
    posterior covariance does not cover every frame."""
    nodes, S = cov_ours
    ixn = {n_: i_ for i_, n_ in enumerate(nodes)}
    jj = [ixn[k] for k in ok_frames if k in ixn]
    if len(jj) != len(ok_frames):
        return None
    P = np.eye(len(jj)) - 1.0 / len(jj)
    s2o = float(np.mean(np.diag(P @ S[np.ix_(jj, jj)] @ P)))
    d = (x4 - x4.mean()) - (an - an.mean())
    s2d = (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2
    return float(np.clip(s2o / s2d, 0.1, 0.9)) if s2d > 0 else 0.9


def score_clip(record, clip, tracks, masks, lifted, depth4d, reliability, baselines, depth4d_variance=None,
               horizon=18, keep_frames=True):
    """Score one clip.  Returns None if the clip has fewer than 5 evaluation frames or no reference motion, else
    {id, group, sequence, n_frames, sd_ref, methods: {name: {r, beta, corr} or None}, frames: per-frame log depth}.
    `baselines`: ordered names of the baselines that decide the evaluation frames (lift_moge2 comes from the record,
    the others from `lifted`)."""
    if set(baselines) - {MONO_BASELINE} - set(lifted):
        return None
    F = _reference_points(record)
    if not F:
        return None
    K = np.asarray(clip["K"], np.float64)
    extra = [n for n in lifted if n not in baselines]
    d4_names = list(depth4d)

    # PS4D on the clip
    est, cov_ours = {}, None
    if tracks is not None:
        tr, vis, qt = tracks["tracks"], tracks["visible"], tracks["query_frame"]
        est["size_unweighted"] = {k: -v for k, v in graph.robust_average(size_measurements(tr, vis, qt, horizon)).items()}
        if reliability is not None:
            meas = measurements(tr, vis, qt, clip["mono_depth"], K, horizon)
            if meas:
                var = reliability.variance(meas)
                est["ps"], _ = graph.solve(meas, var)
                est["ps_equal"], _ = graph.solve(meas, None)
                if d4_names:
                    obs_first = None
                    for name in d4_names:
                        obs = fusion.depth4d_observations(depth4d[name], masks, clip["ref_frames"])
                        est["branch4d_" + name] = fusion.branch_4d(obs)
                        if obs_first is None:
                            obs_first = obs
                    _, _, cov_ours = graph.solve(meas, var, cov=True)
                    if obs_first and depth4d_variance is not None:
                        var4 = depth4d_variance.calib * np.exp(
                            depth4d_variance.model.predict(feature_matrix(obs_first, FEATURES_4D)))
                        s1, _ = graph.solve(meas + obs_first, np.r_[var, var4])
                        est["joint_graph"] = {kk: v for kk, v in s1.items() if kk >= 0}

    kref = max(F, key=lambda k: len(F[k]))
    rows = []
    for k, D in sorted(F.items()):
        cols = []
        for c, x in D.items():
            v = {n: lifted[n][k, c] for n in lifted}
            if all(np.all(np.isfinite(v[n])) and v[n][2] > 0 for n in baselines if n != MONO_BASELINE):
                cols.append((c, x, v))
        if len(cols) < 3:
            continue
        common = [c for c, _, _ in cols if c in F[kref]]
        pairs = [1 for i, j in itertools.combinations(common, 2)
                 if np.linalg.norm(np.subtract(F[kref][i]["uv"], F[kref][j]["uv"])) > 3.0]
        if len(pairs) < 3:
            continue
        rec = {"k": k, "ref": np.log(np.median([x["X_ref"][2] for _, x, _ in cols]))}
        for n in baselines:
            if n == MONO_BASELINE:
                rec[n] = np.log(np.median([x["X_mono"][2] for _, x, _ in cols]))
            else:
                rec[n] = np.log(np.median([v[n][2] for _, _, v in cols]))
        for n in extra:
            zx = [lifted[n][k, c][2] for c, _, _ in cols]
            rec[n] = float(np.log(np.median(zx))) if all(np.isfinite(z) and z > 0 for z in zx) else None
        for n, L in est.items():
            rec[n] = L.get(k)
        for name in d4_names:
            b4 = rec.get("branch4d_" + name)
            rec["ps4d_" + name] = 0.5 * rec["ps"] + 0.5 * b4 if b4 is not None and rec.get("ps") is not None else None
        rows.append(rec)
    if len(rows) < 5:
        return None
    ref = np.array([x["ref"] for x in rows])
    if ref.std() <= 0:
        return None
    out = {"id": record["id"], "group": record.get("group"), "sequence": record.get("sequence"), "n_frames": len(rows),
           "sd_ref": float(ref.std()), "methods": {}}
    for n in baselines:
        out["methods"][n] = scores(np.array([x[n] for x in rows]), ref)
    names = extra + ["size_unweighted", "ps_equal", "ps"] + [p + n for n in d4_names for p in ("branch4d_", "ps4d_")]
    names += ["joint_graph"] if d4_names else []
    for n in names:
        ok = [i for i, x in enumerate(rows) if x.get(n) is not None]
        out["methods"][n] = scores(np.array([rows[i][n] for i in ok]), ref[ok]) if len(ok) >= 5 and ref[ok].std() > 0 else None
    if d4_names and cov_ours is not None:
        b4 = "branch4d_" + d4_names[0]
        ok = [i for i, x in enumerate(rows) if x.get("ps") is not None and x.get(b4) is not None]
        if len(ok) >= 5 and ref[ok].std() > 0:
            an, x4 = np.array([rows[i]["ps"] for i in ok]), np.array([rows[i][b4] for i in ok])
            w4 = _disagreement_weight(an, x4, [rows[i]["k"] for i in ok], cov_ours)
            if w4 is not None:
                out["methods"]["disagreement_weights"] = scores((1 - w4) * an + w4 * x4, ref[ok])
                out["disagreement_weight"] = w4
    out["methods"] = {n: v for n, v in out["methods"].items() if v is not None}
    if keep_frames:
        keys = ["k", "ref"] + list(baselines) + names
        out["frames"] = {kk: [None if x.get(kk) is None else float(x[kk]) for x in rows] for kk in keys}
    return out
