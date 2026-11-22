"""Projective-scale measurements of the object's log-depth change between an anchor frame a and a later frame k.

From the re-anchored tracks (ps4d.tracks) every pair (a, k) with a < k <= a + H gives up to two measurements of
l(k) - l(a), where l is the log depth of the object:
  size   minus the median log ratio of pairwise image distances of the anchor's points visible at k, over pairs at
         least 3 px apart at a (Eq. 3 of the paper);
  rigid  the anchor's points lifted at a with the monocular depth (MoGe-2), posed at k by PnP against their tracked
         positions (SQPnP + RANSAC with a 3 px threshold, Levenberg-Marquardt on the inliers, at least 5 points);
         log median depth of the posed points minus that at a (Eq. 4).
Each measurement carries eleven statistics of its own geometry, stored as twelve input columns (FEATURES; one column
is duplicated, see below); the reliability model maps them to a variance.
"""
import cv2
import numpy as np

MIN_RIGID_POINTS = 5
# The reliability model's twelve input columns (eleven geometric statistics, one duplicated), in model order:
#   dt        frame gap k - a                 n_pts     visible points at k        vis_frac  visible share of the anchor's points
#   spread_px extent of the points at a       disp      median displacement a -> k iqr       IQR of the pairwise log ratios
#   n_pairs   number of point pairs           pnp_inl   PnP inlier share           pnp_rms   PnP reprojection RMS (px)
#   z_spread  sd of monocular log depth at a  zoom_px   duplicate of spread_px     is_pnp    0 = size, 1 = rigid
# zoom_px always equals spread_px (the extent of the anchor's points); the duplicate column is kept because the
# released model was trained with twelve columns, and removing it would change its predictions.
FEATURES = ("dt", "n_pts", "vis_frac", "spread_px", "disp", "iqr", "n_pairs", "pnp_inl", "pnp_rms", "z_spread",
            "zoom_px", "is_pnp")


def pair_log_ratios(P0, P1, min_dist=3.0):
    """log(|P1_i - P1_j| / |P0_i - P0_j|) over all pairs with |P0_i - P0_j| > min_dist (and a non-degenerate P1)."""
    i, j = np.triu_indices(len(P0), 1)
    d0 = np.linalg.norm(P0[i] - P0[j], axis=1)
    d1 = np.linalg.norm(P1[i] - P1[j], axis=1)
    ok = (d0 > min_dist) & (d1 > 1e-3)
    return np.log(d1[ok] / d0[ok])


def _rigid_pose(X, U, K):
    """SQPnP + RANSAC (3 px), then LM on the inliers.  Returns (rvec, tvec, inlier share, inlier RMS) or None."""
    try:
        ok, rv, tv, inl = cv2.solvePnPRansac(X, U, K, None, flags=cv2.SOLVEPNP_SQPNP, reprojectionError=3.0,
                                             iterationsCount=200, confidence=0.999)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < 4:
        return None
    idx = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[idx], U[idx], K, None, rv, tv)
    proj = cv2.projectPoints(X[idx], rv, tv, K, None)[0][:, 0]
    rms = float(np.sqrt(np.mean(np.sum((proj - U[idx]) ** 2, axis=1))))
    return rv, tv, len(idx) / len(X), rms


def measurements(tracks, visible, query_frame, mono_depth, K, horizon=18):
    """All measurements of a clip.

    tracks (T, Q, 2) and visible (T, Q): re-anchored tracks, NaN / False outside each anchor's window;
    query_frame (Q,): anchor frame of every query; mono_depth (T, H, W): monocular metric depth (only the anchor
    frames are read; 0 or NaN where invalid), or None for size measurements only; K (3, 3).
    Returns a list of dicts with a, k, branch ('size' or 'rigid'), m (log-depth change from a to k) and FEATURES.
    """
    out = []
    Kinv = np.linalg.inv(K)
    for a0 in np.unique(query_frame):
        idx = np.where(query_frame == a0)[0]
        U0 = tracks[a0, idx].astype(float)
        spread = float(np.linalg.norm(np.ptp(U0, 0)))
        if mono_depth is not None:
            ui = np.clip(np.round(U0).astype(int), 0, [mono_depth.shape[2] - 1, mono_depth.shape[1] - 1])
            zz = mono_depth[a0, ui[:, 1], ui[:, 0]].astype(float)
            X0 = (np.c_[U0, np.ones(len(U0))] @ Kinv.T) * zz[:, None] * 1000.0   # millimetres (SQPnP conditioning)
            okz = zz > 0
            zsp = float(np.std(np.log(zz[okz]))) if okz.sum() >= 3 else np.nan
        for k in range(int(a0) + 1, min(int(a0) + horizon, len(tracks) - 1) + 1):
            v = visible[k, idx]
            sel = idx[v]
            base = {"a": int(a0), "k": k, "dt": k - int(a0), "n_pts": int(v.sum()), "vis_frac": float(v.mean()),
                    "spread_px": spread, "zoom_px": spread}   # zoom_px: duplicate column
            if len(sel) >= 3:
                d = tracks[k, sel].astype(float) - tracks[a0, sel].astype(float)
                base["disp"] = float(np.median(np.linalg.norm(d, axis=1)))
                lr = pair_log_ratios(tracks[a0, sel].astype(float), tracks[k, sel].astype(float))
                if len(lr) >= 3:
                    q75, q25 = np.percentile(lr, [75, 25])
                    out.append(dict(base, branch="size", m=-float(np.median(lr)), iqr=float(q75 - q25),
                                    n_pairs=len(lr), pnp_inl=np.nan, pnp_rms=np.nan, z_spread=np.nan, is_pnp=0))
            if mono_depth is not None:
                s2 = v & okz
                if s2.sum() >= MIN_RIGID_POINTS:
                    sol = _rigid_pose(X0[s2], tracks[k, idx[s2]].astype(float), K)
                    if sol is not None:
                        zk = (X0[s2] @ cv2.Rodrigues(sol[0])[0].T + sol[1].T)[:, 2]
                        if np.all(zk > 0):
                            m = float(np.log(np.median(zk)) - np.log(np.median(X0[s2][:, 2])))
                            out.append(dict(base, branch="rigid", m=m, disp=base.get("disp", np.nan), iqr=np.nan,
                                            n_pairs=0, pnp_inl=sol[2], pnp_rms=sol[3], z_spread=zsp, is_pnp=1))
    return out


def feature_matrix(meas, features=FEATURES):
    """(n, len(features)) float array; missing statistics are NaN (the gradient-boosted trees handle them)."""
    return np.array([[float(m.get(f, np.nan)) for f in features] for m in meas], float)


def size_measurements(tracks, visible, query_frame, horizon=18):
    """Size-only measurements (a, k, s) with s = median pairwise log distance ratio (= minus the log-depth change),
    as used by the unweighted size-cue ablation."""
    out = []
    for a0 in np.unique(query_frame):
        idx = np.where(query_frame == a0)[0]
        for k in range(a0 + 1, min(a0 + horizon, len(tracks) - 1) + 1):
            sel = idx[visible[k, idx]]
            P0, P1 = tracks[a0, sel].astype(float), tracks[k, sel].astype(float)
            if len(P0) < 3:
                continue
            lr = pair_log_ratios(P0, P1)
            if len(lr) >= 3:
                out.append((int(a0), k, float(np.median(lr))))
    return out
