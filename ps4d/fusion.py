"""The 4D branch and the equal-weight average (Sec. 3.5 / Eq. 7 of the paper).

The 4D branch reads a 4D reconstruction model's per-frame depth inside our own masks: the log median depth inside
the mask eroded by 2 px (at least 20 valid pixels).  PS4D averages it with the projective-scale branch in log depth,
l = (l_ps + l_4d) / 2; the unknown scale of the 4D model is a constant offset in log depth.
"""
import cv2
import numpy as np


def depth4d_observations(depth, masks, frames, min_px=20):
    """Observations of the object's log depth from a 4D model's depth (T, H, W; NaN where the model was not run).

    masks (T, H, W) bool; frames: the frames to read.  Returns a list of dicts with k, m (log median depth inside the
    eroded mask) and the statistics of reliability.FEATURES_4D (used only by the joint-graph ablation), as edges
    from a virtual offset node a = -1 to k."""
    T, H, W = depth.shape
    ker = np.ones((5, 5), np.uint8)
    rec = []
    for k in frames:
        k = int(k)
        if k >= len(masks):
            continue
        mk = cv2.erode(masks[k].astype(np.uint8), ker) > 0
        if mk.sum() < min_px:
            continue
        z = depth[k][mk]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() < min_px:
            continue
        lz = np.log(z[ok])
        ys, xs = np.nonzero(mk)
        q75, q25 = np.percentile(lz, [75, 25])
        rec.append({"k": k, "m": float(np.median(lz)), "area": float(mk.sum()), "valid_frac": float(ok.mean()),
                    "z_iqr": float(q75 - q25), "ctr": (float(xs.mean()), float(ys.mean())),
                    "border_frac": float(np.mean((xs <= 1) | (xs >= W - 2) | (ys <= 1) | (ys >= H - 2)))})
    if not rec:
        return []
    med_a, med_z = np.median([r["area"] for r in rec]), np.median([r["m"] for r in rec])
    out = []
    for i, r in enumerate(rec):
        step = np.nan
        if i > 0:
            step = float(np.hypot(*np.subtract(r["ctr"], rec[i - 1]["ctr"])) / np.sqrt(r["area"])
                         / (r["k"] - rec[i - 1]["k"]))
        out.append({"a": -1, "k": r["k"], "branch": "d4", "m": r["m"], "log_area": float(np.log(r["area"])),
                    "area_rel": float(np.log(r["area"] / med_a)), "valid_frac": r["valid_frac"], "z_iqr": r["z_iqr"],
                    "border_frac": r["border_frac"], "ctr_step": step, "z_rel": r["m"] - float(med_z),
                    "pos": r["k"] / max(T - 1, 1)})
    return out


def branch_4d(observations):
    """{frame: log depth} of the 4D branch."""
    return {o["k"]: o["m"] for o in observations}


def blend(l_ps, l_4d):
    """Equal-weight average in log depth on the frames where both branches exist: {frame: log depth}."""
    return {k: 0.5 * l_ps[k] + 0.5 * l_4d[k] for k in sorted(set(l_ps) & set(l_4d))}
