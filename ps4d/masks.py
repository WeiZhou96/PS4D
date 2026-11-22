"""Object masks: SAM 2 propagation from a first-frame box (Sec. 3.3 of the paper).

On the evaluation sets the box is derived from the reference points of the first frame with at least 3 visible
points: their bounding box padded by 10% of its size plus 4 px.  On new videos the user gives the box.
"""
import numpy as np

from .models.foundation import sam2_propagate


def box_from_points(points_by_frame, min_points=3):
    """points_by_frame: {frame: list of (x, y)}.  Returns (start frame, box (x0, y0, x1, y1) float32) for the first
    frame with >= min_points points, or (None, None)."""
    for k in sorted(points_by_frame):
        if len(points_by_frame[k]) >= min_points:
            P = np.array(points_by_frame[k], float)
            lo, hi = P.min(0), P.max(0)
            pad = 0.1 * (hi - lo) + 4.0
            return k, np.r_[lo - pad, hi + pad].astype(np.float32)
    return None, None


def reference_start_box(record, K, use_projection=False, key="X_ref"):
    """Start frame and box of an evaluation clip.  Hand datasets use the annotated pixels of the reference points;
    DROID (use_projection=True) projects the reference points (or, on the training set, the kinematic labels
    key='X_kin') into the camera."""
    F = {}
    for q in record.get("points", []):
        for x in q.get("obs", []):
            if key in x and x["occluded"] == 0 and "X_mono" in x and x["X_mono"][2] > 0:
                if use_projection:
                    X = np.asarray(x[key], float)
                    F.setdefault(x["k"], []).append((K @ (X / X[2]))[:2])
                else:
                    F.setdefault(x["k"], []).append(x["uv"])
    return box_from_points(F)


def propagate(predictor, frames, start, box, workdir=None):
    """(T, H, W) bool masks from SAM 2.1 video propagation, forward from `start`."""
    return sam2_propagate(predictor, frames, start, box, workdir)
