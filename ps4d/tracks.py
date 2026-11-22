"""Re-anchored point tracks on zoomed crops (Sec. 3.3 of the paper).

From the start frame on, an anchor is placed every W frames at which the mask, eroded by 2 px, holds at least 3 grid
points.  Its queries are a uniform grid inside the eroded mask (at most 64 points, a seeded subset when there are
more).  Each anchor a is tracked by CoTracker3 over frames [a, a + H] on a crop around the union of the masks in that
window, padded by 30% (at least 128 x 96 px, aspect 4:3); CoTracker3 resamples its input to a fixed working
resolution, so the crop magnifies small objects.
"""
import cv2
import numpy as np

from .models.foundation import cotrack

KERNEL = np.ones((5, 5), np.uint8)


def grid_queries(mask, rng, max_points=64):
    """Uniform grid inside the mask eroded by 2 px; at most max_points (a seeded subset)."""
    m = cv2.erode(mask.astype(np.uint8), KERNEL) > 0
    n = int(m.sum())
    if n < 3:
        return np.zeros((0, 2))
    step = max(2.0, np.sqrt(n / float(max_points)))
    ys, xs = np.nonzero(m)
    gx = np.arange(xs.min(), xs.max() + 1, step)
    gy = np.arange(ys.min(), ys.max() + 1, step)
    h, w = m.shape
    P = np.array([(x, y) for y in gy for x in gx if m[min(int(round(y)), h - 1), min(int(round(x)), w - 1)]], float)
    if len(P) > max_points:
        P = P[np.sort(rng.choice(len(P), max_points, replace=False))]
    return P


def crop_box(masks, H, W):
    """Crop (x0, y0, w, h) around the union of the masks' boxes, padded 30%, at least 128 x 96, aspect 4:3."""
    ys, xs = [], []
    for m in masks:
        if m.any():
            yy, xx = np.nonzero(m)
            ys += [yy.min(), yy.max()]
            xs += [xx.min(), xx.max()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w, h = max((max(xs) - min(xs)) * 1.3, 128.0), max((max(ys) - min(ys)) * 1.3, 96.0)
    w, h = max(w, h * 4 / 3), max(h, w * 3 / 4)
    w, h = min(w, W), min(h, H)
    x0 = int(np.clip(round(cx - w / 2), 0, W - int(w)))
    y0 = int(np.clip(round(cy - h / 2), 0, H - int(h)))
    return x0, y0, int(w), int(h)


def anchored_tracks(tracker, frames, masks, start, rng, anchor_every=6, horizon=18, zoom=True, device="cuda"):
    """Re-anchored tracks of a clip.

    frames (T, H, W, 3) uint8; masks (T, H, W) bool; start: first frame of the object specification; rng: numpy
    Generator for the query subsets.  Returns {tracks (T, Q, 2) float32 (NaN outside each anchor's window),
    visible (T, Q) bool, query_frame (Q,)} or None if no anchor could be placed."""
    T, H, W = frames.shape[:3]
    tr = np.full((T, 0, 2), np.nan, np.float32)
    vis = np.zeros((T, 0), bool)
    qt = []
    last = -10 ** 9
    for k in range(int(start), T - 1):
        if k - last < anchor_every:
            continue
        P = grid_queries(masks[k], rng)
        if len(P) < 3:
            continue
        last = k
        k1 = min(k + horizon, T - 1)
        if zoom:
            x0, y0, w, h = crop_box(masks[k:k1 + 1], H, W)
            clip = np.ascontiguousarray(frames[k:k1 + 1, y0:y0 + h, x0:x0 + w])
            t_, v_ = cotrack(tracker, clip, np.c_[np.zeros(len(P)), P - [x0, y0]], device)
            t_ = t_ + np.array([x0, y0], np.float32)
        else:
            t_, v_ = cotrack(tracker, frames[k:k1 + 1], np.c_[np.zeros(len(P)), P], device)
        tr_k = np.full((T, len(P), 2), np.nan, np.float32)
        vis_k = np.zeros((T, len(P)), bool)
        tr_k[k:k1 + 1], vis_k[k:k1 + 1] = t_, (v_ > 0.5) if v_.dtype != bool else v_
        tr, vis = np.concatenate([tr, tr_k], 1), np.concatenate([vis, vis_k], 1)
        qt += [k] * len(P)
    if not qt:
        return None
    return {"tracks": tr, "visible": vis, "query_frame": np.array(qt)}
