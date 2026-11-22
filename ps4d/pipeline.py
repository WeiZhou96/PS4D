"""End-to-end PS4D on a video: frames, intrinsics and a box around the object on the start frame.

    masks        SAM 2.1 propagates a mask from the box
    tracks       anchors every 6 frames, <= 64 grid queries in the eroded mask, CoTracker3 on zoomed crops over 18 frames
    measurements size and anchored-rigid measurements for every (anchor, frame) pair (MoGe-2 depth at the anchors)
    reliability  per-measurement variances from eleven geometric statistics (twelve columns)
    graph        Huber IRLS on the log-depth graph -> projective-scale branch l_ps
    4D branch    a 4D model's (Pi3X by default) median depth inside the eroded masks -> l_4d
    PS4D         (l_ps + l_4d) / 2 on the frames where both exist
All outputs are log depth up to an additive constant; they are returned centred on their common frames.
"""
import numpy as np

from . import fusion, graph
from .masks import propagate
from .measurements import measurements
from .models import depth4d as d4
from .models.foundation import moge_depths, stored_mono_depth
from .tracks import anchored_tracks


class Models:
    """Loads the networks once.  depth4d: name in ps4d.models.depth4d.MODELS, or None for the projective-scale
    branch only."""

    def __init__(self, reliability, depth4d="pi3x", depth4d_repo=None, depth4d_checkpoint=None,
                 cotracker_checkpoint=None, device="cuda"):
        from .models.foundation import load_cotracker, load_moge, load_sam2_video
        self.device = device
        self.reliability = reliability
        self.sam2 = load_sam2_video(device)
        self.tracker = load_cotracker(cotracker_checkpoint, device)
        self.moge = load_moge(device)
        self.depth4d_name = depth4d
        self.depth4d = d4.load(depth4d, depth4d_repo, depth4d_checkpoint, device) if depth4d else None


def estimate(models, frames, K, box, start=0, frames_4d=None, max_frames_4d=100, seed=0, horizon=18, anchor_every=6,
             masks=None):
    """frames (T, H, W, 3) uint8 RGB; K (3, 3); box (x0, y0, x1, y1) on frame `start`.
    frames_4d: frames on which the 4D model runs (default all, at most max_frames_4d uniformly subsampled).
    Returns a dict with per-frame arrays (NaN where a branch has no estimate): log_depth_ps, log_depth_4d, log_depth
    (PS4D), plus masks, tracks and the measurements."""
    K = np.asarray(K, np.float64)
    T = len(frames)
    if masks is None:
        masks = propagate(models.sam2, frames, start, box)
    tracks = anchored_tracks(models.tracker, frames, masks, start, np.random.default_rng(seed), anchor_every, horizon,
                             zoom=True, device=models.device)
    out = {"masks": masks, "tracks": tracks, "measurements": [], "log_depth_ps": np.full(T, np.nan),
           "log_depth_4d": np.full(T, np.nan), "log_depth": np.full(T, np.nan)}
    l_ps = {}
    if tracks is not None:
        anchors = np.unique(tracks["query_frame"])
        mono = stored_mono_depth(moge_depths(models.moge, {int(a): frames[a] for a in anchors}, K, models.device),
                                 frames.shape)
        meas = measurements(tracks["tracks"], tracks["visible"], tracks["query_frame"], mono, K, horizon)
        out["measurements"] = meas
        if meas:
            l_ps, _ = graph.solve(meas, models.reliability.variance(meas))
    l_4d = {}
    if models.depth4d is not None:
        ids = np.arange(T) if frames_4d is None else np.asarray(frames_4d)
        depth, sel = d4.clip_depth(models.depth4d, frames, ids, max_frames_4d)
        out["depth4d_frames"] = sel
        l_4d = fusion.branch_4d(fusion.depth4d_observations(depth.astype(np.float32), masks, sel))
    l_bl = fusion.blend(l_ps, l_4d)
    for key, L in (("log_depth_ps", l_ps), ("log_depth_4d", l_4d), ("log_depth", l_bl)):
        for k, v in L.items():
            out[key][k] = v
    common = np.isfinite(out["log_depth"])
    for key in ("log_depth_ps", "log_depth_4d", "log_depth"):
        ref = common if common.any() else np.isfinite(out[key])
        if ref.any():
            out[key] = out[key] - np.nanmean(out[key][ref])
    return out
