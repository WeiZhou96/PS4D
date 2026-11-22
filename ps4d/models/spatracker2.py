"""SpatialTrackerV2 (offline) tracking the reference points of a clip, camera-only ('rgb') or with depth ('rgbd').

Follows the repository's inference.py (track_num 756, iters_track 4, query_no_BA, fixed_cam False, stage 1,
support_frame T-1, replace_ratio 0.2), with third_party/patches/spatrackerv2_keep_queries.patch applied so that every
user query keeps its output (the stock code may drop low-confidence queries and pad with random duplicates).
  rgb    the VGGT4Track front end gives depth, intrinsics and poses (as the RGB demo); used in the paper
  rgbd   given metric depth and intrinsics, identity extrinsics (static camera)
Frames go through the front end's preprocess_image (width 518, height rounded to a multiple of 14; no crop at the
aspects used here); query pixels and intrinsics follow the half-pixel convention.  The stream tracker runs forward from
each query's window, so frames before a query come from a second pass over the time-reversed clip.  Clips longer than
max_t frames (160), or that run out of memory, are tracked at stride 3.
"""
import gc

import numpy as np

FRONT_MODEL = "Yuxihenry/SpatialTrackerV2_Front"
OFFLINE_MODEL = "Yuxihenry/SpatialTrackerV2-Offline"


def load(repo, mode="rgb", device="cuda"):
    import os
    import sys
    sys.path.insert(0, os.path.abspath(repo))
    from models.SpaTrackV2.models.predictor import Predictor
    from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
    front = VGGT4Track.from_pretrained(FRONT_MODEL).eval().to(device) if mode == "rgb" else None
    model = Predictor.from_pretrained(OFFLINE_MODEL)
    model.spatrack.track_num = 756
    model.eval()
    model.to(device)
    return front, model


def _resize_depth(D, w, h):
    import cv2
    valid = (D > 0).astype(np.float32)
    d = cv2.resize(D, (w, h), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(valid, (w, h), interpolation=cv2.INTER_LINEAR)
    d = d / (m + 1e-6)
    d[m <= 1e-6] = 0.0
    return d


def _run(front, model, video, depth, K, Q, mode):
    import torch
    from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
    T, H, W = video.shape[:3]
    vt = preprocess_image(torch.from_numpy(video).permute(0, 3, 1, 2).float())
    TH, TW = vt.shape[-2:]
    assert TW == 518 and TH == round(H * TW / W / 14) * 14 <= 518, vt.shape     # resized only, never cropped
    sx, sy = TW / W, TH / H
    q = np.c_[Q[:, 0], (Q[:, 1] + 0.5) * sx - 0.5, (Q[:, 2] + 0.5) * sy - 0.5].astype(np.float32)
    if mode == "rgb":
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pr = front(vt[None].cuda() / 255)
        extrs = pr["poses_pred"].squeeze().float().cpu().numpy()
        intrs = pr["intrs"].squeeze().float().cpu().numpy()
        dep = pr["points_map"][..., 2].squeeze().float().cpu().numpy()
        unc = pr["unc_metric"].squeeze().float().cpu().numpy() > 0.5
        del pr
    else:
        dep = np.stack([_resize_depth(d, TW, TH) for d in depth])
        Ks = K.astype(np.float32).copy()
        Ks[0] = [K[0, 0] * sx, 0, (K[0, 2] + 0.5) * sx - 0.5]
        Ks[1] = [0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy - 0.5]
        intrs = np.repeat(Ks[None], T, 0)
        extrs = np.repeat(np.eye(4, dtype=np.float32)[None], T, 0)
        unc = None

    def track(v, dp, it, ex, un, qq):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            c2w, intr_o, _, _, t3, t2, vis, _, _ = model.forward(
                v, depth=dp, intrs=it, extrs=ex, queries=qq, fps=1, full_point=False, iters_track=4,
                query_no_BA=True, fixed_cam=False, stage=1, unc_metric=un, support_frame=len(v) - 1, replace_ratio=0.2)
        c2w = c2w.float().cpu().numpy()
        t3 = t3[..., :3].float().cpu().numpy()
        world = np.einsum("tij,tnj->tni", c2w[:, :3, :3], t3) + c2w[:, :3, 3][:, None]
        return {"cam": t3, "world": world, "c2w": c2w, "intr": intr_o.float().cpu().numpy(),
                "uv": t2[..., :2].float().cpu().numpy(), "vis": vis.float().cpu().numpy().reshape(len(v), -1) > 0.5}

    out = track(vt, dep, intrs, extrs, unc, q)
    if (q[:, 0] > 0).any():
        qb = q.copy()
        qb[:, 0] = T - 1 - q[:, 0]
        bw = track(vt.flip(0).contiguous(), np.ascontiguousarray(dep[::-1]), np.ascontiguousarray(intrs[::-1]),
                   np.ascontiguousarray(extrs[::-1]), None if unc is None else np.ascontiguousarray(unc[::-1]), qb)
        before = np.arange(T)[:, None] < q[:, 0][None]
        for k in ("cam", "world", "uv", "vis"):
            b = bw[k][::-1]
            out[k] = np.where(before[..., None] if out[k].ndim == 3 else before, b, out[k])
    out["uv"] = np.stack([(out["uv"][..., 0] + 0.5) / sx - 0.5, (out["uv"][..., 1] + 0.5) / sy - 0.5], -1)
    return out


def track_clip(front, model, video, depth, K, queries, ref_uv, mode="rgb", max_t=160):
    """Tracks of the queries (N, 3) of (t, x, y).  Returns ({cam, world, c2w, intr, uv, vis}, stride, oom_fallback);
    'cam' holds the camera-frame 3D tracks scored as the SpatialTrackerV2 baseline.  ref_uv (T, N, 2) gives the
    query positions when a clip is tracked at stride 3."""
    import torch
    T = len(video)
    stride, oom = (1, False) if T <= max_t else (3, False)
    if stride == 1:
        try:
            return _run(front, model, video, depth, K, queries, mode), 1, False
        except torch.cuda.OutOfMemoryError:
            stride, oom = 3, True
    gc.collect()
    torch.cuda.empty_cache()
    keep = np.arange(0, T, 3)
    tq = (queries[:, 0] // 3) * 3
    uv = ref_uv[tq.astype(int), np.arange(len(queries))]
    os_ = _run(front, model, video[keep], depth[keep], K, np.c_[tq / 3, uv], mode)
    o = {}
    for k, v in os_.items():
        full = np.full((T,) + v.shape[1:], False if v.dtype == bool else np.nan, bool if v.dtype == bool else np.float32)
        full[keep] = v
        o[k] = full
    return o, 3, oom
