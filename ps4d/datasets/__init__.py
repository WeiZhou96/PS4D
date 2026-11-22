"""Converters from the public datasets to the clip format used by evaluation (see README, "Data preparation").

clip archive <id>.npz
    frames      (T, H, W, 3) uint8 RGB
    mono_depth  (T, H, W) float16 MoGe-2 metric depth, 0 where invalid
    K           (3, 3) intrinsics
    ref_frames  frames with reference observations (all frames on DexYCB and HOT3D, every third frame on DROID)
    ref_queries (N, 3) (t, x, y) of the reference points
    ref_uv      (T, N, 2) their 2D positions (annotated projections; on DROID the CoTracker3 tracks)
    ref_vis     (T, N) visibility
clip record (one json line per clip): id, group (subject / participant / building), sequence (HOT3D), status,
clip_file and points = [{col, obs: [{k, uv, occluded, tracker_visible, X_ref, X_kin, X_mono[, z_sensor]}]}]
    X_ref   reference 3D point in the camera frame (ground truth, or the cross-camera reference on DROID)
    X_kin   kinematic label (DROID; equal to X_ref on the hand datasets)
    X_mono  the MoGe-2 lift at the point's pixel
"""
import json
import os

import numpy as np


def write_clip(path, frames, mono_depth, K, ref_frames, ref_queries, ref_uv, ref_vis, compress=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    save(path, frames=frames, mono_depth=mono_depth, K=K, ref_frames=np.asarray(ref_frames),
         ref_queries=np.asarray(ref_queries), ref_uv=np.asarray(ref_uv, np.float32), ref_vis=np.asarray(ref_vis, bool))


def load_clip(path):
    return dict(np.load(path))


def read_records(paths):
    """Records with status 'ok' from one or more jsonl files."""
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "ok" and "clip_file" in r:
                    out.append(r)
    return out


def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f}
