"""PS4D front end on a prepared set: SAM 2 masks from the first-frame box and re-anchored CoTracker3 tracks.

The box is the bounding box of the reference points on the first frame with at least 3 of them, padded by 10% + 4 px
(hand datasets: their annotated pixels; DROID: --projection, the reference points projected into the camera; the
DROID training set: --projection --key X_kin).  Writes <masks>/<clip>.npz {masks: bits packed along x} and
<tracks>/<clip>.npz {tracks, visible, query_frame}; finished clips are skipped.
Query subsets (anchors with more than 64 grid points) use one seeded generator per process, consumed in record order,
as in the experiments; a different sharding changes those subsets but not the method.
Usage: python scripts/run_masks_tracks.py --records clips.jsonl --clips clips --masks masks --tracks tracks
         [--projection] [--key X_ref] [--shard i/n]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.datasets import read_records  # noqa: E402
from ps4d.utils import pack_masks, shard, unpack_masks  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--projection", action="store_true", help="box from the projected reference points (DROID)")
    ap.add_argument("--key", default="X_ref", help="reference point used for the box (X_kin on the DROID training set)")
    ap.add_argument("--anchor-every", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--no-zoom", action="store_true", help="track on full frames (ablation)")
    ap.add_argument("--cotracker-checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch

    from ps4d.masks import propagate, reference_start_box
    from ps4d.models.foundation import load_cotracker, load_sam2_video
    from ps4d.tracks import anchored_tracks
    os.makedirs(a.masks, exist_ok=True)
    os.makedirs(a.tracks, exist_ok=True)
    tracker = load_cotracker(a.cotracker_checkpoint, a.device)
    sam = None
    rng = np.random.default_rng(a.seed)
    recs = shard(read_records(a.records), a.shard)
    for i, r in enumerate(recs):
        fn = r["clip_file"]
        o = os.path.join(a.tracks, fn)
        if os.path.exists(o):
            continue
        z = np.load(os.path.join(a.clips, fn))
        frames, K = z["frames"], z["K"].astype(np.float64)
        start, box = reference_start_box(r, K, a.projection, a.key)
        if start is None:
            continue
        mp = os.path.join(a.masks, fn)
        if os.path.exists(mp):
            M = unpack_masks(np.load(mp)["masks"], frames.shape[2])
        else:
            sam = sam or load_sam2_video(a.device)
            M = propagate(sam, frames, start, box)
            np.savez_compressed(mp, masks=pack_masks(M))
        tr = anchored_tracks(tracker, frames, M, start, rng, a.anchor_every, a.horizon, not a.no_zoom, a.device)
        if tr is None:
            continue
        np.savez(o + ".tmp.npz", **tr)
        os.replace(o + ".tmp.npz", o)
        print(f"{i + 1}/{len(recs)} {fn} T {len(frames)} start {start} anchors {len(set(tr['query_frame']))} "
              f"queries {len(tr['query_frame'])}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
