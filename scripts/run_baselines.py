"""Baselines on the clips of a prepared set.

  vda          metric Video Depth Anything-L on every frame -> <out>/<clip>.npz {depth}; lift it with lift_tracks.py
               (frame rate: 30 on DexYCB and HOT3D, 15 on DROID, as in the experiments)
  spatracker2  SpatialTrackerV2 (camera-only 'rgb' mode) tracking the reference points -> <out>/<clip>.npz with
               cam (T, N, 3) camera-frame tracks (the scored output), world, c2w, intr, uv, vis
Usage: python scripts/run_baselines.py vda --clips clips --out depth_vda --repo third_party/Video-Depth-Anything
         --checkpoint checkpoints/vda/metric_video_depth_anything_vitl.pth [--fps 30] [--shard i/n]
       python scripts/run_baselines.py spatracker2 --clips clips --out spatracker2 --repo third_party/SpaTrackerV2
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.utils import shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", choices=("vda", "spatracker2"))
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoint", default=None, help="VDA checkpoint")
    ap.add_argument("--fps", type=int, default=30, help="VDA frame rate argument")
    ap.add_argument("--mode", default="rgb", choices=("rgb", "rgbd"), help="SpatialTrackerV2 mode")
    ap.add_argument("--max-t", type=int, default=160, help="SpatialTrackerV2: longer clips run at stride 3")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    import torch
    if a.baseline == "vda":
        from ps4d.models.video_depth import load_vda, vda_depth
        model = load_vda(a.repo, a.checkpoint)
    else:
        from ps4d.models import spatracker2 as st2
        front, model = st2.load(a.repo, a.mode)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f"log_{a.shard.split('/')[0]}.jsonl"), "a")
    for fp in shard(sorted(glob.glob(os.path.join(a.clips, "*.npz"))), a.shard):
        cid = os.path.basename(fp)[:-4]
        out = os.path.join(a.out, cid + ".npz")
        if os.path.exists(out):
            continue
        t0, rec = time.time(), {"id": cid}
        try:
            z = np.load(fp)
            if a.baseline == "vda":
                np.savez_compressed(out, depth=vda_depth(model, z["frames"], a.fps))
                rec["status"] = "ok"
            else:
                o, stride, oom = st2.track_clip(front, model, z["frames"], z["mono_depth"].astype(np.float32), z["K"],
                                                z["ref_queries"].astype(np.float64), z["ref_uv"], a.mode, a.max_t)
                np.savez_compressed(out, **{k: (v if v.dtype == bool else v.astype(np.float32)) for k, v in o.items()})
                rec.update({"status": "ok", "stride": stride, "oom_fallback": oom})
        except Exception as e:
            rec.update({"status": "error", "error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]})
            torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(cid, rec["status"], rec["sec"], flush=True)


if __name__ == "__main__":
    main()
