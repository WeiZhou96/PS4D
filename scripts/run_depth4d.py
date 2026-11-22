"""Per-frame depth of a 4D reconstruction model (pi3x, pi3, vggt, cut3r) on the clips of a prepared set.

The model runs once per clip on the clip's reference frames (all frames on DexYCB and HOT3D, every third frame on
DROID), at most --max-frames of them (uniformly subsampled otherwise).  Writes <out>/<clip>.npz {depth (T, H, W)
float16, NaN on frames not run}; finished clips are skipped; a log line per clip goes to <out>/log_<shard>.jsonl.
Usage: python scripts/run_depth4d.py --model pi3x --repo third_party/Pi3 --checkpoint checkpoints/pi3x/model.safetensors
         --clips clips --out depth_pi3x [--shard i/n]
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
from ps4d.models import depth4d as d4  # noqa: E402
from ps4d.utils import shard  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=d4.MODELS, required=True)
    ap.add_argument("--repo", required=True, help="code of the model (Pi3, vggt or CUT3R repository)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=100)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    import torch
    run = d4.load(a.model, a.repo, a.checkpoint)
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
            depth, sel = d4.clip_depth(run, z["frames"], z["ref_frames"], a.max_frames)
            np.savez_compressed(out, depth=depth)
            rec.update({"status": "ok", "T": int(len(depth)), "n_run": int(len(sel))})
        except Exception as e:
            rec.update({"status": "error", "error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]})
            torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(cid, rec["status"], rec.get("n_run"), rec["sec"], flush=True)


if __name__ == "__main__":
    main()
