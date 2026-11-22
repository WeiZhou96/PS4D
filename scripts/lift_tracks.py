"""Track-lifting baselines: lift the reference points' 2D positions (ref_uv of each clip) with a model's per-frame
depth (VDA, Pi3X, pi3, VGGT, CUT3R, ... as written by run_depth4d.py / run_baselines.py).
Writes <out>/<clip>.npz {coords (T, N, 3)} in camera coordinates, NaN where the depth is invalid.
Usage: python scripts/lift_tracks.py --clips clips --depth depth_pi3x --out lift_pi3x
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.lifting import lift  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--depth-key", default="depth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for fp in sorted(glob.glob(os.path.join(a.clips, "*.npz"))):
        cid = os.path.basename(fp)[:-4]
        dp = os.path.join(a.depth, cid + ".npz")
        if not os.path.exists(dp):
            continue
        z = np.load(fp)
        C = lift(z["ref_uv"], np.load(dp)[a.depth_key].astype(np.float32), z["K"])
        np.savez_compressed(os.path.join(a.out, cid + ".npz"), coords=C)
        n += 1
    print("lifted", n)


if __name__ == "__main__":
    main()
