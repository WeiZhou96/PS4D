"""Convert DexYCB, HOT3D or DROID into evaluation clips (clip archives + a jsonl of clip records).

  dexycb  --root <DexYCB root> --split test_06_10 | dev_03_05 | dev_01_02 [--n 500 | 100]
  hot3d   --root <HOT3D-Clips train split dir> --models <object models dir> --selection splits/hot3d_quest3_test.json
          --stream 1201-1 (Quest3 left SLAM camera; 214-1 = Aria RGB for splits/hot3d_aria_dev.json)
  droid   --segments splits/droid_test.json (or droid_dev.json / droid_train.json); needs $PS4D_DROID_ROOT
Every converter runs MoGe-2 on every frame (stored as mono_depth); DROID additionally runs the label certification
(SAM 2 image predictor, CoTracker3).  Shards (--shard i/n) can run in parallel; finished clips are skipped.
Usage: python scripts/prepare_data.py dexycb --root data/dexycb --split test_06_10 --n 500 --out data/prepared/dexycb_test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=("dexycb", "hot3d", "droid"))
    ap.add_argument("--root", help="dataset root (DexYCB, or the HOT3D-Clips directory with clip-*.tar)")
    ap.add_argument("--split", default="test_06_10", help="DexYCB subject split")
    ap.add_argument("--n", type=int, default=500, help="DexYCB: number of sequences (500 = all of subjects 06-10)")
    ap.add_argument("--models", help="HOT3D object models directory (obj_*.glb)")
    ap.add_argument("--selection", help="HOT3D clip selection json")
    ap.add_argument("--stream", default="1201-1", help="HOT3D stream: 1201-1 (Quest3) or 214-1 (Aria)")
    ap.add_argument("--segments", help="DROID segment list json")
    ap.add_argument("--out", required=True, help="output directory (clips/ and clips.jsonl)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    clip_dir = os.path.join(a.out, "clips")
    jsonl = os.path.join(a.out, f"clips_{a.shard.replace('/', 'of')}.jsonl" if a.shard != "0/1" else "clips.jsonl")
    os.makedirs(a.out, exist_ok=True)
    if a.dataset == "dexycb":
        from ps4d.datasets import dexycb
        dexycb.convert(a.root, jsonl, clip_dir, dexycb.SUBJECTS[a.split], a.n, a.shard, a.device)
    elif a.dataset == "hot3d":
        from ps4d.datasets import hot3d
        hot3d.convert(a.root, a.models, a.selection, jsonl, clip_dir, a.stream, True, 0, a.shard, a.device)
    else:
        from ps4d.datasets import droid
        segments = json.load(open(a.segments))["segments"]
        droid.convert(segments, jsonl, clip_dir, None, a.shard, a.device)


if __name__ == "__main__":
    main()
