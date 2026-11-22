"""Score PS4D, its ablations and the baselines on a prepared evaluation set; writes a per-clip results file.

Example (DexYCB test set, layout of README "Reproduction"):
  python scripts/evaluate.py --records $D/clips.jsonl --clips $D/clips --tracks $D/tracks --masks $D/masks \
      --lifted lift_vda=$D/lift_vda spatracker2=$D/spatracker2:cam lift_pi3x=$D/lift_pi3x \
      --extra-lifted lift_pi3=$D/lift_pi3 lift_vggt=$D/lift_vggt lift_cut3r=$D/lift_cut3r \
      --depth4d pi3x=$D/depth_pi3x pi3=$D/depth_pi3 vggt=$D/depth_vggt cut3r=$D/depth_cut3r \
      --out results/test/dexycb.json.gz
The baselines (--baselines; lift_moge2 is read from the records) decide the evaluation frames; --extra-lifted
pipelines are scored on those frames.  The first --depth4d model is the 4D branch of PS4D and of the two
weighting ablations; every model gives branch4d_<name> and ps4d_<name>.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.datasets import read_records  # noqa: E402
from ps4d.evaluation import score_clip  # noqa: E402
from ps4d.io import load_depth, load_lifted, load_masks, load_npz, load_tracks, parse_sources, write_results  # noqa: E402
from ps4d.reliability import FEATURES_4D, VarianceModel  # noqa: E402
from ps4d.utils import DEPTH4D_VARIANCE_SHA256, default_depth4d_variance_model, default_reliability_model  # noqa: E402

DEFAULT_BASELINES = "lift_moge2,lift_vda,spatracker2,lift_pi3x"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", required=True, help="clip records (jsonl)")
    ap.add_argument("--clips", required=True, help="directory of clip archives")
    ap.add_argument("--tracks", default=None, help="directory of PS4D re-anchored tracks")
    ap.add_argument("--masks", default=None, help="directory of PS4D masks")
    ap.add_argument("--lifted", nargs="*", default=[], help="baselines: name=dir[:key] (key default coords)")
    ap.add_argument("--extra-lifted", nargs="*", default=[], help="further lifted pipelines: name=dir[:key]")
    ap.add_argument("--baselines", default=DEFAULT_BASELINES, help="comma-separated baselines that define the frames")
    ap.add_argument("--depth4d", nargs="*", default=[], help="4D models: name=dir (the first is PS4D's 4D branch)")
    ap.add_argument("--model", default=default_reliability_model(), help="reliability model")
    ap.add_argument("--depth4d-variance", default=default_depth4d_variance_model(),
                    help="variance model of 4D observations (joint-graph ablation; 'none' to skip)")
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--no-frames", action="store_true", help="do not store per-frame log depth")
    ap.add_argument("--out", required=True, help="results file (.json or .json.gz)")
    a = ap.parse_args()

    reliability = VarianceModel.load(a.model)
    var4 = None
    if a.depth4d_variance and a.depth4d_variance != "none" and os.path.exists(a.depth4d_variance):
        sha = DEPTH4D_VARIANCE_SHA256 if a.depth4d_variance == default_depth4d_variance_model() else None
        var4 = VarianceModel.load(a.depth4d_variance, FEATURES_4D, sha)
    baselines = tuple(a.baselines.split(","))
    lifted_src = parse_sources(a.lifted)
    extra_src = parse_sources(a.extra_lifted)
    d4_src = dict(s.split("=", 1) for s in a.depth4d)
    clips = []
    records = read_records(a.records)
    for i, r in enumerate(records):
        fn = r["clip_file"]
        lifted = {}
        for name, (d, key) in list(lifted_src.items()) + list(extra_src.items()):
            x = load_lifted(os.path.join(d, fn), key)
            if x is not None:
                lifted[name] = x
        if set(baselines) - {"lift_moge2"} - set(lifted):
            continue
        clip = load_npz(os.path.join(a.clips, fn))
        clip["K"] = clip["K"].astype(np.float64)
        tracks = load_tracks(os.path.join(a.tracks, fn)) if a.tracks else None
        depth4d, masks = {}, None
        for name, d in d4_src.items():
            D = load_depth(os.path.join(d, fn))
            if D is not None:
                depth4d[name] = D
        if depth4d:
            masks = load_masks(os.path.join(a.masks, fn), next(iter(depth4d.values())).shape[2])
            if masks is None:
                depth4d = {}
        e = score_clip(r, clip, tracks, masks, lifted, depth4d, reliability, baselines, var4, a.horizon,
                       keep_frames=not a.no_frames)
        if e is not None:
            clips.append(e)
        print(f"{i + 1}/{len(records)} {r['id']} {'scored' if e else 'skipped'}", flush=True)
    write_results(a.out, clips, {"baselines": list(baselines), "depth4d": list(d4_src), "horizon": a.horizon})
    print(f"{len(clips)} clips scored -> {a.out}")


if __name__ == "__main__":
    main()
