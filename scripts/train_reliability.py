"""Train the reliability model from robot kinematics.

Step 1 (features): for each certified DROID training clip (prepared with prepare_data.py droid and tracked with
run_masks_tracks.py --projection --key X_kin), compute every measurement and its statistics (twelve columns), and its target
l*(k) - l*(a) from the kinematic labels (log median depth of the visible certified points, >= 3 points per frame):
  python scripts/train_reliability.py features --records train/clips.jsonl --clips train/clips --tracks train/tracks
      --out train_features.npz
Step 2 (fit): leave-building-out evaluation, final fit on all buildings, calibration (median chi-square ratio 0.455):
  python scripts/train_reliability.py fit --features train_features.npz --out reliability_model.pkl
The released model was trained on 134,274 labelled measurements from 963 clips in 15 buildings.
For the joint-graph ablation, 'features4d' / 'fit4d' do the same for the 4D observations (--depth: a 4D model's
depth, --masks: PS4D masks).
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d import fusion, reliability  # noqa: E402
from ps4d.datasets import read_records  # noqa: E402
from ps4d.measurements import FEATURES, feature_matrix, measurements  # noqa: E402
from ps4d.utils import unpack_masks  # noqa: E402


def label_log_depth(record, key):
    """{frame: log median depth of the visible labelled points} for frames with >= 3 of them."""
    Z = {}
    for q in record.get("points", []):
        for x in q.get("obs", []):
            if key in x and x["occluded"] == 0 and x[key][2] > 0:
                Z.setdefault(x["k"], []).append(x[key][2])
    return {k: float(np.log(np.median(v))) for k, v in Z.items() if len(v) >= 3}


def features(a):
    rows, seg_id, seg_group = [], [], []
    for r in read_records(a.records):
        tf = os.path.join(a.tracks, r["clip_file"])
        if not os.path.exists(tf):
            continue
        z = np.load(tf)
        c = np.load(os.path.join(a.clips, r["clip_file"]))
        meas = measurements(z["tracks"], z["visible"], z["query_frame"], c["mono_depth"], c["K"].astype(np.float64),
                            a.horizon)
        L = label_log_depth(r, a.key)
        si = len(seg_id)
        seg_id.append(r["id"])
        seg_group.append(r.get("group", ""))
        for m in meas:
            t = L[m["k"]] - L[m["a"]] if m["k"] in L and m["a"] in L else np.nan
            rows.append((m, t, si))
        print(f"{si + 1} {r['clip_file']} measurements {len(meas)}", flush=True)
    np.savez(a.out, X=feature_matrix([m for m, _, _ in rows]), feats=np.array(FEATURES),
             is_pnp=np.array([m["is_pnp"] for m, _, _ in rows]), m=np.array([m["m"] for m, _, _ in rows]),
             t=np.array([t for _, t, _ in rows]), seg=np.array([s for _, _, s in rows]),
             a=np.array([m["a"] for m, _, _ in rows]), k=np.array([m["k"] for m, _, _ in rows]),
             seg_id=np.array(seg_id), seg_group=np.array(seg_group))
    print(f"clips {len(seg_id)}, measurements {len(rows)}, with label {int(np.isfinite([t for _, t, _ in rows]).sum())}")


def features4d(a):
    rows, seg_id, seg_group = [], [], []
    for r in read_records(a.records):
        fm, fd = os.path.join(a.masks, r["clip_file"]), os.path.join(a.depth, r["clip_file"])
        if not (os.path.exists(fm) and os.path.exists(fd)):
            continue
        D = np.load(fd)["depth"].astype(np.float32)
        c = np.load(os.path.join(a.clips, r["clip_file"]))
        obs = fusion.depth4d_observations(D, unpack_masks(np.load(fm)["masks"], D.shape[2]), c["ref_frames"])
        L = label_log_depth(r, a.key)
        withref = [o for o in obs if o["k"] in L]
        if len(withref) < 3:
            continue
        off = float(np.median([o["m"] - L[o["k"]] for o in withref]))
        si = len(seg_id)
        seg_id.append(r["id"])
        seg_group.append(r.get("group", ""))
        rows += [(o, o["m"] - L[o["k"]] - off, si) for o in withref]
    np.savez(a.out, X=feature_matrix([o for o, _, _ in rows], reliability.FEATURES_4D),
             feats=np.array(reliability.FEATURES_4D), e=np.array([e for _, e, _ in rows]),
             m=np.array([o["m"] for o, _, _ in rows]), k=np.array([o["k"] for o, _, _ in rows]),
             seg=np.array([s for _, _, s in rows]), seg_id=np.array(seg_id), seg_group=np.array(seg_group))
    print(f"clips {len(seg_id)}, observations with label {len(rows)}")


def fit(a):
    z = np.load(a.features, allow_pickle=True)
    payload = reliability.train_measurement_model(z["X"], z["m"], z["t"], z["seg"], z["seg_group"], z["is_pnp"])
    payload["tau"] = reliability.clip_noise_threshold(payload, z["X"], z["m"], z["seg"], z["a"], z["k"])
    print(f"calibration factor {payload['calib']:.3f}; clip noise threshold {payload['tau']:.3f}")
    pickle.dump({"model": payload["model"], "calib": payload["calib"], "tau": payload["tau"], "feats": payload["feats"]},
                open(a.out, "wb"))


def fit4d(a):
    z = np.load(a.features, allow_pickle=True)
    payload = reliability.train_depth4d_model(z["X"], z["e"], z["seg"], z["seg_group"])
    print(f"calibration factor {payload['calib']:.3f}")
    pickle.dump(payload, open(a.out, "wb"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("features", "fit", "features4d", "fit4d"))
    ap.add_argument("--records", nargs="+")
    ap.add_argument("--clips")
    ap.add_argument("--tracks")
    ap.add_argument("--masks")
    ap.add_argument("--depth")
    ap.add_argument("--key", default="X_kin", help="label: X_kin (kinematic) or X_ref")
    ap.add_argument("--horizon", type=int, default=18)
    ap.add_argument("--features")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"features": features, "fit": fit, "features4d": features4d, "fit4d": fit4d}[a.step](a)


if __name__ == "__main__":
    main()
