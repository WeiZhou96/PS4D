"""DROID hold segments as clips, with certified kinematic labels and the cross-camera reference.

Runs the certification (ps4d.certification.labels, labels from C_g and kinematics only) on each segment of a segment
list (splits/droid_*.json: episode uuid, grasp index, closure frame t_c, building group) and writes
  clip archive: frames (T, 360, 640, 3) of C_g over the hold, mono_depth (MoGe-2, every frame), K at 640 x 360,
                ref_frames = every third hold frame, ref_queries / ref_uv / ref_vis = the certified points' own
                CoTracker3 queries and tracks
  clip record : per certified point and reference frame: uv, occluded (geometric, from the robot render),
                tracker_visible, X_kin (kinematic label), X_ref (cross-camera reference, if the C_a closure passed),
                X_mono (MoGe-2 lift at the tracked pixel), z_sensor (raw stereo depth there), all in C_g coordinates.
The reliability model is trained on X_kin of the training split; evaluation uses X_ref.
"""
import json
import os
import time
import traceback
from dataclasses import fields

import numpy as np

from ..utils import sample_depth
from . import done_ids


def _make_hook(store, moge):
    from ..models.foundation import moge_depths

    def hook(c):
        if "mono" not in c["depth"]:
            return
        K, T1, a, f1, cert = c["K1"], c["T1"], c["a"], c["f1"], c["cert"]
        T_cw = np.linalg.inv(T1)
        frames = c["frames"]
        mono_l = c["depth"]["mono"]
        stereo = c["depth"].get("stereo", {})
        rest = moge_depths(moge, {k: f1[k] for k in frames if k not in mono_l}, K, c["device"])
        mono = np.stack([mono_l[k] if k in mono_l else rest[k] for k in frames])
        js = [p["j"] for p in cert]
        store["npz"] = {"frames": f1[frames[0]:frames[-1] + 1], "mono_depth": np.nan_to_num(mono, nan=0.0).astype(np.float16),
                        "K": K, "ref_frames": np.array([k - a for k in c["ks_l"]]), "ref_queries": c["queries"][js],
                        "ref_uv": c["tr1"][:, js], "ref_vis": c["vis1"][:, js]}
        for col, p in enumerate(cert):
            Xe, j = np.array(p["X"]), p["j"]
            Xa = np.array(p["X_a"]) if "X_a" in p else None
            p["col"] = col                            # column of this point in the clip archive
            obs = []
            for k in c["ks_l"]:
                Tw = c["Twe1"][k]
                uv = c["tr1"][k - a, j]
                r = {"k": int(k - a), "uv": [float(uv[0]), float(uv[1])], "occluded": int(p["occ_mask"][k - a]),
                     "tracker_visible": bool(c["vis1"][k - a, j]),
                     "X_kin": (T_cw[:3, :3] @ (Tw[:3, :3] @ Xe + Tw[:3, 3]) + T_cw[:3, 3]).tolist()}
                if Xa is not None:
                    r["X_ref"] = (T_cw[:3, :3] @ (Tw[:3, :3] @ Xa + Tw[:3, 3]) + T_cw[:3, 3]).tolist()
                z = sample_depth(mono_l[k], uv)
                if np.isfinite(z):
                    r["X_mono"] = [(uv[0] - K[0, 2]) / K[0, 0] * z, (uv[1] - K[1, 2]) / K[1, 1] * z, z]
                if k in stereo:                       # raw stereo depth (1280 x 720) at the tracked pixel
                    zs = sample_depth(stereo[k], (uv + 0.5) * 2 - 0.5)
                    if np.isfinite(zs):
                        r["z_sensor"] = float(zs)
                obs.append(r)
            p["obs"] = obs
        store["K"] = K.tolist()
    return hook


def convert(segments, out_jsonl, clip_dir, config=None, shard="0/1", device="cuda", cotracker_checkpoint=None):
    """segments: list of dicts with uuid, grasp, t_c, group (and optionally lab, building)."""
    from ..certification.droid_raw import load_raw
    from ..certification.grasps import segment_grasps
    from ..certification.labels import LabelConfig, process_segment
    from ..models.foundation import load_cotracker, load_moge, load_sam2_image
    cfg = LabelConfig()
    for k, v in (config or {}).items():
        if k in {f.name for f in fields(LabelConfig)}:
            setattr(cfg, k, v)
    os.makedirs(clip_dir, exist_ok=True)
    i, n = (int(x) for x in shard.split("/"))
    jobs = segments[i::n]
    trk = load_cotracker(cotracker_checkpoint, device)
    sam = load_sam2_image(device)
    moge = load_moge(device)
    done = done_ids(out_jsonl)
    with open(out_jsonl, "a") as f:
        for job in jobs:
            sid = f"{job['uuid']}:{job['grasp']}"
            if sid in done:
                continue
            t0, store = time.time(), {}
            try:
                ep = load_raw(job["uuid"])
                gr = next(g for g in segment_grasps(ep.g) if g.t_c == job["t_c"])
                rec = process_segment(ep, gr, cfg, trk, sam, moge, device, hook=_make_hook(store, moge))
                rec["K"] = store.get("K")
                if "npz" in store:
                    fn = sid.replace("/", "_").replace(":", "_") + ".npz"
                    np.savez_compressed(os.path.join(clip_dir, fn), **store["npz"])
                    rec["clip_file"] = fn
                for p in rec.get("points", []):
                    p.pop("audit", None)
            except Exception as e:
                rec = {"status": "error", "error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]}
            rec.update({"id": sid, "group": job.get("group"), "lab": job.get("lab"), "building": job.get("building"),
                        "sec": round(time.time() - t0, 1)})
            f.write(json.dumps(rec, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
            f.flush()
            print(sid, rec.get("status"), "certified", rec.get("n_cert"), rec["sec"], flush=True)
