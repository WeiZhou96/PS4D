"""DexYCB sequences as evaluation clips.  Reference: the annotated object poses (pose_y, camera frame).

Selection (seed 20261102): the sequences of the given subjects in a seeded order (the first n); per sequence the
cameras are tried in a seeded order and the first one where the grasped object (ycb_grasp_ind) has >= 3 visible model
points in >= 80% of the frames is used.  A model point (points.xyz posed by pose_y) is visible if it projects inside
the image, the segmentation label at its pixel is the object's class and it lies within 5 mm of the object's own
z-buffer (2 px cells).  Point set: up to 64 points visible at the first frame t0 with >= 3 visible points (seeded).
All frames are reference frames; ref_uv are the annotated projections.  The per-sequence seeds depend only on the
sequence's position in the seeded order, so sharding does not change any clip.
"""
import glob
import json
import os
import time
import traceback

import numpy as np

from ..utils import sample_depth
from . import done_ids, write_clip

SEED = 20261102
SUBJECTS = {
    "dev_01_02": ("20200709-subject-01", "20200813-subject-02"),
    "dev_03_05": ("20200820-subject-03", "20200903-subject-04", "20200908-subject-05"),
    "test_06_10": ("20200918-subject-06", "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
                   "20201022-subject-10"),
}
YCB = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can",
       5: "006_mustard_bottle", 6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box",
       9: "010_potted_meat_can", 10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser", 13: "024_bowl",
       14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 17: "037_scissors", 18: "040_large_marker",
       19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick"}
MAX_PTS, CELL, TOL = 64, 2, 0.005


def visible(X, K, seg, cls):
    """Camera-frame points (P, 3) -> pixels (P, 2) and visibility (P,)."""
    H, W = seg.shape
    z = X[:, 2]
    uv = (K @ (X / np.maximum(z[:, None], 1e-6)).T).T[:, :2]
    x, y = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    vis = np.zeros(len(X), bool)
    if not ok.any():
        return uv, vis
    cx, cy = x // CELL, y // CELL
    zb = {}
    for i in np.where(ok)[0]:
        key = (cx[i], cy[i])
        zb[key] = min(zb.get(key, np.inf), z[i])
    for i in np.where(ok)[0]:
        vis[i] = seg[y[i], x[i]] == cls and z[i] <= zb[(cx[i], cy[i])] + TOL
    return uv, vis


def chosen_sequences(root, subjects, n):
    seqs = sorted(d for s in subjects for d in glob.glob(os.path.join(root, s, "*")) if os.path.isdir(d))
    rng = np.random.default_rng(SEED)
    return [seqs[i] for i in rng.permutation(len(seqs))[:n]]


def convert(root, out_jsonl, clip_dir, subjects, n=100, shard="0/1", device="cuda"):
    import cv2
    import yaml

    from ..models.foundation import load_moge, moge_depths
    os.makedirs(clip_dir, exist_ok=True)
    chosen = chosen_sequences(root, subjects, n)
    print("sequences chosen", len(chosen), flush=True)
    moge = load_moge(device)
    done_seq = {"/".join(d.split("/")[:2]) for d in done_ids(out_jsonl)}
    si, sn = (int(x) for x in shard.split("/"))
    with open(out_jsonl, "a") as f:
        for qi, sd in enumerate(chosen):
            if qi % sn != si:
                continue
            subj, seq = sd.split(os.sep)[-2], sd.split(os.sep)[-1]
            t_start, rec = time.time(), {"id": f"{subj}/{seq}", "group": subj}
            if rec["id"] in done_seq:
                continue
            try:
                meta = yaml.load(open(os.path.join(sd, "meta.yml")), Loader=yaml.FullLoader)
                g = meta["ycb_grasp_ind"]
                cls = meta["ycb_ids"][g]
                P = np.loadtxt(os.path.join(root, "models", YCB[cls], "points.xyz"))[:, :3]
                T = int(meta["num_frames"])
                serials = list(meta["serials"])
                order = [serials[i] for i in np.random.default_rng([SEED, qi]).permutation(len(serials))]
                pick = None
                for ser in order:
                    intr = yaml.load(open(os.path.join(root, "calibration/intrinsics", f"{ser}_640x480.yml")),
                                     Loader=yaml.FullLoader)["color"]
                    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], float)
                    V, U, XS = [], [], []
                    for t in range(T):
                        L = np.load(os.path.join(sd, ser, f"labels_{t:06d}.npz"))
                        pose = L["pose_y"][g]
                        X = P @ pose[:3, :3].T + pose[:3, 3]
                        uv, vis = visible(X, K, L["seg"], cls)
                        V.append(vis)
                        U.append(uv)
                        XS.append(X)
                    if np.mean([v.sum() >= 3 for v in V]) >= 0.8:
                        pick = (ser, K, np.stack(V), np.stack(U), np.stack(XS))
                        break
                if pick is None:
                    rec.update({"status": "skip", "reason": "no camera with the object visible in >= 80% of frames"})
                    raise StopIteration
                ser, K, V, U, XS = pick
                rec["id"] += f"/{ser}"
                t0 = int(np.argmax(V.sum(1) >= 3))
                cand = np.where(V[t0])[0]
                sel = np.sort(np.random.default_rng([SEED, qi, 1]).choice(cand, min(MAX_PTS, len(cand)), replace=False))
                frames = np.stack([cv2.cvtColor(cv2.imread(os.path.join(sd, ser, f"color_{t:06d}.jpg")), cv2.COLOR_BGR2RGB)
                                   for t in range(T)])
                sens = [cv2.imread(os.path.join(sd, ser, f"aligned_depth_to_color_{t:06d}.png"),
                                   cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 for t in range(T)]
                mono = moge_depths(moge, {t: frames[t] for t in range(T)}, K, device)
                fn = rec["id"].replace("/", "_") + ".npz"
                write_clip(os.path.join(clip_dir, fn), frames,
                           np.nan_to_num(np.stack([mono[t] for t in range(T)]), nan=0.0).astype(np.float16), K,
                           np.arange(T), np.stack([np.r_[t0, U[t0, j]] for j in sel]), U[:, sel], V[:, sel])
                pts = []
                for col, j in enumerate(sel):
                    obs = []
                    for t in range(T):
                        uv = U[t, j]
                        r = {"k": t, "uv": [float(uv[0]), float(uv[1])], "occluded": int(not V[t, j]),
                             "tracker_visible": bool(V[t, j]), "X_ref": XS[t, j].tolist(), "X_kin": XS[t, j].tolist()}
                        z = sample_depth(mono[t], uv)
                        if np.isfinite(z):
                            r["X_mono"] = [(uv[0] - K[0, 2]) / K[0, 0] * z, (uv[1] - K[1, 2]) / K[1, 1] * z, z]
                        zs = sample_depth(sens[t], uv)
                        if np.isfinite(zs) and zs > 0:
                            r["z_sensor"] = float(zs)
                        obs.append(r)
                    pts.append({"j": int(j), "col": col, "obs": obs})
                rec.update({"status": "ok", "clip_file": fn, "serial": ser, "object": YCB[cls], "T": T, "t0": t0,
                            "n_pts": len(sel), "points": pts, "K": K.tolist()})
            except StopIteration:
                pass
            except Exception as e:
                rec.update({"status": "error", "error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]})
            rec["sec"] = round(time.time() - t_start, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(rec["id"], rec["status"], rec.get("object"), rec.get("n_pts"), rec["sec"], flush=True)
