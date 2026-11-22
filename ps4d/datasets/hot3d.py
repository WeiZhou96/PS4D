"""HOT3D clips as evaluation clips.  Reference: HOT3D's per-frame object and camera poses.

Conversion (fixed before any clip was scored):
- The Aria RGB stream 214-1 (1408 x 1408, FISHEYE624) or the Quest3 left monochrome SLAM stream 1201-1 (1280 x 1024,
  FISHEYE624, replicated to 3 channels) is resampled to a virtual pinhole camera with the same centre, rolled 90 deg so
  that images are upright, 512 x 512 px, f = 224 px; the first 100 frames (3.3 s at 30 fps) of each clip are used.
- The manipulated object is the one with the longest world-frame path over those frames among the objects whose modal
  (visible) mask covers >= 100 px in >= 80% of the frames; its path must be >= 5 cm, otherwise the clip is skipped.
- Model points: 4000 area-weighted surface samples of the object's mesh (seed 20261031).  A point is visible if it
  projects inside the image, lies within 5 mm of the object's own z-buffer (2 px cells) and falls inside the object's
  modal mask (resampled to the pinhole image).  Up to 64 points visible at the first frame t0 with >= 3 visible points
  are kept (seeded); clips where >= 3 of them are visible in fewer than 20% of the frames are skipped.
All frames are reference frames; ref_uv are the projections of the reference points.
"""
import json
import os
import struct
import tarfile
import time
import traceback

import cv2
import numpy as np

from ..utils import sample_depth
from . import done_ids

SEED = 20261031
W = H = 512
FOC = 224.0
N_FRAMES = 100
N_SURF = 4000
MAX_PTS = 64
# the Aria RGB and Quest3 SLAM sensors are mounted rotated; the virtual camera is rolled 90 deg clockwise
RV = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def glb_mesh(path):
    """Vertices (V, 3) and faces (F, 3) of all mesh primitives of a binary glTF, with node transforms applied."""
    b = open(path, "rb").read()
    assert b[:4] == b"glTF"
    off, js, binc = 12, None, None
    while off < len(b):
        n, t = struct.unpack("<I4s", b[off:off + 8])
        if t == b"JSON":
            js = json.loads(b[off + 8:off + 8 + n])
        elif t == b"BIN\x00":
            binc = b[off + 8:off + 8 + n]
        off += 8 + n
    ct = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
    nc = {"SCALAR": 1, "VEC3": 3}

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        k = nc[a["type"]]
        dt = np.dtype(ct[a["componentType"]])
        stride = bv.get("byteStride") or k * dt.itemsize   # interleaved vertex buffers (HOT3D models use stride 32)
        x = np.ndarray((a["count"], k), dt, buffer=binc, offset=start, strides=(stride, dt.itemsize))
        return np.array(x) if k > 1 else np.array(x[:, 0])

    def node_mat(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"], float).reshape(4, 4).T
        M = np.eye(4)
        if "scale" in nd:
            M = np.diag(list(nd["scale"]) + [1.0]) @ M
        if "rotation" in nd:
            x, y, z, w = nd["rotation"]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            M = np.r_[np.c_[R, np.zeros(3)], [[0, 0, 0, 1]]] @ M
        if "translation" in nd:
            T = np.eye(4)
            T[:3, 3] = nd["translation"]
            M = T @ M
        return M

    Vs, Fs, nv = [], [], 0

    def walk(i, M):
        nonlocal nv
        nd = js["nodes"][i]
        M = M @ node_mat(nd)
        if "mesh" in nd:
            for p in js["meshes"][nd["mesh"]]["primitives"]:
                V = acc(p["attributes"]["POSITION"]).astype(float)
                V = (np.c_[V, np.ones(len(V))] @ M.T)[:, :3]
                F = acc(p["indices"]).reshape(-1, 3).astype(np.int64) if "indices" in p else np.arange(len(V)).reshape(-1, 3)
                Vs.append(V)
                Fs.append(F + nv)
                nv += len(V)
        for c in nd.get("children", []):
            walk(c, M)
    for r in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    return np.concatenate(Vs), np.concatenate(Fs)


def surface_samples(V, F, n, rng):
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    f = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return T[f, 0] + u[:, None] * (T[f, 1] - T[f, 0]) + v[:, None] * (T[f, 2] - T[f, 0])


def fisheye624(P, prm):
    """Aria FisheyeRadTanThinPrism (projectaria_tools): params f, cx, cy, k0..k5, p0, p1, s0..s3."""
    f, cx, cy = prm[0], prm[1], prm[2]
    k, p0, p1, s = prm[3:9], prm[9], prm[10], prm[11:15]
    a, b = P[..., 0] / P[..., 2], P[..., 1] / P[..., 2]
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    th2 = th * th
    rad, t = np.ones_like(th), th2.copy()
    for ki in k:
        rad += ki * t
        t = t * th2
    tor = np.where(r < 1e-12, 1.0, th / np.maximum(r, 1e-12))
    xr, yr = rad * tor * a, rad * tor * b
    rd2 = xr * xr + yr * yr
    dot = 2.0 * (xr * p0 + yr * p1)
    u = xr + dot * xr + rd2 * p0 + s[0] * rd2 + s[1] * rd2 * rd2
    v = yr + dot * yr + rd2 * p1 + s[2] * rd2 + s[3] * rd2 * rd2
    return np.stack([f * u + cx, f * v + cy], -1)


def pose(d):
    w, x, y, z = d["quaternion_wxyz"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def rle_mask(e):
    m = np.zeros(e["height"] * e["width"], np.uint8)
    r = e["rle"]
    for s, n in zip(r[0::2], r[1::2]):
        m[s:s + n] = 1
    return m.reshape(e["height"], e["width"])


def convert(clips_dir, models_dir, selection, out_jsonl, clip_dir, stream="214-1", with_mono=True, limit=0,
            shard="0/1", device="cuda"):
    """selection: json with 'clips' (clip numbers) and 'clip_sequence' (clip -> recording sequence)."""
    sel = json.load(open(selection))
    clips = sel["clips"][:limit] if limit else sel["clips"]
    si, sn = (int(x) for x in shard.split("/"))
    clips = clips[si::sn]
    K = np.array([[FOC, 0, (W - 1) / 2], [0, FOC, (H - 1) / 2], [0, 0, 1.0]])
    gx, gy = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    rays = np.stack([(gx - K[0, 2]) / FOC, (gy - K[1, 2]) / FOC, np.ones_like(gx)], -1)
    moge = None
    if with_mono:
        from ..models.foundation import load_moge, moge_depths
        moge = load_moge(device)
    os.makedirs(clip_dir, exist_ok=True)
    done = done_ids(out_jsonl)
    meshes = {}
    with open(out_jsonl, "a") as fo:
        for ci, clip in enumerate(clips):
            cid = f"clip-{clip:06d}"
            if cid in done:
                continue
            t0 = time.time()
            seq = sel["clip_sequence"][str(clip)]
            rec = {"id": cid, "group": seq.split("_")[0], "sequence": seq}
            try:
                tf = tarfile.open(os.path.join(clips_dir, f"{cid}.tar"))
                get = lambda n: tf.extractfile(n).read()  # noqa: E731
                cams = [json.loads(get(f"{t:06d}.cameras.json"))[stream] for t in range(N_FRAMES)]
                objs = [json.loads(get(f"{t:06d}.objects.json")) for t in range(N_FRAMES)]
                prm = np.array(cams[0]["calibration"]["projection_params"], float)
                # the virtual pinhole shares the source camera's pose: its rays are resampled from the fisheye image
                src = fisheye624(rays @ RV.T, prm).astype(np.float32)
                mx, my = src[..., 0], src[..., 1]
                frames = np.stack([cv2.remap(cv2.imdecode(np.frombuffer(get(f"{t:06d}.image_{stream}.jpg"), np.uint8),
                                                          cv2.IMREAD_COLOR)[..., ::-1],
                                             mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                                   for t in range(N_FRAMES)])
                # manipulated object: longest world path among objects visible (modal >= 100 px) in >= 80% of frames
                cand = {}
                for oid in objs[0]:
                    if not all(oid in o for o in objs):
                        continue
                    area, pos = [], []
                    for t, o in enumerate(objs):
                        e = o[oid][0]
                        mm = e.get("masks_modal", {}).get(stream)
                        area.append(int(cv2.remap(rle_mask(mm), mx, my, cv2.INTER_NEAREST).sum()) if mm else 0)
                        pos.append(pose(e["T_world_from_object"])[:3, 3])
                    if np.mean(np.array(area) >= 100) >= 0.8:
                        cand[oid] = float(np.sum(np.linalg.norm(np.diff(np.array(pos), axis=0), axis=1)))
                if not cand:
                    raise RuntimeError("no object visible in >= 80% of frames")
                oid = max(cand, key=cand.get)
                if cand[oid] < 0.05:
                    raise RuntimeError(f"longest object path {cand[oid]:.3f} m < 5 cm")
                bop = objs[0][oid][0]["object_bop_id"]
                if bop not in meshes:
                    meshes[bop] = glb_mesh(os.path.join(models_dir, f"obj_{int(bop):06d}.glb"))
                rng = np.random.default_rng([SEED, clip])
                S = surface_samples(*meshes[bop], N_SURF, rng)
                U = np.zeros((N_FRAMES, N_SURF, 2))
                Xc = np.zeros((N_FRAMES, N_SURF, 3))
                V = np.zeros((N_FRAMES, N_SURF), bool)
                for t in range(N_FRAMES):
                    Tco = np.linalg.inv(pose(cams[t]["T_world_from_camera"])) @ pose(objs[t][oid][0]["T_world_from_object"])
                    X = (S @ Tco[:3, :3].T + Tco[:3, 3]) @ RV   # into the upright virtual camera
                    z = X[:, 2]
                    uv = (X / np.maximum(z, 1e-6)[:, None]) @ K.T
                    uv = uv[:, :2]
                    ins = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
                    cell = np.floor(uv / 2).astype(int)
                    zb = np.full((H // 2 + 1, W // 2 + 1), np.inf)
                    np.minimum.at(zb, (cell[ins, 1], cell[ins, 0]), z[ins])
                    mm = objs[t][oid][0].get("masks_modal", {}).get(stream)   # no modal mask: not visible here
                    mod = cv2.remap(rle_mask(mm), mx, my, cv2.INTER_NEAREST) if mm else np.zeros((H, W), np.uint8)
                    vis = ins.copy()
                    iu, iv = np.clip(np.round(uv).astype(int), 0, [W - 1, H - 1]).T
                    vis[ins] &= (z[ins] <= zb[cell[ins, 1], cell[ins, 0]] + 0.005) & (mod[iv[ins], iu[ins]] > 0)
                    U[t], Xc[t], V[t] = uv, X, vis
                t0f = next((t for t in range(N_FRAMES) if V[t].sum() >= 3), None)
                if t0f is None:
                    raise RuntimeError("never >= 3 visible points")
                cand_pts = np.where(V[t0f])[0]
                pick = np.sort(rng.choice(cand_pts, min(MAX_PTS, len(cand_pts)), replace=False))
                share = np.mean(V[:, pick].sum(1) >= 3)
                if share < 0.2:
                    raise RuntimeError(f">= 3 visible points in only {share:.2f} of frames")
                mono = None
                if moge is not None:
                    md = moge_depths(moge, {t: frames[t] for t in range(N_FRAMES)}, K, device)
                    mono = np.stack([md[t] for t in range(N_FRAMES)])
                np.savez(os.path.join(clip_dir, f"{cid}.npz"), frames=frames,
                         mono_depth=np.nan_to_num(mono, nan=0.0).astype(np.float16) if mono is not None
                         else np.zeros((N_FRAMES, H, W), np.float16),
                         K=K, ref_frames=np.arange(N_FRAMES), ref_queries=np.c_[np.full(len(pick), t0f), U[t0f, pick]],
                         ref_uv=U[:, pick].astype(np.float32), ref_vis=V[:, pick])
                pts = []
                for col, j in enumerate(pick):
                    obs = []
                    for t in range(N_FRAMES):
                        r = {"k": t, "uv": U[t, j].tolist(), "occluded": int(not V[t, j]),
                             "tracker_visible": bool(V[t, j]), "X_ref": Xc[t, j].tolist(), "X_kin": Xc[t, j].tolist()}
                        if mono is not None:
                            zz = sample_depth(mono[t], U[t, j])
                            if np.isfinite(zz):
                                u_, v_ = U[t, j]
                                r["X_mono"] = [(u_ - K[0, 2]) / K[0, 0] * zz, (v_ - K[1, 2]) / K[1, 1] * zz, float(zz)]
                        obs.append(r)
                    pts.append({"j": int(j), "col": col, "obs": obs})
                rec.update({"status": "ok", "clip_file": f"{cid}.npz", "object": objs[0][oid][0]["object_name"],
                            "object_bop_id": bop, "path_m": cand[oid], "T": N_FRAMES, "t0": int(t0f),
                            "n_pts": len(pick), "vis3_share": float(share), "points": pts})
            except Exception as ex:
                rec.update({"status": "error", "error": repr(ex), "trace": traceback.format_exc()[-600:]})
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"{ci + 1}/{len(clips)} {cid} {rec['status']} {rec.get('object', rec.get('error', ''))} "
                  f"{rec.get('n_pts', '')} {time.time() - t0:.1f}s", flush=True)
