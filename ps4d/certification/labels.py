"""Certified labels for one hold segment (generation on C_g = DROID's first exterior camera; C_a = the second exterior
camera is used only for the cross-camera reference).

  1. hold segment [t_c + 2, t_release - 2]; queries from the between-finger SAM 2 mask in C_g (arm links removed;
     gripper pixels kept, points that are really on the gripper are rejected later by the 3D hull test);
  2. CoTracker3 over the hold in C_g; gripper-frame rays (forward kinematics at capture time + tau) in two passes:
     (i) tracker-visible frames off the arm links, (ii) after a first estimate, frames where the point is not behind
     the rendered robot;
  3. gripper-frame triangulation, ray-only and, where C_g stereo exists, with an along-ray stereo supplement;
     certificate = inlier rays, residual, covariance, outside the gripper-rigid hulls (+ depth inliers);
  4. amodal labels over the hold with geometric visibility from the robot render (known occluder);
  5. reference: C_a-only triangulation of each certified point from its projection (cross-camera closure);
  6. lifted C_g tracks (MoGe-2, stereo) at every third hold frame, compared with the label and the reference.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import asdict, dataclass

import av
import cv2
import numpy as np

from ..models.foundation import cotrack, moge_depths
from ..utils import sample_depth
from .calibration import episode_calib, load_calib
from .geometry import PoseInterp, project
from .gripper import finger_box, robot_negatives, step_index
from .paths import RAW_DIR
from .robot_render import arm_kind, joints_at, render
from .triangulation import inside_any, rays_in_hand, robot_hulls, triangulate


@dataclass
class LabelConfig:
    """Parameters of the certification; the defaults are the values used for every label of the paper."""
    tau_ms: float = 37.3           # residual camera-robot time offset
    n_queries: int = 64            # per query frame
    query_frames: int = 3          # 1: k_h only; 3: k_h, mid-hold and late-hold (covers what C_g sees over the hold)
    hold_offset: int = 8
    dil_px: int = 5
    n_min: int = 8                 # inlier rays
    r_max_px: float = 1.5          # reprojection RMS at 640x360
    sigma_max_mm: float = 10.0     # position uncertainty
    sigma_u_px: float = 0.63       # per-axis tracking noise at 640x360
    sigma_d_px: float = 1.0        # per-observation stereo disparity noise at 1280x720; 0 disables the supplement
    sigma_db_px: float = 0.5       # systematic disparity error left after a per-segment bias estimate
    sigma_db_prior_px: float = 2.0 # systematic disparity error when no per-segment estimate is possible
    bias_min_points: int = 3       # well-conditioned ray-only points needed to estimate the segment's stereo bias
    bias_min_obs: int = 30
    disp_bias_px: float = 0.0      # unused (kept for configuration compatibility)
    min_depth_inliers: int = 3
    stereo_every: int = 2
    lift_every: int = 3            # reference frames: every third hold frame
    eps_vis_m: float = 0.01
    hull_margin_m: float = 0.003
    min_hold: int = 10


def stereo_depths(uuid, serial, frames, fx_full, baseline):
    """Depth (1280x720) from C_g's own side-by-side stereo MP4 at the given frames; None if absent."""
    p = os.path.join(RAW_DIR, uuid, "recordings/MP4", f"{serial}-stereo.mp4")
    if not os.path.exists(p) or not len(frames):
        return None
    want, ims = set(int(k) for k in frames), {}
    with av.open(p) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i in want:
                ims[i] = f.to_ndarray(format="gray")
            if i >= max(want):
                break
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=256, blockSize=7, P1=8 * 49, P2=32 * 49,
                               uniquenessRatio=5, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_HH)
    out = {}
    for k, im in ims.items():
        d = sg.compute(im[:, :1280], im[:, 1280:]).astype(np.float32) / 16.0
        out[k] = np.where(d > 1.0, fx_full * baseline / np.maximum(d, 1e-6), np.nan)
    return out






class View:
    """One external camera over the hold frames: intrinsics, extrinsics, hand poses, arm masks, robot render depth."""

    def __init__(self, ep, role, K, T_wc, Twe, frames, kind, dq, tau_ms, dil_px, hw):
        self.K, self.T_wc, self.Twe = K, T_wc, Twe
        self.T_cw = np.linalg.inv(T_wc)
        q, g = joints_at(ep, role, tau_ms)
        ker = np.ones((2 * dil_px + 1, 2 * dil_px + 1), np.uint8)
        self.arm, self.zr = {}, {}
        for k in frames:
            R = render(K, T_wc, q[k], g[k], hw, kind=kind, dq=dq)
            arm = (R["link"] >= 0) & (R["link"] <= 7)
            self.arm[k] = cv2.dilate(arm.astype(np.uint8), ker) > 0
            self.zr[k] = R["z"]
        self.H, self.W = hw

    def inside(self, uv):
        return 0 <= uv[0] < self.W - 0.5 and 0 <= uv[1] < self.H - 0.5

    def depth_of(self, Xe, k):
        Xw = self.Twe[k][:3, :3] @ Xe + self.Twe[k][:3, 3]
        return float(self.T_cw[2, :3] @ Xw + self.T_cw[2, 3])

    def project(self, Xe, k):
        Xw = self.Twe[k][:3, :3] @ Xe + self.Twe[k][:3, 3]
        uv, z = project(self.K, self.T_wc, Xw[None, None])
        return uv[0, 0], float(z[0, 0])

    def frames_for(self, tr, vis, j, a, frames, Xe=None, eps=0.01):
        """Pass (i) without Xe: tracker-visible, inside, off the arm links. Pass (ii) with Xe: also not behind the
        rendered robot (any part, gripper included) at the tracked pixel."""
        out = []
        for k in frames:
            uv = tr[k - a, j]
            if not (vis[k - a, j] and self.inside(uv)):
                continue
            x, y = int(round(uv[0])), int(round(uv[1]))
            if self.arm[k][y, x]:
                continue
            if Xe is not None:
                zr = self.zr[k][y, x]
                if np.isfinite(zr) and zr < self.depth_of(Xe, k) - eps:
                    continue
            out.append(k)
        return out

    def rays(self, tr, j, a, ks):
        return rays_in_hand(np.array([self.Twe[k] for k in ks]), self.K, self.T_wc, tr[np.array(ks) - a, j])


def _gate(t, cfg, hulls, use_depth):
    if t is None:
        return "few_rays"
    if t.n < cfg.n_min:
        return "few_inliers"
    if t.rms_px > cfg.r_max_px:
        return "residual"
    if np.sqrt(np.trace(t.cov)) * 1e3 > cfg.sigma_max_mm:
        return "sigma"
    if use_depth and t.n_depth < cfg.min_depth_inliers:
        return "few_depth"
    if inside_any(t.X, hulls):
        return "robot_part"
    return "ok"


def _two_pass(view, tr, vis, j, a, frames, cfg):
    ks = view.frames_for(tr, vis, j, a, frames)
    if len(ks) < 3:
        return None, ks
    O, D = view.rays(tr, j, a, ks)
    t0 = triangulate(O, D, view.K[0, 0], cfg.sigma_u_px)
    if t0 is None:
        return None, ks
    ks2 = view.frames_for(tr, vis, j, a, frames, t0.X, cfg.eps_vis_m)
    if len(ks2) < 3:
        return t0, ks
    O, D = view.rays(tr, j, a, ks2)
    return triangulate(O, D, view.K[0, 0], cfg.sigma_u_px), ks2


def process_segment(ep, grasp, cfg: LabelConfig, trk, sam, moge=None, device="cuda", closure=True, audit=True,
                    calib_dump=False, hook=None) -> dict:
    """Certify the points of one hold segment.  hook (optional): called once after the audit with the intermediate
    state (used by ps4d.datasets.droid to write the evaluation clip); it may add keys to the certified point dicts."""
    calh, calf = episode_calib(ep, 0.5), episode_calib(ep, 1.0)
    K1, K2, T1, T2 = calh["K"]["ext1"], calh["K"]["ext2"], calh["T_wc"]["ext1"], calh["T_wc"]["ext2"]
    Kf = calf["K"]["ext1"]
    pi = PoseInterp(calh["ee_track"])
    f1, f2 = ep.frames("ext1", 0.5), ep.frames("ext2", 0.5)
    n_fr = min(len(f1), len(f2), ep.T)
    a, b = grasp.t_c + 2, min(grasp.t_release - 2, n_fr - 1)
    rec = {"uuid": ep.uuid, "lab": ep.meta.get("lab"), "calib": calh["source"], "t_c": int(grasp.t_c),
           "t_release": int(grasp.t_release), "hold": [int(a), int(b)], "cfg": asdict(cfg)}
    if b - a < cfg.min_hold:
        return {**rec, "status": "short_hold"}
    k_h = min(grasp.t_c + cfg.hold_offset, (a + b) // 2)
    frames = list(range(a, b + 1))
    hw = f1.shape[1:3]
    kind = arm_kind(ep.meta.get("robot_serial", ""))
    Twe1 = {k: T for k, T in zip(frames, pi(step_index(ep, "ext1", cfg.tau_ms, frames)))}
    Twe2 = {k: T for k, T in zip(frames, pi(step_index(ep, "ext2", cfg.tau_ms, frames)))}
    v1 = View(ep, "ext1", K1, T1, Twe1, frames, kind, calh["dq"], cfg.tau_ms, cfg.dil_px, hw)
    v2 = View(ep, "ext2", K2, T2, Twe2, frames, kind, calh["dq"], cfg.tau_ms, cfg.dil_px, hw)
    # queries on the held object: between-finger SAM2 mask at k_h (and, with query_frames=3, at mid- and late-hold)
    q1, g1 = joints_at(ep, "ext1", cfg.tau_ms)
    qframes = [k_h]
    if cfg.query_frames >= 3:
        k_mid, k_late = (a + b) // 2, b - cfg.hold_offset
        qframes += [k for k in (k_mid, k_late) if k > qframes[-1] + 5]
    queries, masks_px = [], []
    for qi, kq in enumerate(qframes):
        link = render(K1, T1, q1[kq], g1[kq], hw, kind=kind, dq=calh["dq"])["link"]
        box, centre = finger_box(ep, calh, cfg.tau_ms, kq, hw)
        negs = robot_negatives(link, box)
        sam.set_image(f1[kq])
        ms, sc, _ = sam.predict(point_coords=np.r_[[centre], negs],
                                point_labels=np.r_[1, np.zeros(len(negs))].astype(int), box=box, multimask_output=True)
        mask = ms[int(np.argmax(sc))].astype(bool) & ~v1.arm[kq]
        masks_px.append(int(mask.sum()))
        if qi == 0:
            rec.update({"k_h": int(k_h), "sam_score": float(np.max(sc))})
        if mask.sum() < 30:
            continue
        gray = cv2.cvtColor(f1[kq], cv2.COLOR_RGB2GRAY)
        c = cv2.goodFeaturesToTrack(gray, cfg.n_queries, 0.01, 3, mask=mask.astype(np.uint8))
        pq = c.reshape(-1, 2) if c is not None else np.zeros((0, 2))
        if len(pq) < cfg.n_queries:
            vv, uu = np.nonzero(mask)
            i = np.random.default_rng(qi).choice(len(uu), min(cfg.n_queries - len(pq), len(uu)), replace=False)
            pq = np.r_[pq, np.c_[uu[i], vv[i]].astype(float)]
        queries.append(np.c_[np.full(len(pq), kq - a), pq])
    rec.update({"query_frames": [int(k) for k in qframes], "mask_px": masks_px})
    if not queries:
        return {**rec, "status": "no_object_mask"}
    queries = np.concatenate(queries)
    pts = queries[:, 1:]
    tr1, vis1 = cotrack(trk, f1[a:b + 1], queries, device)
    # stereo supplement
    c3 = load_calib(ep.uuid)
    base = (c3.stereo.get("ext1", {}) or {}).get("baseline_m") if c3 else None
    st = None
    if cfg.sigma_d_px > 0 or calib_dump or audit:
        st = stereo_depths(ep.uuid, ep.serial["ext1"], frames[::cfg.stereo_every], Kf[0, 0], base or 0.12)
    rec["has_stereo"] = st is not None
    rec["stereo_fb"] = float(Kf[0, 0] * (base or 0.12))
    use_depth = st is not None and cfg.sigma_d_px > 0
    g_hold = float(np.median(ep.g[grasp.t_c:grasp.t_release]))
    hulls = robot_hulls(g_hold, kind, cfg.hull_margin_m)
    pts_out, reasons, dump = [], Counter(), []
    fb = Kf[0, 0] * (base or 0.12)
    # pass 1: ray-only triangulation and the stereo observations along the same rays
    pre = []
    for j in range(len(pts)):
        t_ray, ks = _two_pass(v1, tr1, vis1, j, a, frames, cfg)
        obs = None
        if st is not None and t_ray is not None:
            zs, Os, Ds, rfs = [], [], [], []
            for k in ks:
                if k not in st:
                    continue
                uvf = (tr1[k - a, j] + 0.5) * 2 - 0.5
                z = sample_depth(st[k], uvf)
                if not np.isfinite(z):
                    continue
                ray_c = np.array([(uvf[0] - Kf[0, 2]) / Kf[0, 0], (uvf[1] - Kf[1, 2]) / Kf[1, 1], 1.0])
                Ei = np.linalg.inv(Twe1[k])
                d = Ei[:3, :3] @ (T1[:3, :3] @ ray_c)
                Os.append(Ei[:3, :3] @ T1[:3, 3] + Ei[:3, 3])    # C_g centre in the hand frame
                rfs.append(np.linalg.norm(d))                     # ray length per unit camera-axial depth
                Ds.append(d / np.linalg.norm(d))
                zs.append(z)
            if zs:
                obs = (np.array(Os), np.array(Ds), np.array(rfs), np.array(zs))
        pre.append((t_ray, ks, obs))
    # per-segment stereo bias from well-conditioned ray-only points (sigma <= 3 mm, span >= 15 deg); the bias is
    # camera-specific (dev: 0.6-2.8 px), so it is estimated here when possible and otherwise covered by a prior
    s_bias = []
    for t_ray, ks, obs in pre:
        if obs is None or np.sqrt(np.trace(t_ray.cov)) * 1e3 > 3.0 or t_ray.span_deg < 15:
            continue
        Os, Dz, rfs, zs = obs
        along = np.einsum("fi,fi->f", Os + Dz * (zs * rfs)[:, None] - t_ray.X, Dz)
        s_bias += list(along * fb / (rfs * zs ** 2))
        if calib_dump:
            dump += [{"z": float(z_), "along_m": float(a_)} for z_, a_ in zip(zs, along)]
    n_bias_pts = sum(1 for t_ray, _, obs in pre if obs is not None and t_ray is not None
                     and np.sqrt(np.trace(t_ray.cov)) * 1e3 <= 3.0 and t_ray.span_deg >= 15)
    if n_bias_pts >= cfg.bias_min_points and len(s_bias) >= cfg.bias_min_obs:
        bias_seg, sdb = float(np.median(s_bias)), cfg.sigma_db_px
    else:
        bias_seg, sdb = 0.0, cfg.sigma_db_prior_px
    rec.update({"stereo_bias_px": bias_seg if s_bias else None, "stereo_bias_points": n_bias_pts,
                "stereo_bias_used_px": bias_seg, "stereo_sigma_db_used_px": sdb})
    # pass 2: joint triangulation with the (bias-corrected) stereo supplement, then the certificate
    for j, (t_ray, ks, obs) in enumerate(pre):
        row = {"j": j, "n_rays": len(ks)}
        Z = Dz = sz = None
        if use_depth and obs is not None:
            Os, Dz, rfs, zs = obs
            zc = fb / (fb / zs + bias_seg)
            Z = Os + Dz * (zc * rfs)[:, None]
            # per-observation noise plus a per-segment systematic term that does not average out:
            # total along-ray variance of the N observations = (z^2 r / fb)^2 (sigma_d^2 / N + sigma_db^2)
            sz = zc ** 2 * rfs / fb * np.sqrt(cfg.sigma_d_px ** 2 + len(zc) * sdb ** 2)
        if sz is not None:
            O, D = v1.rays(tr1, j, a, ks)
            t = triangulate(O, D, K1[0, 0], cfg.sigma_u_px, Z=Z, Dz=Dz, sz=sz)
        else:
            t = t_ray
        state = _gate(t, cfg, hulls, use_depth and sz is not None)
        state_ray = _gate(t_ray, cfg, hulls, False)
        reasons[state] += 1
        row.update({"state": state, "state_ray": state_ray, "used_depth": bool(use_depth and sz is not None)})
        for nm, tt in (("", t), ("ray_", t_ray)):
            if tt is not None:
                row.update({nm + "X": tt.X.tolist(), nm + "sigma_mm": float(np.sqrt(np.trace(tt.cov)) * 1e3),
                            nm + "rms_px": tt.rms_px, nm + "span_deg": tt.span_deg, nm + "n": tt.n})
        if t is not None:
            row.update({"n_depth": t.n_depth, "depth_chi": t.depth_chi, "_t": t, "_ks": ks})
        pts_out.append(row)
    cert = [p for p in pts_out if p["state"] == "ok"]
    rec.update({"n_queries": len(pts), "n_cert": len(cert), "reasons": dict(reasons)})
    # amodal labels + geometric visibility in C_g
    for p in cert:
        Xe = np.array(p["X"])
        occ = []
        for k in frames:
            uv, z = v1.project(Xe, k)
            if z <= 0 or not v1.inside(uv):
                occ.append(1)
                continue
            zr = v1.zr[k][int(round(uv[1])), int(round(uv[0]))]
            occ.append(int(np.isfinite(zr) and zr < z - cfg.eps_vis_m))
        p["occluded_frames"] = int(sum(occ))
        p["occ_mask"] = occ
    rec["n_point_frames"] = len(cert) * len(frames)
    rec["occluded_share"] = float(np.mean([o for p in cert for o in p["occ_mask"]])) if cert else None
    # acceptance: cross-camera closure
    if closure and cert:
        qs2, idx2 = [], []
        for p in cert:
            t = p["_t"]
            _, D1j = v1.rays(tr1, p["j"], a, p["_ks"])
            D1j = D1j[t.inliers]
            best = None
            for k in frames:
                uv, z = v2.project(t.X, k)
                if z <= 0 or not v2.inside(uv):
                    continue
                x, y = int(round(uv[0])), int(round(uv[1]))
                zr = v2.zr[k][y, x]
                if v2.arm[k][y, x] or (np.isfinite(zr) and zr < z - cfg.eps_vis_m):
                    continue
                _, d2 = rays_in_hand(Twe2[k][None], K2, T2, uv[None])
                ang = float(np.degrees(np.arccos(np.clip((D1j @ d2[0]).max(), -1, 1))))
                if best is None or ang < best[0]:
                    best = (ang, k, uv)
            if best is not None:
                qs2.append([best[1] - a, best[2][0], best[2][1]])
                idx2.append((p, best[0]))
        if qs2:
            tr2, vis2 = cotrack(trk, f2[a:b + 1], np.array(qs2), device)
            for i, (p, ang) in enumerate(idx2):
                t2, _ = _two_pass(v2, tr2, vis2, i, a, frames, cfg)
                st2 = _gate(t2, cfg, hulls, False)
                p.update({"c2_state": st2, "view_angle_deg": ang})
                if t2 is not None:
                    p["sigma_a_mm"] = float(np.sqrt(np.trace(t2.cov)) * 1e3)
                if st2 == "ok":
                    d = p["_t"].X - t2.X
                    p.update({"closure_mm": float(np.linalg.norm(d) * 1e3), "X_a": t2.X.tolist(),
                              "mahal": float(np.sqrt(d @ np.linalg.solve(p["_t"].cov + t2.cov, d)))})
    # audit: lifted C_g tracks vs label and vs the C_a-based reference, split by geometric visibility
    if audit and cert:
        ks_l = frames[::cfg.lift_every]
        depth = {}
        if moge is not None:
            depth["mono"] = moge_depths(moge, {k: f1[k] for k in ks_l}, K1, device)
        if st is not None:
            depth["stereo"] = {k: st[k] for k in ks_l if k in st}       # as an off-the-shelf pipeline would use it
            if bias_seg:   # per-segment self-calibrated stereo (not an off-the-shelf pipeline; reported separately)
                depth["stereo_corr"] = {k: fb / (fb / st[k] + bias_seg) for k in ks_l if k in st}
        for p in cert:
            Xe, j = np.array(p["X"]), p["j"]
            Xa = np.array(p["X_a"]) if "X_a" in p else None
            # ray-only label (no stereo inside) for the C_g internal-consistency check against stereo lifting
            Xr = np.array(p["ray_X"]) if p.get("state_ray") == "ok" and "ray_X" in p else None
            rows = []
            for k in ks_l:
                uv = tr1[k - a, j]
                Xl = Twe1[k][:3, :3] @ Xe + Twe1[k][:3, 3]
                r = {"k": k, "occ": p["occ_mask"][k - a], "trk_vis": bool(vis1[k - a, j])}
                if Xa is not None:
                    r["lab_vs_ref_mm"] = float(np.linalg.norm(Xl - (Twe1[k][:3, :3] @ Xa + Twe1[k][:3, 3])) * 1e3)
                for dn, Dm in depth.items():
                    if k not in Dm:
                        continue
                    z = sample_depth(Dm[k], (uv + 0.5) * 2 - 0.5 if dn.startswith("stereo") else uv)
                    if not np.isfinite(z):
                        continue
                    Xc = np.array([(uv[0] - K1[0, 2]) / K1[0, 0] * z, (uv[1] - K1[1, 2]) / K1[1, 1] * z, z])
                    Xt = T1[:3, :3] @ Xc + T1[:3, 3]
                    r[f"trk_{dn}_vs_lab_mm"] = float(np.linalg.norm(Xt - Xl) * 1e3)
                    if Xr is not None:
                        r[f"trk_{dn}_vs_raylab_mm"] = float(np.linalg.norm(Xt - (Twe1[k][:3, :3] @ Xr + Twe1[k][:3, 3])) * 1e3)
                    if Xa is not None:
                        r[f"trk_{dn}_vs_ref_mm"] = float(np.linalg.norm(Xt - (Twe1[k][:3, :3] @ Xa + Twe1[k][:3, 3])) * 1e3)
                rows.append(r)
            p["audit"] = rows
        if hook is not None:
            hook({"ep": ep, "cfg": cfg, "cert": cert, "depth": depth, "ks_l": ks_l, "frames": frames, "a": a,
                  "tr1": tr1, "vis1": vis1, "K1": K1, "T1": T1, "Twe1": Twe1, "kind": kind, "dq": calh["dq"],
                  "q1": q1, "g1": g1, "hw": hw, "f1": f1, "device": device, "queries": queries})
    for p in pts_out:
        for key in ("_t", "_ks", "occ_mask"):
            p.pop(key, None)
    rec["points"] = pts_out
    if calib_dump:
        rec["stereo_dump"] = dump
    rec["status"] = "ok"
    return rec
