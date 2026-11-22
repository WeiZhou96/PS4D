"""Minimal robot renderer: theoretical projection of the robot model.

Arm: MuJoCo Menagerie Panda / FR3 visual meshes (Apache-2.0) placed on the forward-kinematics link frames
(panda_link0..7).  Gripper: DROID-VP CAD gripper points (CC BY 4.0) tabulated in the flange frame versus the gripper
opening g, rendered as convex hulls of the rigid base and of each finger.  Output per pixel: link id (-1 none; 0..7
arm links, 9 gripper base, 10/11 fingers), camera-axial depth z and validity.  The render is a theoretical projection,
not the visible robot region (objects may be in front of the robot), and its depth is not ground truth.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import open3d as o3d
from scipy.interpolate import CubicSpline

from .paths import GRIPPER_MODEL, MENAGERIE
from .fk import franka_link_frames

MODELS = {"panda": ("franka_emika_panda", "panda_nohand.xml", ""),
          "fr3": ("franka_fr3", "fr3.xml", "fr3_")}
ID_GRIPPER_BASE, ID_FINGER0, ID_FINGER1 = 9, 10, 11


def arm_kind(robot_serial: str) -> str:
    return "fr3" if "fr3" in str(robot_serial).lower() else "panda"


@lru_cache(maxsize=2)
def arm_meshes(kind: str, max_tris_per_link: int = 4000) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """link index -> (V (n,3) in link frame, F (m,3)) merged visual meshes."""
    sub, xml, prefix = MODELS[kind]
    root = ET.parse(os.path.join(MENAGERIE, sub, xml)).getroot()
    meshdir = os.path.join(MENAGERIE, sub, root.find("compiler").get("meshdir", ""))
    files = {}
    for m in root.iter("mesh"):
        f = m.get("file")
        files[m.get("name") or os.path.splitext(f)[0]] = os.path.join(meshdir, f)
    out = {}
    for body in root.iter("body"):
        name = body.get("name", "")
        if not name.startswith(prefix + "link"):
            continue
        li = int(name[len(prefix) + 4:])
        merged = o3d.geometry.TriangleMesh()
        for g in body.findall("geom"):
            if "visual" not in (g.get("class") or "") or g.get("mesh") is None:
                continue
            if g.get("pos") or g.get("quat"):
                raise NotImplementedError(f"geom offset in {name}")
            merged += o3d.io.read_triangle_mesh(files[g.get("mesh")])
        if len(merged.triangles) > max_tris_per_link:
            merged = merged.simplify_quadric_decimation(max_tris_per_link)
        out[li] = (np.asarray(merged.vertices, np.float64), np.asarray(merged.triangles, np.int32))
    return out


@lru_cache(maxsize=1)
def gripper_model() -> dict:
    z = np.load(GRIPPER_MODEL)
    return {k: z[k] for k in ("base", "g_grid", "fingers", "side")}


def _hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    h, _ = pc.compute_convex_hull()
    return np.asarray(h.vertices, np.float64), np.asarray(h.triangles, np.int32)


def gripper_parts(g: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Gripper hull meshes in the flange frame at opening g (0 open .. 1 closed)."""
    m = gripper_model()
    g = float(np.clip(g, 0.0, 1.0))
    fing = np.stack([np.stack([np.interp(g, m["g_grid"], m["fingers"][:, j, d]) for d in range(3)])
                     for j in range(m["fingers"].shape[1])])
    return {ID_GRIPPER_BASE: _hull(m["base"]), ID_FINGER0: _hull(fing[m["side"] == 0]),
            ID_FINGER1: _hull(fing[m["side"] == 1])}


def render(K: np.ndarray, T_wc: np.ndarray, q: np.ndarray, g: float, hw: tuple[int, int],
           kind: str = "panda", dq: np.ndarray | None = None) -> dict:
    """Ray-cast the robot into a pinhole camera (OpenCV axes, pixel centres at integer coords)."""
    H, W = hw
    frames = franka_link_frames(q, dq)                         # (9, 4, 4) link poses in base
    scene = o3d.t.geometry.RaycastingScene()
    gid2link = {}
    parts = [(li, frames[li], V, F) for li, (V, F) in arm_meshes(kind).items()]
    parts += [(lid, frames[8], V, F) for lid, (V, F) in gripper_parts(g).items()]
    for lid, T, V, F in parts:
        Vw = V @ T[:3, :3].T + T[:3, 3]
        gid = scene.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        gid2link[gid] = lid
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    d_w = d_cam @ T_wc[:3, :3].T                               # unnormalised: t_hit == axial depth Z
    o_w = np.broadcast_to(T_wc[:3, 3], d_w.shape)
    rays = o3d.core.Tensor(np.concatenate([o_w, d_w], -1).reshape(-1, 6).astype(np.float32))
    ans = scene.cast_rays(rays)
    t = ans["t_hit"].numpy().reshape(H, W)
    gid = ans["geometry_ids"].numpy().reshape(H, W).astype(np.int64)
    valid = np.isfinite(t)
    link = np.full((H, W), -1, np.int8)
    for k, lid in gid2link.items():
        link[(gid == k) & valid] = lid
    return {"link": link, "z": np.where(valid, t, np.nan).astype(np.float32), "valid": valid}


def joints_at(ep, role: str, tau_ms: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """q (n,7) and g (n,) at each frame's capture time of camera `role` (NUC clock) + residual tau."""
    n = len(ep.t_cap[role])
    idx = np.arange(ep.T)
    s = np.interp(ep.t_cap[role] + tau_ms, ep.t_read, idx)
    q = CubicSpline(idx, ep.q, axis=0)(s)
    g = np.interp(s, idx, ep.g)
    return q[:n], g[:n]
