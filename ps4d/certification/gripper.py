"""Gripper-frame helpers: capture-time poses, the between-finger box and SAM 2 prompts."""
from __future__ import annotations

import numpy as np

from .geometry import PoseInterp, project, transform
from .robot_render import gripper_model


def step_index(ep, role: str, tau_ms: float, frames) -> np.ndarray:
    """Continuous robot-step index at each frame's capture time (NUC clock) + tau."""
    return np.interp(ep.t_cap[role][np.asarray(frames)] + tau_ms, ep.t_read, np.arange(ep.T))


def pose_at(ep, ee_track: np.ndarray, role: str, tau_ms: float, frames=None) -> np.ndarray:
    """End-effector pose at camera `role`'s capture time of each frame + tau (NUC clock)."""
    t = ep.t_cap[role] if frames is None else ep.t_cap[role][np.asarray(frames)]
    s = np.interp(t + tau_ms, ep.t_read, np.arange(ep.T))
    return PoseInterp(ee_track)(s)


def g_at(ep, role: str, tau_ms: float, frame: int) -> float:
    return float(np.interp(np.interp(ep.t_cap[role][frame] + tau_ms, ep.t_read, np.arange(ep.T)), np.arange(ep.T), ep.g))


def grasp_box_flange(g: float, z_tip_band: float = 0.04, half_x: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Box between the fingers in the flange frame at opening g: (centre (3,), corners (8,3))."""
    m = gripper_model()
    fing = np.stack([np.stack([np.interp(g, m["g_grid"], m["fingers"][:, j, d]) for d in range(3)])
                     for j in range(m["fingers"].shape[1])])
    c0, c1 = fing[m["side"] == 0].mean(0), fing[m["side"] == 1].mean(0)
    zmax = fing[:, 2].max()
    centre = np.array([(c0[0] + c1[0]) / 2, (c0[1] + c1[1]) / 2, zmax - z_tip_band / 2])
    hy = abs(c0[1] - c1[1]) / 2
    corners = np.array([[centre[0] + sx * half_x, centre[1] + sy * hy, zmax + sz * z_tip_band / 2 - z_tip_band / 2]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return centre, corners


def robot_negatives(link: np.ndarray, box: np.ndarray, k: int = 4, margin: int = 8, seed: int = 0) -> np.ndarray:
    """Up to k spread-out rendered-robot pixels outside the (expanded) prompt box, as (x, y)."""
    vv, uu = np.nonzero(link >= 0)
    x0, y0, x1, y1 = box
    keep = (uu < x0 - margin) | (uu > x1 + margin) | (vv < y0 - margin) | (vv > y1 + margin)
    pts = np.c_[uu[keep], vv[keep]].astype(float)
    if not len(pts):
        return np.zeros((0, 2))
    rng = np.random.default_rng(seed)
    out = [pts[rng.integers(len(pts))]]
    for _ in range(k - 1):  # farthest-point sampling
        d = np.min(np.linalg.norm(pts[:, None] - np.array(out)[None], axis=2), axis=1)
        out.append(pts[int(np.argmax(d))])
    return np.array(out)


def finger_box(ep, cal: dict, tau_ms: float, k: int, hw) -> tuple[np.ndarray, np.ndarray]:
    """Between-finger box (x0, y0, x1, y1) and its centre in C1 at frame k (working resolution of cal['K'])."""
    H, W = hw
    Twe = pose_at(ep, cal["ee_track"], "ext1", tau_ms, [k])[0]
    c, corners = grasp_box_flange(g_at(ep, "ext1", tau_ms, k))
    uv, _ = project(cal["K"]["ext1"], cal["T_wc"]["ext1"], transform(Twe, np.r_[[c], corners][None]))
    uv = uv[0]
    box = np.array([uv[1:, 0].min(), uv[1:, 1].min(), uv[1:, 0].max(), uv[1:, 1].max()]).clip(0, [W - 1, H - 1, W - 1, H - 1])
    return box, uv[0]
