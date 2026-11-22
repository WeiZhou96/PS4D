"""Franka Panda / FR3 forward kinematics (modified DH, base -> panda_link8 flange), numpy.

Ported from DepthWorld `depth_extras/franka_fk.py` (MIT License, (c) 2026 Bardhan, Sivic, Petrik;
(c) 2025 Tsinghua University), so that read-only DROID-3D joint offsets `dq` are applied with the
same chain that the published extrinsics were optimized against: T_WE(t) = FK(q(t) + dq).
"""
from __future__ import annotations

import numpy as np

# (a, d, alpha) for joints 1..7, then the fixed link8 flange offset (theta = 0)
_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
    (0.0, 0.107, 0.0),
]


def _mdh(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """Trans_x(a) Rot_x(alpha) Rot_z(theta) Trans_z(d), batched over theta."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    M = np.zeros(theta.shape + (4, 4))
    M[..., 0, 0], M[..., 0, 1], M[..., 0, 3] = ct, -st, a
    M[..., 1, 0], M[..., 1, 1], M[..., 1, 2], M[..., 1, 3] = st * ca, ct * ca, -sa, -sa * d
    M[..., 2, 0], M[..., 2, 1], M[..., 2, 2], M[..., 2, 3] = st * sa, ct * sa, ca, ca * d
    M[..., 3, 3] = 1.0
    return M


def franka_fk(q: np.ndarray, dq: np.ndarray | None = None) -> np.ndarray:
    """(..., 7) joint angles (+ optional published offsets dq) -> (..., 4, 4) flange pose in base."""
    q = np.asarray(q, np.float64)
    if dq is not None:
        q = q + np.asarray(dq, np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i, (a, d, al) in enumerate(_DH[:7]):
        T = T @ _mdh(a, d, al, q[..., i])
    a, d, al = _DH[7]
    return T @ _mdh(a, d, al, np.zeros(q.shape[:-1]))


def franka_link_frames(q: np.ndarray, dq: np.ndarray | None = None) -> np.ndarray:
    """(..., 7) -> (..., 9, 4, 4) poses of panda_link0..panda_link8 in base (link0 = identity).

    The modified-DH frames coincide with the URDF / MuJoCo-menagerie link frames, so visual meshes
    defined in link frames can be placed directly.
    """
    q = np.asarray(q, np.float64)
    if dq is not None:
        q = q + np.asarray(dq, np.float64)
    T = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    out = [T.copy()]
    for i, (a, d, al) in enumerate(_DH[:7]):
        T = T @ _mdh(a, d, al, q[..., i])
        out.append(T.copy())
    a, d, al = _DH[7]
    out.append(T @ _mdh(a, d, al, np.zeros(q.shape[:-1])))
    return np.stack(out, axis=-3)
