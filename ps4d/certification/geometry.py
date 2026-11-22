"""SE(3) helpers and pinhole projection (OpenCV axes, metres, robot base = world)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline


def pose6_to_mat(p: np.ndarray) -> np.ndarray:
    """[..., 6] DROID pose (x, y, z, roll, pitch, yaw; extrinsic xyz Euler) -> [..., 4, 4]."""
    p = np.asarray(p, np.float64)
    M = np.zeros(p.shape[:-1] + (4, 4))
    M[..., :3, :3] = Rotation.from_euler("xyz", p.reshape(-1, 6)[:, 3:]).as_matrix().reshape(p.shape[:-1] + (3, 3))
    M[..., :3, 3] = p[..., :3]
    M[..., 3, 3] = 1.0
    return M


def transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply (..., 4, 4) to (..., N, 3)."""
    return np.einsum("...ij,...nj->...ni", T[..., :3, :3], X) + T[..., None, :3, 3]


def project(K: np.ndarray, T_wc: np.ndarray, X_w: np.ndarray):
    """World points (..., N, 3) -> pixels (..., N, 2), depth (..., N). T_wc is camera-to-world."""
    T_cw = np.linalg.inv(T_wc)
    Xc = transform(T_cw, X_w)
    z = Xc[..., 2]
    uv = Xc[..., :2] / np.clip(z[..., None], 1e-6, None) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def angular_speed(R: np.ndarray, fps: float) -> np.ndarray:
    """Rotation magnitude between consecutive frames, rad/s; sample k sits at time k + 0.5 frames."""
    rel = Rotation.from_matrix(np.einsum("tji,tjk->tik", R[:-1], R[1:]))
    return np.linalg.norm(rel.as_rotvec(), axis=1) * fps


class PoseInterp:
    """Continuous-time interpolation of a pose track sampled at frame indices 0..T-1.

    Positions use a cubic spline, rotations a cubic RotationSpline (continuous angular velocity;
    SLERP is piecewise linear and biases sub-frame shifts of the angular-speed signal by a few ms).
    Queries outside [0, T-1] are clamped.
    """

    def __init__(self, T: np.ndarray):
        n = len(T)
        self.t = np.arange(n, dtype=np.float64)
        self.pos = CubicSpline(self.t, T[:, :3, 3], axis=0)
        self.rot = RotationSpline(self.t, Rotation.from_matrix(T[:, :3, :3]))
        self.n = n

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, np.float64), 0, self.n - 1)
        M = np.zeros(t.shape + (4, 4))
        M[..., :3, :3] = self.rot(t.ravel()).as_matrix().reshape(t.shape + (3, 3))
        M[..., :3, 3] = self.pos(t)
        M[..., 3, 3] = 1.0
        return M
