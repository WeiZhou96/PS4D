"""Gripper-frame triangulation of held-object points.

While an object is held rigidly, a static camera C moves in the gripper (flange) frame: T_EC(t) = T_WE(t)^-1 T_WC.
A point fixed in the gripper frame and observed by C at many hold frames is triangulated from those rays.  The
over-determination yields a rigidity residual (the certificate) and a covariance (the error budget), with no depth
network.  Parts rigidly attached to the gripper (base, fingers, link 7) also triangulate consistently, so points
inside their hulls are rejected explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay


def rays_in_hand(T_we: np.ndarray, K: np.ndarray, T_wc: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand poses (F, 4, 4) at the frames' capture times and pixels (F, 2) -> ray origins and unit directions (F, 3)
    expressed in the hand frame."""
    d_c = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))]
    Ei = np.linalg.inv(T_we)
    O = np.einsum("fij,j->fi", Ei[:, :3, :3], T_wc[:3, 3]) + Ei[:, :3, 3]
    D = np.einsum("fij,fj->fi", Ei[:, :3, :3], d_c @ T_wc[:3, :3].T)
    return O, D / np.linalg.norm(D, axis=1, keepdims=True)


def _solve(O, D, w):
    P = np.eye(3)[None] - D[:, :, None] * D[:, None, :]
    A = np.einsum("f,fij->ij", w, P)
    return np.linalg.solve(A, np.einsum("f,fij,fj->i", w, P, O)), A


def _residual_px(X, O, D, f_px):
    v = X - O
    return np.linalg.norm(np.cross(v, D), axis=1) / np.maximum(np.einsum("fi,fi->f", v, D), 1e-6) * f_px


def _span_deg(D: np.ndarray, max_rays: int = 400) -> float:
    if len(D) > max_rays:
        D = D[np.linspace(0, len(D) - 1, max_rays).round().astype(int)]
    c = np.clip(D @ D.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(c.min())))


@dataclass
class Tri:
    X: np.ndarray        # (3,) hand-frame point
    cov: np.ndarray      # (3, 3) first-order covariance for angular noise sigma_px / f (and depth noise if used)
    inliers: np.ndarray  # (F,) bool, rays
    rms_px: float        # reprojection RMS over inlier rays
    span_deg: float      # largest angle between inlier rays
    n: int
    n_depth: int = 0     # inlier depth observations (stereo supplement)
    depth_chi: float = float("nan")  # RMS of along-ray depth residuals / their sigma, over depth inliers


def _solve_joint(O, D, w, Z, Dz, wz):
    P = np.eye(3)[None] - D[:, :, None] * D[:, None, :]
    A = np.einsum("f,fij->ij", w, P)
    b = np.einsum("f,fij,fj->i", w, P, O)
    if len(Z):
        Q = Dz[:, :, None] * Dz[:, None, :]
        A = A + np.einsum("f,fij->ij", wz, Q)
        b = b + np.einsum("f,fij,fj->i", wz, Q, Z)
    return np.linalg.solve(A, b), A


def triangulate(O: np.ndarray, D: np.ndarray, f_px: float, sigma_px: float = 1.0, k_mad: float = 3.0,
                iters: int = 3, Z: np.ndarray | None = None, Dz: np.ndarray | None = None,
                sz: np.ndarray | None = None, depth_gate: float = 3.0) -> Tri | None:
    """Weighted least-squares point closest to the rays, with a MAD gate on the angular residual.

    Optional depth supplement: hand-frame points Z (M, 3) measured by a depth sensor along rays Dz
    (M, 3) with along-ray std sz (M,) constrain only the along-ray coordinate, (Dz . (X - Z))^2 / sz^2; rays keep
    the lateral constraint. Depth terms with a normalised residual above `depth_gate` are dropped iteratively.
    """
    if len(O) < 3:
        return None
    use_z = Z is not None and len(Z) > 0
    Z = Z if use_z else np.zeros((0, 3))
    Dz = Dz if use_z else np.zeros((0, 3))
    wz_all = 1.0 / np.asarray(sz) ** 2 if use_z else np.zeros(0)
    keep = np.ones(len(O), bool)
    zkeep = np.ones(len(Z), bool)
    w = np.ones(len(O))
    X, _ = _solve_joint(O, D, w, Z, Dz, wz_all * 0.0 + (wz_all > 0))   # unweighted start
    for _ in range(iters):
        rng = np.maximum(np.einsum("fi,fi->f", X - O, D), 1e-3)
        w = 1.0 / (rng * sigma_px / f_px) ** 2
        e = _residual_px(X, O, D, f_px)
        med = np.median(e[keep])
        new = e <= med + k_mad * 1.4826 * (np.median(np.abs(e[keep] - med)) + 1e-3)
        if new.sum() < 3:
            break
        keep = new
        if use_z:
            chi = np.abs(np.einsum("fi,fi->f", X - Z, Dz)) * np.sqrt(wz_all)
            zkeep = chi <= depth_gate
        X, _ = _solve_joint(O[keep], D[keep], w[keep], Z[zkeep], Dz[zkeep], wz_all[zkeep])
    X, A = _solve_joint(O[keep], D[keep], w[keep], Z[zkeep], Dz[zkeep], wz_all[zkeep])
    e = _residual_px(X, O, D, f_px)[keep]
    chi = (np.abs(np.einsum("fi,fi->f", X - Z, Dz)) * np.sqrt(wz_all))[zkeep] if use_z else np.zeros(0)
    return Tri(X, np.linalg.inv(A), keep, float(np.sqrt(np.mean(e ** 2))), _span_deg(D[keep]), int(keep.sum()),
               int(zkeep.sum()) if use_z else 0, float(np.sqrt(np.mean(chi ** 2))) if len(chi) else float("nan"))


def robot_hulls(g: float, kind: str, margin_m: float = 0.003) -> list[Delaunay]:
    """Hulls (flange frame) of the parts rigid with the hand at opening g: gripper base, both fingers, link7."""
    from .robot_render import arm_meshes, gripper_model
    m = gripper_model()
    g = float(np.clip(g, 0.0, 1.0))
    fing = np.stack([np.stack([np.interp(g, m["g_grid"], m["fingers"][:, j, d]) for d in range(3)])
                     for j in range(m["fingers"].shape[1])])
    link7 = arm_meshes(kind)[7][0] - np.array([0.0, 0.0, 0.107])   # link8 = link7 . Trans_z(0.107)
    out = []
    for P in (m["base"], fing[m["side"] == 0], fing[m["side"] == 1], link7):
        c = P.mean(0)
        r = np.linalg.norm(P - c, axis=1, keepdims=True)
        out.append(Delaunay(c + (P - c) * (1.0 + margin_m / np.maximum(r, 1e-6))))
    return out


def inside_any(X: np.ndarray, hulls: list[Delaunay]) -> bool:
    return any(h.find_simplex(X[None])[0] >= 0 for h in hulls)
