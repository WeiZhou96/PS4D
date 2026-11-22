"""Read-only adapter for the DROID-3D calibration release (HF jaibrdhn/droid_3d_extrinsics, CC BY 4.0).

Conventions (dataset card + DepthWorld depth_extras/extrinsics.py, MIT):
  - T0_ext1_in_world / T1_ext2_in_world are world->camera VIEW matrices; camera pose = inv(view).
  - dq (7) are per-robot joint-encoder offsets: the published rig was optimized against FK(q + dq),
    so the end-effector track used for propagation is T_WE(t) = FK(q(t) + dq).
  - dT_gw is a per-robot right-multiplied gripper correction; it cancels in rigid propagation
    (^E X = T_WE(tc)^-1 X, X_hat = T_WE(t) ^E X) and is only needed for the wrist camera:
        T_wristcam_from_base(t) = FLIP . T_gw_static . dT_gw . inv(FK(q(t) + dq))
    with T_gw_static from the per-robot gripper2wrist asset; stored wrist images are rotated 180 deg
    in plane relative to that frustum, so the image-aligned pose is inv(view) . Rz(180).
Nothing here estimates or writes calibration; episodes are joined on episode_uuid only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .paths import D3D_DIR, G2W_ASSET
from .fk import franka_fk

FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])
RZ180 = np.diag([-1.0, -1.0, 1.0, 1.0])


@dataclass
class D3DCalib:
    uuid: str
    T_wc: dict            # 'ext1'/'ext2' -> 4x4 camera-to-base (left lens)
    K_hd: dict            # role -> 3x3 left intrinsics at 1280x720 (None if absent)
    stereo: dict          # role -> {'baseline_m', 'T_right_from_left', 'K_right', 'disto_left'}
    dq: np.ndarray
    dT_gw: np.ndarray
    robot_serial: str

    def ee_track(self, q: np.ndarray) -> np.ndarray:
        """Flange track consistent with the published rig: FK(q + dq)."""
        return franka_fk(q, self.dq)

    def wrist_c2w(self, q: np.ndarray, image_aligned: bool = True) -> np.ndarray | None:
        T_gw = wrist_static(self.robot_serial)
        if T_gw is None:
            return None
        view = FLIP @ T_gw @ self.dT_gw @ np.linalg.inv(self.ee_track(q))
        pose = np.linalg.inv(view)
        return pose @ RZ180 if image_aligned else pose


@lru_cache(maxsize=1)
def g2w_asset() -> dict:
    d = json.load(open(G2W_ASSET))
    return {k: np.asarray(v["mean_mat"], np.float64) for k, v in d.items()}


def wrist_static(robot_serial: str) -> np.ndarray | None:
    """T_gw_static lookup as in DepthWorld get_wrist_static_transform: literal key, then case-insensitive
    (asset keys are e.g. 'fr3-...' while records carry 'FR3-...')."""
    a = g2w_asset()
    if robot_serial in a:
        return a[robot_serial]
    hits = [v for k, v in a.items() if k.lower() == robot_serial.lower()]
    return hits[0] if hits else None


def _K(c: dict) -> np.ndarray:
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict]:
    ext, intr = {}, {}
    with open(os.path.join(D3D_DIR, "extrinsics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            ext[r["episode_uuid"]] = r
    p = os.path.join(D3D_DIR, "camera_intrinsics.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                intr[r["episode_uuid"]] = r
    return ext, intr


def load_calib(uuid: str) -> D3DCalib | None:
    ext, intr = _tables()
    r = ext.get(uuid)
    if r is None:
        return None
    T_wc = {"ext1": np.linalg.inv(np.asarray(r["T0_ext1_in_world"], np.float64)),
            "ext2": np.linalg.inv(np.asarray(r["T1_ext2_in_world"], np.float64))}
    K_hd, stereo = {}, {}
    cams = intr.get(uuid, {}).get("cameras", {})
    for role in ("ext1", "ext2", "wrist"):
        c = cams.get(role)
        if not c:
            K_hd[role] = None
            continue
        K_hd[role] = _K(c["left"])
        stereo[role] = {"baseline_m": c.get("stereo_baseline_m"),
                        "T_right_from_left": np.asarray(c["T_right_from_left"]) if "T_right_from_left" in c else None,
                        "K_right": _K(c["right"]) if "right" in c else None,
                        "disto_left": np.asarray(c["left"].get("disto", []))}
    return D3DCalib(uuid=uuid, T_wc=T_wc, K_hd=K_hd, stereo=stereo,
                    dq=np.asarray(r["dq_joint_correction"], np.float64),
                    dT_gw=np.asarray(r["dT_gw_gripper_in_world_correction"], np.float64),
                    robot_serial=str(r.get("robot_serial", "")))


def scale_K(K_hd: np.ndarray, s: float) -> np.ndarray:
    """1280x720 intrinsics -> decoded resolution, same pixel-centre convention as RawEpisode.intrinsics."""
    return np.array([[K_hd[0, 0] * s, 0, (K_hd[0, 2] + 0.5) * s - 0.5],
                     [0, K_hd[1, 1] * s, (K_hd[1, 2] + 0.5) * s - 0.5], [0, 0, 1.0]])


def episode_calib(ep, scale: float = 0.5) -> dict:
    """Calibration of an episode: DROID-3D when published, otherwise the factory set. Each set is used whole
    (extrinsics with their own intrinsics and FK offsets), never mixed. A missing published calibration
    never rejects a segment by itself; the pre-specified quality checks do. `source` must be carried into
    every output so DROID-3D and factory segments are reported separately.
    """
    c = load_calib(ep.uuid)
    if c is not None:
        if any(c.K_hd.get(r) is None for r in ("ext1", "ext2")):
            raise ValueError(f"{ep.uuid}: DROID-3D extrinsics without intrinsics")
        return {"source": "droid3d", "T_wc": dict(c.T_wc), "dq": c.dq, "ee_track": c.ee_track(ep.q),
                "K": {r: scale_K(c.K_hd[r], scale) for r in ("ext1", "ext2")}}
    return {"source": "factory", "T_wc": dict(ep.T_wc), "dq": None, "ee_track": franka_fk(ep.q),
            "K": {r: ep.intrinsics(r, scale) for r in ("ext1", "ext2")}}
