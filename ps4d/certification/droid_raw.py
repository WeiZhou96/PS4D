"""DROID raw release 1.0.1 (gs://gresearch/robotics/droid_raw/1.0.1, public HTTPS).

Per episode: metadata json, trajectory.h5 (states and per-step timestamps) and three 720p MP4s; MP4 frame k belongs
to trajectory step k.  The factory extrinsics come from the metadata; intrinsics are looked up by camera serial in the
DROID-VP table.  The DROID-3D calibration (ps4d.certification.calibration) replaces both when available.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache

import av
import h5py
import numpy as np

from .paths import DROIDVP, RAW_DIR
from .geometry import pose6_to_mat

BUCKET = "https://storage.googleapis.com/gresearch/robotics/droid_raw/1.0.1"
LIST_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
ROLES = {"wrist": "wrist", "ext1": "exterior_1_left", "ext2": "exterior_2_left"}


def _get(url: str, timeout: float = 60, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2.0 * (i + 1))


def list_prefixes(prefix: str) -> list[str]:
    out, token = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            q["pageToken"] = token
        d = json.loads(_get(LIST_API + "?" + urllib.parse.urlencode(q)))
        out += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def fetch_episode(rel_dir: str, with_video: bool = True) -> str:
    """Download one raw episode (rel_dir like 'TRI/success/2023-11-16/Thu_Nov_16_14:20:16_2023/')."""
    rel_dir = rel_dir.rstrip("/") + "/"
    items = json.loads(_get(LIST_API + "?" + urllib.parse.urlencode(
        {"prefix": "robotics/droid_raw/1.0.1/" + rel_dir})))["items"]
    names = [o["name"].split(rel_dir, 1)[1] for o in items]
    meta_name = next(n for n in names if n.startswith("metadata_"))
    uuid = meta_name[len("metadata_"):-len(".json")]
    dst = os.path.join(RAW_DIR, uuid)
    want = [meta_name, "trajectory.h5"] + ([n for n in names if n.startswith("recordings/MP4/")] if with_video else [])
    for n in want:
        p = os.path.join(dst, n)
        if os.path.exists(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = _get(BUCKET + "/" + urllib.parse.quote(rel_dir + n), timeout=600)
        with open(p + ".tmp", "wb") as f:
            f.write(data)
        os.replace(p + ".tmp", p)
    return uuid


@lru_cache(maxsize=1)
def calib_tables() -> dict:
    cam = dict(np.load(os.path.join(DROIDVP, "droid/cameras.npz")))
    pts = dict(np.load(os.path.join(DROIDVP, "droid_points/episodes.npz")))
    return {"cam": cam, "ego_id": pts["ego_id"]}


@lru_cache(maxsize=1)
def serial_intrinsics() -> dict:
    """camera serial -> (fx, fy, cx, cy) at 1280x720, from DROID-VP rows whose source is the serial itself."""
    cam = calib_tables()["cam"]
    ser, K, src = cam["serial"], cam["intrinsics_hd"], cam["intrinsics_src"]
    out = {}
    for s, k in zip(ser[src == 0], K[src == 0]):
        out.setdefault(int(s), k)
    return out


@dataclass
class RawEpisode:
    uuid: str
    meta: dict
    ee: np.ndarray                  # (T, 4, 4) cartesian_position
    q: np.ndarray
    g: np.ndarray
    t_robot: np.ndarray             # (T,) ms, robot clock
    t_read: np.ndarray              # (T,) ms, NUC robot_state read_start
    t_cap: dict                     # role -> (T,) ms, camera estimated capture (NUC clock)
    serial: dict
    K: dict = field(default_factory=dict)     # role -> 3x3 at the decoded resolution
    T_wc: dict = field(default_factory=dict)  # ext role -> 4x4 camera-to-base (factory)
    _frames: dict = field(default_factory=dict, repr=False)

    @property
    def T(self) -> int:
        return len(self.g)

    def frames(self, role: str, scale: float = 0.5) -> np.ndarray:
        key = (role, scale)
        if key not in self._frames:
            p = os.path.join(RAW_DIR, self.uuid, "recordings/MP4", f"{self.serial[role]}.mp4")
            out = []
            with av.open(p) as c:
                for f in c.decode(c.streams.video[0]):
                    out.append(f.to_ndarray(format="rgb24", width=int(1280 * scale), height=int(720 * scale)))
            self._frames[key] = np.stack(out)
        return self._frames[key]

    def intrinsics(self, role: str, scale: float = 0.5) -> np.ndarray | None:
        k = serial_intrinsics().get(int(self.serial[role]))
        if k is None:
            return None
        fx, fy, cx, cy = k
        return np.array([[fx * scale, 0, (cx + 0.5) * scale - 0.5], [0, fy * scale, (cy + 0.5) * scale - 0.5], [0, 0, 1.0]])


def load_raw(uuid: str) -> RawEpisode:
    d = os.path.join(RAW_DIR, uuid)
    meta = json.load(open(os.path.join(d, f"metadata_{uuid}.json")))
    with h5py.File(os.path.join(d, "trajectory.h5"), "r") as f:
        rs = f["observation/robot_state"]
        ts = f["observation/timestamp"]
        serial = {r: str(meta[f"{r}_cam_serial"]) for r in ROLES}
        ep = RawEpisode(
            uuid=uuid, meta=meta, ee=pose6_to_mat(rs["cartesian_position"][:]), q=rs["joint_positions"][:],
            g=rs["gripper_position"][:].ravel(),
            t_robot=ts["robot_state/robot_timestamp_seconds"][:] * 1e3 + ts["robot_state/robot_timestamp_nanos"][:] / 1e6,
            t_read=ts["robot_state/read_start"][:].astype(np.float64),
            t_cap={r: ts[f"cameras/{s}_estimated_capture"][:].astype(np.float64) for r, s in serial.items()},
            serial=serial)
    for r in ("ext1", "ext2"):
        ep.T_wc[r] = pose6_to_mat(np.asarray(meta[f"{r}_cam_extrinsics"]))
    return ep
