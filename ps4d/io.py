"""Reading and writing the per-clip files of a prepared dataset and the results files."""
import gzip
import json
import os

import numpy as np

from .utils import unpack_masks


def load_npz(path):
    return dict(np.load(path)) if os.path.exists(path) else None


def load_tracks(path):
    """{tracks, visible, query_frame} written by scripts/run_masks_tracks.py, or None."""
    return load_npz(path)


def load_masks(path, width):
    z = load_npz(path)
    return None if z is None else unpack_masks(z["masks"], width)


def load_depth(path):
    """(T, H, W) float32 depth (stored as float16), or None."""
    return None if not os.path.exists(path) else np.load(path)["depth"].astype(np.float32)


def load_lifted(path, key="coords"):
    """(T, N, 3) camera-frame coordinates as stored (float32), or None."""
    return None if not os.path.exists(path) else np.load(path)[key]


def parse_sources(specs):
    """['name=dir[:key]', ...] -> {name: (dir, key)} preserving order; key defaults to 'coords'."""
    out = {}
    for s in specs or []:
        name, rest = s.split("=", 1)
        if ":" in rest and not os.path.exists(rest):
            d, key = rest.rsplit(":", 1)
        else:
            d, key = rest, "coords"
        out[name] = (d, key)
    return out


def write_results(path, clips, meta=None):
    payload = {"meta": meta or {}, "clips": clips}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def read_results(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)
