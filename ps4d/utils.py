"""Small shared helpers: file locations, mask packing, depth sampling."""
import hashlib
import os

import numpy as np

# sha256 of the released reliability model (weights/reliability_model.pkl)
RELIABILITY_SHA256 = "dea113f85ebc50f02107b57d239f45075293d65cad975db37a0bd79aa9b02a94"
# sha256 of the released variance model of 4D observations (weights/depth4d_variance_model.pkl, ablation only)
DEPTH4D_VARIANCE_SHA256 = "a8f440ced26137472f52c3c40c4aecd8b848b37d9ccb8fcd15e7e354dc0cc8c1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def checkpoint_dir():
    """Directory with third-party checkpoints (CoTracker3, Pi3X, ...): $PS4D_CHECKPOINTS, default ./checkpoints."""
    return os.environ.get("PS4D_CHECKPOINTS", os.path.join(REPO_ROOT, "checkpoints"))


def third_party_dir():
    """Directory with third-party code checkouts (Pi3, VGGT, ...): $PS4D_THIRD_PARTY, default ./third_party."""
    return os.environ.get("PS4D_THIRD_PARTY", os.path.join(REPO_ROOT, "third_party"))


def default_reliability_model():
    return os.path.join(REPO_ROOT, "weights", "reliability_model.pkl")


def default_depth4d_variance_model():
    return os.path.join(REPO_ROOT, "weights", "depth4d_variance_model.pkl")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_masks(masks):
    """(T, H, W) bool -> (T, H, ceil(W / 8)) uint8, bits packed along x."""
    return np.packbits(np.asarray(masks, bool), axis=-1)


def unpack_masks(packed, width):
    """Inverse of pack_masks."""
    return np.unpackbits(packed, axis=2)[:, :, :width].astype(bool)


def sample_depth(depth, uv, r=1):
    """Median of the valid depths in the (2r + 1)^2 window around the rounded pixel uv; NaN if none."""
    H, W = depth.shape
    x, y = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= x < W and 0 <= y < H):
        return np.nan
    w = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    w = w[np.isfinite(w) & (w > 0)]
    return float(np.median(w)) if len(w) else np.nan


def shard(items, spec):
    """Every n-th item starting at i, for spec 'i/n'."""
    i, n = (int(x) for x in spec.split("/"))
    return items[i::n]
