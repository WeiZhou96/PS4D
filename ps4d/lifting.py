"""Track-lifting baselines: the reference points' 2D positions lifted per frame with a model's depth map.

Each point is lifted at its pixel with the median valid depth of the 3 x 3 window, with the clip intrinsics, giving
camera-frame coordinates (T, N, 3), NaN where the depth is invalid.  The positions are the clip's ref_uv: CoTracker3
tracks of the certified points on DROID, the annotated projections of the reference points on DexYCB and HOT3D.
"""
import numpy as np

from .utils import sample_depth


def lift(uv, depth, K):
    """uv (T, N, 2), depth (T, H, W), K (3, 3) -> (T, N, 3) float32."""
    T, N = uv.shape[:2]
    C = np.full((T, N, 3), np.nan, np.float32)
    for t in range(T):
        for n in range(N):
            z = sample_depth(depth[t], uv[t, n])
            if np.isfinite(z) and z > 0:
                u, v = uv[t, n]
                C[t, n] = [(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    return C
