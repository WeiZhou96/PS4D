"""Grasp segmentation from the gripper signal g(t): closing start, closure, hold and release frames.
Rule-based; the thresholds are the defaults of segment_grasps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grasp:
    t_close_start: int   # g starts rising
    t_c: int             # closure complete (g plateau)
    t_release: int       # g starts falling (exclusive end of hold)
    g_hold: float        # median g during hold
    full_closed: bool    # hold plateau in the fully-closed band -> nothing in hand
    transport: tuple[int, int]  # [start, end) frames with stable g after t_c


def segment_grasps(g: np.ndarray, closed_thr: float = 0.3, full_thr: float = 0.95,
                   dg_thr: float = 0.01, settle: int = 2, min_hold: int = 8) -> list[Grasp]:
    g = np.asarray(g, np.float64)
    T = len(g)
    dg = np.abs(np.diff(g, prepend=g[0]))
    closed = g > closed_thr
    out = []
    k = 0
    while k < T:
        if not closed[k]:
            k += 1
            continue
        a = k
        while k < T and closed[k]:
            k += 1
        b = k  # [a, b) closed
        if b - a < min_hold:
            continue
        s = a
        while s > 0 and g[s - 1] < g[s] - 1e-4:   # walk back to the rise onset
            s -= 1
        tc = a
        while tc < b - settle and not (dg[tc:tc + settle] <= dg_thr).all():
            tc += 1
        r = b
        while r > tc + 1 and g[r - 1] < g[r - 2] - dg_thr:   # walk back from the drop
            r -= 1
        if r - tc < min_hold:
            continue
        gh = float(np.median(g[tc:r]))
        stable = np.where(dg[tc:r] <= dg_thr)[0] + tc
        tr = (int(stable[0]), int(stable[-1]) + 1) if len(stable) else (tc, tc)
        out.append(Grasp(int(s), int(tc), int(r), gh, gh >= full_thr, tr))
    return out
