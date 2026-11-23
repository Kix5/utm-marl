"""
conflict_detector.py
====================
Vectorized pairwise separation checker (CPU / NumPy).

Replaces the O(N^2) Python double-loop with a single vectorized distance
computation. Results are IDENTICAL to the brute-force version; only far faster
at scale (100 / 500 UAVs). Ready for 3D: check3d() adds a separate vertical
minimum (the "hockey-puck" separation standard).
"""

import numpy as np
from typing import Dict

MIN_SEPARATION_M   = 500   # horizontal separation minimum (m)
MIN_VSEPARATION_M  = 30    # vertical separation minimum (m) -- used by check3d


class ConflictDetector:
    def __init__(self, min_separation: float = MIN_SEPARATION_M,
                 min_vseparation: float = MIN_VSEPARATION_M):
        self.min_sep  = min_separation
        self.min_vsep = min_vseparation

    # ---- 2D horizontal separation (vectorized) --------------------------
    def check(self, positions: np.ndarray) -> Dict:
        n = len(positions)
        per_drone = np.zeros(n, dtype=bool)
        if n < 2:
            return {"pairs": [], "count": 0, "rate": 0.0, "per_drone": per_drone}

        xy = np.asarray(positions[:, :2], dtype=np.float64)   # (N,2)
        # all pairwise horizontal distances in one shot
        diff = xy[:, None, :] - xy[None, :, :]                # (N,N,2)
        dist = np.sqrt((diff * diff).sum(axis=2))             # (N,N)

        conflict = dist < self.min_sep
        np.fill_diagonal(conflict, False)                     # exclude self-pairs

        iu = np.triu_indices(n, k=1)                          # upper triangle only
        mask = conflict[iu]
        pi, pj = iu[0][mask], iu[1][mask]
        pairs = list(zip(pi.tolist(), pj.tolist()))

        if len(pairs):
            per_drone[pi] = True
            per_drone[pj] = True

        total = n * (n - 1) / 2
        return {"pairs": pairs, "count": len(pairs),
                "rate": len(pairs) / max(total, 1), "per_drone": per_drone}

    # ---- 3D separation: horizontal AND vertical minima (vectorized) -----
    def check3d(self, positions: np.ndarray) -> Dict:
        """Two aircraft conflict only if within BOTH the horizontal and the
        vertical minimum simultaneously (hockey-puck standard)."""
        n = len(positions)
        per_drone = np.zeros(n, dtype=bool)
        if n < 2:
            return {"pairs": [], "count": 0, "rate": 0.0, "per_drone": per_drone}

        p  = np.asarray(positions, dtype=np.float64)
        xy = p[:, :2]; z = p[:, 2]
        diff = xy[:, None, :] - xy[None, :, :]
        dh = np.sqrt((diff * diff).sum(axis=2))               # horizontal (N,N)
        dv = np.abs(z[:, None] - z[None, :])                  # vertical   (N,N)

        conflict = (dh < self.min_sep) & (dv < self.min_vsep)
        np.fill_diagonal(conflict, False)

        iu = np.triu_indices(n, k=1)
        mask = conflict[iu]
        pi, pj = iu[0][mask], iu[1][mask]
        pairs = list(zip(pi.tolist(), pj.tolist()))
        if len(pairs):
            per_drone[pi] = True; per_drone[pj] = True
        total = n * (n - 1) / 2
        return {"pairs": pairs, "count": len(pairs),
                "rate": len(pairs) / max(total, 1), "per_drone": per_drone}

    def risk_score(self, positions: np.ndarray) -> float:
        n = len(positions)
        if n < 2:
            return 0.0
        xy = np.asarray(positions[:, :2], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        iu = np.triu_indices(n, k=1)
        d  = dist[iu]
        close = d < self.min_sep
        return float(((self.min_sep - d[close]) / self.min_sep).sum())
