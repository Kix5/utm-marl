"""
conflict_detector.py
====================
Pairwise separation checker — same as original PoC.
Works on XY meter coordinates (converted from lat/lon in env).
"""

import numpy as np
from typing import Dict

MIN_SEPARATION_M = 500  # 500m horizontal separation for UTM


class ConflictDetector:
    def __init__(self, min_separation: float = MIN_SEPARATION_M):
        self.min_sep = min_separation

    def check(self, positions: np.ndarray) -> Dict:
        n = len(positions)
        pairs, per_drone = [], np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < self.min_sep:
                    pairs.append((i, j))
                    per_drone[i] = per_drone[j] = True
        total = n*(n-1)/2
        return {
            "pairs":     pairs,
            "count":     len(pairs),
            "rate":      len(pairs) / max(total, 1),
            "per_drone": per_drone,
        }

    def risk_score(self, positions: np.ndarray) -> float:
        n = len(positions)
        score = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < self.min_sep:
                    score += (self.min_sep - d) / self.min_sep
        return score
