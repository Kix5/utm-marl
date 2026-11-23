import numpy as np
from typing import List, Tuple

ORIGIN_LAT = 25.0
ORIGIN_LON = 55.0
SECTOR_DEG = 0.1
GRID_ROWS  = 5
GRID_COLS  = 5

SECTOR_CAPACITIES = {
    0:2,  1:3,  2:2,  3:3,  4:2,
    5:3,  6:4,  7:3,  8:4,  9:3,
    10:3, 11:4, 12:1, 13:4, 14:3,
    15:3, 16:4, 17:3, 18:4, 19:3,
    20:2, 21:3, 22:2, 23:3, 24:2
    }
NO_FLY_ZONES      = []


class SectorGraph:
    def __init__(self, rows=None, cols=None, capacities=None, no_fly=None):
        # Grid size now comes from the env (get_config) rather than being hardcoded,
        # so the graph and the environment can never disagree about the airspace.
        self.rows       = GRID_ROWS if rows is None else rows
        self.cols       = GRID_COLS if cols is None else cols
        self.n_sectors  = self.rows * self.cols
        self.no_fly     = set(NO_FLY_ZONES if no_fly is None else no_fly)
        base_caps       = SECTOR_CAPACITIES if capacities is None else capacities
        self.capacities = {s: base_caps.get(s, 2) for s in range(self.n_sectors)}
        for s in self.no_fly:
            self.capacities[s] = 0
        self._adj = {s: self._neighbours(s) for s in range(self.n_sectors)}

    def _neighbours(self, sid):
        row, col = divmod(sid, self.cols)
        nbrs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = row+dr, col+dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                nb = nr*self.cols + nc
                if nb not in self.no_fly:
                    nbrs.append(nb)
        return nbrs

    def neighbours(self, sid):
        return self._adj[sid]

    def capacity(self, sid):
        return self.capacities[sid]

    def is_no_fly(self, sid):
        return sid in self.no_fly

    def sector_center_latlon(self, sid):
        row, col = divmod(sid, self.cols)
        lat = ORIGIN_LAT + (row + 0.5) * SECTOR_DEG
        lon = ORIGIN_LON + (col + 0.5) * SECTOR_DEG
        return lat, lon

    def pos_to_sector(self, lat, lon):
        col = int(np.clip((lon - ORIGIN_LON) / SECTOR_DEG, 0, self.cols-1))
        row = int(np.clip((lat - ORIGIN_LAT) / SECTOR_DEG, 0, self.rows-1))
        return row * self.cols + col

    def shortest_path(self, src, dst):
        if src == dst:
            return 0
        visited, queue = {src}, [(src, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in self.neighbours(node):
                if nb == dst:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist+1))
        return 999

    def __repr__(self):
        lines = []
        for row in range(self.rows):
            parts = []
            for col in range(self.cols):
                sid = row*self.cols + col
                tag = "X" if self.is_no_fly(sid) else str(self.capacities[sid])
                parts.append(f"[{sid}:{tag}]")
            lines.append(" ".join(parts))
        return "\n".join(lines)