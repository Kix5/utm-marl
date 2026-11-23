"""
bluesky_utm_env.py  (FIXED)
===========================
Key fix: BlueSky sim/traf are None until bs.init() runs.
Import them lazily inside methods AFTER initialization.
Falls back to mock movement gracefully if BlueSky fails.
"""


from typing import Optional, List, Tuple, Dict
import math
import sys , os 

import numpy as np
import gymnasium as gym
from gymnasium import spaces
# Protect argparse from BlueSky's argument hijacking.
# BlueSky parses sys.argv on import, which would swallow the trainer's own CLI
# flags (e.g. --updates). We hide argv for the duration of the bluesky import
# ONLY, then restore it so downstream argparse (train_continuous.py) sees the
# real command line. Previously argv was truncated and never restored, which is
# why --updates silently had no effect.
_argv_backup = sys.argv[:]
sys.argv = sys.argv[:1]
try:
    import bluesky as bs
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False
    print("[WARNING] BlueSky not found — install with: pip install bluesky-gym")
finally:
    sys.argv = _argv_backup   # restore real CLI args for the trainer's argparse

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
# Import robustly whether this module is loaded as 'bluesky_utm_env' (bare)
# or as 'env.bluesky_utm_env' (package). Bare-first avoids the package
# re-entering itself, which was the source of the circular import.
try:
    from sector_graph import SectorGraph
    from conflict_detector import ConflictDetector
except ImportError:
    from env.sector_graph import SectorGraph
    from env.conflict_detector import ConflictDetector

def get_config(scale: int = 5):
    """Returns (rows, cols, caps, od, no_fly) for a given number of drones.

    Grid scales with fleet so density stays near ~1 UAV/sector at large scale.
    Difficulty at large scale comes from SCATTERED NO-FLY REGIONS the fleet must
    route around, rather than tight per-sector capacities.
    """
    import numpy as np

    # -- small validated base cases (unchanged) ------------------------------
    if scale <= 5:
        rows, cols = 3, 3
        caps = {0:2,1:3,2:2,3:3,4:1,5:3,6:2,7:3,8:2}
        od = [(0,8),(2,6),(0,2),(6,2),(3,5)]
        no_fly = []
        return rows, cols, caps, od[:scale], no_fly

    elif scale <= 25:
        rows, cols = 5, 5
        n = rows * cols
        caps = {}
        for s in range(n):
            r, c = divmod(s, cols)
            is_corner = r in [0,rows-1] and c in [0,cols-1]
            is_center = r == rows//2 and c == cols//2
            if is_corner:    caps[s] = 2
            elif is_center:  caps[s] = 1
            else:            caps[s] = 3
        od = [(0,24),(4,20),(0,4),(20,24),(2,22),
              (10,14),(1,23),(3,21),(5,19),(6,18),
              (0,20),(4,24),(1,13),(11,23),(7,17),
              (0,12),(12,24),(3,11),(13,21),(8,16),
              (0,14),(10,24),(2,12),(12,22),(6,14)]
        no_fly = []
        return rows, cols, caps, od[:scale], no_fly

    elif scale <= 50:
        rows, cols = 7, 7            # 49 sectors, ~1/sector at 50 UAVs
    elif scale <= 100:
        rows, cols = 10, 10          # 100 sectors, ~1/sector at 100 UAVs
    else:
        rows, cols = 22, 22          # 484 sectors, ~1/sector at 500 UAVs

    # -- large-scale generator (50 / 100 / 500) ------------------------------
    n = rows * cols
    rng = np.random.RandomState(42)  # fixed seed -> reproducible scenarios

    # Scattered no-fly regions: a handful of small forbidden blocks.
    # Count scales gently with grid area; each block is 2x2 (or 1x1 on 7x7).
    n_zones   = max(2, (rows * cols) // 120)      # 7x7->2, 10x10->2, 22x22->4
    block     = 1 if rows < 10 else 2             # block edge length
    no_fly    = set()
    zone_seeds = []
    attempts = 0
    while len(zone_seeds) < n_zones and attempts < 200:
        attempts += 1
        r0 = rng.randint(1, rows - block - 1)
        c0 = rng.randint(1, cols - block - 1)
        # keep zones apart so they don't merge into one wall
        if any(abs(r0-zr) + abs(c0-zc) < 4 for zr, zc in zone_seeds):
            continue
        zone_seeds.append((r0, c0))
        for dr in range(block):
            for dc in range(block):
                no_fly.add((r0+dr)*cols + (c0+dc))
    no_fly = sorted(no_fly)

    # Capacities: uniform, roomy. Difficulty is the no-fly geometry, not caps.
    caps = {s: 3 for s in range(n)}
    for s in no_fly:
        caps[s] = 0

    # O-D pairs: spread across the grid, Manhattan-separated, endpoints not in
    # a no-fly sector. Count == number of drones (scale).
    nf_set = set(no_fly)
    min_sep = max(3, (rows + cols) // 4)   # require a real journey
    od = []
    tries = 0
    while len(od) < scale and tries < scale * 200:
        tries += 1
        o = rng.randint(0, n)
        d = rng.randint(0, n)
        if o in nf_set or d in nf_set:
            continue
        orr, oc = divmod(o, cols)
        drr, dc = divmod(d, cols)
        if abs(orr-drr) + abs(oc-dc) >= min_sep:
            od.append((o, d))
    return rows, cols, caps, od, no_fly



ORIGIN_LAT  = 25.0
ORIGIN_LON  = 55.0
SECTOR_DEG  = 0.1
CRUISE_ALT  = 300
N_LAYERS    = 5          # discrete altitude layers
LAYER_SEP_M = 60         # vertical spacing between layers (m); > vertical sep min
LAYER_ALTS  = [CRUISE_ALT + k*LAYER_SEP_M for k in range(N_LAYERS)]  # ft-ish units
DRONE_SPEED = 50
GRID_ROWS   = 3
GRID_COLS   = 3

def sector_to_latlon(sid):
    row, col = divmod(sid, GRID_COLS)
    return ORIGIN_LAT + (row+0.5)*SECTOR_DEG, ORIGIN_LON + (col+0.5)*SECTOR_DEG

def latlon_to_sector(lat, lon):
    col = int(np.clip((lon-ORIGIN_LON)/SECTOR_DEG, 0, GRID_COLS-1))
    row = int(np.clip((lat-ORIGIN_LAT)/SECTOR_DEG, 0, GRID_ROWS-1))
    return row*GRID_COLS + col


class BlueSkyUTMEnv(gym.Env):
    metadata   = {"render_modes": ["human","none"]}
    # 3*3 = 9 sectors
    CAPACITIES = {0:2, 1:3, 2:2, 3:3, 4:1, 5:3, 6:2, 7:3, 8:2}

    # 10 diverse O-D pairs crossing different parts of grid
    OD_PAIRS = [(0,8),(2,6),(0,2),(6,2),(3,5)] 
    NO_FLY     = []

    _SCALE     = int(os.environ.get("UTM_SCALE", "5"))
    N_DRONES   = _SCALE
    LOG_DIR    = f"./logs_mappo_{_SCALE}"
    CKPT_DIR   = f"./checkpoints_mappo_{_SCALE}"

    def __init__(self, n_drones=5, render_mode="none", max_steps=200, scale=None):
        super().__init__()
        if scale is None:
            scale = n_drones
        rows, cols, caps, od, no_fly = get_config(scale)
        global GRID_ROWS, GRID_COLS
        GRID_ROWS = rows
        GRID_COLS = cols
        self.n_drones    = n_drones
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.n_sectors   = GRID_ROWS * GRID_COLS
        self.graph       = SectorGraph(rows=rows, cols=cols, capacities=caps, no_fly=no_fly)
        self.detector    = ConflictDetector(min_separation=500)
        self.CAPACITIES  = caps
        self.OD_PAIRS    = od[:n_drones]
        self.NO_FLY      = list(no_fly)

        
        self.K_NEAREST     = int(os.environ.get("UTM_KNN", "10"))
        # k-nearest intruders -> FIXED observation size at every fleet scale.
        # If fewer than K other drones exist, remaining slots are zero-padded.
        k_slots            = min(self.K_NEAREST, max(n_drones - 1, 0))
        self.K_NEAREST     = k_slots
        self.WIN = int(os.environ.get("UTM_WIN", "3"))   # local sector window edge (odd)
        self.WIN_CELLS = self.WIN * self.WIN               # FIXED load-map size at any grid
        # +2 = [normalized distance to nearest intruder, conflict flag]
        # Gives the policy a DIRECT conflict signal to react to, instead of
        # having to infer risk from raw neighbor coordinates.
        self.agent_obs_dim = self.WIN_CELLS + 3 + 1 + 1 + self.K_NEAREST * 3 + 2
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(self.agent_obs_dim,), dtype=np.float32)
        self.N_LAYERS          = N_LAYERS
        # Action = sector * layer  (drone chooses WHERE horizontally AND which altitude)
        self.action_space      = spaces.Discrete(self.n_sectors * N_LAYERS)

        self._step_count    = 0
        self._callsigns     = [f"UAV{i:02d}" for i in range(n_drones)]
        self._drone_sectors = np.zeros(n_drones, dtype=int)
        self._drone_goals   = np.zeros(n_drones, dtype=int)
        self._drone_done    = np.zeros(n_drones, dtype=bool)
        self._positions     = np.zeros((n_drones,3), dtype=np.float32)
        self._ep_conflicts  = self._ep_overloads = self._ep_completed = 0
        self._bs_ready      = False

        self._start_bluesky()

    # ── BlueSky helpers ────────────────────────────────────────────────────

    def _start_bluesky(self):
        """Only init BlueSky when explicitly rendering. Skip for training."""
        if self.render_mode != "human":
            print("[UTMEnv] Training mode — using mock movement (no BlueSky)")
            return
        if not BLUESKY_AVAILABLE:
            print("[UTMEnv] BlueSky not installed")
            return
        try:
            bs.init(mode="sim", gui="pygame")
            import time
            time.sleep(1.0)
            self._bs_ready = True
            print("[UTMEnv] BlueSky ready")
        except Exception as e:
            print(f"[UTMEnv] BlueSky init failed: {e}")
    def _stack(self, cmd):
        if not self._bs_ready:
            return
        try:
            from bluesky import stack
            if stack is not None:
                stack.stack(cmd)
        except Exception:
            pass

    def _step_sim(self, n=10):
        if not self._bs_ready:
            return
        try:
            from bluesky import sim
            if sim is not None:
                for _ in range(n):
                    sim.step()
        except Exception:
            pass

    def _read_positions(self):
        if not self._bs_ready:
            return
        try:
            from bluesky import traf
            if traf is None:
                return
            for i, cs in enumerate(self._callsigns):
                if cs in traf.id:
                    idx = traf.id.index(cs)
                    self._positions[i] = [
                        float(traf.lat[idx]),
                        float(traf.lon[idx]),
                        float(traf.alt[idx]) * 3.28084
                    ]
        except Exception:
            pass

    # ── Gym API ────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count   = 0
        self._ep_conflicts = self._ep_overloads = self._ep_completed = 0
        self._drone_done   = np.zeros(self.n_drones, dtype=bool)
        self._drone_layers = np.zeros(self.n_drones, dtype=np.int64)

        self._stack("DELETEALL")
        self._step_sim(1)

        for i, (origin, goal) in enumerate(self.OD_PAIRS[:self.n_drones]):
            self._drone_sectors[i] = origin
            self._drone_layers[i]  = 0
            self._drone_goals[i]   = goal
            lat, lon               = sector_to_latlon(origin)
            self._positions[i]     = [lat, lon, CRUISE_ALT]
            self._stack(f"CRE {self._callsigns[i]} B744 {lat:.4f} {lon:.4f} 0 {CRUISE_ALT} {DRONE_SPEED}")

        self._step_sim(5)
        self._read_positions()
        return self.get_agent_obs(0), self._get_info()

    def step_joint(self, actions):
        self._step_count += 1
        for i, tgt in enumerate(actions):
            if self._drone_done[i]:
                continue
            a = int(tgt)
            sec = a % self.n_sectors          # horizontal sector
            lay = a // self.n_sectors         # altitude layer 0..N_LAYERS-1
            if sec in self.NO_FLY:
                continue
            self._drone_sectors[i] = sec
            self._drone_layers[i]  = lay
            self._positions[i,2]   = LAYER_ALTS[lay]
            lat, lon = sector_to_latlon(sec)
            self._stack(f"MOVE {self._callsigns[i]} {lat:.4f} {lon:.4f} {LAYER_ALTS[lay]}")

        self._step_sim(10)
        self._read_positions()
        if not self._bs_ready:
            self._mock_move()

        newly_done = 0
        for i in range(self.n_drones):
            if self._drone_done[i]:
                continue
            glat, glon = sector_to_latlon(self._drone_goals[i])
            if self._drone_sectors[i] == self._drone_goals[i]:
                self._drone_done[i] = True
                newly_done += 1
                self._ep_completed += 1

        loads    = self._compute_loads()
        overload = sum(max(0, loads[s]-self.CAPACITIES.get(s,2)) for s in range(self.n_sectors))
        self._ep_overloads += int(overload > 0)

        pos_m = self._to_meters(self._positions)
        cinfo = self.detector.check3d(pos_m)   # 3D: same layer + horizontally close
        self._ep_conflicts += cinfo["count"]

        rewards = np.full(self.n_drones, -0.01, dtype=np.float32)
        rewards += newly_done  * 5.0  / self.n_drones
        rewards -= cinfo["count"] * 2.0  / self.n_drones
        rewards -= overload    * 1.5  / self.n_drones

        terminated = bool(np.all(self._drone_done))
        truncated  = self._step_count >= self.max_steps
        return (self.get_all_agent_obs(), rewards, self._drone_done.copy(),
                terminated, truncated,
                self._get_info(conflict_count=cinfo["count"],
                               overload_count=int(overload),
                               newly_completed=newly_done))

    def step(self, action, agent_idx=0):
        actions = [action if i==agent_idx else self._drone_sectors[i]
                   for i in range(self.n_drones)]
        return self.step_joint(actions)

    def get_agent_obs(self, idx):
        loads    = self._compute_loads() / max(max(self.CAPACITIES.values()),1)
        sp_lat   = GRID_ROWS * SECTOR_DEG
        sp_lon   = GRID_COLS * SECTOR_DEG
        own_pos  = np.array([
            (self._positions[idx,0]-ORIGIN_LAT)/sp_lat,
            (self._positions[idx,1]-ORIGIN_LON)/sp_lon,
            self._positions[idx,2]/1000.0
        ], dtype=np.float32)
        own_sec  = np.array([self._drone_sectors[idx]/(self.n_sectors-1)], dtype=np.float32)
        own_goal = np.array([self._drone_goals[idx]  /(self.n_sectors-1)], dtype=np.float32)
        # --- k-nearest intruders (fixed-size, local observation) ---
        K = self.K_NEAREST
        if K > 0:
            oi = self._positions[idx, :2]
            # squared horizontal distance to every other drone
            d2 = np.full(self.n_drones, np.inf, dtype=np.float64)
            for j in range(self.n_drones):
                if j == idx:
                    continue
                dx = self._positions[j,0] - oi[0]
                dy = self._positions[j,1] - oi[1]
                d2[j] = dx*dx + dy*dy
            nearest = np.argsort(d2)[:K]
            slots = []
            for j in nearest:
                if np.isinf(d2[j]):        # fewer than K neighbors -> zero-pad
                    slots.extend([0.0, 0.0, 0.0])
                else:
                    slots.extend([(self._positions[j,0]-ORIGIN_LAT)/sp_lat,
                                  (self._positions[j,1]-ORIGIN_LON)/sp_lon,
                                  self._positions[j,2]/1000.0])
            others = np.array(slots, dtype=np.float32)
        else:
            others = np.zeros(0, dtype=np.float32)
        return np.concatenate([loads, own_pos, own_sec, own_goal, others])

    def get_all_agent_obs(self):
        """VECTORIZED: build all agents' observations at once.
        - sector loads via bincount (no loop)
        - k-nearest intruders via one pairwise distance matrix (no per-agent loop)
        """
        N = self.n_drones
        K = self.K_NEAREST
        sp_lat = GRID_ROWS * SECTOR_DEG
        sp_lon = GRID_COLS * SECTOR_DEG
        cap_max = max(max(self.CAPACITIES.values()), 1)

        # --- LOCAL sector-load window (fixed size, per agent) ---
        loads = self._compute_loads() / cap_max            # (n_sectors,)
        W = self.WIN; half = W // 2
        grid = loads.reshape(GRID_ROWS, GRID_COLS)         # (R,C) congestion map
        # pad so the window is defined at the edges
        gpad = np.pad(grid, ((half, half), (half, half)), mode='constant')
        arow = np.clip(self._drone_sectors // GRID_COLS, 0, GRID_ROWS - 1)
        acol = np.clip(self._drone_sectors %  GRID_COLS, 0, GRID_COLS - 1)
        loads_row = np.empty((N, W * W), dtype=np.float32)
        for a in range(N):
            r0, c0 = arow[a], acol[a]
            loads_row[a] = gpad[r0:r0+W, c0:c0+W].ravel()


        # --- own features (vectorized) ---
        own_pos = np.empty((N, 3), dtype=np.float32)
        own_pos[:, 0] = (self._positions[:, 0] - ORIGIN_LAT) / sp_lat
        own_pos[:, 1] = (self._positions[:, 1] - ORIGIN_LON) / sp_lon
        own_pos[:, 2] = self._positions[:, 2] / 1000.0
        own_sec  = (self._drone_sectors / max(self.n_sectors - 1, 1)).reshape(N, 1).astype(np.float32)
        own_goal = (self._drone_goals   / max(self.n_sectors - 1, 1)).reshape(N, 1).astype(np.float32)

        # normalized positions of ALL drones (for the "others" block)
        norm_xyz = own_pos  # same normalization used for neighbors

        if K > 0:
            xy = self._positions[:, :2].astype(np.float64)          # (N,2)
            diff = xy[:, None, :] - xy[None, :, :]                  # (N,N,2)
            d2 = (diff * diff).sum(axis=2)                          # (N,N) squared dist
            np.fill_diagonal(d2, np.inf)                           # ignore self
            # indices of K nearest others for every agent, in one call
            nn_idx = np.argpartition(d2, kth=min(K, N-1)-1, axis=1)[:, :K]   # (N,K)
            # gather neighbor normalized coords -> (N, K, 3) -> (N, K*3)
            neigh = norm_xyz[nn_idx]                                # (N,K,3)
            # zero-pad slots where a neighbor doesn't exist (d2==inf)
            gathered_d2 = np.take_along_axis(d2, nn_idx, axis=1)    # (N,K)
            valid = np.isfinite(gathered_d2)[:, :, None]           # (N,K,1)
            neigh = np.where(valid, neigh, 0.0)
            others = neigh.reshape(N, K * 3).astype(np.float32)
        else:
            others = np.zeros((N, 0), dtype=np.float32)

        # --- direct conflict signal: nearest-neighbor distance + conflict flag ---
        if K > 0:
            # nearest squared distance per agent (already have d2 & nn_idx above)
            nn_d2 = gathered_d2[:, 0]                       # closest neighbor, squared
            nn_d2 = np.where(np.isfinite(nn_d2), nn_d2, np.inf)
            nn_dist_m = np.sqrt(nn_d2) * 111000.0           # deg->meters (approx)
            SEP = self.detector.min_sep                     # 500 m
            # normalized distance (clipped to [0,1] at 2x separation), and a hard flag
            nn_norm = np.clip(nn_dist_m / (2.0 * SEP), 0.0, 1.0).astype(np.float32)
            conflict_flag = (nn_dist_m < SEP).astype(np.float32)
        else:
            nn_norm = np.ones(N, dtype=np.float32)
            conflict_flag = np.zeros(N, dtype=np.float32)
        conflict_sig = np.stack([nn_norm, conflict_flag], axis=1)   # (N,2)

        obs = np.concatenate([loads_row.astype(np.float32),
                              own_pos, own_sec, own_goal, others, conflict_sig], axis=1)
        return obs

    def render(self):
        pass

    def close(self):
        self._stack("QUIT")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _mock_move(self):
        # VECTORIZED: move all active drones 50% toward their target sector centers.
        active = ~self._drone_done
        if not active.any():
            return
        sec = self._drone_sectors[active]
        rows = sec // GRID_COLS
        cols = sec %  GRID_COLS
        tlat = ORIGIN_LAT + (rows + 0.5) * SECTOR_DEG
        tlon = ORIGIN_LON + (cols + 0.5) * SECTOR_DEG
        self._positions[active, 0] += (tlat - self._positions[active, 0]) * 0.5
        self._positions[active, 1] += (tlon - self._positions[active, 1]) * 0.5

    def _compute_loads(self):
        # VECTORIZED: sector index for every active drone, then bincount.
        active = ~self._drone_done
        if not active.any():
            return np.zeros(self.n_sectors, dtype=np.float32)
        lat = self._positions[active, 0]
        lon = self._positions[active, 1]
        col = np.clip(((lon - ORIGIN_LON) / SECTOR_DEG).astype(int), 0, GRID_COLS - 1)
        row = np.clip(((lat - ORIGIN_LAT) / SECTOR_DEG).astype(int), 0, GRID_ROWS - 1)
        sec = row * GRID_COLS + col
        return np.bincount(sec, minlength=self.n_sectors).astype(np.float32)

    def _to_meters(self, pos):
        ref = ORIGIN_LAT + (GRID_ROWS/2)*SECTOR_DEG
        out = np.zeros_like(pos)
        out[:,0] = (pos[:,0]-ref)*111320
        out[:,1] = (pos[:,1]-ORIGIN_LON)*111320*math.cos(math.radians(ref))
        out[:,2] = pos[:,2]*0.3048
        return out

    @staticmethod
    def _haversine(la1,lo1,la2,lo2):
        R=6371000; p1,p2=math.radians(la1),math.radians(la2)
        dp,dl=math.radians(la2-la1),math.radians(lo2-lo1)
        a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

    def _get_info(self, **kw):
        return {"step":self._step_count,"ep_conflicts":self._ep_conflicts,
                "ep_overloads":self._ep_overloads,"ep_completed":self._ep_completed,
                "drones_done":int(np.sum(self._drone_done)),
                "bluesky_ready":self._bs_ready,**kw}
