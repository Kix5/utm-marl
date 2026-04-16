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
# Protect argparse from BlueSky's argument hijacking
_argv_backup = sys.argv[:]
sys.argv = sys.argv[:1]

try:
    import bluesky as bs
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False
    print("[WARNING] BlueSky not found — install with: pip install bluesky-gym")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sector_graph import SectorGraph
from conflict_detector import ConflictDetector

def get_config(scale: int = 5):
    """Returns grid config for a given number of drones."""
    if scale <= 5:
        rows, cols = 3, 3
        caps = {0:2,1:3,2:2,3:3,4:1,5:3,6:2,7:3,8:2}
        od = [(0,8),(2,6),(0,2),(6,2),(3,5)]
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
    else:
        rows, cols = 7, 7
        n = rows * cols
        caps = {}
        for s in range(n):
            r, c = divmod(s, cols)
            is_corner = r in [0,rows-1] and c in [0,cols-1]
            is_center = abs(r-rows//2) <= 1 and abs(c-cols//2) <= 1
            if is_corner:   caps[s] = 2
            elif is_center: caps[s] = 2
            else:           caps[s] = 4
        # 50 diverse O-D pairs
        import numpy as np
        rng = np.random.RandomState(42)
        od = []
        for _ in range(50):
            while True:
                o = rng.randint(0, n)
                d = rng.randint(0, n)
                or_, oc = divmod(o, cols)
                dr_, dc = divmod(d, cols)
                if abs(or_-dr_) + abs(oc-dc) >= 3:
                    od.append((o, d))
                    break
    return rows, cols, caps, od



ORIGIN_LAT  = 25.0
ORIGIN_LON  = 55.0
SECTOR_DEG  = 0.1
CRUISE_ALT  = 300
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
        rows, cols, caps, od = get_config(scale)
        global GRID_ROWS, GRID_COLS
        GRID_ROWS = rows
        GRID_COLS = cols
        self.n_drones    = n_drones
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.n_sectors   = GRID_ROWS * GRID_COLS
        self.graph       = SectorGraph()
        self.detector    = ConflictDetector(min_separation=500)
        self.CAPACITIES  = caps
        self.OD_PAIRS    = od[:n_drones]
        self.NO_FLY      = []

        
        self.agent_obs_dim = self.n_sectors + 3 + 1 + 1 + (n_drones - 1) * 3
        self.observation_space = spaces.Box(-1.0, 5.0, shape=(self.agent_obs_dim,), dtype=np.float32)
        self.action_space      = spaces.Discrete(self.n_sectors)

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

        self._stack("DELETEALL")
        self._step_sim(1)

        for i, (origin, goal) in enumerate(self.OD_PAIRS[:self.n_drones]):
            self._drone_sectors[i] = origin
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
            tgt = int(np.clip(tgt, 0, self.n_sectors-1))
            if tgt in self.NO_FLY:
                continue
            self._drone_sectors[i] = tgt
            lat, lon = sector_to_latlon(tgt)
            self._stack(f"MOVE {self._callsigns[i]} {lat:.4f} {lon:.4f} {CRUISE_ALT}")

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
        cinfo = self.detector.check(pos_m)
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
        others   = np.concatenate([
            np.array([(self._positions[j,0]-ORIGIN_LAT)/sp_lat,
                      (self._positions[j,1]-ORIGIN_LON)/sp_lon,
                      self._positions[j,2]/1000.0])
            for j in range(self.n_drones) if j!=idx
        ]).astype(np.float32)
        return np.concatenate([loads, own_pos, own_sec, own_goal, others])

    def get_all_agent_obs(self):
        return np.stack([self.get_agent_obs(i) for i in range(self.n_drones)])

    def render(self):
        pass

    def close(self):
        self._stack("QUIT")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _mock_move(self):
        for i in range(self.n_drones):
            if self._drone_done[i]:
                continue
            tlat, tlon = sector_to_latlon(self._drone_sectors[i])
            # Move 40% toward target per step instead of 10%
            self._positions[i,0] += (tlat - self._positions[i,0]) * 0.5
            self._positions[i,1] += (tlon - self._positions[i,1]) * 0.5

    def _compute_loads(self):
        loads = np.zeros(self.n_sectors, dtype=np.float32)
        for i in range(self.n_drones):
            if not self._drone_done[i]:
                loads[latlon_to_sector(self._positions[i,0], self._positions[i,1])] += 1.0
        return loads

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
