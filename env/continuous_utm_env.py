"""
continuous_utm_env.py
=====================
Continuous-control UAV traffic environment.

Action per drone = [d_heading, d_speed, vertical_rate]  (continuous, Box).
Movement is KINEMATIC: each drone integrates its velocity, so it can make
fine steering adjustments to avoid neighbours while progressing to its goal.
This removes the discrete-action ceiling of the sector-picking env.

Reuses the sector graph only for capacity/goal bookkeeping; movement itself
is continuous in (lat, lon, alt).
"""

import os, math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from bluesky_utm_env import get_config, ORIGIN_LAT, ORIGIN_LON, SECTOR_DEG, CRUISE_ALT
except ImportError:
    from env.bluesky_utm_env import get_config, ORIGIN_LAT, ORIGIN_LON, SECTOR_DEG, CRUISE_ALT

try:
    from conflict_detector import ConflictDetector
except ImportError:
    from env.conflict_detector import ConflictDetector

M_PER_DEG   = 111_320.0            # metres per degree (approx)
SEP_H       = 500.0               # horizontal separation minimum (m)
SEP_V       = 30.0                # vertical separation minimum (m)
# --- proactive (soft) separation shaping ---
# A drone starts paying a small, smoothly growing penalty as soon as its nearest
# neighbour enters the WARNING band (WARN_H), long before an actual conflict at
# SEP_H. This gives the policy a dense gradient to keep a margin, instead of only
# a sparse signal once separation is already lost.
WARN_H      = float(os.environ.get('UTM_WARN_RADIUS', '1500.0'))   # warning band outer radius (m)
W_WARN      = float(os.environ.get('UTM_WARN_PENALTY', '3.0'))     # max soft penalty per drone/step
DT          = 5.0                # seconds per step
# Speed is set in reset() relative to the airspace size so goals are always
# reachable within the episode, regardless of grid size.
V_NOM       = 400.0              # default; overridden per-scenario in reset()
V_MIN, V_MAX = 100.0, 900.0      # wide bounds (m/s) -- scaled airspace
DPSI_MAX    = math.radians(30)   # max heading change per step (rad)
VZ_MAX      = 5.0                # max vertical rate (m/s)
ALT_MIN, ALT_MAX = 60.0, 300.0   # altitude band (m)
GOAL_RADIUS = 2000.0             # arrival radius (m); relative to ~11km sectors
# --- reward weights (ablation-configurable; defaults identical to the thesis) ---
PROG_W    = float(os.environ.get('UTM_PROGRESS_W',   '0.001'))  # progress reward weight
ARR_BONUS = float(os.environ.get('UTM_ARRIVAL_BONUS','10.0'))   # arrival bonus


class ContinuousUTMEnv(gym.Env):
    def __init__(self, n_drones=100, render_mode="none", max_steps=200, scale=None):
        super().__init__()
        if scale is None:
            scale = n_drones
        rows, cols, caps, od, no_fly = get_config(scale)
        self.rows, self.cols = rows, cols
        self.n_sectors = rows * cols
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.CAPACITIES = caps
        self.NO_FLY = set(no_fly)
        self.OD_PAIRS = od[:n_drones]
        self._od_counter = 0          # episode counter for randomized-OD generalization mode
        self.detector = ConflictDetector(min_separation=SEP_H, min_vseparation=SEP_V)
        # per-sector capacity as an array, and a no-fly mask, for the constraint
        # terms in step_joint. caps is a dict {sector_id: capacity}; missing -> inf.
        self.cap_arr = np.array([caps.get(s, 10**6) for s in range(self.n_sectors)],
                                dtype=np.float64)
        self.nfz_mask = np.zeros(self.n_sectors, dtype=bool)
        for s in self.NO_FLY:
            if 0 <= s < self.n_sectors:
                self.nfz_mask[s] = True
        # geometry for mapping a drone (x,y) back to its sector
        _ref = ORIGIN_LAT + (self.rows / 2) * SECTOR_DEG
        self._sx = SECTOR_DEG * M_PER_DEG * math.cos(math.radians(_ref))
        self._sy = SECTOR_DEG * M_PER_DEG

        self.K = min(int(os.environ.get("UTM_KNN", "10")), max(n_drones - 1, 0))

        # --- CONTINUOUS ACTION: [d_heading, d_speed, vertical_rate], each in [-1,1] ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # observation per drone (fixed size)
        # [own: dx_goal, dy_goal, dz_goal, speed, heading_sin, heading_cos, alt] (7)
        #  + K nearest intruders * (rel_x, rel_y, rel_z) (K*3)
        #  + [nearest_dist_norm, conflict_flag] (2)
        self.agent_obs_dim = 7 + self.K * 3 + 2
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.agent_obs_dim,), dtype=np.float32)

    # ---- helpers ----
    def _sector_center_xy(self, sid):
        r, c = divmod(sid, self.cols)
        lat = ORIGIN_LAT + (r + 0.5) * SECTOR_DEG
        lon = ORIGIN_LON + (c + 0.5) * SECTOR_DEG
        return self._to_xy(lat, lon)

    def _to_xy(self, lat, lon):
        ref = ORIGIN_LAT + (self.rows / 2) * SECTOR_DEG
        x = (lon - ORIGIN_LON) * M_PER_DEG * math.cos(math.radians(ref))
        y = (lat - ORIGIN_LAT) * M_PER_DEG
        return x, y

    def _sector_of(self, xy):
        """Map drone (x,y) positions to sector ids. Returns (sid, inside_grid)."""
        c = np.floor(xy[:, 0] / self._sx).astype(int)
        r = np.floor(xy[:, 1] / self._sy).astype(int)
        inside = (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows)
        c = np.clip(c, 0, self.cols - 1); r = np.clip(r, 0, self.rows - 1)
        return (r * self.cols + c), inside

    def _generate_od(self, seed):
        """Generate a random set of origin-destination pairs for ONE episode,
        on the SAME fixed airspace (same grid and no-fly layout). Used for the
        generalization study: training sees a stream of random traffic layouts,
        evaluation uses held-out layouts the policy never trained on. Endpoints
        avoid no-fly sectors and are Manhattan-separated, matching get_config's
        rule, so a randomized scenario is drawn from the same distribution as the
        fixed one, only with different demands."""
        rng = np.random.RandomState(int(seed) % (2**31 - 1))
        n = self.rows * self.cols
        nf_set = set(self.NO_FLY)
        min_sep = max(3, (self.rows + self.cols) // 4)
        od, tries = [], 0
        while len(od) < self.n_drones and tries < self.n_drones * 400:
            tries += 1
            o = rng.randint(0, n); d = rng.randint(0, n)
            if o in nf_set or d in nf_set:
                continue
            orr, oc = divmod(o, self.cols); drr, dc = divmod(d, self.cols)
            if abs(orr - drr) + abs(oc - dc) >= min_sep:
                od.append((o, d))
        while len(od) < self.n_drones:        # relax separation if grid is small
            o = rng.randint(0, n); d = rng.randint(0, n)
            if o != d and o not in nf_set and d not in nf_set:
                od.append((o, d))
        return od

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # --- randomized-OD generalization mode (UTM_RANDOM_OD=1) ---
        # Draw a fresh traffic layout each episode. With an explicit seed, the
        # layout is reproducible (used for held-out evaluation); without one, it
        # advances a counter from UTM_OD_SEED_BASE (varied training layouts).
        if os.environ.get("UTM_RANDOM_OD", "0") == "1":
            if seed is not None:
                _od_seed = int(seed)
            else:
                _od_seed = int(os.environ.get("UTM_OD_SEED_BASE", "0")) + self._od_counter
                self._od_counter += 1
            self.OD_PAIRS = self._generate_od(_od_seed)
        self._step = 0
        self._ep_conflicts = self._ep_completed = 0
        self._ep_overloads = self._ep_nfz = 0
        self._done = np.zeros(self.n_drones, dtype=bool)
        # state: x, y (metres), alt, heading (rad), speed (m/s)
        self.pos = np.zeros((self.n_drones, 3))      # x, y, alt
        self.hdg = np.zeros(self.n_drones)
        # scale nominal speed so a drone can cross the whole grid in ~half the episode
        sector_m   = SECTOR_DEG * M_PER_DEG
        grid_diag  = sector_m * math.hypot(self.rows, self.cols)
        v_needed   = grid_diag / (0.5 * self.max_steps * DT)   # cross grid in half the steps
        self._v_nom = float(np.clip(v_needed, V_MIN, V_MAX))
        self.spd = np.full(self.n_drones, self._v_nom)
        self.goal = np.zeros((self.n_drones, 2))     # goal x, y
        self.target_alt = np.full(self.n_drones, float(CRUISE_ALT))
        # --- structured (layered) airspace ---
        # UTM_LAYERS=1 assigns each drone a cruise flight level from its heading
        # (a learned semicircular rule). Crossing traffic then sits on vertically
        # separated levels (>SEP_V apart), so a horizontal crossing is no longer a
        # 3D conflict. This uses the vertical dimension the flat policy ignores.
        self._layers = os.environ.get("UTM_LAYERS", "0") == "1"
        levels = np.arange(ALT_MIN, ALT_MAX + 1e-6, 40.0)   # 40m apart (>SEP_V=30) -> distinct FLs
        for i, (o, g) in enumerate(self.OD_PAIRS):
            ox, oy = self._sector_center_xy(o)
            gx, gy = self._sector_center_xy(g)
            h = math.atan2(gy - oy, gx - ox)                # heading toward goal
            self.goal[i] = [gx, gy]
            self.hdg[i] = h
            if self._layers:
                lvl = int(((h % (2*math.pi)) / (2*math.pi)) * len(levels)) % len(levels)
                a0 = float(levels[lvl])
            else:
                a0 = float(CRUISE_ALT)
            self.target_alt[i] = a0
            self.pos[i] = [ox, oy, a0]
        return self.get_all_agent_obs(), {}

    def _apply_shield(self, active):
        """Reactive separation-assurance filter (velocity-obstacle style).

        For each active drone it predicts the next-step position from the
        commanded heading/speed, and for any pair predicted to violate the 3D
        minima it (1) steers away from the intruder, (2) brakes to cut the
        closing rate, and (3) splits the pair vertically. Iterated a few times so
        dense tangles settle. Runs entirely on kinematic state, so it overrides
        whatever the policy asked for -- the guarantee does not depend on the
        network. Tunable via UTM_SHIELD_* env vars.
        """
        idx = np.where(active)[0]
        if len(idx) < 2:
            return
        iters  = int(os.environ.get("UTM_SHIELD_ITERS", "8"))
        margin = float(os.environ.get("UTM_SHIELD_MARGIN", "1.3"))   # target = margin * minimum
        rep_k  = float(os.environ.get("UTM_SHIELD_REP", "1.0"))       # repulsion strength
        brake  = float(os.environ.get("UTM_SHIELD_BRAKE", "0.4"))     # speed multiplier when in conflict
        vmin   = float(os.environ.get("UTM_SHIELD_VMIN", "0.0"))      # floor speed (0 = may hover)
        SEPH = SEP_H * margin
        SEPV = SEP_V * margin
        A = len(idx)
        order = np.arange(A)
        for _ in range(iters):
            hdg = self.hdg[idx]; spd = self.spd[idx]
            vx = np.cos(hdg) * spd; vy = np.sin(hdg) * spd
            px = self.pos[idx, 0] + vx * DT
            py = self.pos[idx, 1] + vy * DT
            pz = self.pos[idx, 2]
            dx = px[:, None] - px[None, :]
            dy = py[:, None] - py[None, :]
            dz = pz[:, None] - pz[None, :]
            dh = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dh, np.inf)
            conflict = (dh < SEPH) & (np.abs(dz) < SEPV)
            any_conf = conflict.any(1)
            if not any_conf.any():
                break
            w = np.where(conflict, np.clip((SEPH - dh) / SEPH, 0.0, 1.0), 0.0)
            inv = 1.0 / np.maximum(dh, 1e-6)
            rep_x = (w * dx * inv).sum(1)      # dx = xi - xj -> points away from neighbour
            rep_y = (w * dy * inv).sum(1)
            nx = np.cos(hdg) + rep_k * rep_x
            ny = np.sin(hdg) + rep_k * rep_y
            nrm = np.hypot(nx, ny)
            ok = (nrm > 1e-6) & any_conf
            new_hdg = hdg.copy()
            new_hdg[ok] = np.arctan2(ny[ok], nx[ok])
            self.hdg[idx] = new_hdg
            self.spd[idx] = np.where(any_conf, np.clip(spd * brake, vmin, V_MAX), spd)
            # vertical resolution: the higher drone of each conflicting pair climbs,
            # the lower descends; exact ties broken by index so the pair always
            # splits to opposite levels (the earlier bug split by parity, which
            # could send both the same way).
            zsign = np.sign(dz)
            tie = np.abs(dz) < 1.0
            zsign = np.where(tie, np.sign(order[:, None] - order[None, :]), zsign)
            vdir = np.sign(np.where(conflict, zsign, 0.0).sum(1))
            self.pos[idx, 2] = np.clip(np.where(any_conf, pz + vdir * VZ_MAX * DT, pz),
                                       ALT_MIN, ALT_MAX)

    def step_joint(self, actions):
        self._step += 1
        a = np.asarray(actions, dtype=np.float32).reshape(self.n_drones, 3)
        active = ~self._done

        # apply continuous controls
        dpsi = a[:, 0] * DPSI_MAX
        self.hdg[active] += dpsi[active]
        self.spd[active] = np.clip(self.spd[active] + a[active, 1] * (0.1*self._v_nom), V_MIN, V_MAX)
        self.pos[active, 2] = np.clip(self.pos[active, 2] + a[active, 2] * VZ_MAX * DT, ALT_MIN, ALT_MAX)

        # --- reactive safety shield (separation assurance layer) ---
        # Overrides the policy just before motion to enforce separation. This is
        # what turns a low-conflict LEARNED policy into a UTM with tactical
        # separation assurance: no matter what the policy commands, the shield
        # steers, brakes and vertically splits any pair predicted to lose
        # separation on the next step.
        if os.environ.get("UTM_SHIELD", "0") == "1":
            self._apply_shield(active)

        # integrate horizontal motion
        self.pos[active, 0] += np.cos(self.hdg[active]) * self.spd[active] * DT
        self.pos[active, 1] += np.sin(self.hdg[active]) * self.spd[active] * DT

        # completion: within GOAL_RADIUS of goal (horizontal)
        dg = self.pos[:, :2] - self.goal
        dist_goal = np.linalg.norm(dg, axis=1)
        newly = active & (dist_goal < GOAL_RADIUS)
        self._done[newly] = True
        self._ep_completed += int(newly.sum())

        # conflicts (3D) among ACTIVE drones
        pos3 = self.pos.copy()
        cinfo = self.detector.check3d(pos3[~self._done]) if (~self._done).sum() > 1 else {"count":0,"per_drone":np.zeros(0,bool)}
        n_conf = cinfo["count"]
        self._ep_conflicts += n_conf

        # per-agent reward: progress toward goal - conflict penalty
        prev_dist = getattr(self, "_prev_dist", dist_goal)
        progress = (prev_dist - dist_goal)                      # positive if moving toward goal
        self._prev_dist = dist_goal
        rewards = np.zeros(self.n_drones, dtype=np.float32)
        rewards[active] += progress[active] * PROG_W            # scaled progress
        rewards[newly] += ARR_BONUS                                  # arrival bonus
        # conflict penalty spread to involved drones
        overloads = np.zeros(self.n_drones, dtype=np.float32)
        conflicts_pa = np.zeros(self.n_drones, dtype=np.float32)
        if n_conf > 0 and len(cinfo["per_drone"]):
            # map back to global indices
            act_idx = np.where(~self._done)[0]
            pd = cinfo["per_drone"]
            conflicts_pa[act_idx[:len(pd)]] = pd.astype(np.float32)
        rewards -= conflicts_pa * float(os.environ.get('UTM_CONF_PENALTY', '2.0'))

        # --- proactive soft-separation penalty (dense shaping) ---
        # For each active drone, ramp a penalty from 0 at WARN_H up to W_WARN at
        # SEP_H, based on its single nearest active neighbour (horizontal). This
        # is bounded per drone (no blow-up in dense clusters) and gives the policy
        # a smooth gradient to open up spacing BEFORE a conflict occurs.
        act_i = np.where(active)[0]
        if len(act_i) > 1:
            P = self.pos[act_i, :2]
            d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn_d = np.sqrt(d2.min(axis=1))
            ramp = np.clip((WARN_H - nn_d) / (WARN_H - SEP_H), 0.0, 1.0)
            rewards[act_i] -= ramp.astype(np.float32) * W_WARN

        # --- altitude-hold shaping (only in layered mode) ---
        # Mild penalty for drifting off the assigned flight level, so the policy
        # keeps the vertical separation that layering sets up.
        if getattr(self, "_layers", False):
            alt_dev = np.abs(self.pos[:, 2] - self.target_alt) / 100.0
            rewards[active] -= alt_dev[active].astype(np.float32) * float(os.environ.get('UTM_ALT_PENALTY', '1.0'))

        # --- sector constraints: capacity + no-fly zones ---
        # Counts are ALWAYS computed (so they can be reported as metrics even when
        # unpenalised); penalties apply only when their env var is > 0, so existing
        # runs are unchanged until you switch them on for training.
        sid, _inside = self._sector_of(self.pos[:, :2])
        loads = np.bincount(sid[active], minlength=self.n_sectors)   # active drones per sector
        over_sector = np.maximum(loads - self.cap_arr, 0.0)          # overload per sector
        drone_over = (over_sector[sid] > 0) & active                # drones sitting over capacity
        in_nfz = self.nfz_mask[sid] & active                        # drones inside a no-fly sector
        n_over = int(drone_over.sum()); n_nfz = int(in_nfz.sum())
        self._ep_overloads += n_over
        self._ep_nfz += n_nfz
        cap_pen = float(os.environ.get('UTM_CAP_PENALTY', '0.0'))
        nfz_pen = float(os.environ.get('UTM_NFZ_PENALTY', '0.0'))
        if cap_pen > 0:
            # penalty scales with how far over capacity the drone's sector is
            rewards[drone_over] -= cap_pen * over_sector[sid][drone_over].astype(np.float32)
        if nfz_pen > 0:
            rewards[in_nfz] -= nfz_pen
        overloads = drone_over.astype(np.float32)

        terminated = bool(self._done.all())
        truncated = self._step >= self.max_steps
        info = {"step": self._step, "ep_conflicts": self._ep_conflicts,
                "ep_completed": self._ep_completed,
                "ep_overloads": self._ep_overloads, "ep_nfz": self._ep_nfz,
                "conflict_count": n_conf, "overload_count": n_over, "nfz_count": n_nfz,
                "conflicts_per_agent": conflicts_pa, "overloads_per_agent": overloads}
        return self.get_all_agent_obs(), rewards, self._done.copy(), terminated, truncated, info

    def get_all_agent_obs(self):
        N, K = self.n_drones, self.K
        obs = np.zeros((N, self.agent_obs_dim), dtype=np.float32)
        dg = self.goal - self.pos[:, :2]
        obs[:, 0] = dg[:, 0] / 50000.0
        obs[:, 1] = dg[:, 1] / 50000.0
        obs[:, 2] = (CRUISE_ALT - self.pos[:, 2]) / 300.0
        obs[:, 3] = (self.spd - self._v_nom) / (V_MAX - V_MIN)
        obs[:, 4] = np.sin(self.hdg)
        obs[:, 5] = np.cos(self.hdg)
        obs[:, 6] = self.pos[:, 2] / 300.0
        if K > 0 and N > 1:
            xy = self.pos[:, :2]
            diff = xy[:, None, :] - xy[None, :, :]
            d2 = (diff * diff).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.argpartition(d2, kth=min(K, N-1)-1, axis=1)[:, :K]
            rel = (self.pos[nn] - self.pos[:, None, :])         # (N,K,3)
            gathered = np.take_along_axis(d2, nn, axis=1)
            valid = np.isfinite(gathered)[:, :, None]
            rel = np.where(valid, rel / 5000.0, 0.0)
            obs[:, 7:7+K*3] = rel.reshape(N, K*3)
            nn_d = np.sqrt(np.where(np.isfinite(gathered[:,0]), gathered[:,0], (2*SEP_H)**2))
            obs[:, -2] = np.clip(nn_d / (2*SEP_H), 0, 1)
            obs[:, -1] = (nn_d < SEP_H).astype(np.float32)
        else:
            obs[:, -2] = 1.0
        return obs

    # ---- graph observation (for the ST-GNN policy) ----
    def get_node_features(self):
        """Per-drone own-state node features (N,7): the ego info the GNN starts
        from before message passing fills in neighbour context."""
        N = self.n_drones
        f = np.zeros((N, 7), dtype=np.float32)
        dg = self.goal - self.pos[:, :2]
        f[:, 0] = dg[:, 0] / 50000.0
        f[:, 1] = dg[:, 1] / 50000.0
        f[:, 2] = (CRUISE_ALT - self.pos[:, 2]) / 300.0
        f[:, 3] = (self.spd - self._v_nom) / (V_MAX - V_MIN)
        f[:, 4] = np.sin(self.hdg)
        f[:, 5] = np.cos(self.hdg)
        f[:, 6] = self.pos[:, 2] / 300.0
        return f

    def get_graph(self, k=None):
        """Build the proximity graph for the current state.

        Returns (node_feats (N,7), edge_index (2,E) int64, edge_attr (E,4)).
        Edges connect each ACTIVE drone to its k nearest ACTIVE neighbours in 3D
        (a done drone has left the airspace, so it neither sends nor receives).
        Edge features carry the 3D relative geometry: [dx,dy,dz, distance].
        Directed neighbour->ego, so ego aggregates messages from its neighbours.
        """
        N = self.n_drones
        if k is None:
            k = self.K
        nf = self.get_node_features()
        active = np.where(~self._done)[0]
        if len(active) < 2 or k < 1:
            return nf, np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
        P = self.pos[active]                                  # (A,3)
        diff = P[:, None, :] - P[None, :, :]                  # (A,A,3)
        d2 = (diff * diff).sum(-1)
        np.fill_diagonal(d2, np.inf)
        kk = min(k, len(active) - 1)
        nn = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]  # (A,kk) local indices
        ego_local = np.repeat(np.arange(len(active)), kk)     # (A*kk,)
        nbr_local = nn.reshape(-1)                            # (A*kk,)
        ego = active[ego_local]                               # global ego ids
        nbr = active[nbr_local]                               # global neighbour ids
        rel = self.pos[nbr] - self.pos[ego]                   # (E,3) neighbour relative to ego
        dist = np.linalg.norm(rel, axis=1, keepdims=True)
        edge_attr = np.concatenate(
            [rel[:, :2] / 5000.0, rel[:, 2:3] / 100.0, dist / 5000.0], axis=1).astype(np.float32)
        # message flows neighbour -> ego  (src=nbr, dst=ego)
        edge_index = np.stack([nbr, ego]).astype(np.int64)
        return nf, edge_index, edge_attr