"""
compare_all.py — run from utm_bluesky/: python compare_all.py
Reads obs_dim and n_actions from checkpoint, builds matching env.
"""
import sys, os, numpy as np, json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "env"))
sys.path.insert(0, os.path.join(ROOT, "training"))

import torch, torch.nn as nn
from torch.distributions import Categorical

# ── Flexible policy ────────────────────────────────────────────────────────
class FlexPolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh())
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        t = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = self(t)
        return int(logits.argmax().item()) if deterministic \
               else int(Categorical(logits=logits).sample().item())

def load_policy(path):
    ckpt    = torch.load(path, map_location='cpu')
    state   = ckpt.get('policy', ckpt)
    obs_dim = state['shared.0.weight'].shape[1]
    n_act   = state['actor.weight'].shape[0]
    p = FlexPolicy(obs_dim, n_act)
    p.load_state_dict(state, strict=True)
    p.eval()
    return p, obs_dim, n_act

def infer_config(obs_dim, n_actions, n_drones):
    """
    Reverse-engineer env config from checkpoint dims.
    obs_dim = n_sectors + 3 + 1 + 1 + (n_drones-1)*3
    n_actions = n_sectors
    """
    n_sectors = n_actions
    # verify
    expected = n_sectors + 3 + 1 + 1 + (n_drones - 1) * 3
    if expected != obs_dim:
        # Try to find n_drones that matches
        for nd in range(1, 200):
            if n_sectors + 5 + (nd-1)*3 == obs_dim:
                n_drones = nd
                break
    import math
    grid = int(math.sqrt(n_sectors))
    return grid, n_drones

# ── Build env matching checkpoint dims ────────────────────────────────────
def make_matching_env(n_drones, n_sectors):
    """Build env with exactly the right grid for given n_sectors."""
    import math
    grid = int(math.sqrt(n_sectors))
    
    from bluesky_utm_env import BlueSkyUTMEnv, get_config
    import bluesky_utm_env as benv
    
    # Build capacities and OD pairs for this grid
    rows = cols = grid
    n = rows * cols
    
    caps = {}
    for s in range(n):
        r, c = divmod(s, cols)
        is_corner = r in [0,rows-1] and c in [0,cols-1]
        is_center = r == rows//2 and c == cols//2
        if is_corner:   caps[s] = 2
        elif is_center: caps[s] = 1
        else:           caps[s] = 3
    
    # Generate OD pairs
    import numpy as npx
    rng = npx.random.RandomState(42)
    od = []
    corners  = [0, grid-1, (grid-1)*grid, n-1]
    attempts = 0
    while len(od) < n_drones and attempts < 10000:
        o = int(rng.randint(0, n))
        d = int(rng.randint(0, n))
        or_, oc = divmod(o, cols)
        dr_, dc = divmod(d, cols)
        if o != d and abs(or_-dr_)+abs(oc-dc) >= 2:
            od.append((o, d))
        attempts += 1
    if len(od) < n_drones:
        # fallback: repeat pairs
        while len(od) < n_drones:
            od.append(od[len(od) % max(len(od),1)])
    
    # Patch globals
    benv.GRID_ROWS = rows
    benv.GRID_COLS = cols
    
    env = BlueSkyUTMEnv.__new__(BlueSkyUTMEnv)
    env.n_drones    = n_drones
    env.render_mode = "none"
    env.max_steps   = 300
    env.n_sectors   = n_sectors
    env.CAPACITIES  = caps
    env.OD_PAIRS    = od
    env.NO_FLY      = []
    env.graph       = type('G', (), {'n_sectors': n_sectors})()
    
    from conflict_detector import ConflictDetector
    env.detector = ConflictDetector(min_separation=500)
    
    env.agent_obs_dim = n_sectors + 3 + 1 + 1 + (n_drones-1)*3
    
    from gymnasium import spaces
    env.observation_space = spaces.Box(-1., 5., shape=(env.agent_obs_dim,))
    env.action_space      = spaces.Discrete(n_sectors)
    
    env._step_count    = 0
    env._callsigns     = [f"UAV{i:02d}" for i in range(n_drones)]
    env._drone_sectors = np.zeros(n_drones, dtype=int)
    env._drone_goals   = np.zeros(n_drones, dtype=int)
    env._drone_done    = np.zeros(n_drones, dtype=bool)
    env._positions     = np.zeros((n_drones, 3), dtype=np.float32)
    env._ep_conflicts  = env._ep_overloads = env._ep_completed = 0
    env._bs_ready      = False
    
    return env

# ── Run episodes ───────────────────────────────────────────────────────────
def run_episodes(env, policy, N, n_ep=10):
    from bluesky_utm_env import BlueSkyUTMEnv
    conflicts, throughput = [], []
    for _ in range(n_ep):
        env.reset()
        obs = env.get_all_agent_obs()
        done, step = False, 0
        while not done and step < 300:
            acts = ([env.action_space.sample() for _ in range(N)]
                    if policy is None else
                    [policy.act(obs[i]) for i in range(N)])
            obs, _, _, term, trunc, info = env.step_joint(acts)
            done = term or trunc; step += 1
        conflicts.append(info['ep_conflicts'])
        throughput.append(info['ep_completed'] / N)
    return float(np.mean(throughput)), float(np.mean(conflicts))

# ── Main ───────────────────────────────────────────────────────────────────
SCALES = [
    (5,  'checkpoints_mappo_5/mappo_bluesky_final.pt',
          'checkpoints_constrained_5/constrained_final.pt'),
    (25, 'checkpoints_mappo_25/mappo_bluesky_final.pt',
          'checkpoints_constrained_25/constrained_final.pt'),
    (50, 'checkpoints_mappo_50/mappo_bluesky_final.pt',
          'checkpoints_constrained_50/constrained_final.pt'),
]

print('\n' + '='*72)
print('Full Scalability Comparison — MAPPO vs Constrained MARL')
print('='*72)
print(f'{"Drones":<8}{"Grid":<7}{"Method":<24}{"Throughput":>11}{"Conflicts":>10}')
print('-'*72)

all_results = []

for scale, mckpt, cckpt in SCALES:
    try:
        # Load policies and read their exact dims
        vanilla,     od_v, na_v = load_policy(mckpt)
        constrained, od_c, na_c = load_policy(cckpt)

        # Both should agree on obs_dim (trained on same env)
        assert od_v == od_c, f"obs mismatch: {od_v} vs {od_c}"
        assert na_v == na_c, f"act mismatch: {na_v} vs {na_c}"

        n_sectors = na_v
        import math
        grid = int(math.sqrt(n_sectors))

        print(f'\nScale {scale} | {grid}x{grid} grid | '
              f'{n_sectors} sectors | obs={od_v}')

        # Build env matching these exact dims
        env = make_matching_env(scale, n_sectors)
        assert env.agent_obs_dim == od_v, \
            f"env obs {env.agent_obs_dim} != ckpt {od_v}"

        rt, rc = run_episodes(env, None,        scale)
        vt, vc = run_episodes(env, vanilla,     scale)
        ct, cc = run_episodes(env, constrained, scale)

        print(f'{scale:<8}{grid}x{grid:<4}{"Random":<24}{rt:>10.1%}{rc:>10.1f}')
        print(f'{"":8}{"":7}{"Vanilla MAPPO":<24}{vt:>10.1%}{vc:>10.1f}')
        print(f'{"":8}{"":7}{"Constrained MARL":<24}{ct:>10.1%}{cc:>10.1f}')
        print('-'*72)

        all_results.append({"scale":scale,"grid":f"{grid}x{grid}",
            "random":{"throughput":rt,"conflicts":rc},
            "mappo": {"throughput":vt,"conflicts":vc},
            "constrained":{"throughput":ct,"conflicts":cc}})

    except Exception as e:
        print(f'Scale {scale}: ERROR — {e}')
        import traceback; traceback.print_exc()
        print('-'*72)

print('='*72)
os.makedirs('results', exist_ok=True)
with open('results/final_comparison.json','w') as f:
    json.dump(all_results, f, indent=2)
print('\nSaved: results/final_comparison.json')