# UTM PoC — BlueSky Integration
## Multi-Agent UAV Traffic Management with BlueSky Simulator

This extends the original AirSim-based UTM PoC to use **BlueSky**
as the simulation backend — the standard open-source ATM/UTM simulator
used in published research papers. Works natively on **Apple Silicon Mac**.

---

## Architecture

```
BlueSky Simulator (physics + visualization)
         │
         │  traf.lat / traf.lon / traf.alt
         │  stack.stack("CRE / MOVE / DELETE")
         ▼
BlueSkyUTMEnv (Gymnasium wrapper)
  ├── SectorGraph    → maps lat/lon to 3×3 sector grid
  ├── ConflictDetector → pairwise UAV separation check
  └── Reward function  → throughput + safety + capacity
         │
         ▼
MAPPO Trainer (shared policy, parameter sharing)
  └── SharedActorCritic (256×256 MLP, same as original PoC)
```

---

## Airspace Layout (Dubai UTM scenario)

```
Sector 0 [cap:2] | Sector 1 [cap:3] | Sector 2 [cap:2]
  25.05N,55.05E  |   25.05N,55.15E  |   25.05N,55.25E
─────────────────┼──────────────────┼─────────────────
Sector 3 [cap:3] | Sector 4 [cap:1] | Sector 5 [cap:3]
  25.15N,55.05E  |   25.15N,55.15E  |   25.15N,55.25E
─────────────────┼──────────────────┼─────────────────
Sector 6 [cap:2] | Sector 7 [cap:3] | Sector 8 [cap:2]
  25.25N,55.05E  |   25.25N,55.15E  |   25.25N,55.25E
```

Each sector = ~11km × 11km  
Cruise altitude = 300 ft (low-altitude UTM)

---

## Installation

### Step 1 — Install BlueSky-Gym
```bash
# Due to pip bug (Feb 2025), install from source:
git clone https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
cd bluesky-gym
pip install -r requirements.txt
pip install -e .
cd ..
```

### Step 2 — Install other dependencies
```bash
pip install torch stable-baselines3 gymnasium tensorboard numpy
```

### Step 3 — Verify everything works
```bash
cd utm_bluesky
python setup_and_test.py
```

---

## Running

```bash
# Headless training (fast):
python training/train_mappo_bluesky.py

# With BlueSky GUI (see drones flying in real airspace):
python training/train_mappo_bluesky.py --render

# Quick proof-of-concept (5000 steps):
python training/train_mappo_bluesky.py --timesteps 5000

# Monitor training:
tensorboard --logdir logs_bluesky/
```

---

## What You'll See

When running with `--render`:
- BlueSky opens a 2D radar view showing the 3×3 sector grid
- 5 UAVs appear at their origin sectors
- As training progresses, UAVs navigate toward their destinations
- Conflicts (UAVs too close) are highlighted
- Sector overloads visible when capacity exceeded

---

## Connection to Original PoC

| Component | Original (AirSim) | This version (BlueSky) |
|---|---|---|
| Physics backend | AirSim / Mock | BlueSky real simulation |
| Coordinate system | AirSim XYZ meters | WGS84 lat/lon/alt |
| Multi-drone control | AirSim API | BlueSky stack commands |
| RL framework | Same MAPPO | Same MAPPO |
| Sector graph | Grid in meters | Grid in degrees |
| Conflict detection | 3D distance | Haversine + altitude |
| Visualization | AirSim 3D | BlueSky 2D radar |

The RL logic (MAPPO, reward function, observation space) is **identical**
to the original PoC — only the physics backend changed.

---

## Citation

```bibtex
@misc{bluesky-gym,
  author = {Groot, DJ and Leto, G and Vlaskin, A and Moec, A and Ellerbroek, J},
  title  = {BlueSky-Gym: Reinforcement Learning Environments for Air Traffic Applications},
  year   = {2024},
  journal = {SESAR Innovation Days 2024},
}
```
