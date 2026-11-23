# UTM PoC — Multi-Agent UAV Traffic Management

Constrained Multi-Agent Reinforcement Learning for UAV Traffic Management,
built on the [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky-gym)
open-source ATM/UTM simulator. Works natively on Apple Silicon Mac.

**Main contribution:** Lagrangian-constrained MARL that achieves 100% throughput
with zero separation conflicts, versus ~2 conflicts/episode for vanilla MAPPO.

---

## Architecture

```
BlueSky Simulator (physics + visualization)
         │
         │  traf.lat / traf.lon / traf.alt
         │  stack.stack("CRE / MOVE / DELETE")
         ▼
BlueSkyUTMEnv (Gymnasium wrapper)              ContinuousUTMEnv (kinematic, no BlueSky)
  ├── SectorGraph    → 3×3 sector grid            ├── Continuous [d_heading, d_speed, v_rate]
  ├── ConflictDetector → pairwise separation       ├── Reactive safety shield
  └── Reward function                              └── Multi-layer altitude separation
         │                                                │
         ▼                                                ▼
MAPPO / Constrained MARL                        Continuous PPO (Gaussian policy)
  └── SharedActorCritic (256×256 MLP)              └── GraphActorCritic (ST-GNN)
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

Each sector ≈ 11 km × 11 km · Cruise altitude = 300 ft (low-altitude UTM)

---

## Installation

### Step 1 — Install BlueSky-Gym
```bash
git clone https://github.com/TUDelft-CNS-ATM/bluesky-gym.git
cd bluesky-gym
pip install -r requirements.txt
pip install -e .
cd ..
```

### Step 2 — Install dependencies
```bash
pip install torch stable-baselines3 gymnasium tensorboard numpy
```

### Step 3 — Verify setup
```bash
python setup_and_test.py
```

---

## Running the Simulation

### Step 1 — Verify installation
```bash
python setup_and_test.py
```
This runs the environment for 5 steps, testing the sector graph, conflict
detector, and env loop. It works even without BlueSky installed (mock mode).

### Step 2 — Quick simulation demo

Visualize 5 UAVs navigating the sector grid (uses matplotlib, no BlueSky
GUI needed):

```bash
# Random-policy demo (runs immediately, no training needed):
python visualize_flights.py --random

# Save as MP4 video:
python visualize_flights.py --random --save
```

### Step 3 — Train and visualize a learned policy

**Constrained MARL** (Lagrangian PPO — main contribution, ~2 min):
```bash
python training/train_constrained_marl.py
python visualize_flights.py --checkpoint checkpoints_constrained_5/constrained_final.pt
```

**Vanilla MAPPO** (discrete sector-based):
```bash
python training/train_mappo_bluesky.py
python visualize_flights.py --checkpoint checkpoints_bluesky/mappo_bluesky_final.pt
```

**With BlueSky GUI** (shows drones in real airspace radar view):
```bash
python training/train_mappo_bluesky.py --render
```

**Continuous control** (kinematic env, Gaussian policy):
```bash
UTM_SCALE=100 python training/train_continuous.py --updates 100
```

### 3D demo video (after training continuous controller)
```bash
python demo_video_3d.py --drones 100 --scale 100 --steps 180 \
    --ckpt checkpoints_continuous_100/continuous_best.pt --out demo3d.mp4
```

### Compare methods
```bash
python training/train_constrained_marl.py --compare    # constrained vs vanilla
python run_matched_comparison.py                        # full comparison suite
```

### Monitor training
```bash
tensorboard --logdir logs_bluesky/        # vanilla MAPPO
tensorboard --logdir logs_constrained/    # constrained MARL
```

---

## Evaluation & Analysis

### Full metrics suite
```bash
python eval_metrics.py
```
Reports: conflicts/episode, completion rate, path efficiency, sector overloads,
inference latency, and more.

### MILP baseline comparison
```bash
python milp_baseline.py
```

### Reward ablation study
```bash
python run_ablation.py
```

### Method comparison (MAPPO vs Constrained vs MILP)
```bash
python run_matched_comparison.py
```

### Generate publication figures
```bash
python generate_figures.py
```

---

## Generalization Study

The generalization experiment (`run_generalization.py`) evaluates whether the
ST-GNN policy generalizes to unseen traffic patterns better than the MLP.
This experiment **runs on Google Colab** (not locally):

```bash
# On Colab:
python run_generalization.py --scale 100 --seeds 0 1 2 \
    --updates 400 --steps 400 --layers 1 --n-test 20 --resume
```

---

## Project Structure

```
env/
  bluesky_utm_env.py        Gymnasium wrapper for BlueSky
  continuous_utm_env.py      Kinematic continuous-control env
  conflict_detector.py       Pairwise separation checking
  sector_graph.py            3×3 sector grid mapping

training/
  train_mappo_bluesky.py     Vanilla MAPPO trainer
  train_constrained_marl.py  Constrained MARL (Lagrangian PPO)
  train_continuous.py        Continuous-control PPO
  attention_policy.py        Attention-based policy network
  eval_best.py               Evaluate best checkpoint

visualize_flights.py         2D matplotlib animation of flights
visualize_run.py             3D GNN policy visualization (GIF)
demo_video_3d.py             3D rotating demo video (MP4)
eval_metrics.py              Full evaluation metrics suite
milp_baseline.py             MILP optimization baseline
compare_all.py               Cross-method comparison
run_matched_comparison.py    Matched comparison experiments
run_ablation.py              Reward ablation study
run_generalization.py        Generalization study (run on Colab)
generate_figures.py          Generate publication figures
setup_and_test.py            Installation verification
```

---

## Pre-trained Checkpoints

| Directory | Method | Scale |
|---|---|---|
| `checkpoints_bluesky/` | Vanilla MAPPO | 5 drones |
| `checkpoints_constrained_5/` | Constrained MARL | 5 drones |
| `checkpoints_constrained_25/` | Constrained MARL | 25 drones |
| `checkpoints_constrained_50/` | Constrained MARL | 50 drones |
| `checkpoints_constrained_100/` | Constrained MARL | 100 drones |
| `checkpoints_continuous_100/` | Continuous PPO | 100 drones |
| `checkpoints_continuous_500/` | Continuous PPO | 500 drones |
| `checkpoints_gnn_100/` | ST-GNN | 100 drones |
| `checkpoints_gnn_500/` | ST-GNN | 500 drones |

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
