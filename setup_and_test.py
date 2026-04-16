"""
setup_and_test.py
=================
Run this first to verify BlueSky + UTM integration works.

Usage:
    python setup_and_test.py
"""

import sys
import subprocess
import os 

ROOT    = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(ROOT, "env")
sys.path.insert(0, ROOT)
sys.path.insert(0, ENV_DIR)   # ← this is what was missing

print("=" * 55)
print("UTM BlueSky Integration — Setup & Test")
print("=" * 55)

# ── Step 1: Check BlueSky-Gym ──────────────────────────────────────────────
print("\n[1] Checking BlueSky-Gym installation...")
try:
    import bluesky as bs
    print("    ✅ BlueSky found")
    BLUESKY_OK = True
except ImportError:
    print("    ❌ BlueSky not found")
    print("\n    Install with:")
    print("      git clone https://github.com/TUDelft-CNS-ATM/bluesky-gym.git")
    print("      cd bluesky-gym")
    print("      pip install -r requirements.txt")
    print("      pip install -e .")
    BLUESKY_OK = False

# ── Step 2: Check core dependencies ───────────────────────────────────────
print("\n[2] Checking dependencies...")
deps = ["numpy", "torch", "gymnasium", "stable_baselines3"]
for dep in deps:
    try:
        __import__(dep)
        print(f"    ✅ {dep}")
    except ImportError:
        print(f"    ❌ {dep} — install with: pip install {dep}")

# ── Step 3: Test sector graph ──────────────────────────────────────────────
print("\n[3] Testing SectorGraph...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
    from sector_graph import SectorGraph
    g = SectorGraph()
    assert g.n_sectors == 9
    lat, lon = g.sector_center_latlon(4)
    assert abs(lat - 25.15) < 0.01
    assert abs(lon - 55.15) < 0.01
    print(f"    ✅ 3×3 grid created — sector 4 center: ({lat:.2f}N, {lon:.2f}E)")
    print(f"    Grid layout:\n")
    for line in str(g).split("\n"):
        print(f"        {line}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback
    traceback.print_exc()

# ── Step 4: Test conflict detector ────────────────────────────────────────
print("\n[4] Testing ConflictDetector...")
try:
    import numpy as np
    from conflict_detector import ConflictDetector
    det = ConflictDetector(min_separation=500)
    pos = np.array([[0,0,300],[100,0,300],[10000,0,300]], dtype=np.float32)
    result = det.check(pos)
    assert result["count"] == 1
    print(f"    ✅ Detected {result['count']} conflict (drones 0 and 1 within 500m)")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback
    traceback.print_exc()

# ── Step 5: Test BlueSky UTM env ───────────────────────────────────────────
print("\n[5] Testing BlueSkyUTMEnv (headless)...")
try:
    from bluesky_utm_env import BlueSkyUTMEnv
    env = BlueSkyUTMEnv(n_drones=5, render_mode="none", max_steps=10)
    obs, info = env.reset()
    print(f"    ✅ Environment created")
    print(f"    Obs shape: {obs.shape}")

    # Run 5 random steps
    total_reward = 0.0
    for step in range(5):
        actions = [env.action_space.sample() for _ in range(5)]
        all_obs, rewards, dones, term, trunc, info = env.step_joint(actions)
        total_reward += rewards.mean()

    print(f"    ✅ 5 steps completed — total reward: {total_reward:.3f}")
    print(f"    Info: {info}")
except Exception as e:
    print(f"    ❌ {e}")
    import traceback; 
    traceback.print_exc()

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if BLUESKY_OK:
    print("✅ Ready! Run training with:")
    print()
    print("  # Headless (fast training):")
    print("  python training/train_mappo_bluesky.py")
    print()
    print("  # With BlueSky GUI (shows simulation):")
    print("  python training/train_mappo_bluesky.py --render")
    print()
    print("  # Quick test run:")
    print("  python training/train_mappo_bluesky.py --timesteps 5000")
    print()
    print("  # TensorBoard:")
    print("  tensorboard --logdir logs_bluesky/")
else:
    print("⚠️  BlueSky not installed — install it first (see Step 1 above)")
    print()
    print("  After installing BlueSky, the env will run with real simulation.")
    print("  Without it, the env falls back to mock movement.")
    print()
    print("  You can still run training in mock mode:")
    print("  python training/train_mappo_bluesky.py")
print("=" * 55)
