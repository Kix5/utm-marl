"""Evaluate the BEST (conflict-gated) checkpoint.

Reads UTM_SCALE and UTM_STEPS from the environment so the eval episode length
MATCHES training. Run it exactly like your training command, e.g.:

    UTM_SCALE=500 UTM_STEPS=400 python -u eval_best.py
"""
import sys, os, numpy as np, torch
sys.path.insert(0, "env"); sys.path.insert(0, "training")
try:
    from continuous_utm_env import ContinuousUTMEnv
except ImportError:
    from env.continuous_utm_env import ContinuousUTMEnv
try:
    from train_continuous import ContinuousActorCritic
except ImportError:
    from training.train_continuous import ContinuousActorCritic

SCALE = int(os.environ.get("UTM_SCALE", "500"))
STEPS = int(os.environ.get("UTM_STEPS", "400"))   # MUST match training episode length
env = ContinuousUTMEnv(n_drones=SCALE, scale=SCALE, max_steps=STEPS)
pol = ContinuousActorCritic(env.agent_obs_dim)

ckpt_path = f"checkpoints_continuous_{SCALE}/continuous_best.pt"
ck = torch.load(ckpt_path, map_location="cpu")
pol.load_state_dict(ck["policy"]); pol.eval()

meta = {k: v for k, v in ck.items() if k != "policy"}
print(f"Loaded {ckpt_path}")
print(f"  checkpoint meta: {meta}")
print(f"  eval: scale={SCALE}  steps={STEPS}")

N_EVAL = 20   # more episodes -> see consistency
tps, confs = [], []
for _ in range(N_EVAL):
    o, _ = env.reset(); done = False; s = 0
    while not done and s < STEPS:
        a = pol.act(o, deterministic=True)      # deterministic = consistent
        o, r, dn, term, trunc, info = env.step_joint(a); done = term or trunc; s += 1
    tps.append(info["ep_completed"] / SCALE); confs.append(info["ep_conflicts"])

tps = np.array(tps); confs = np.array(confs)
print(f"\nBEST checkpoint over {N_EVAL} episodes:")
print(f"  throughput: mean={tps.mean():.1%}  min={tps.min():.1%}  max={tps.max():.1%}  std={tps.std():.1%}")
print(f"  conflicts:  mean={confs.mean():.1f}  min={confs.min():.0f}  max={confs.max():.0f}")
print(f"  consistency: {(tps > 0.9).sum()}/{N_EVAL} episodes above 90% throughput")
