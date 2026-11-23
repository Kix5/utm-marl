"""
visualize_run.py
================
Run the trained ST-GNN policy for one episode and animate the drone movement in
3D so the layered flight levels are visible: points = drones (colour = altitude),
red = a drone currently in conflict, green = goals (at their flight level), grey
translucent columns = no-fly sectors. Saves an animated GIF and shows it live if
a display is available.

Env vars (same as training/eval):
  UTM_SCALE, UTM_STEPS, UTM_LAYERS, UTM_SHIELD, UTM_CONF_PENALTY, UTM_WARN_*
Extra:
  UTM_VIEW   "3d" (default) or "2d"
  UTM_CKPT   checkpoint path (default: auto-find under checkpoints_gnn_<scale>)
  UTM_OUT    output gif path (default: utm_run.gif)
  UTM_FRAME_EVERY  record every Nth step (default 2)

Run:  UTM_SCALE=100 UTM_STEPS=400 UTM_LAYERS=1 UTM_SHIELD=1 \
      UTM_CKPT=checkpoints_gnn_100/gnn_final.pt python visualize_run.py
"""
import os, sys, glob
import numpy as np

for f in ["continuous_utm_env.py", "gnn_policy.py"]:
    for m in glob.glob("**/" + f, recursive=True):
        sys.path.insert(0, os.path.dirname(os.path.abspath(m)) or ".")
import torch
from continuous_utm_env import (ContinuousUTMEnv, M_PER_DEG, SECTOR_DEG,
                                ALT_MIN, ALT_MAX)
from gnn_policy import GraphActorCritic

import matplotlib
if not os.environ.get("DISPLAY") and sys.platform != "darwin":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

S = int(os.environ.get("UTM_SCALE", "100"))
T = int(os.environ.get("UTM_STEPS", "400"))
OUT = os.environ.get("UTM_OUT", "utm_run.gif")
FRAME_EVERY = int(os.environ.get("UTM_FRAME_EVERY", "2"))
VIEW = os.environ.get("UTM_VIEW", "3d").lower()

ckpt = os.environ.get("UTM_CKPT", "")
if not ckpt:
    pref = f"checkpoints_gnn_{S}"
    cands = (glob.glob(f"{pref}/gnn_best.pt") or glob.glob(f"{pref}/gnn_final.pt")
             or sorted(glob.glob(f"{pref}/gnn_*.pt"))
             or glob.glob("**/gnn_best.pt", recursive=True)
             or glob.glob("**/gnn_final.pt", recursive=True)
             or sorted(glob.glob("**/gnn_*.pt", recursive=True)))
    assert cands, "no gnn checkpoint found; set UTM_CKPT"
    ckpt = cands[0]
print("checkpoint:", ckpt, "| view:", VIEW, "| shield:", os.environ.get("UTM_SHIELD", "0"))

env = ContinuousUTMEnv(n_drones=S, scale=S, max_steps=T)
ck = torch.load(ckpt, map_location="cpu"); cfg = ck.get("cfg", {"hidden": 128, "layers": 2, "heads": 4})
pol = GraphActorCritic(7, 4, 3, cfg["hidden"], cfg["layers"], cfg["heads"]); pol.load_state_dict(ck["policy"]); pol.eval()
goal_alt = env.target_alt.copy() if hasattr(env, "target_alt") else np.full(S, (ALT_MIN + ALT_MAX) / 2)

# ---- roll out one episode ----
frames = []
o = env.reset()[0]
if hasattr(env, "target_alt"):
    goal_alt = env.target_alt.copy()
for step in range(T):
    nf, ei, ea = env.get_graph()
    a = pol.act(torch.as_tensor(nf, dtype=torch.float32),
                torch.as_tensor(ei, dtype=torch.long),
                torch.as_tensor(ea, dtype=torch.float32), deterministic=True).numpy()
    _, _, _, term, trunc, info = env.step_joint(a)
    active = ~env._done
    cflt = np.zeros(env.n_drones, dtype=bool)
    idx = np.where(active)[0]
    if len(idx) > 1:
        ci = env.detector.check3d(env.pos[idx]); pd = ci.get("per_drone", np.zeros(0, bool))
        if len(pd):
            cflt[idx[:len(pd)]] = pd.astype(bool)
    if step % FRAME_EVERY == 0 or term or trunc:
        frames.append((env.pos.copy(), active.copy(), cflt.copy(),
                       step + 1, int(info["conflict_count"]), int(info["ep_completed"])))
    if term or trunc:
        break
print(f"episode: {len(frames)} frames | final conflicts={info['ep_conflicts']} | "
      f"throughput={info['ep_completed']/S:.1%}")

allxy = np.vstack([env.goal, env.pos[:, :2]])
pad = 0.05 * (allxy.max(0) - allxy.min(0) + 1)
xmin, ymin = allxy.min(0) - pad; xmax, ymax = allxy.max(0) + pad
sm = SECTOR_DEG * M_PER_DEG
nofly = list(getattr(env, "NO_FLY", set()))


def nofly_boxes(ax, three_d):
    for sid in nofly:
        cx, cy = env._sector_center_xy(sid)
        if three_d:
            ax.bar3d(cx - sm/2, cy - sm/2, ALT_MIN, sm, sm, ALT_MAX - ALT_MIN,
                     color="0.6", alpha=0.10, shade=False, zorder=0)
        else:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((cx - sm/2, cy - sm/2), sm, sm, color="0.85", zorder=0))


if VIEW == "3d":
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(fi):
        pos, active, cflt, stp, nc, comp = frames[fi]
        ax.clear()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_zlim(ALT_MIN, ALT_MAX)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("altitude (m)")
        ax.set_title(f"UAV traffic (3D)   step {stp}/{T}   arrived {comp}/{S}   "
                     f"conflicts this step: {nc}")
        nofly_boxes(ax, True)
        # goals at their flight level
        ax.scatter(env.goal[:, 0], env.goal[:, 1], goal_alt, marker="^", s=10,
                   color="tab:green", alpha=0.35, depthshade=False)
        safe = active & ~cflt; bad = active & cflt
        P = pos[safe]
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="viridis",
                       vmin=ALT_MIN, vmax=ALT_MAX, s=16, depthshade=False)
        B = pos[bad]
        if len(B):
            ax.scatter(B[:, 0], B[:, 1], B[:, 2], color="red", s=45,
                       marker="o", depthshade=False)
        ax.view_init(elev=22, azim=(fi * 1.5) % 360)   # slow rotate
        return ()
else:
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(fi):
        pos, active, cflt, stp, nc, comp = frames[fi]
        ax.clear()
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"UAV traffic (2D)   step {stp}/{T}   arrived {comp}/{S}   "
                     f"conflicts this step: {nc}")
        nofly_boxes(ax, False)
        ax.scatter(env.goal[:, 0], env.goal[:, 1], marker="s", s=12, facecolors="none",
                   edgecolors="tab:green", alpha=0.4)
        safe = active & ~cflt; bad = active & cflt
        P = pos[safe]
        if len(P):
            ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="viridis",
                       vmin=ALT_MIN, vmax=ALT_MAX, s=16)
        B = pos[bad]
        if len(B):
            ax.scatter(B[:, 0], B[:, 1], color="red", s=60)
        return ()

anim = FuncAnimation(fig, update, frames=len(frames), interval=70, blit=False)
anim.save(OUT, writer=PillowWriter(fps=15))
print("saved animation ->", os.path.abspath(OUT))
try:
    plt.show()
except Exception:
    pass