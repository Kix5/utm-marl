"""
demo_video_3d.py  --  REAL 3D demo of the continuous UTM controller.

Renders a rotating 3D view (x, y, altitude) of one episode driven by your
TRAINED policy, with the reactive shield on, so the vertical separation that is
the point of the continuous-3D formulation is actually visible. Drops next to
continuous_utm_env.py.

    python demo_video_3d.py --drones 100 --scale 100 --steps 180 --seed 7 \
        --ckpt checkpoints_continuous_100/continuous_best.pt --out demo3d.mp4 --fps 20

--split renders two 3D panels (shield OFF vs ON). --noshield shows policy alone.
Falls back to a greedy straight-to-goal controller only if torch/ckpt are missing.
"""
import os, sys, math, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for p in ("env", os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")):
    if os.path.isdir(p):
        sys.path.insert(0, p)

import continuous_utm_env as C
from continuous_utm_env import ContinuousUTMEnv

SEP_H, SEP_V = C.SEP_H, C.SEP_V
DT, ALT_MIN, ALT_MAX, VZ_MAX = C.DT, C.ALT_MIN, C.ALT_MAX, C.VZ_MAX
V_MIN, V_MAX = C.V_MIN, C.V_MAX
SECTOR_M = C.SECTOR_DEG * C.M_PER_DEG
KM = 1000.0


# ---------------------------------------------------------------- policy
def load_policy(ckpt_path, obs_dim):
    try:
        import torch
        from train_continuous import ContinuousActorCritic
        pol = ContinuousActorCritic(obs_dim)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        pol.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
        pol.eval()
        print(f"[policy] loaded trained network from {ckpt_path}")
        return (lambda obs: pol.act(np.asarray(obs, np.float32), deterministic=True)), "Trained policy"
    except Exception as e:
        print(f"[policy] FALLBACK greedy controller ({type(e).__name__}: {e})")
        return (lambda obs: None), "Greedy baseline (no checkpoint)"


# ---------------------------------------------------------------- shield
def apply_shield(env, passes=8, target=1.3 * SEP_H, brake=0.4):
    for _ in range(passes):
        act = np.where(~env._done)[0]
        if len(act) < 2:
            break
        P = env.pos[act]; any_fix = False
        for a in range(len(act)):
            for b in range(a + 1, len(act)):
                i, j = act[a], act[b]
                dh = math.hypot(P[a, 0] - P[b, 0], P[a, 1] - P[b, 1]); dv = abs(P[a, 2] - P[b, 2])
                if dh < target and dv < SEP_V:
                    any_fix = True
                    ux, uy = ((P[a, 0]-P[b, 0])/dh, (P[a, 1]-P[b, 1])/dh) if dh > 1e-6 else (1.0, 0.0)
                    push = 0.5 * (target - dh)
                    env.pos[i, 0] += ux*push; env.pos[i, 1] += uy*push
                    env.pos[j, 0] -= ux*push; env.pos[j, 1] -= uy*push
                    hi, lo = (i, j) if env.pos[i, 2] >= env.pos[j, 2] else (j, i)
                    env.pos[hi, 2] = min(ALT_MAX, env.pos[hi, 2] + VZ_MAX*DT)
                    env.pos[lo, 2] = max(ALT_MIN, env.pos[lo, 2] - VZ_MAX*DT)
                    env.spd[i] *= brake; env.spd[j] *= brake
        if not any_fix:
            break


# ---------------------------------------------------------------- rollout
def greedy_actions(env):
    dg = env.goal - env.pos[:, :2]
    tgt = np.arctan2(dg[:, 1], dg[:, 0])
    herr = np.arctan2(np.sin(tgt - env.hdg), np.cos(tgt - env.hdg))
    a = np.zeros((env.n_drones, 3), np.float32)
    a[:, 0] = np.clip(herr / C.DPSI_MAX, -1, 1); a[:, 1] = 0.5
    return a


def rollout(drones, scale, steps, seed, act_fn, shield):
    env = ContinuousUTMEnv(n_drones=drones, scale=scale, max_steps=steps)
    obs, _ = env.reset(seed=seed)
    frames = []; cum = 0
    for t in range(steps):
        a = act_fn(obs)
        if a is None:
            a = greedy_actions(env)
        env.step_joint(a)
        if shield:
            apply_shield(env)
        ai = np.where(~env._done)[0]
        cinfo = env.detector.check3d(env.pos[ai]) if len(ai) > 1 else {"count": 0, "per_drone": np.zeros(0, bool)}
        per = np.zeros(env.n_drones, bool)
        if len(cinfo["per_drone"]):
            per[ai[:len(cinfo["per_drone"])]] = cinfo["per_drone"]
        cum += int(cinfo["count"])
        frames.append(dict(pos=env.pos.copy(), done=env._done.copy(), conf=per.copy(),
                           cum=cum, completed=int(env._done.sum()), step=t + 1))
        obs = env.get_all_agent_obs()
    return env, frames


# ---------------------------------------------------------------- 3D render
def draw_static(ax, env):
    ax.set_xlim(0, env.cols * SECTOR_M / KM); ax.set_ylim(0, env.rows * SECTOR_M / KM)
    ax.set_zlim(ALT_MIN, ALT_MAX)
    ax.set_xlabel("x (km)", fontsize=9); ax.set_ylabel("y (km)", fontsize=9)
    ax.set_zlabel("altitude (m)", fontsize=9)
    ax.set_box_aspect((1, 1, 0.45))
    # no-fly zones as translucent vertical columns
    for sid in env.NO_FLY:
        r, c = divmod(sid, env.cols)
        x0, y0 = c * SECTOR_M / KM, r * SECTOR_M / KM
        s = SECTOR_M / KM
        rect = Rectangle((x0, y0), s, s, facecolor="#d62728", alpha=0.12, edgecolor="none")
        ax.add_patch(rect); art3d.pathpatch_2d_to_3d(rect, z=ALT_MIN, zdir="z")
    # goals on the ground
    ax.scatter(env.goal[:, 0] / KM, env.goal[:, 1] / KM, np.full(env.n_drones, ALT_MIN),
               marker="x", s=12, c="#bbb", depthshade=False)


def colors(fr, env):
    alt = (fr["pos"][:, 2] - ALT_MIN) / max(1.0, ALT_MAX - ALT_MIN)
    c = plt.cm.viridis(np.clip(alt, 0, 1))
    c[fr["conf"]] = (0.84, 0.15, 0.16, 1.0)
    c[fr["done"]] = (0.6, 0.6, 0.6, 0.18)
    return c


def render(panels, out, fps, title):
    """panels: list of (env, frames, subtitle)."""
    n = min(len(f) for _, f, _ in panels)
    ncol = len(panels)
    fig = plt.figure(figsize=(7.2 * ncol, 6.6))
    axes = [fig.add_subplot(1, ncol, i + 1, projection="3d") for i in range(ncol)]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def frame(k):
        az = 35 + 0.35 * k
        for ax, (env, frames, sub) in zip(axes, panels):
            ax.clear(); draw_static(ax, env); ax.view_init(elev=22, azim=az)
            ax.set_title(sub, fontsize=11, fontweight="bold")
            # trails
            for j in range(max(0, k - 6), k):
                m = ~frames[j]["done"]
                P = frames[j]["pos"][m]
                ax.scatter(P[:, 0]/KM, P[:, 1]/KM, P[:, 2], s=3, c=[(0.3, 0.5, 0.8, 0.08)], depthshade=False)
            fr = frames[k]; P = fr["pos"]
            ax.scatter(P[:, 0]/KM, P[:, 1]/KM, P[:, 2], s=26, c=colors(fr, env),
                       edgecolors="white", linewidths=0.3, depthshade=True)
            ax.text2D(0.02, 0.97, f"step {fr['step']:3d}   active {env.n_drones-fr['completed']:3d}   "
                      f"arrived {fr['completed']:3d}   conflicts(cum) {fr['cum']:4d}",
                      transform=ax.transAxes, fontsize=9, family="monospace",
                      bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))

    anim = manim.FuncAnimation(fig, frame, frames=n, interval=1000 / fps)
    try:
        anim.save(out, writer=manim.FFMpegWriter(fps=fps, bitrate=2800))
        print(f"[video] wrote {out}")
    except Exception as e:
        gif = os.path.splitext(out)[0] + ".gif"
        anim.save(gif, writer=manim.PillowWriter(fps=fps)); print(f"[video] ffmpeg failed ({e}); wrote {gif}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=100)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", type=str, default="checkpoints_continuous_100/continuous_best.pt")
    ap.add_argument("--out", type=str, default="demo3d.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--split", action="store_true", help="two panels: shield OFF vs ON")
    ap.add_argument("--noshield", action="store_true", help="single panel, policy alone")
    args = ap.parse_args()
    scale = args.scale or args.drones
    probe = ContinuousUTMEnv(n_drones=args.drones, scale=scale, max_steps=args.steps)
    act_fn, pol_label = load_policy(args.ckpt, probe.agent_obs_dim)
    title = f"Continuous 3D UTM  |  {args.drones} UAVs  |  {pol_label}"

    if args.split:
        eo, fo = rollout(args.drones, scale, args.steps, args.seed, act_fn, False)
        en, fn = rollout(args.drones, scale, args.steps, args.seed, act_fn, True)
        print(f"[summary] OFF conflicts={fo[-1]['cum']}  ON conflicts={fn[-1]['cum']}")
        render([(eo, fo, "shield OFF"), (en, fn, "shield ON")], args.out, args.fps, title)
    else:
        shield = not args.noshield
        e, f = rollout(args.drones, scale, args.steps, args.seed, act_fn, shield)
        print(f"[summary] conflicts(cum)={f[-1]['cum']}  arrived={f[-1]['completed']}/{args.drones}")
        sub = "deployed system (shield ON)" if shield else "policy alone (shield OFF)"
        render([(e, f, sub)], args.out, args.fps, title)


if __name__ == "__main__":
    main()