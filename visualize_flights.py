"""
visualize_flights.py
====================
Standalone animated visualization of UAV flights in the UTM sector grid.
Does NOT require BlueSky GUI — uses matplotlib animation instead.
Much more reliable on Apple Silicon Mac.

Usage:
    # Show animation live:
    python visualize_flights.py

    # Save as MP4 video file:
    python visualize_flights.py --save

    # Use trained policy checkpoint:
    python visualize_flights.py --checkpoint checkpoints_bluesky/mappo_bluesky_final.pt

    # More episodes:
    python visualize_flights.py --episodes 3
"""

import sys, os, argparse, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import torch
import torch.nn as nn
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
from bluesky_utm_env import BlueSkyUTMEnv, sector_to_latlon, ORIGIN_LAT, ORIGIN_LON, SECTOR_DEG, GRID_ROWS, GRID_COLS

# ── Colours per drone ──────────────────────────────────────────────────────
DRONE_COLOURS = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
SECTOR_COLOUR = "#ECF0F1"
OVERLOAD_COLOUR = "#FADBD8"
NO_FLY_COLOUR  = "#BDC3C7"

# ── Minimal policy (same architecture as trainer) ──────────────────────────
class Policy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        t = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = self(t)
        return int(logits.argmax().item())
def run_episode(env, policy, max_steps=80):
    """Run one episode and record all drone positions."""
    env.reset()
    all_obs = env.get_all_agent_obs()

    history = {i: {"lats":[], "lons":[], "done":[]} for i in range(env.n_drones)}
    info_history = []

    # Record initial positions
    for i in range(env.n_drones):
        history[i]["lats"].append(float(env._positions[i,0]))
        history[i]["lons"].append(float(env._positions[i,1]))
        history[i]["done"].append(False)

    for step in range(max_steps):
        if policy:
            acts = [policy.act(all_obs[i]) for i in range(env.n_drones)]
        else:
            acts = [env.action_space.sample() for _ in range(env.n_drones)]

        all_obs, rewards, dones, terminated, truncated, info = env.step_joint(acts)
        info_history.append(info.copy())

        for i in range(env.n_drones):
            history[i]["lats"].append(float(env._positions[i,0]))
            history[i]["lons"].append(float(env._positions[i,1]))
            history[i]["done"].append(bool(dones[i]))

        if terminated or truncated:
            break

    return history, info_history, env._drone_goals.copy()


def draw_sector_grid(ax):
    """Draw the 3x3 sector grid with capacities."""
    caps = {0:2,1:3,2:2,3:3,4:1,5:3,6:2,7:3,8:2}

    for sid in range(9):
        row, col = divmod(sid, GRID_COLS)
        lat0 = ORIGIN_LAT + row * SECTOR_DEG
        lon0 = ORIGIN_LON + col * SECTOR_DEG

        color = OVERLOAD_COLOUR if caps[sid] == 1 else SECTOR_COLOUR
        rect = plt.Rectangle(
            (lon0, lat0), SECTOR_DEG, SECTOR_DEG,
            linewidth=1.5, edgecolor="#2C3E50",
            facecolor=color, alpha=0.6, zorder=1
        )
        ax.add_patch(rect)

        # Sector label
        clat = lat0 + SECTOR_DEG/2
        clon = lon0 + SECTOR_DEG/2
        ax.text(clon, clat + 0.025, f"S{sid}",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#2C3E50", zorder=3)
        ax.text(clon, clat - 0.025, f"cap:{caps[sid]}",
                ha="center", va="center",
                fontsize=7, color="#7F8C8D", zorder=3)


def make_animation(histories, goals_list, info_list, save_path=None):
    """Create animated figure showing drone flights."""
    n_ep    = len(histories)
    n_steps = max(len(h[0]["lats"]) for h in histories)
    n_drones = len(histories[0])

    fig, axes = plt.subplots(1, n_ep, figsize=(7*n_ep, 7))
    if n_ep == 1:
        axes = [axes]

    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle("UTM Multi-Agent Simulation\n5 UAVs — 3×3 Sector Grid (Dubai Airspace)",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)

    all_artists = []

    for ep_idx, (ax, history, goals, infos) in enumerate(zip(axes, histories, goals_list, info_list)):
        ax.set_facecolor("#EBF5FB")
        draw_sector_grid(ax)

        # Axis formatting
        ax.set_xlim(ORIGIN_LON - 0.02, ORIGIN_LON + GRID_COLS*SECTOR_DEG + 0.02)
        ax.set_ylim(ORIGIN_LAT - 0.02, ORIGIN_LAT + GRID_ROWS*SECTOR_DEG + 0.02)
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.set_title(f"Episode {ep_idx+1}", fontsize=11, color="#2C3E50")
        ax.grid(True, alpha=0.3, linestyle="--")

        # Draw goal markers
        for i in range(n_drones):
            glat, glon = sector_to_latlon(goals[i])
            ax.plot(glon, glat, marker="*", markersize=18,
                    color=DRONE_COLOURS[i], alpha=0.4,
                    markeredgecolor="white", markeredgewidth=1, zorder=2)
            ax.text(glon + 0.005, glat + 0.005, f"G{i}",
                    fontsize=7, color=DRONE_COLOURS[i], fontweight="bold", zorder=4)

        # Draw origin markers
        for i in range(n_drones):
            olat = history[i]["lats"][0]
            olon = history[i]["lons"][0]
            ax.plot(olon, olat, marker="s", markersize=10,
                    color=DRONE_COLOURS[i], alpha=0.5,
                    markeredgecolor="white", markeredgewidth=1, zorder=2)

        # Trail lines (will be updated)
        trails = [
            ax.plot([], [], "-", color=DRONE_COLOURS[i],
                    alpha=0.4, linewidth=1.5, zorder=4)[0]
            for i in range(n_drones)
        ]
        # Drone markers
        markers = [
            ax.plot([], [], "o", color=DRONE_COLOURS[i],
                    markersize=12, markeredgecolor="white",
                    markeredgewidth=2, zorder=6)[0]
            for i in range(n_drones)
        ]
        # Drone ID labels
        labels = [
            ax.text(0, 0, f"D{i}", fontsize=8,
                    color="white", fontweight="bold",
                    ha="center", va="center", zorder=7)
            for i in range(n_drones)
        ]
        # Step counter
        step_text = ax.text(
            ORIGIN_LON + 0.01,
            ORIGIN_LAT + GRID_ROWS*SECTOR_DEG + 0.01,
            "Step: 0", fontsize=9, color="#2C3E50",
            fontweight="bold", zorder=8
        )
        # Conflict / overload counter
        info_text = ax.text(
            ORIGIN_LON + GRID_COLS*SECTOR_DEG - 0.01,
            ORIGIN_LAT + GRID_ROWS*SECTOR_DEG + 0.01,
            "", fontsize=9, color="#E74C3C",
            ha="right", fontweight="bold", zorder=8
        )

        all_artists.append({
            "trails": trails, "markers": markers,
            "labels": labels, "step_text": step_text,
            "info_text": info_text, "history": history,
            "infos": infos, "ax": ax
        })

    def animate(frame):
        artists_out = []
        for ep_data in all_artists:
            hist   = ep_data["history"]
            infos  = ep_data["infos"]
            step   = min(frame, len(hist[0]["lats"]) - 1)

            for i in range(n_drones):
                lats = hist[i]["lats"][:step+1]
                lons = hist[i]["lons"][:step+1]

                ep_data["trails"][i].set_data(lons, lats)
                ep_data["markers"][i].set_data([lons[-1]], [lats[-1]])
                ep_data["labels"][i].set_position((lons[-1], lats[-1]))

                # Fade out completed drones
                if hist[i]["done"][step]:
                    ep_data["markers"][i].set_alpha(0.3)
                    ep_data["labels"][i].set_alpha(0.3)
                else:
                    ep_data["markers"][i].set_alpha(1.0)
                    ep_data["labels"][i].set_alpha(1.0)

            ep_data["step_text"].set_text(f"Step: {step}")

            if step > 0 and step - 1 < len(infos):
                info = infos[step-1]
                conflicts = info.get("ep_conflicts", 0)
                completed = info.get("ep_completed", 0)
                ep_data["info_text"].set_text(
                    f"✓ {completed}/{n_drones}  ⚠ {conflicts} conflicts"
                )

            artists_out.extend(ep_data["trails"])
            artists_out.extend(ep_data["markers"])
            artists_out.extend(ep_data["labels"])
            artists_out.append(ep_data["step_text"])
            artists_out.append(ep_data["info_text"])

        return artists_out

    # Legend
    legend_elements = [
        mpatches.Patch(color=DRONE_COLOURS[i], label=f"Drone {i}",
                       alpha=0.8) for i in range(n_drones)
    ]
    legend_elements += [
        mpatches.Patch(facecolor=OVERLOAD_COLOUR, edgecolor="gray",
                       label="Tight sector (cap=1)"),
        plt.Line2D([0],[0], marker="*", color="gray",
                   markersize=10, label="Goal", linestyle="None"),
        plt.Line2D([0],[0], marker="s", color="gray",
                   markersize=8, label="Origin", linestyle="None"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    anim = animation.FuncAnimation(
        fig, animate,
        frames=n_steps,
        interval=150,       # ms per frame
        blit=False,
        repeat=True
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path} ...")
        writer = animation.FFMpegWriter(fps=8, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    return anim


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained policy .pt file")
    parser.add_argument("--episodes",   type=int, default=2,
                        help="Number of episodes to visualize side by side")
    parser.add_argument("--steps",      type=int, default=80,
                        help="Max steps per episode")
    parser.add_argument("--save",       action="store_true",
                        help="Save as MP4 instead of showing live")
    parser.add_argument("--random",     action="store_true",
                        help="Use random policy (no checkpoint needed)")
    args = parser.parse_args()

    # ── Environment ────────────────────────────────────────────────────────
    env = BlueSkyUTMEnv(n_drones=5, render_mode="none", max_steps=args.steps)

    # ── Policy ─────────────────────────────────────────────────────────────
    policy = None
    if not args.random:
        ckpt_path = args.checkpoint or \
                    "checkpoints_bluesky/mappo_bluesky_final.pt"
        if os.path.exists(ckpt_path):
            print(f"Loading policy from: {ckpt_path}")
            policy = Policy(env.agent_obs_dim, env.n_sectors)
            ckpt   = torch.load(ckpt_path, map_location="cpu")
            # Handle both full trainer checkpoint and bare state dict
            state  = ckpt.get("policy", ckpt)
            policy.load_state_dict(state, strict=True)
            policy.eval()
            print("Policy loaded.")
        else:
            print(f"No checkpoint at {ckpt_path} — using random policy")

    # ── Run episodes ────────────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} episode(s)...")
    histories, goals_list, info_list = [], [], []
    for ep in range(args.episodes):
        print(f"  Episode {ep+1}/{args.episodes} ...", end=" ")
        history, infos, goals = run_episode(env, policy, max_steps=args.steps)
        final = infos[-1] if infos else {}
        print(f"completed={final.get('ep_completed',0)}/5  "
              f"conflicts={final.get('ep_conflicts',0)}")
        histories.append(history)
        goals_list.append(goals)
        info_list.append(infos)

    # ── Animate ─────────────────────────────────────────────────────────────
    save_path = "utm_simulation.mp4" if args.save else None
    print("\nLaunching visualization...")
    make_animation(histories, goals_list, info_list, save_path=save_path)
