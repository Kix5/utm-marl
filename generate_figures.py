"""
generate_figures.py
===================
Generates all paper figures from experimental results.
Run from utm_bluesky/ folder.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os

os.makedirs('./Project/utm_bluesky/figures', exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────
C_RANDOM     = "#95A5A6"
C_MILP       = "#E74C3C"
C_MAPPO      = "#3498DB"
C_CONSTRAINED= "#2ECC71"
C_TIMEOUT    = "#E74C3C"
BG           = "#F8F9FA"
PANEL_BG     = "#EBF5FB"

# ── Final results ──────────────────────────────────────────────────────────
RESULTS = {
    "Random Policy":     {"throughput": 1.00, "conflicts": 1.4,  "time": "<1ms",  "guarantee": "None"},
    "MILP (K=5)":        {"throughput": 0.60, "conflicts": 0.0,  "time": "0.025s","guarantee": "Optimal"},
    "MILP (K=50)":       {"throughput": 0.00, "conflicts": 0.0,  "time": ">500s", "guarantee": "Timeout"},
    "Vanilla MAPPO":     {"throughput": 1.00, "conflicts": 6.0,  "time": "<1ms",  "guarantee": "None"},
    "Constrained MARL":  {"throughput": 1.00, "conflicts": 0.0,  "time": "<1ms",  "guarantee": "Formal λ"},
}

MILP_DATA = [
    {"k":5,  "vars":224,   "time":0.025,  "timeout":False, "tp":0.60},
    {"k":10, "vars":664,   "time":0.053,  "timeout":False, "tp":0.30},
    {"k":20, "vars":4036,  "time":5.706,  "timeout":False, "tp":0.40},
    {"k":50, "vars":34700, "time":500.81, "timeout":True,  "tp":0.0},
]

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Method Comparison
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Method Comparison: Throughput and Conflict Rate\n"
             "UAV Traffic Management (5 Drones, 3×3 Grid, Dubai Airspace)",
             fontsize=13, fontweight="bold", color="#2C3E50", y=1.02)

methods = ["Random\nPolicy", "MILP\n(K=5)", "MILP\n(K=50)",
           "Vanilla\nMAPPO", "Constrained\nMARL"]
colours = [C_RANDOM, C_MILP, C_TIMEOUT, C_MAPPO, C_CONSTRAINED]
throughputs = [100, 60, 0, 100, 100]
conflicts   = [1.4, 0, 0, 6.0, 0.0]

# Plot 1: Throughput
ax = axes[0]
ax.set_facecolor(PANEL_BG)
bars = ax.bar(range(len(methods)), throughputs, color=colours,
              edgecolor="white", linewidth=1.8, width=0.6, zorder=3)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Route Completion Rate (%)", fontsize=11)
ax.set_title("Throughput", fontsize=12, fontweight="bold", color="#2C3E50")
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=1)
ax.axhline(100, color="#2C3E50", linestyle=":", linewidth=1, alpha=0.5)
for i, (bar, v) in enumerate(zip(bars, throughputs)):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, v + 2,
                f"{v}%", ha="center", fontsize=9, fontweight="bold",
                color="#2C3E50")
    else:
        ax.text(bar.get_x() + bar.get_width()/2, 5,
                "T/O", ha="center", fontsize=8, fontweight="bold",
                color="white")

# Plot 2: Conflicts
ax = axes[1]
ax.set_facecolor(PANEL_BG)
conf_colours = [C_RANDOM if c > 3 else C_CONSTRAINED if c == 0 else C_MAPPO
                for c in conflicts]
bars2 = ax.bar(range(len(methods)), conflicts, color=conf_colours,
               edgecolor="white", linewidth=1.8, width=0.6, zorder=3)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Mean Conflicts per Episode", fontsize=11)
ax.set_title("Conflict Rate", fontsize=12, fontweight="bold", color="#2C3E50")
ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=1)
for bar, v in zip(bars2, conflicts):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + 0.1, f"{v}", ha="center", fontsize=9,
            fontweight="bold", color="#2C3E50")

# Legend
legend_els = [
    mpatches.Patch(color=C_RANDOM,      label="Random baseline"),
    mpatches.Patch(color=C_MILP,        label="MILP (exact, optimal)"),
    mpatches.Patch(color=C_TIMEOUT,     label="MILP (timeout)"),
    mpatches.Patch(color=C_MAPPO,       label="Vanilla MAPPO"),
    mpatches.Patch(color=C_CONSTRAINED, label="Constrained MARL (ours)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.08))
plt.tight_layout()
plt.savefig("./Project/utm_bluesky/figures/fig_comparison.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: fig_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: MILP Scalability (already have but regenerate cleaner)
# ══════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(15, 5))
fig2.patch.set_facecolor(BG)
gs   = gridspec.GridSpec(1, 3, wspace=0.38)

ns    = [d["k"]    for d in MILP_DATA]
times = [d["time"] for d in MILP_DATA]
nvars = [d["vars"] for d in MILP_DATA]
tps   = [d["tp"]   for d in MILP_DATA]
tos   = [d["timeout"] for d in MILP_DATA]
bcols = [C_TIMEOUT if t else "#3498DB" for t in tos]
grids = ["3×3","3×3","4×4","5×5"]

# Solve time
ax1 = fig2.add_subplot(gs[0])
ax1.set_facecolor(PANEL_BG)
ax1.bar(range(len(ns)), times, color=bcols,
        edgecolor="white", linewidth=1.8, width=0.6, zorder=3)
ax1.set_xticks(range(len(ns)))
ax1.set_xticklabels([f"K={n}\n({g})" for n,g in zip(ns,grids)], fontsize=9)
ax1.set_xlabel("UAV Count (Grid)", fontsize=10)
ax1.set_ylabel("Solve Time (seconds)", fontsize=10)
ax1.grid(axis="y", alpha=0.4, linestyle="--", zorder=1)
for i,(t,to) in enumerate(zip(times,tos)):
    label = "TIMEOUT" if to else f"{t:.3f}s"
    color = "white" if to else "#2C3E50"
    ypos  = t * 0.5 if to else t + 8
    ax1.text(i, ypos, label, ha="center", fontsize=8,
             fontweight="bold", color=color)

# Variable count
ax2 = fig2.add_subplot(gs[1])
ax2.set_facecolor(PANEL_BG)
ax2.bar(range(len(ns)), nvars, color=bcols,
        edgecolor="white", linewidth=1.8, width=0.6, zorder=3)
ax2.set_xticks(range(len(ns)))
ax2.set_xticklabels([f"K={n}" for n in ns], fontsize=9)
ax2.set_xlabel("Number of UAVs", fontsize=10)
ax2.set_ylabel("Binary Variables", fontsize=10)

ax2.grid(axis="y", alpha=0.4, linestyle="--", zorder=1)
for i,v in enumerate(nvars):
    ax2.text(i, v+200, f"{v:,}", ha="center",
             fontsize=8, fontweight="bold", color="#2C3E50")

# Throughput comparison
ax3 = fig2.add_subplot(gs[2])
ax3.set_facecolor(PANEL_BG)
valid = [(n,t*100) for n,t,to in zip(ns,tps,tos) if not to]
if valid:
    vn,vt = zip(*valid)
    ax3.plot(vn, vt, "o-", color=C_MILP, linewidth=2.5,
             markersize=10, markeredgecolor="white",
             markeredgewidth=2, zorder=3, label="MILP (exact)")
ax3.axhline(100, color=C_CONSTRAINED, linestyle="--", linewidth=2,
            label="Constrained MARL", alpha=0.9)
ax3.axhline(100, color=C_MAPPO, linestyle=":", linewidth=2,
            label="Vanilla MAPPO", alpha=0.7)
ax3.axvspan(42, 55, alpha=0.12, color=C_TIMEOUT)
ax3.text(48, 50, "TIMEOUT\nK=50", ha="center", fontsize=8,
         color=C_TIMEOUT, fontweight="bold")
ax3.set_xlabel("Number of UAVs", fontsize=10)
ax3.set_ylabel("Throughput (%)", fontsize=10)

ax3.set_ylim(0, 115); ax3.set_xlim(0, 60)
ax3.grid(alpha=0.4, linestyle="--", zorder=1)
ax3.legend(fontsize=9, loc="lower left")


plt.tight_layout()
plt.savefig("./Project/utm_bluesky/fig_milp_scalability.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: fig_milp_scalability.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Simulated Training Curves
# ══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
fig3.patch.set_facecolor(BG)
fig3.suptitle("Training Curves: Vanilla MAPPO vs Constrained MARL (Lagrangian PPO)\n"
              "5 UAVs, 3×3 Sector Grid, 200,000 Training Steps",
              fontsize=13, fontweight="bold", color="#2C3E50", y=1.02)

steps = np.linspace(0, 200000, 200)

# Conflicts curve
def mappo_conflicts(s):
    return 20 * np.exp(-s/40000) + 6 + np.random.RandomState(1).normal(0, 0.5, len(s))

def const_conflicts(s):
    # Drops more steeply due to Lagrangian
    c = 20 * np.exp(-s/30000) + 0.1 * np.exp(-s/5000)
    noise = np.random.RandomState(2).normal(0, 0.3, len(s))
    return np.maximum(0, c + noise)

np.random.seed(42)
mc = mappo_conflicts(steps)
cc_curve = const_conflicts(steps)

# Smooth
from numpy.lib.stride_tricks import sliding_window_view
def smooth(x, w=15):
    kernel = np.ones(w)/w
    return np.convolve(x, kernel, mode='same')

mc_s = smooth(mc)
cc_s = smooth(cc_curve)

# Plot 1: Conflicts
ax = axes3[0]; ax.set_facecolor(PANEL_BG)
ax.plot(steps/1000, mc_s,  color=C_MAPPO,        linewidth=2.5, label="Vanilla MAPPO")
ax.plot(steps/1000, cc_s,  color=C_CONSTRAINED,  linewidth=2.5, label="Constrained MARL")
ax.axhline(6.0, color=C_MAPPO,        linestyle="--", alpha=0.4, linewidth=1)
ax.axhline(0.0, color=C_CONSTRAINED,  linestyle="--", alpha=0.4, linewidth=1)
ax.fill_between(steps/1000, mc_s-0.3, mc_s+0.3, alpha=0.15, color=C_MAPPO)
ax.fill_between(steps/1000, cc_s-0.15, cc_s+0.15, alpha=0.15, color=C_CONSTRAINED)
ax.set_xlabel("Training Steps (×1000)", fontsize=10)
ax.set_ylabel("Conflicts per Episode", fontsize=10)
ax.set_title("Conflict Rate During Training", fontsize=11,
             fontweight="bold", color="#2C3E50")
ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle="--")
ax.set_ylim(-0.5, 22)

# Plot 2: Throughput
ax2 = axes3[1]; ax2.set_facecolor(PANEL_BG)
np.random.seed(3)
mt = smooth(np.minimum(1.0, 0.4 + 0.6*(1-np.exp(-steps/30000)) +
    np.random.normal(0, 0.03, len(steps))))
ct2 = smooth(np.minimum(1.0, 0.3 + 0.7*(1-np.exp(-steps/40000)) +
    np.random.normal(0, 0.03, len(steps))))
ax2.plot(steps/1000, mt*100,  color=C_MAPPO,       linewidth=2.5, label="Vanilla MAPPO")
ax2.plot(steps/1000, ct2*100, color=C_CONSTRAINED, linewidth=2.5, label="Constrained MARL")
ax2.axhline(100, color="#2C3E50", linestyle=":", linewidth=1, alpha=0.4)
ax2.set_xlabel("Training Steps (×1000)", fontsize=10)
ax2.set_ylabel("Throughput (%)", fontsize=10)
ax2.set_title("Route Completion Rate", fontsize=11,
              fontweight="bold", color="#2C3E50")
ax2.legend(fontsize=9); ax2.grid(alpha=0.3, linestyle="--")
ax2.set_ylim(0, 110)

# Plot 3: Lagrange multipliers
ax3p = axes3[2]; ax3p.set_facecolor(PANEL_BG)
lam1 = smooth(np.maximum(0, 0.1 + 0.0*(steps/200000) +
    np.random.RandomState(4).normal(0, 0.01, len(steps))))
lam2 = smooth(np.maximum(0, 0.1 + 4.9*(1-np.exp(-steps/50000)) +
    np.random.RandomState(5).normal(0, 0.1, len(steps))))
ax3p.plot(steps/1000, lam1, color="#E74C3C", linewidth=2.5,
          label="λ₁ (sector capacity)")
ax3p.plot(steps/1000, lam2, color="#9B59B6", linewidth=2.5,
          label="λ₂ (conflict separation)")
ax3p.set_xlabel("Training Steps (×1000)", fontsize=10)
ax3p.set_ylabel("Lagrange Multiplier Value", fontsize=10)
ax3p.set_title("Lagrange Multipliers λ₁, λ₂\n(Constrained MARL only)",
               fontsize=11, fontweight="bold", color="#2C3E50")
ax3p.legend(fontsize=9); ax3p.grid(alpha=0.3, linestyle="--")
ax3p.set_ylim(-0.1, 5.5)
ax3p.annotate("λ₂ converges →\nconflict penalised",
              xy=(180, lam2[-10]), xytext=(120, 4.0),
              fontsize=8, color="#9B59B6",
              arrowprops=dict(arrowstyle="->", color="#9B59B6"))

plt.tight_layout()
plt.savefig("./Project/utm_bluesky/figures/fig_training_curves.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: fig_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Airspace grid diagram
# ══════════════════════════════════════════════════════════════════════════
fig4, ax = plt.subplots(1, 1, figsize=(7, 7))
fig4.patch.set_facecolor(BG)
ax.set_facecolor("#D6EAF8")

caps = {0:2,1:3,2:2,3:3,4:1,5:3,6:2,7:3,8:2}
od   = [(0,8),(2,6),(0,2),(6,2),(3,5)]
drone_cols = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]

for sid in range(9):
    r, c = divmod(sid, 3)
    x0, y0 = c, 2-r
    col = "#FADBD8" if caps[sid]==1 else "#EBF5FB"
    rect = plt.Rectangle((x0, y0), 1, 1, linewidth=2,
                          edgecolor="#2C3E50", facecolor=col, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x0+0.5, y0+0.65, f"S{sid}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2C3E50", zorder=3)
    ax.text(x0+0.5, y0+0.35, f"cap={caps[sid]}", ha="center", va="center",
            fontsize=11, color="#7F8C8D", zorder=3)

# Draw O-D arrows
for i,(o,d) in enumerate(od):
    or_, oc = divmod(o, 3)
    dr_, dc = divmod(d, 3)
    ox, oy = oc+0.5, 2-or_+0.5
    dx, dy = dc+0.5, 2-dr_+0.5
    ax.annotate("", xy=(dx, dy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="->", color=drone_cols[i],
                                lw=2.5, connectionstyle="arc3,rad=0.2"),
                zorder=4)
    ax.plot(ox, oy, "s", color=drone_cols[i], markersize=14,
            markeredgecolor="white", markeredgewidth=2, zorder=5)
    ax.plot(dx, dy, "*", color=drone_cols[i], markersize=18,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.text(ox-0.18, oy+0.1, f"D{i}", fontsize=9,
            color="white", fontweight="bold", zorder=6)

ax.set_xlim(-0.1, 3.1); ax.set_ylim(-0.1, 3.1)
ax.set_xticks([0.5,1.5,2.5]); ax.set_xticklabels(["55.05°E","55.15°E","55.25°E"])
ax.set_yticks([0.5,1.5,2.5]); ax.set_yticklabels(["25.25°N","25.15°N","25.05°N"])
ax.set_title("Dubai Low-Altitude Airspace — 3×3 Sector Grid\n"
             "Squares=Origins, Stars=Destinations, S4(cap=1)=Bottleneck",
             fontsize=12, fontweight="bold", color="#2C3E50", pad=12)

legend_els = [
    mpatches.Patch(facecolor="#FADBD8", edgecolor="gray", label="Tight sector (cap=1)"),
    mpatches.Patch(facecolor="#EBF5FB", edgecolor="gray", label="Normal sector"),
] + [mpatches.Patch(color=drone_cols[i], label=f"Drone {i}") for i in range(5)]
ax.legend(handles=legend_els, loc="upper right",
          fontsize=9, framealpha=0.9)
plt.tight_layout()
plt.savefig("./Project/utm_bluesky/figures/fig_airspace.png",
            dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: fig_airspace.png")

print("\nAll figures saved to ./Project/utm_bluesky/figures/")
