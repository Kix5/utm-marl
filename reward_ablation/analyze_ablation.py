#!/usr/bin/env python3
"""
analyze_ablation.py -- turn reward_ablation.csv into a table and figures.

Produces, per evaluation scale (e.g. 100 and 500):
  - sensitivity_s<scale>.png : 4 panels (conflict/warning/arrival/progress),
      conflicts (left axis) and throughput (right axis) vs the swept weight,
      with std error bars and a dashed line at the chosen value.
  - minsep_s<scale>.png      : 4 panels, minimum separation vs the swept weight.
  - pareto_s<scale>.png      : throughput vs conflicts for every config, the
      chosen weights highlighted.
And one results_table.md across all configs/scales.

Usage:
    python analyze_ablation.py --csv reward_ablation.csv --outdir results
"""
import os, csv, argparse, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHOSEN = dict(progress=0.001, arrival=10.0, conflict=2.0, warning=3.0)
GROUPS = ["conflict", "warning", "arrival", "progress"]


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    scales = sorted({int(m.group(1)) for c in rows[0]
                     for m in [re.match(r"conflicts_mean_s(\d+)", c)] if m})
    return rows, scales


def _series(rows, group, metric, scale):
    """(x, mean, std) for a weight group vs a metric at a scale.
    Columns are named '<metric>_mean_s<scale>' / '<metric>_std_s<scale>'."""
    mcol, scol = f"{metric}_mean_s{scale}", f"{metric}_std_s{scale}"
    pts = []
    for r in rows:
        name = r["config"]
        if name == "chosen" or name.startswith(group + "="):
            pts.append((float(r[group]), float(r[mcol]), float(r[scol])))
    pts.sort()
    if not pts:
        return None
    xs, ms, ss = zip(*pts)
    return np.array(xs), np.array(ms), np.array(ss)


def sensitivity_fig(rows, scale, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, g in zip(axes.flat, GROUPS):
        c = _series(rows, g, "conflicts", scale)
        t = _series(rows, g, "throughput", scale)
        if c is None:
            ax.set_visible(False); continue
        ax.errorbar(c[0], c[1], yerr=c[2], marker="o", color="#a6413b",
                    capsize=3, label="conflicts/ep")
        ax.set_xlabel(f"{g} weight"); ax.set_ylabel("conflicts / episode", color="#a6413b")
        ax.tick_params(axis="y", labelcolor="#a6413b")
        ax.axvline(CHOSEN[g], ls="--", color="#555", lw=1)
        ax.annotate("chosen", (CHOSEN[g], ax.get_ylim()[1]), fontsize=8,
                    ha="center", va="top", color="#555")
        if t is not None:
            ax2 = ax.twinx()
            ax2.errorbar(t[0], t[1] * 100, yerr=t[2] * 100, marker="s",
                         color="#2f7a55", capsize=3)
            ax2.set_ylabel("throughput (%)", color="#2f7a55")
            ax2.tick_params(axis="y", labelcolor="#2f7a55")
        ax.set_title(f"{g}  (chosen = {CHOSEN[g]})")
    fig.suptitle(f"Reward-weight sensitivity @ {scale} UAVs "
                 f"(red = conflicts, green = throughput)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, f"sensitivity_s{scale}.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def minsep_fig(rows, scale, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, g in zip(axes.flat, GROUPS):
        m = _series(rows, g, "minsep", scale)
        if m is None:
            ax.set_visible(False); continue
        ax.errorbar(m[0], m[1], yerr=m[2], marker="o", color="#35607f", capsize=3)
        ax.axhline(500, ls=":", color="#a6413b", lw=1, label="S_H = 500 m")
        ax.axvline(CHOSEN[g], ls="--", color="#555", lw=1)
        ax.set_xlabel(f"{g} weight"); ax.set_ylabel("min separation (m)")
        ax.set_title(f"{g}  (chosen = {CHOSEN[g]})"); ax.legend(fontsize=8)
    fig.suptitle(f"Minimum separation vs reward weights @ {scale} UAVs", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(outdir, f"minsep_s{scale}.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def pareto_fig(rows, scale, outdir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in rows:
        x = float(r[f"throughput_mean_s{scale}"]) * 100
        y = float(r[f"conflicts_mean_s{scale}"])
        chosen = (r["config"] == "chosen")
        ax.scatter(x, y, s=90 if chosen else 45,
                   color="#d08a1e" if chosen else "#35607f",
                   edgecolor="black" if chosen else "none", zorder=3 if chosen else 2)
        if chosen:
            ax.annotate("chosen", (x, y), fontsize=9, fontweight="bold",
                        xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("throughput (%)  -->  better")
    ax.set_ylabel("conflicts / episode  -->  worse")
    ax.set_title(f"Safety / throughput trade-off @ {scale} UAVs\n"
                 f"(bottom-right is best; is 'chosen' on the frontier?)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, f"pareto_s{scale}.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def table_md(rows, scales, outdir):
    lines = ["# Reward-weight ablation results\n"]
    hdr = ["config", "prog", "arr", "conf", "warn"]
    for s in scales:
        hdr += [f"conf/ep@{s}", f"tp%@{s}", f"minsep@{s}"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [r["config"], r["progress"], r["arrival"], r["conflict"], r["warning"]]
        for s in scales:
            cells.append(f"{float(r[f'conflicts_mean_s{s}']):.1f}±{float(r[f'conflicts_std_s{s}']):.1f}")
            cells.append(f"{float(r[f'throughput_mean_s{s}'])*100:.0f}±{float(r[f'throughput_std_s{s}'])*100:.0f}")
            cells.append(f"{float(r[f'minsep_mean_s{s}']):.0f}")
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    p = os.path.join(outdir, "results_table.md")
    open(p, "w").write("\n".join(lines) + "\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reward_ablation.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rows, scales = load(args.csv)
    if not scales:
        raise SystemExit("No scale columns found in CSV (expected conflicts_mean_s<scale>).")
    made = [table_md(rows, scales, args.outdir)]
    for s in scales:
        made += [sensitivity_fig(rows, s, args.outdir),
                 minsep_fig(rows, s, args.outdir),
                 pareto_fig(rows, s, args.outdir)]
    print(f"Analyzed {len(rows)} configs at scales {scales}. Wrote:")
    for m in made:
        print("  " + m)


if __name__ == "__main__":
    main()
