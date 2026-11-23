"""
run_ablation.py
===============
Ablation study for the constrained UTM system. Shows what each component of the
tactical/safety stack contributes, by switching one component off at a time and
measuring the effect on conflicts and throughput. Every configuration is then
evaluated BOTH with the reactive shield OFF and ON, so the shield's contribution
(residual conflicts driven toward zero) appears as its own column.

This directly answers "where does the UTM management come from": the table
attributes the conflict reduction to layering, reward shaping, and the shield.

CONFIGURATIONS (default; each is one component removed from the full system)
  full        layered airspace + soft-separation shaping + conflict penalty
  no_layer    layering OFF  (tests the altitude-separation contribution)
  no_warn     soft-separation warning penalty OFF
  no_conf     conflict penalty OFF
  sparse      no shaping at all (only progress + arrival reward)

Each trained policy is evaluated at shield=0 and shield=1.

HOW IT STAYS HONEST
  - Same scenario (fixed OD pairs), same PPO hyperparameters, same seeds for
    every configuration. Only the ablated component changes.
  - Variance is over training seeds; results carry a 95% CI.
  - The shield toggle is evaluation-only (same trained policy), so the shield
    column is a clean, retrain-free attribution.

USAGE
  # self-test (tiny), proves the pipeline:
  python run_ablation.py --smoke

  # real ablation (pick a scale that fits your deadline; 300 is a good middle):
  python run_ablation.py --arch mlp --scale 300 --seeds 0 1 2 \
      --updates 400 --steps 400

  --arch mlp|gnn   which policy to ablate (default mlp, the stronger tactical layer;
                   run gnn too if you want the ablation for your named method)
  --scale N        fleet size for the ablation (default 300; smaller = faster)
  --seeds ...      training seeds (default 0 1 2)
  --configs ...    subset of: full no_layer no_warn no_conf sparse (default all)
  --epochs 10      UTM_EPOCHS for the GNN (match the MLP's 10 for fairness)
  --outdir PATH    default ./ablation_results

OUTPUTS
  ablation_raw.csv      one row per (config, seed, shield): conflicts, throughput,
                        overloads, nfz
  ablation_summary.csv  mean + 95% CI per (config, shield)
  ablation_table.txt    formatted table (drops into the Results chapter)
  ablation_conflicts.png  bar chart: conflicts per config, shield off vs on

DEADLINE NOTE
  Each configuration is a full training run, so cost = (#configs) x (#seeds)
  training runs. At scale 300 with 5 configs x 3 seeds that is 15 runs. If time
  is tight, drop to --scale 100, fewer --seeds, or a --configs subset
  (e.g. full no_layer sparse) to get the core story fast.
"""
import argparse, os, sys, subprocess, csv, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH_ROOTS = [os.getcwd(), HERE]


def _find(name):
    for root in _SEARCH_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        hits = [h for h in hits if "ablation_results" not in h and "comparison_results" not in h
                and "/run_" not in h] or hits
        if hits:
            return os.path.abspath(hits[0])
    raise FileNotFoundError(f"Could not find {name}; run this from your project root.")


ENV_FILE    = _find("continuous_utm_env.py")
POLICY_FILE = _find("gnn_policy.py")
TRAIN_MLP   = _find("train_continuous.py")
TRAIN_GNN   = _find("train_gnn.py")
for _d in {os.path.dirname(ENV_FILE), os.path.dirname(POLICY_FILE)}:
    sys.path.insert(0, _d)

import torch
from continuous_utm_env import ContinuousUTMEnv
from gnn_policy import GraphActorCritic
import importlib.util
_spec = importlib.util.spec_from_file_location("train_continuous", TRAIN_MLP)
_tc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tc)
ContinuousActorCritic = _tc.ContinuousActorCritic


# ---- configurations: env toggles per ablation (training-time reward/airspace) ----
# keys map to env vars the trainer/env read.
CONFIGS = {
    "full":        dict(UTM_LAYERS="1", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="2.0"),
    "no_layer":    dict(UTM_LAYERS="0", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="2.0"),
    "no_warn":     dict(UTM_LAYERS="1", UTM_WARN_PENALTY="0.0", UTM_CONF_PENALTY="2.0"),
    "no_conf":     dict(UTM_LAYERS="1", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="0.0"),
    "sparse":      dict(UTM_LAYERS="1", UTM_WARN_PENALTY="0.0", UTM_CONF_PENALTY="0.0"),
    # RQ5: constraint enforcement ON (capacity + no-fly penalties). Compare its
    # overloads / nfz columns against 'full' (which has them OFF) to show the
    # Lagrangian constraint layer reduces violations.
    "constrained": dict(UTM_LAYERS="1", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="2.0",
                        UTM_CAP_PENALTY="0.5", UTM_NFZ_PENALTY="5.0"),
    # RQ5 follow-up (a): stronger FIXED penalties. If a larger hand-set weight
    # reduces overloads / nfz vs 'constrained', it shows the earlier weights were
    # simply too weak (a tuning issue), not that penalties cannot work.
    "constrained_strong": dict(UTM_LAYERS="1", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="2.0",
                        UTM_CAP_PENALTY="4.0", UTM_NFZ_PENALTY="20.0"),
    # RQ5 follow-up (b): ADAPTIVE penalties (dual ascent / PID-Lagrangian). The
    # multipliers start small and are raised automatically during training when
    # the measured per-episode violation count exceeds its budget. Needs the
    # UTM_ADAPTIVE hook in train_continuous.py (backward compatible, off by
    # default). Compare its overloads / nfz against 'constrained' and 'full'.
    "constrained_adaptive": dict(UTM_LAYERS="1", UTM_WARN_PENALTY="3.0", UTM_CONF_PENALTY="2.0",
                        UTM_CAP_PENALTY="0.5", UTM_NFZ_PENALTY="5.0",
                        UTM_ADAPTIVE="1", UTM_CAP_TARGET="100", UTM_NFZ_TARGET="50",
                        UTM_CAP_LR="0.05", UTM_NFZ_LR="0.05",
                        UTM_CAP_MAX="20.0", UTM_NFZ_MAX="60.0"),
}
CONFIG_DESC = {
    "full":        "layered + soft-separation + conflict penalty (constraints OFF)",
    "no_layer":    "layering OFF",
    "no_warn":     "soft-separation warning penalty OFF",
    "no_conf":     "conflict penalty OFF",
    "sparse":      "no shaping (progress + arrival only)",
    "constrained": "full + capacity & no-fly penalties ON (fixed, weak weights)",
    "constrained_strong":   "full + capacity & no-fly penalties ON (fixed, strong weights)",
    "constrained_adaptive": "full + adaptive (dual-ascent) capacity & no-fly penalties",
}


def _make_env(scale, steps, layers, shield):
    os.environ["UTM_LAYERS"] = str(layers)
    os.environ["UTM_SHIELD"] = str(shield)
    return ContinuousUTMEnv(n_drones=scale, scale=scale, max_steps=steps)


def evaluate(arch, ckpt, scale, steps, layers, shield):
    """Deterministic eval; returns (throughput, conflicts, overloads, nfz)."""
    env = _make_env(scale, steps, layers, shield)
    if arch == "gnn":
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ck.get("cfg", {"hidden": 128, "layers": 2, "heads": 4})
        pol = GraphActorCritic(7, 4, 3, cfg["hidden"], cfg["layers"], cfg["heads"])
        pol.load_state_dict(ck["policy"]); pol.eval()
        env.reset(); info = {}
        for _ in range(steps):
            nf, ei, ea = env.get_graph()
            a = pol.act(torch.as_tensor(nf, dtype=torch.float32),
                        torch.as_tensor(ei, dtype=torch.long),
                        torch.as_tensor(ea, dtype=torch.float32), deterministic=True).numpy()
            _, _, _, term, trunc, info = env.step_joint(a)
            if term or trunc:
                break
    else:
        pol = ContinuousActorCritic(env.agent_obs_dim)
        ck = torch.load(ckpt, map_location="cpu")
        pol.load_state_dict(ck["policy"]); pol.eval()
        obs = env.reset()[0]; info = {}
        for _ in range(steps):
            a = pol.act(obs, deterministic=True)
            obs, _, _, term, trunc, info = env.step_joint(a)
            if term or trunc:
                break
    return (info["ep_completed"] / scale, info["ep_conflicts"],
            info.get("ep_overloads", 0), info.get("ep_nfz", 0))


def _ckpt_paths(arch, scale, run_dir):
    sub = "gnn" if arch == "gnn" else "continuous"
    best = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_best.pt")
    final = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_final.pt")
    return best, final


def train(arch, cfgname, scale, seed, updates, steps, epochs, run_dir, threads=0, resume=False):
    os.makedirs(run_dir, exist_ok=True)
    best, final = _ckpt_paths(arch, scale, run_dir)
    # RESUME: if this run already finished (checkpoint present), skip the expensive
    # training and reuse it. Re-evaluation downstream is cheap. This makes the whole
    # study robust to a laptop restart: just re-run the SAME command and it continues.
    if resume and os.path.exists(final):   # final.pt is written ONLY when training completes
        print(f"  [skip ] {arch} {cfgname:9s} seed={seed} (completed, --resume)", flush=True)
        return best if os.path.exists(best) else final
    env = dict(os.environ)
    env.update(CONFIGS[cfgname])
    env["UTM_SCALE"] = str(scale); env["UTM_STEPS"] = str(steps); env["UTM_SHIELD"] = "0"
    if epochs:
        env["UTM_EPOCHS"] = str(epochs)
    if threads:                              # cap CPU so the laptop does not overheat/crash
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
    script = TRAIN_GNN if arch == "gnn" else TRAIN_MLP
    cmd = [sys.executable, "-u", script, "--updates", str(updates), "--seed", str(seed)]
    print(f"  [train] {arch} {cfgname:9s} scale={scale} seed={seed} "
          f"({CONFIGS[cfgname]}) ...", flush=True)
    r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"training failed: {arch} {cfgname} seed={seed}")
    return best if os.path.exists(best) else final


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.5 + conf / 2, df=max(n - 1, 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * sd / math.sqrt(n) if n > 1 else 0.0
    return m, sd, half, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gnn"], default="mlp")
    ap.add_argument("--scale", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--threads", type=int, default=0,
                    help="cap CPU threads (e.g. 2 or 4) so the laptop stays cool and "
                         "usable. 0 = no limit (uses all cores, hotter).")
    ap.add_argument("--resume", action="store_true",
                    help="skip any run whose checkpoint already exists; re-run the SAME "
                         "command after a crash/restart to continue where it stopped.")
    ap.add_argument("--outdir", default=os.path.join(HERE, "ablation_results"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.scale = 40; args.seeds = [0, 1]; args.updates = 5; args.steps = 35
        args.configs = ["full", "no_warn", "sparse"]

    if args.threads:                         # also throttle THIS process's eval torch
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = str(args.threads)
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

    os.makedirs(args.outdir, exist_ok=True)
    raw = os.path.join(args.outdir, "ablation_raw.csv")
    rows = []
    with open(raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "arch", "scale", "seed", "shield",
                    "conflicts", "throughput", "overloads", "nfz"])
        for cfg in args.configs:
            layers = int(CONFIGS[cfg]["UTM_LAYERS"])
            for seed in args.seeds:
                run_dir = os.path.join(args.outdir, f"run_{args.arch}_{cfg}_s{seed}")
                ck = train(args.arch, cfg, args.scale, seed, args.updates,
                           args.steps, args.epochs, run_dir,
                           threads=args.threads, resume=args.resume)
                for shield in (0, 1):
                    tp, cf, ov, nf = evaluate(args.arch, ck, args.scale, args.steps,
                                              layers, shield)
                    print(f"  [eval ] {cfg:9s} seed={seed} shield={shield} -> "
                          f"conflicts={cf} tp={tp:.1%} overloads={ov} nfz={nf}", flush=True)
                    w.writerow([cfg, args.arch, args.scale, seed, shield,
                                cf, f"{tp:.4f}", ov, nf]); f.flush()
                    rows.append((cfg, seed, shield, cf, tp, ov, nf))

    # aggregate per (config, shield)
    summ = os.path.join(args.outdir, "ablation_summary.csv")
    agg = {}
    with open(summ, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "description", "shield", "metric", "mean", "std", "ci95_halfwidth", "n"])
        for cfg in args.configs:
            for shield in (0, 1):
                sel = [r for r in rows if r[0] == cfg and r[2] == shield]
                for metric, idx in [("conflicts", 3), ("throughput", 4), ("overloads", 5), ("nfz", 6)]:
                    vals = [r[idx] for r in sel]
                    if not vals:
                        continue
                    m, sd, half, n = mean_ci(vals)
                    w.writerow([cfg, CONFIG_DESC[cfg], shield, metric,
                                f"{m:.4f}", f"{sd:.4f}", f"{half:.4f}", n])
                    agg[(cfg, shield, metric)] = (m, sd, half, n)

    # formatted table
    tbl = os.path.join(args.outdir, "ablation_table.txt")
    with open(tbl, "w") as f:
        hdr = (f"Ablation study  |  arch={args.arch}  scale={args.scale}  "
               f"seeds={args.seeds}  (mean +/- 95% CI over seeds)\n")
        line = "=" * 92 + "\n"
        f.write(hdr); f.write(line)
        f.write(f"{'configuration':34s} {'conflicts (shield OFF)':>22s} {'conflicts (shield ON)':>22s} {'throughput':>11s}\n")
        f.write("-" * 92 + "\n")
        for cfg in args.configs:
            coff = agg.get((cfg, 0, "conflicts")); con = agg.get((cfg, 1, "conflicts"))
            tp = agg.get((cfg, 0, "throughput"))
            if not coff:
                continue
            f.write(f"{cfg + ' - ' + CONFIG_DESC[cfg]:34.34s} "
                    f"{coff[0]:8.1f} +/- {coff[2]:<7.1f} "
                    f"{(con[0] if con else 0):8.1f} +/- {(con[2] if con else 0):<7.1f} "
                    f"{tp[0]*100:7.1f}%\n")
        f.write(line)
    print("\n" + open(tbl).read())

    # plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cfgs = args.configs
        x = np.arange(len(cfgs)); width = 0.38
        fig, ax = plt.subplots(figsize=(2 + 1.6 * len(cfgs), 4.8))
        for j, shield in enumerate((0, 1)):
            means = [agg.get((c, shield, "conflicts"), (0,0,0,0))[0] for c in cfgs]
            errs = [agg.get((c, shield, "conflicts"), (0,0,0,0))[2] for c in cfgs]
            ax.bar(x + (j - 0.5) * width, means, width, yerr=errs, capsize=5,
                   label=("shield OFF" if shield == 0 else "shield ON"),
                   color=("#C0504D" if shield == 0 else "#1F3864"))
        ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=20, ha="right")
        ax.set_ylabel("Conflicts per episode (deterministic eval)")
        ax.set_title(f"Ablation ({args.arch}, {args.scale} drones): component contributions")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "ablation_conflicts.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("[warn] plotting skipped:", e)

    print("DONE. Wrote:", raw, summ, tbl, "and ablation_conflicts.png in", args.outdir)


if __name__ == "__main__":
    main()