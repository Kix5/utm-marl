"""
run_matched_comparison.py
=========================
Matched, reproducible ST-GNN vs MLP comparison for the thesis.

WHY THIS EXISTS
---------------
The proposal reported a preliminary ST-GNN result (15 conflicts) but flagged a
matched comparison as "ongoing". This harness produces that matched comparison:
the SAME environment, the SAME traffic scenario (OD pairs are fixed by
RandomState(42) per scale), and the SAME PPO hyperparameters for both
architectures. The ONLY thing that differs is the policy network (MLP vs
graph-attention), so any difference in conflicts/throughput is attributable to
the architecture, not to luck or tuning.

The honest source of variance is the TRAINING SEED (network initialization and
action sampling). Evaluation is deterministic (greedy policy on the fixed
scenario), so each trained policy yields one reproducible eval number; running
several seeds gives a distribution we can put error bars and a significance test
on.

WHAT IT DOES
------------
For every (architecture, scale, seed):
  1. trains the policy by calling your real trainer as a subprocess
     (train_continuous.py for MLP, train_gnn.py for GNN) with that seed,
  2. loads the conflict-gated best checkpoint it saved,
  3. evaluates it with ONE shared deterministic evaluator (identical for both
     architectures), recording conflicts and throughput.
Then it aggregates to mean, std, and a 95% confidence interval per
(architecture, scale), and runs a Welch t-test (and Mann-Whitney if SciPy is
present) between the two architectures.

OUTPUTS (written to --outdir, default ./comparison_results)
  - results_raw.csv       one row per (arch, scale, seed): conflicts, throughput
  - summary.csv           mean, std, 95% CI, n per (arch, scale)
  - significance.csv      per scale: difference, t-stat, p-values
  - comparison.png        grouped bar chart of conflicts with 95% CI error bars
  - comparison_tp.png     grouped bar chart of throughput with 95% CI error bars

USAGE
-----
  # quick self-test on this machine (tiny, ~2 min, proves the pipeline runs):
  python run_matched_comparison.py --smoke

  # the real run (do this on your training machine; it is the slow part):
  python run_matched_comparison.py \
      --scales 500 --seeds 0 1 2 3 4 \
      --updates 400 --steps 400 --layers 1

  # evaluation-time options:
  --eval-shield 0|1   turn the reactive shield ON during EVALUATION only
                      (default 0; the core architecture comparison is shield-OFF,
                      since the shield drives both architectures to near-zero
                      conflicts and would mask the policy difference). A separate
                      shield-ON pass is useful to show the deployed system.

NOTE ON FAIRNESS OF EPOCHS
--------------------------
train_continuous.py uses 10 PPO epochs/update; train_gnn.py uses 4 (UTM_EPOCHS).
These are each architecture's working settings, not a handicap. If a reviewer
asks, you can equalize with  UTM_EPOCHS=10  for the GNN, or set N_EPOCHS in the
MLP trainer; this harness passes UTM_EPOCHS through so you can match them.
"""
import argparse, os, sys, subprocess, csv, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# locate project files automatically, so this script works whether everything
# sits in one folder OR is split into env/ and training/ subfolders. Searches
# the current working directory first (your project root), then this script's
# own folder. Run the harness from your project root.
# --------------------------------------------------------------------------
_SEARCH_ROOTS = [os.getcwd(), HERE]


def _find(name):
    for root in _SEARCH_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        # prefer a hit NOT inside a results/checkpoints/run folder
        hits = [h for h in hits if "comparison_results" not in h and "/run_" not in h] or hits
        if hits:
            return os.path.abspath(hits[0])
    raise FileNotFoundError(
        f"Could not find {name}. Run this from your project root (the folder "
        f"that contains your env and training files).")


ENV_FILE     = _find("continuous_utm_env.py")
POLICY_FILE  = _find("gnn_policy.py")
TRAIN_MLP    = _find("train_continuous.py")
TRAIN_GNN    = _find("train_gnn.py")
for _d in {os.path.dirname(ENV_FILE), os.path.dirname(POLICY_FILE)}:
    sys.path.insert(0, _d)
print(f"[harness] env     : {ENV_FILE}")
print(f"[harness] policy  : {POLICY_FILE}")
print(f"[harness] mlp     : {TRAIN_MLP}")
print(f"[harness] gnn     : {TRAIN_GNN}")

import torch
from continuous_utm_env import ContinuousUTMEnv
from gnn_policy import GraphActorCritic
# ContinuousActorCritic lives in the MLP trainer module
import importlib.util
_spec = importlib.util.spec_from_file_location("train_continuous", TRAIN_MLP)
_tc = importlib.util.module_from_spec(_spec)
# guard: importing the module must NOT run main()
_spec.loader.exec_module(_tc)
ContinuousActorCritic = _tc.ContinuousActorCritic


# --------------------------------------------------------------------------
# one shared deterministic evaluator (identical for both architectures)
# --------------------------------------------------------------------------
def eval_mlp(ckpt_path, scale, steps, layers, shield):
    env = _make_env(scale, steps, layers, shield)
    pol = ContinuousActorCritic(env.agent_obs_dim)
    ck = torch.load(ckpt_path, map_location="cpu")
    pol.load_state_dict(ck["policy"]); pol.eval()
    obs = env.reset()[0]
    info = {}
    for _ in range(steps):
        a = pol.act(obs, deterministic=True)
        obs, _, _, term, trunc, info = env.step_joint(a)
        if term or trunc:
            break
    return info["ep_completed"] / scale, info["ep_conflicts"]


def eval_gnn(ckpt_path, scale, steps, layers, shield):
    env = _make_env(scale, steps, layers, shield)
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("cfg", {"hidden": 128, "layers": 2, "heads": 4})
    pol = GraphActorCritic(7, 4, 3, cfg["hidden"], cfg["layers"], cfg["heads"])
    pol.load_state_dict(ck["policy"]); pol.eval()
    env.reset()
    info = {}
    for _ in range(steps):
        nf, ei, ea = env.get_graph()
        a = pol.act(torch.as_tensor(nf, dtype=torch.float32),
                    torch.as_tensor(ei, dtype=torch.long),
                    torch.as_tensor(ea, dtype=torch.float32),
                    deterministic=True).numpy()
        _, _, _, term, trunc, info = env.step_joint(a)
        if term or trunc:
            break
    return info["ep_completed"] / scale, info["ep_conflicts"]


def _make_env(scale, steps, layers, shield):
    # env reads UTM_LAYERS / UTM_SHIELD from the environment at step/reset time
    os.environ["UTM_LAYERS"] = str(layers)
    os.environ["UTM_SHIELD"] = str(shield)
    return ContinuousUTMEnv(n_drones=scale, scale=scale, max_steps=steps)


# --------------------------------------------------------------------------
# training (drives your real trainers as subprocesses)
# --------------------------------------------------------------------------
def train_one(arch, scale, seed, updates, steps, layers, epochs, run_dir,
              init_path="", threads=0, resume=False):
    os.makedirs(run_dir, exist_ok=True)
    sub = "gnn" if arch == "gnn" else "continuous"
    best = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_best.pt")
    final = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_final.pt")
    # RESUME: final.pt is written ONLY when training completes, so its presence
    # means this run finished. Skip it and reuse the checkpoint. Re-run the SAME
    # command after a crash/restart and the study continues where it stopped.
    if resume and os.path.exists(final):
        print(f"  [skip ] {arch} scale={scale} seed={seed} layers={layers} (completed, --resume)",
              flush=True)
        return best if os.path.exists(best) else final
    env = dict(os.environ)
    env["UTM_SCALE"] = str(scale)
    env["UTM_STEPS"] = str(steps)
    env["UTM_LAYERS"] = str(layers)
    env["UTM_SHIELD"] = "0"                 # never shield during training
    if epochs:
        env["UTM_EPOCHS"] = str(epochs)     # only the GNN reads this
    if threads:                             # cap CPU so the laptop does not overheat/crash
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
    warm = ""
    if init_path and os.path.exists(init_path):
        env["UTM_INIT"] = init_path         # only the GNN reads this (curriculum warm-start)
        warm = f" warm<-{os.path.basename(os.path.dirname(os.path.dirname(init_path)))}"
    script = TRAIN_GNN if arch == "gnn" else TRAIN_MLP
    cmd = [sys.executable, "-u", script,
           "--updates", str(updates), "--seed", str(seed)]
    # run inside run_dir so checkpoints land there (trainers save to ./checkpoints_*_<N>)
    print(f"  [train] {arch} scale={scale} seed={seed} layers={layers} updates={updates}{warm} ...", flush=True)
    r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"training failed: {arch} scale={scale} seed={seed} layers={layers}")
    return best if os.path.exists(best) else final


# --------------------------------------------------------------------------
# stats helpers (no hard SciPy dependency)
# --------------------------------------------------------------------------
def mean_ci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    # t critical via normal approx fallback; use SciPy if available
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.5 + conf / 2, df=max(n - 1, 1))
    except Exception:
        tcrit = 1.96
    half = tcrit * sd / math.sqrt(n) if n > 1 else 0.0
    return m, sd, half, n


def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    if va == 0 and vb == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / math.sqrt(va / na + vb / nb)
    try:
        from scipy import stats
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        p = float("nan")
    return t, p


def mann_whitney(a, b):
    try:
        from scipy import stats
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return u, p
    except Exception:
        return float("nan"), float("nan")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[500])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--layers", type=int, nargs="+", default=[1],
                    help="airspace mode(s): 1=layered, 0=unlayered. Give both (--layers 0 1) "
                         "to sweep the regime where the GNN should help most (unlayered).")
    ap.add_argument("--epochs", type=int, default=0,
                    help="UTM_EPOCHS for the GNN (0=trainer default of 4). Use 10 to MATCH the "
                         "MLP's 10 PPO epochs for a fair comparison.")
    ap.add_argument("--gnn-curriculum", action="store_true",
                    help="warm-start each GNN run from the previous (smaller) scale's best "
                         "checkpoint of the same seed+layers. Reported as a separate, labeled "
                         "condition; the MLP is unaffected (it trains from scratch at every scale).")
    ap.add_argument("--eval-shield", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0,
                    help="cap CPU threads (e.g. 2 or 4) so the laptop stays cool and usable. "
                         "0 = no limit (uses all cores, hotter).")
    ap.add_argument("--resume", action="store_true",
                    help="skip any run whose training already completed; re-run the SAME "
                         "command after a crash/restart to continue where it stopped.")
    ap.add_argument("--outdir", default=os.path.join(HERE, "comparison_results"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny self-test: scale=30, 2 seeds, few updates")
    args = ap.parse_args()

    if args.smoke:
        args.scales = [20, 30]; args.seeds = [0, 1]
        args.updates = 5; args.steps = 35; args.layers = [1]
        args.gnn_curriculum = True          # exercise the curriculum path too

    if args.threads:                        # also throttle THIS process's eval torch
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = str(args.threads)
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

    scales = sorted(args.scales)            # ascending, so curriculum can chain upward
    os.makedirs(args.outdir, exist_ok=True)
    raw_path = os.path.join(args.outdir, "results_raw.csv")
    rows = []
    # remembers each GNN run's best checkpoint, keyed by (seed, layers), for the
    # curriculum warm-start of the next-larger scale.
    prev_gnn_ckpt = {}
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "scale", "layers", "seed", "conflicts", "throughput", "ckpt"])
        for lay in args.layers:
            for scale in scales:
                for seed in args.seeds:
                    for arch in ["mlp", "gnn"]:
                        run_dir = os.path.join(args.outdir, f"run_{arch}_L{lay}_{scale}_s{seed}")
                        init = ""
                        if arch == "gnn" and args.gnn_curriculum:
                            init = prev_gnn_ckpt.get((seed, lay), "")
                        ck = train_one(arch, scale, seed, args.updates, args.steps,
                                       lay, args.epochs, run_dir, init_path=init,
                                       threads=args.threads, resume=args.resume)
                        if arch == "gnn" and args.gnn_curriculum:
                            prev_gnn_ckpt[(seed, lay)] = ck
                        ev = eval_gnn if arch == "gnn" else eval_mlp
                        tp, cf = ev(ck, scale, args.steps, lay, args.eval_shield)
                        print(f"  [eval ] {arch} L{lay} scale={scale} seed={seed} -> "
                              f"conflicts={cf} throughput={tp:.1%}", flush=True)
                        w.writerow([arch, scale, lay, seed, cf, f"{tp:.4f}", ck]); f.flush()
                        rows.append((arch, scale, lay, seed, cf, tp))

    # ---- aggregate (keyed by arch, scale, layers) ----
    # row indices: 0 arch, 1 scale, 2 layers, 3 seed, 4 conflicts, 5 throughput
    summ_path = os.path.join(args.outdir, "summary.csv")
    sig_path = os.path.join(args.outdir, "significance.csv")
    agg = {}
    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "scale", "layers", "metric", "mean", "std", "ci95_halfwidth", "n"])
        for lay in args.layers:
            for scale in scales:
                for arch in ["mlp", "gnn"]:
                    cfs = [r[4] for r in rows if r[0] == arch and r[1] == scale and r[2] == lay]
                    tps = [r[5] for r in rows if r[0] == arch and r[1] == scale and r[2] == lay]
                    for metric, vals in [("conflicts", cfs), ("throughput", tps)]:
                        if not vals:
                            continue
                        m, sd, half, n = mean_ci(vals)
                        w.writerow([arch, scale, lay, metric, f"{m:.4f}", f"{sd:.4f}", f"{half:.4f}", n])
                        agg[(arch, scale, lay, metric)] = (m, sd, half, n)

    with open(sig_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "layers", "metric", "mlp_mean", "gnn_mean", "diff(gnn-mlp)",
                    "welch_t", "welch_p", "mannwhitney_u", "mannwhitney_p", "winner"])
        for lay in args.layers:
            for scale in scales:
                for metric, idx in [("conflicts", 4), ("throughput", 5)]:
                    a = [r[idx] for r in rows if r[0] == "mlp" and r[1] == scale and r[2] == lay]
                    b = [r[idx] for r in rows if r[0] == "gnn" and r[1] == scale and r[2] == lay]
                    if not a or not b:
                        continue
                    t, p = welch_t(b, a)          # gnn vs mlp
                    u, pu = mann_whitney(b, a)
                    diff = np.mean(b) - np.mean(a)
                    # winner on this metric (fewer conflicts better; higher throughput better)
                    better = ("gnn" if diff < 0 else "mlp") if metric == "conflicts" \
                        else ("gnn" if diff > 0 else "mlp")
                    sig = "" if (not np.isnan(p) and p < 0.05) else " (n.s.)"
                    w.writerow([scale, lay, metric, f"{np.mean(a):.4f}", f"{np.mean(b):.4f}",
                                f"{diff:.4f}", f"{t:.3f}", f"{p:.4g}", f"{u:.1f}", f"{pu:.4g}",
                                better + sig])

    # ---- plots (one pair per layers value) ----
    try:
        _plots(args.outdir, scales, args.layers, agg)
    except Exception as e:
        print("[warn] plotting skipped:", e)

    print("\nDONE. Wrote:", raw_path, summ_path, sig_path, "and plots in", args.outdir)
    _print_table(scales, args.layers, agg)


def _plots(outdir, scales, layers_list, agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for lay in layers_list:
        tag = "layered" if lay == 1 else "unlayered"
        for metric, ylab in [("conflicts", "Conflicts per episode (deterministic eval)"),
                             ("throughput", "Throughput (fraction arrived)")]:
            if not any((arch, s, lay, metric) in agg for arch in ["mlp", "gnn"] for s in scales):
                continue
            fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(scales), 4.5))
            x = np.arange(len(scales)); width = 0.36
            for j, arch in enumerate(["mlp", "gnn"]):
                means = [agg.get((arch, s, lay, metric), (0, 0, 0, 0))[0] for s in scales]
                errs = [agg.get((arch, s, lay, metric), (0, 0, 0, 0))[2] for s in scales]
                ax.bar(x + (j - 0.5) * width, means, width, yerr=errs, capsize=5,
                       label=("MLP (flat PPO)" if arch == "mlp" else "ST-GNN"),
                       color=("#9CA3AF" if arch == "mlp" else "#1F3864"))
            ax.set_xticks(x); ax.set_xticklabels([f"{s}" for s in scales])
            ax.set_xlabel("number of drones")
            ax.set_ylabel(ylab); ax.legend()
            ax.set_title(f"ST-GNN vs MLP, {tag} airspace (error bars = 95% CI over seeds)")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"comparison_{metric}_{tag}.png"), dpi=150)
            plt.close(fig)


def _print_table(scales, layers_list, agg):
    print("\n" + "=" * 74)
    print(f"{'layers':>7} {'scale':>7} {'arch':>5} {'conflicts (mean+/-CI)':>26} {'throughput':>14}")
    print("-" * 74)
    for lay in layers_list:
        for s in scales:
            for arch in ["mlp", "gnn"]:
                if (arch, s, lay, "conflicts") not in agg:
                    continue
                cm, _, ch, n = agg[(arch, s, lay, "conflicts")]
                tm, _, th, _ = agg[(arch, s, lay, "throughput")]
                print(f"{lay:>7} {s:>7} {arch:>5} {cm:>10.1f} +/- {ch:<10.1f} "
                      f"{tm*100:>9.1f}% +/- {th*100:.1f}%")
    print("=" * 74)


if __name__ == "__main__":
    main()
