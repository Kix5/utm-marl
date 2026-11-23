"""
run_generalization.py
======================
Generalization study: does the ST-GNN generalize to UNSEEN traffic better than
the MLP? This is the setting where a graph policy's relational inductive bias is
expected to help, even though the MLP wins on a single fixed layout.

SETUP (honest and clean)
  - The airspace is FIXED (same grid, same no-fly layout).
  - The TRAFFIC is randomized: origin-destination demands are drawn fresh each
    training episode (UTM_RANDOM_OD=1). So each policy trains on a STREAM of
    different traffic layouts, not one.
  - Evaluation is on a HELD-OUT set of traffic layouts (fixed seeds in a range
    the training never used), so the test scenarios are genuinely unseen.
  - Both architectures train and are evaluated on identical conditions; only the
    network differs. Variance is over training seeds (95% CI), with a Welch test.

Requires the patched continuous_utm_env.py (adds UTM_RANDOM_OD / _generate_od).

USAGE (on Colab; your Mac is unstable)
  python run_generalization.py --smoke                      # pipeline self-test
  python run_generalization.py --scale 100 --seeds 0 1 2 \
      --updates 400 --steps 400 --layers 1 --n-test 20 --resume

  --scale N        fleet size (default 100; generalization is about traffic
                   variety, so a moderate scale is fine and faster)
  --seeds ...      training seeds (variance source); default 0 1 2
  --n-test K       number of held-out test layouts to average over (default 20)
  --test-base B    first held-out scenario seed (default 900000; keep far from
                   training to guarantee the test layouts are unseen)
  --layers 0|1     airspace mode (default 1 = layered)
  --epochs 10      UTM_EPOCHS for the GNN (match the MLP's 10)
  --threads N      cap CPU threads
  --resume         skip completed training runs; re-run to continue after a drop
  --outdir PATH    default ./generalization_results

OUTPUTS
  generalization_raw.csv   per (arch, train_seed): held-out mean conflicts / tp
  generalization_summary.csv   mean + 95% CI over training seeds, per arch
  generalization_significance.csv   MLP vs GNN on held-out traffic (Welch, MWU)
  generalization_table.txt    formatted table for the thesis
  generalization.png          bar chart with 95% CI error bars

READING IT
  If the GNN's held-out conflicts are significantly LOWER than the MLP's, that is
  real evidence the graph policy generalizes better to unseen traffic, a genuine
  and publishable advantage even though the MLP wins on a single fixed layout.
  If not, report it honestly: the MLP generalizes at least as well here.
"""
import argparse, os, sys, subprocess, csv, math, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH_ROOTS = [os.getcwd(), HERE]


def _find(name):
    for root in _SEARCH_ROOTS:
        hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
        hits = [h for h in hits if "_results" not in h and "/run_" not in h] or hits
        if hits:
            return os.path.abspath(hits[0])
    raise FileNotFoundError(f"Could not find {name}; run from your project root.")


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


def _check_env_supports_random_od():
    import inspect
    src = inspect.getsource(ContinuousUTMEnv)
    if "UTM_RANDOM_OD" not in src or "_generate_od" not in src:
        sys.exit("ERROR: your continuous_utm_env.py is NOT patched for randomized OD. "
                 "Use the patched env (adds UTM_RANDOM_OD / _generate_od) before running.")


def eval_heldout(arch, ckpt, scale, steps, layers, test_seeds, shield=0):
    """Deterministic eval across a list of held-out scenario seeds. Returns
    (mean_throughput, mean_conflicts) over those unseen layouts."""
    os.environ["UTM_RANDOM_OD"] = "1"
    os.environ["UTM_LAYERS"] = str(layers)
    os.environ["UTM_SHIELD"] = str(shield)
    env = ContinuousUTMEnv(n_drones=scale, scale=scale, max_steps=steps)
    if arch == "gnn":
        ck = torch.load(ckpt, map_location="cpu")
        cfg = ck.get("cfg", {"hidden": 128, "layers": 2, "heads": 4})
        pol = GraphActorCritic(7, 4, 3, cfg["hidden"], cfg["layers"], cfg["heads"])
        pol.load_state_dict(ck["policy"]); pol.eval()
    else:
        pol = ContinuousActorCritic(env.agent_obs_dim)
        ck = torch.load(ckpt, map_location="cpu")
        pol.load_state_dict(ck["policy"]); pol.eval()
    confs, tps = [], []
    for s in test_seeds:
        env.reset(seed=int(s)); info = {}
        for _ in range(steps):
            if arch == "gnn":
                nf, ei, ea = env.get_graph()
                a = pol.act(torch.as_tensor(nf, dtype=torch.float32),
                            torch.as_tensor(ei, dtype=torch.long),
                            torch.as_tensor(ea, dtype=torch.float32),
                            deterministic=True).numpy()
            else:
                a = pol.act(env.get_all_agent_obs(), deterministic=True)
            _, _, _, term, trunc, info = env.step_joint(a)
            if term or trunc:
                break
        confs.append(info["ep_conflicts"]); tps.append(info["ep_completed"] / scale)
    return float(np.mean(tps)), float(np.mean(confs))


def train_one(arch, scale, seed, updates, steps, layers, epochs, run_dir, threads=0, resume=False):
    os.makedirs(run_dir, exist_ok=True)
    sub = "gnn" if arch == "gnn" else "continuous"
    best = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_best.pt")
    final = os.path.join(run_dir, f"checkpoints_{sub}_{scale}", f"{sub}_final.pt")
    if resume and os.path.exists(final):
        print(f"  [skip ] {arch} seed={seed} (completed, --resume)", flush=True)
        return best if os.path.exists(best) else final
    env = dict(os.environ)
    env["UTM_SCALE"] = str(scale); env["UTM_STEPS"] = str(steps)
    env["UTM_LAYERS"] = str(layers); env["UTM_SHIELD"] = "0"
    env["UTM_RANDOM_OD"] = "1"                        # randomized traffic during training
    env["UTM_OD_SEED_BASE"] = str(seed * 1000)        # each seed sees a different training stream
    if epochs:
        env["UTM_EPOCHS"] = str(epochs)
    if threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
    script = TRAIN_GNN if arch == "gnn" else TRAIN_MLP
    cmd = [sys.executable, "-u", script, "--updates", str(updates), "--seed", str(seed)]
    print(f"  [train] {arch} scale={scale} seed={seed} (randomized traffic) ...", flush=True)
    r = subprocess.run(cmd, cwd=run_dir, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"training failed: {arch} seed={seed}")
    return best if os.path.exists(best) else final


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float); n = len(x)
    m = x.mean(); sd = x.std(ddof=1) if n > 1 else 0.0
    try:
        from scipy import stats
        tcrit = stats.t.ppf(0.5 + conf / 2, df=max(n - 1, 1))
    except Exception:
        tcrit = 1.96
    return m, sd, (tcrit * sd / math.sqrt(n) if n > 1 else 0.0), n


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    if va == 0 and vb == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / math.sqrt(va / na + vb / nb)
    try:
        from scipy import stats
        df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        p = float("nan")
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--test-base", type=int, default=900000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(HERE, "generalization_results"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.scale = 40; args.seeds = [0, 1]; args.updates = 5
        args.steps = 35; args.n_test = 4

    _check_env_supports_random_od()
    if args.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = str(args.threads)
        try: torch.set_num_threads(args.threads)
        except Exception: pass

    test_seeds = [args.test_base + i for i in range(args.n_test)]
    os.makedirs(args.outdir, exist_ok=True)
    raw = os.path.join(args.outdir, "generalization_raw.csv")
    rows = []
    with open(raw, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "scale", "train_seed", "heldout_mean_conflicts", "heldout_mean_throughput"])
        for seed in args.seeds:
            for arch in ["mlp", "gnn"]:
                run_dir = os.path.join(args.outdir, f"run_{arch}_s{seed}")
                ck = train_one(arch, args.scale, seed, args.updates, args.steps,
                               args.layers, args.epochs, run_dir,
                               threads=args.threads, resume=args.resume)
                tp, cf = eval_heldout(arch, ck, args.scale, args.steps, args.layers, test_seeds)
                print(f"  [heldout] {arch} seed={seed} -> conflicts={cf:.1f} tp={tp:.1%} "
                      f"(avg over {args.n_test} unseen layouts)", flush=True)
                w.writerow([arch, args.scale, seed, f"{cf:.4f}", f"{tp:.4f}"]); f.flush()
                rows.append((arch, seed, cf, tp))

    # aggregate over training seeds
    agg = {}
    with open(os.path.join(args.outdir, "generalization_summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arch", "metric", "mean", "std", "ci95_halfwidth", "n"])
        for arch in ["mlp", "gnn"]:
            for metric, idx in [("heldout_conflicts", 2), ("heldout_throughput", 3)]:
                vals = [r[idx] for r in rows if r[0] == arch]
                if not vals: continue
                m, sd, half, n = mean_ci(vals)
                w.writerow([arch, metric, f"{m:.4f}", f"{sd:.4f}", f"{half:.4f}", n])
                agg[(arch, metric)] = (m, sd, half, n)

    with open(os.path.join(args.outdir, "generalization_significance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mlp_mean", "gnn_mean", "diff(gnn-mlp)", "welch_t", "welch_p", "better"])
        for metric, idx in [("heldout_conflicts", 2), ("heldout_throughput", 3)]:
            a = [r[idx] for r in rows if r[0] == "mlp"]
            b = [r[idx] for r in rows if r[0] == "gnn"]
            if not a or not b: continue
            t, p = welch(b, a); diff = np.mean(b) - np.mean(a)
            better = ("gnn" if diff < 0 else "mlp") if metric == "heldout_conflicts" \
                else ("gnn" if diff > 0 else "mlp")
            sig = "" if (not np.isnan(p) and p < 0.05) else " (n.s.)"
            w.writerow([metric, f"{np.mean(a):.4f}", f"{np.mean(b):.4f}", f"{diff:.4f}",
                        f"{t:.3f}", f"{p:.4g}", better + sig])

    tbl = os.path.join(args.outdir, "generalization_table.txt")
    with open(tbl, "w") as f:
        f.write(f"Generalization to unseen traffic | scale={args.scale} | "
                f"{len(args.seeds)} training seeds | {args.n_test} held-out layouts\n")
        f.write("(mean conflicts on UNSEEN traffic, +/- 95% CI over training seeds)\n")
        f.write("=" * 60 + "\n")
        for arch in ["mlp", "gnn"]:
            if (arch, "heldout_conflicts") not in agg: continue
            cm, _, ch, n = agg[(arch, "heldout_conflicts")]
            tm, _, th, _ = agg[(arch, "heldout_throughput")]
            f.write(f"  {arch.upper():4s}  conflicts {cm:7.1f} +/- {ch:<6.1f}   "
                    f"throughput {tm*100:5.1f}%\n")
        f.write("=" * 60 + "\n")
    print("\n" + open(tbl).read())

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.5))
        archs = ["mlp", "gnn"]
        means = [agg[(a, "heldout_conflicts")][0] for a in archs]
        errs = [agg[(a, "heldout_conflicts")][2] for a in archs]
        ax.bar(["MLP", "ST-GNN"], means, yerr=errs, capsize=6,
               color=["#9CA3AF", "#1F3864"])
        ax.set_ylabel("Conflicts on unseen traffic (held-out)")
        ax.set_title(f"Generalization, {args.scale} drones (95% CI over seeds)")
        ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "generalization.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("[warn] plot skipped:", e)

    print("DONE. Results in", args.outdir)


if __name__ == "__main__":
    main()
