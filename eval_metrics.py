"""
eval_metrics.py
===============
Compute a full performance-metrics suite for a trained UTM policy (MLP or GNN),
for the thesis Results chapter. Evaluates one checkpoint over one or more
scenarios and reports safety, efficiency, constraint, real-time, and stability
metrics.

METRICS
  Safety
    conflicts_per_episode     mean total separation violations per episode
    conflicts_per_drone       conflicts_per_episode / n_drones
    los_free_rate             fraction of drones never in conflict (approx: 1 - involved/n)
    min_separation_m          smallest 3D distance reached between any two drones (m)
  Efficiency / throughput
    completion_rate           fraction of drones that reached their goal
    mean_flight_time_s        mean time to reach goal, over arrived drones (s)
    path_efficiency           mean( ideal straight-line dist / actual path length )  in (0,1], higher better
    mean_delay_s              mean( actual flight time - ideal time ) over arrived drones (s)
  Constraints
    sector_overloads          mean per-episode count of drones over sector capacity
    nofly_incursions          mean per-episode count of drones inside a no-fly sector
  Real-time
    inference_ms_per_step     mean wall-clock time of one policy decision for the whole fleet (ms)
  (variance across scenarios is reported as +/- standard deviation)

USAGE (Colab or local)
  # auto-find the best checkpoint for an arch at a scale, fixed scenario:
  python eval_metrics.py --arch mlp --scale 100 --layers 1

  # evaluate on held-out RANDOM traffic (needs the patched env), shield on/off:
  python eval_metrics.py --arch gnn --scale 100 --layers 1 \
      --random-od --test-base 900000 --n-eval 20 --shield 0

  --ckpt PATH     explicit checkpoint (else auto-find checkpoints_{sub}_{scale}/..best/final)
  --shield 0|1    evaluate with the reactive shield off (policy only) or on (deployed)
  --random-od     draw held-out random traffic (patched env); else the fixed scenario
  --n-eval K      number of scenarios to average over (random-od); default 1 for fixed
  --out PATH      write a CSV row of metrics (appends if exists)
"""
import argparse, os, sys, glob, time, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for root in [os.getcwd(), HERE]:
    for d in ("", "env", "training", "stgnn_bundle"):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            sys.path.insert(0, p)

import torch
from continuous_utm_env import ContinuousUTMEnv, SEP_H, SEP_V, DT
from gnn_policy import GraphActorCritic
import importlib.util
_tc_path = (glob.glob(os.path.join(os.getcwd(), "**", "train_continuous.py"), recursive=True)
            or glob.glob(os.path.join(HERE, "**", "train_continuous.py"), recursive=True))[0]
_spec = importlib.util.spec_from_file_location("train_continuous", _tc_path)
_tc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tc)
ContinuousActorCritic = _tc.ContinuousActorCritic


def load_policy(arch, ckpt, obs_dim):
    ck = torch.load(ckpt, map_location="cpu")
    if arch == "gnn":
        cfg = ck.get("cfg", {"hidden": 128, "layers": 2, "heads": 4})
        pol = GraphActorCritic(7, 4, 3, cfg["hidden"], cfg["layers"], cfg["heads"])
    else:
        pol = ContinuousActorCritic(obs_dim)
    pol.load_state_dict(ck["policy"]); pol.eval()
    return pol


def rollout_metrics(arch, pol, env, steps):
    """One episode; returns a dict of metrics for this scenario."""
    N = env.n_drones
    obs = env.reset()[0] if arch != "gnn" else (env.reset(), env.get_all_agent_obs())[1]
    start_xy = env.pos[:, :2].copy()
    goal_xy = env.goal.copy()
    ideal_dist = np.linalg.norm(goal_xy - start_xy, axis=1)           # straight-line need
    prev_xy = env.pos[:, :2].copy()
    prev_done = env._done.copy()
    path_len = np.zeros(N)
    arrival_step = np.full(N, -1)
    # Safety margin: minimum HORIZONTAL separation among CO-ALTITUDE pairs, i.e.
    # pairs that are within the vertical minimum (SEP_V) of each other and could
    # therefore actually conflict. This is the meaningful "how close did drones
    # that share a flight level get horizontally" and is NOT distorted by pairs
    # on different levels (which are vertically separated and safe).
    min_hco = np.inf
    t_inf = 0.0
    info = {}
    for step in range(steps):
        t0 = time.perf_counter()
        if arch == "gnn":
            nf, ei, ea = env.get_graph()
            a = pol.act(torch.as_tensor(nf, dtype=torch.float32),
                        torch.as_tensor(ei, dtype=torch.long),
                        torch.as_tensor(ea, dtype=torch.float32), deterministic=True).numpy()
        else:
            a = pol.act(env.get_all_agent_obs(), deterministic=True)
        t_inf += time.perf_counter() - t0
        _, _, _, term, trunc, info = env.step_joint(a)
        # path length (horizontal) for drones that were active before this step
        now_xy = env.pos[:, :2]
        disp = np.linalg.norm(now_xy - prev_xy, axis=1)
        active_before = ~prev_done
        path_len[active_before] += disp[active_before]
        prev_xy = now_xy.copy()
        # arrivals
        newly = env._done & (~prev_done)
        arrival_step[newly & (arrival_step < 0)] = step + 1
        prev_done = env._done.copy()
        # minimum horizontal separation among CO-ALTITUDE pairs (vertical gap < SEP_V)
        act = np.where(~env._done)[0]
        if len(act) > 1:
            P = env.pos[act]
            dh = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=-1)
            dv = np.abs(P[:, None, 2] - P[None, :, 2])
            coalt = dv < SEP_V            # pairs sharing (nearly) the same altitude
            np.fill_diagonal(coalt, False)
            if coalt.any():
                min_hco = min(min_hco, float(dh[coalt].min()))
        if term or trunc:
            break
    nsteps = step + 1
    arrived = arrival_step > 0
    comp = env._done.copy()
    # efficiency over arrived drones
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where((path_len > 1) & arrived, ideal_dist / path_len, np.nan)
        eff = np.minimum(eff, 1.0)   # cap at 1 (arrival tolerance can make path < straight line)
    flight_time = np.where(arrived, arrival_step * DT, np.nan)
    confs = info.get("ep_conflicts", 0)
    return {
        "conflicts_per_episode": confs,
        "conflicts_per_drone": confs / N,
        "completion_rate": info.get("ep_completed", comp.sum()) / N,
        "min_horiz_sep_coalt_m": (min_hco if np.isfinite(min_hco) else np.nan),
        "mean_flight_time_s": np.nanmean(flight_time) if arrived.any() else np.nan,
        "path_efficiency": np.nanmean(eff) if arrived.any() else np.nan,
        "sector_overloads": info.get("ep_overloads", 0),
        "nofly_incursions": info.get("ep_nfz", 0),
        "inference_ms_per_step": 1000.0 * t_inf / nsteps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "gnn"], required=True)
    ap.add_argument("--scale", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--shield", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random-od", action="store_true")
    ap.add_argument("--test-base", type=int, default=900000)
    ap.add_argument("--n-eval", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    os.environ["UTM_LAYERS"] = str(args.layers)
    os.environ["UTM_SHIELD"] = str(args.shield)
    os.environ["UTM_RANDOM_OD"] = "1" if args.random_od else "0"

    env = ContinuousUTMEnv(n_drones=args.scale, scale=args.scale, max_steps=args.steps)
    sub = "gnn" if args.arch == "gnn" else "continuous"
    ckpt = args.ckpt
    if not ckpt:
        cands = (glob.glob(f"**/checkpoints_{sub}_{args.scale}/{sub}_best.pt", recursive=True)
                 or glob.glob(f"**/checkpoints_{sub}_{args.scale}/{sub}_final.pt", recursive=True))
        assert cands, f"no checkpoint found for {args.arch} scale={args.scale}; pass --ckpt"
        ckpt = cands[0]
    print("checkpoint:", ckpt)
    pol = load_policy(args.arch, ckpt, env.agent_obs_dim)

    n_eval = args.n_eval if args.random_od else 1
    runs = []
    for i in range(n_eval):
        if args.random_od:
            # force the held-out scenario seed for this eval
            env.reset(seed=args.test_base + i)
        runs.append(rollout_metrics(args.arch, pol, env, args.steps))

    keys = list(runs[0].keys())
    agg = {k: (np.nanmean([r[k] for r in runs]), np.nanstd([r[k] for r in runs])) for k in keys}

    print("\n" + "=" * 62)
    print(f"Performance metrics | {args.arch.upper()} | scale={args.scale} | "
          f"layers={args.layers} | shield={args.shield} | "
          f"{'held-out '+str(n_eval)+' scenarios' if args.random_od else 'fixed scenario'}")
    print("=" * 62)
    units = {"min_horiz_sep_coalt_m": "m",
             "mean_flight_time_s": "s", "inference_ms_per_step": "ms",
             "completion_rate": "", "path_efficiency": ""}
    for k in keys:
        m, s = agg[k]
        tail = f" +/- {s:.2f}" if n_eval > 1 else ""
        print(f"  {k:24s} {m:10.3f}{tail} {units.get(k,'')}")
    print("=" * 62)

    if args.out:
        head = ["arch", "scale", "layers", "shield", "random_od", "n_eval"] + keys
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(head)
            w.writerow([args.arch, args.scale, args.layers, args.shield,
                        int(args.random_od), n_eval] + [f"{agg[k][0]:.4f}" for k in keys])
        print("wrote", args.out)


if __name__ == "__main__":
    main()
