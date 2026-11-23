#!/usr/bin/env python3
"""
run_reward_ablation.py  (publication-grade)
===========================================
Reward-weight sensitivity ablation for the continuous UTM controller.

It retrains the SAME policy (ContinuousActorCritic) with the SAME PPO setup as
train_continuous.py, changing ONE reward weight at a time away from the chosen
values, over several seeds, and reports three metrics (conflicts/episode,
throughput, minimum separation) at TWO densities (100 and 500, the 500 being
zero-shot). Every configuration is evaluated on the SAME fixed set of held-out
layouts, so the comparison across configs is fair and reproducible.

Chosen weights (the "chosen" baseline row):
    progress = 0.001   arrival = +10.0   conflict = -2.0   warning = -3.0

The training loop is copied verbatim from train_continuous.main() (same
hyperparameters, imported from that file; checkpoint-saving removed, print typo
best_tp->best_score fixed). Each row trains exactly the way your thesis policy
was, differing only in reward weights.

REQUIRES the 2-line patch to continuous_utm_env.py (see PATCH_INSTRUCTIONS.md);
defaults are identical, so thesis behaviour is unchanged.

Usage
-----
Full, publication-grade (train@100, eval zero-shot @100 and @500, 5 seeds,
20 held-out layouts, dense grid). This is what you report:
    python run_reward_ablation.py --updates 400 --seeds 5 \
        --eval-scales 100,500 --eval-layouts 20 --out reward_ablation.csv

Fast preview (shape only, NOT paper numbers):
    python run_reward_ablation.py --updates 60 --seeds 1 --quick \
        --eval-scales 100 --eval-layouts 4 --out preview.csv
"""
import os, sys, csv, time, argparse
import numpy as np
import torch

def _discover_repo_paths():
    """Put YOUR modules on sys.path. continuous_utm_env.py and train_continuous.py
    may live in DIFFERENT folders (e.g. env/ and training/), so we locate each
    one: quick checks first, then a bounded walk of your repo. Set UTM_REPO to the
    repo root to skip the search."""
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    repo = os.environ.get("UTM_REPO", "")
    bases = [b for b in (repo, cwd, here, os.path.dirname(here),
                         os.path.dirname(cwd)) if b]
    needed = ["continuous_utm_env.py", "train_continuous.py"]
    found = {}
    # 1) quick: each base and base/env
    for name in needed:
        for base in bases:
            for d in (base, os.path.join(base, "env")):
                if os.path.isfile(os.path.join(d, name)):
                    found[name] = d
                    break
            if name in found:
                break
    # 2) walk the repo roots for anything still missing
    missing = [n for n in needed if n not in found]
    if missing:
        roots = [repo] if repo else list(dict.fromkeys(
            [os.path.dirname(here), os.path.dirname(cwd), here, cwd]))
        skip = {"__pycache__", "node_modules", "venv", ".venv", "site-packages", ".git"}
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dp, dns, fns in os.walk(root):
                dns[:] = [d for d in dns if not d.startswith(".") and d not in skip]
                for n in list(missing):
                    if n in fns:
                        found[n] = dp
                        missing.remove(n)
                if not missing:
                    break
            if not missing:
                break
    for d in dict.fromkeys(found.values()):
        if d not in sys.path:
            sys.path.insert(0, d)
    still = [n for n in needed if n not in found]
    if still:
        sys.exit("ERROR: could not find %s in your repo. Set "
                 "UTM_REPO=/path/to/your/repo and re-run." % ", ".join(still))
    return found

_discover_repo_paths()

import continuous_utm_env as C
from continuous_utm_env import ContinuousUTMEnv, SEP_H, SEP_V
from train_continuous import (ContinuousActorCritic, GAMMA, LAM, CLIP, N_EPOCHS,
                              BATCH, ENT_COEF, ACT_DIM, TARGET_KL, VF_CLIP, LR, N_STEPS)

CHOSEN = dict(progress=0.001, arrival=10.0, conflict=2.0, warning=3.0)
EVAL_SEED_BASE = 10_000   # held-out layouts: fixed, shared by every config, disjoint from training


def _require_patch():
    missing = [a for a in ("PROG_W", "ARR_BONUS") if not hasattr(C, a)]
    if missing:
        sys.exit("continuous_utm_env.py is not patched: missing %s. See "
                 "PATCH_INSTRUCTIONS.md (defaults identical)." % ", ".join(missing))


def set_weights(progress, arrival, conflict, warning):
    """All four read per step in step_joint: PROG_W/ARR_BONUS/W_WARN as module
    globals, conflict via os.environ."""
    C.PROG_W = float(progress); C.ARR_BONUS = float(arrival); C.W_WARN = float(warning)
    os.environ['UTM_CONF_PENALTY'] = repr(float(conflict))


def _min_horizontal_sep(pos, done):
    """Smallest horizontal distance (m) between any two active drones this step.
    This is the worst-case separation; the episode minimum is the safety number."""
    act = np.where(~done)[0]
    if len(act) < 2:
        return np.inf
    P = pos[act]
    dx = P[:, 0][:, None] - P[:, 0][None, :]
    dy = P[:, 1][:, None] - P[:, 1][None, :]
    dh = np.hypot(dx, dy)
    iu = np.triu_indices(len(act), 1)
    return float(dh[iu].min())


def train_one(n_drones, scale, n_updates, seed):
    """Faithful copy of train_continuous.main()'s training loop."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = ContinuousUTMEnv(n_drones=n_drones, scale=scale, max_steps=N_STEPS)
    obs_dim = env.agent_obs_dim
    pol = ContinuousActorCritic(obs_dim)
    opt = torch.optim.Adam(pol.parameters(), lr=LR, eps=1e-5)
    buf = N_STEPS * n_drones
    for upd in range(1, n_updates + 1):
        frac = 1.0 - (upd - 1) / n_updates
        for g in opt.param_groups:
            g['lr'] = LR * frac
        with torch.no_grad():
            floor = np.log(0.05)
            target = floor + (0.0 - floor) * frac
            pol.log_std.clamp_(max=target)
        obs_b = np.zeros((buf, obs_dim), np.float32); act_b = np.zeros((buf, ACT_DIM), np.float32)
        logp_b = np.zeros(buf, np.float32); rew_b = np.zeros(buf, np.float32)
        val_b = np.zeros(buf, np.float32); done_b = np.zeros(buf, np.float32)
        obs, _ = env.reset()
        for step in range(N_STEPS):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, logp, _, v = pol.get_action_and_value(ot)
            a_np = a.cpu().numpy()
            nobs, r, dones, term, trunc, info = env.step_joint(a_np)
            idx = slice(step * n_drones, (step + 1) * n_drones)
            obs_b[idx] = obs; act_b[idx] = a_np; logp_b[idx] = logp.numpy()
            rew_b[idx] = r; val_b[idx] = v.numpy(); done_b[idx] = dones.astype(float)
            obs = nobs
            if term or trunc:
                obs, _ = env.reset()
        adv = np.zeros(buf, np.float32); last = np.zeros(n_drones)
        for step in reversed(range(N_STEPS)):
            idx = slice(step * n_drones, (step + 1) * n_drones)
            nxt = np.zeros(n_drones) if step == N_STEPS - 1 else val_b[(step + 1) * n_drones:(step + 2) * n_drones]
            nd = done_b[idx]
            delta = rew_b[idx] + GAMMA * nxt * (1 - nd) - val_b[idx]
            last = delta + GAMMA * LAM * (1 - nd) * last
            adv[idx] = last
        ret = adv + val_b
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ot = torch.as_tensor(obs_b); at = torch.as_tensor(act_b)
        lt = torch.as_tensor(logp_b); rt = torch.as_tensor(ret); advt = torch.as_tensor(adv)
        vt = torch.as_tensor(val_b); n = buf; stop = False
        for _ in range(N_EPOCHS):
            if stop: break
            perm = np.random.permutation(n)
            for s in range(0, n, BATCH):
                mb = perm[s:s + BATCH]
                _, nlp, ent, nv = pol.get_action_and_value(ot[mb], at[mb])
                logratio = nlp - lt[mb]; ratio = torch.exp(logratio)
                approx_kl = ((ratio - 1) - logratio).mean().item()
                pg = torch.max(-advt[mb] * ratio, -advt[mb] * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                v_clipped = vt[mb] + torch.clamp(nv - vt[mb], -VF_CLIP, VF_CLIP)
                vf = 0.5 * torch.max((nv - rt[mb]) ** 2, (v_clipped - rt[mb]) ** 2).mean()
                loss = pg + 0.5 * vf - ENT_COEF * ent.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), 0.5); opt.step()
                if approx_kl > 1.5 * TARGET_KL:
                    stop = True; break
    return pol


def evaluate(scale, pol, eval_seeds, track_minsep=True):
    """Deterministic, policy-only eval (shield off) on a FIXED set of held-out
    layouts (eval_seeds). n_drones = scale, so 500 is zero-shot. Returns arrays
    of per-layout conflicts, throughput, and minimum separation (m)."""
    n_drones = scale
    env = ContinuousUTMEnv(n_drones=n_drones, scale=scale, max_steps=N_STEPS)
    confs = []; tps = []; minseps = []
    for sd in eval_seeds:
        obs, _ = env.reset(seed=int(sd)); done = False; st = 0; info = {}
        ep_min = np.inf
        while not done and st < N_STEPS:
            a = pol.act(obs, deterministic=True)
            obs, r, dn, term, trunc, info = env.step_joint(a); done = term or trunc; st += 1
            if track_minsep:
                m = _min_horizontal_sep(env.pos, env._done)
                if m < ep_min:
                    ep_min = m
        confs.append(info.get("ep_conflicts", np.nan))
        tps.append(info.get("ep_completed", np.nan) / n_drones)
        minseps.append(ep_min)
    return np.array(confs, float), np.array(tps, float), np.array(minseps, float)


def build_grid(quick=False):
    """Dense one-at-a-time sweep around the chosen values. 'chosen' is the shared
    baseline; every other row changes exactly one weight."""
    g = [("chosen", dict(CHOSEN))]
    for v in (0.0, 0.5, 1.0, 3.0, 4.0, 6.0):        # conflict penalty (chosen 2.0)
        g.append((f"conflict={v}", {**CHOSEN, "conflict": v}))
    for v in (0.0, 1.0, 1.5, 2.0, 4.0, 6.0):        # warning weight (chosen 3.0)
        g.append((f"warning={v}", {**CHOSEN, "warning": v}))
    for v in (2.5, 5.0, 15.0, 20.0):                # arrival bonus (chosen 10.0)
        g.append((f"arrival={v}", {**CHOSEN, "arrival": v}))
    for v in (0.0, 0.0005, 0.002, 0.004):           # progress weight (chosen 0.001)
        g.append((f"progress={v}", {**CHOSEN, "progress": v}))
    if quick:
        g = [g[0],
             ("conflict=0.0", {**CHOSEN, "conflict": 0.0}),
             ("warning=0.0",  {**CHOSEN, "warning": 0.0})]
    return g


def _group_of(name):
    return "chosen" if name == "chosen" else name.split("=")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5, help="training seeds per config")
    ap.add_argument("--train-drones", type=int, default=100)
    ap.add_argument("--eval-scales", type=str, default="100,500")
    ap.add_argument("--eval-layouts", type=int, default=20,
                    help="number of fixed held-out layouts, shared across all configs")
    ap.add_argument("--only", type=str, default="",
                    help="comma list of weight groups to run (conflict,warning,arrival,progress); "
                         "'chosen' is always included")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in --out and append the rest")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default="reward_ablation.csv")
    args = ap.parse_args()
    eval_scales = [int(x) for x in args.eval_scales.split(",") if x.strip()]
    eval_seeds = [EVAL_SEED_BASE + i for i in range(args.eval_layouts)]
    _require_patch()
    grid = build_grid(args.quick)
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        grid = [(n, w) for (n, w) in grid if n == "chosen" or _group_of(n) in keep]
    rows = []; done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f):
                rows.append(r); done.add(r["config"])
        print(f"[resume] {len(done)} configs already in {args.out}, skipping them.")
    todo = [(n, w) for (n, w) in grid if n not in done]
    print(f"Configs: {len(grid)} ({len(todo)} to run) | seeds: {args.seeds} | "
          f"runs left: {len(todo) * args.seeds}")
    print(f"updates: {args.updates} | train@{args.train_drones} | "
          f"eval@{eval_scales} on {args.eval_layouts} fixed held-out layouts")
    for name, w in todo:
        acc = {s: {"conf": [], "tp": [], "msep": []} for s in eval_scales}
        for seed in range(args.seeds):
            set_weights(**w)
            t0 = time.time()
            pol = train_one(args.train_drones, args.train_drones, args.updates, seed)
            msg = []
            for s in eval_scales:
                conf, tp, msep = evaluate(s, pol, eval_seeds)
                acc[s]["conf"].append(np.nanmean(conf))
                acc[s]["tp"].append(np.nanmean(tp))
                acc[s]["msep"].append(np.nanmean(msep))
                msg.append(f"@{s}: conf={np.nanmean(conf):.1f} tp={np.nanmean(tp):.0%} "
                           f"minsep={np.nanmean(msep):.0f}m")
            print(f"  [{name}] seed {seed}: " + "  ".join(msg) + f"  ({time.time()-t0:.0f}s)")
        row = dict(config=name, progress=w["progress"], arrival=w["arrival"],
                   conflict=w["conflict"], warning=w["warning"], seeds=args.seeds,
                   updates=args.updates, eval_layouts=args.eval_layouts)
        for s in eval_scales:
            c = np.array(acc[s]["conf"]); t = np.array(acc[s]["tp"]); m = np.array(acc[s]["msep"])
            row[f"conflicts_mean_s{s}"] = float(c.mean()); row[f"conflicts_std_s{s}"] = float(c.std())
            row[f"throughput_mean_s{s}"] = float(t.mean()); row[f"throughput_std_s{s}"] = float(t.std())
            row[f"minsep_mean_s{s}"] = float(m.mean()); row[f"minsep_std_s{s}"] = float(m.std())
        rows.append(row)
        summ = " | ".join(
            f"@{s} conf {row[f'conflicts_mean_s{s}']:.1f}+/-{row[f'conflicts_std_s{s}']:.1f}"
            for s in eval_scales)
        print(f"==> {name:14s}: " + summ)
        # write incrementally so a long run is never lost; fieldnames from a
        # freshly computed row so resumed (string) rows can't dictate a stale header
        with open(args.out, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            wri.writeheader(); wri.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} configs, eval scales {eval_scales}, "
          f"{args.seeds} seeds, {args.eval_layouts} held-out layouts).")


if __name__ == "__main__":
    main()
