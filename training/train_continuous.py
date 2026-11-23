"""
train_continuous.py
===================
Minimal, clean continuous-control PPO for the ContinuousUTMEnv.
Gaussian (tanh-squashed) policy over [d_heading, d_speed, vertical_rate].
Parameter-shared across all drones. No Lagrangian layer yet (added once this
learns) -- conflict penalty is in the reward for now.

Run:  UTM_SCALE=100 python train_continuous.py --updates 100
"""
import argparse, os, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
try:
    from continuous_utm_env import ContinuousUTMEnv
except ImportError:
    from env.continuous_utm_env import ContinuousUTMEnv

SCALE     = int(os.environ.get("UTM_SCALE", "100"))
N_DRONES  = SCALE
N_UPDATES = 400  # ~convergence; KL-stopping keeps it stable
N_STEPS   = int(os.environ.get('UTM_STEPS', '200'))   # episode length; more steps = more time to avoid AND arrive
N_EPOCHS  = 10
BATCH     = 4096
LR        = 3e-4
GAMMA     = 0.99
LAM       = 0.95
CLIP      = 0.2
ENT_COEF  = 0.0
ACT_DIM   = 3
TARGET_KL  = 0.02        # KL early-stopping threshold (prevents catastrophic steps)
VF_CLIP    = 0.2         # value-function clip range

class ContinuousActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.mu   = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        self.v    = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

    def dist(self, obs):
        h = self.body(obs)
        mu = self.mu(h)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std), self.v(h).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        d, v = self.dist(obs)
        if action is None:
            raw = d.rsample()
            action = torch.tanh(raw)                    # squash to [-1,1]
            logp = d.log_prob(raw).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            return action, logp, d.entropy().sum(-1), v
        # recompute logp for a given (already squashed) action
        raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        logp = d.log_prob(raw).sum(-1) - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, logp, d.entropy().sum(-1), v

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        t = torch.as_tensor(obs, dtype=torch.float32)
        d, _ = self.dist(t)
        a = d.mean if deterministic else d.rsample()
        return torch.tanh(a).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=N_UPDATES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    env = ContinuousUTMEnv(n_drones=N_DRONES, scale=SCALE, max_steps=N_STEPS)
    obs_dim = env.agent_obs_dim
    pol = ContinuousActorCritic(obs_dim).to(device)
    opt = optim.Adam(pol.parameters(), lr=LR, eps=1e-5)

    ckpt_dir = f"./checkpoints_continuous_{N_DRONES}"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_score = -1e9      # selection score (throughput units until first gated ckpt, then conflict-gated)
    have_gated = False     # True once any checkpoint has crossed the throughput gate
    print("="*60)
    print(f"Continuous PPO | drones={N_DRONES} | obs={obs_dim} | act={ACT_DIM}")
    print(f"Checkpoints -> {ckpt_dir}")
    print(f"Updates={args.updates} | rollout={N_STEPS} steps x {N_DRONES} drones")
    print("="*60)

    buf = N_STEPS * N_DRONES

    # --- Adaptive Lagrangian constraint penalties (dual ascent / PID-Lagrangian) ---
    # When UTM_ADAPTIVE=1, the capacity and no-fly penalty weights are not fixed:
    # after each update they are raised when the measured per-episode violation
    # count sits above its budget and lowered when below, driving violations
    # toward the target. Off by default, so existing runs are unchanged.
    ADAPTIVE = os.environ.get("UTM_ADAPTIVE", "0") == "1"
    if ADAPTIVE:
        lam_cap = float(os.environ.get("UTM_CAP_PENALTY", "0.5"))
        lam_nfz = float(os.environ.get("UTM_NFZ_PENALTY", "5.0"))
        cap_target = float(os.environ.get("UTM_CAP_TARGET", "100"))  # per-episode overload budget
        nfz_target = float(os.environ.get("UTM_NFZ_TARGET", "50"))   # per-episode incursion budget
        cap_lr  = float(os.environ.get("UTM_CAP_LR", "0.3"))
        nfz_lr  = float(os.environ.get("UTM_NFZ_LR", "0.3"))
        cap_max = float(os.environ.get("UTM_CAP_MAX", "20.0"))
        nfz_max = float(os.environ.get("UTM_NFZ_MAX", "60.0"))
        ema_over = ema_nfz = None
        lam_hist = []
        print(f"ADAPTIVE constraints ON | cap_target={cap_target} nfz_target={nfz_target} "
              f"| lr=({cap_lr},{nfz_lr}) | max=({cap_max},{nfz_max})")

    for upd in range(1, args.updates+1):
        # linear LR decay -> stabilizes late training, prevents the collapse
        frac = 1.0 - (upd - 1) / args.updates
        for g in opt.param_groups:
            g['lr'] = LR * frac
        # anneal exploration: shrink policy std as training converges -> fewer
        # unlucky sampled actions -> fewer catastrophic single-episode failures.
        with torch.no_grad():
            floor = np.log(0.05)          # minimum std (log-space)
            target = floor + (0.0 - floor) * frac    # start ~0, decay toward floor
            pol.log_std.clamp_(max=target)
        obs_b  = np.zeros((buf, obs_dim), np.float32)
        act_b  = np.zeros((buf, ACT_DIM), np.float32)
        logp_b = np.zeros(buf, np.float32)
        rew_b  = np.zeros(buf, np.float32)
        val_b  = np.zeros(buf, np.float32)
        done_b = np.zeros(buf, np.float32)

        obs, _ = env.reset()
        ep_conf = ep_comp = ep_over = ep_nfz = 0
        for step in range(N_STEPS):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, logp, _, v = pol.get_action_and_value(ot)
            a_np = a.cpu().numpy()
            nobs, r, dones, term, trunc, info = env.step_joint(a_np)
            idx = slice(step*N_DRONES, (step+1)*N_DRONES)
            obs_b[idx]=obs; act_b[idx]=a_np; logp_b[idx]=logp.numpy()
            rew_b[idx]=r; val_b[idx]=v.numpy(); done_b[idx]=dones.astype(float)
            obs = nobs
            ep_conf = info["ep_conflicts"]; ep_comp = info["ep_completed"]
            ep_over = info.get("ep_overloads", 0); ep_nfz = info.get("ep_nfz", 0)
            if term or trunc:
                obs,_ = env.reset()

        # GAE
        adv = np.zeros(buf, np.float32); last=np.zeros(N_DRONES)
        for step in reversed(range(N_STEPS)):
            idx = slice(step*N_DRONES,(step+1)*N_DRONES)
            nxt = np.zeros(N_DRONES) if step==N_STEPS-1 else val_b[(step+1)*N_DRONES:(step+2)*N_DRONES]
            nd  = done_b[idx]
            delta = rew_b[idx] + GAMMA*nxt*(1-nd) - val_b[idx]
            last = delta + GAMMA*LAM*(1-nd)*last
            adv[idx]=last
        ret = adv + val_b
        adv = (adv-adv.mean())/(adv.std()+1e-8)

        ot=torch.as_tensor(obs_b); at=torch.as_tensor(act_b)
        lt=torch.as_tensor(logp_b); rt=torch.as_tensor(ret); advt=torch.as_tensor(adv)
        vt=torch.as_tensor(val_b)
        n=buf
        stop=False
        for _ in range(N_EPOCHS):
            if stop: break
            perm=np.random.permutation(n)
            for s in range(0,n,BATCH):
                mb=perm[s:s+BATCH]
                _,nlp,ent,nv = pol.get_action_and_value(ot[mb], at[mb])
                logratio=nlp-lt[mb]
                ratio=torch.exp(logratio)
                # approximate KL (Schulman); if too large, stop this update -> no catastrophic step
                approx_kl = ((ratio - 1) - logratio).mean().item()
                pg=torch.max(-advt[mb]*ratio, -advt[mb]*torch.clamp(ratio,1-CLIP,1+CLIP)).mean()
                # clipped value loss (stabilizes the critic)
                v_clipped = vt[mb] + torch.clamp(nv - vt[mb], -VF_CLIP, VF_CLIP)
                vf = 0.5*torch.max((nv-rt[mb])**2, (v_clipped-rt[mb])**2).mean()
                loss=pg+0.5*vf-ENT_COEF*ent.mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(),0.5); opt.step()
                if approx_kl > 1.5 * TARGET_KL:
                    stop=True; break   # policy moved too far -> stop before it collapses

        tp = ep_comp / N_DRONES

        # --- dual ascent on the Lagrange multipliers ---
        # EMA smooths the noisy per-episode violation signal; each multiplier
        # moves proportionally to how far the smoothed violation count is above
        # (raise) or below (lower) its budget, then is clipped to [0, max].
        if ADAPTIVE:
            ema_over = ep_over if ema_over is None else 0.8*ema_over + 0.2*ep_over
            ema_nfz  = ep_nfz  if ema_nfz  is None else 0.8*ema_nfz  + 0.2*ep_nfz
            lam_cap = float(np.clip(lam_cap + cap_lr*(ema_over-cap_target)/max(cap_target,1.0), 0.0, cap_max))
            lam_nfz = float(np.clip(lam_nfz + nfz_lr*(ema_nfz -nfz_target)/max(nfz_target,1.0), 0.0, nfz_max))
            os.environ["UTM_CAP_PENALTY"] = f"{lam_cap:.4f}"
            os.environ["UTM_NFZ_PENALTY"] = f"{lam_nfz:.4f}"
            lam_hist.append((upd, lam_cap, lam_nfz, ep_over, ep_nfz))

        # Checkpoint selection with a THROUGHPUT GATE.
        #  - Until some checkpoint reaches TP_GATE throughput, 'best' just tracks
        #    the highest-throughput policy (so best.pt always exists and is sane).
        #  - Once throughput is high, selection switches to CONFLICT-driven: among
        #    high-throughput checkpoints, the one with fewest conflicts wins. The
        #    conflict weight is large enough that a ~50-conflict improvement beats
        #    throughput noise, so best.pt captures the genuinely safe policy rather
        #    than a lucky-throughput, high-conflict one.
        CONF_W  = float(os.environ.get('UTM_SCORE_CONF_W', '0.002'))
        TP_GATE = float(os.environ.get('UTM_SCORE_TP_GATE', '0.95'))
        saved = False
        if tp >= TP_GATE:
            gscore = tp - CONF_W * ep_conf
            if (not have_gated) or (gscore > best_score):
                have_gated = True; best_score = gscore; saved = True
        elif not have_gated:
            if tp > best_score:
                best_score = tp; saved = True
        if saved:
            torch.save({"policy": pol.state_dict(), "update": upd,
                        "throughput": tp, "conflicts": ep_conf, "score": best_score,
                        "gated": have_gated},
                       f"{ckpt_dir}/continuous_best.pt")
        if upd % 5 == 0 or upd == 1:
            _lam = (f" | lam_cap={lam_cap:.2f} lam_nfz={lam_nfz:.2f} "
                    f"over={ep_over} nfz={ep_nfz}") if ADAPTIVE else ""
            print(f"Update {upd:3d}/{args.updates} | conflicts={ep_conf:5d} | "
                  f"completed={ep_comp:3d}/{N_DRONES} | throughput={tp:.1%} | "
                  f"lr={opt.param_groups[0]['lr']:.1e} | best_score={best_score:.3f}{_lam}")
        if upd % 20 == 0:
            torch.save({"policy": pol.state_dict(), "update": upd},
                       f"{ckpt_dir}/continuous_{upd}.pt")

    torch.save({"policy": pol.state_dict(), "update": args.updates},
               f"{ckpt_dir}/continuous_final.pt")
    if ADAPTIVE:
        import csv as _csv
        with open(f"{ckpt_dir}/lambda_trace.csv", "w", newline="") as _f:
            _w = _csv.writer(_f); _w.writerow(["update", "lam_cap", "lam_nfz", "ep_over", "ep_nfz"])
            _w.writerows(lam_hist)
        print(f"Saved multiplier trace -> {ckpt_dir}/lambda_trace.csv "
              f"(final lam_cap={lam_cap:.3f}, lam_nfz={lam_nfz:.3f})")
    print(f"\nSaved: {ckpt_dir}/continuous_final.pt and continuous_best.pt (score={best_score:.3f})")

    # eval (report spread, not just mean -- conflicts swing episode to episode)
    print("\nEvaluating final policy (10 episodes)...")
    confs=[]; tps=[]
    for _ in range(10):
        obs,_=env.reset(); done=False; st=0
        while not done and st<N_STEPS:
            a=pol.act(obs, deterministic=True)
            obs,r,dn,term,trunc,info=env.step_joint(a); done=term or trunc; st+=1
        confs.append(info["ep_conflicts"]); tps.append(info["ep_completed"]/N_DRONES)
    confs=np.array(confs); tps=np.array(tps)
    print(f"  Conflicts/episode: mean={confs.mean():.1f} min={confs.min()} max={confs.max()} std={confs.std():.1f}")
    print(f"  Throughput:        mean={tps.mean():.1%} min={tps.min():.1%}")
    print("  NOTE: continuous_best.pt (conflict-gated) is usually better than final; run eval_best.py.")

if __name__=="__main__":
    main()