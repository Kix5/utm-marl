"""
train_constrained_marl.py
=========================
Constrained Multi-Agent RL using Lagrangian PPO.

Key difference from vanilla MAPPO:
  - Two Lagrange multipliers: λ₁ (sector overload), λ₂ (conflict rate)
  - Augmented reward: r̃ = r - λ₁·C₁ - λ₂·C₂
  - Multiplier update: λᵢ ← max(0, λᵢ + α_λ · Cᵢ(θ))
  - Hard constraint budgets: b_cap = 0, b_sep = 0

Expected result vs vanilla MAPPO:
  Vanilla MAPPO:      100% throughput, ~2 conflicts/ep
  Constrained MARL:   100% throughput, ~0 conflicts/ep ← main contribution

Usage:
    python training/train_constrained_marl.py
    python training/train_constrained_marl.py --timesteps 100000
    python training/train_constrained_marl.py --compare   # compare vs vanilla
"""

import argparse, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "env"))

from env.bluesky_utm_env import BlueSkyUTMEnv

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

_SCALE     = int(os.environ.get("UTM_SCALE", "5"))
# ── Hyperparameters ────────────────────────────────────────────────────────
N_DRONES    = _SCALE
TOTAL_STEPS = 200_000
N_STEPS     = 256
N_EPOCHS    = 8
BATCH_SIZE  = 128
LR          = 3e-4
GAMMA       = 0.99
GAE_LAMBDA  = 0.95
CLIP_RANGE  = 0.2
ENT_COEF    = 0.01

# ── Lagrangian hyperparameters ─────────────────────────────────────────────
LR_LAMBDA   = 0.01    # dual learning rate α_λ
B_CAP       = 0.0     # sector overload budget (0 = zero tolerance)
B_SEP       = 0.0     # conflict budget (0 = zero tolerance)
LAMBDA_INIT = 0.5     # initial multiplier value
LAMBDA_MAX  = 10.0    # clip multipliers to prevent explosion

LOG_DIR    = f"./logs_constrained_{_SCALE}"
CKPT_DIR   = f"./checkpoints_constrained_{_SCALE}"


# ══════════════════════════════════════════════════════════════════════════
# SHARED ACTOR-CRITIC (identical architecture to vanilla MAPPO)
# ══════════════════════════════════════════════════════════════════════════

class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs):
        h = self.shared(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        dist  = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        t = torch.FloatTensor(obs).unsqueeze(0).to(
            next(self.parameters()).device)
        logits, _ = self(t)
        if deterministic:
            return int(logits.argmax().item())
        return int(Categorical(logits=logits).sample().item())


# ══════════════════════════════════════════════════════════════════════════
# LAGRANGE MULTIPLIERS
# ══════════════════════════════════════════════════════════════════════════

class LagrangeMultipliers:
    """
    Adaptive Lagrange multipliers for two constraints:
      C1: sector overload  (λ₁)
      C2: conflict rate    (λ₂)

    Update rule:
      λᵢ ← clip(λᵢ + α_λ · (Cᵢ - bᵢ), 0, λ_max)
    """
    def __init__(self, lr: float = LR_LAMBDA,
                 init: float = LAMBDA_INIT,
                 lmax: float = LAMBDA_MAX):
        self.lr   = lr
        self.lmax = lmax
        self.lam1 = init   # sector overload multiplier
        self.lam2 = init   # conflict rate multiplier
        self.history = {"lam1": [], "lam2": []}

    def update(self, c1: float, c2: float):
        """
        c1 = mean sector overload per step (constraint cost)
        c2 = mean conflict count per step  (constraint cost)
        """
        self.lam1 = float(np.clip(
            self.lam1 + self.lr * (c1 - B_CAP), 0, self.lmax))
        self.lam2 = float(np.clip(
            self.lam2 + self.lr * (c2 - B_SEP), 0, self.lmax))
        self.history["lam1"].append(self.lam1)
        self.history["lam2"].append(self.lam2)

    def augment_reward(self, rewards: np.ndarray,
                       overloads: np.ndarray,
                       conflicts: np.ndarray) -> np.ndarray:
        """
        Augmented reward: r̃ = r - λ₁·C₁ - λ₂·C₂
        Shapes: rewards (N,), overloads (N,), conflicts (N,)
        """
        return rewards - self.lam1 * overloads - self.lam2 * conflicts

    def state_dict(self):
        return {"lam1": self.lam1, "lam2": self.lam2}

    def load_state_dict(self, d):
        self.lam1 = d["lam1"]
        self.lam2 = d["lam2"]


# ══════════════════════════════════════════════════════════════════════════
# CONSTRAINED MARL ENVIRONMENT WRAPPER
# ══════════════════════════════════════════════════════════════════════════

class ConstrainedUTMEnv(BlueSkyUTMEnv):
    """
    Extends BlueSkyUTMEnv to return per-agent constraint costs
    alongside rewards — needed for Lagrangian augmented rewards.
    """

    def step_joint(self, actions):
        all_obs, rewards, dones, terminated, truncated, info = \
            super().step_joint(actions)

        # Per-agent constraint costs
        n = self.n_drones
        overloads = np.full(n, info.get("overload_count", 0) , dtype=np.float32)
        conflicts = np.full(n, info.get("conflict_count", 0) , dtype=np.float32)
        info["overloads_per_agent"] = overloads
        info["conflicts_per_agent"] = conflicts
        return all_obs, rewards, dones, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════

class Metrics:
    def __init__(self, window: int = 50):
        self.rewards     = deque(maxlen=window)
        self.conflicts   = deque(maxlen=window)
        self.overloads   = deque(maxlen=window)
        self.throughput  = deque(maxlen=window)
        self.lam1_hist   = deque(maxlen=window)
        self.lam2_hist   = deque(maxlen=window)
        self.ep_count    = 0

    def record(self, reward, conflicts, overloads, completed,
               n_drones, lam1, lam2):
        self.ep_count   += 1
        self.rewards.append(reward)
        self.conflicts.append(conflicts)
        self.overloads.append(overloads)
        self.throughput.append(completed / max(n_drones, 1))
        self.lam1_hist.append(lam1)
        self.lam2_hist.append(lam2)

    def mean(self, q):
        return float(np.mean(list(q))) if q else 0.0

    def print(self, prefix=""):
        print(
            f"{prefix}Ep {self.ep_count:4d} | "
            f"Reward {self.mean(self.rewards):+7.2f} | "
            f"Throughput {self.mean(self.throughput):.1%} | "
            f"Conflicts {self.mean(self.conflicts):.2f} | "
            f"Overloads {self.mean(self.overloads):.2f} | "
            f"λ₁={self.mean(self.lam1_hist):.3f} "
            f"λ₂={self.mean(self.lam2_hist):.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════
# CONSTRAINED MARL TRAINER
# ══════════════════════════════════════════════════════════════════════════

class ConstrainedMARLTrainer:

    def __init__(self, render: bool = False, device: str = "cpu"):
        self.device  = torch.device(device)
        self.metrics = Metrics()
        self.lagrange = LagrangeMultipliers()

        self.env = ConstrainedUTMEnv(
            n_drones    = N_DRONES,
            render_mode = "none" , 
            #render_mode = "human" if render else "none",
            max_steps   = 200
        )

        obs_dim   = self.env.agent_obs_dim
        n_actions = self.env.n_sectors

        self.policy = SharedActorCritic(obs_dim, n_actions).to(self.device)
        self.optim  = optim.Adam(
            self.policy.parameters(), lr=LR, eps=1e-5)

        # Rollout buffers
        buf = N_STEPS * N_DRONES
        self.buf_obs      = np.zeros((buf, obs_dim),  dtype=np.float32)
        self.buf_actions  = np.zeros(buf,             dtype=np.int64)
        self.buf_rewards  = np.zeros(buf,             dtype=np.float32)  # augmented
        self.buf_raw_rew  = np.zeros(buf,             dtype=np.float32)  # original
        self.buf_dones    = np.zeros(buf,             dtype=np.float32)
        self.buf_values   = np.zeros(buf,             dtype=np.float32)
        self.buf_logprobs = np.zeros(buf,             dtype=np.float32)
        self.buf_advs     = np.zeros(buf,             dtype=np.float32)
        self.buf_returns  = np.zeros(buf,             dtype=np.float32)
        self.buf_c1       = np.zeros(buf,             dtype=np.float32)  # overload cost
        self.buf_c2       = np.zeros(buf,             dtype=np.float32)  # conflict cost

        os.makedirs(LOG_DIR,  exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        self.writer = SummaryWriter(LOG_DIR) if HAS_TB else None
        self.total_steps = 0

    # ── Rollout collection ─────────────────────────────────────────────────

    def collect(self):
        self.policy.eval()
        ptr = 0
        ep_rew = np.zeros(N_DRONES)
        ep_con = ep_ovl = ep_cmp = 0

        self.env.reset()
        all_obs = self.env.get_all_agent_obs()

        # Accumulate constraint costs over rollout for multiplier update
        rollout_c1 = []
        rollout_c2 = []

        for step in range(N_STEPS):
            obs_t = torch.FloatTensor(all_obs).to(self.device)
            with torch.no_grad():
                acts, logps, _, vals = \
                    self.policy.get_action_and_value(obs_t)
            acts_np  = acts.cpu().numpy()
            logps_np = logps.cpu().numpy()
            vals_np  = vals.cpu().numpy()

            new_obs, raw_rewards, dones, terminated, truncated, info = \
                self.env.step_joint(acts_np.tolist())

            c1 = info["overloads_per_agent"]   # per-agent overload cost
            c2 = info["conflicts_per_agent"]   # per-agent conflict cost

            # Augmented reward: r̃ = r - λ₁·c₁ - λ₂·c₂
            aug_rewards = self.lagrange.augment_reward(
                raw_rewards, c1, c2)

            rollout_c1.append(c1.mean())
            rollout_c2.append(c2.mean())

            # Store
            idx = slice(ptr * N_DRONES, (ptr + 1) * N_DRONES)
            self.buf_obs[idx]      = all_obs
            self.buf_actions[idx]  = acts_np
            self.buf_rewards[idx]  = aug_rewards
            self.buf_raw_rew[idx]  = raw_rewards
            self.buf_dones[idx]    = dones.astype(float)
            self.buf_values[idx]   = vals_np
            self.buf_logprobs[idx] = logps_np
            self.buf_c1[idx]       = c1
            self.buf_c2[idx]       = c2

            ptr += 1
            self.total_steps += N_DRONES

            ep_rew += raw_rewards
            ep_con  = info.get("ep_conflicts", 0)
            ep_ovl  = info.get("ep_overloads", 0)
            ep_cmp  = info.get("ep_completed", 0)

            all_obs = new_obs

            if terminated or truncated:
                self.metrics.record(
                    ep_rew.mean(), ep_con, ep_ovl, ep_cmp,
                    N_DRONES, self.lagrange.lam1, self.lagrange.lam2
                )
                if self.metrics.ep_count % 10 == 0:
                    self.metrics.print("[Constrained-MARL] ")
                    self._log_tb()
                self.env.reset()
                all_obs = self.env.get_all_agent_obs()
                ep_rew  = np.zeros(N_DRONES)
                ep_con  = ep_ovl = ep_cmp = 0

        # Update Lagrange multipliers after rollout
        mean_c1 = float(np.mean(rollout_c1))
        mean_c2 = float(np.mean(rollout_c2))
        self.lagrange.update(mean_c1, mean_c2)

        # Compute GAE on augmented rewards
        obs_t = torch.FloatTensor(all_obs).to(self.device)
        with torch.no_grad():
            _, _, _, last_vals = \
                self.policy.get_action_and_value(obs_t)
        self._compute_gae(last_vals.cpu().numpy(),
                          np.zeros(N_DRONES))

    def _compute_gae(self, last_values, last_dones):
        gae = np.zeros(N_DRONES, dtype=np.float32)
        for step in reversed(range(N_STEPS)):
            idx      = slice(step * N_DRONES, (step+1) * N_DRONES)
            next_idx = slice((step+1)*N_DRONES, (step+2)*N_DRONES)
            if step == N_STEPS - 1:
                next_vals  = last_values
                next_dones = last_dones
            else:
                next_vals  = self.buf_values[next_idx]
                next_dones = self.buf_dones[next_idx]
            delta = (self.buf_rewards[idx]
                     + GAMMA * next_vals * (1 - next_dones)
                     - self.buf_values[idx])
            gae = (delta + GAMMA * GAE_LAMBDA
                   * (1 - self.buf_dones[idx]) * gae)
            self.buf_advs[idx]    = gae
            self.buf_returns[idx] = gae + self.buf_values[idx]

    # ── Policy update ──────────────────────────────────────────────────────

    def update(self):
        self.policy.train()
        obs_t  = torch.FloatTensor(self.buf_obs).to(self.device)
        acts_t = torch.LongTensor(self.buf_actions).to(self.device)
        rets_t = torch.FloatTensor(self.buf_returns).to(self.device)
        advs_t = torch.FloatTensor(self.buf_advs).to(self.device)
        logp_t = torch.FloatTensor(self.buf_logprobs).to(self.device)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        n = obs_t.shape[0]
        pg_losses, vf_losses, ent_losses = [], [], []

        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                mb = idx[start:start+BATCH_SIZE]
                _, new_logp, entropy, new_val = \
                    self.policy.get_action_and_value(
                        obs_t[mb], acts_t[mb])
                ratio = torch.exp(new_logp - logp_t[mb])
                pg1   = -advs_t[mb] * ratio
                pg2   = -advs_t[mb] * torch.clamp(
                    ratio, 1-CLIP_RANGE, 1+CLIP_RANGE)
                pg    = torch.max(pg1, pg2).mean()
                vf    = 0.5 * ((new_val - rets_t[mb])**2).mean()
                ent   = -entropy.mean()
                loss  = pg + 0.5*vf + ENT_COEF*ent
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 0.5)
                self.optim.step()
                pg_losses.append(pg.item())
                vf_losses.append(vf.item())
                ent_losses.append(ent.item())

        return {
            "pg":  np.mean(pg_losses),
            "vf":  np.mean(vf_losses),
            "ent": np.mean(ent_losses)
        }

    # ── Training loop ──────────────────────────────────────────────────────

    def train(self, total_steps: int = TOTAL_STEPS):
        print("=" * 65)
        print("Constrained MARL — Lagrangian PPO")
        print(f"  Drones:       {N_DRONES}")
        print(f"  Constraints:  b_cap={B_CAP}, b_sep={B_SEP}")
        print(f"  λ init:       λ₁={self.lagrange.lam1:.3f}  "
              f"λ₂={self.lagrange.lam2:.3f}")
        print(f"  Dual LR:      {LR_LAMBDA}")
        print(f"  Steps:        {total_steps:,}")
        print("=" * 65)

        updates = total_steps // (N_STEPS * N_DRONES)
        for upd in range(1, updates + 1):
            self.collect()
            losses = self.update()

            if upd % 5 == 0:
                m = self.metrics
                print(
                    f"Update {upd:4d}/{updates} | "
                    f"pg={losses['pg']:.4f} | "
                    f"λ₁={self.lagrange.lam1:.3f} "
                    f"λ₂={self.lagrange.lam2:.3f} | "
                    f"conflicts={m.mean(m.conflicts):.2f} | "
                    f"throughput={m.mean(m.throughput):.1%}"
                )

            if upd % 20 == 0:
                path = os.path.join(
                    CKPT_DIR, f"constrained_{upd}.pt")
                self.save(path)

        self.save(os.path.join(CKPT_DIR, "constrained_final.pt"))
        print("\nTraining complete!")
        results = self.evaluate()
        return results

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 10):
        print(f"\nEvaluating ({n_episodes} episodes)...")
        self.policy.eval()
        m = Metrics(window=n_episodes)

        for ep in range(n_episodes):
            self.env.reset()
            all_obs = self.env.get_all_agent_obs()
            ep_rew, done, step = 0.0, False, 0

            while not done and step < 100:   # ← hard cap at 100 steps
                acts = [self.policy.act(all_obs[i], deterministic=True)
                        for i in range(N_DRONES)]
                all_obs, rewards, _, terminated, truncated, info = \
                    self.env.step_joint(acts)
                ep_rew += rewards.mean()
                done    = terminated or truncated
                step   += 1

            m.record(ep_rew, info["ep_conflicts"], info["ep_overloads"],
                    info["ep_completed"], N_DRONES, self.lagrange.lam1, self.lagrange.lam2)

        print("── Evaluation Results ──────────────────────────")
        print(f"  Mean reward:     {m.mean(m.rewards):+.2f}")
        print(f"  Mean throughput: {m.mean(m.throughput):.1%}")
        print(f"  Mean conflicts:  {m.mean(m.conflicts):.1f}")
        print(f"  Mean overloads:  {m.mean(m.overloads):.1f}")
        print("─" * 48)

    def save(self, path: str):
        torch.save({
            "policy":  self.policy.state_dict(),
            "optim":   self.optim.state_dict(),
            "lagrange": self.lagrange.state_dict(),
            "steps":   self.total_steps,
        }, path)
        print(f"Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optim.load_state_dict(ckpt["optim"])
        self.lagrange.load_state_dict(ckpt["lagrange"])
        self.total_steps = ckpt.get("steps", 0)
        print(f"Loaded: {path}")

    def _log_tb(self):
        if not self.writer:
            return
        m  = self.metrics
        s  = self.total_steps
        self.writer.add_scalar("constrained/throughput",
                               m.mean(m.throughput),  s)
        self.writer.add_scalar("constrained/reward",
                               m.mean(m.rewards),     s)
        self.writer.add_scalar("constrained/conflicts",
                               m.mean(m.conflicts),   s)
        self.writer.add_scalar("constrained/overloads",
                               m.mean(m.overloads),   s)
        self.writer.add_scalar("constrained/lambda1",
                               self.lagrange.lam1,    s)
        self.writer.add_scalar("constrained/lambda2",
                               self.lagrange.lam2,    s)


# ══════════════════════════════════════════════════════════════════════════
# COMPARISON UTILITY
# ══════════════════════════════════════════════════════════════════════════

def compare_methods(constrained_results: dict,
                    vanilla_ckpt: str = None):
    """Compare Constrained MARL vs Vanilla MAPPO vs Random."""
    print("\n" + "=" * 65)
    print("Method Comparison Table")
    print("=" * 65)
    print(f"{'Method':<28} {'Throughput':>11} "
          f"{'Conflicts':>10} {'Overloads':>10} {'Time':>8}")
    print("-" * 65)

    # Random baseline — quick evaluation
    env = BlueSkyUTMEnv(n_drones=N_DRONES,
                        render_mode="none", max_steps=50)
    rand_cons, rand_ovl, rand_cmp = [], [], []
    for _ in range(10):
        env.reset()
        obs = env.get_all_agent_obs()
        done = False
        while not done:
            acts = [env.action_space.sample()
                    for _ in range(N_DRONES)]
            obs, _, _, term, trunc, info = env.step_joint(acts)
            done = term or trunc
        rand_cons.append(info["ep_conflicts"])
        rand_ovl.append(info["ep_overloads"])
        rand_cmp.append(info["ep_completed"] / N_DRONES)

    print(f"{'Random Policy':<28} "
          f"{np.mean(rand_cmp):>10.1%} "
          f"{np.mean(rand_cons):>10.1f} "
          f"{np.mean(rand_ovl):>10.1f} "
          f"{'<1ms':>8}")

    # Vanilla MAPPO
    if vanilla_ckpt and os.path.exists(vanilla_ckpt):
        from training.train_mappo_bluesky import (
            SharedActorCritic as VanillaAC)
        vanilla = VanillaAC(env.agent_obs_dim, env.n_sectors)
        ckpt    = torch.load(vanilla_ckpt, map_location="cpu")
        vanilla.load_state_dict(ckpt.get("policy", ckpt), strict=False)
        vanilla.eval()

        van_cons, van_ovl, van_cmp = [], [], []
        for _ in range(10):
            env.reset()
            obs  = env.get_all_agent_obs()
            done = False
            step = 0
            while not done and step < 50:
                acts = [vanilla.act(obs[i], deterministic=True)
                        for i in range(N_DRONES)]
                obs, _, _, term, trunc, info = env.step_joint(acts)
                done = term or trunc
                step += 1
            van_cons.append(info["ep_conflicts"])
            van_ovl.append(info["ep_overloads"])
            van_cmp.append(info["ep_completed"] / N_DRONES)

        print(f"{'Vanilla MAPPO':<28} "
              f"{np.mean(van_cmp):>10.1%} "
              f"{np.mean(van_cons):>10.1f} "
              f"{np.mean(van_ovl):>10.1f} "
              f"{'<1ms':>8}")

    # Constrained MARL
    print(f"{'Constrained MARL (ours)':<28} "
          f"{constrained_results['throughput']:>10.1%} "
          f"{constrained_results['conflicts']:>10.2f} "
          f"{constrained_results['overloads']:>10.2f} "
          f"{'<1ms':>8}")

    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",    type=int,  default=TOTAL_STEPS)
    parser.add_argument("--render",       action="store_true")
    parser.add_argument("--compare",      action="store_true",
                        help="Run comparison vs vanilla MAPPO after training")
    parser.add_argument("--vanilla-ckpt", type=str,
                        default="checkpoints_bluesky/mappo_bluesky_final.pt")
    parser.add_argument("--load",         type=str,  default=None,
                        help="Load existing checkpoint and evaluate only")
    parser.add_argument("--seed",         type=int,  default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cpu"  # MPS can cause device mismatches — CPU is stable
    print(f"Device: {device}")

    trainer = ConstrainedMARLTrainer(
        render=args.render, device=device)

    if args.load:
        trainer.load(args.load)
        results = trainer.evaluate()
    else:
        results = trainer.train(total_steps=args.timesteps)

    if args.compare:
        compare_methods(results, vanilla_ckpt=args.vanilla_ckpt)
