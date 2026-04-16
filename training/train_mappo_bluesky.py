"""
train_mappo_bluesky.py
======================
Multi-Agent PPO training using BlueSky as the simulation backend.
Connects your existing MAPPO logic to the BlueSky UTM environment.

Usage:
    # With BlueSky GUI (shows simulation visually):
    python training/train_mappo_bluesky.py --render

    # Headless (faster training, no window):
    python training/train_mappo_bluesky.py

    # Quick test (1000 steps):
    python training/train_mappo_bluesky.py --timesteps 1000
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "env"))

from env.bluesky_utm_env import BlueSkyUTMEnv
_SCALE   = int(os.environ.get("UTM_SCALE", "5"))
N_DRONES = _SCALE
LOG_DIR  = f"./logs_mappo_{_SCALE}"
CKPT_DIR = f"./checkpoints_mappo_{_SCALE}"
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

# ── Hyperparameters ────────────────────────────────────────────────────────
TOTAL_STEPS    = 200_000
N_STEPS        = 256       # rollout length
N_EPOCHS       = 8
BATCH_SIZE     = 128
LR             = 3e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.01
LOG_DIR        = "./logs_bluesky"


# ── Shared Actor-Critic ────────────────────────────────────────────────────

class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
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
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic=False) -> int:
        t = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
        logits, _ = self(t)
        if deterministic:
            return int(logits.argmax().item())
        return int(Categorical(logits=logits).sample().item())


# ── Metrics tracker ────────────────────────────────────────────────────────

class Metrics:
    def __init__(self, window=50):
        self.rewards    = deque(maxlen=window)
        self.conflicts  = deque(maxlen=window)
        self.overloads  = deque(maxlen=window)
        self.throughput = deque(maxlen=window)
        self.ep_count   = 0

    def record(self, reward, conflicts, overloads, completed, n_drones):
        self.ep_count += 1
        self.rewards.append(reward)
        self.conflicts.append(conflicts)
        self.overloads.append(overloads)
        self.throughput.append(completed / max(n_drones, 1))

    def mean(self, q):
        return float(np.mean(list(q))) if q else 0.0

    def print(self, prefix=""):
        print(
            f"{prefix}Ep {self.ep_count:4d} | "
            f"Reward {self.mean(self.rewards):+7.2f} | "
            f"Throughput {self.mean(self.throughput):.1%} | "
            f"Conflicts {self.mean(self.conflicts):.1f} | "
            f"Overloads {self.mean(self.overloads):.1f}"
        )


# ── MAPPO Trainer ──────────────────────────────────────────────────────────

class MAPPOBlueSky:

    def __init__(self, render: bool = False, device: str = "cpu"):
        self.device  = torch.device(device)
        self.metrics = Metrics()
        self.render  = render

        # Environment
        render_mode = "human" if render else "none"
        self.env = BlueSkyUTMEnv(
            n_drones    = N_DRONES,
            render_mode = "none",
            #render_mode = render_mode,
            max_steps   = 200
        )

        obs_dim   = self.env.agent_obs_dim
        n_actions = self.env.n_sectors

        # Shared policy
        self.policy = SharedActorCritic(obs_dim, n_actions).to(self.device)
        self.optim  = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)

        # Rollout buffers (flattened: N_STEPS * N_DRONES entries)
        buf_size = N_STEPS * N_DRONES
        self.buf_obs      = np.zeros((buf_size, obs_dim),   dtype=np.float32)
        self.buf_actions  = np.zeros(buf_size,              dtype=np.int64)
        self.buf_rewards  = np.zeros(buf_size,              dtype=np.float32)
        self.buf_dones    = np.zeros(buf_size,              dtype=np.float32)
        self.buf_values   = np.zeros(buf_size,              dtype=np.float32)
        self.buf_logprobs = np.zeros(buf_size,              dtype=np.float32)
        self.buf_advs     = np.zeros(buf_size,              dtype=np.float32)
        self.buf_returns  = np.zeros(buf_size,              dtype=np.float32)

        os.makedirs(LOG_DIR,  exist_ok=True)
        os.makedirs(CKPT_DIR, exist_ok=True)
        self.writer = SummaryWriter(LOG_DIR) if HAS_TB else None
        self.total_steps = 0
        self.ptr = 0

    def collect(self):
        """Collect N_STEPS of experience from all agents."""
        self.ptr = 0
        self.policy.eval()

        # Reset episode tracking
        ep_rew = np.zeros(N_DRONES)
        ep_con = ep_ovl = ep_cmp = ep_len = 0

        self.env.reset()
        all_obs = self.env.get_all_agent_obs()

        for step in range(N_STEPS):
            obs_t = torch.FloatTensor(all_obs).to(self.device)
            with torch.no_grad():
                acts, logps, _, vals = self.policy.get_action_and_value(obs_t)
            acts_np  = acts.cpu().numpy()
            logps_np = logps.cpu().numpy()
            vals_np  = vals.cpu().numpy()

            new_obs, rewards, dones, terminated, truncated, info = \
                self.env.step_joint(acts_np.tolist())

            ep_rew += rewards
            ep_con  = info.get("ep_conflicts", 0)
            ep_ovl  = info.get("ep_overloads", 0)
            ep_cmp  = info.get("ep_completed", 0)
            ep_len += 1

            # Store in buffer (flattened)
            idx = slice(self.ptr * N_DRONES, (self.ptr + 1) * N_DRONES)
            self.buf_obs[idx]      = all_obs
            self.buf_actions[idx]  = acts_np
            self.buf_rewards[idx]  = rewards
            self.buf_dones[idx]    = dones.astype(float)
            self.buf_values[idx]   = vals_np
            self.buf_logprobs[idx] = logps_np
            self.ptr += 1
            self.total_steps += N_DRONES

            all_obs = new_obs

            if terminated or truncated:
                self.metrics.record(ep_rew.mean(), ep_con, ep_ovl, ep_cmp, N_DRONES)
                if self.metrics.ep_count % 10 == 0:
                    self.metrics.print("[BlueSky-MAPPO] ")
                    self._log_tb()
                # Reset
                self.env.reset()
                all_obs  = self.env.get_all_agent_obs()
                ep_rew   = np.zeros(N_DRONES)
                ep_con   = ep_ovl = ep_cmp = ep_len = 0

        # Compute GAE
        obs_t = torch.FloatTensor(all_obs).to(self.device)
        with torch.no_grad():
            _, _, _, last_vals = self.policy.get_action_and_value(obs_t)
        self._compute_gae(last_vals.cpu().numpy(), np.zeros(N_DRONES))

    def _compute_gae(self, last_values, last_dones):
        gae = np.zeros(N_DRONES, dtype=np.float32)
        for step in reversed(range(N_STEPS)):
            idx      = slice(step * N_DRONES, (step+1) * N_DRONES)
            next_idx = slice((step+1) * N_DRONES, (step+2) * N_DRONES)
            if step == N_STEPS - 1:
                next_vals  = last_values
                next_dones = last_dones
            else:
                next_vals  = self.buf_values[next_idx]
                next_dones = self.buf_dones[next_idx]
            delta = (self.buf_rewards[idx]
                     + GAMMA * next_vals * (1 - next_dones)
                     - self.buf_values[idx])
            gae = delta + GAMMA * GAE_LAMBDA * (1 - self.buf_dones[idx]) * gae
            self.buf_advs[idx]    = gae
            self.buf_returns[idx] = gae + self.buf_values[idx]

    def update(self):
        self.policy.train()
        obs_t  = torch.FloatTensor(self.buf_obs).to(self.device)
        acts_t = torch.LongTensor(self.buf_actions).to(self.device)
        rets_t = torch.FloatTensor(self.buf_returns).to(self.device)
        advs_t = torch.FloatTensor(self.buf_advs).to(self.device)
        logp_t = torch.FloatTensor(self.buf_logprobs).to(self.device)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        n = obs_t.shape[0]
        pg_losses, vf_losses = [], []

        for _ in range(N_EPOCHS):
            idx = np.random.permutation(n)
            for start in range(0, n, BATCH_SIZE):
                mb = idx[start:start+BATCH_SIZE]
                _, new_logp, entropy, new_val = \
                    self.policy.get_action_and_value(obs_t[mb], acts_t[mb])
                ratio = torch.exp(new_logp - logp_t[mb])
                pg1   = -advs_t[mb] * ratio
                pg2   = -advs_t[mb] * torch.clamp(ratio, 1-CLIP_RANGE, 1+CLIP_RANGE)
                pg    = torch.max(pg1, pg2).mean()
                vf    = 0.5 * ((new_val - rets_t[mb])**2).mean()
                ent   = -entropy.mean()
                loss  = pg + 0.5*vf + ENT_COEF*ent
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()
                pg_losses.append(pg.item())
                vf_losses.append(vf.item())

        return {"pg": np.mean(pg_losses), "vf": np.mean(vf_losses)}

    def train(self, total_steps: int = TOTAL_STEPS):
        print("=" * 60)
        print("UTM MARL — BlueSky Backend")
        print(f"  Drones:    {N_DRONES}")
        print(f"  Sectors:   3×3 = 9 (Dubai airspace)")
        print(f"  Render:    {'Yes (BlueSky GUI)' if self.render else 'Headless'}")
        print(f"  Steps:     {total_steps:,}")
        print("=" * 60)

        updates = total_steps // (N_STEPS * N_DRONES)
        for upd in range(1, updates + 1):
            self.collect()
            losses = self.update()

            if upd % 5 == 0:
                m = self.metrics
                print(
                    f"Update {upd:4d}/{updates} | "
                    f"pg={losses['pg']:.4f} vf={losses['vf']:.4f} | "
                    f"throughput={m.mean(m.throughput):.1%}"
                )

            if upd % 20 == 0:
                path = os.path.join(CKPT_DIR, f"mappo_bluesky_{upd}.pt")
                self.save(path)

        self.save(os.path.join(CKPT_DIR, "mappo_bluesky_final.pt"))
        print("\nTraining complete!")
        self.evaluate()

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
                    info["ep_completed"], N_DRONES)

        print("── Evaluation Results ──────────────────────────")
        print(f"  Mean reward:     {m.mean(m.rewards):+.2f}")
        print(f"  Mean throughput: {m.mean(m.throughput):.1%}")
        print(f"  Mean conflicts:  {m.mean(m.conflicts):.1f}")
        print(f"  Mean overloads:  {m.mean(m.overloads):.1f}")
        print("─" * 48)

    def save(self, path):
        torch.save({"policy": self.policy.state_dict(),
                    "optim":  self.optim.state_dict(),
                    "steps":  self.total_steps}, path)
        print(f"Saved: {path}")

    def _log_tb(self):
        if not self.writer:
            return
        m = self.metrics
        self.writer.add_scalar("utm/throughput",  m.mean(m.throughput), self.total_steps)
        self.writer.add_scalar("utm/reward",      m.mean(m.rewards),    self.total_steps)
        self.writer.add_scalar("utm/conflicts",   m.mean(m.conflicts),  self.total_steps)
        self.writer.add_scalar("utm/overloads",   m.mean(m.overloads),  self.total_steps)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render",     action="store_true",
                        help="Open BlueSky GUI window")
    parser.add_argument("--timesteps",  type=int, default=TOTAL_STEPS)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = MAPPOBlueSky(render=args.render, device=device)
    trainer.train(total_steps=args.timesteps)
