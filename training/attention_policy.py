# Attention-based SharedActorCritic — DROP-IN replacement.
# Same interface (forward / get_action_and_value / act), same constructor signature.
# Internally: splits obs into an "own-context" token + K intruder tokens,
# runs self-attention so each drone attends over its nearest neighbors,
# pools, then actor/critic heads.

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# --- Observation layout (must match the env) ---
# [ WIN_CELLS(9) | own_pos(3) | own_sec(1) | own_goal(1) | K*3 intruders | conflict(2) ]
WIN_CELLS = 9
OWN_EXTRA = 3 + 1 + 1        # pos + sec + goal
CONF      = 2
INTRUDER_FEAT = 3            # each intruder = 3 coords

class SharedActorCritic(nn.Module):
    """Attention policy. Signature kept identical: (obs_dim, n_actions, hidden)."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        # infer number of intruder tokens K from obs_dim
        own_len = WIN_CELLS + OWN_EXTRA + CONF            # = 16
        self.K = max(0, (obs_dim - own_len) // INTRUDER_FEAT)
        self.own_len = own_len
        d = hidden                                        # embedding dim

        # token embedders: own-context token, and per-intruder token
        self.own_embed  = nn.Linear(own_len, d)
        self.intr_embed = nn.Linear(INTRUDER_FEAT, d)
        # learnable type markers so the model distinguishes own vs intruder
        self.own_type  = nn.Parameter(torch.zeros(1, 1, d))
        self.intr_type = nn.Parameter(torch.zeros(1, 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=d*2,
            dropout=0.0, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                     enable_nested_tensor=False)

        self.actor  = nn.Linear(d, n_actions)
        self.critic = nn.Linear(d, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _tokenize(self, obs):
        # obs: (B, obs_dim) -> tokens (B, 1+K, d), plus padding mask (B,1+K)
        B = obs.shape[0]
        win_own = obs[:, :WIN_CELLS + OWN_EXTRA]                 # (B,12)
        conf    = obs[:, self.obs_dim - CONF:]                   # (B,2)
        own_vec = torch.cat([win_own, conf], dim=1)             # (B,16)
        own_tok = self.own_embed(own_vec).unsqueeze(1) + self.own_type   # (B,1,d)

        if self.K > 0:
            start = WIN_CELLS + OWN_EXTRA
            intr = obs[:, start:start + self.K*INTRUDER_FEAT]    # (B,K*3)
            intr = intr.view(B, self.K, INTRUDER_FEAT)          # (B,K,3)
            intr_tok = self.intr_embed(intr) + self.intr_type    # (B,K,d)
            tokens = torch.cat([own_tok, intr_tok], dim=1)      # (B,1+K,d)
            # mask out zero-padded intruders (all-zero coords -> absent neighbor)
            pad = (intr.abs().sum(-1) == 0)                     # (B,K) True where padded
            mask = torch.cat([torch.zeros(B,1,dtype=torch.bool,device=obs.device), pad], dim=1)
            # own token (col 0) is always visible; guarantees no fully-masked row
        else:
            tokens = own_tok
            mask = torch.zeros(B,1,dtype=torch.bool,device=obs.device)
        return tokens, mask

    def forward(self, obs):
        single = (obs.dim() == 1)
        if single: obs = obs.unsqueeze(0)
        tokens, mask = self._tokenize(obs)
        enc = self.encoder(tokens, src_key_padding_mask=mask)  # (B,1+K,d)
        h = enc[:, 0]                                           # own token = summary
        logits = self.actor(h)
        value  = self.critic(h).squeeze(-1)
        if single:
            logits = logits.squeeze(0); value = value.squeeze(0)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> int:
        t = torch.FloatTensor(obs).unsqueeze(0).to(next(self.parameters()).device)
        logits, _ = self(t)
        if deterministic:
            return int(logits.argmax().item())
        return int(Categorical(logits=logits).sample().item())