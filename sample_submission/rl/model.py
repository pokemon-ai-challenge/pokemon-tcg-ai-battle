"""
PPO Actor-Critic network for Pokemon TCG.

Architecture:
  - State encoder:  MLP(STATE_DIM → 128 → 128)
  - Option encoder: MLP(OPTION_DIM → 64)
  - Scorer:         Linear(128+64 → 1) per option → softmax = policy
  - Value head:     Linear(128 → 1)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import STATE_DIM, OPTION_DIM


class PokemonRLModel(nn.Module):
    def __init__(self, hidden: int = 128, opt_hidden: int = 64):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.opt_enc = nn.Sequential(
            nn.Linear(OPTION_DIM, opt_hidden),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden + opt_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Linear(hidden, 1)

    def forward(
        self,
        state: torch.Tensor,          # (B, STATE_DIM)
        options: torch.Tensor,        # (B, N, OPTION_DIM)  N = num options
        option_mask: torch.Tensor,    # (B, N) bool, True = valid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: (B, N) log probabilities over options (invalid = -inf)
            values:    (B,)   state value estimates
        """
        B, N, _ = options.shape
        s = self.state_enc(state)               # (B, hidden)
        o = self.opt_enc(options.view(B * N, -1)).view(B, N, -1)  # (B, N, opt_hidden)

        s_exp = s.unsqueeze(1).expand(-1, N, -1)   # (B, N, hidden)
        combined = torch.cat([s_exp, o], dim=-1)    # (B, N, hidden+opt_hidden)
        scores = self.scorer(combined).squeeze(-1)  # (B, N)

        # Mask invalid options
        scores = scores.masked_fill(~option_mask, float('-inf'))
        log_probs = F.log_softmax(scores, dim=-1)

        values = self.value_head(s).squeeze(-1)     # (B,)
        return log_probs, values

    def act(
        self,
        state: torch.Tensor,    # (STATE_DIM,)
        options: torch.Tensor,  # (N, OPTION_DIM)
        greedy: bool = False,
    ) -> tuple[int, float, float]:
        """
        Sample one action from the policy.
        Returns: (action_index, log_prob, value)
        """
        with torch.no_grad():
            s = state.unsqueeze(0)
            o = options.unsqueeze(0)
            mask = torch.ones(1, options.shape[0], dtype=torch.bool)
            log_probs, values = self.forward(s, o, mask)
            log_probs = log_probs.squeeze(0)  # (N,)
            if greedy:
                idx = log_probs.argmax().item()
            else:
                idx = torch.distributions.Categorical(logits=log_probs).sample().item()
            return int(idx), log_probs[idx].item(), values.squeeze().item()
