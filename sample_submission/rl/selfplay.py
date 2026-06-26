"""
Self-play game runner.

Runs a full game using battle_start / battle_select.
Returns trajectories for both players to use in PPO training.

Trajectory step:
  state_feat   : list[float]  (STATE_DIM,)
  opt_feats    : list[list[float]]  (N, OPTION_DIM)
  action_idx   : int
  log_prob     : float
  value        : float
  reward       : float   (filled in after game ends)
  done         : bool
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from dataclasses import dataclass, field

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class
from .encoder import encode_state, encode_option
from .model import PokemonRLModel


@dataclass
class Step:
    player: int
    state_feat: list[float]
    opt_feats: list[list[float]]
    action_idx: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False


def run_game(
    model: PokemonRLModel,
    deck0: list[int],
    deck1: list[int],
    device: torch.device,
    greedy: bool = False,
) -> tuple[list[Step], list[Step], int]:
    """
    Run one game. Both players use the same model.

    Returns:
        steps0: trajectory for player 0
        steps1: trajectory for player 1
        winner: 0, 1, or 2 (draw)
    """
    obs_dict, start = battle_start(deck0, deck1)
    if obs_dict is None:
        battle_finish()
        return [], [], -1

    steps0: list[Step] = []
    steps1: list[Step] = []

    while True:
        obs = to_observation_class(obs_dict)

        if obs.current is not None and obs.current.result != -1:
            winner = obs.current.result
            break

        # Deck selection (select is None) — use rule-based deck (already chosen)
        if obs.select is None:
            # This shouldn't happen in mid-game; just pick deck 0 or 1
            obs_dict = battle_select([])
            continue

        player = obs.current.yourIndex if obs.current else 0

        # Encode state
        state_feat = encode_state(obs)
        opt_feats = [encode_option(opt, obs) for opt in obs.select.option]

        n_opts = len(opt_feats)
        if n_opts == 0:
            obs_dict = battle_select([])
            continue

        # Must select minCount..maxCount options; model picks one at a time
        # For simplicity, pick exactly minCount options greedily after the first
        min_c = obs.select.minCount
        max_c = obs.select.maxCount

        s_t = torch.tensor(state_feat, dtype=torch.float32, device=device)
        o_t = torch.tensor(opt_feats, dtype=torch.float32, device=device)

        chosen = []
        first_log_prob = 0.0
        first_value = 0.0
        available = list(range(n_opts))

        for pick_i in range(max(min_c, 1)):
            if not available:
                break
            # Build sub-option tensor from available indices
            avail_feats = torch.stack([o_t[i] for i in available])
            idx_in_avail, lp, val = model.act(s_t, avail_feats, greedy=greedy)
            global_idx = available[idx_in_avail]
            chosen.append(global_idx)
            if pick_i == 0:
                first_log_prob = lp
                first_value = val
            # remove chosen from available (no duplicates)
            available.remove(global_idx)
            if len(chosen) >= max_c:
                break

        step = Step(
            player=player,
            state_feat=state_feat,
            opt_feats=opt_feats,
            action_idx=chosen[0] if chosen else 0,
            log_prob=first_log_prob,
            value=first_value,
        )
        if player == 0:
            steps0.append(step)
        else:
            steps1.append(step)

        obs_dict = battle_select(chosen)

    battle_finish()

    # Assign terminal rewards
    winner_val = obs.current.result if obs.current else -1
    for steps, pid in [(steps0, 0), (steps1, 1)]:
        if not steps:
            continue
        if winner_val == pid:
            reward = 1.0
        elif winner_val == 2:
            reward = 0.0
        else:
            reward = -1.0
        steps[-1].reward = reward
        steps[-1].done = True

    return steps0, steps1, winner_val


def collect_episodes(
    model: PokemonRLModel,
    deck: list[int],
    n_games: int,
    device: torch.device,
) -> list[Step]:
    """Run n_games self-play games and return all steps."""
    all_steps: list[Step] = []
    wins = [0, 0, 0]
    for _ in range(n_games):
        s0, s1, w = run_game(model, deck, deck, device)
        all_steps.extend(s0)
        all_steps.extend(s1)
        if 0 <= w <= 2:
            wins[w] += 1
    print(f"  wins: p0={wins[0]}, p1={wins[1]}, draw={wins[2]}")
    return all_steps
