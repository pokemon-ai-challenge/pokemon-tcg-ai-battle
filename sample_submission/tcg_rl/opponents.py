"""Opponent policies used inside the training environment.

An opponent is a callable: opponent(obs: Observation) -> list[int]
returning a full, legal selection for ONE SelectData (the opponent does not use
the factored buffer; it returns the complete index list at once).

Provided opponents:
    - random_opponent           : uniform random legal action
    - make_selfplay_opponent     : mirror of the current PPO policy (torch)
    - make_mc_opponent           : the Flat Monte Carlo agent (tcg_rl.mc_agent)
    - CurriculumOpponent         : samples among opponents by training progress
"""

from __future__ import annotations

import random

from cg.api import Observation
from tcg_rl.mlp_numpy import decide_full_action


def _legal_random(sel) -> list[int]:
    n = len(sel.option)
    lo = sel.minCount
    hi = min(sel.maxCount, n)
    if hi < lo:
        return []
    k = random.randint(lo, hi)
    if k <= 0:
        return []
    return random.sample(range(n), k)


def random_opponent(obs: Observation) -> list[int]:
    sel = obs.select
    if sel is None:
        return []
    return _legal_random(sel)


def make_selfplay_opponent(model, deterministic: bool = False):
    """Opponent that plays with the given MaskablePPO model (factored loop)."""
    import numpy as np

    def choose_fn(vec, mask):
        action, _ = model.predict(
            np.asarray(vec, dtype=np.float32),
            action_masks=np.asarray(mask, dtype=bool),
            deterministic=deterministic,
        )
        return int(action)

    def opp(obs: Observation) -> list[int]:
        return decide_full_action(obs, choose_fn)

    return opp


def make_mc_opponent(rollouts: int = 8, depth: int = 20, time_budget: float = 0.4):
    """Opponent backed by the Flat Monte Carlo agent (loose budget)."""
    from tcg_rl.mc_agent import MonteCarloAgent

    agent = MonteCarloAgent(rollouts=rollouts, depth=depth, time_budget=time_budget)

    def opp(obs: Observation) -> list[int]:
        return agent.act(obs)

    return opp


class CurriculumOpponent:
    """Picks an opponent per episode based on a schedule of (threshold, weights).

    schedule: list of (min_steps, {name: weight}). The last entry whose
    min_steps <= current global step is used. Names map to factory callables in
    `pool`. Call .set_step(n) from a training callback to advance the curriculum.
    """

    def __init__(self, pool: dict, schedule: list):
        self.pool = pool          # name -> opponent callable
        self.schedule = schedule  # list[(min_steps, {name: weight})]
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    def _weights(self) -> dict:
        chosen = self.schedule[0][1]
        for min_steps, weights in self.schedule:
            if self._step >= min_steps:
                chosen = weights
        return chosen

    def __call__(self):
        weights = self._weights()
        names = list(weights.keys())
        w = [weights[n] for n in names]
        name = random.choices(names, weights=w, k=1)[0]
        return self.pool[name]
