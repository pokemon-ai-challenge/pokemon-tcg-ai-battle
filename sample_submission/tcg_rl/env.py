"""Gymnasium environment wrapping the local cg.game battle engine.

The learner is always player 0. The opponent (player 1) is driven internally by
a pluggable opponent policy. Each Gymnasium step corresponds to ONE sub-decision
of the factored action space (pick an option, or STOP to submit the current
multi-select). See docs/mppo-spec.md for the full design.

IMPORTANT: cg.game uses a process-global battle pointer, so only ONE env battle
can run per process. Do not put multiple instances of this env in a DummyVecEnv;
use SubprocVecEnv (separate processes) for parallelism.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cg.game import battle_start, battle_select, battle_finish
from cg.api import to_observation_class, Observation
from tcg_rl.features import (
    ACTION_DIM,
    MAX_OPTIONS,
    OBS_DIM,
    STOP_ACTION,
    action_mask,
    encode_obs,
)

LEARNER = 0


def _default_opponent_provider():
    from tcg_rl.opponents import random_opponent
    return random_opponent


class PokemonTCGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        learner_deck: list[int],
        opp_deck: list[int] | None = None,
        opponent_provider=None,
        max_steps: int = 2000,
        max_engine_steps: int = 4000,
        reward_shaping: bool = False,
    ):
        super().__init__()
        self.learner_deck = list(learner_deck)
        self.opp_deck = list(opp_deck) if opp_deck else list(learner_deck)
        # opponent_provider() -> opponent callable, called once per episode
        self.opponent_provider = opponent_provider or _default_opponent_provider
        self.opponent_fn = None
        self.max_steps = max_steps
        self.max_engine_steps = max_engine_steps
        self.reward_shaping = reward_shaping

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        self.obs_dict = None
        self.buffer: list[int] = []
        self.steps = 0
        self._battle_active = False
        self._prev_prize_diff = 0

    # ----------------------------------------------------------------- masks
    def action_masks(self) -> np.ndarray:
        obs = to_observation_class(self.obs_dict)
        return action_mask(obs, self.buffer)

    # ----------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._battle_active:
            try:
                battle_finish()
            except Exception:
                pass
            self._battle_active = False

        self.opponent_fn = self.opponent_provider()
        self.obs_dict, start = battle_start(self.learner_deck, self.opp_deck)
        if start.errorType != 0 or self.obs_dict is None:
            # extremely rare; retry once
            self.obs_dict, start = battle_start(self.learner_deck, self.opp_deck)
        self._battle_active = True
        self.buffer = []
        self.steps = 0
        self._prev_prize_diff = self._prize_diff()

        obs, terminal, winner = self._advance_to_learner()
        if terminal:
            # degenerate immediate end; start a fresh battle
            return self.reset(seed=seed)
        return encode_obs(obs, self.buffer), {}

    # ------------------------------------------------------------------ step
    def step(self, action):
        obs = to_observation_class(self.obs_dict)
        sel = obs.select
        a = int(action)
        n = min(len(sel.option), MAX_OPTIONS)

        commit = False
        if a == STOP_ACTION:
            if len(self.buffer) >= sel.minCount:
                commit = True
            else:
                a = self._first_unused(n)
        if not commit:
            if a is not None and 0 <= a < n and a not in self.buffer:
                self.buffer.append(a)
            if len(self.buffer) >= sel.maxCount:
                commit = True

        if not commit:
            # need more sub-picks for this same SelectData
            return encode_obs(obs, self.buffer), 0.0, False, False, {}

        # commit the accumulated selection
        full = self._ensure_min(sel, list(self.buffer))
        self.buffer = []
        try:
            self.obs_dict = battle_select(full)
            self.steps += 1
            nobs, terminal, winner = self._advance_to_learner()
        except Exception:
            # The global battle pointer was invalidated (e.g., another
            # PokemonTCGEnv started a battle in this process). End the episode
            # as truncated; the auto-reset will create a fresh battle.
            self._battle_active = False
            return (
                np.zeros(OBS_DIM, dtype=np.float32),
                0.0, False, True, {"engine_error": True},
            )

        if terminal:
            reward = self._terminal_reward(winner)
            return encode_obs(nobs, self.buffer), reward, True, False, {"winner": winner}

        reward = self._shaping_reward() if self.reward_shaping else 0.0
        truncated = self.steps >= self.max_steps
        return encode_obs(nobs, self.buffer), reward, False, truncated, {}

    # --------------------------------------------------------------- helpers
    def _advance_to_learner(self):
        """Play opponent (and skip non-learner) selections until the learner must
        act or the battle ends. Returns (Observation, terminal, winner)."""
        for _ in range(self.max_engine_steps):
            obs = to_observation_class(self.obs_dict)
            st = obs.current
            if st is not None and st.result != -1:
                return obs, True, st.result
            if obs.select is None:
                # deck-selection phase is not expected in local play; guard anyway
                self.obs_dict = battle_select(self.learner_deck)
                continue
            if int(st.yourIndex) == LEARNER:
                return obs, False, None
            act = self.opponent_fn(obs)
            act = self._sanitize(obs.select, act)
            self.obs_dict = battle_select(act)
        return to_observation_class(self.obs_dict), True, None

    def _sanitize(self, sel, action) -> list[int]:
        """Coerce an opponent's action into a legal selection."""
        n = len(sel.option)
        seen, out = set(), []
        for x in (action or []):
            xi = int(x)
            if 0 <= xi < n and xi not in seen and len(out) < sel.maxCount:
                seen.add(xi)
                out.append(xi)
        i = 0
        while len(out) < sel.minCount and i < n:
            if i not in seen:
                out.append(i)
                seen.add(i)
            i += 1
        return out

    def _ensure_min(self, sel, full) -> list[int]:
        n = min(len(sel.option), MAX_OPTIONS)
        i = 0
        seen = set(full)
        while len(full) < sel.minCount and i < n:
            if i not in seen:
                full.append(i)
                seen.add(i)
            i += 1
        return full

    def _first_unused(self, n: int):
        for i in range(n):
            if i not in self.buffer:
                return i
        return STOP_ACTION

    def _prize_diff(self) -> int:
        obs = to_observation_class(self.obs_dict)
        st = obs.current
        if st is None:
            return 0
        me = st.players[LEARNER]
        opp = st.players[1 - LEARNER]
        # fewer remaining prizes is better; diff = opp_remaining - my_remaining
        return len(opp.prize) - len(me.prize)

    def _shaping_reward(self) -> float:
        diff = self._prize_diff()
        delta = diff - self._prev_prize_diff
        self._prev_prize_diff = diff
        return 0.1 * float(delta)

    def _terminal_reward(self, winner) -> float:
        if winner == LEARNER:
            return 1.0
        if winner == 1 - LEARNER:
            return -1.0
        return 0.0

    def close(self):
        if self._battle_active:
            try:
                battle_finish()
            except Exception:
                pass
            self._battle_active = False
