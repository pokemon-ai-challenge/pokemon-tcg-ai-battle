"""Kaggle submission entry point — Maskable PPO Dragapult ex agent.

Inference is pure numpy (no torch needed): the trained policy is a small MLP, and
tcg_rl.mlp_numpy reproduces the torch forward pass exactly. This keeps the
submission small and guaranteed to run regardless of the runtime's packages.

Decision flow per call:
  * initial deck selection (obs.select is None) -> return the 60-card deck
  * otherwise -> encode the observation and run the factored policy (pick options
    one at a time, with action masking, until STOP / maxCount) to build the list
  * any failure falls back to a guaranteed-legal action so the agent never crashes
"""

import os
import sys

# Make the submission folder importable regardless of the working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cg.api import Observation, to_observation_class  # noqa: E402

_POLICY = None
_POLICY_TRIED = False


def _candidate_paths(filename: str):
    return [
        os.path.join(_HERE, filename),
        filename,
        "/kaggle_simulations/agent/" + filename,
    ]


def read_deck_csv() -> list[int]:
    """Read deck.csv and return a list of 60 card IDs."""
    for path in _candidate_paths("deck.csv"):
        if os.path.exists(path):
            with open(path, "r") as f:
                rows = [r for r in f.read().split("\n") if r.strip()]
            return [int(rows[i]) for i in range(60)]
    raise FileNotFoundError("deck.csv not found")


def _get_policy():
    """Lazily load policy.npz; return None if unavailable (triggers fallback)."""
    global _POLICY, _POLICY_TRIED
    if _POLICY_TRIED:
        return _POLICY
    _POLICY_TRIED = True
    try:
        from tcg_rl.mlp_numpy import load_policy
        for path in _candidate_paths("policy.npz"):
            if os.path.exists(path):
                _POLICY = load_policy(path)
                break
    except Exception:
        _POLICY = None
    return _POLICY


def _fallback_action(obs: Observation) -> list[int]:
    """A guaranteed-legal action used when the policy is unavailable or errors."""
    sel = obs.select
    if sel is None:
        return read_deck_csv()
    n = len(sel.option)
    if sel.minCount <= 0:
        return []
    return list(range(min(sel.minCount, n)))


def agent(obs_dict: dict) -> list[int]:
    """Competition entry point. Returns option indices (or the deck on turn 0)."""
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        return read_deck_csv()

    try:
        policy = _get_policy()
        if policy is None:
            return _fallback_action(obs)
        from tcg_rl.mlp_numpy import decide_full_action
        action = decide_full_action(obs, policy.choose)
        # final legality guard
        n = len(obs.select.option)
        action = [i for i in action if 0 <= i < n]
        if len(action) < obs.select.minCount:
            return _fallback_action(obs)
        return action
    except Exception:
        return _fallback_action(obs)
