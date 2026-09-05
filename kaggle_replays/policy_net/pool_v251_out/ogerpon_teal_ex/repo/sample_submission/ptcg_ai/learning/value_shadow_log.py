"""Value network shadow-mode logging (Stage1 Step4, ``stage1-wiring-implementation-plan.md``).

Records ``ValueModel.predict_win_prob(obs)`` once per turn without using it
for any decision (shadow mode only: Stage1 explicitly stops short of using
the value network to change action selection). A module-level singleton
buffer, following the same reset pattern as
``hidden_information.match_context`` (``obs.select is None`` marks a new
match; the buffer holds exactly the current match's turns in this process).

Callers (``rule_based_agent.agent()`` / ``ml_policy_agent.agent()``) call
``record(obs)`` once per turn, gated behind the ``value_shadow_logging``
config flag. ``record()`` never raises: any failure here must not affect
the actual game decision.
"""

from __future__ import annotations

from cg.api import Observation

from ptcg_ai.learning.value_model import ValueModel

_model: ValueModel | None = None
_log: list[dict] = []


def _get_model() -> ValueModel:
    global _model
    if _model is None:
        _model = ValueModel()
    return _model


def reset() -> None:
    """Discard the current match's log. Call when a new match starts."""
    global _log
    _log = []


def get_log() -> list[dict]:
    """Return a copy of the current match's recorded (turn, win_prob) entries."""
    return list(_log)


def record(obs: Observation) -> None:
    """Predict and append this turn's win probability. Never raises.

    ``obs.select is None`` (deck-selection turn) resets the log for the new
    match instead of recording (``obs.current`` is not available yet).
    """
    try:
        if obs.select is None:
            reset()
            return
        state = obs.current
        if state is None:
            return
        prob = _get_model().predict_win_prob(obs)
        _log.append({
            "turn": state.turn,
            "your_index": state.yourIndex,
            "win_prob": prob,
        })
    except Exception:
        pass
