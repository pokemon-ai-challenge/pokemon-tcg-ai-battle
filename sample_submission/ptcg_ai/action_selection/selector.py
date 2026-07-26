"""Action selection: try lethal search, then the learned policy prior, then the
rule-based router.

``select_action()`` is the normal-turn entry point called from
``ptcg_ai.rule_based.rule_based_agent``. Search modules never talk to
the agent entry points directly; this module wires the config, builds
the hidden state for the search, validates whatever the search returns
and falls back to the rule-based ``router.route()``, guaranteeing a
legal action even when everything else fails.

## Learned policy prior (opt-in, MAIN only)

Between lethal search and the rule-based router, ``SelectContext.MAIN``
decisions can optionally be routed through ``ptcg_ai.learning.policy_model``:
resolve every option with ``semantic_action.resolve_option()``, extract
features with ``ptcg_ai.learning.policy_features``, score them, and take
the argmax. Gated by ``ml_policy.enabled`` in the config (default
``False``), overridable by ``PTCG_ML_POLICY=0``/``1`` -- same idiom as
``use_real_hidden_state`` below. Whatever it returns still goes through
``is_valid_action()``; missing weights, an unexpected exception, or a
non-MAIN context all fall through to ``router.route()`` untouched.

## Hidden state supplied to the search (FR-WIRE)

The search needs concrete card IDs for every hidden zone. Two sources
exist and are selected by the ``use_real_hidden_state`` config flag:

- ``False`` (default): ``hidden_information.search_state_stub`` — dummy
  data, not a prediction. Historical behaviour, kept as the A/B control.
- ``True``: the real estimation layer
  (``hidden_information.match_context`` + ``search_adapter``), i.e. the
  ``OpponentHiddenState`` / ``OwnHiddenState`` posteriors that
  ``rule_based_agent`` already updates every turn via
  ``match_context.update(obs)`` *before* this module runs.

``PTCG_REAL_HIDDEN_STATE=1`` / ``=0`` overrides the config flag (same
idiom as ``PTCG_SETUP_BEFORE_ATTACK`` in ``rule_based``), so an A/B run
needs no config edit.

Either factory is called once per verification replay by the search, so
each call re-samples the hidden zones; that is what lets the search
reject lines which only work for one particular shuffle.
"""

from __future__ import annotations

import dataclasses
import os

from cg.api import Observation, SelectContext, SelectData

from ptcg_ai.action_selection import fallback, router
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information import match_context
from ptcg_ai.hidden_information.search_adapter import to_search_begin_kwargs
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.learning import policy_model
from ptcg_ai.learning.observable_state import observable_state
from ptcg_ai.learning.semantic_action import resolve_option
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.search import lethal_simple

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
}

_CONFIG_CACHE: dict | None = None
_POLICY_MODEL_CACHE: policy_model.PolicyModel | None = None


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def _policy_model() -> policy_model.PolicyModel:
    global _POLICY_MODEL_CACHE
    if _POLICY_MODEL_CACHE is None:
        _POLICY_MODEL_CACHE = policy_model.PolicyModel()
    return _POLICY_MODEL_CACHE


def _use_ml_policy(config: dict) -> bool:
    """Whether to try the learned policy prior before falling back to ``router.route``.

    Same idiom as ``_use_real_hidden_state``: ``PTCG_ML_POLICY`` wins over the config
    when set to ``0``/``1`` (anything else is ignored). Defaults to ``False`` (config's
    ``ml_policy.enabled``, itself defaulting to ``False``), so this step is a no-op
    until both the config and the weights file are in place.
    """
    override = os.environ.get("PTCG_ML_POLICY")
    if override in ("0", "1"):
        return override == "1"
    return bool(((config or {}).get("ml_policy") or {}).get("enabled", False))


def _ml_policy_action(obs: Observation, select: SelectData) -> list[int] | None:
    """Resolve options, score them with the learned policy prior, and argmax.

    Restricted to ``SelectContext.MAIN`` (the only context the policy prior is trained
    on). Returns ``None`` -- meaning "fall through to the rule-based router" -- when the
    context isn't MAIN, the model isn't loaded, or anything raises; this function must
    never be allowed to stop the turn. The caller still runs ``is_valid_action()`` on
    whatever this returns, exactly like every other action source in this module.
    """
    if select.context != SelectContext.MAIN or obs.current is None:
        return None
    model = _policy_model()
    if not model.is_ready:
        return None
    try:
        current_dict = dataclasses.asdict(obs.current)
        me = current_dict.get("yourIndex", 0)
        state = observable_state(current_dict, me)
        actions = [
            resolve_option(dataclasses.asdict(option), current_dict, me)
            for option in select.option
        ]
        choice = model.select(state, actions)
    except Exception:  # noqa: BLE001 -- never let the policy prior break the turn
        return None
    if choice is None:
        return None
    return [choice]


def _use_real_hidden_state(config: dict) -> bool:
    """Whether to feed the search the real estimation instead of the dummy stub.

    ``PTCG_REAL_HIDDEN_STATE`` wins over the config when set to ``0``/``1``
    (anything else is ignored, so a stray value cannot silently flip the
    behaviour). Defaults to ``False``: the stub stays the control arm until
    the A/B says otherwise.
    """
    override = os.environ.get("PTCG_REAL_HIDDEN_STATE")
    if override in ("0", "1"):
        return override == "1"
    return bool((config or {}).get("use_real_hidden_state", False))


def _hidden_state_factory(obs: Observation, full_deck: list[int], use_real: bool):
    """Build the zero-argument callable the search calls per verification replay.

    Returning ``None`` is part of the search's factory contract (that replay
    is skipped), so an estimation failure degrades the search instead of
    breaking the turn.
    """
    if not use_real:
        return lambda: build_dummy_search_state(obs, full_deck)

    def factory():
        try:
            # match_context is already updated for this turn by
            # rule_based_agent.agent() before select_action() runs.
            return to_search_begin_kwargs(
                match_context.get_own_state(),
                match_context.get_opponent_state(),
                obs,
            )
        except Exception:  # noqa: BLE001 -- never let estimation break the turn
            return None

    return factory


def is_valid_action(action, select: SelectData) -> bool:
    """Check the contract required by the competition runner."""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def select_action(obs: Observation, full_deck: list[int], config: dict | None = None) -> list[int]:
    """Choose the action for the current selection.

    Args:
        obs: Observation passed to the agent (``obs.select`` must be set).
        full_deck: Our own 60-card deck list (used only by the dummy
            hidden state; the real estimation reads the deck itself).
        config: Agent config dict (``lethal_search`` section and the
            ``use_real_hidden_state`` flag are used).
            Defaults to ``core.config.load_config()``.

    Returns:
        list[int]: A legal selection.
    """
    select = obs.select
    if config is None:
        config = _config()
    lethal_config = (config or {}).get("lethal_search") or {}

    # 相手デッキ予測の更新は lethal_search / router のどちらに進む前にも必ず通したいので、
    # この関数の一番手前で行う。予測結果は priorities/*.py が
    # opponent_tracker.current_matchup_plan() 経由で参照する。失敗しても通常運用は継続する。
    if obs.current is not None:
        try:
            opponent_tracker.update(obs)
        except Exception:
            pass

    if lethal_config.get("enabled", False) and obs.current is not None:
        module = _SEARCH_MODULES.get(lethal_config.get("module", "lethal_simple"))
        if module is not None:
            context = {
                "observation": obs,
                "config": lethal_config,
                # The search takes the hidden state from the outside; which
                # source is used is the FR-WIRE flag (module docstring).
                "hidden_state_factory": _hidden_state_factory(
                    obs, full_deck, _use_real_hidden_state(config)
                ),
            }
            try:
                action = module.search(obs.current, select.option, context)
            except Exception:
                action = None
            if action is not None and is_valid_action(action, select):
                return action

    # No certain lethal: try the learned policy prior (MAIN only, opt-in via config/env).
    if _use_ml_policy(config):
        action = _ml_policy_action(obs, select)
        if action is not None and is_valid_action(action, select):
            return action

    # No certain lethal / no ML policy: play the normal rule-based policy.
    try:
        action = router.route(obs)
    except Exception:
        action = None
    if action is not None and is_valid_action(action, select):
        return action
    return fallback.safe_choice(obs)
