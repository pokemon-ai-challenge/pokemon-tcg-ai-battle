"""Action selection: try lethal search first, then the rule-based router.

``select_action()`` is the normal-turn entry point called from
``ptcg_ai.rule_based.rule_based_agent``. Search modules never talk to
the agent entry points directly; this module wires the config, builds
the hidden state for the search (the dummy stub in
``hidden_information.search_state_stub`` by default, or the real
estimate from ``hidden_information.search_adapter`` when the config
opts in via ``lethal_search.hidden_state_source``), validates whatever
the search returns and falls back to the rule-based ``router.route()``,
guaranteeing a legal action even when everything else fails.
"""

from __future__ import annotations

from cg.api import Observation, SelectData

from ptcg_ai.action_selection import fallback, router
from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information import match_context, search_adapter
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.search import lethal_simple, pimc

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
    "pimc": pimc,
}

_CONFIG_CACHE: dict | None = None


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


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
        full_deck: Our own 60-card deck list (used to build the dummy
            hidden state handed to the search).
        config: Agent config dict (``lethal_search`` section is used).
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
            hidden_state_source = lethal_config.get("hidden_state_source", "dummy")
            if hidden_state_source == "estimated":
                # Real hidden-information estimate (hidden_information.match_context
                # is kept up to date once per turn by rule_based_agent.agent()).
                factory = lambda: search_adapter.to_search_begin_kwargs(
                    match_context.get_own_state(obs.current.yourIndex),
                    match_context.get_opponent_state(obs.current.yourIndex),
                    obs,
                )
            else:
                # Dummy stub until real estimation is opted into (default,
                # unchanged behavior for existing configs).
                factory = lambda: build_dummy_search_state(obs, full_deck)
            context = {
                "observation": obs,
                "config": lethal_config,
                "hidden_state_factory": factory,
            }
            try:
                action = module.search(obs.current, select.option, context)
            except Exception:
                action = None
            if action is not None and is_valid_action(action, select):
                return action

    # No certain lethal: play the normal rule-based policy.
    try:
        action = router.route(obs)
    except Exception:
        action = None
    if action is not None and is_valid_action(action, select):
        return action
    return fallback.safe_choice(obs)
