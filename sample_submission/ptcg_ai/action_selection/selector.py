"""Action selection: try lethal search first, then fall back.

``select_action()`` is the single entry point used by ``main.py``.
Search modules never talk to ``main.py`` directly; this module wires the
config, builds the search context, validates whatever the search returns
and guarantees a legal action even when everything else fails.
"""

from __future__ import annotations

import random

from cg.api import Observation, SelectData

from ptcg_ai.search import lethal_simple

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
}


def fallback_action(select: SelectData) -> list[int]:
    """Stable baseline policy: a random legal selection."""
    return random.sample(range(len(select.option)), select.maxCount)


def is_valid_action(action, select: SelectData) -> bool:
    """Check the contract required by the competition runner."""
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def select_action(obs: Observation, full_deck: list[int], config: dict | None) -> list[int]:
    """Choose the action for the current selection.

    Args:
        obs: Observation passed to the agent (``obs.select`` must be set).
        full_deck: Our own 60-card deck list (used to predict hidden info).
        config: Agent config dict (``lethal_search`` section is used).

    Returns:
        list[int]: A legal selection.
    """
    select = obs.select
    lethal_config = (config or {}).get("lethal_search") or {}

    if lethal_config.get("enabled", False) and obs.current is not None:
        module = _SEARCH_MODULES.get(lethal_config.get("module", "lethal_simple"))
        if module is not None:
            context = {
                "observation": obs,
                "full_deck": full_deck,
                "config": lethal_config,
            }
            try:
                action = module.search(obs.current, select.option, context)
            except Exception:
                action = None
            if action is not None and is_valid_action(action, select):
                return action

    return fallback_action(select)
