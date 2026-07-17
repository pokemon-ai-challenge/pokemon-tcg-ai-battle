"""Deterministic lethal search (issues #57 / #58).

Searches, within the current own turn only, for an action sequence that
ends the game in our victory, using the competition search API
(``cg.api.search_begin`` / ``search_step``). Only "certain" lethals are
targeted: a found line is re-verified against reshuffled hidden
information, so lines that depend on unknown card order (or coin luck)
are rejected.

Phase 1 (#57) covered the last-prize case; Phase 2 (#58) triggers at
<= 2 remaining prizes (lethal via KOing a Pokemon ex / Mega ex,
multi-KO effects, or non-attack card effects — all detected uniformly
through ``State.result``) and adds pruning and instrumentation
(``get_stats()`` / ``reset_stats()``).

Entry point (team common interface)::

    def search(state, legal_actions, context) -> list[int] | None

``context`` keys:

- ``observation`` (required): the original ``Observation`` given to the agent.
- ``config``: the ``lethal_search`` section of the agent config
  (missing keys fall back to ``DEFAULTS``).
- ``full_deck``: our own 60-card deck list, used to predict hidden cards.
- ``predictions`` / ``predictions_factory``: pre-built hidden-information
  dict (see ``ptcg_ai.hidden_information.naive.predict_hidden``) or a
  zero-argument callable returning one. Mainly for tests; when absent,
  ``predict_hidden`` is used with ``full_deck``.
- ``rng``: optional ``random.Random`` for reproducible predictions.

Returns the first selection (option index list) of a winning line, or
None when there is no certain lethal / on timeout / on any error, in
which case the caller falls back to its normal policy.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Callable, Iterator

from cg import api as cg_api
from cg.api import Observation, OptionType, SelectData, SelectType, State

from ptcg_ai.hidden_information.naive import predict_hidden

DEFAULTS: dict = {
    "enabled": True,
    "module": "lethal_simple",
    "max_remaining_prizes": 2,
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000,
    # Cap on selection combinations generated per node (multi-select
    # prompts can explode combinatorially).
    "max_combinations_per_select": 128,
    # Extra replays with reshuffled hidden info to confirm the line is
    # deterministic. 0 disables verification.
    "verify_shuffles": 1,
}

# Option ordering for MAIN selections (#58 探索優先順位): attacks first
# (multi-prize KOs and win-now lines), then damage raisers
# (ability/evolve), energy acceleration, retreat/switch, other card use.
# END is excluded from MAIN candidates entirely: the search only covers
# the current own turn, so ending the turn can never reach a win.
_MAIN_OPTION_PRIORITY = {
    OptionType.ATTACK: 0,
    OptionType.ABILITY: 1,
    OptionType.EVOLVE: 2,
    OptionType.ATTACH: 3,
    OptionType.RETREAT: 4,
    OptionType.PLAY: 5,
    OptionType.DISCARD: 6,
}
_DEFAULT_PRIORITY = 50


class _SearchAbort(Exception):
    """Raised internally when a time/node budget is exhausted."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Instrumentation (#58): cumulative counters for local measurement.
_STATS_ZERO = {
    "searches": 0,        # searches actually started (gates passed)
    "found": 0,           # searches that returned a lethal action
    "timeouts": 0,        # aborted by time_limit_ms
    "node_limit_hits": 0, # aborted by max_nodes
    "verify_rejects": 0,  # lines rejected by the determinism replay
    "total_time_ms": 0.0,
    "max_time_ms": 0.0,
}
_stats = dict(_STATS_ZERO)


def get_stats() -> dict:
    """Return cumulative search statistics (see ``_STATS_ZERO``)."""
    stats = dict(_stats)
    searches = stats["searches"]
    stats["avg_time_ms"] = stats["total_time_ms"] / searches if searches else 0.0
    return stats


def reset_stats() -> None:
    _stats.update(_STATS_ZERO)


def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """Search for an action winning within the current own turn.

    Args:
        state: Current ``State``.
        legal_actions: Current ``obs.select.option``.
        context: See module docstring.

    Returns:
        list[int] | None: First selection of a winning line, else None.
    """
    try:
        config = {**DEFAULTS, **(context.get("config") or {})}
        if not config["enabled"]:
            return None

        obs: Observation | None = context.get("observation")
        if obs is None or obs.select is None or obs.current is None:
            return None
        if state is None:
            state = obs.current

        me = state.yourIndex
        if state.result != -1:
            return None
        if not _is_my_turn(state, me):
            return None
        if len(state.players[me].prize) > config["max_remaining_prizes"]:
            return None

        predictions_factory = _predictions_factory(obs, context)
        if predictions_factory is None:
            return None
        predictions = predictions_factory()
        if predictions is None:
            return None

        start_time = time.perf_counter()
        _stats["searches"] += 1
        deadline = start_time + config["time_limit_ms"] / 1000.0
        try:
            path = _find_winning_path(obs, me, predictions, config, deadline)
            if path is None:
                return None

            first = path[0]
            if not _is_legal_selection(first, obs.select):
                return None

            # Verification gets its own small budget: the DFS may have
            # consumed the whole deadline, and a replay is only
            # len(path) engine steps.
            verify_deadline = (
                time.perf_counter() + config["time_limit_ms"] / 1000.0
            )
            for _ in range(int(config["verify_shuffles"])):
                verify_predictions = predictions_factory()
                if verify_predictions is None:
                    return None
                if not _replay_wins(obs, me, verify_predictions, path, verify_deadline):
                    _stats["verify_rejects"] += 1
                    return None
            _stats["found"] += 1
            return first
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            _stats["total_time_ms"] += elapsed_ms
            _stats["max_time_ms"] = max(_stats["max_time_ms"], elapsed_ms)
            try:
                cg_api.search_end()
            except Exception:
                pass
    except Exception:
        return None


def _predictions_factory(obs: Observation, context: dict) -> Callable[[], dict | None] | None:
    factory = context.get("predictions_factory")
    if factory is not None:
        return factory
    predictions = context.get("predictions")
    if predictions is not None:
        return lambda: predictions
    full_deck = context.get("full_deck")
    if not full_deck:
        return None
    rng = context.get("rng") or random
    return lambda: predict_hidden(obs, full_deck, rng)


def _is_my_turn(state: State, me: int) -> bool:
    if state.turn < 1 or state.firstPlayer < 0:
        return False
    starter_turn = state.turn % 2 == 1
    return starter_turn == (state.firstPlayer == me)


def _is_legal_selection(selection: list[int], select: SelectData) -> bool:
    if not isinstance(selection, list):
        return False
    if not (select.minCount <= len(selection) <= select.maxCount):
        return False
    if len(selection) != len(set(selection)):
        return False
    return all(
        isinstance(index, int) and 0 <= index < len(select.option)
        for index in selection
    )


def _find_winning_path(
    obs: Observation,
    me: int,
    predictions: dict,
    config: dict,
    deadline: float,
) -> list[list[int]] | None:
    """Iterative-deepening DFS from the current observation.

    Returns the list of selections leading to our victory, or None.
    Raises nothing: budget exhaustion is converted to None.
    """
    root = _begin(obs, predictions)
    budget = {"nodes": 0, "depth_cutoff": False}
    try:
        for depth_limit in range(1, int(config["max_depth"]) + 1):
            visited: dict[str, int] = {}
            budget["depth_cutoff"] = False
            path = _dfs(root, me, depth_limit, config, deadline, budget, visited)
            if path is not None:
                return path
            if not budget["depth_cutoff"]:
                # The whole reachable tree fits within this depth limit
                # and holds no win; deeper iterations cannot find one.
                return None
        return None
    except _SearchAbort as abort:
        if abort.reason == "time":
            _stats["timeouts"] += 1
        else:
            _stats["node_limit_hits"] += 1
        return None
    finally:
        try:
            cg_api.search_release(root.searchId)
        except Exception:
            pass


def _dfs(
    node,
    me: int,
    depth_left: int,
    config: dict,
    deadline: float,
    budget: dict,
    visited: dict[str, int],
) -> list[list[int]] | None:
    obs = node.observation
    if obs.select is None or not obs.select.option:
        return None
    if depth_left <= 0:
        budget["depth_cutoff"] = True
        return None

    key = _state_key(obs)
    if visited.get(key, -1) >= depth_left:
        return None
    visited[key] = depth_left

    for selection in _candidate_selections(obs.select, config):
        if time.perf_counter() > deadline:
            raise _SearchAbort("time")
        if budget["nodes"] >= int(config["max_nodes"]):
            raise _SearchAbort("nodes")
        budget["nodes"] += 1

        try:
            child = cg_api.search_step(node.searchId, selection)
        except ValueError:
            continue
        try:
            child_state = child.observation.current
            if child_state.result == me:
                return [selection]
            if child_state.result != -1:
                continue  # opponent won
            if child_state.yourIndex != me:
                continue  # turn ended or control moved to the opponent
            sub_path = _dfs(
                child, me, depth_left - 1, config, deadline, budget, visited
            )
            if sub_path is not None:
                return [selection] + sub_path
        finally:
            try:
                cg_api.search_release(child.searchId)
            except Exception:
                pass
    return None


def _candidate_selections(select: SelectData, config: dict) -> Iterator[list[int]]:
    """Generate index selections satisfying min/max count, no duplicates.

    MAIN options are reordered by ``_MAIN_OPTION_PRIORITY`` and END is
    pruned (sound: the search only covers the current own turn, so
    ending the turn can never lead to a win). Other select types keep
    their natural order. Output is capped by
    ``max_combinations_per_select``.
    """
    order = list(range(len(select.option)))
    if select.type == SelectType.MAIN:
        order = [i for i in order if select.option[i].type != OptionType.END]
        order.sort(
            key=lambda i: _MAIN_OPTION_PRIORITY.get(
                select.option[i].type, _DEFAULT_PRIORITY
            )
        )

    min_count = max(select.minCount, 0)
    max_count = min(select.maxCount, len(order))
    limit = int(config["max_combinations_per_select"])
    produced = 0
    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(order, count):
            yield list(combo)
            produced += 1
            if produced >= limit:
                return


def _replay_wins(
    obs: Observation,
    me: int,
    predictions: dict,
    path: list[list[int]],
    deadline: float,
) -> bool:
    """Replay ``path`` under different hidden info; True if we still win."""
    try:
        node = _begin(obs, predictions)
    except Exception:
        return False
    search_ids = [node.searchId]
    try:
        for selection in path:
            if time.perf_counter() > deadline:
                return False
            try:
                node = cg_api.search_step(node.searchId, selection)
            except ValueError:
                return False  # line is not legal under this shuffle
            search_ids.append(node.searchId)
            result = node.observation.current.result
            if result != -1:
                return result == me
        return node.observation.current.result == me
    finally:
        for search_id in search_ids:
            try:
                cg_api.search_release(search_id)
            except Exception:
                pass


def _begin(obs: Observation, predictions: dict):
    return cg_api.search_begin(
        obs,
        predictions["your_deck"],
        predictions["your_prize"],
        predictions["opponent_deck"],
        predictions["opponent_prize"],
        predictions["opponent_hand"],
        predictions["opponent_active"],
    )


def _state_key(obs: Observation) -> str:
    # Dataclass repr is deterministic for identical states; good enough
    # to suppress re-exploring transpositions within one search.
    return f"{obs.current!r}|{obs.select!r}"
