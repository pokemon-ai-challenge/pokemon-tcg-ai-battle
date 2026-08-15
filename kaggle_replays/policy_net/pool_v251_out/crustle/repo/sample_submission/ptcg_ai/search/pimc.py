"""PIMC v0: value-scored own-turn search (Stage2, ``stage2-pimc-implementation-plan.md``).

Searches, within the current own turn only, for the action sequence whose
end-of-turn (or certain-win) outcome scores highest under the value network
(``ptcg_ai.learning.value_model.ValueModel``), aggregated over several
determinizations (independent samples of hidden information) via
``context["hidden_state_factory"]``. This is *not* a general information-set
search: it never models the opponent's turn, so no opponent action model is
needed (see the module docstring rationale in the implementation plan,
Non-goals section).

A certain win (``State.result == me``) found anywhere in a determinization's
tree is scored 1.0 and short-circuits that determinization's exploration
(mirrors ``lethal_simple``'s "adopt a certain win immediately" property), but
does *not* short-circuit the outer determinization loop: a line's win/loss
under different hidden-info samples is exactly what the averaging step is
meant to expose (an action that wins under every sample averages near 1.0; an
action that only wins under one sample gets pulled down by the samples where
it does not).

Entry point (team common interface, shared with ``lethal_simple``)::

    def search(state, legal_actions, context) -> list[int] | None

``context`` keys are identical to ``lethal_simple``'s (see that module's
docstring): ``observation``, ``config`` (the ``lethal_search`` config
section), and ``hidden_state`` / ``hidden_state_factory``.

This file is intentionally independent of ``lethal_simple.py`` (which is
never imported here) and does not modify it. The action-generation helper
(``_candidate_selections``) and small validation helpers are copied rather
than shared, following the existing team convention of duplicating these
across search-adjacent modules (e.g. ``ml_policy_agent.py`` duplicating
``selector._is_valid_action``).
"""

from __future__ import annotations

import itertools
import time
from typing import Callable, Iterator

from cg import api as cg_api
from cg.api import Observation, OptionType, SelectData, SelectType, State

from ptcg_ai.learning.value_model import ValueModel

DEFAULTS: dict = {
    "enabled": True,
    "module": "pimc",
    "time_limit_ms": 300,
    "max_depth": 20,
    "max_nodes": 10000,
    "max_combinations_per_select": 128,
    "num_determinizations": 4,
}

# Same MAIN-option ordering as lethal_simple: attacks first, END excluded
# from candidates entirely (the search only covers the current own turn, so
# ending the turn can never be improved upon by continuing the search here;
# its value is already captured by the leaf evaluation of whichever action
# preceded it).
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
    """Raised internally when a single determinization's time/node budget is exhausted."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Instrumentation: cumulative counters for local measurement (Step4).
_STATS_ZERO = {
    "searches": 0,                        # top-level search() calls that passed the gates
    "found": 0,                           # searches that returned a non-None action
    "determinizations_run": 0,            # individual hidden-state samples explored
    "determinization_timeouts": 0,        # determinizations aborted by time_limit_ms
    "determinization_node_limit_hits": 0, # determinizations aborted by max_nodes
    "total_time_ms": 0.0,                 # wall time of the whole search() call
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


_model: ValueModel | None = None


def _get_model() -> ValueModel:
    global _model
    if _model is None:
        _model = ValueModel()
    return _model


def _leaf_score(state: State) -> float:
    return _get_model().predict_win_prob_from_state(state)


def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """Search for the best-scoring action within the current own turn.

    Args:
        state: Current ``State``.
        legal_actions: Current ``obs.select.option``.
        context: See module docstring.

    Returns:
        list[int] | None: First selection of the best-scoring line found
        across all determinizations, or None (no candidate could be scored /
        gates not met / any error), in which case the caller falls back to
        its normal policy.
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

        hidden_state_factory = _hidden_state_factory(context)
        if hidden_state_factory is None:
            return None

        start_time = time.perf_counter()
        _stats["searches"] += 1
        deadline = start_time + config["time_limit_ms"] / 1000.0
        num_determinizations = max(1, int(config["num_determinizations"]))

        try:
            aggregate = _run_determinizations(
                obs, me, hidden_state_factory, config, deadline, num_determinizations
            )
            if not aggregate:
                return None

            best_key = max(aggregate, key=lambda k: sum(aggregate[k]) / len(aggregate[k]))
            best_selection = list(best_key)
            if not _is_legal_selection(best_selection, obs.select):
                return None
            _stats["found"] += 1
            return best_selection
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


def _run_determinizations(
    obs: Observation,
    me: int,
    hidden_state_factory: Callable[[], dict | None],
    config: dict,
    deadline: float,
    num_determinizations: int,
) -> dict[tuple[int, ...], list[float]]:
    """Run up to ``num_determinizations`` samples, aggregating per-first-move scores.

    Each determinization contributes at most one score per first-move
    selection (only for the selections it actually explored: a
    determinization that finds a certain win early only contributes that one
    key). The shared ``deadline`` is a wall-clock budget for the whole call,
    not per determinization.
    """
    aggregate: dict[tuple[int, ...], list[float]] = {}
    for _ in range(num_determinizations):
        if time.perf_counter() > deadline:
            break
        hidden_state = hidden_state_factory()
        if hidden_state is None:
            continue
        try:
            root = _begin(obs, hidden_state)
        except Exception:
            continue
        _stats["determinizations_run"] += 1
        budget = {"nodes": 0}
        try:
            scores = _explore(root, me, config, deadline, budget)
        except _SearchAbort as abort:
            if abort.reason == "time":
                _stats["determinization_timeouts"] += 1
            else:
                _stats["determinization_node_limit_hits"] += 1
            scores = {}
        finally:
            try:
                cg_api.search_release(root.searchId)
            except Exception:
                pass
        for key, score in scores.items():
            aggregate.setdefault(key, []).append(score)
    return aggregate


def _explore(
    node,
    me: int,
    config: dict,
    deadline: float,
    budget: dict,
) -> dict[tuple[int, ...], float]:
    """Explore one determinization's tree; return best score per first move.

    Mirrors ``lethal_simple._dfs``'s shape (candidate generation, time/node
    budget checks, search_step/search_release), but instead of returning the
    first winning path, it records a score for every first-move candidate
    (value-network leaf score, or 1.0 on a certain win) and returns
    immediately once a certain win is found anywhere in the tree (further
    exploration cannot beat a score of 1.0).
    """
    scores: dict[tuple[int, ...], float] = {}
    obs = node.observation
    if obs.select is None or not obs.select.option:
        return scores

    visited: dict[str, tuple[int, float, bool]] = {}
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
            score, found_win = _dfs_score(
                child, me, int(config["max_depth"]) - 1, config, deadline, budget, visited
            )
        finally:
            try:
                cg_api.search_release(child.searchId)
            except Exception:
                pass

        scores[tuple(selection)] = score
        if found_win:
            return scores
    return scores


def _dfs_score(
    node,
    me: int,
    depth_left: int,
    config: dict,
    deadline: float,
    budget: dict,
    visited: dict[str, tuple[int, float, bool]],
) -> tuple[float, bool]:
    """Return ``(best_score, found_certain_win)`` reachable from ``node``."""
    obs = node.observation
    child_state = obs.current

    if child_state.result == me:
        return 1.0, True
    if child_state.result != -1:
        return 0.0, False  # opponent won: worst possible outcome
    if child_state.yourIndex != me:
        return _leaf_score(child_state), False  # turn ended / control moved
    if obs.select is None or not obs.select.option:
        return _leaf_score(child_state), False
    if depth_left <= 0:
        return _leaf_score(child_state), False

    key = _state_key(obs)
    cached = visited.get(key)
    if cached is not None and cached[0] >= depth_left:
        return cached[1], cached[2]

    best_score = _leaf_score(child_state)
    found_win = False
    for selection in _candidate_selections(obs.select, config):
        if time.perf_counter() > deadline:
            raise _SearchAbort("time")
        if budget["nodes"] >= int(config["max_nodes"]):
            raise _SearchAbort("nodes")
        budget["nodes"] += 1

        try:
            grandchild = cg_api.search_step(node.searchId, selection)
        except ValueError:
            continue
        try:
            score, win = _dfs_score(
                grandchild, me, depth_left - 1, config, deadline, budget, visited
            )
        finally:
            try:
                cg_api.search_release(grandchild.searchId)
            except Exception:
                pass

        if score > best_score:
            best_score = score
        if win:
            found_win = True
            break  # 1.0 is the maximum possible score; no need to check siblings

    visited[key] = (depth_left, best_score, found_win)
    return best_score, found_win


def _hidden_state_factory(context: dict) -> Callable[[], dict | None] | None:
    """Resolve the caller-provided hidden state; None if none was given."""
    factory = context.get("hidden_state_factory")
    if factory is not None:
        return factory
    hidden_state = context.get("hidden_state")
    if hidden_state is not None:
        return lambda: hidden_state
    return None


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


def _candidate_selections(select: SelectData, config: dict) -> Iterator[list[int]]:
    """Generate index selections satisfying min/max count, no duplicates.

    Copied from ``lethal_simple._candidate_selections`` (see module
    docstring: kept independent on purpose). MAIN options are reordered by
    ``_MAIN_OPTION_PRIORITY`` and END is pruned. Output is capped by
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


def _begin(obs: Observation, hidden_state: dict):
    return cg_api.search_begin(
        obs,
        hidden_state["your_deck"],
        hidden_state["your_prize"],
        hidden_state["opponent_deck"],
        hidden_state["opponent_prize"],
        hidden_state["opponent_hand"],
        hidden_state["opponent_active"],
    )


def _state_key(obs: Observation) -> str:
    # Dataclass repr is deterministic for identical states; good enough to
    # suppress re-exploring transpositions within one determinization.
    return f"{obs.current!r}|{obs.select!r}"
