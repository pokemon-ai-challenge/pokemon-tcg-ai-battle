"""Beam search over the current own turn (design: ``test_plan/ptcg_search_design.md``).

Unlike ``lethal_simple`` (which only ever confirms *certain* lethal lines and
prunes everything else), this module searches for the *best* action sequence
within the current own turn -- ending either when control passes to the
opponent (an ``END`` was chosen, or the last action ends the turn
automatically) or when the game itself ends. It does not model the
opponent's turn at all (design §2.1): once control leaves us, whatever state
we reached is evaluated as a leaf.

Full-width search is not affordable (design §1: 7^7 branches * 5.2ms would be
~4,282s per decision, ~300x the 14.4s/decision budget). This module instead
keeps, at every depth, only the top ``beam_width`` partial lines ranked by
the learned policy's score (``ptcg_ai.learning.policy_model``) -- cost is
then linear in depth instead of exponential (design §2.2/§2.4):
``beam_width * depth * branching * 0.745ms``.

## What gets returned

A "line" here is a sequence of ``search_step`` selections starting from the
*current* MAIN decision. Only the very first selection of the best line is
ever returned (design §2.2: "返すのは「今この瞬間の decision に対する1手」")
-- the rest of the line is only used to decide which first move is best.
Concretely this means the tree is bucketed by "which option did we pick at
the root", and only the best leaf value reachable under each bucket is
tracked; nothing about the middle of the line is kept once traversed.

## Leaf evaluation: real values, not the learned value function (§2.3)

The learned value model is not used here. It is only about as strong as the
prize difference alone (AUC 0.719 vs 0.696 per the design doc), so there is
no reason to mix in a weak estimate when simulating our own turn to the end
gives us the *exact* resulting board. See ``evaluate_leaf`` /
``DEFAULT_LEAF_WEIGHTS`` below for the four confirmed-value components and
why their weights are separated by orders of magnitude.

## Failure handling

Every public entry point (``search``) never raises: gating failures, a
missing/failed hidden-state factory, ``search_step`` failures (~1.2% per the
design doc's measurement, cause unknown -- design §5 risk #1), and time/node
budget exhaustion all degrade to returning ``None`` (or, for exhausted
budgets, the best move found so far -- see "budget exhaustion" below), never
an exception. The caller (``action_selection.selector``) falls through to
the ML policy / rule-based router when this happens, exactly like
``lethal_simple``.

## Budget exhaustion: never discard partial progress

Per design §4 stage 1's acceptance criterion, hitting ``time_limit_ms`` or
``max_depth`` must still return the best move found *so far*, not ``None``.
Nodes still open in the beam frontier when the deadline/depth limit is hit
are evaluated in place (as if the turn had ended there) instead of being
thrown away.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Callable, Iterator

from cg import api as cg_api
from cg.api import Observation, PlayerState, Pokemon, SelectContext, SelectData, SelectType, State

from ptcg_ai.board_evaluation import board_features
from ptcg_ai.learning.observable_state import observable_state
from ptcg_ai.learning.policy_model import PolicyModel
from ptcg_ai.learning.semantic_action import resolve_option

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS: dict = {
    "enabled": False,
    # Design §2.2/§2.4: 30 keeps one decision to ~1.1s (30 * 7 * 7 * 0.745ms),
    # ~45s/game against the measured median of 7 MAIN options / 7 steps to
    # END -- 12x headroom under the 600s/game budget. Raise only after §4
    # stage 2 shows a win-rate gain at width 30 (per design, "not worth it
    # otherwise").
    "beam_width": 30,
    # Design §5 risk #5: the 600s/game (14.4s/decision) budget comes from
    # replay analysis; the local simulator cannot observe
    # remainingOverageTime, so it cannot be verified locally. Start
    # conservative (design explicitly suggests ~3s as the starting point)
    # and only raise once real submissions show headroom.
    "time_limit_ms": 3000,
    # A full own turn is a median 7 / max 24 steps (design §1); 20 covers
    # the common case with room to spare without letting a pathological
    # loop run unbounded.
    "max_depth": 20,
    # Bounds combinatorial explosion on multi-select nodes (e.g. "discard 2
    # of 5 cards") the same way lethal_simple's cap does. Kept lower than
    # lethal_simple's 128 because here every candidate also gets stepped
    # through search_step *before* the beam-width prune can discard it, and
    # the cost is multiplied by beam_width across the whole frontier.
    "max_combinations_per_select": 64,
    # Design §2.7: the opponent's hidden hand/deck are decretized per
    # search_begin call, so a single sample is biased by that one shuffle.
    # Averaging leaf values over several samples reduces that bias, but
    # each extra sample re-runs the whole beam search (roughly another
    # beam_width * depth * branching * 0.745ms), so it trades directly
    # against time_limit_ms. Default 1: time budget conservatism (see
    # time_limit_ms above) comes first for stage 1; §4 stage 3 is where
    # this gets tuned against measured win rate.
    "hidden_state_samples": 1,
}

# Leaf-evaluation weights (design §2.3 priority order: prizes taken > damage
# dealt > KO risk avoided > board development). Each tier's weight is chosen
# to be at least ~10x the maximum plausible total contribution of every
# lower tier combined, so ties are broken lexicographically in practice
# (a single extra prize always outweighs any amount of damage/KO-risk/
# development difference; a single HP point of extra damage always
# outweighs KO risk + development, etc.) without needing an actual
# lexicographic comparison (a plain weighted sum is simpler to reason about
# and to override partially via config).
DEFAULT_LEAF_WEIGHTS: dict = {
    # Priority 1: prizes taken this turn -- the win condition itself.
    # damage_dealt tops out around a few hundred HP (a handful of attacks
    # in one turn at most) * 1,000 = a few hundred thousand; one prize
    # (1,000,000) still exceeds that comfortably, and a 2nd prize taken
    # dwarfs it further.
    "prizes_taken": 1_000_000.0,
    # Priority 2: total damage dealt to the opponent's Pokemon this turn
    # (HP points, confirmed from the simulated end state). KO risk (0/1)
    # and development (bench count + energy count, tens at most) together
    # max out in the low hundreds; 1,000 per HP point keeps even a single
    # point of extra damage decisive over both.
    "damage_dealt": 1_000.0,
    # Priority 3: would our active likely be knocked out next opponent
    # turn (board_features.is_likely_ko_next_turn, reused as-is rather than
    # reimplemented -- it is exactly the "1手先の相手の攻撃" check design
    # §2.3 asks for). Penalty; magnitude chosen to exceed development's max
    # (bench <=5 + energy count, rarely above a few dozen).
    "ko_risk": -100.0,
    # Priority 4: board development (bench count + total energy in play).
    # Pure tie-breaker among lines that are otherwise identical on the
    # first three, confirmed-value components.
    "development": 1.0,
}

# Sentinels for definite win/loss leaves (State.result is the winning
# player index, -1 while undecided). Not part of DEFAULT_LEAF_WEIGHTS
# because they are not a linear function of any single feature -- an
# outright win must outrank every possible combination of the other
# components, and an outright loss must be avoided over all of them.
_WIN_SCORE = 1e12
_LOSE_SCORE = -1e12

_POLICY_MODEL_CACHE: PolicyModel | None = None


def _policy_model() -> PolicyModel:
    global _POLICY_MODEL_CACHE
    if _POLICY_MODEL_CACHE is None:
        _POLICY_MODEL_CACHE = PolicyModel()
    return _POLICY_MODEL_CACHE


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

_STATS_ZERO = {
    "searches": 0,            # gates passed, a search was actually attempted
    "returned": 0,            # a move was returned (beam found >=1 leaf)
    "none_returned": 0,       # None was returned (gate failure or no leaves)
    "leaves_evaluated": 0,    # cumulative leaves evaluated across all calls
    "max_depth_reached": 0,   # high-water mark across all calls
    "total_depth_reached": 0, # sum, for computing an average depth
    "step_failures": 0,       # cumulative search_step ValueError/exceptions
    "total_time_ms": 0.0,
    "max_time_ms": 0.0,
}
_stats = dict(_STATS_ZERO)


def get_stats() -> dict:
    """Return cumulative turn_beam statistics (see ``_STATS_ZERO``)."""
    stats = dict(_stats)
    searches = stats["searches"]
    stats["avg_time_ms"] = stats["total_time_ms"] / searches if searches else 0.0
    stats["avg_depth_reached"] = stats["total_depth_reached"] / searches if searches else 0.0
    return stats


def reset_stats() -> None:
    _stats.update(_STATS_ZERO)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def search(
    obs: Observation,
    config: dict | None,
    hidden_state_factory: Callable[[], dict | None] | None,
    policy_model: PolicyModel | None = None,
) -> list[int] | None:
    """Beam-search the current own turn; return the best first selection.

    Args:
        obs: The observation for the current decision. Only run when
            ``obs.select.context == SelectContext.MAIN`` (design: "配線先
            ... SelectContext.MAIN のときだけ動かす"); anything else
            returns None immediately without touching stats.
        config: The ``turn_search`` section of the agent config (missing
            keys fall back to ``DEFAULTS``).
        hidden_state_factory: Zero-argument callable building a
            ``search_begin()``-ready dict (same contract as
            ``lethal_simple``'s ``hidden_state_factory``/``hidden_state``),
            or None. Called once per hidden-state sample (design §2.7);
            each call may resample, which is what lets averaging over
            ``hidden_state_samples`` reduce shuffle bias.
        policy_model: Optional injected ``PolicyModel`` (mainly for tests).
            Defaults to a module-level cached instance, same pattern as
            ``action_selection.selector._policy_model``.

    Returns:
        list[int] | None: The first selection of the best line found, or
        None on any gate failure, exception, or a genuinely empty search
        (never partial/garbage data).
    """
    try:
        cfg = {**DEFAULTS, **(config or {})}
        if not cfg.get("enabled", False):
            return None
        if obs is None or obs.select is None or obs.current is None:
            return None
        if obs.select.context != SelectContext.MAIN:
            return None
        if hidden_state_factory is None:
            return None

        me = obs.current.yourIndex
        policy = policy_model if policy_model is not None else _policy_model()

        start_time = time.perf_counter()
        _stats["searches"] += 1
        deadline = start_time + float(cfg["time_limit_ms"]) / 1000.0
        try:
            action, info = _search_with_samples(obs, me, cfg, hidden_state_factory, policy, deadline)
            _stats["leaves_evaluated"] += info["leaves"]
            _stats["max_depth_reached"] = max(_stats["max_depth_reached"], info["max_depth"])
            _stats["total_depth_reached"] += info["max_depth"]
            _stats["step_failures"] += info["step_failures"]
            if action is None:
                _stats["none_returned"] += 1
                return None
            _stats["returned"] += 1
            return action
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


# --------------------------------------------------------------------------
# Sampling over decretized hidden state (§2.7)
# --------------------------------------------------------------------------

def _search_with_samples(
    obs: Observation,
    me: int,
    cfg: dict,
    hidden_state_factory: Callable[[], dict | None],
    policy: PolicyModel,
    deadline: float,
) -> tuple[list[int] | None, dict]:
    """Run one beam search per hidden-state sample and average leaf scores
    per root-level selection. Returns (best first selection or None, info).
    """
    samples_wanted = max(1, int(cfg.get("hidden_state_samples", 1)))
    sums: dict[tuple[int, ...], float] = {}
    counts: dict[tuple[int, ...], int] = {}
    leaves = 0
    max_depth = 0
    step_failures = 0

    for _ in range(samples_wanted):
        if time.perf_counter() > deadline:
            break
        try:
            hidden_state = hidden_state_factory()
        except Exception:
            continue
        if hidden_state is None:
            continue
        try:
            best, sample_leaves, depth_reached, sample_failures = _run_beam(
                obs, me, hidden_state, cfg, deadline, policy
            )
        except Exception:
            continue
        leaves += sample_leaves
        max_depth = max(max_depth, depth_reached)
        step_failures += sample_failures
        for key, value in best.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    info = {"leaves": leaves, "max_depth": max_depth, "step_failures": step_failures}
    if not sums:
        return None, info
    best_key = max(sums, key=lambda k: sums[k] / counts[k])
    return list(best_key), info


# --------------------------------------------------------------------------
# One beam search under a single decretized hidden state
# --------------------------------------------------------------------------

def _run_beam(
    obs: Observation,
    me: int,
    hidden_state: dict,
    cfg: dict,
    deadline: float,
    policy: PolicyModel,
) -> tuple[dict[tuple[int, ...], float], int, int, int]:
    """Beam search from ``obs``'s current MAIN decision under one hidden
    state. Returns (best leaf score per root-level selection, leaves
    evaluated, depth reached, search_step failures).
    """
    root = cg_api.search_begin(
        obs,
        hidden_state["your_deck"],
        hidden_state["your_prize"],
        hidden_state["opponent_deck"],
        hidden_state["opponent_prize"],
        hidden_state["opponent_hand"],
        hidden_state["opponent_active"],
    )
    root_state = obs.current
    beam_width = max(1, int(cfg.get("beam_width", DEFAULTS["beam_width"])))
    max_depth = max(1, int(cfg.get("max_depth", DEFAULTS["max_depth"])))
    weights = {**DEFAULT_LEAF_WEIGHTS, **(cfg.get("leaf_weights") or {})}

    live_ids: set[int] = {root.searchId}
    best: dict[tuple[int, ...], float] = {}
    leaves = 0
    step_failures = 0
    depth_reached = 0

    def release(search_id: int) -> None:
        if search_id in live_ids:
            live_ids.discard(search_id)
            try:
                cg_api.search_release(search_id)
            except Exception:
                pass

    def record(first_action: tuple[int, ...] | None, score: float) -> None:
        nonlocal leaves
        if first_action is None:
            return
        leaves += 1
        prev = best.get(first_action)
        if prev is None or score > prev:
            best[first_action] = score

    frontier = [
        {
            "search_id": root.searchId,
            "obs": root.observation,
            "first_action": None,
            "score": 0.0,
        }
    ]

    try:
        depth = 0
        while frontier and depth < max_depth:
            if time.perf_counter() > deadline:
                break
            next_frontier: list[dict] = []
            for node in frontier:
                if time.perf_counter() > deadline:
                    break
                n_obs = node["obs"]
                select = n_obs.select
                if select is None or not select.option:
                    record(node["first_action"], evaluate_leaf(root_state, n_obs.current, me, weights))
                    release(node["search_id"])
                    continue

                candidates = list(_candidate_selections(select, cfg))
                branch_scores = _branch_scores(n_obs, me, select, candidates, policy)
                for selection, branch_score in zip(candidates, branch_scores):
                    if time.perf_counter() > deadline:
                        break
                    try:
                        child = cg_api.search_step(node["search_id"], selection)
                    except Exception:
                        step_failures += 1
                        continue

                    live_ids.add(child.searchId)
                    child_state = child.observation.current
                    first_action = (
                        node["first_action"] if node["first_action"] is not None else tuple(selection)
                    )

                    if child_state.result == me:
                        record(first_action, _WIN_SCORE)
                        release(child.searchId)
                        continue
                    if child_state.result != -1:
                        record(first_action, _LOSE_SCORE)
                        release(child.searchId)
                        continue
                    if child_state.yourIndex != me:
                        record(first_action, evaluate_leaf(root_state, child_state, me, weights))
                        release(child.searchId)
                        continue

                    next_frontier.append(
                        {
                            "search_id": child.searchId,
                            "obs": child.observation,
                            "first_action": first_action,
                            "score": node["score"] + branch_score,
                        }
                    )
                release(node["search_id"])

            next_frontier.sort(key=lambda n: n["score"], reverse=True)
            kept, dropped = next_frontier[:beam_width], next_frontier[beam_width:]
            for dropped_node in dropped:
                release(dropped_node["search_id"])
            frontier = kept
            depth += 1
            depth_reached = max(depth_reached, depth)

        # Budget exhausted (time or max_depth) with nodes still open:
        # evaluate them where they stand rather than discarding progress.
        for node in frontier:
            record(node["first_action"], evaluate_leaf(root_state, node["obs"].current, me, weights))
            release(node["search_id"])

        return best, leaves, depth_reached, step_failures
    finally:
        for search_id in list(live_ids):
            release(search_id)


def _candidate_selections(select: SelectData, cfg: dict) -> Iterator[list[int]]:
    """Generate index selections satisfying min/max count, no duplicates.

    Unlike ``lethal_simple``'s version, END is not pruned and MAIN options
    are not reordered by a hand-written priority table: the branch's
    priority for beam pruning comes from the policy score computed
    separately in ``_branch_scores`` and applied via the global
    depth-frontier sort in ``_run_beam``, not from generation order here.
    """
    order = list(range(len(select.option)))
    min_count = max(select.minCount, 0)
    max_count = min(select.maxCount, len(order))
    limit = int(cfg.get("max_combinations_per_select", DEFAULTS["max_combinations_per_select"]))
    produced = 0
    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(order, count):
            yield list(combo)
            produced += 1
            if produced >= limit:
                return


def _branch_scores(
    obs: Observation,
    me: int,
    select: SelectData,
    candidates: list[list[int]],
    policy: PolicyModel,
) -> list[float]:
    """Score each candidate selection with the learned policy prior, for
    beam ranking only (never used as the leaf evaluation).

    Only meaningful for ``SelectType.MAIN`` nodes: the policy prior
    (``policy_model.py`` / ``policy_features.py``) was trained on MAIN
    decisions exclusively. Non-MAIN nodes (sub-selections needed to
    complete a MAIN action, e.g. choosing an attack target) get a neutral
    0.0 for every candidate -- their relative order does not matter beyond
    being subject to the same beam-width prune as everything else at that
    depth.
    """
    if select.type != SelectType.MAIN or not getattr(policy, "is_ready", False):
        return [0.0] * len(candidates)
    try:
        current_dict = dataclasses.asdict(obs.current)
        state = observable_state(current_dict, me)
        actions = [
            resolve_option(dataclasses.asdict(option), current_dict, me) for option in select.option
        ]
        option_scores = policy.score_options(state, actions)
    except Exception:
        return [0.0] * len(candidates)
    return [sum(option_scores[i] for i in combo) if combo else 0.0 for combo in candidates]


# --------------------------------------------------------------------------
# Leaf evaluation (§2.3): confirmed values only, no learned value function
# --------------------------------------------------------------------------

def evaluate_leaf(
    root_state: State,
    leaf_state: State,
    me: int,
    weights: dict | None = None,
) -> float:
    """Score a leaf state reached by simulating our own turn.

    All four components are confirmed values computed from the simulated
    end state (design §2.3), not learned estimates:

    1. ``prizes_taken`` -- prizes we took this turn (``root`` vs ``leaf``).
    2. ``damage_dealt`` -- total HP damage dealt to the opponent's Pokemon
       this turn (matched by ``serial`` across ``root``/``leaf`` so a
       Pokemon knocked out entirely still counts its full remaining HP).
    3. ``ko_risk`` -- would our active likely be knocked out next
       opponent turn, via ``board_features.is_likely_ko_next_turn`` (reused
       unmodified; it already implements exactly this "1手先の相手の攻撃"
       check against the decretized opponent board).
    4. ``development`` -- bench count + total energy in play, a tie-breaker
       only.
    """
    weights = weights or DEFAULT_LEAF_WEIGHTS
    opponent = 1 - me

    prizes = _prizes_taken(root_state, leaf_state, me)
    damage = _damage_dealt(root_state.players[opponent], leaf_state.players[opponent])
    ko_risk = _ko_risk(leaf_state, me)
    development = _development_score(leaf_state, me)

    return (
        weights.get("prizes_taken", 0.0) * prizes
        + weights.get("damage_dealt", 0.0) * damage
        + weights.get("ko_risk", 0.0) * ko_risk
        + weights.get("development", 0.0) * development
    )


def _prizes_taken(root_state: State, leaf_state: State, me: int) -> int:
    """Prizes we took between ``root_state`` and ``leaf_state`` (our own
    ``prize`` list shrinks as we take prizes -- see ``cg/api.py``)."""
    return len(root_state.players[me].prize) - len(leaf_state.players[me].prize)


def _pokemon_by_serial(player_state: PlayerState) -> dict[int, Pokemon]:
    result: dict[int, Pokemon] = {}
    for mon in list(player_state.active or []) + list(player_state.bench or []):
        if mon is not None:
            result[mon.serial] = mon
    return result


def _damage_dealt(opponent_root: PlayerState, opponent_leaf: PlayerState) -> int:
    """Total HP damage dealt to the opponent's in-play Pokemon.

    Matched by ``serial`` so a Pokemon that was knocked out (and thus
    disappeared from ``active``/``bench``) still contributes its full
    remaining HP at ``root`` as damage dealt, instead of silently vanishing
    from the total.
    """
    root_map = _pokemon_by_serial(opponent_root)
    leaf_map = _pokemon_by_serial(opponent_leaf)
    total = 0
    for serial, mon in root_map.items():
        leaf_mon = leaf_map.get(serial)
        if leaf_mon is None:
            total += mon.hp
        else:
            total += max(0, mon.hp - leaf_mon.hp)
    return total


def _ko_risk(leaf_state: State, me: int) -> float:
    player = leaf_state.players[me]
    if not player.active or player.active[0] is None:
        return 0.0
    return 1.0 if board_features.is_likely_ko_next_turn(player.active[0], leaf_state, me) else 0.0


def _development_score(state: State, me: int) -> float:
    player = state.players[me]
    in_play = list(player.bench)
    if player.active and player.active[0] is not None:
        in_play.append(player.active[0])
    total_energy = sum(len(mon.energies) for mon in in_play)
    return float(len(player.bench) + total_energy)
