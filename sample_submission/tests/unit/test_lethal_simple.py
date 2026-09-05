"""Unit tests for ptcg_ai.search.lethal_simple (issue #57).

The competition search API is replaced by a small fake engine so the
search logic (iterative deepening, pruning, budgets, fallback) can be
tested without a real battle.

Run from sample_submission/:
    python -m pytest tests/unit/test_lethal_simple.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import (
    AreaType,
    Card,
    Observation,
    Option,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    SelectData,
    SelectType,
    State,
)
from ptcg_ai.action_selection import selector
from ptcg_ai.hidden_information import search_state_stub
from ptcg_ai.search import lethal_simple


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_player(prizes: int = 1, deck_count: int = 10, hand_count: int = 0) -> PlayerState:
    return PlayerState(
        active=[],
        bench=[],
        benchMax=5,
        deckCount=deck_count,
        discard=[],
        prize=[None] * prizes,
        handCount=hand_count,
        hand=[],
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def make_state(
    *,
    your_index: int = 0,
    turn: int = 1,
    first_player: int = 0,
    result: int = -1,
    my_prizes: int = 1,
    action_count: int = 0,
) -> State:
    mine = make_player(prizes=my_prizes)
    theirs = make_player(prizes=3)
    players = [mine, theirs] if your_index == 0 else [theirs, mine]
    return State(
        turn=turn,
        turnActionCount=action_count,
        yourIndex=your_index,
        firstPlayer=first_player,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=result,
        stadium=[],
        looking=None,
        players=players,
    )


def make_select(
    option_types: list[OptionType],
    *,
    select_type: SelectType = SelectType.MAIN,
    min_count: int = 1,
    max_count: int = 1,
) -> SelectData:
    return SelectData(
        type=select_type,
        context=SelectContext.MAIN,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=[Option(type=t) for t in option_types],
        deck=None,
        contextCard=None,
        effect=None,
    )


def make_obs(state: State, select: SelectData | None) -> Observation:
    return Observation(select=select, logs=[], current=state, search_begin_input="{}")


def make_opt_select(
    options: list[Option],
    *,
    select_type: SelectType = SelectType.MAIN,
    min_count: int = 1,
    max_count: int = 1,
) -> SelectData:
    """Like ``make_select`` but takes pre-built ``Option`` objects, so a
    PLAY option can carry the ``index`` that maps it to a hand card."""
    return SelectData(
        type=select_type,
        context=SelectContext.MAIN,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=list(options),
        deck=None,
        contextCard=None,
        effect=None,
    )


def set_hand(state: State, card_ids: list[int], your_index: int = 0) -> State:
    state.players[your_index].hand = [
        Card(id=c, serial=i, playerIndex=your_index) for i, c in enumerate(card_ids)
    ]
    return state


DUMMY_HIDDEN_STATE = {
    "your_deck": [],
    "your_prize": [],
    "opponent_deck": [],
    "opponent_prize": [],
    "opponent_hand": [],
    "opponent_active": [],
}


# ---------------------------------------------------------------------------
# Fake search engine
# ---------------------------------------------------------------------------

class FakeNode:
    def __init__(self, obs: Observation, transitions: dict | None = None):
        self.obs = obs
        # keys: tuple(sorted(selection)); values: next node name
        self.transitions = transitions or {}


class FakeEngine:
    """Mimics cg.api search_* using a hand-built decision tree."""

    def __init__(self, nodes: dict[str, FakeNode], root: str = "root", default: str | None = None):
        self.nodes = nodes
        self.roots = [root] if isinstance(root, str) else list(root)
        self.default = default
        self.next_id = 0
        self.id_to_name: dict[int, str] = {}
        self.released: set[int] = set()
        self.begin_count = 0
        self.end_count = 0
        self.step_log: list[tuple[str, tuple[int, ...]]] = []

    def _make_state(self, name: str) -> SearchState:
        self.next_id += 1
        self.id_to_name[self.next_id] = name
        return SearchState(observation=self.nodes[name].obs, searchId=self.next_id)

    def search_begin(self, obs, *args, **kwargs) -> SearchState:
        root = self.roots[min(self.begin_count, len(self.roots) - 1)]
        self.begin_count += 1
        return self._make_state(root)

    def search_step(self, search_id: int, select: list[int]) -> SearchState:
        if search_id in self.released:
            raise ValueError("Released item.")
        name = self.id_to_name[search_id]
        node = self.nodes[name]
        obs = node.obs
        if obs.current.result != -1:
            raise ValueError("Cannot be selected because the battle has ended.")
        sel = obs.select
        if not (sel.minCount <= len(select) <= sel.maxCount):
            raise ValueError("count out of range")
        if any(not (0 <= i < len(sel.option)) for i in select):
            raise ValueError("index out of range")
        if len(select) != len(set(select)):
            raise ValueError("duplicate")
        self.step_log.append((name, tuple(select)))
        key = tuple(sorted(select))
        target = node.transitions.get(key, self.default)
        if target is None:
            raise AssertionError(f"no transition from {name} for {key}")
        return self._make_state(target)

    def search_end(self) -> None:
        self.end_count += 1

    def search_release(self, search_id: int) -> None:
        self.released.add(search_id)


@pytest.fixture
def install_engine(monkeypatch):
    def _install(engine: FakeEngine):
        monkeypatch.setattr(lethal_simple, "cg_api", engine)
        return engine
    return _install


def base_context(obs: Observation, **config) -> dict:
    cfg = {"enabled": True, "verify_shuffles": 1, "time_limit_ms": 1000}
    cfg.update(config)
    return {"observation": obs, "config": cfg, "hidden_state": DUMMY_HIDDEN_STATE}


def run_search(obs: Observation, **config):
    return lethal_simple.search(obs.current, obs.select.option, base_context(obs, **config))


WIN = FakeNode(make_obs(make_state(result=0, action_count=90), None))
OPP_TURN = FakeNode(
    make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=91),
        make_select([OptionType.END]),
    )
)


# ---------------------------------------------------------------------------
# Lethal detection
# ---------------------------------------------------------------------------

def test_attack_now_wins(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [0]
    assert engine.end_count == 1


def test_energy_attach_then_attack_wins(install_engine):
    # Direct attack does not win; attach energy first, then attack.
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(
        make_state(action_count=1),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    engine = install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
            "mid": FakeNode(mid_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_ability_then_attack_wins(install_engine):
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ABILITY, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(
        make_state(action_count=1),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
            "mid": FakeNode(mid_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_retreat_choose_new_active_then_attack_wins(install_engine):
    # Retreat -> CARD selection for the new Active -> attack. Covers a
    # multi-step line that goes through a non-MAIN selection.
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.RETREAT, OptionType.ATTACK, OptionType.END]),
    )
    choose_obs = make_obs(
        make_state(action_count=1),
        make_select([OptionType.CARD, OptionType.CARD], select_type=SelectType.CARD),
    )
    after_obs = make_obs(
        make_state(action_count=2),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "choose", (1,): "opp"}),
            "choose": FakeNode(choose_obs, {(0,): "opp2", (1,): "after"}),
            "after": FakeNode(after_obs, {(0,): "win"}),
            "opp2": FakeNode(make_obs(
                make_state(action_count=5),
                make_select([OptionType.ATTACK, OptionType.END]),
            ), {}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_multi_count_selection_handles_min_max_and_duplicates(install_engine):
    # A 2-of-3 card selection: only the pair (0, 2) leads to the win.
    root_obs = make_obs(
        make_state(),
        make_select(
            [OptionType.CARD, OptionType.CARD, OptionType.CARD],
            select_type=SelectType.CARD,
            min_count=2,
            max_count=2,
        ),
    )
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0, 2): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    action = run_search(root_obs)
    assert action is not None
    assert sorted(action) == [0, 2]
    assert len(action) == len(set(action))


def test_attack_tried_before_end(install_engine):
    # END is listed first, but the attack option must be tried first.
    root_obs = make_obs(make_state(), make_select([OptionType.END, OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(1,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [1]
    assert engine.step_log[0] == ("root", (1,))


# ---------------------------------------------------------------------------
# No lethal / fallback to None
# ---------------------------------------------------------------------------

def test_no_win_returns_none(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {}), "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, max_depth=5) is None
    assert engine.end_count == 1


def test_attack_without_win_returns_none(install_engine):
    # The attack resolves but the game continues on the opponent's side.
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "opp"}), "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, max_depth=5) is None


def test_opponent_win_branch_is_pruned(install_engine):
    lose = FakeNode(make_obs(make_state(result=1, action_count=99), None))
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "lose"}), "lose": lose, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, max_depth=5) is None


def test_time_limit_returns_none(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, time_limit_ms=0) is None
    assert engine.end_count == 1  # search_end still runs


def test_node_limit_returns_none(install_engine):
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(
        make_state(action_count=1),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
            "mid": FakeNode(mid_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    assert run_search(root_obs, max_nodes=1) is None


# ---------------------------------------------------------------------------
# Gates (config / prize / turn)
# ---------------------------------------------------------------------------

def test_disabled_config_skips_search(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs, enabled=False) is None
    assert engine.begin_count == 0


def test_over_prize_threshold_skips_search(install_engine):
    # Phase 2 default threshold is 2 remaining prizes; 3 must not search.
    root_obs = make_obs(make_state(my_prizes=3), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs) is None
    assert engine.begin_count == 0


def test_phase1_threshold_still_configurable(install_engine):
    # max_remaining_prizes=1 (Phase 1 behavior) must keep gating at 2.
    root_obs = make_obs(make_state(my_prizes=2), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs, max_remaining_prizes=1) is None
    assert engine.begin_count == 0


def test_not_my_turn_skips_search(install_engine):
    # turn=2 with firstPlayer == me (0) means it is the opponent's turn.
    root_obs = make_obs(make_state(turn=2), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs) is None
    assert engine.begin_count == 0


def test_missing_hidden_state_skips_search(install_engine):
    # The search never builds hidden information itself: without a
    # hidden_state / hidden_state_factory in the context it must bail out.
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    context = {"observation": root_obs, "config": {"enabled": True}}
    assert lethal_simple.search(root_obs.current, root_obs.select.option, context) is None
    assert engine.begin_count == 0


# ---------------------------------------------------------------------------
# Phase 2 (#58): two remaining prizes
# ---------------------------------------------------------------------------

def test_two_prizes_ex_knockout_wins(install_engine):
    # KOing a Pokemon ex takes both remaining prizes -> State.result flips.
    root_obs = make_obs(
        make_state(my_prizes=2), make_select([OptionType.ATTACK, OptionType.END])
    )
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_two_prizes_normal_knockout_is_not_lethal(install_engine):
    # A normal KO only takes 1 of the 2 prizes: the game continues on the
    # opponent's side, so no lethal must be claimed.
    root_obs = make_obs(
        make_state(my_prizes=2), make_select([OptionType.ATTACK, OptionType.END])
    )
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "opp"}), "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, max_depth=5) is None


def test_two_prizes_multi_knockout_selection_wins(install_engine):
    # Attack -> choose 2 damage targets -> both KO'd -> win. Covers
    # effects that knock out multiple Pokemon at once.
    root_obs = make_obs(
        make_state(my_prizes=2), make_select([OptionType.ATTACK, OptionType.END])
    )
    targets_obs = make_obs(
        make_state(my_prizes=2, action_count=1),
        make_select(
            [OptionType.CARD, OptionType.CARD, OptionType.CARD],
            select_type=SelectType.CARD,
            min_count=2,
            max_count=2,
        ),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "targets"}),
            "targets": FakeNode(targets_obs, {(1, 2): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_two_prizes_card_effect_win_without_attack(install_engine):
    # A win produced by playing a card (no attack) must also be found.
    root_obs = make_obs(
        make_state(my_prizes=2),
        make_select([OptionType.PLAY, OptionType.ATTACK, OptionType.END]),
    )
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [0]


def test_end_option_is_never_stepped(install_engine):
    # END cannot lead to a win within the own turn, so it must be pruned
    # from MAIN candidates entirely.
    root_obs = make_obs(make_state(), make_select([OptionType.END, OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {}), "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, max_depth=5) is None
    assert ("root", (0,)) not in engine.step_log


def test_stats_record_searches_and_findings(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    lethal_simple.reset_stats()
    assert run_search(root_obs) == [0]
    stats = lethal_simple.get_stats()
    assert stats["searches"] == 1
    assert stats["found"] == 1
    assert stats["timeouts"] == 0
    assert stats["max_time_ms"] >= 0.0
    assert stats["avg_time_ms"] == stats["total_time_ms"]


def test_stats_record_timeouts(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    lethal_simple.reset_stats()
    assert run_search(root_obs, time_limit_ms=0) is None
    stats = lethal_simple.get_stats()
    assert stats["searches"] == 1
    assert stats["found"] == 0
    assert stats["timeouts"] == 1


# ---------------------------------------------------------------------------
# Gust prioritization (Boss's Orders): try the gust Trainer right after
# attacks so "gust an ex Active, then KO it" lethals are reached before the
# time/node budget is spent on the rest of a big hand.
# ---------------------------------------------------------------------------

GUST_ID = 1182  # Boss's Orders


def _root_gust_scenario() -> tuple[Observation, FakeEngine]:
    """A root where attacking now loses tempo but Boss's Orders (option 3,
    hand[0]) drags a benched ex Active and the follow-up attack wins."""
    root_obs = make_obs(
        set_hand(make_state(my_prizes=2), [GUST_ID, 700, 701, 702]),
        make_opt_select([
            Option(type=OptionType.ATTACH, index=1),  # 0 decoy
            Option(type=OptionType.EVOLVE, index=2),  # 1 decoy
            Option(type=OptionType.PLAY, index=3),    # 2 non-gust play decoy
            Option(type=OptionType.PLAY, index=0),    # 3 gust (hand[0] == 1182)
            Option(type=OptionType.ATTACK),           # 4 attack (does not win)
            Option(type=OptionType.END),              # 5
        ]),
    )
    target_obs = make_obs(
        make_state(my_prizes=2, action_count=1),
        make_opt_select(
            [Option(type=OptionType.CARD), Option(type=OptionType.CARD)],
            select_type=SelectType.CARD,
        ),
    )
    after_obs = make_obs(
        make_state(my_prizes=2, action_count=2),
        make_opt_select([Option(type=OptionType.ATTACK), Option(type=OptionType.END)]),
    )
    engine = FakeEngine(
        {
            "root": FakeNode(root_obs, {(3,): "target", (4,): "opp"}),
            "target": FakeNode(target_obs, {(1,): "after", (0,): "opp"}),
            "after": FakeNode(after_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    )
    return root_obs, engine


def _first_step(step_log, entry) -> float:
    return step_log.index(entry) if entry in step_log else float("inf")


def test_candidate_selections_orders_gust_right_after_attack():
    state = set_hand(make_state(my_prizes=2), [GUST_ID, 999])
    select = make_opt_select([
        Option(type=OptionType.ATTACH, index=1),  # 0
        Option(type=OptionType.PLAY, index=1),    # 1 non-gust play
        Option(type=OptionType.PLAY, index=0),    # 2 gust play
        Option(type=OptionType.EVOLVE, index=1),  # 3
        Option(type=OptionType.ATTACK),           # 4
        Option(type=OptionType.END),              # 5
    ])
    order = [
        sel[0]
        for sel in lethal_simple._candidate_selections(
            select,
            {"max_combinations_per_select": 128, "priority_play_card_ids": [GUST_ID]},
            state,
        )
    ]
    assert 5 not in order      # END pruned
    assert order[0] == 4       # ATTACK still first
    assert order[1] == 2       # gust PLAY right after the attack
    assert order.index(2) < order.index(1)  # before the non-gust PLAY
    assert order.index(2) < order.index(0)  # before ATTACH
    assert order.index(2) < order.index(3)  # before EVOLVE


def test_candidate_selections_keeps_normal_order_without_gust_ids():
    state = set_hand(make_state(my_prizes=2), [GUST_ID])
    select = make_opt_select([
        Option(type=OptionType.ATTACH, index=0),  # 0
        Option(type=OptionType.PLAY, index=0),    # 1 (would be gust, but not configured)
        Option(type=OptionType.ATTACK),           # 2
    ])
    order = [
        sel[0]
        for sel in lethal_simple._candidate_selections(
            select, {"max_combinations_per_select": 128}, state
        )
    ]
    # PLAY keeps its default (lowest) priority: attack, then attach, then play.
    assert order == [2, 0, 1]


def test_gust_lethal_is_found_end_to_end(install_engine):
    root_obs, engine = _root_gust_scenario()
    install_engine(engine)
    assert run_search(root_obs, priority_play_card_ids=[GUST_ID]) == [3]


def test_gust_is_explored_before_decoys_when_configured(install_engine):
    root_obs, engine = _root_gust_scenario()
    install_engine(engine)
    assert run_search(root_obs, priority_play_card_ids=[GUST_ID]) == [3]
    log = engine.step_log
    assert _first_step(log, ("root", (3,))) < _first_step(log, ("root", (0,)))
    assert _first_step(log, ("root", (3,))) < _first_step(log, ("root", (1,)))
    assert _first_step(log, ("root", (3,))) < _first_step(log, ("root", (2,)))


def test_gust_is_explored_after_decoys_without_configuration(install_engine):
    # Same winning line, but with no gust card configured the gust PLAY keeps
    # its low priority, so the decoys are stepped first -- exactly the
    # ordering that let the 100ms budget run out before the gust line in the
    # replayed game (episode 88154084, turn 16).
    root_obs, engine = _root_gust_scenario()
    install_engine(engine)
    assert run_search(root_obs, priority_play_card_ids=[]) == [3]  # still found, just later
    log = engine.step_log
    assert _first_step(log, ("root", (3,))) > _first_step(log, ("root", (0,)))
    assert _first_step(log, ("root", (3,))) > _first_step(log, ("root", (1,)))


# ---------------------------------------------------------------------------
# Verification and reuse
# ---------------------------------------------------------------------------

def test_verification_rejects_shuffle_dependent_line(install_engine):
    # First shuffle finds a "win"; on the verification replay the same
    # selection is no longer legal -> must be rejected as non-deterministic.
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    root2_obs = make_obs(
        make_state(action_count=7),
        make_select([OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "win"}),
            "root2": FakeNode(root2_obs, {}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        root=["root", "root2"],
        default="opp",
    ))
    assert run_search(root_obs, verify_shuffles=1) is None


def test_search_can_run_again_after_finishing(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    nodes = {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN}
    engine = install_engine(FakeEngine(nodes, default="opp"))
    assert run_search(root_obs) == [0]
    assert run_search(root_obs) == [0]
    assert engine.end_count == 2


def test_returned_action_is_legal_for_current_select(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    action = run_search(root_obs)
    assert selector.is_valid_action(action, root_obs.select)


# ---------------------------------------------------------------------------
# 優先展開 (config-gated: priority_play_card_ids / priority_play_max_remaining_prizes)
#
# G1調査: 「ボスの指令→入替→攻撃→勝ち」の列は現行順序だと PLAY が最後尾(rank 5)で
# 予算内に到達できない。対象カードの手札PLAYだけを ATTACK の直後へ引き上げる。
# キーが無ければ順序は一切変わらない(既存挙動バイト不変)ことも同値テストで固定する。
# ---------------------------------------------------------------------------

BOSS_ORDERS_ID = 1182   # ボスの指令(data/JP_Card_Data.csv)
BRIAR_ID = 1201         # ブライア(同上)
OTHER_CARD_ID = 999     # 優先対象でないカード


def make_hand_state(*, my_prizes: int = 1, hand_ids: tuple[int, ...] = ()) -> State:
    """自分の手札に指定IDのカードを持たせた State。"""
    state = make_state(my_prizes=my_prizes)
    state.players[state.yourIndex].hand = [
        Card(id=card_id, serial=100 + i, playerIndex=state.yourIndex)
        for i, card_id in enumerate(hand_ids)
    ]
    return state


def make_option_select(options: list[Option], *, min_count: int = 1, max_count: int = 1) -> SelectData:
    return SelectData(
        type=SelectType.MAIN,
        context=SelectContext.MAIN,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )


def legacy_candidate_selections(select: SelectData, config: dict) -> list[list[int]]:
    """変更前の `_candidate_selections` をそのまま写した参照実装(同値テスト用)。"""
    import itertools

    order = list(range(len(select.option)))
    if select.type == SelectType.MAIN:
        order = [i for i in order if select.option[i].type != OptionType.END]
        order.sort(
            key=lambda i: lethal_simple._MAIN_OPTION_PRIORITY.get(
                select.option[i].type, lethal_simple._DEFAULT_PRIORITY
            )
        )
    min_count = max(select.minCount, 0)
    max_count = min(select.maxCount, len(order))
    limit = int(config["max_combinations_per_select"])
    out: list[list[int]] = []
    for count in range(min_count, max_count + 1):
        for combo in itertools.combinations(order, count):
            out.append(list(combo))
            if len(out) >= limit:
                return out
    return out


# 「ATTACH / その他PLAY / ABILITY / ATTACK / 対象PLAY / RETREAT / EVOLVE / END」の混在盤面。
# 手札は [0]=その他カード, [1]=ボスの指令, [2]=ブライア。
def _mixed_select() -> SelectData:
    return make_option_select([
        Option(type=OptionType.ATTACH, index=0),        # 0
        Option(type=OptionType.PLAY, index=0),          # 1: 手札0=対象外カード
        Option(type=OptionType.ABILITY),                # 2
        Option(type=OptionType.ATTACK),                 # 3
        Option(type=OptionType.PLAY, index=1),          # 4: 手札1=ボスの指令
        Option(type=OptionType.RETREAT),                # 5
        Option(type=OptionType.EVOLVE),                 # 6
        Option(type=OptionType.END),                    # 7
    ])


def test_priority_play_absent_keys_keep_existing_order():
    """新キーが無い config では生成順が変更前と完全一致する(既存挙動バイト不変)。"""
    select = _mixed_select()
    state = make_hand_state(hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID, BRIAR_ID))
    config = {k: v for k, v in lethal_simple.DEFAULTS.items() if not k.startswith("priority_play")}

    got = list(lethal_simple._candidate_selections(select, config, state))
    assert got == legacy_candidate_selections(select, config)
    # 参照実装と一致していることに加えて、期待順序そのものも明示しておく。
    assert got == [[3], [2], [6], [0], [5], [1], [4]]


def test_priority_play_empty_list_keeps_existing_order():
    """既定(空リスト)でも順序は変わらない。state を渡す/渡さないでも同一。"""
    select = _mixed_select()
    state = make_hand_state(hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID, BRIAR_ID))
    config = dict(lethal_simple.DEFAULTS)

    legacy = legacy_candidate_selections(select, config)
    assert list(lethal_simple._candidate_selections(select, config, state)) == legacy
    assert list(lethal_simple._candidate_selections(select, config, None)) == legacy
    assert list(lethal_simple._candidate_selections(select, config)) == legacy


def test_priority_play_promotes_target_play_right_after_attack():
    """対象PLAYが ATTACK の直後に来る: ATTACK→対象PLAY→ABILITY→EVOLVE→ATTACH→RETREAT→その他PLAY。"""
    select = _mixed_select()
    state = make_hand_state(my_prizes=2, hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID, BRIAR_ID))
    config = {
        **lethal_simple.DEFAULTS,
        "priority_play_card_ids": [BOSS_ORDERS_ID, BRIAR_ID],
        "priority_play_max_remaining_prizes": 2,
    }

    got = list(lethal_simple._candidate_selections(select, config, state))
    assert got == [[3], [4], [2], [6], [0], [5], [1]]
    # 枝刈りではなく並べ替えのみ: 集合は変わらない(END を除いた全選択肢が出る)。
    assert sorted(got) == sorted(
        legacy_candidate_selections(select, config)
    )


def test_priority_play_prize_threshold_boundary():
    """自分の残りサイドが閾値以下でのみ発動する(境界: 2=発動 / 3=発動しない)。"""
    select = _mixed_select()
    config = {
        **lethal_simple.DEFAULTS,
        "priority_play_card_ids": [BOSS_ORDERS_ID],
        "priority_play_max_remaining_prizes": 2,
    }

    on_state = make_hand_state(my_prizes=2, hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID))
    assert list(lethal_simple._candidate_selections(select, config, on_state))[1] == [4]

    off_state = make_hand_state(my_prizes=3, hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID))
    assert list(lethal_simple._candidate_selections(select, config, off_state)) == (
        legacy_candidate_selections(select, config)
    )


def test_priority_play_resolves_card_from_hand_when_cardid_is_none():
    """MAIN の Option.cardId は実測で常に None。手札実体 hand[index].id で同定できること。"""
    state = make_hand_state(hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID))
    play_boss = Option(type=OptionType.PLAY, index=1)
    assert play_boss.cardId is None
    assert lethal_simple._play_option_card_id(play_boss, state) == BOSS_ORDERS_ID
    # cardId が入っている場合はそちらを優先(将来 API が埋めてきても壊れない)。
    assert lethal_simple._play_option_card_id(
        Option(type=OptionType.PLAY, index=1, cardId=OTHER_CARD_ID), state
    ) == OTHER_CARD_ID
    # 判定不能(index 範囲外 / index なし / state なし)は None。
    assert lethal_simple._play_option_card_id(Option(type=OptionType.PLAY, index=9), state) is None
    assert lethal_simple._play_option_card_id(Option(type=OptionType.PLAY), state) is None
    assert lethal_simple._play_option_card_id(play_boss, None) is None


def test_priority_play_only_applies_to_play_options():
    """同じ手札インデックスでも PLAY 以外(ATTACH 等)は引き上げない。"""
    select = make_option_select([
        Option(type=OptionType.ATTACH, index=1),   # 0: 手札1(ボスの指令)を指すが ATTACH
        Option(type=OptionType.ATTACK),            # 1
        Option(type=OptionType.PLAY, index=0),     # 2: 対象外カード
    ])
    state = make_hand_state(my_prizes=1, hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID))
    config = {
        **lethal_simple.DEFAULTS,
        "priority_play_card_ids": [BOSS_ORDERS_ID],
        "priority_play_max_remaining_prizes": 2,
    }
    assert list(lethal_simple._candidate_selections(select, config, state)) == (
        legacy_candidate_selections(select, config)
    )


def test_priority_play_changes_search_expansion_order(install_engine):
    """探索本体でも優先展開が効く: ATTACK の次に踏まれるのが対象PLAYになる。

    盤面: ATTACK は turn が終わるだけ、ATTACH も無意味、ボスの指令 PLAY のあとの攻撃で勝ち。
    現行順序では ATTACH(rank3) が PLAY(rank5) より先に踏まれる。
    """
    state = make_hand_state(my_prizes=2, hand_ids=(BOSS_ORDERS_ID,))
    root_select = make_option_select([
        Option(type=OptionType.ATTACK),          # 0
        Option(type=OptionType.ATTACH, index=0), # 1
        Option(type=OptionType.PLAY, index=0),   # 2: ボスの指令
    ])
    root_obs = make_obs(state, root_select)
    mid_obs = make_obs(
        make_hand_state(my_prizes=2, hand_ids=()),
        make_select([OptionType.ATTACK, OptionType.END]),
    )
    nodes = {
        "root": FakeNode(root_obs, {(0,): "opp", (1,): "opp", (2,): "mid"}),
        "mid": FakeNode(mid_obs, {(0,): "win"}),
        "win": WIN,
        "opp": OPP_TURN,
    }

    engine = install_engine(FakeEngine(dict(nodes), default="opp"))
    assert run_search(
        root_obs,
        max_remaining_prizes=2,
        priority_play_card_ids=[BOSS_ORDERS_ID],
        priority_play_max_remaining_prizes=2,
    ) == [2]
    assert engine.step_log[0] == ("root", (0,))
    assert engine.step_log[1] == ("root", (2,))  # 対象PLAYが ATTACK の直後

    # 同じ盤面を新キー無しで走らせると、現行どおり ATTACH が先に踏まれる。
    engine2 = install_engine(FakeEngine(dict(nodes), default="opp"))
    assert run_search(root_obs, max_remaining_prizes=2) == [2]
    assert engine2.step_log[0] == ("root", (0,))
    assert engine2.step_log[1] == ("root", (1,))


# ---------------------------------------------------------------------------
# Fix-A(1) 相手対象ソート (config-gated: priority_sort_switch_targets)
#
# 実ラダー 93335550 T15 の機械診断: ボスの指令の直後に来る CARD/SWITCH(どの相手ベンチを
# 引きずり出すか)は自然順のままだと高HPのメガ ex から踏むため、確1圏の弱い対象(HP100)に
# 届く前に予算が尽きる。相手ポケモンを指す選択肢だけを残りHP昇順へ並べ替える。
# ---------------------------------------------------------------------------

def _mon(card_id: int, hp: int, max_hp: int | None = None) -> Pokemon:
    return Pokemon(
        id=card_id, serial=card_id * 10, hp=hp, maxHp=max_hp if max_hp is not None else hp,
        appearThisTurn=False, energies=[], energyCards=[], tools=[], preEvolution=[],
    )


def make_switch_state(bench_hps: list[int], *, my_prizes: int = 1, your_index: int = 0) -> State:
    """相手ベンチに指定HPのポケモンを並べた State(ボスの指令直後の盤面を模す)。"""
    state = make_state(my_prizes=my_prizes, your_index=your_index)
    opp = state.players[1 - your_index]
    opp.active = [_mon(900, 330)]
    opp.bench = [_mon(800 + i, hp) for i, hp in enumerate(bench_hps)]
    return state


def make_switch_select(options: list[Option], *, min_count: int = 1, max_count: int = 1) -> SelectData:
    """ボスの指令直後の CARD/SWITCH 実測スキーマ(cardId は None、area/index/playerIndex で同定)。"""
    return SelectData(
        type=SelectType.CARD,
        context=SelectContext.SWITCH,
        minCount=min_count,
        maxCount=max_count,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )


def _bench_options(count: int, opponent_index: int = 1) -> list[Option]:
    return [
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=i, playerIndex=opponent_index)
        for i in range(count)
    ]


_SORT_ON = {
    **lethal_simple.DEFAULTS,
    "priority_sort_switch_targets": True,
    "priority_play_max_remaining_prizes": 2,
}


def test_switch_target_sort_absent_key_keeps_natural_order():
    """新キーが無い config では非MAIN select の順序は自然順のまま(既存挙動不変)。"""
    select = make_switch_select(_bench_options(4))
    state = make_switch_state([310, 100, 140, 100])
    config = {k: v for k, v in lethal_simple.DEFAULTS.items()
              if k != "priority_sort_switch_targets"}
    assert list(lethal_simple._candidate_selections(select, config, state)) == [[0], [1], [2], [3]]
    # 既定(False)でも同じ。
    assert list(
        lethal_simple._candidate_selections(select, dict(lethal_simple.DEFAULTS), state)
    ) == [[0], [1], [2], [3]]


def test_switch_target_sort_orders_by_remaining_hp_ascending():
    """有効時は残りHP昇順。同HPは元の順(安定ソート)。"""
    select = make_switch_select(_bench_options(4))
    state = make_switch_state([310, 100, 140, 100])
    got = list(lethal_simple._candidate_selections(select, _SORT_ON, state))
    assert got == [[1], [3], [2], [0]]
    # 枝刈りではなく並べ替えのみ: 集合は変わらない。
    assert sorted(got) == [[0], [1], [2], [3]]


def test_switch_target_sort_puts_unresolvable_options_last():
    """相手ポケモンに紐づかない/判定不能な選択肢は末尾(安全側)。自分側の選択肢も対象外。"""
    options = [
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=9, playerIndex=1),   # 0: 範囲外
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),   # 1: 自分側
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=1),   # 2: HP100
        Option(type=OptionType.CARD, area=AreaType.ACTIVE, playerIndex=1),           # 3: HP330
    ]
    state = make_switch_state([310, 100])
    state.players[0].bench = [_mon(700, 50)]
    got = list(lethal_simple._candidate_selections(make_switch_select(options), _SORT_ON, state))
    assert got == [[2], [3], [0], [1]]


def test_switch_target_sort_prize_threshold_boundary():
    """自サイド残が priority_play_max_remaining_prizes 以下のときだけ効く(境界 2=ON / 3=OFF)。"""
    select = make_switch_select(_bench_options(3))
    on_state = make_switch_state([310, 100, 140], my_prizes=2)
    assert list(lethal_simple._candidate_selections(select, _SORT_ON, on_state)) == [[1], [2], [0]]
    off_state = make_switch_state([310, 100, 140], my_prizes=3)
    assert list(lethal_simple._candidate_selections(select, _SORT_ON, off_state)) == [[0], [1], [2]]
    # state が無い(判定不能)ときも自然順。
    assert list(lethal_simple._candidate_selections(select, _SORT_ON, None)) == [[0], [1], [2]]


def test_switch_target_sort_does_not_touch_main_selects():
    """MAIN は従来どおり `_MAIN_OPTION_PRIORITY` / 優先展開の順序のまま(相手対象ソートは非MAIN限定)。"""
    select = _mixed_select()
    state = make_hand_state(my_prizes=1, hand_ids=(OTHER_CARD_ID, BOSS_ORDERS_ID, BRIAR_ID))
    config = {**lethal_simple.DEFAULTS, "priority_sort_switch_targets": True}
    assert list(lethal_simple._candidate_selections(select, config, state)) == (
        legacy_candidate_selections(select, config)
    )


def test_switch_target_sort_multi_select_keeps_same_combination_set():
    """複数選択(maxCount>1)でも並べ替えのみで組み合わせの集合は不変。"""
    select = make_switch_select(_bench_options(3), min_count=2, max_count=2)
    state = make_switch_state([310, 100, 140])
    got = list(lethal_simple._candidate_selections(select, _SORT_ON, state))
    assert got == [[1, 2], [1, 0], [2, 0]]
    assert sorted(sorted(c) for c in got) == [[0, 1], [0, 2], [1, 2]]


def test_switch_target_sort_changes_search_expansion_order(install_engine):
    """探索本体でも効く: 弱い(HP100)対象の枝を先に踏む。

    盤面: ボスの指令直後の SWITCH で、HP100 の相手を選んだ枝だけが勝ちに繋がる。
    自然順では HP310 の枝を先に踏む。
    """
    state = make_switch_state([310, 100], my_prizes=1)
    root_select = make_switch_select(_bench_options(2))
    root_obs = make_obs(state, root_select)
    after_obs = make_obs(make_state(action_count=1), make_select([OptionType.ATTACK, OptionType.END]))
    nodes = {
        "root": FakeNode(root_obs, {(0,): "opp", (1,): "after"}),
        "after": FakeNode(after_obs, {(0,): "win"}),
        "win": WIN,
        "opp": OPP_TURN,
    }

    engine = install_engine(FakeEngine(dict(nodes), default="opp"))
    assert run_search(root_obs, priority_sort_switch_targets=True) == [1]
    assert engine.step_log[0] == ("root", (1,))  # HP100 を最初に踏む

    engine2 = install_engine(FakeEngine(dict(nodes), default="opp"))
    assert run_search(root_obs) == [1]
    assert engine2.step_log[0] == ("root", (0,))  # 現行は自然順=HP310 が先


def test_option_opponent_pokemon_resolution():
    """`_option_opponent_pokemon` の解決規則(相手側 / area / index)。"""
    state = make_switch_state([310, 100])
    bench1 = Option(type=OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=1)
    assert lethal_simple._option_opponent_pokemon(bench1, state).hp == 100
    active = Option(type=OptionType.CARD, area=AreaType.ACTIVE, playerIndex=1)
    assert lethal_simple._option_opponent_pokemon(active, state).hp == 330
    # 自分側・範囲外・エリア不明・state無しはすべて None。
    assert lethal_simple._option_opponent_pokemon(
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0), state) is None
    assert lethal_simple._option_opponent_pokemon(
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=9, playerIndex=1), state) is None
    assert lethal_simple._option_opponent_pokemon(
        Option(type=OptionType.CARD, area=AreaType.HAND, index=0, playerIndex=1), state) is None
    assert lethal_simple._option_opponent_pokemon(bench1, None) is None


# ---------------------------------------------------------------------------
# Fix-A(2) 終盤予算エスカレーション
# (config-gated: endgame_time_limit_ms / endgame_max_nodes / endgame_max_remaining_prizes)
# ---------------------------------------------------------------------------

def test_endgame_budget_absent_keys_return_same_config_object():
    """新キーが無ければ config オブジェクトはそのまま返る(=差し替えゼロ、本番不変)。"""
    config = dict(lethal_simple.DEFAULTS)
    state = make_state(my_prizes=1)
    assert lethal_simple._apply_endgame_budget(config, state, 0) is config


def test_endgame_budget_replaces_limits_below_threshold():
    """自サイド残 <= 閾値 のときだけ時間/ノード予算を差し替える。元 config は変更しない。"""
    config = {**lethal_simple.DEFAULTS, "time_limit_ms": 400, "max_nodes": 10000,
              "endgame_time_limit_ms": 2500, "endgame_max_nodes": 30000,
              "endgame_max_remaining_prizes": 2}
    escalated = lethal_simple._apply_endgame_budget(config, make_state(my_prizes=2), 0)
    assert escalated["time_limit_ms"] == 2500
    assert escalated["max_nodes"] == 30000
    assert config["time_limit_ms"] == 400 and config["max_nodes"] == 10000  # 呼び出し側は不変


def test_endgame_budget_threshold_boundary():
    """境界: サイド残2=発動 / 3=非発動。"""
    config = {**lethal_simple.DEFAULTS, "time_limit_ms": 400,
              "endgame_time_limit_ms": 2500, "endgame_max_remaining_prizes": 2}
    assert lethal_simple._apply_endgame_budget(config, make_state(my_prizes=2), 0)["time_limit_ms"] == 2500
    assert lethal_simple._apply_endgame_budget(config, make_state(my_prizes=3), 0) is config


def test_endgame_budget_only_one_key_given():
    """片方のキーだけでも効く(もう片方は現行値のまま)。"""
    config = {**lethal_simple.DEFAULTS, "time_limit_ms": 400, "max_nodes": 10000,
              "endgame_max_nodes": 30000}
    escalated = lethal_simple._apply_endgame_budget(config, make_state(my_prizes=1), 0)
    assert escalated["max_nodes"] == 30000
    assert escalated["time_limit_ms"] == 400


def test_endgame_budget_lets_search_find_lethal_that_node_limit_missed(install_engine):
    """ノード上限で見逃していた勝ち筋が、終盤エスカレーションで見つかる。"""
    root_obs = make_obs(
        make_state(my_prizes=1),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(make_state(my_prizes=1, action_count=1),
                       make_select([OptionType.ATTACK, OptionType.END]))
    nodes = {
        "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
        "mid": FakeNode(mid_obs, {(0,): "win"}),
        "win": WIN,
        "opp": OPP_TURN,
    }
    install_engine(FakeEngine(dict(nodes), default="opp"))
    assert run_search(root_obs, max_nodes=1) is None  # 現行予算では届かない

    install_engine(FakeEngine(dict(nodes), default="opp"))
    lethal_simple.reset_stats()
    assert run_search(root_obs, max_nodes=1, endgame_max_nodes=1000) == [0]
    assert lethal_simple.get_stats()["endgame_escalations"] == 1


def test_endgame_budget_not_applied_above_threshold_in_search(install_engine):
    """サイド残が閾値超なら search 内でも差し替わらない(現行予算のまま見逃す)。"""
    root_obs = make_obs(
        make_state(my_prizes=3),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(make_state(my_prizes=3, action_count=1),
                       make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
            "mid": FakeNode(mid_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    lethal_simple.reset_stats()
    assert run_search(
        root_obs, max_remaining_prizes=3, max_nodes=1,
        endgame_max_nodes=1000, endgame_max_remaining_prizes=2,
    ) is None
    assert lethal_simple.get_stats()["endgame_escalations"] == 0


# ---------------------------------------------------------------------------
# Hidden-state stub (dummy data for search_begin)
# ---------------------------------------------------------------------------

def _card(card_id: int, player_index: int = 0) -> Card:
    return Card(id=card_id, serial=card_id, playerIndex=player_index)


@pytest.fixture(autouse=True)
def filler_cache(monkeypatch):
    monkeypatch.setattr(search_state_stub, "_FILLER_CACHE", {"energy": 999, "pokemon": 888})


def test_build_dummy_search_state_accounts_for_visible_cards():
    state = make_state(my_prizes=2)
    me = state.players[0]
    me.hand = [_card(1)]
    me.discard = [_card(2)]
    me.deckCount = 1
    full_deck = [1, 2, 3, 4, 5]
    obs = make_obs(state, make_select([OptionType.ATTACK]))
    hidden_state = search_state_stub.build_dummy_search_state(obs, full_deck)
    assert hidden_state is not None
    assert len(hidden_state["your_prize"]) == 2
    assert len(hidden_state["your_deck"]) == 1
    assert sorted(hidden_state["your_deck"] + hidden_state["your_prize"]) == [3, 4, 5]
    assert len(hidden_state["opponent_prize"]) == 3
    assert len(hidden_state["opponent_deck"]) == state.players[1].deckCount


def test_build_dummy_search_state_returns_none_on_count_mismatch():
    state = make_state(my_prizes=2)
    state.players[0].hand = [_card(1)]
    state.players[0].deckCount = 10  # unseen pool cannot cover this
    obs = make_obs(state, make_select([OptionType.ATTACK]))
    assert search_state_stub.build_dummy_search_state(obs, [1, 2, 3, 4, 5]) is None


def test_build_dummy_search_state_returns_none_on_unknown_visible_card():
    state = make_state(my_prizes=2)
    state.players[0].hand = [_card(42)]  # not in the deck list
    state.players[0].deckCount = 2
    obs = make_obs(state, make_select([OptionType.ATTACK]))
    assert search_state_stub.build_dummy_search_state(obs, [1, 2, 3, 4, 5]) is None


def _hand_state_for_rng_test() -> tuple[State, list[int]]:
    """`build_dummy_search_state` が実際にプールをシャッフルする(=乱数を1回以上消費する)
    ような、未確認プールが複数枚残る局面を組み立てる。"""
    state = make_state(my_prizes=2)
    me = state.players[0]
    me.hand = [_card(1)]
    me.discard = [_card(2)]
    me.deckCount = 1
    return state, [1, 2, 3, 4, 5]


def test_build_dummy_search_state_without_rng_consumes_global_random():
    """rng 未指定(既定)は現行どおりグローバル `random` モジュールを消費する(対照)。"""
    import random

    state, full_deck = _hand_state_for_rng_test()
    obs = make_obs(state, make_select([OptionType.ATTACK]))
    random.seed(123)
    before = random.getstate()
    search_state_stub.build_dummy_search_state(obs, full_deck)
    after = random.getstate()
    assert before != after


def test_build_dummy_search_state_with_rng_does_not_touch_global_random():
    """専用の `random.Random` を渡せば、グローバル `random` の状態は一切変化しない。"""
    import random

    state, full_deck = _hand_state_for_rng_test()
    obs = make_obs(state, make_select([OptionType.ATTACK]))
    before = random.getstate()
    search_state_stub.build_dummy_search_state(obs, full_deck, rng=random.Random(7))
    after = random.getstate()
    assert before == after


# ---------------------------------------------------------------------------
# 専用RNG (config-gated `lethal_search.isolated_rng`、乱数消費の交絡除去)
#
# 問題: リーサル探索の隠れ状態スタブは既定でグローバル `random` を消費するため、探索の
# 実行量(発火有無・verify_shuffles 回数)が変わるだけで探索と無関係な後続の乱数列まで
# ずれ、paired seed A/B のペア性が壊れる。`lethal_simple` が専用の `random.Random` を
# 持ち、試合開始時に決定論的にシードし直せるようにする(実際の切替は
# `ml_policy_agent._try_lethal` 側、test_ml_policy_agent.py で検証)。
# ---------------------------------------------------------------------------

def test_reset_rng_then_get_rng_is_reproducible_for_same_seed():
    """同一seedで reset すれば、以後に引く乱数列は再現する。"""
    lethal_simple.reset_rng(42)
    seq1 = [lethal_simple.get_rng().random() for _ in range(5)]
    lethal_simple.reset_rng(42)
    seq2 = [lethal_simple.get_rng().random() for _ in range(5)]
    assert seq1 == seq2


def test_reset_rng_different_seeds_diverge():
    lethal_simple.reset_rng(1)
    seq1 = [lethal_simple.get_rng().random() for _ in range(5)]
    lethal_simple.reset_rng(2)
    seq2 = [lethal_simple.get_rng().random() for _ in range(5)]
    assert seq1 != seq2


def test_get_rng_returns_same_instance_until_reset():
    lethal_simple.reset_rng(1)
    rng_a = lethal_simple.get_rng()
    rng_b = lethal_simple.get_rng()
    assert rng_a is rng_b
    lethal_simple.reset_rng(2)
    rng_c = lethal_simple.get_rng()
    assert rng_c is not rng_a


def test_get_rng_lazy_initializes_without_touching_global_random(monkeypatch):
    """`reset_rng()` を一度も呼ばずに `get_rng()` を呼んでも、グローバル `random` の
    状態には一切触れない(自己初期化はOSエントロピー等、モジュール完結)。"""
    import random

    monkeypatch.setattr(lethal_simple, "_rng", None)
    before = random.getstate()
    rng = lethal_simple.get_rng()
    after = random.getstate()
    assert before == after
    assert isinstance(rng, random.Random)


# ---------------------------------------------------------------------------
# Selector fallback
# ---------------------------------------------------------------------------

def test_selector_falls_back_when_search_returns_none(monkeypatch):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    monkeypatch.setattr(lethal_simple, "search", lambda *a, **k: None)
    action = selector.select_action(obs, [1] * 60, {"lethal_search": {"enabled": True}})
    assert selector.is_valid_action(action, obs.select)


def test_selector_falls_back_when_search_returns_illegal(monkeypatch):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    monkeypatch.setattr(lethal_simple, "search", lambda *a, **k: [5])
    action = selector.select_action(obs, [1] * 60, {"lethal_search": {"enabled": True}})
    assert selector.is_valid_action(action, obs.select)


def test_selector_uses_search_result(monkeypatch):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    monkeypatch.setattr(lethal_simple, "search", lambda *a, **k: [0])
    action = selector.select_action(obs, [1] * 60, {"lethal_search": {"enabled": True}})
    assert action == [0]


def test_selector_ignores_search_when_disabled(monkeypatch):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    def _boom(*a, **k):
        raise AssertionError("search must not be called when disabled")
    monkeypatch.setattr(lethal_simple, "search", _boom)
    action = selector.select_action(obs, [1] * 60, {"lethal_search": {"enabled": False}})
    assert selector.is_valid_action(action, obs.select)
