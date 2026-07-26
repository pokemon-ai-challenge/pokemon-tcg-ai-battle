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
    Card,
    Observation,
    Option,
    OptionType,
    PlayerState,
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
    hand = [Card(id=GUST_ID, serial=0, playerIndex=0), Card(id=999, serial=1, playerIndex=0)]
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
            select, {"max_combinations_per_select": 128}, hand, frozenset({GUST_ID})
        )
    ]
    assert 5 not in order      # END pruned
    assert order[0] == 4       # ATTACK still first
    assert order[1] == 2       # gust PLAY right after the attack
    assert order.index(2) < order.index(1)  # before the non-gust PLAY
    assert order.index(2) < order.index(0)  # before ATTACH
    assert order.index(2) < order.index(3)  # before EVOLVE


def test_candidate_selections_keeps_normal_order_without_gust_ids():
    hand = [Card(id=GUST_ID, serial=0, playerIndex=0)]
    select = make_opt_select([
        Option(type=OptionType.ATTACH, index=0),  # 0
        Option(type=OptionType.PLAY, index=0),    # 1 (would be gust, but not configured)
        Option(type=OptionType.ATTACK),           # 2
    ])
    order = [
        sel[0]
        for sel in lethal_simple._candidate_selections(
            select, {"max_combinations_per_select": 128}, hand, frozenset()
        )
    ]
    # PLAY keeps its default (lowest) priority: attack, then attach, then play.
    assert order == [2, 0, 1]


def test_gust_lethal_is_found_end_to_end(install_engine):
    root_obs, engine = _root_gust_scenario()
    install_engine(engine)
    assert run_search(root_obs, gust_card_ids=[GUST_ID]) == [3]


def test_gust_is_explored_before_decoys_when_configured(install_engine):
    root_obs, engine = _root_gust_scenario()
    install_engine(engine)
    assert run_search(root_obs, gust_card_ids=[GUST_ID]) == [3]
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
    assert run_search(root_obs, gust_card_ids=[]) == [3]  # still found, just later
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
