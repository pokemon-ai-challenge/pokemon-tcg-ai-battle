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


def test_two_prizes_left_skips_search(install_engine):
    root_obs = make_obs(make_state(my_prizes=2), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs) is None
    assert engine.begin_count == 0


def test_not_my_turn_skips_search(install_engine):
    # turn=2 with firstPlayer == me (0) means it is the opponent's turn.
    root_obs = make_obs(make_state(turn=2), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN},
    ))
    assert run_search(root_obs) is None
    assert engine.begin_count == 0


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
# Hidden-information prediction
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
    prediction = search_state_stub.build_dummy_search_state(obs, full_deck)
    assert prediction is not None
    assert len(prediction["your_prize"]) == 2
    assert len(prediction["your_deck"]) == 1
    assert sorted(prediction["your_deck"] + prediction["your_prize"]) == [3, 4, 5]
    assert len(prediction["opponent_prize"]) == 3
    assert len(prediction["opponent_deck"]) == state.players[1].deckCount


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
