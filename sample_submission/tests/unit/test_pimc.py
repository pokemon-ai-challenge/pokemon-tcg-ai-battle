"""Unit tests for ptcg_ai.search.pimc (Stage2, stage2-pimc-implementation-plan.md).

Mirrors the fake-engine approach of ``tests/unit/test_lethal_simple.py`` (the
competition search API is replaced by a small hand-built decision tree), and
additionally fakes the value network (``pimc._get_model``) so leaf scores can
be controlled precisely.

Run from sample_submission/:
    python -m pytest tests/unit/test_pimc.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

import pytest

from cg.api import (
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
from ptcg_ai.search import pimc


# ---------------------------------------------------------------------------
# Builders (duplicated from test_lethal_simple.py on purpose: pimc.py is
# kept independent of lethal_simple.py, and tests follow the same shape).
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
# Fake search engine (same shape as test_lethal_simple.py's FakeEngine)
# ---------------------------------------------------------------------------

class FakeNode:
    def __init__(self, obs: Observation, transitions: dict | None = None):
        self.obs = obs
        self.transitions = transitions or {}


class FakeEngine:
    """Mimics cg.api search_* using a hand-built decision tree."""

    def __init__(self, nodes: dict[str, FakeNode], root="root", default: str | None = None):
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
        monkeypatch.setattr(pimc, "cg_api", engine)
        return engine
    return _install


class FakeValueModel:
    def __init__(self, score_fn):
        self._score_fn = score_fn

    def predict_win_prob_from_state(self, state) -> float:
        return self._score_fn(state)


@pytest.fixture
def install_value_model(monkeypatch):
    """Install a fake value model keyed off State.turnActionCount by default."""
    def _install(score_by_action_count: dict[int, float], default: float = 0.5):
        model = FakeValueModel(
            lambda state: score_by_action_count.get(state.turnActionCount, default)
        )
        monkeypatch.setattr(pimc, "_get_model", lambda: model)
        return model
    return _install


def base_context(obs: Observation, **config) -> dict:
    cfg = {"enabled": True, "time_limit_ms": 1000, "num_determinizations": 1}
    cfg.update(config)
    return {"observation": obs, "config": cfg, "hidden_state": DUMMY_HIDDEN_STATE}


def run_search(obs: Observation, **config):
    return pimc.search(obs.current, obs.select.option, base_context(obs, **config))


WIN = FakeNode(make_obs(make_state(result=0, action_count=90), None))
OPP_TURN = FakeNode(
    make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=91),
        make_select([OptionType.END]),
    )
)


# ---------------------------------------------------------------------------
# Step1-equivalent: certain win detection (same property as lethal_simple)
# ---------------------------------------------------------------------------

def test_attack_now_wins(install_engine, install_value_model):
    install_value_model({})  # leaf scores never matter here
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [0]
    assert engine.end_count == 1


def test_end_option_is_never_stepped(install_engine, install_value_model):
    install_value_model({91: 0.1})
    root_obs = make_obs(make_state(), make_select([OptionType.END, OptionType.ATTACK]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(1,): "opp"}), "opp": OPP_TURN},
        default="opp",
    ))
    run_search(root_obs)
    assert ("root", (0,)) not in engine.step_log


def test_multi_step_line_to_certain_win(install_engine, install_value_model):
    # Attach energy first, then attack: covers a multi-step certain-win line.
    install_value_model({})
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(make_state(action_count=1), make_select([OptionType.ATTACK, OptionType.END]))
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


def test_opponent_win_branch_is_not_chosen(install_engine, install_value_model):
    # One branch loses immediately (result != me); the other ends the turn
    # with a mediocre value score. The losing branch must never be chosen.
    lose = FakeNode(make_obs(make_state(result=1, action_count=99), None))
    install_value_model({91: 0.4})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.RETREAT]))
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "lose", (1,): "opp"}),
            "lose": lose,
            "opp": OPP_TURN,
        },
    ))
    assert run_search(root_obs) == [1]


# ---------------------------------------------------------------------------
# Step2-equivalent: value-network leaf scoring picks the better line
# ---------------------------------------------------------------------------

def test_higher_value_leaf_is_preferred(install_engine, install_value_model):
    # Two options, neither wins outright; the value network clearly prefers
    # the RETREAT line (0.9) over the ATTACH line (0.2).
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.RETREAT]),
    )
    attach_leaf = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=1),
        make_select([OptionType.END]),
    )
    retreat_leaf = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=2),
        make_select([OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "attach_leaf", (1,): "retreat_leaf"}),
            "attach_leaf": FakeNode(attach_leaf, {}),
            "retreat_leaf": FakeNode(retreat_leaf, {}),
        },
    ))
    install_value_model({1: 0.2, 2: 0.9})
    assert run_search(root_obs) == [1]


def test_lower_value_leaf_not_chosen_over_higher(install_engine, install_value_model):
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.RETREAT]),
    )
    attach_leaf = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=1),
        make_select([OptionType.END]),
    )
    retreat_leaf = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=2),
        make_select([OptionType.END]),
    )
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "attach_leaf", (1,): "retreat_leaf"}),
            "attach_leaf": FakeNode(attach_leaf, {}),
            "retreat_leaf": FakeNode(retreat_leaf, {}),
        },
    ))
    # Reversed from the previous test: now ATTACH scores higher.
    install_value_model({1: 0.95, 2: 0.1})
    assert run_search(root_obs) == [0]


# ---------------------------------------------------------------------------
# Step3-equivalent: determinization loop + aggregation
# ---------------------------------------------------------------------------

def test_aggregation_averages_across_determinizations(install_engine, install_value_model):
    # Same two first-move options (ATTACH=0, RETREAT=1) under two different
    # determinizations ("root1", "root2"), each scoring the leaves
    # differently. ATTACH averages (0.9+0.7)/2=0.8, RETREAT averages
    # (0.4+0.5)/2=0.45: ATTACH must win despite neither sample being
    # decisive on its own.
    root1 = make_obs(make_state(), make_select([OptionType.ATTACH, OptionType.RETREAT]))
    root2 = make_obs(make_state(), make_select([OptionType.ATTACH, OptionType.RETREAT]))
    a1 = make_obs(make_state(your_index=1, first_player=0, turn=2, action_count=1), make_select([OptionType.END]))
    r1 = make_obs(make_state(your_index=1, first_player=0, turn=2, action_count=2), make_select([OptionType.END]))
    a2 = make_obs(make_state(your_index=1, first_player=0, turn=2, action_count=3), make_select([OptionType.END]))
    r2 = make_obs(make_state(your_index=1, first_player=0, turn=2, action_count=4), make_select([OptionType.END]))
    install_engine(FakeEngine(
        {
            "root1": FakeNode(root1, {(0,): "a1", (1,): "r1"}),
            "root2": FakeNode(root2, {(0,): "a2", (1,): "r2"}),
            "a1": FakeNode(a1, {}),
            "r1": FakeNode(r1, {}),
            "a2": FakeNode(a2, {}),
            "r2": FakeNode(r2, {}),
        },
        root=["root1", "root2"],
    ))
    install_value_model({1: 0.9, 2: 0.4, 3: 0.7, 4: 0.5})

    factory_values = iter([DUMMY_HIDDEN_STATE, DUMMY_HIDDEN_STATE])
    context = {
        "observation": root1,
        "config": {"enabled": True, "time_limit_ms": 1000, "num_determinizations": 2},
        "hidden_state_factory": lambda: next(factory_values),
    }
    assert pimc.search(root1.current, root1.select.option, context) == [0]


def test_aggregation_uses_partial_samples_when_a_win_short_circuits(
    install_engine, install_value_model
):
    # Determinization 1: RETREAT (ATTACK-priority option, tried first) wins
    # outright, so ATTACH is never explored in that sample. Determinization
    # 2: neither wins; ATTACH scores much higher than RETREAT.
    # RETREAT's average is (1.0 + 0.3) / 2 = 0.65; ATTACH's average is
    # 0.9 / 1 = 0.9 (only one sample). ATTACH must still win: the
    # aggregation must average over however many samples a key actually
    # has, not require every determinization to cover every key.
    root1 = make_obs(make_state(), make_select([OptionType.ABILITY, OptionType.ATTACK]))
    root2 = make_obs(make_state(), make_select([OptionType.ABILITY, OptionType.ATTACK]))
    # index0 = ABILITY ("ATTACH-like" option, priority 1), index1 = ATTACK (priority 0, tried first).
    ability_leaf2 = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=1), make_select([OptionType.END])
    )
    attack_leaf2 = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=2), make_select([OptionType.END])
    )
    install_engine(FakeEngine(
        {
            "root1": FakeNode(root1, {(1,): "win"}),
            "root2": FakeNode(root2, {(0,): "ability_leaf2", (1,): "attack_leaf2"}),
            "win": WIN,
            "ability_leaf2": FakeNode(ability_leaf2, {}),
            "attack_leaf2": FakeNode(attack_leaf2, {}),
        },
        root=["root1", "root2"],
    ))
    install_value_model({1: 0.9, 2: 0.3})

    factory_values = iter([DUMMY_HIDDEN_STATE, DUMMY_HIDDEN_STATE])
    context = {
        "observation": root1,
        "config": {"enabled": True, "time_limit_ms": 1000, "num_determinizations": 2},
        "hidden_state_factory": lambda: next(factory_values),
    }
    # index 0 == ABILITY == the higher-average-but-single-sample option.
    assert pimc.search(root1.current, root1.select.option, context) == [0]


def test_missing_hidden_state_sample_is_skipped(install_engine, install_value_model):
    # A factory that returns None for one determinization must not crash;
    # the other determinization's result should still be used.
    install_value_model({1: 0.8})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK]))
    leaf = make_obs(
        make_state(your_index=1, first_player=0, turn=2, action_count=1), make_select([OptionType.END])
    )
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "leaf"}), "leaf": FakeNode(leaf, {})},
    ))
    factory_values = iter([None, DUMMY_HIDDEN_STATE])
    context = {
        "observation": root_obs,
        "config": {"enabled": True, "time_limit_ms": 1000, "num_determinizations": 2},
        "hidden_state_factory": lambda: next(factory_values),
    }
    assert pimc.search(root_obs.current, root_obs.select.option, context) == [0]


# ---------------------------------------------------------------------------
# Gates / config
# ---------------------------------------------------------------------------

def test_disabled_config_skips_search(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN}))
    assert run_search(root_obs, enabled=False) is None
    assert engine.begin_count == 0


def test_no_prize_gate_still_searches_at_high_prize_count(install_engine, install_value_model):
    # Unlike lethal_simple, pimc has no max_remaining_prizes gate (contract
    # section 4 of the plan): it must still search even far from lethal.
    install_value_model({})
    root_obs = make_obs(make_state(my_prizes=6), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs) == [0]
    assert engine.begin_count == 1


def test_not_my_turn_skips_search(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(turn=2), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN}))
    assert run_search(root_obs) is None
    assert engine.begin_count == 0


def test_missing_hidden_state_factory_skips_search(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK]))
    engine = install_engine(FakeEngine({"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN}))
    context = {"observation": root_obs, "config": {"enabled": True}}
    assert pimc.search(root_obs.current, root_obs.select.option, context) is None
    assert engine.begin_count == 0


def test_node_limit_returns_none_and_records_stat(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(
        make_state(),
        make_select([OptionType.ATTACH, OptionType.ATTACK, OptionType.END]),
    )
    mid_obs = make_obs(make_state(action_count=1), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "mid", (1,): "opp"}),
            "mid": FakeNode(mid_obs, {(0,): "win"}),
            "win": WIN,
            "opp": OPP_TURN,
        },
        default="opp",
    ))
    pimc.reset_stats()
    assert run_search(root_obs, max_nodes=1) is None
    stats = pimc.get_stats()
    assert stats["determinization_node_limit_hits"] >= 1


def test_time_limit_returns_none(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    assert run_search(root_obs, time_limit_ms=0) is None
    assert engine.end_count == 1  # search_end still runs


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_record_searches_and_findings(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    pimc.reset_stats()
    assert run_search(root_obs) == [0]
    stats = pimc.get_stats()
    assert stats["searches"] == 1
    assert stats["found"] == 1
    assert stats["determinizations_run"] == 1
    assert stats["max_time_ms"] >= 0.0
    assert stats["avg_time_ms"] == stats["total_time_ms"]


# ---------------------------------------------------------------------------
# Selector fallback (registration in _SEARCH_MODULES)
# ---------------------------------------------------------------------------

def test_returned_action_is_legal_for_current_select(install_engine, install_value_model):
    install_value_model({})
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "win"}), "win": WIN, "opp": OPP_TURN},
        default="opp",
    ))
    action = run_search(root_obs)
    assert selector.is_valid_action(action, root_obs.select)


def test_pimc_registered_in_selector_search_modules():
    assert selector._SEARCH_MODULES["pimc"] is pimc


def test_selector_falls_back_when_pimc_returns_none(monkeypatch):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    monkeypatch.setattr(pimc, "search", lambda *a, **k: None)
    action = selector.select_action(
        obs, [1] * 60, {"lethal_search": {"enabled": True, "module": "pimc"}}
    )
    assert selector.is_valid_action(action, obs.select)
