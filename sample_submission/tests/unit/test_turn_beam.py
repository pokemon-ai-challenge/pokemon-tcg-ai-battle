"""Unit tests for ptcg_ai.search.turn_beam (自ターン内ビーム探索).

Mirrors ``test_lethal_simple.py``'s approach: the competition search API is
replaced by a small fake engine so beam expansion, pruning, leaf
evaluation, and budget handling can be tested without a real battle.

Run from sample_submission/:
    python -m pytest tests/unit/test_turn_beam.py -q
"""

from __future__ import annotations

import sys
import time
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
    Pokemon,
    SearchState,
    SelectContext,
    SelectData,
    SelectType,
    State,
)
from ptcg_ai.action_selection import selector
from ptcg_ai.board_evaluation import board_features
from ptcg_ai.search import turn_beam


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_pokemon(serial: int = 1, hp: int = 60, max_hp: int = 60, energies=None) -> Pokemon:
    return Pokemon(
        id=serial,
        serial=serial,
        hp=hp,
        maxHp=max_hp,
        appearThisTurn=False,
        energies=list(energies or []),
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def make_player(
    *, prizes: int = 1, deck_count: int = 10, hand_count: int = 0,
    active: Pokemon | None = None, bench=None, bench_max: int = 5,
) -> PlayerState:
    return PlayerState(
        active=[active] if active is not None else [],
        bench=list(bench or []),
        benchMax=bench_max,
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
    *, your_index: int = 0, turn: int = 3, first_player: int = 0, result: int = -1,
    my_player: PlayerState | None = None, opp_player: PlayerState | None = None,
    action_count: int = 0,
) -> State:
    mine = my_player if my_player is not None else make_player()
    theirs = opp_player if opp_player is not None else make_player(prizes=6)
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
        players=[mine, theirs],
    )


def make_select(
    option_types, *, select_type: SelectType = SelectType.MAIN, min_count: int = 1,
    max_count: int = 1, context: SelectContext = SelectContext.MAIN,
) -> SelectData:
    return SelectData(
        type=select_type,
        context=context,
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
    "your_deck": [], "your_prize": [], "opponent_deck": [],
    "opponent_prize": [], "opponent_hand": [], "opponent_active": [],
}


class NotReadyPolicy:
    """A policy model stand-in that must never be consulted for scores."""

    is_ready = False

    def score_options(self, state, actions):  # pragma: no cover
        raise AssertionError("score_options must not be called when not is_ready")


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


class AlwaysFailsEngine(FakeEngine):
    """search_step always raises -- for step-failure handling tests."""

    def search_step(self, search_id, select):
        raise RuntimeError("engine exploded")


class SlowStepEngine(FakeEngine):
    """search_step sleeps before responding -- for real-time budget tests."""

    def __init__(self, *args, sleep_seconds: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self._sleep_seconds = sleep_seconds

    def search_step(self, search_id, select):
        time.sleep(self._sleep_seconds)
        return super().search_step(search_id, select)


@pytest.fixture
def install_engine(monkeypatch):
    def _install(engine: FakeEngine):
        monkeypatch.setattr(turn_beam, "cg_api", engine)
        return engine
    return _install


def base_config(**overrides) -> dict:
    cfg = {"enabled": True, "beam_width": 30, "time_limit_ms": 3000, "max_depth": 20}
    cfg.update(overrides)
    return cfg


def run_search(obs, hidden_state_factory=None, policy_model=None, **config_overrides):
    factory = hidden_state_factory or (lambda: dict(DUMMY_HIDDEN_STATE))
    return turn_beam.search(obs, base_config(**config_overrides), factory, policy_model)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_disabled_config_skips_search(install_engine):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine({}))
    assert run_search(obs, policy_model=NotReadyPolicy(), enabled=False) is None
    assert engine.begin_count == 0


def test_non_main_context_skips_search(install_engine):
    obs = make_obs(
        make_state(),
        make_select([OptionType.CARD], select_type=SelectType.CARD, context=SelectContext.TO_HAND),
    )
    engine = install_engine(FakeEngine({}))
    assert run_search(obs, policy_model=NotReadyPolicy()) is None
    assert engine.begin_count == 0


def test_missing_select_or_current_skips_search(install_engine):
    engine = install_engine(FakeEngine({}))
    obs_no_current = Observation(select=make_select([OptionType.END]), logs=[], current=None)
    assert run_search(obs_no_current, policy_model=NotReadyPolicy()) is None
    obs_no_select = Observation(select=None, logs=[], current=make_state())
    assert run_search(obs_no_select, policy_model=NotReadyPolicy()) is None
    assert engine.begin_count == 0


def test_missing_hidden_state_factory_skips_search(install_engine):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine({}))
    assert turn_beam.search(obs, base_config(), None, NotReadyPolicy()) is None
    assert engine.begin_count == 0


def test_hidden_state_factory_returning_none_yields_none(install_engine):
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(FakeEngine({}))
    result = run_search(obs, hidden_state_factory=lambda: None, policy_model=NotReadyPolicy())
    assert result is None
    # Gates passed, so search_begin should never have been reached either
    # (the sample loop skips a None hidden state before calling _run_beam).
    assert engine.begin_count == 0


# ---------------------------------------------------------------------------
# Exceptions never stop the turn
# ---------------------------------------------------------------------------

def test_search_step_failures_are_swallowed_and_return_none(install_engine):
    root_obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    engine = install_engine(AlwaysFailsEngine({"root": FakeNode(root_obs, {})}))
    turn_beam.reset_stats()
    result = run_search(root_obs, policy_model=NotReadyPolicy())
    assert result is None
    stats = turn_beam.get_stats()
    assert stats["step_failures"] >= 2  # both root options failed to step
    assert stats["none_returned"] == 1


def test_search_begin_exception_returns_none(monkeypatch):
    class BoomEngine:
        def search_begin(self, *a, **k):
            raise RuntimeError("boom")

        def search_end(self):
            pass

    monkeypatch.setattr(turn_beam, "cg_api", BoomEngine())
    obs = make_obs(make_state(), make_select([OptionType.ATTACK, OptionType.END]))
    assert run_search(obs, policy_model=NotReadyPolicy()) is None


# ---------------------------------------------------------------------------
# Leaf selection: picks the branch with the better confirmed outcome
# ---------------------------------------------------------------------------

def test_picks_branch_that_takes_a_prize(install_engine):
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.ATTACK]))

    leaf_no_prize = make_state(your_index=1, my_player=make_player(prizes=1))
    leaf_takes_prize = make_state(your_index=1, my_player=make_player(prizes=0))

    engine = install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "no_prize", (1,): "takes_prize"}),
            "no_prize": FakeNode(make_obs(leaf_no_prize, None), {}),
            "takes_prize": FakeNode(make_obs(leaf_takes_prize, None), {}),
        },
    ))
    assert run_search(root_obs, policy_model=NotReadyPolicy()) == [1]
    assert engine.begin_count == 1


def test_win_leaf_beats_everything_else(install_engine):
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.ATTACK]))

    # Branch 0 takes several prizes but doesn't win outright.
    big_prize_gain = make_state(your_index=1, my_player=make_player(prizes=0))
    # Branch 1 is an outright win (State.result == me), despite otherwise
    # looking identical -- it must still be preferred.
    outright_win = make_state(your_index=0, result=0, my_player=make_player(prizes=1))

    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "big_prize", (1,): "win"}),
            "big_prize": FakeNode(make_obs(big_prize_gain, None), {}),
            "win": FakeNode(make_obs(outright_win, None), {}),
        },
    ))
    assert run_search(root_obs, policy_model=NotReadyPolicy()) == [1]


def test_avoids_branch_that_loses_outright(install_engine):
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.ATTACK]))

    # Branch 0 loses outright despite an apparent prize gain.
    outright_loss = make_state(your_index=0, result=1, my_player=make_player(prizes=0))
    # Branch 1 gains nothing but doesn't lose.
    neutral = make_state(your_index=1, my_player=make_player(prizes=1))

    install_engine(FakeEngine(
        {
            "root": FakeNode(root_obs, {(0,): "lose", (1,): "neutral"}),
            "lose": FakeNode(make_obs(outright_loss, None), {}),
            "neutral": FakeNode(make_obs(neutral, None), {}),
        },
    ))
    assert run_search(root_obs, policy_model=NotReadyPolicy()) == [1]


def test_returned_action_is_legal_for_current_select(install_engine):
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.END]))
    leaf = make_state(your_index=1, my_player=make_player(prizes=1))
    install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "leaf", (1,): "leaf"}), "leaf": FakeNode(make_obs(leaf, None), {})},
    ))
    action = run_search(root_obs, policy_model=NotReadyPolicy())
    assert selector.is_valid_action(action, root_obs.select)


# ---------------------------------------------------------------------------
# Beam width actually prunes (fewer leaves explored with a narrower beam)
# ---------------------------------------------------------------------------

def _wide_branching_engine():
    # root has 5 first-level options, each leading to a distinct
    # non-terminal "mid" node that itself has exactly one further option
    # leading to a shared terminal ("opp") leaf. Beam pruning after depth 0
    # decides how many of the 5 mid nodes survive to be expanded (and thus
    # how many terminal leaves get evaluated at depth 1).
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK] * 5, max_count=1))
    opp_state = make_state(your_index=1, my_player=make_player(prizes=1))
    opp_obs = make_obs(opp_state, None)

    nodes = {"root": FakeNode(root_obs, {(i,): f"mid{i}" for i in range(5)}), "opp": FakeNode(opp_obs, {})}
    for i in range(5):
        mid_state = make_state(my_player=make_player(prizes=1))
        mid_obs = make_obs(mid_state, make_select([OptionType.END]))
        nodes[f"mid{i}"] = FakeNode(mid_obs, {(0,): "opp"})
    return root_obs, nodes


def test_narrow_beam_width_evaluates_fewer_leaves(install_engine):
    root_obs, nodes = _wide_branching_engine()
    install_engine(FakeEngine(nodes))
    turn_beam.reset_stats()
    run_search(root_obs, policy_model=NotReadyPolicy(), beam_width=1, max_depth=3)
    narrow_leaves = turn_beam.get_stats()["leaves_evaluated"]

    root_obs2, nodes2 = _wide_branching_engine()
    install_engine(FakeEngine(nodes2))
    turn_beam.reset_stats()
    run_search(root_obs2, policy_model=NotReadyPolicy(), beam_width=5, max_depth=3)
    wide_leaves = turn_beam.get_stats()["leaves_evaluated"]

    assert narrow_leaves == 1
    assert wide_leaves == 5
    assert narrow_leaves < wide_leaves


# ---------------------------------------------------------------------------
# Time budget: returns the best move found so far, never discards progress
# ---------------------------------------------------------------------------

def test_time_limit_returns_best_effort_not_none(install_engine):
    # 3 first-level options, each leading to a non-terminal "mid" node.
    # search_step sleeps 10ms; with a 5ms budget, only the first candidate
    # at depth 0 gets stepped before the deadline check aborts the rest,
    # and the loop itself stops before depth 1 is processed. The single
    # surviving frontier node must still be evaluated in place instead of
    # being thrown away.
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK] * 3, max_count=1))
    nodes = {"root": FakeNode(root_obs, {(i,): f"mid{i}" for i in range(3)})}
    for i in range(3):
        mid_state = make_state(my_player=make_player(prizes=1))
        nodes[f"mid{i}"] = FakeNode(make_obs(mid_state, make_select([OptionType.END])), {})

    install_engine(SlowStepEngine(nodes, sleep_seconds=0.01))
    turn_beam.reset_stats()
    action = run_search(root_obs, policy_model=NotReadyPolicy(), time_limit_ms=5, max_depth=5)
    assert action is not None
    assert action == [0]  # only the first candidate was stepped before the deadline
    assert turn_beam.get_stats()["returned"] == 1


def test_max_depth_cutoff_returns_best_effort_not_none(install_engine):
    # Same idea, but the cutoff is max_depth instead of wall-clock time --
    # exercises the same "evaluate remaining frontier in place" fallback.
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.END]))
    mid_state = make_state(my_player=make_player(prizes=1))
    mid_obs = make_obs(mid_state, make_select([OptionType.END]))
    install_engine(FakeEngine({"root": FakeNode(root_obs, {(0,): "mid"}), "mid": FakeNode(mid_obs, {})}))
    action = run_search(root_obs, policy_model=NotReadyPolicy(), max_depth=1)
    assert action is not None


# ---------------------------------------------------------------------------
# hidden_state_samples: averages leaf scores across decretized samples
# ---------------------------------------------------------------------------

def test_hidden_state_samples_averages_across_shuffles(install_engine):
    opp_full_hp = make_pokemon(serial=1, hp=100, max_hp=100)
    root_state = make_state(
        my_player=make_player(prizes=1),
        opp_player=make_player(prizes=6, active=opp_full_hp),
    )
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.ATTACK]))

    def leaf(hp: int) -> State:
        return make_state(
            your_index=1,
            my_player=make_player(prizes=1),
            opp_player=make_player(prizes=6, active=make_pokemon(serial=1, hp=hp, max_hp=100)),
        )

    # Sample A: action0 deals 100 damage, action1 deals 90 -> action0 wins alone.
    # Sample B: action0 deals 80 damage, action1 deals 95 -> action1 wins alone.
    # Averaged: action0 = 90, action1 = 92.5 -> action1 must win overall.
    nodes = {
        "rootA": FakeNode(root_obs, {(0,): "a0", (1,): "a1"}),
        "a0": FakeNode(make_obs(leaf(hp=0), None), {}),      # 100 damage
        "a1": FakeNode(make_obs(leaf(hp=10), None), {}),     # 90 damage
        "rootB": FakeNode(root_obs, {(0,): "b0", (1,): "b1"}),
        "b0": FakeNode(make_obs(leaf(hp=20), None), {}),     # 80 damage
        "b1": FakeNode(make_obs(leaf(hp=5), None), {}),      # 95 damage
    }
    engine = install_engine(FakeEngine(nodes, root=["rootA", "rootB"]))
    action = run_search(root_obs, policy_model=NotReadyPolicy(), hidden_state_samples=2)
    assert action == [1]
    assert engine.begin_count == 2


def test_single_sample_by_default(install_engine):
    root_state = make_state(my_player=make_player(prizes=1))
    root_obs = make_obs(root_state, make_select([OptionType.ATTACK, OptionType.END]))
    leaf = make_state(your_index=1, my_player=make_player(prizes=0))
    engine = install_engine(FakeEngine(
        {"root": FakeNode(root_obs, {(0,): "leaf", (1,): "leaf"}), "leaf": FakeNode(make_obs(leaf, None), {})},
    ))
    run_search(root_obs, policy_model=NotReadyPolicy())
    assert engine.begin_count == 1  # DEFAULTS["hidden_state_samples"] == 1


# ---------------------------------------------------------------------------
# evaluate_leaf and its components (no fake engine needed)
# ---------------------------------------------------------------------------

def test_prizes_taken_counts_shrinkage_of_own_prize_list():
    root = make_state(my_player=make_player(prizes=3))
    leaf = make_state(my_player=make_player(prizes=1))
    assert turn_beam._prizes_taken(root, leaf, me=0) == 2


def test_damage_dealt_counts_partial_damage_by_serial():
    root_opp = make_player(prizes=6, active=make_pokemon(serial=1, hp=100, max_hp=100))
    leaf_opp = make_player(prizes=6, active=make_pokemon(serial=1, hp=40, max_hp=100))
    assert turn_beam._damage_dealt(root_opp, leaf_opp) == 60


def test_damage_dealt_counts_full_remaining_hp_for_knocked_out_pokemon():
    # The Pokemon that had 30 damage on it (hp=70/100) is knocked out and
    # vanishes from the leaf's active/bench -- the KO must still count as
    # 70 damage dealt (its full remaining HP), not 0.
    root_opp = make_player(prizes=6, active=make_pokemon(serial=1, hp=70, max_hp=100))
    leaf_opp = make_player(prizes=5, active=None)  # replaced/empty after KO
    assert turn_beam._damage_dealt(root_opp, leaf_opp) == 70


def test_ko_risk_is_zero_without_an_active_pokemon():
    leaf = make_state(my_player=make_player(prizes=1, active=None))
    assert turn_beam._ko_risk(leaf, me=0) == 0.0


def test_ko_risk_reuses_board_features_is_likely_ko_next_turn(monkeypatch):
    active = make_pokemon(serial=1, hp=10, max_hp=100)
    leaf = make_state(my_player=make_player(prizes=1, active=active))
    monkeypatch.setattr(board_features, "is_likely_ko_next_turn", lambda pokemon, state, me: pokemon is active)
    assert turn_beam._ko_risk(leaf, me=0) == 1.0


def test_development_score_counts_bench_and_energy():
    bench = [make_pokemon(serial=2, energies=[0]), make_pokemon(serial=3, energies=[0, 0])]
    active = make_pokemon(serial=1, energies=[0])
    state = make_state(my_player=make_player(prizes=1, active=active, bench=bench))
    # bench_count(2) + total energy (1 + 1 + 2 = 4) == 6
    assert turn_beam._development_score(state, me=0) == 6.0


def test_evaluate_leaf_priority_ordering_prizes_dominate_damage():
    root = make_state(
        my_player=make_player(prizes=2),
        opp_player=make_player(prizes=6, active=make_pokemon(serial=1, hp=100, max_hp=100)),
    )
    # Takes no prize but deals massive damage.
    big_damage_no_prize = make_state(
        my_player=make_player(prizes=2),
        opp_player=make_player(prizes=6, active=make_pokemon(serial=1, hp=0, max_hp=100)),
    )
    # Takes one prize but deals no damage.
    prize_no_damage = make_state(
        my_player=make_player(prizes=1),
        opp_player=make_player(prizes=6, active=make_pokemon(serial=1, hp=100, max_hp=100)),
    )
    score_damage = turn_beam.evaluate_leaf(root, big_damage_no_prize, me=0)
    score_prize = turn_beam.evaluate_leaf(root, prize_no_damage, me=0)
    assert score_prize > score_damage


def test_evaluate_leaf_weights_are_configurable():
    root = make_state(my_player=make_player(prizes=1))
    leaf = make_state(my_player=make_player(prizes=1))
    default_score = turn_beam.evaluate_leaf(root, leaf, me=0)
    custom_score = turn_beam.evaluate_leaf(root, leaf, me=0, weights={"development": 5.0})
    assert default_score == 0.0
    assert custom_score == 0.0  # no bench/energy difference either way, but no KeyError


# ---------------------------------------------------------------------------
# _candidate_selections: unlike lethal_simple, END is never pruned
# ---------------------------------------------------------------------------

def test_candidate_selections_includes_end():
    select = make_select([OptionType.ATTACK, OptionType.END])
    candidates = list(turn_beam._candidate_selections(select, {"max_combinations_per_select": 64}))
    assert [1] in candidates  # END's index


def test_candidate_selections_respects_min_max_and_cap():
    select = make_select(
        [OptionType.CARD, OptionType.CARD, OptionType.CARD],
        select_type=SelectType.CARD, min_count=2, max_count=2,
    )
    candidates = list(turn_beam._candidate_selections(select, {"max_combinations_per_select": 64}))
    assert all(len(c) == 2 for c in candidates)
    assert sorted(tuple(c) for c in candidates) == [(0, 1), (0, 2), (1, 2)]


def test_candidate_selections_caps_combinations():
    select = make_select(
        [OptionType.CARD] * 6, select_type=SelectType.CARD, min_count=0, max_count=6,
    )
    candidates = list(turn_beam._candidate_selections(select, {"max_combinations_per_select": 3}))
    assert len(candidates) == 3
