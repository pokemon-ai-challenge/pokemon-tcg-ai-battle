"""**製品の Phase 1/2** を toy oracle と突き合わせる(Step 1-3)。

Step 1-2 では toy 専用の候補実装を比較したが、ここで比較するのは
``ptcg_ai.search.lethal.phase1`` / ``phase2`` そのもの。実エンジン用の
backend と同じインタフェース越しに動かしている。

観点:
1. 全シナリオで oracle と proof・root action が一致する
2. `UNKNOWN` を `PROVEN_NO_WIN` に丸めない(予算・深さ・未対応効果)
3. outcome 列挙が不完全・拒否されたら `PROVEN_WIN` にならない
4. transposition の hash 衝突が結果を変えない
5. マクロ(順序付け)を通しても結果が変わらない
"""

from fractions import Fraction

import pytest

from ptcg_ai.search.lethal import phase1, phase2
from ptcg_ai.search.lethal.budget import Budget
from ptcg_ai.search.lethal.enumeration import Outcome, OutcomeSet
from ptcg_ai.search.lethal.backend import OutcomeEnumeration, Transition
from ptcg_ai.search.lethal.types import ChanceClass, Proof, StopReason

from tests.toy import macros, oracle
from tests.toy.toy_game import DUD, STRIKE, make_state
from tests.toy.toy_lethal_backend import ConstantHashKey, ToyLethalBackend
from tests.unit.lethal.test_toy_oracle import SCENARIOS, scenario_ids


def budget_for(depth: int, **overrides) -> Budget:
    defaults = dict(time_limit_ms=2000.0, max_nodes=200_000, max_depth=depth, max_chance_depth=2)
    defaults.update(overrides)
    return Budget(**defaults)


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_production_phase1_matches_oracle(name, state, depth):
    truth = oracle.analyze_deterministic(state, depth)
    result = phase1.search(ToyLethalBackend(state), budget_for(depth))
    assert result.proof is truth.proof, name
    if truth.proof is Proof.PROVEN_WIN:
        assert result.depth == truth.min_depth, name
        assert result.first_action in truth.winning_root_actions, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_production_phase2_matches_oracle(name, state, depth):
    truth = oracle.analyze(state, depth)
    result = phase2.search(ToyLethalBackend(state), budget_for(depth))
    assert result.proof is truth.proof, name
    if truth.proof is Proof.PROVEN_WIN:
        assert result.first_action in truth.winning_root_actions, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_production_phase2_matches_with_macro_ordering(name, state, depth):
    """マクロで順序を変えても結果は同じ(順序付けであって枝刈りではない)。"""
    truth = oracle.analyze(state, depth)
    backend = ToyLethalBackend(state, action_source=macros.macro_actions)
    result = phase2.search(backend, budget_for(depth))
    assert result.proof is truth.proof, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_transposition_hash_collisions_do_not_change_results(name, state, depth):
    """観測キーの hash が全部衝突しても結果が変わらない(等価性で比較している)。"""
    plain = phase2.search(ToyLethalBackend(state), budget_for(depth))
    colliding = phase2.search(
        ToyLethalBackend(state, key_wrapper=ConstantHashKey), budget_for(depth)
    )
    assert plain.proof is colliding.proof, name


# ------------------------------------------------- UNKNOWN に倒れることの確認


def test_node_limit_yields_unknown_not_no_win():
    name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(ToyLethalBackend(state), budget_for(depth, max_nodes=1))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.NODE_LIMIT in result.stop_reasons


def test_time_limit_yields_unknown_not_no_win():
    name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(ToyLethalBackend(state), budget_for(depth, time_limit_ms=0.0))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.TIME_LIMIT in result.stop_reasons


def test_chance_depth_limit_yields_unknown():
    name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(ToyLethalBackend(state), budget_for(depth, max_chance_depth=0))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.CHANCE_DEPTH_LIMIT in result.stop_reasons


def test_refused_outcomes_yield_unknown():
    """未対応効果・シャッフル後など、列挙を拒否されたら UNKNOWN。"""
    name, state, depth = _entry("draw_then_always_win")
    backend = ToyLethalBackend(state, refuse_outcomes=StopReason.SHUFFLE_ENCOUNTERED)
    result = phase2.search(backend, budget_for(depth))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.SHUFFLE_ENCOUNTERED in result.stop_reasons


def test_incomplete_action_set_blocks_proven_no_win():
    """候補が不完全なら「勝てない」と言わない。"""
    name, state, depth = _entry("no_lethal")
    backend = ToyLethalBackend(state, incomplete_actions=True)
    result = phase2.search(backend, budget_for(depth))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.INCOMPLETE_ACTION_SET in result.stop_reasons
    result1 = phase1.search(ToyLethalBackend(state, incomplete_actions=True), budget_for(depth))
    assert result1.proof is Proof.UNKNOWN


def test_incomplete_outcome_mass_yields_unknown():
    """質量合計が 1 にならない outcome 集合では PROVEN_WIN にしない。"""
    name, state, depth = _entry("draw_then_always_win")

    class TruncatingBackend(ToyLethalBackend):
        def enumerate_outcomes(self, node, action):
            full = super().enumerate_outcomes(node, action)
            kept = full.outcome_set.outcomes[:-1]
            dropped = full.outcome_set.outcomes[-1].mass
            return OutcomeEnumeration(
                OutcomeSet(kept, full.outcome_set.source, coverage_certified=True,
                           unprocessed_mass=dropped),
                ChanceClass.CONTROLLED_SUPPLY_ORDER,
            )

    result = phase2.search(TruncatingBackend(state), budget_for(depth))
    assert result.proof is not Proof.PROVEN_WIN, name
    assert StopReason.OUTCOMES_NOT_ENUMERABLE in result.stop_reasons


def test_unconstructible_outcome_yields_unknown():
    """outcome を状態として構築できなければ PROVEN_WIN にしない。"""
    name, state, depth = _entry("draw_then_always_win")

    class UnbuildableBackend(ToyLethalBackend):
        def apply_outcome(self, node, action, outcome):
            return Transition(None, stop_reason=StopReason.OUTCOMES_NOT_ENUMERABLE)

    result = phase2.search(UnbuildableBackend(state), budget_for(depth))
    assert result.proof is Proof.UNKNOWN, name


def test_engine_random_class_yields_unknown():
    """クラス S と宣言された事象は証明に使えない。"""
    name, state, depth = _entry("draw_then_always_win")

    class EngineRandomBackend(ToyLethalBackend):
        def enumerate_outcomes(self, node, action):
            full = super().enumerate_outcomes(node, action)
            return OutcomeEnumeration(full.outcome_set, ChanceClass.ENGINE_RANDOM)

    result = phase2.search(EngineRandomBackend(state), budget_for(depth))
    assert result.proof is Proof.UNKNOWN, name
    assert StopReason.OUTCOMES_NOT_ENUMERABLE in result.stop_reasons


def test_unsupported_transition_blocks_proven_no_win():
    """未対応効果に当たった枝があるとき、PROVEN_NO_WIN を主張しない。"""
    name, state, depth = _entry("no_lethal")

    class UnsupportedBackend(ToyLethalBackend):
        def apply(self, node, action):
            return Transition(None, stop_reason=StopReason.UNSUPPORTED_EFFECT)

    result2 = phase2.search(UnsupportedBackend(state), budget_for(depth))
    assert result2.proof is Proof.UNKNOWN, name
    result1 = phase1.search(UnsupportedBackend(state), budget_for(depth))
    assert result1.proof is Proof.UNKNOWN, name


# ------------------------------------------------- B4: 相手選択(AND ノード)

# STRIKE は 3 ダメージ。KO するとサイドを 1 枚取り、まだ残っていてベンチがいれば
# **相手が**次のバトルポケモンを選ぶ。
B4_SCENARIOS = [
    # ケース A: どちらを出されても倒し切れる → PROVEN_WIN
    ("b4_all_choices_lose",
     make_state([STRIKE, STRIKE], [DUD, DUD], 3, prizes_left=2, opponent_bench=(3, 3)), 4),
    # ケース B: 片方(HP9)を出されると倒せない → PROVEN_NO_WIN
    ("b4_one_choice_survives",
     make_state([STRIKE, STRIKE], [DUD, DUD], 3, prizes_left=2, opponent_bench=(3, 9)), 4),
    # 選択肢が1つしかない場合(相手に選択の余地が無い)
    ("b4_single_choice",
     make_state([STRIKE, STRIKE], [DUD, DUD], 3, prizes_left=2, opponent_bench=(3,)), 4),
]


@pytest.mark.parametrize("name,state,depth", B4_SCENARIOS, ids=[s[0] for s in B4_SCENARIOS])
def test_opponent_choice_matches_oracle(name, state, depth):
    """相手選択ノードでも oracle と一致する(Phase 1 / Phase 2 とも)。"""
    truth1 = oracle.analyze_deterministic(state, depth)
    result1 = phase1.search(ToyLethalBackend(state), budget_for(depth))
    assert result1.proof is truth1.proof, f"{name}: phase1"

    truth2 = oracle.analyze(state, depth)
    result2 = phase2.search(ToyLethalBackend(state), budget_for(depth))
    assert result2.proof is truth2.proof, f"{name}: phase2"


def test_case_a_all_opponent_choices_win_is_proven_win():
    name, state, depth = _b4("b4_all_choices_lose")
    for search in (phase1.search, phase2.search):
        result = search(ToyLethalBackend(state), budget_for(depth))
        assert result.proof is Proof.PROVEN_WIN, name
        assert result.first_action == ("play", STRIKE)


def test_case_b_one_bad_opponent_choice_is_proven_no_win():
    """相手が有利な方(倒せない方)を選べるなら、確定勝ちにしない。"""
    name, state, depth = _b4("b4_one_choice_survives")
    for search in (phase1.search, phase2.search):
        result = search(ToyLethalBackend(state), budget_for(depth))
        assert result.proof is Proof.PROVEN_NO_WIN, name


def test_case_c_incomplete_opponent_choice_set_is_unknown():
    """相手選択を全部は確認できない場合、勝ちを主張しない。"""
    name, state, depth = _b4("b4_all_choices_lose")
    backend = ToyLethalBackend(state, incomplete_actions=True)
    for search in (phase1.search, phase2.search):
        result = search(backend, budget_for(depth))
        assert result.proof is Proof.UNKNOWN, name
        assert StopReason.INCOMPLETE_ACTION_SET in result.stop_reasons


def test_opponent_choice_is_not_averaged_or_cherry_picked():
    """相手選択を平均しない・都合のよい選択だけを採らない。

    ケース B は「2 択のうち 1 つは勝てる」局面。もし平均していたり、
    有利な選択だけを見ていれば PROVEN_WIN になってしまう。
    """
    _name, state, depth = _b4("b4_one_choice_survives")
    result = phase2.search(ToyLethalBackend(state), budget_for(depth))
    assert result.proof is not Proof.PROVEN_WIN
    # 実際に「片方の選択なら勝てる」ことを確認しておく(テストの前提)。
    from tests.toy.toy_game import BACKEND

    after_ko = BACKEND.apply(state, ("play", STRIKE))
    assert BACKEND.is_opponent_node(after_ko)
    outcomes = []
    for action in BACKEND.legal_actions(after_ko):
        branch = phase2.search(
            ToyLethalBackend(BACKEND.apply(after_ko, action)), budget_for(depth)
        )
        outcomes.append(branch.proof)
    assert Proof.PROVEN_WIN in outcomes and Proof.PROVEN_NO_WIN in outcomes


def test_opponent_node_is_not_terminal():
    from tests.toy.toy_game import BACKEND

    _name, state, _depth = _b4("b4_all_choices_lose")
    after_ko = BACKEND.apply(state, ("play", STRIKE))
    assert BACKEND.is_opponent_node(after_ko)
    assert not BACKEND.is_terminal(after_ko)
    assert not BACKEND.is_win(after_ko)


def _b4(name):
    for entry in B4_SCENARIOS:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def test_masses_must_be_exact_fractions():
    with pytest.raises(TypeError):
        Outcome(label=(), mass=0.5)  # type: ignore[arg-type]
    assert Outcome(label=(), mass=Fraction(1)).mass == 1


def _entry(name):
    for entry in SCENARIOS:
        if entry[0] == name:
            return entry
    raise KeyError(name)


# ------------------- outcome 構築失敗の 3 ケース(Step 1-7a: 安全側フォールバック)


def _failing_backend(state, failing_indices):
    """指定した index の outcome だけ構築に失敗する backend。"""

    class PartiallyUnbuildable(ToyLethalBackend):
        def apply_outcome(self, node, action, outcome):
            enumeration = self.enumerate_outcomes(node, action)
            labels = [o.label for o in enumeration.outcome_set.outcomes]
            if labels.index(outcome.label) in failing_indices:
                return Transition(None, stop_reason=StopReason.OUTCOMES_NOT_ENUMERABLE)
            return super().apply_outcome(node, action, outcome)

    return PartiallyUnbuildable(state)


def test_single_outcome_build_failure_is_unknown():
    """1 つでも構築できなければ、全 outcome を覆えていないので UNKNOWN。"""
    _name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(_failing_backend(state, {0}), budget_for(depth))
    assert result.proof is Proof.UNKNOWN
    assert StopReason.OUTCOMES_NOT_ENUMERABLE in result.stop_reasons


def test_partial_outcome_build_failure_is_unknown():
    _name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(_failing_backend(state, {0, 2}), budget_for(depth))
    assert result.proof is Proof.UNKNOWN


def test_all_outcome_build_failures_are_unknown():
    _name, state, depth = _entry("draw_then_always_win")
    result = phase2.search(_failing_backend(state, set(range(99))), budget_for(depth))
    assert result.proof is Proof.UNKNOWN
    # 「構築できなかった」を勝ちにも負けにも丸めない
    assert result.proof is not Proof.PROVEN_NO_WIN
