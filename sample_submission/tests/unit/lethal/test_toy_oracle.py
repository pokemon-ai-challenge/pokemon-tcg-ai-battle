"""toy oracle との比較ハーネス(Step 1-2c〜1-2f)。

ここで確認するのは **toy state の範囲**での次の性質だけである
(実エンジンに対する保証ではない):

- 探索候補の判定が、独立実装の ground-truth と全ケース一致する
- ``PROVEN_WIN`` / ``PROVEN_NO_WIN`` / ``UNKNOWN`` を厳密に区別している
- outcome 列挙が完全で、確率質量が厳密に 1 になる
- ドロー前は隠れた順序に依存せず、ドロー後は公開結果で手を変えられる
- マクロ化しても勝ち筋を落とさない
- 意図的に壊した実装をハーネスが検出できる
"""

from fractions import Fraction
from itertools import permutations

import pytest

from ptcg_ai.search.lethal.enumeration import (
    SOURCE_SAMPLED,
    Outcome,
    OutcomeSet,
    certify_for_proof,
    draw_outcomes,
)
from ptcg_ai.search.lethal.types import ChanceClass, Proof, StopReason

from tests.toy import candidates, macros, oracle
from tests.toy.toy_game import BOOST, DRAW2, DUD, STRIKE, make_state

# --------------------------------------------------------------- toy シナリオ
# (名前, 状態, 探索深さ)。深さは primitive 行動数。
SCENARIOS = [
    # 手札だけで確定リーサル(Phase 1 相当)
    ("deterministic_two_strikes", make_state([STRIKE, STRIKE], [DUD, DUD, DUD], 6), 3),
    # ブーストしてから殴る必要がある(固定優先順位だと落としやすい)
    ("boost_then_strike", make_state([BOOST, STRIKE], [DUD, DUD, DUD], 6), 3),
    # 引かないと勝てないが、どの outcome でも勝てる(Phase 2 相当の PROVEN_WIN)
    ("draw_then_always_win", make_state([DRAW2, STRIKE], [STRIKE, BOOST, STRIKE, BOOST], 6), 3),
    # 引いても負ける outcome がある(PROVEN_NO_WIN。確率は 5/6)
    ("draw_sometimes_win", make_state([DRAW2, STRIKE], [STRIKE, BOOST, DUD, DUD], 6), 3),
    # 負ける outcome の確率が単独最小(outcome を1つ落とす実装が検出できる形)
    ("draw_rare_miss", make_state([DRAW2, STRIKE], [STRIKE, STRIKE, STRIKE, DUD, DUD], 6), 3),
    # そもそも届かない
    ("no_lethal", make_state([STRIKE], [DUD, DUD, DUD], 20), 3),
    # 手数が足りない
    ("not_enough_actions", make_state([STRIKE, STRIKE], [DUD, DUD], 6, actions_left=1), 3),
    # 深さで打ち切られる(UNKNOWN にならなければならない)
    ("depth_limited", make_state([STRIKE, STRIKE], [DUD, DUD, DUD], 6), 1),
]

DETERMINIZATION_SCENARIOS = [
    ("draw_then_always_win", make_state([DRAW2, STRIKE], [STRIKE, BOOST, STRIKE, BOOST], 6), 3),
    ("draw_sometimes_win", make_state([DRAW2, STRIKE], [STRIKE, BOOST, DUD, DUD], 6), 3),
]


def scenario_ids():
    return [name for name, _state, _depth in SCENARIOS]


# ------------------------------------------------------- Step 1-2c 比較ハーネス


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_phase1_candidate_matches_oracle(name, state, depth):
    """Phase 1 相当: 確定勝利の有無・最短手数・root action が一致する。"""
    truth = oracle.analyze_deterministic(state, depth)
    result = candidates.phase1(state, depth)
    assert result.proof is truth.proof, name
    assert result.min_depth == truth.min_depth, name
    assert result.winning_root_actions == truth.winning_root_actions, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_phase2_candidate_matches_oracle(name, state, depth):
    """Phase 2 相当: proof と root action の集合が一致する。"""
    truth = oracle.analyze(state, depth)
    result = candidates.phase2(state, depth)
    assert result.proof is truth.proof, name
    assert result.winning_root_actions == truth.winning_root_actions, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_phase2_policy_tree_covers_every_outcome(name, state, depth):
    """勝ち筋の方策木が、全非0 outcome を覆っていること。"""
    truth = oracle.analyze(state, depth)
    result = candidates.phase2(state, depth)
    if truth.proof is not Proof.PROVEN_WIN:
        assert result.policy_tree is None, name
        return
    assert result.policy_tree is not None, name
    assert _tree_shape(result.policy_tree) == _tree_shape(truth.policy_tree), name


def _tree_shape(node):
    """方策木を(行動, 子の観測キー集合)の再帰構造へ落として比較する。"""
    if node is None or node[0] == "win":
        return ("win",)
    _kind, action, children = node
    return (action, frozenset((key, _tree_shape(child)) for key, child in children.items()))


def test_comparison_counts():
    """比較件数の記録(完了条件の報告用)。"""
    compared = 0
    matched = 0
    for _name, state, depth in SCENARIOS:
        for truth, result in (
            (oracle.analyze_deterministic(state, depth), candidates.phase1(state, depth)),
            (oracle.analyze(state, depth), candidates.phase2(state, depth)),
        ):
            compared += 1
            if (
                truth.proof is result.proof
                and truth.winning_root_actions == result.winning_root_actions
            ):
                matched += 1
    assert compared == len(SCENARIOS) * 2
    assert matched == compared, f"{matched}/{compared} しか一致していない"


# -------------------------------------------- 3値の区別(見つからない != 勝てない)


def test_three_valued_proof_examples():
    """PROVEN_WIN / PROVEN_NO_WIN / UNKNOWN の判定例。"""
    win = candidates.phase2(*_scenario("deterministic_two_strikes"))
    assert win.proof is Proof.PROVEN_WIN

    no_win = candidates.phase2(*_scenario("no_lethal"))
    assert no_win.proof is Proof.PROVEN_NO_WIN

    unknown = candidates.phase2(*_scenario("depth_limited"))
    assert unknown.proof is Proof.UNKNOWN
    assert unknown.stop_reason is StopReason.DEPTH_LIMIT
    # 深さで打ち切っただけなのに「勝てない」と言わないこと。
    assert unknown.proof is not Proof.PROVEN_NO_WIN


def test_depth_limited_mutant_is_detected():
    """「見つからなかった」を PROVEN_NO_WIN にする実装を検出できる。"""
    name, state, depth = _scenario_entry("depth_limited")
    truth = oracle.analyze(state, depth)
    broken = candidates.phase2_no_win_on_depth_limit(state, depth)
    assert truth.proof is Proof.UNKNOWN, name
    assert broken.proof is Proof.PROVEN_NO_WIN
    assert broken.proof is not truth.proof  # ハーネスが差を検出する


def test_incomplete_enumeration_never_yields_proven_win():
    """outcome 列挙が不完全なら PROVEN_WIN にしない(証明可否の判定が効く)。"""
    name, state, depth = _scenario_entry("draw_rare_miss")

    def truncated_builder(node):
        full = draw_outcomes(node.deck_multiset(), 2)
        kept = full.outcomes[:-1]
        unprocessed = full.outcomes[-1].mass
        return OutcomeSet(kept, full.source, coverage_certified=True,
                          unprocessed_mass=unprocessed)

    result = candidates.phase2(state, depth, outcome_builder=truncated_builder)
    assert result.proof is not Proof.PROVEN_WIN, name
    assert result.stop_reason is StopReason.OUTCOMES_NOT_ENUMERABLE


# ------------------------------------------ Step 1-2e outcome completeness


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_outcome_masses_are_exact_and_complete(name, state, depth):
    """厳密列挙の質量が Fraction で合計 1、ラベル一意、全て構築可能。"""
    outcome_set = draw_outcomes(state.deck_multiset(), 2)
    assert outcome_set.total_mass() == Fraction(1), name
    assert outcome_set.labels_unique(), name
    assert all(isinstance(o.mass, Fraction) and o.mass > 0 for o in outcome_set.outcomes)
    certification = certify_for_proof(
        outcome_set, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER
    )
    assert certification.admissible, certification.failures
    # 各 outcome を実際に構築できること(構築可能性)。
    canonical = candidates.canonical(state)
    for outcome in outcome_set.outcomes:
        drawn = candidates._label_to_cards(outcome.label)
        realized = candidates.realize(canonical, drawn)
        assert realized.deck[: len(drawn)] == drawn, name


@pytest.mark.parametrize(
    "name,state,depth",
    [s for s in SCENARIOS if DRAW2 in s[1].hand],
    ids=[s[0] for s in SCENARIOS if DRAW2 in s[1].hand],
)
def test_candidate_coverage_matches_oracle_coverage(name, state, depth):
    """探索が覆った outcome 集合が、実際に起こりうる集合と一致する。"""
    truth = oracle.chance_coverage(state, depth)
    result = candidates.phase2(state, depth)
    assert result.coverage, name
    for path, observed in result.coverage.items():
        assert path in truth, f"{name}: 想定外の chance ノード {path}"
        assert observed == truth[path], (
            f"{name}: outcome coverage 不一致 path={path} "
            f"探索={sorted(observed)} 正解={sorted(truth[path])}"
        )


def test_hypergeometric_matches_counting_by_permutation():
    """確率の導出が独立であることの確認: 解析式 vs 順列の数え上げ。"""
    deck = (STRIKE, STRIKE, BOOST, DUD, DUD)
    analytic = {o.label: o.mass for o in draw_outcomes({
        STRIKE: 2, BOOST: 1, DUD: 2
    }, 2).outcomes}

    counted: dict = {}
    orders = sorted(set(permutations(deck)))
    for order in orders:
        drawn = tuple(sorted(order[:2]))
        label = tuple(sorted((card, drawn.count(card)) for card in set(drawn)))
        counted[label] = counted.get(label, Fraction(0)) + Fraction(1, len(orders))
    assert analytic == counted


def test_sampled_outcome_set_is_rejected_for_proof():
    """サンプリング由来の outcome 集合は証明に使えない。"""
    sampled = OutcomeSet(
        (Outcome(label=((STRIKE, 2),), mass=Fraction(1)),),
        SOURCE_SAMPLED,
        coverage_certified=False,
    )
    certification = certify_for_proof(
        sampled, chance_class=ChanceClass.CONTROLLED_SUPPLY_ORDER
    )
    assert not certification.admissible
    assert certification.stop_reason is StopReason.OUTCOMES_NOT_ENUMERABLE
    assert "source_not_provable:sampled" in certification.failures


def test_engine_random_class_is_rejected_for_proof():
    """クラス S(エンジン内部 RNG)の事象は、質量が揃っていても証明に使えない。"""
    outcome_set = draw_outcomes({STRIKE: 2, DUD: 2}, 2)
    certification = certify_for_proof(
        outcome_set, chance_class=ChanceClass.ENGINE_RANDOM
    )
    assert not certification.admissible
    assert any("chance_class_not_enumerable" in f for f in certification.failures)


# ----------------------------------------- 意図的に壊した実装(検出力の確認)


def test_dropping_an_outcome_produces_a_false_proven_win():
    """低確率 outcome を捨てる実装は、oracle 比較で検出される。"""
    name, state, depth = _scenario_entry("draw_rare_miss")
    truth = oracle.analyze(state, depth)
    broken = candidates.phase2_dropping_outcome(state, depth)
    assert truth.proof is Proof.PROVEN_NO_WIN, name
    assert broken.proof is Proof.PROVEN_WIN, "壊した実装が誤検出を起こさなかった"
    assert broken.proof is not truth.proof


def test_sampling_implementation_produces_a_false_proven_win():
    """1サンプルで確定判定する実装も検出される。"""
    name, state, depth = _scenario_entry("draw_rare_miss")
    truth = oracle.analyze(state, depth)
    broken = candidates.phase2_sampling(state, depth, seed=1)
    assert truth.proof is Proof.PROVEN_NO_WIN, name
    assert broken.proof is Proof.PROVEN_WIN
    assert broken.proof is not truth.proof


# ------------------------------------- Step 1-2d determinization 境界テスト


@pytest.mark.parametrize(
    "name,state,depth", DETERMINIZATION_SCENARIOS, ids=[s[0] for s in DETERMINIZATION_SCENARIOS]
)
def test_results_do_not_depend_on_hidden_deck_order(name, state, depth):
    """ドロー前: 隠れた順序が違っても proof・root action・観測キーが同じ。"""
    proofs = set()
    roots = set()
    keys = set()
    for order in sorted(set(permutations(state.deck))):
        variant = state.__class__(**{**state.__dict__, "deck": order})
        keys.add(variant.observable())
        result1 = candidates.phase1(variant, depth)
        result2 = candidates.phase2(variant, depth)
        truth = oracle.analyze(variant, depth)
        proofs.add((result1.proof, result2.proof, truth.proof))
        roots.add((result1.winning_root_actions, result2.winning_root_actions,
                   truth.winning_root_actions))
    assert len(keys) == 1, f"{name}: 観測キーが隠れた順序に依存した"
    assert len(proofs) == 1, f"{name}: proof が隠れた順序に依存した"
    assert len(roots) == 1, f"{name}: root action が隠れた順序に依存した"


def test_peeking_implementation_is_detected():
    """順序を覗く実装は、この境界テストで検出される。

    勝てる root action が2つある局面(STRIKE→STRIKE と BOOST→STRIKE のどちらでも
    ちょうど6ダメージ)を使う。1つしか無いと、覗いていても選択が変わらないため。
    """
    state = make_state([STRIKE, STRIKE, BOOST], [STRIKE, BOOST, DUD], 6)
    baseline = candidates.phase1(state, 3)
    assert len(baseline.winning_root_actions) > 1, "前提: 勝てる root action が複数あること"
    roots = set()
    for order in sorted(set(permutations(state.deck))):
        variant = state.__class__(**{**state.__dict__, "deck": order})
        roots.add(candidates.phase1_peeking(variant, 3).winning_root_actions)
    assert len(roots) > 1, "順序を覗く実装を検出できなかった"


def test_actions_may_depend_on_the_revealed_outcome():
    """ドロー後: 公開された結果に応じて後続行動が変わってよい(むしろ変わるべき)。"""
    name, state, depth = _scenario_entry("draw_then_always_win")
    result = candidates.phase2(state, depth)
    assert result.proof is Proof.PROVEN_WIN, name
    _kind, _action, children = result.policy_tree
    follow_ups = {child[1] for child in children.values() if child[0] == "action"}
    assert len(follow_ups) > 1, (
        "全 outcome で同じ後続行動になっている。"
        "公開結果に応じた再選択が働いていない可能性がある"
    )


# ------------------------------------------------- Step 1-2f マクロ前後の一致


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_macro_search_matches_primitive_search(name, state, depth):
    """マクロ(順序付けのみ)を通しても、結果がプリミティブ全探索と一致する。"""
    primitive1 = candidates.phase1(state, depth)
    macro1 = candidates.phase1(state, depth, action_source=macros.macro_actions)
    assert macro1.proof is primitive1.proof, name
    assert macro1.winning_root_actions == primitive1.winning_root_actions, name
    assert macro1.min_depth == primitive1.min_depth, name

    primitive2 = candidates.phase2(state, depth)
    macro2 = candidates.phase2(state, depth, action_source=macros.macro_actions)
    assert macro2.proof is primitive2.proof, name
    assert macro2.winning_root_actions == primitive2.winning_root_actions, name


@pytest.mark.parametrize("name,state,depth", SCENARIOS, ids=scenario_ids())
def test_macro_search_matches_the_oracle(name, state, depth):
    """マクロ探索の結果が ground-truth とも一致する(勝ち筋を落としていない)。"""
    truth1 = oracle.analyze_deterministic(state, depth)
    macro1 = candidates.phase1(state, depth, action_source=macros.macro_actions)
    assert macro1.proof is truth1.proof, name
    assert macro1.winning_root_actions == truth1.winning_root_actions, name

    truth2 = oracle.analyze(state, depth)
    macro2 = candidates.phase2(state, depth, action_source=macros.macro_actions)
    assert macro2.proof is truth2.proof, name
    assert macro2.winning_root_actions == truth2.winning_root_actions, name


def test_lossy_macro_pruning_is_detected():
    """健全でない枝刈りを入れたマクロは、勝ち筋を落として検出される。"""
    name, state, depth = _scenario_entry("boost_then_strike")
    truth = oracle.analyze(state, depth)
    lossy = candidates.phase2(state, depth, action_source=macros.lossy_macro_actions)
    assert truth.proof is Proof.PROVEN_WIN, name
    assert lossy.proof is not Proof.PROVEN_WIN, "枝刈りの欠陥を検出できなかった"


def test_macro_generator_does_not_prune_legal_actions():
    """安全側のマクロは合法手を1つも落とさない(順序を変えるだけ)。"""
    from tests.toy.toy_game import BACKEND

    for _name, state, _depth in SCENARIOS:
        canonical = candidates.canonical(state)
        assert set(macros.macro_actions(canonical)) == set(BACKEND.legal_actions(canonical))


# --------------------------------------------------------------------- utils


def _scenario_entry(name):
    for entry in SCENARIOS:
        if entry[0] == name:
            return entry
    raise KeyError(name)


def _scenario(name):
    entry = _scenario_entry(name)
    return entry[1], entry[2]
