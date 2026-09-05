"""ptcg_ai.search.leaf_eval のユニットテスト。

末端評価は先読みの葉で盤面を ``me`` 視点の勝ち見込み(0..1)へ写す。cg の完全な State を
組むのは重いので、``HandcraftedEvaluator`` が実際に参照するフィールド(result / players の
active・bench・prize、Pokemon の hp・maxHp・energies)だけを持つ軽量なダミーで検証する。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))


@pytest.fixture(scope="module")
def leaf_eval():
    try:
        from ptcg_ai.search import leaf_eval as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"leaf_eval / cg engine unavailable: {exc}")
    return mod


def _pokemon(hp=100, max_hp=100, energies=0):
    return SimpleNamespace(hp=hp, maxHp=max_hp, energies=[0] * energies)


def _player(prize=3, active=True, bench=0, energies=0):
    return SimpleNamespace(
        prize=[None] * prize,
        active=[_pokemon(energies=energies)] if active else [None],
        bench=[_pokemon() for _ in range(bench)],
    )


def _state(me=0, result=-1, my_prize=3, opp_prize=3, my_energy=0, opp_energy=0,
           my_bench=0, opp_bench=0, your_index=0):
    players = [None, None]
    players[me] = _player(prize=my_prize, energies=my_energy, bench=my_bench)
    players[1 - me] = _player(prize=opp_prize, energies=opp_energy, bench=opp_bench)
    return SimpleNamespace(result=result, players=players, yourIndex=your_index)


def test_decided_states_are_hard_scored(leaf_eval):
    ev = leaf_eval.HandcraftedEvaluator()
    assert ev.evaluate(_state(me=0, result=0), me=0) == 1.0   # 自分の勝ち
    assert ev.evaluate(_state(me=0, result=1), me=0) == 0.0   # 相手の勝ち


def test_score_is_in_unit_interval(leaf_eval):
    ev = leaf_eval.HandcraftedEvaluator()
    for opp_prize in range(0, 7):
        s = ev.evaluate(_state(my_prize=3, opp_prize=opp_prize), me=0)
        assert 0.0 <= s <= 1.0


def test_prize_advantage_is_monotonic(leaf_eval):
    """自分の残サイドが少ない(=勝ちに近い)ほどスコアが高い。"""
    ev = leaf_eval.HandcraftedEvaluator()
    ahead = ev.evaluate(_state(my_prize=1, opp_prize=5), me=0)
    even = ev.evaluate(_state(my_prize=3, opp_prize=3), me=0)
    behind = ev.evaluate(_state(my_prize=5, opp_prize=1), me=0)
    assert ahead > even > behind
    assert even == pytest.approx(0.5, abs=1e-9)  # 完全対称なら中立


def test_perspective_flips_with_me(leaf_eval):
    """同じ盤面を相手視点で見るとスコアは 1 - x になる(サイド以外対称なとき)。"""
    ev = leaf_eval.HandcraftedEvaluator()
    st = _state(me=0, my_prize=2, opp_prize=5)
    a = ev.evaluate(st, me=0)
    b = ev.evaluate(st, me=1)
    assert a + b == pytest.approx(1.0, abs=1e-9)


def test_value_evaluator_aligns_perspective(leaf_eval):
    """ValueModelEvaluator は state.yourIndex 視点の勝率を me 視点へ整列する。"""
    class _FakeModel:
        def predict_win_prob_from_state(self, state):
            return 0.7  # state.yourIndex 視点

    ev = leaf_eval.ValueModelEvaluator(model=_FakeModel())
    # yourIndex==me ならそのまま。
    assert ev.evaluate(_state(your_index=0), me=0) == pytest.approx(0.7)
    # yourIndex!=me なら反転。
    assert ev.evaluate(_state(your_index=1), me=0) == pytest.approx(0.3)


def test_build_evaluator_selects_kind(leaf_eval):
    assert isinstance(leaf_eval.build_evaluator(None), leaf_eval.HandcraftedEvaluator)
    assert isinstance(leaf_eval.build_evaluator({"kind": "handcrafted"}), leaf_eval.HandcraftedEvaluator)
    assert isinstance(leaf_eval.build_evaluator({"kind": "value"}), leaf_eval.ValueModelEvaluator)


def _state_hand(me=0, my_hand=5, opp_hand=5):
    mine = SimpleNamespace(prize=[None] * 3, active=[_pokemon()], bench=[], handCount=my_hand)
    opp = SimpleNamespace(prize=[None] * 3, active=[_pokemon()], bench=[], handCount=opp_hand)
    players = [None, None]
    players[me] = mine
    players[1 - me] = opp
    return SimpleNamespace(result=-1, players=players, yourIndex=me)


def test_card_advantage_coeff_zero_is_noop(leaf_eval):
    """card_advantage_coeff=0(既定)なら手札枚数はスコアに影響しない(従来挙動)。"""
    ev = leaf_eval.HandcraftedEvaluator(card_advantage_coeff=0.0)
    assert ev.evaluate(_state_hand(my_hand=12, opp_hand=3), me=0) == ev.evaluate(
        _state_hand(my_hand=3, opp_hand=3), me=0
    )


def test_card_advantage_rewards_bigger_hand(leaf_eval):
    """coeff>0 なら自分の手札が多い(=引いた)局面ほど高評価。ドローエンジンを後押しする。"""
    ev = leaf_eval.HandcraftedEvaluator(card_advantage_coeff=0.1)
    drew = ev.evaluate(_state_hand(my_hand=12, opp_hand=5), me=0)   # +4/+3ドロー後を想定
    base = ev.evaluate(_state_hand(my_hand=5, opp_hand=5), me=0)
    assert drew > base
    # 相手視点では符号が反転する。
    assert ev.evaluate(_state_hand(me=0, my_hand=12, opp_hand=5), me=1) < 0.5


def test_build_evaluator_passes_card_advantage_coeff(leaf_eval):
    assert leaf_eval.build_evaluator({"kind": "handcrafted"}).card_advantage_coeff == 0.0
    assert leaf_eval.build_evaluator(
        {"kind": "handcrafted", "card_advantage_coeff": 0.07}
    ).card_advantage_coeff == 0.07


# --- BlendedEvaluator(climb v1.5: 学習Valueを葉へ入れる混合口) ---


class _FakeValue:
    """ValueModel を差し替える固定値スタブ(重みJSON非依存でブレンド則だけ検証する)。"""

    def __init__(self, value):
        self.value = value

    def evaluate(self, state, me):
        return self.value


def test_blend_alpha_zero_matches_handcrafted(leaf_eval):
    """α=0.0 は handcrafted と数値的に一致する(既定挙動を壊さないことの保証)。"""
    hand = leaf_eval.HandcraftedEvaluator()
    ev = leaf_eval.BlendedEvaluator(alpha=0.0, handcrafted=hand, value=_FakeValue(0.9))
    s = _state(my_prize=2, opp_prize=5)
    assert ev.evaluate(s, me=0) == hand.evaluate(s, me=0)


def test_blend_alpha_one_matches_value(leaf_eval):
    """α=1.0 は learned value 単体と一致する。"""
    ev = leaf_eval.BlendedEvaluator(alpha=1.0, value=_FakeValue(0.83))
    assert ev.evaluate(_state(), me=0) == pytest.approx(0.83)


def test_blend_is_convex_combination(leaf_eval):
    """0<α<1 は両者の凸結合。handcrafted と value の間に必ず入る。"""
    hand = leaf_eval.HandcraftedEvaluator()
    s = _state(my_prize=2, opp_prize=5)
    h = hand.evaluate(s, me=0)
    ev = leaf_eval.BlendedEvaluator(alpha=0.5, handcrafted=hand, value=_FakeValue(0.0))
    assert ev.evaluate(s, me=0) == pytest.approx(0.5 * 0.0 + 0.5 * h)
    assert min(h, 0.0) <= ev.evaluate(s, me=0) <= max(h, 0.0)


def test_blend_keeps_decided_states_hard(leaf_eval):
    """決着局面はブレンドしても確定値(1.0/0.0)。学習Valueの誤差が勝敗を濁らせない。"""
    ev = leaf_eval.BlendedEvaluator(alpha=0.5, value=_FakeValue(0.5))
    assert ev.evaluate(_state(me=0, result=0), me=0) == 1.0
    assert ev.evaluate(_state(me=0, result=1), me=0) == 0.0


def test_blend_alpha_is_clamped(leaf_eval):
    assert leaf_eval.BlendedEvaluator(alpha=-3.0).alpha == 0.0
    assert leaf_eval.BlendedEvaluator(alpha=9.0).alpha == 1.0


def test_build_evaluator_blend_kind(leaf_eval):
    ev = leaf_eval.build_evaluator({"kind": "blend", "alpha": 0.8})
    assert isinstance(ev, leaf_eval.BlendedEvaluator)
    assert ev.alpha == 0.8
    # 既定 kind は従来どおり handcrafted のまま(本番不変)。
    assert isinstance(leaf_eval.build_evaluator({}), leaf_eval.HandcraftedEvaluator)


def test_shared_value_model_is_reused(leaf_eval):
    """`build_evaluator` は select 毎に呼ばれるため、ValueModel は共有インスタンスであること
    (毎回重みJSONを読み直すと探索予算を食う)。"""
    a = leaf_eval._get_shared_value_model()
    b = leaf_eval._get_shared_value_model()
    assert a is b


def test_value_model_ready_reports_true_when_weights_present(leaf_eval):
    """本リポジトリには value_weights.json が同梱されているので ready であるべき。

    ここが False のまま kind="value"/"blend" を走らせると、葉評価が定数 0.5 になっているのに
    探索は動くという気付きにくい失敗になる(A/B の前に必ず確認する)。
    """
    assert leaf_eval.value_model_ready() is True


class _FakeValueModel:
    """`predict_win_prob_from_state` だけを持つ ValueModel スタブ。"""

    def __init__(self, prob):
        self.prob = prob
        self.calls = 0

    def predict_win_prob_from_state(self, state):
        self.calls += 1
        return self.prob


def test_value_evaluator_flips_perspective_for_opponent(leaf_eval):
    """ValueModel は state.yourIndex 視点の勝率を返す。``me`` がそれと違うなら 1-p にする。

    ここを取り違えると、探索の葉で **相手の勝率を自分の勝率として最大化** することになり、
    勝率だけ見ても気付けない(実際に診断スクリプトで同じ取り違えを踏んだ)。
    """
    ev = leaf_eval.ValueModelEvaluator(model=_FakeValueModel(0.8))
    s = _state(your_index=0)
    assert ev.evaluate(s, me=0) == pytest.approx(0.8)     # 視点一致 → そのまま
    assert ev.evaluate(s, me=1) == pytest.approx(0.2)     # 視点反転 → 1-p


def test_blend_uses_flipped_value_for_opponent(leaf_eval):
    """ブレンドも視点反転を引き継ぐ(α=1 は value 単体と一致)。"""
    ev = leaf_eval.BlendedEvaluator(
        alpha=1.0, value=leaf_eval.ValueModelEvaluator(model=_FakeValueModel(0.9)))
    s = _state(your_index=0)
    assert ev.evaluate(s, me=1) == pytest.approx(0.1)


def test_blend_alpha_one_equals_value_kind(leaf_eval):
    """α=1.0 の blend は kind="value" と数値的に一致する。

    α スイープでは全 arm を同じ実装経路(BlendedEvaluator)に揃えたいが、
    それが従来の kind="value" と等価であることを保証しておく。
    """
    fake = _FakeValueModel(0.73)
    v = leaf_eval.ValueModelEvaluator(model=fake)
    b = leaf_eval.BlendedEvaluator(alpha=1.0, value=leaf_eval.ValueModelEvaluator(model=fake))
    s = _state(my_prize=2, opp_prize=5, your_index=0)
    assert b.evaluate(s, me=0) == pytest.approx(v.evaluate(s, me=0))
    assert b.evaluate(s, me=1) == pytest.approx(v.evaluate(s, me=1))
