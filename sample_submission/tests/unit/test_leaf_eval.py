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
