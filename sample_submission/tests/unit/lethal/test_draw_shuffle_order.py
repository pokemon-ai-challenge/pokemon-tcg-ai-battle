"""DRAW / SHUFFLE の境界判定(Step 1-7a の誤拒否を再発させないための回帰)。

Step 1-7a のバグ: 1 ステップ内の「引いてからシャッフル」を、ログ順を見ずに
一律で無効としていた(167 outcome の構築が全滅した)。

判定は ``draw_is_supply_ordered(sequence, order_lost)`` に切り出してある。
ここでは指示のケース A〜D をそのまま固定する。
"""

from ptcg_ai.search.lethal.cg_backend import draw_is_supply_ordered
from ptcg_ai.search.lethal.engine import COIN, DRAW, PRIZE, SHUFFLE


def test_case_a_draw_then_shuffle_is_supply_ordered():
    """ケース A: DRAW × 3 → SHUFFLE → (ターン終了)。3 枚は供給順どおり。"""
    assert draw_is_supply_ordered((DRAW, DRAW, DRAW, SHUFFLE), order_lost=False) is True


def test_case_b_shuffle_then_draw_is_not_usable():
    """ケース B: SHUFFLE → DRAW。未知の順序から引いているので使えない。"""
    assert draw_is_supply_ordered((SHUFFLE, DRAW), order_lost=False) is False


def test_case_c_draws_after_a_previous_shuffle_are_not_usable():
    """ケース C: DRAW × 3 → SHUFFLE の**後**のステップのドローは使えない。

    最初の 3 枚は exact outcome として扱えるが、シャッフル後の 4 枚目は
    実際の隠れた山札順に依存するので確定証明に使ってはいけない。
    """
    # 1 ステップ目(引いてからシャッフル)は有効
    assert draw_is_supply_ordered((DRAW, DRAW, DRAW, SHUFFLE), order_lost=False) is True
    # 2 ステップ目以降は order_lost=True で入る
    assert draw_is_supply_ordered((DRAW,), order_lost=True) is False


def test_case_d_mixed_draw_and_shuffle_in_one_step():
    """ケース D: 同一ステップに draw と shuffle が混在する場合。"""
    # シャッフル後にもう一度引いていたら無効
    assert draw_is_supply_ordered((DRAW, SHUFFLE, DRAW), order_lost=False) is False
    # 全てのドローがシャッフルより前なら有効
    assert draw_is_supply_ordered((DRAW, DRAW, SHUFFLE, PRIZE), order_lost=False) is True
    # シャッフルが 2 回あっても、ドローが全部より前なら有効
    assert draw_is_supply_ordered((DRAW, SHUFFLE, SHUFFLE), order_lost=False) is True
    # ドローが 2 回目のシャッフルより後なら無効
    assert draw_is_supply_ordered((SHUFFLE, DRAW, SHUFFLE), order_lost=False) is False


def test_no_shuffle_at_all_is_supply_ordered():
    assert draw_is_supply_ordered((DRAW, DRAW), order_lost=False) is True
    assert draw_is_supply_ordered((COIN, DRAW), order_lost=False) is True


def test_order_lost_dominates_everything():
    """経路上で一度シャッフルが起きたら、以降はログ順に関係なく無効。"""
    for sequence in ((DRAW,), (DRAW, SHUFFLE), (DRAW, DRAW, DRAW)):
        assert draw_is_supply_ordered(sequence, order_lost=True) is False
