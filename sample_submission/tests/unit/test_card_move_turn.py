"""クラスタ⑨ 検証（汎用シナリオ）／担当B

decision.card_move 配下の choose() が、合成した Option リストに対して常に合法手
（minCount <= len(result) <= maxCount、重複なし）を返すことを、架空のカードで検証する。
"""

import pytest


def test_discard_choose_respects_protected_cards():
    """decks.active.deck_plan.protected_card_ids に含まれるカードを discard.choose が避けることを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")


def test_choose_result_is_always_legal():
    """各 choose() の戻り値が minCount/maxCount の範囲に収まり重複が無いことを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")
