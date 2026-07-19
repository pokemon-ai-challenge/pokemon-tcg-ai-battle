"""クラスタ⑨ 検証（汎用シナリオ）／担当B

decision.router.route が、router.py の対応表通りに SelectContext を正しい handler へ
振り分けるかを、架空のカードで検証する（デッキ非依存）。
"""

import pytest


def test_main_context_routes_to_main_turn_handler():
    """SelectContext.MAIN が handlers.main_turn.handle に渡ることを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")


def test_unknown_context_falls_back_to_fallback():
    """対応表に無い SelectContext が decision.fallback.safe_choice に渡ることを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")
