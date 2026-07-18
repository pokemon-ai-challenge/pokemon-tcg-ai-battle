"""クラスタ⑨ 検証（デッキ固有シナリオ）／担当A

decks/new_deck/*_profiles.py と deck_plan.py の整合性を検証する
（例: main_attacker_ids に含まれる全カードIDが pokemon_profiles にも存在するか）。
"""

import pytest


def test_all_deck_plan_ids_have_pokemon_profile():
    """deck_plan.PLAN が参照するカードIDが pokemon_profiles.PROFILES に存在することを確認する。"""
    pytest.skip("TODO: 新デッキ確定後にシナリオを書く")
