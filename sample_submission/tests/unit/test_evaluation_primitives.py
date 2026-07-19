"""クラスタ⑨ 検証（汎用シナリオ）／担当B

decision.evaluation 配下の計算関数（ダメージ計算、エネルギー充足判定など）を、
架空の CardData/Attack を使って検証する（デッキ非依存）。
"""

import pytest


def test_resolve_damage_applies_weakness():
    """attack_features.resolve_damage が弱点タイプに対してダメージを正しく増加させることを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")


def test_energy_shortfall_accounts_for_colorless():
    """energy_requirements.energy_shortfall が無色エネルギーの充当を正しく計算することを確認する。"""
    pytest.skip("TODO: 実装後に有効化する")
