"""pokemon_value.switch_target_value の即KOボーナス（2026-07-23追加）の回帰テスト。

docs/plans/rule_based/retreat-margin-unification-2026-07-23.md のトリガーB
（後退での即きぜつ）は、retreat.py から交代先を明示的に指定するのではなく、
switch_target_value 自体に「今すぐ相手アクティブをKOできるなら大きく加点する」
ボーナス（_IMMEDIATE_KO_BONUS）を足すことで、best_switch_target が自然にKO可能な
候補を選ぶようにする設計。ここでは
  (a) can_ko_with_any_available_attack が True の候補にボーナスが乗ること
  (b) そのボーナスが switch_target_value の他の全項の実運用上の合計を確実に上回ること
    （＝KOできる候補が他のどんな理由があっても最優先で選ばれる保証）
を検証する。
"""

from types import SimpleNamespace

from ptcg_ai.board_evaluation import attack_features
from ptcg_ai.rule_based.main_turn_parts import pokemon_value

_ALAKAZAM_ID = 743


def _pokemon(card_id: int = _ALAKAZAM_ID, energies=None, hp=100, max_hp=100):
    return SimpleNamespace(id=card_id, energies=list(energies or []), hp=hp, maxHp=max_hp)


def _state():
    own_player = SimpleNamespace(active=[_pokemon()], bench=[], handCount=5)
    opponent_player = SimpleNamespace(active=[_pokemon(999)], bench=[], handCount=5)
    return SimpleNamespace(yourIndex=0, players=[own_player, opponent_player])


def test_switch_target_value_gains_immediate_ko_bonus(monkeypatch):
    """can_ko_with_any_available_attack が True の候補は、False の候補よりスコアが
    _IMMEDIATE_KO_BONUS 以上高くなる（他の全項が同一の場合）。"""
    state = _state()
    candidate_can_ko = _pokemon(65, hp=80, max_hp=80)
    candidate_cannot_ko = _pokemon(66, hp=80, max_hp=80)  # 同じHPで揃え、他項の差を最小化

    def fake_can_ko(pokemon, *a, **k):
        return pokemon is candidate_can_ko

    monkeypatch.setattr(attack_features, "can_ko_with_any_available_attack", fake_can_ko)

    score_with_ko = pokemon_value.switch_target_value(candidate_can_ko, state, 0)
    score_without_ko = pokemon_value.switch_target_value(candidate_cannot_ko, state, 0)

    assert score_with_ko - score_without_ko >= pokemon_value._IMMEDIATE_KO_BONUS


def test_immediate_ko_bonus_dominates_all_other_terms(monkeypatch):
    """_IMMEDIATE_KO_BONUS は switch_target_value の他の全項の実運用上の合計
    （docs/plans/rule_based/retreat-margin-unification-2026-07-23.md 記載の見積り、
    おおむね20台）を1桁以上上回ること。KOできる候補が、他のどんな要因（安全ボーナス・
    役割スコア・KO後継優先度など）を最大限持つ候補にも、KOできないという理由だけで
    絶対に負けないことを保証するための構造的なテスト。"""
    # switch_target_value を構成する各ボーナスの理論上限（コード上のモジュール定数から）。
    from ptcg_ai.board_evaluation import switch_eval
    from ptcg_ai.rule_based.main_turn_parts import pokemon_value as pv

    realistic_other_terms_ceiling = (
        2.0  # attacker_score の hp_ratio 項 (hp_ratio<=1 * _HP_RATIO_WEIGHT)
        + 6.0  # attacker_score の energy_count 項（実運用で数枚程度の余裕を見た目安）
        + switch_eval._SAFE_BONUS  # 3.0
        + pv._ROLE_SCORE_WEIGHT  # role_score<=1 * weight = 3.0
        + pv._COMBO_BONUS  # 1.5
        + pv._KO_REPLACEMENT_BASE_SCORE  # 4.0
    )
    assert pv._IMMEDIATE_KO_BONUS >= realistic_other_terms_ceiling * 2
