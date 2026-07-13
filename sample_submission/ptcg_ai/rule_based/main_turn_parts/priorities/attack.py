"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: attack（ワザを使う）

decision.evaluation.attack_features / energy_requirements を使い、使用可能なワザの中から
最も価値の高いものを提案する。相手をきぜつさせられる（can_ko）ワザは最優先とする。
きぜつが取れるかに関わらず、AttackProfile（担当A）が示す追加効果
（ベンチ狙撃・状態異常・ドロー・次ターン攻撃封じ）があるワザには加点し、
単純なダメージ量だけでは測れない価値も評価に含める。
"""

from cg.api import Observation, OptionType

from ptcg_ai.board_evaluation import attack_features, energy_requirements
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import card_cache, profile_registry

# きぜつを取れるワザは、ダメージ量に関わらず最優先にする。
_KO_BONUS = 1000.0

# AttackProfile（担当A）の追加効果分類に対する加点。ダメージと同じスケール（点）で加算する。
_BENCH_SNIPE_BONUS = 8.0
_SPECIAL_CONDITION_BONUS = 6.0
_DRAW_BONUS = 4.0
_DISABLE_NEXT_ATTACK_BONUS = 10.0


def propose(obs: Observation) -> ActionProposal | None:
    """ワザ使用の行動を1つ提案する。使用可能なワザが無ければ None。"""
    state = obs.current
    your_index = state.yourIndex
    player = state.players[your_index]
    opponent = state.players[1 - your_index]
    if not player.active or player.active[0] is None:
        return None
    if not opponent.active or opponent.active[0] is None:
        return None

    attacker = player.active[0]
    defender = opponent.active[0]
    defender_card = card_cache.get_card(defender.id)

    usable: list[tuple[int, float]] = []  # (option_index, value)
    ko_candidates: list[tuple[int, float]] = []  # (option_index, value)
    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ATTACK or option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        if not energy_requirements.is_energy_sufficient(attack, attacker.energies):
            continue
        value = float(attack.damage) + _effect_bonus(option.attackId)
        usable.append((i, value))
        if attack_features.can_ko(attack, attacker, defender, defender_card.weakness, defender_card.resistance):
            ko_candidates.append((i, value))

    if not usable:
        return None

    if ko_candidates:
        best_index, best_value = max(ko_candidates, key=lambda item: item[1])
        return ActionProposal(category="attack", select=[best_index], score=_KO_BONUS + best_value, reason="lethal attack")

    best_index, best_value = max(usable, key=lambda item: item[1])
    return ActionProposal(category="attack", select=[best_index], score=best_value, reason="highest value attack")


def _effect_bonus(attack_id: int) -> float:
    """AttackProfile（担当A）の追加効果分類に応じた加点を返す（プロファイルが無ければ0）。"""
    profile = profile_registry.get_attack_profile(attack_id)
    if profile is None:
        return 0.0

    bonus = 0.0
    if profile.bench_snipe:
        bonus += _BENCH_SNIPE_BONUS
    if profile.inflicts_special_condition:
        bonus += _SPECIAL_CONDITION_BONUS
    if profile.draws_cards:
        bonus += _DRAW_BONUS
    if profile.disables_next_attack:
        bonus += _DISABLE_NEXT_ATTACK_BONUS
    return bonus
