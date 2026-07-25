"""攻撃の選択・ダメージ計算・確定リーサル判定。

ダメージ計算は既存の board_evaluation.attack_features / energy_requirements
（フーディンのハンドパワーの手札枚数依存ダメージの正規表現推定を含む）を再利用する。
"""

from __future__ import annotations

from cg.api import Attack, Observation, OptionType, Pokemon, SelectContext, SelectData, State

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import attack_features, energy_requirements
from ptcg_ai.custom_agent import board_context
from ptcg_ai.shared import card_cache


def usable_attacks(pokemon: Pokemon) -> list[Attack]:
    """今すぐ使えるだけのエネルギーが揃っているワザの一覧。"""
    card = card_cache.get_card(pokemon.id)
    attacks = [card_cache.get_attack(attack_id) for attack_id in card.attacks]
    return [attack for attack in attacks if energy_requirements.is_energy_sufficient(attack, pokemon.energies)]


def damage_against(attack: Attack, attacker: Pokemon, defender: Pokemon, attacker_index: int, state: State) -> int:
    defender_card = card_cache.get_card(defender.id)
    attacker_hand_size = state.players[attacker_index].handCount
    return attack_features.resolve_damage(
        attack, attacker, defender_card.weakness, defender_card.resistance, attacker_hand_size
    )


def best_attack_for(
    attacker: Pokemon, defender: Pokemon, attacker_index: int, state: State
) -> tuple[Attack, int] | None:
    """attacker が defender に与えられる最大ダメージのワザを返す（使えるワザが無ければNone）。"""
    best: tuple[Attack, int] | None = None
    for attack in usable_attacks(attacker):
        damage = damage_against(attack, attacker, defender, attacker_index, state)
        if best is None or damage > best[1]:
            best = (attack, damage)
    return best


def max_damage_against(defender: Pokemon, state: State) -> int | None:
    """自分のバトルポケモンが今すぐ defender に与えられる最大ダメージ。攻撃不可ならNone。"""
    own_active = board_context.own(state).active
    if not own_active or own_active[0] is None:
        return None
    result = best_attack_for(own_active[0], defender, state.yourIndex, state)
    return result[1] if result else None


def find_lethal_attack(state: State) -> Attack | None:
    """今すぐ相手のバトルポケモンを倒し切り、勝利が確定するワザがあれば返す。

    「勝利確定」は、このきぜつで自分の残りサイドが0枚になる場合とする
    （公式ルール同様、相手を倒すと"自分の"サイドから引く）。
    """
    own_active = board_context.own(state).active
    opp_active = board_context.opponent_active(state)
    if not own_active or own_active[0] is None or opp_active is None:
        return None

    my_prize_remaining = len(board_context.own(state).prize)
    prize_gain = board_context.prize_value(opp_active.id)
    if prize_gain < my_prize_remaining:
        return None

    result = best_attack_for(own_active[0], opp_active, state.yourIndex, state)
    if result is None:
        return None
    attack, damage = result
    if damage >= opp_active.hp:
        return attack
    return None


def choose_main_attack_option(select: SelectData, state: State) -> int | None:
    """MAINの選択肢の中から、相手のバトルポケモンへの最大ダメージを与えるATTACKオプションのインデックスを返す。"""
    own_active = board_context.own(state).active
    opp_active = board_context.opponent_active(state)
    if not own_active or own_active[0] is None or opp_active is None:
        return None
    attacker = own_active[0]

    best_index = None
    best_damage = -1
    for i, option in enumerate(select.option):
        if option.type != OptionType.ATTACK or option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        damage = damage_against(attack, attacker, opp_active, state.yourIndex, state)
        if damage > best_damage:
            best_damage = damage
            best_index = i
    return best_index


def handle_attack_select(obs: Observation) -> list[int]:
    """SelectContext.ATTACK / DISABLE_ATTACK（MAIN外のネストした技選択）を処理する。"""
    if obs.select.context == SelectContext.DISABLE_ATTACK:
        return _choose_most_dangerous(obs)
    return _choose_best(obs)


def _choose_best(obs: Observation) -> list[int]:
    state = obs.current
    index = choose_main_attack_option(obs.select, state)
    if index is None:
        return fallback.safe_choice(obs)
    return [index]


def _choose_most_dangerous(obs: Observation) -> list[int]:
    """相手のワザを1つ無効化する効果向け: 最もダメージの大きいワザを対象にする。"""
    best_index = None
    best_damage = -1
    for i, option in enumerate(obs.select.option):
        if option.attackId is None:
            continue
        attack = card_cache.get_attack(option.attackId)
        if attack.damage > best_damage:
            best_damage = attack.damage
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]
