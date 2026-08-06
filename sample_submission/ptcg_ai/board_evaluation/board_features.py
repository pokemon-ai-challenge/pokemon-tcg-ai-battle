"""クラスタ② 盤面評価／担当B

「盤面の状態 -> 数値」に変換する共通部品。サイド差・テンポ・主力アタッカーらしさなど、
main_turn_parts/priorities/*.py や handlers/damage_target_turn.py から使われる。
"""

from cg.api import Pokemon, State

from ptcg_ai.board_evaluation import attack_features, energy_requirements
from ptcg_ai.shared import card_cache

_HP_RATIO_WEIGHT = 2.0
_ENERGY_UNIT_SCORE = 1.0
# 相手が今エネルギー1枚足りないだけの技は、次の相手ターンには使えるようになるとみなす。
_NEXT_TURN_ENERGY_ALLOWANCE = 1


def attacker_score(pokemon: Pokemon) -> float:
    """「主力アタッカーらしさ」をスコア化する（HP、付いているエネルギー量などから算出）。"""
    hp_ratio = pokemon.hp / pokemon.maxHp if pokemon.maxHp else 0.0
    return hp_ratio * _HP_RATIO_WEIGHT + len(pokemon.energies) * _ENERGY_UNIT_SCORE


def prize_diff(state: State, your_index: int) -> int:
    """自分と相手の残りサイド枚数の差を返す（正なら自分が有利）。"""
    your_prize = len(state.players[your_index].prize)
    opponent_prize = len(state.players[1 - your_index].prize)
    return your_prize - opponent_prize


def is_likely_ko_next_turn(pokemon: Pokemon, state: State, your_index: int) -> bool:
    """このポケモンが次の相手ターンで倒されやすいかを判定する。

    相手のバトルポケモンが持つワザのうち、今エネルギーが十分（または次ターンに1枚
    付ければ十分になる程度の不足）なものを対象に、現在のHPが削り切られるかを見積もる
    （相手の手札にある道具/サポートによる追加のダメージ増減は考慮しない簡易版）。
    """
    opponent = state.players[1 - your_index]
    if not opponent.active or opponent.active[0] is None:
        return False
    attacker = opponent.active[0]
    attacker_card = card_cache.get_card(attacker.id)
    defender_card = card_cache.get_card(pokemon.id)
    # 可変ダメージ技（手札枚数依存など）の推定用。相手の手札枚数は非公開だが件数
    # （handCount）だけは見えるため、それを使う。次ターンのドロー分までは考慮しない簡易版。
    attacker_hand_size = opponent.handCount

    # pokemon は自分側(your_index)の active/bench どちらも渡され得る唯一の呼び出し口なので、
    # ベンチ庇い型防壁（74 Rabsca / 343 Shaymin）を判定するための defender_side_pokemon /
    # defender_is_benched をここで計算する。同一性判定は Pokemon.serial（試合中に一意な値）で
    # 行う。id（cardId）は同名カードで重複するため使えない。
    your_side = state.players[your_index]
    defender_side_pokemon: list[Pokemon | None] = list(your_side.active) + list(your_side.bench)
    defender_is_benched = any(
        bench_pokemon is not None and bench_pokemon.serial == pokemon.serial
        for bench_pokemon in your_side.bench
    )

    for attack_id in attacker_card.attacks:
        attack = card_cache.get_attack(attack_id)
        shortfall = energy_requirements.energy_shortfall(attack, attacker.energies)
        if sum(shortfall.values()) > _NEXT_TURN_ENERGY_ALLOWANCE:
            continue
        damage_is_effect = attack_features.damage_is_effect_based(attack)
        damage = attack_features.resolve_damage(
            attack, attacker, defender_card.weakness, defender_card.resistance, attacker_hand_size,
            defender=pokemon, defender_side_pokemon=defender_side_pokemon,
            defender_is_benched=defender_is_benched, damage_is_effect=damage_is_effect,
        )
        if damage >= pokemon.hp:
            return True
    return False
