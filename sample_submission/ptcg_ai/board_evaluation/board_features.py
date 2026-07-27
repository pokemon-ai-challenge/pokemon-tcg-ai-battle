"""クラスタ② 盤面評価／担当B

「盤面の状態 -> 数値」に変換する共通部品。サイド差・テンポ・主力アタッカーらしさなど、
main_turn_parts/priorities/*.py や handlers/damage_target_turn.py から使われる。
"""

from cg.api import EnergyType, Pokemon, State

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

    for attack_id in attacker_card.attacks:
        attack = card_cache.get_attack(attack_id)
        shortfall = energy_requirements.energy_shortfall(attack, attacker.energies)
        if sum(shortfall.values()) > _NEXT_TURN_ENERGY_ALLOWANCE:
            continue
        damage = attack_features.resolve_damage(
            attack, attacker, defender_card.weakness, defender_card.resistance, attacker_hand_size
        )
        if damage >= pokemon.hp:
            return True
    return False


def likely_ko_probability_next_turn(pokemon: Pokemon, state: State, your_index: int) -> float:
    """このポケモンが次の相手ターンで倒される確率を見積もる（0.0〜1.0）。

    is_likely_ko_next_turn() と同じワザ・ダメージ判定を使うが、「あと1エネルギーで足りる」
    ケース（shortfall合計が1）を無条件の True にはせず、そのエネルギーを相手が手札に
    持っている確率（hidden_information.OpponentHiddenState.marginals() の hand 確率、
    超幾何分布ベースの較正済みベイズ推定）で重み付けする。

    - shortfall == 0（今すぐ確実にKO可能）は確率 1.0（is_likely_ko_next_turn と同じ判定）。
    - 非公開情報が未準備（is_ready=False や marginals() が空、必要なエネルギー種別が
      card_cache 側でマッピングできない）ときは 1.0 にフォールバックする
      （= is_likely_ko_next_turn の True と同じ挙動。今日の挙動から後退しない）。
    - 複数のワザが該当する場合はワザごとの確率の最大値を採る。真の和集合確率
      （1 - Π(1-p_i)）よりは低く見積もる保守的な近似だが、過大評価はしない
      （opponent_hidden_state._shrink_hand_confidence と同じ「慎重側に倒す」方針）。
    - 特殊エネルギーでその種別を賄えるケースは今回モデル化していない
      （基本エネルギーのみを見るため、確率をやや低く見積もる既知のギャップ）。
    """
    opponent = state.players[1 - your_index]
    if not opponent.active or opponent.active[0] is None:
        return 0.0
    attacker = opponent.active[0]
    attacker_card = card_cache.get_card(attacker.id)
    defender_card = card_cache.get_card(pokemon.id)
    attacker_hand_size = opponent.handCount

    marginals: dict[int, dict[str, float]] | None = None
    best = 0.0
    for attack_id in attacker_card.attacks:
        attack = card_cache.get_attack(attack_id)
        shortfall = energy_requirements.energy_shortfall(attack, attacker.energies)
        total_shortfall = sum(shortfall.values())
        if total_shortfall > _NEXT_TURN_ENERGY_ALLOWANCE:
            continue
        damage = attack_features.resolve_damage(
            attack, attacker, defender_card.weakness, defender_card.resistance, attacker_hand_size
        )
        if damage < pokemon.hp:
            continue
        if total_shortfall == 0:
            attack_prob = 1.0
        else:
            if marginals is None:
                marginals = _opponent_hand_marginals()
            needed_type = next(iter(shortfall))
            attack_prob = _energy_in_hand_probability(needed_type, marginals)
        best = max(best, attack_prob)
    return best


def _opponent_hand_marginals() -> dict[int, dict[str, float]]:
    """match_context の現在の OpponentHiddenState から marginals() を取得する（未準備なら {}）。

    board_evaluation -> hidden_information の import は match_context.py 側が
    board_evaluation を参照しない（循環参照が無い）ことを確認済みだが、既存コードの
    「必要な箇所で遅延 import する」慣例に合わせて関数内 import にする。
    """
    from ptcg_ai.hidden_information import match_context

    opponent_state = match_context.get_opponent_state()
    if not opponent_state.is_ready:
        return {}
    return opponent_state.marginals()


def _energy_in_hand_probability(energy_type: EnergyType, marginals: dict[int, dict[str, float]]) -> float:
    """必要エネルギー種別を相手が手札に持っている確率。

    同タイプの再録違い（複数 card_id）や COLORLESS（全基本エネルギーで代替可）は
    1 - Π(1-p_i) で和集合として合成する。マッピングできない/データが無い場合は
    1.0（安全側 = 今日の挙動と同じ確実視）にフォールバックする。
    """
    if not marginals:
        return 1.0
    ids_by_type = card_cache.energy_card_ids_by_type()
    if energy_type == EnergyType.COLORLESS:
        candidate_ids = [card_id for ids in ids_by_type.values() for card_id in ids]
    else:
        candidate_ids = ids_by_type.get(energy_type, [])
    if not candidate_ids:
        return 1.0
    complement = 1.0
    for card_id in candidate_ids:
        p = marginals.get(card_id, {}).get("hand", 0.0)
        complement *= 1.0 - p
    return 1.0 - complement
