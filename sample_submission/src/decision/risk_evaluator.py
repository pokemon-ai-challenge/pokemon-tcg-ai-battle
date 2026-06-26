"""KO リスクとエネルギー上限を評価するユーティリティ。"""
from cg.api import State, Pokemon
from src.knowledge.card_database import get_card_db, get_attack_db


def _max_opponent_damage(state: State) -> int:
    """相手アクティブが出せる最大ダメージを返す（攻撃不可なら 0）。"""
    card_db = get_card_db()
    attack_db = get_attack_db()
    opp_idx = 1 - state.yourIndex
    opp_active = state.players[opp_idx].active
    if not opp_active or not opp_active[0]:
        return 0
    opp = opp_active[0]
    opp_card = card_db.get(opp.id)
    if not opp_card or not opp_card.attacks:
        return 0
    max_dmg = 0
    for atk_id in opp_card.attacks:
        atk = attack_db.get(atk_id)
        if atk:
            max_dmg = max(max_dmg, atk.damage)
    return max_dmg


def _weakness_multiplier(pokemon: Pokemon, state: State) -> float:
    """弱点倍率を返す（簡易実装: 弱点があれば 2.0、なければ 1.0）。"""
    card_db = get_card_db()
    opp_idx = 1 - state.yourIndex
    opp_active = state.players[opp_idx].active
    if not opp_active or not opp_active[0]:
        return 1.0
    opp = opp_active[0]
    opp_card = card_db.get(opp.id)
    my_card = card_db.get(pokemon.id)
    if not opp_card or not my_card:
        return 1.0
    # CardData.weakness, energyType はどちらも int（エネルギータイプ ID）
    opp_energy = getattr(opp_card, "energyType", None)
    my_weakness = getattr(my_card, "weakness", None)
    if opp_energy is not None and my_weakness is not None and opp_energy == my_weakness:
        return 2.0
    return 1.0


def will_be_ko_next_turn(state: State, pokemon: Pokemon) -> bool:
    """相手が次ターンにこのポケモンをKOできるか推定する。

    判定: 相手アクティブの最大ダメージ × 弱点倍率 >= pokemon.hp
    """
    max_dmg = _max_opponent_damage(state)
    if max_dmg <= 0:
        return False
    multiplier = _weakness_multiplier(pokemon, state)
    effective_dmg = int(max_dmg * multiplier)
    return effective_dmg >= pokemon.hp


def energy_needed_for_attack(card_id: int) -> int:
    """カードが攻撃するのに必要な最小エネルギー枚数を返す。"""
    card_db = get_card_db()
    attack_db = get_attack_db()
    card = card_db.get(card_id)
    if not card or not card.attacks:
        return 2
    min_cost = min(
        len(attack_db[atk_id].energies) if atk_id in attack_db else 2
        for atk_id in card.attacks
    )
    return max(1, min_cost)


def energy_cap(card_id: int) -> int:
    """有効なエネルギー上限 = 攻撃コスト + 逃げコストを返す。

    この枚数以上付けても戦術的に無意味（ポケモンのリソース効率を考慮）。
    """
    card_db = get_card_db()
    card = card_db.get(card_id)
    if not card:
        return 3
    attack_cost = energy_needed_for_attack(card_id)
    retreat_cost = card.retreatCost or 0
    return attack_cost + retreat_cost
