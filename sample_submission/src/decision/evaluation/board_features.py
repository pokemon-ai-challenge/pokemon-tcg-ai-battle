from cg.api import AreaType, Attack, CardData, Observation, Pokemon

from src.decision.evaluation.attack_features import resolve_attacks
from src.decision.evaluation.energy_requirements import energy_gap_for_attack
from src.knowledge.deck_profiles import pokemon_active_role_bonus


def resolve_in_play_pokemon(
    active: list[Pokemon | None],
    bench: list[Pokemon],
    area: AreaType | None,
    index: int | None,
) -> Pokemon | None:
    """ACTIVE/BENCH の座標から対象ポケモンを取り出す。"""
    if area == AreaType.ACTIVE:
        if index is None or index < 0 or index >= len(active):
            return None
        return active[index]
    if area == AreaType.BENCH:
        if index is None or index < 0 or index >= len(bench):
            return None
        return bench[index]
    return None


def is_main_attacker(
    obs: Observation,
    target: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> bool:
    """現在の盤面で主力候補と言えるポケモンかをざっくり判定する。"""
    if obs.current is None:
        return False

    player = obs.current.players[obs.current.yourIndex]
    in_play = [pokemon for pokemon in player.active if pokemon is not None] + player.bench
    if not in_play:
        return False

    target_score = attacker_priority_score(target, card_data_by_id, attack_by_id)
    best_score = max(
        attacker_priority_score(pokemon, card_data_by_id, attack_by_id)
        for pokemon in in_play
    )
    return target_score == best_score


def attacker_priority_score(
    pokemon: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> int:
    """技の最大打点と耐久を元に、主力適性の粗いスコアを作る。"""
    card_data = card_data_by_id.get(pokemon.id)
    if card_data is None:
        return 0

    attacks = resolve_attacks(card_data, attack_by_id)
    best_damage = max((attack.damage for attack in attacks), default=0)
    stage_bonus = 8 if card_data.stage2 else 4 if card_data.stage1 else 0
    ex_bonus = 6 if card_data.ex else 0
    # 自デッキ固有の「主力アタッカーらしさ」を最後に少しだけ足す。
    role_bonus = pokemon_active_role_bonus(card_data.cardId)
    return best_damage + card_data.hp // 10 + stage_bonus + ex_bonus + role_bonus


def is_likely_knocked_out_next_turn(
    obs: Observation,
    target: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Attack],
) -> bool:
    """相手の次ターンに倒されやすい前衛かを保守的に見る。"""
    if obs.current is None:
        return False

    opponent_index = 1 - obs.current.yourIndex
    opponent_active = obs.current.players[opponent_index].active
    if not opponent_active or opponent_active[0] is None:
        return target.hp <= 30

    opponent_card_data = card_data_by_id.get(opponent_active[0].id)
    if opponent_card_data is None:
        return target.hp <= 30

    opponent_attacks = resolve_attacks(opponent_card_data, attack_by_id)
    threatening_damage = 0
    for attack in opponent_attacks:
        # 相手がほぼ今すぐ使えそうな技を優先して脅威打点を見る。
        if energy_gap_for_attack(attack, list(opponent_active[0].energies)) <= 1:
            threatening_damage = max(threatening_damage, attack.damage)

    if threatening_damage == 0:
        # エネ条件が読みにくい相手には、印字最大打点だけでも拾っておく。
        threatening_damage = max((attack.damage for attack in opponent_attacks), default=0)

    return threatening_damage >= target.hp or target.hp <= 30
