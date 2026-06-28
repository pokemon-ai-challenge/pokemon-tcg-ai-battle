from dataclasses import dataclass
from typing import Callable, Protocol

from cg.api import AreaType, Card, CardData, CardType, EnergyType, Observation, Option, Pokemon

from src.decision.evaluation.attack_features import build_attack_lookup, choose_main_attack, resolve_attacks
from src.decision.evaluation.board_features import attacker_priority_score
from src.decision.evaluation.energy_requirements import (
    best_payable_attack_damage,
    energy_gap_for_attack,
    has_payable_attack,
)
from src.knowledge.deck_profiles import (
    get_attack_effect_profile,
    get_item_profile,
    get_pokemon_profile,
    get_supporter_profile,
    get_tool_profile,
)


class CardMoveOptionViewLike(Protocol):
    option_index: int
    owner_is_self: bool | None
    area: AreaType | None
    card_id: int | None
    is_pokemon: bool | None
    is_energy: bool | None


@dataclass(frozen=True)
class ToHandOptionEvaluation:
    option_index: int
    score: float


def choose_to_hand_action(
    obs: Observation,
    *,
    analyze_option: Callable[[Observation, int], CardMoveOptionViewLike | None],
    card_data_lookup: Callable[[], dict[int, CardData]],
    resolve_option_entry: Callable[[Observation, int], object | None],
) -> list[int] | None:
    if obs.select is None or obs.current is None:
        return None

    card_data_by_id = card_data_lookup()
    attack_by_id = build_attack_lookup()
    views = [
        analyze_option(obs, option_index)
        for option_index in range(len(obs.select.option))
    ]
    views = [view for view in views if view is not None]
    if not views:
        return None

    selected: list[int] = []
    selected_card_ids: dict[int, int] = {}

    while len(selected) < obs.select.maxCount:
        # 複数枚選ぶときも、まずは 1 枚ずつ価値を見て貪欲に積む。
        remaining = [
            view
            for view in views
            if view.option_index not in selected
        ]
        if not remaining:
            break

        evaluations = [
            evaluate_to_hand_option(
                obs,
                view,
                selected_card_ids,
                card_data_by_id,
                attack_by_id,
                resolve_option_entry,
            )
            for view in remaining
        ]
        best = max(
            evaluations,
            key=lambda evaluation: (evaluation.score, -evaluation.option_index),
        )
        if len(selected) >= obs.select.minCount and best.score <= 0:
            break

        selected.append(best.option_index)
        view = next(view for view in remaining if view.option_index == best.option_index)
        if view.card_id is not None:
            selected_card_ids[view.card_id] = selected_card_ids.get(view.card_id, 0) + 1

    if not selected and obs.select.minCount == 0:
        return []

    if len(selected) < obs.select.minCount:
        remaining = [
            view
            for view in views
            if view.option_index not in selected
        ]
        ranked = sorted(
            (
                evaluate_to_hand_option(
                    obs,
                    view,
                    selected_card_ids,
                    card_data_by_id,
                    attack_by_id,
                    resolve_option_entry,
                )
                for view in remaining
            ),
            key=lambda evaluation: (evaluation.score, -evaluation.option_index),
            reverse=True,
        )
        for evaluation in ranked:
            if len(selected) >= obs.select.minCount:
                break
            selected.append(evaluation.option_index)

    return selected if selected else None


def evaluate_to_hand_option(
    obs: Observation,
    view: CardMoveOptionViewLike,
    selected_card_ids: dict[int, int],
    card_data_by_id: dict[int, CardData],
    attack_by_id,
    resolve_option_entry: Callable[[Observation, int], object | None],
) -> ToHandOptionEvaluation:
    if obs.current is None or view.card_id is None:
        return ToHandOptionEvaluation(option_index=view.option_index, score=-1000)

    option = obs.select.option[view.option_index]
    entry = resolve_option_entry(obs, view.option_index)
    card_data = card_data_by_id.get(view.card_id)
    if card_data is None:
        return ToHandOptionEvaluation(option_index=view.option_index, score=-1000)

    owner_is_self = infer_owner_is_self(obs, view, option)
    if owner_is_self is True:
        # 自分のカードなら「取りにいく価値」を見る。
        score = score_self_to_hand_option(
            obs,
            view,
            card_data,
            entry,
            card_data_by_id,
            attack_by_id,
        )
    elif owner_is_self is False:
        # 相手のカードなら「相手の次ターンをどれだけ弱くできるか」で見る。
        score = score_opponent_to_hand_option(
            obs,
            view,
            card_data,
            entry,
            card_data_by_id,
            attack_by_id,
        )
    else:
        score = 0.0

    duplicate_count = selected_card_ids.get(view.card_id, 0)
    if duplicate_count > 0 and card_data.cardType != CardType.BASIC_ENERGY:
        score -= 6 * duplicate_count

    return ToHandOptionEvaluation(option_index=view.option_index, score=score)


def infer_owner_is_self(
    obs: Observation,
    view: CardMoveOptionViewLike,
    option: Option,
) -> bool | None:
    if view.owner_is_self is not None:
        return view.owner_is_self
    if obs.current is None:
        return None
    if option.playerIndex is not None:
        return option.playerIndex == obs.current.yourIndex
    if view.area in {
        AreaType.HAND,
        AreaType.DECK,
        AreaType.DISCARD,
        AreaType.PRIZE,
        AreaType.LOOKING,
    }:
        return True
    return None


def score_self_to_hand_option(
    obs: Observation,
    view: CardMoveOptionViewLike,
    card_data: CardData,
    entry: object | None,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    if view.area in {AreaType.DECK, AreaType.DISCARD, AreaType.PRIZE, AreaType.LOOKING, AreaType.HAND}:
        # 山札・トラッシュなどから手札に加えるケース。
        return score_self_gain_card(
            obs,
            card_data,
            from_discard=view.area == AreaType.DISCARD,
            card_data_by_id=card_data_by_id,
            attack_by_id=attack_by_id,
        )

    if view.area in {AreaType.ACTIVE, AreaType.BENCH} and isinstance(entry, Pokemon):
        # 場のポケモンを手札へ戻すケース。
        return score_self_pokemon_bounce(
            obs,
            entry,
            card_data,
            card_data_by_id,
            attack_by_id,
        )

    return 0.0


def score_self_gain_card(
    obs: Observation,
    card_data: CardData,
    *,
    from_discard: bool,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    if card_data.cardType == CardType.POKEMON:
        return score_self_pokemon_gain(
            obs,
            card_data,
            from_discard=from_discard,
            card_data_by_id=card_data_by_id,
            attack_by_id=attack_by_id,
        )
    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        return score_self_energy_gain(
            obs,
            card_data,
            from_discard=from_discard,
            card_data_by_id=card_data_by_id,
            attack_by_id=attack_by_id,
        )
    return score_self_trainer_gain(
        obs,
        card_data,
        from_discard=from_discard,
        card_data_by_id=card_data_by_id,
        attack_by_id=attack_by_id,
    )


def score_self_pokemon_gain(
    obs: Observation,
    card_data: CardData,
    *,
    from_discard: bool,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    player = obs.current.players[obs.current.yourIndex]
    profile = get_pokemon_profile(card_data.cardId)
    active_role_bonus = profile.active_role_bonus if profile is not None else 0
    bench_setup_bonus = profile.bench_setup_bonus if profile is not None else 0

    score = 2.0 + bench_setup_bonus * 1.8 + active_role_bonus * 0.8
    if from_discard:
        score += 4

    if card_data.evolvesFrom:
        if has_named_pokemon_in_play(player, card_data.evolvesFrom, card_data_by_id):
            # 進化元が場にいるなら、進化先の回収はかなり価値が高い。
            score += 30
            if from_discard:
                score += 14
        elif has_named_card(player.hand, card_data.evolvesFrom, card_data_by_id):
            score += 14
        else:
            score -= 8
    elif card_data.basic:
        bench_open_slots = max(player.benchMax - len(player.bench), 0)
        if bench_open_slots > 0:
            score += 8
        if count_ready_attackers(player, card_data_by_id, attack_by_id) == 0 and active_role_bonus >= 12:
            score += 16

    if is_player_under_pressure(player):
        score += 8

    return score


def score_self_energy_gain(
    obs: Observation,
    card_data: CardData,
    *,
    from_discard: bool,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    player = obs.current.players[obs.current.yourIndex]
    in_play = in_play_pokemon(player)
    if not in_play:
        return -6

    best_score = max(
        score_energy_gain_to_target(
            obs,
            pokemon,
            card_data,
            card_data_by_id,
            attack_by_id,
        )
        for pokemon in in_play
    )

    if from_discard and benched_energy_acceleration_is_online(player, card_data_by_id):
        # すぐ場へ貼り直せるデッキでは、単純なエネ回収を少し下げる。
        best_score -= 14

    if player.handCount == 0:
        best_score += 3

    return best_score


def score_energy_gain_to_target(
    obs: Observation,
    target: Pokemon,
    energy_card: CardData,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    target_card_data = card_data_by_id.get(target.id)
    if target_card_data is None:
        return -10

    attacks = resolve_attacks(target_card_data, attack_by_id)
    if not attacks:
        return -8

    current_energies = list(target.energies)
    next_energies = current_energies + [energy_card.energyType]
    main_attack = choose_main_attack(attacks)

    before_main_gap = energy_gap_for_attack(main_attack, current_energies)
    after_main_gap = energy_gap_for_attack(main_attack, next_energies)
    best_damage_before = best_payable_attack_damage(attacks, current_energies)
    best_damage_after = best_payable_attack_damage(attacks, next_energies)
    can_attack_before = has_payable_attack(attacks, current_energies)
    can_attack_after = has_payable_attack(attacks, next_energies)

    score = 0.0
    if target == active_pokemon(obs.current.players[obs.current.yourIndex]) and not can_attack_before and can_attack_after:
        score += 36
    if before_main_gap > 1 and after_main_gap == 1:
        score += 22
    if after_main_gap < before_main_gap:
        score += 12
    if best_damage_after > best_damage_before:
        score += 8
    if is_main_attacker_for_player(
        obs.current.players[obs.current.yourIndex],
        target,
        card_data_by_id,
        attack_by_id,
    ):
        score += 10
    if is_likely_to_fall_next_turn(obs, target, card_data_by_id, attack_by_id):
        score -= 10
    return score


def score_self_trainer_gain(
    obs: Observation,
    card_data: CardData,
    *,
    from_discard: bool,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    player = obs.current.players[obs.current.yourIndex]
    active = active_pokemon(player)
    active_card_data = card_data_by_id.get(active.id) if active is not None else None
    active_attacks = resolve_attacks(active_card_data, attack_by_id) if active_card_data is not None else []
    active_main_gap = energy_gap_for_attack(
        choose_main_attack(active_attacks),
        list(active.energies),
    ) if active is not None else 99

    item_profile = get_item_profile(card_data.cardId)
    if item_profile is not None:
        score = 0.0
        if item_profile.switches_own_active and active is not None and player.bench:
            score += 30 if is_likely_to_fall_next_turn(obs, active, card_data_by_id, attack_by_id) else 12
        if item_profile.active_damage_bonus > 0 and active_card_data is not None:
            if not item_profile.only_for_fighting_pokemon or active_card_data.energyType == EnergyType.FIGHTING:
                if active_main_gap <= 1:
                    score += 18
        if item_profile.searches_basic_energy > 0 and not has_basic_energy_in_hand(player.hand, card_data_by_id):
            score += 16
        if item_profile.searches_pokemon > 0 and count_ready_attackers(player, card_data_by_id, attack_by_id) == 0:
            score += 14
        if item_profile.recovers_pokemon_from_discard > 0 and has_pokemon_in_discard(player.discard, card_data_by_id):
            score += 12
        if item_profile.recovers_basic_energy_from_discard > 0 and has_basic_energy_in_discard(player.discard, card_data_by_id):
            score += 9
        if from_discard and score > 0:
            score += 2
        return score

    supporter_profile = get_supporter_profile(card_data.cardId)
    if supporter_profile is not None:
        score = float(min(18, supporter_profile.draw_cards * 2))
        if supporter_profile.gust_effect:
            score += 18 if active_main_gap <= 1 else 8
        if supporter_profile.both_players_redraw:
            score -= 4
        if supporter_profile.hand_reset and player.handCount >= 5:
            score -= 3
        return score

    tool_profile = get_tool_profile(card_data.cardId)
    if tool_profile is not None:
        score = 0.0
        if tool_profile.retreat_cost_reduction > 0 and active_card_data is not None and active_card_data.retreatCost >= 2:
            score += 8
        if tool_profile.active_damage_bonus > 0 or tool_profile.active_damage_bonus_vs_ex > 0:
            if active_main_gap <= 1:
                score += 10
        return score

    return 0.0


def score_self_pokemon_bounce(
    obs: Observation,
    pokemon: Pokemon,
    card_data: CardData,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    player = obs.current.players[obs.current.yourIndex]
    score = -18.0

    if is_likely_to_fall_next_turn(obs, pokemon, card_data_by_id, attack_by_id):
        score += 24
    if card_data.basic and max(player.benchMax - len(player.bench), 0) > 0:
        score += 6
    if len(pokemon.energies) >= 2:
        score -= 10
    if pokemon.preEvolution or card_data.stage1 or card_data.stage2:
        score -= 8
    if is_main_attacker_for_player(player, pokemon, card_data_by_id, attack_by_id):
        score -= 4

    return score


def score_opponent_to_hand_option(
    obs: Observation,
    view: CardMoveOptionViewLike,
    card_data: CardData,
    entry: object | None,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    if view.area in {AreaType.ACTIVE, AreaType.BENCH} and isinstance(entry, Pokemon):
        # 相手の場を手札に戻すときは、テンポ妨害として評価する。
        return score_opponent_pokemon_bounce(
            obs,
            view,
            entry,
            card_data,
            card_data_by_id,
            attack_by_id,
        )

    return -score_opponent_gain_value(
        obs,
        card_data,
        card_data_by_id,
        attack_by_id,
    )


def score_opponent_pokemon_bounce(
    obs: Observation,
    view: CardMoveOptionViewLike,
    pokemon: Pokemon,
    card_data: CardData,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    opponent = obs.current.players[1 - obs.current.yourIndex]
    attacks = resolve_attacks(card_data, attack_by_id)
    main_attack = choose_main_attack(attacks)
    can_attack_now = has_payable_attack(attacks, list(pokemon.energies))
    main_gap = energy_gap_for_attack(main_attack, list(pokemon.energies))

    score = 0.0
    if view.area == AreaType.ACTIVE:
        score += 16
    else:
        score += 8
    if can_attack_now:
        # 次ターンの攻撃を止められるなら最優先に近い。
        score += 26
    elif main_gap == 1:
        score += 12
    if is_main_attacker_for_player(opponent, pokemon, card_data_by_id, attack_by_id):
        score += 10
    if card_data.stage1 or card_data.stage2 or card_data.ex:
        score += 8
    if pokemon.maxHp > 0 and pokemon.hp / pokemon.maxHp <= 0.4:
        score -= 24
    if view.area == AreaType.ACTIVE and self_can_knock_out_opponent_active(obs, pokemon, card_data_by_id, attack_by_id):
        score -= 28

    return score


def score_opponent_gain_value(
    obs: Observation,
    card_data: CardData,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> float:
    opponent = obs.current.players[1 - obs.current.yourIndex]
    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        return 14

    if card_data.cardType == CardType.POKEMON:
        score = 4.0
        if card_data.evolvesFrom:
            if has_named_pokemon_in_play(opponent, card_data.evolvesFrom, card_data_by_id):
                score += 18
            else:
                score -= 2
        elif card_data.basic:
            score += 8
        score += attacker_priority_score(
            pokemon_from_card_data(card_data),
            card_data_by_id,
            attack_by_id,
        ) / 8
        return score

    item_profile = get_item_profile(card_data.cardId)
    if item_profile is not None:
        score = 8.0
        if item_profile.switches_own_active:
            score += 8
        if item_profile.searches_pokemon > 0 or item_profile.searches_basic_pokemon > 0:
            score += 10
        if item_profile.searches_basic_energy > 0:
            score += 10
        if item_profile.active_damage_bonus > 0:
            score += 8
        return score

    supporter_profile = get_supporter_profile(card_data.cardId)
    if supporter_profile is not None:
        score = 10.0 + min(10, supporter_profile.draw_cards * 2)
        if supporter_profile.gust_effect:
            score += 10
        return score

    return 6.0


def has_named_pokemon_in_play(player, name: str, card_data_by_id: dict[int, CardData]) -> bool:
    for pokemon in in_play_pokemon(player):
        current = card_data_by_id.get(pokemon.id)
        if current is not None and current.name == name:
            return True
    return False


def has_named_card(cards: list[Card] | None, name: str, card_data_by_id: dict[int, CardData]) -> bool:
    if cards is None:
        return False
    for card in cards:
        card_data = card_data_by_id.get(card.id)
        if card_data is not None and card_data.name == name:
            return True
    return False


def has_basic_energy_in_hand(cards: list[Card] | None, card_data_by_id: dict[int, CardData]) -> bool:
    if cards is None:
        return False
    return any(
        (card_data := card_data_by_id.get(card.id)) is not None
        and card_data.cardType == CardType.BASIC_ENERGY
        for card in cards
    )


def has_pokemon_in_discard(cards: list[Card] | None, card_data_by_id: dict[int, CardData]) -> bool:
    if cards is None:
        return False
    return any(
        (card_data := card_data_by_id.get(card.id)) is not None
        and card_data.cardType == CardType.POKEMON
        for card in cards
    )


def has_basic_energy_in_discard(cards: list[Card] | None, card_data_by_id: dict[int, CardData]) -> bool:
    if cards is None:
        return False
    return any(
        (card_data := card_data_by_id.get(card.id)) is not None
        and card_data.cardType == CardType.BASIC_ENERGY
        for card in cards
    )


def count_ready_attackers(player, card_data_by_id: dict[int, CardData], attack_by_id) -> int:
    count = 0
    for pokemon in in_play_pokemon(player):
        card_data = card_data_by_id.get(pokemon.id)
        if card_data is None:
            continue
        if has_payable_attack(resolve_attacks(card_data, attack_by_id), list(pokemon.energies)):
            count += 1
    return count


def benched_energy_acceleration_is_online(player, card_data_by_id: dict[int, CardData]) -> bool:
    for pokemon in in_play_pokemon(player):
        card_data = card_data_by_id.get(pokemon.id)
        if card_data is None:
            continue
        for attack_id in card_data.attacks:
            profile = get_attack_effect_profile(attack_id)
            if profile is not None and profile.accelerates_energy_to_bench > 0:
                return True
    return False


def is_player_under_pressure(player) -> bool:
    active = active_pokemon(player)
    if active is None or active.maxHp <= 0:
        return False
    return active.hp / active.maxHp <= 0.55


def is_main_attacker_for_player(player, target: Pokemon, card_data_by_id: dict[int, CardData], attack_by_id) -> bool:
    in_play = in_play_pokemon(player)
    if not in_play:
        return False
    target_score = attacker_priority_score(target, card_data_by_id, attack_by_id)
    best_score = max(
        attacker_priority_score(pokemon, card_data_by_id, attack_by_id)
        for pokemon in in_play
    )
    return target_score == best_score


def is_likely_to_fall_next_turn(
    obs: Observation,
    target: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> bool:
    opponent = obs.current.players[1 - obs.current.yourIndex]
    opponent_active = active_pokemon(opponent)
    if opponent_active is None:
        return target.hp <= 30

    opponent_card_data = card_data_by_id.get(opponent_active.id)
    if opponent_card_data is None:
        return target.hp <= 30

    threatening_damage = best_payable_attack_damage(
        resolve_attacks(opponent_card_data, attack_by_id),
        list(opponent_active.energies),
    )
    return threatening_damage >= target.hp or target.hp <= 30


def self_can_knock_out_opponent_active(
    obs: Observation,
    opponent_active: Pokemon,
    card_data_by_id: dict[int, CardData],
    attack_by_id,
) -> bool:
    player = obs.current.players[obs.current.yourIndex]
    active = active_pokemon(player)
    if active is None:
        return False
    active_card_data = card_data_by_id.get(active.id)
    if active_card_data is None:
        return False
    return best_payable_attack_damage(
        resolve_attacks(active_card_data, attack_by_id),
        list(active.energies),
    ) >= opponent_active.hp


def active_pokemon(player) -> Pokemon | None:
    return player.active[0] if player.active else None


def in_play_pokemon(player) -> list[Pokemon]:
    return [pokemon for pokemon in list(player.active) + list(player.bench) if pokemon is not None]


def pokemon_from_card_data(card_data: CardData) -> Pokemon:
    return Pokemon(
        id=card_data.cardId,
        serial=0,
        hp=card_data.hp,
        maxHp=card_data.hp,
        appearThisTurn=False,
        energies=[],
        energyCards=[],
        tools=[],
        preEvolution=[],
    )
