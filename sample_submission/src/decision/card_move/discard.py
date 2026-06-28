from typing import Any

from cg.api import AreaType, CardData, CardType, EnergyType, Observation, Pokemon

from src.decision.card_move.common import (
    CardMoveOptionView,
    _card_data_lookup,
    analyze_card_move_option,
    count_basic_energy_cards,
    resolve_option_entry,
)
from src.decision.evaluation.attack_features import (
    build_attack_lookup,
    choose_main_attack,
    resolve_attacks,
)
from src.decision.evaluation.energy_requirements import energy_gap_for_attack
from src.decision.fallback import choose_random_legal_action
from src.knowledge.deck_profiles import (
    get_attack_effect_profile,
    get_item_profile,
    get_pokemon_profile,
    get_supporter_profile,
)


def choose_discard_action(obs: Observation) -> list[int]:
    """Choose DISCARD targets by comparing discard gain against future loss."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during discard decisions.")

    option_count = len(obs.select.option)
    if option_count == 0:
        return [] if obs.select.minCount == 0 else choose_random_legal_action(obs)
    if obs.select.minCount == obs.select.maxCount == option_count:
        return list(range(option_count))

    views = [
        analyze_card_move_option(obs, option_index)
        for option_index in range(option_count)
    ]
    views = [view for view in views if view is not None]
    if not views:
        return choose_random_legal_action(obs)

    card_data_by_id = _card_data_lookup()
    attack_by_id = build_attack_lookup()
    selected: list[int] = []
    selected_card_ids: dict[int, int] = {}

    while len(selected) < obs.select.maxCount:
        remaining = [
            view
            for view in views
            if view.option_index not in selected
        ]
        if not remaining:
            break

        scored_views = [
            (
                score_discard_option(
                    obs,
                    view,
                    selected_card_ids,
                    views,
                    card_data_by_id,
                    attack_by_id,
                ),
                view,
            )
            for view in remaining
        ]
        best_score, best_view = max(scored_views, key=lambda item: item[0])

        if len(selected) >= obs.select.minCount and best_score <= 0:
            break

        selected.append(best_view.option_index)
        if best_view.card_id is not None:
            selected_card_ids[best_view.card_id] = (
                selected_card_ids.get(best_view.card_id, 0) + 1
            )

    if len(selected) < obs.select.minCount:
        remaining = [
            view
            for view in views
            if view.option_index not in selected
        ]
        remaining.sort(
            key=lambda view: score_discard_option(
                obs,
                view,
                selected_card_ids,
                views,
                card_data_by_id,
                attack_by_id,
            ),
            reverse=True,
        )
        for view in remaining:
            if len(selected) >= obs.select.minCount:
                break
            selected.append(view.option_index)
            if view.card_id is not None:
                selected_card_ids[view.card_id] = (
                    selected_card_ids.get(view.card_id, 0) + 1
                )

    if not selected and obs.select.minCount == 0:
        return []
    if len(selected) < obs.select.minCount:
        return choose_random_legal_action(obs)

    return selected


def score_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    all_views: list[CardMoveOptionView],
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Any],
) -> float:
    if view.owner_is_self is True:
        return _score_self_discard_option(
            obs,
            view,
            selected_card_ids,
            card_data_by_id,
            attack_by_id,
        )
    if view.owner_is_self is False:
        return _score_opponent_discard_option(
            obs,
            view,
            selected_card_ids,
            all_views,
            card_data_by_id,
        )
    return -1.0


def _score_self_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Any],
) -> float:
    if obs.current is None:
        return -1.0

    if view.area == AreaType.HAND:
        return _score_self_hand_discard_option(
            obs,
            view,
            selected_card_ids,
            card_data_by_id,
            attack_by_id,
        )
    if view.area in {AreaType.ACTIVE, AreaType.BENCH}:
        return _score_self_field_discard_option(obs, view, card_data_by_id)
    return -3.0


def _score_opponent_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    all_views: list[CardMoveOptionView],
    card_data_by_id: dict[int, CardData],
) -> float:
    if view.area == AreaType.HAND:
        return _score_opponent_hand_discard_option(
            obs,
            view,
            selected_card_ids,
            all_views,
            card_data_by_id,
        )
    if view.area in {AreaType.ACTIVE, AreaType.BENCH}:
        return _score_opponent_field_discard_option(obs, view, card_data_by_id)
    return 0.0


def _score_self_hand_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Any],
) -> float:
    if obs.current is None or view.card_id is None:
        return -1.0

    player = obs.current.players[obs.current.yourIndex]
    card_data = card_data_by_id.get(view.card_id)
    if card_data is None:
        return 0.0

    current_hand_copy_count = _remaining_hand_copy_count(
        player.hand,
        selected_card_ids,
        view.card_id,
    )
    remaining_after_discard = current_hand_copy_count - 1
    in_play_count = _in_play_copy_count(obs, owner_is_self=True, card_id=view.card_id)
    score = 0.0

    if card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        score -= 5
        if profile is not None:
            score -= profile.active_role_bonus * 0.45
            score -= profile.bench_setup_bonus * 0.6
            total_future_copies = in_play_count + remaining_after_discard
            if total_future_copies >= profile.max_useful_copies:
                score += 4 * (total_future_copies - profile.max_useful_copies + 1)
        elif current_hand_copy_count > 1:
            score += 2 * (current_hand_copy_count - 1)

        if card_data.ex or card_data.megaEx:
            score -= 6
        if card_data.evolvesFrom:
            if _name_in_play(
                obs,
                owner_is_self=True,
                name=card_data.evolvesFrom,
                card_data_by_id=card_data_by_id,
            ):
                score -= 8
            elif _name_in_remaining_hand(
                player.hand,
                selected_card_ids,
                card_data.evolvesFrom,
                card_data_by_id,
            ):
                score -= 4
            else:
                score += 1
        elif _has_remaining_evolution_in_hand(
            player.hand,
            selected_card_ids,
            evolves_from=card_data.name,
            card_data_by_id=card_data_by_id,
        ):
            score -= 6

    elif card_data.cardType == CardType.BASIC_ENERGY:
        current_energy_count = _remaining_self_hand_energy_count(
            player.hand,
            selected_card_ids,
            card_data_by_id,
        )
        remaining_energy_count = current_energy_count - 1

        score -= 4
        if current_energy_count >= 3:
            score += 5 + (current_energy_count - 3) * 2
        elif current_energy_count == 2:
            score += 1

        if _active_wants_more_energy(obs, attack_by_id, card_data_by_id):
            score -= 5
            if remaining_energy_count <= 0:
                score -= 8

        if view.energy_type == EnergyType.FIGHTING:
            score += _fighting_energy_discard_bonus(
                obs,
                selected_card_ids,
                card_data_by_id,
                attack_by_id,
            )

    elif card_data.cardType == CardType.SPECIAL_ENERGY:
        score -= 14
        if remaining_after_discard <= 0:
            score -= 4

    elif card_data.cardType == CardType.SUPPORTER:
        profile = get_supporter_profile(card_data.cardId)
        score -= 7
        if obs.current.supporterPlayed:
            score += 2
        if profile is not None:
            score -= profile.draw_cards * 0.6
            if profile.gust_effect:
                score -= 4
            if profile.hand_reset:
                score -= 3
        if current_hand_copy_count > 1:
            score += current_hand_copy_count - 1

    elif card_data.cardType == CardType.ITEM:
        profile = get_item_profile(card_data.cardId)
        score -= 5
        if profile is not None:
            score -= profile.searches_pokemon * 2
            score -= profile.searches_basic_pokemon * 1.5
            score -= profile.searches_basic_energy * 1.5
            score -= profile.recovers_pokemon_from_discard * 1.5
            score -= profile.recovers_basic_energy_from_discard * 1.5
            if profile.switches_own_active:
                score -= 2
            score -= profile.discard_cost * 0.5
        if current_hand_copy_count > 1:
            score += current_hand_copy_count - 1

    elif card_data.cardType in {CardType.TOOL, CardType.STADIUM}:
        score -= 4

    else:
        score -= 3

    return score


def _score_opponent_hand_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    selected_card_ids: dict[int, int],
    all_views: list[CardMoveOptionView],
    card_data_by_id: dict[int, CardData],
) -> float:
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 1.0

    duplicate_count = _remaining_view_copy_count(
        all_views,
        selected_card_ids,
        owner_is_self=False,
        area=AreaType.HAND,
        card_id=card_data.cardId,
    )
    score = 0.0

    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        score += 16
        if _opponent_active_wants_more_energy(obs):
            score += 4
    elif card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        score += 10
        if card_data.evolvesFrom and _name_in_play(
            obs,
            owner_is_self=False,
            name=card_data.evolvesFrom,
            card_data_by_id=card_data_by_id,
        ):
            score += 6
        if profile is not None:
            score += profile.active_role_bonus * 0.5
            score += profile.bench_setup_bonus * 0.35
        if card_data.ex or card_data.megaEx:
            score += 4
    elif card_data.cardType == CardType.SUPPORTER:
        profile = get_supporter_profile(card_data.cardId)
        score += 10
        if profile is not None:
            score += profile.draw_cards * 0.8
            if profile.gust_effect:
                score += 4
            if profile.hand_reset:
                score += 3
    elif card_data.cardType == CardType.ITEM:
        profile = get_item_profile(card_data.cardId)
        score += 7
        if profile is not None:
            score += profile.searches_pokemon * 3
            score += profile.searches_basic_pokemon * 2
            score += profile.searches_basic_energy * 2
            score += profile.recovers_pokemon_from_discard * 2
            score += profile.recovers_basic_energy_from_discard * 2
            if profile.switches_own_active:
                score += 2
    else:
        score += 4

    if duplicate_count > 1:
        score -= min(duplicate_count - 1, 2) * 1.5

    return score


def _score_self_field_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    card_data_by_id: dict[int, CardData],
) -> float:
    entry = resolve_option_entry(obs, view.option_index)
    if not isinstance(entry, Pokemon):
        return -8.0

    card_data = card_data_by_id.get(entry.id)
    score = -16.0
    if card_data is None:
        return score

    profile = get_pokemon_profile(card_data.cardId)
    if profile is not None:
        score -= profile.active_role_bonus * 0.7
        score -= profile.bench_setup_bonus * 0.45
    if card_data.ex or card_data.megaEx:
        score -= 6
    score -= len(entry.energies) * 3

    if entry.maxHp > 0 and entry.hp / entry.maxHp <= 0.4:
        score += 3
    if view.area == AreaType.BENCH and not entry.energies:
        score += 2

    return score


def _score_opponent_field_discard_option(
    obs: Observation,
    view: CardMoveOptionView,
    card_data_by_id: dict[int, CardData],
) -> float:
    entry = resolve_option_entry(obs, view.option_index)
    if not isinstance(entry, Pokemon):
        return 2.0

    card_data = card_data_by_id.get(entry.id)
    if card_data is None:
        return 3.0

    profile = get_pokemon_profile(card_data.cardId)
    score = 8.0
    if profile is not None:
        score += profile.active_role_bonus * 0.55
        score += profile.bench_setup_bonus * 0.3
    if view.area == AreaType.ACTIVE:
        score += 4
    if card_data.ex or card_data.megaEx:
        score += 4
    score += len(entry.energies) * 3
    return score


def _remaining_hand_copy_count(
    hand: list[Any] | None,
    selected_card_ids: dict[int, int],
    card_id: int,
) -> int:
    total = sum(1 for card in hand or [] if getattr(card, "id", None) == card_id)
    return max(0, total - selected_card_ids.get(card_id, 0))


def _remaining_self_hand_energy_count(
    hand: list[Any] | None,
    selected_card_ids: dict[int, int],
    card_data_by_id: dict[int, CardData],
) -> int:
    remaining_selected = dict(selected_card_ids)
    count = 0

    for card in hand or []:
        if remaining_selected.get(card.id, 0) > 0:
            remaining_selected[card.id] -= 1
            continue
        card_data = card_data_by_id.get(card.id)
        if card_data is None:
            continue
        if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
            count += 1

    return count


def _fighting_energy_discard_bonus(
    obs: Observation,
    selected_card_ids: dict[int, int],
    card_data_by_id: dict[int, CardData],
    attack_by_id: dict[int, Any],
) -> float:
    if obs.current is None:
        return 0.0

    player = obs.current.players[obs.current.yourIndex]
    hand_count = _remaining_basic_energy_count_in_hand(
        player.hand,
        selected_card_ids,
        EnergyType.FIGHTING,
        card_data_by_id,
    )
    remaining_after_discard = hand_count - 1
    if remaining_after_discard < 0:
        return 0.0
    if remaining_after_discard <= 0 and _active_wants_more_energy(obs, attack_by_id, card_data_by_id):
        return -2.0
    if not _energy_acceleration_is_online(player, card_data_by_id):
        return 0.0
    if not player.bench:
        return 0.0

    discard_count = count_basic_energy_cards(player.discard, EnergyType.FIGHTING)
    discard_count += _selected_basic_energy_count(
        selected_card_ids,
        EnergyType.FIGHTING,
        card_data_by_id,
    )

    bonus = 0.0
    if discard_count == 0:
        bonus += 6
    elif discard_count == 1:
        bonus += 3
    elif discard_count == 2:
        bonus += 1

    if hand_count >= 3:
        bonus += 2

    return bonus


def _remaining_basic_energy_count_in_hand(
    hand: list[Any] | None,
    selected_card_ids: dict[int, int],
    energy_type: EnergyType,
    card_data_by_id: dict[int, CardData],
) -> int:
    remaining_selected = dict(selected_card_ids)
    count = 0

    for card in hand or []:
        if remaining_selected.get(card.id, 0) > 0:
            remaining_selected[card.id] -= 1
            continue
        card_data = card_data_by_id.get(card.id)
        if card_data is None:
            continue
        if card_data.cardType == CardType.BASIC_ENERGY and card_data.energyType == energy_type:
            count += 1

    return count


def _selected_basic_energy_count(
    selected_card_ids: dict[int, int],
    energy_type: EnergyType,
    card_data_by_id: dict[int, CardData],
) -> int:
    count = 0
    for card_id, selected_count in selected_card_ids.items():
        if selected_count <= 0:
            continue
        card_data = card_data_by_id.get(card_id)
        if card_data is None:
            continue
        if card_data.cardType == CardType.BASIC_ENERGY and card_data.energyType == energy_type:
            count += selected_count
    return count


def _active_wants_more_energy(
    obs: Observation,
    attack_by_id: dict[int, Any],
    card_data_by_id: dict[int, CardData],
) -> bool:
    if obs.current is None:
        return False

    player = obs.current.players[obs.current.yourIndex]
    active = player.active[0] if player.active else None
    if active is None:
        return False

    card_data = card_data_by_id.get(active.id)
    if card_data is None:
        return False

    attacks = resolve_attacks(card_data, attack_by_id)
    main_attack = choose_main_attack(attacks)
    if main_attack is None:
        return False

    return energy_gap_for_attack(main_attack, active.energies) >= 1


def _opponent_active_wants_more_energy(obs: Observation) -> bool:
    if obs.current is None:
        return False

    opponent = obs.current.players[1 - obs.current.yourIndex]
    active = opponent.active[0] if opponent.active else None
    if active is None:
        return False
    return len(active.energies) <= 1


def _energy_acceleration_is_online(
    player: Any,
    card_data_by_id: dict[int, CardData],
) -> bool:
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            continue
        card_data = card_data_by_id.get(pokemon.id)
        if card_data is None:
            continue
        for attack_id in card_data.attacks:
            profile = get_attack_effect_profile(attack_id)
            if profile is not None and profile.accelerates_energy_to_bench > 0:
                return True
    return False


def _in_play_copy_count(obs: Observation, *, owner_is_self: bool, card_id: int) -> int:
    player = _player(obs, owner_is_self)
    if player is None:
        return 0

    count = 0
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is not None and pokemon.id == card_id:
            count += 1
    return count


def _remaining_view_copy_count(
    views: list[CardMoveOptionView],
    selected_card_ids: dict[int, int],
    *,
    owner_is_self: bool,
    area: AreaType,
    card_id: int,
) -> int:
    total = sum(
        1
        for view in views
        if view.owner_is_self is owner_is_self
        and view.area == area
        and view.card_id == card_id
    )
    return max(0, total - selected_card_ids.get(card_id, 0))


def _name_in_play(
    obs: Observation,
    *,
    owner_is_self: bool,
    name: str,
    card_data_by_id: dict[int, CardData],
) -> bool:
    player = _player(obs, owner_is_self)
    if player is None:
        return False

    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            continue
        card_data = card_data_by_id.get(pokemon.id)
        if card_data is not None and card_data.name == name:
            return True
    return False


def _name_in_remaining_hand(
    hand: list[Any] | None,
    selected_card_ids: dict[int, int],
    name: str,
    card_data_by_id: dict[int, CardData],
) -> bool:
    remaining_selected = dict(selected_card_ids)
    for card in hand or []:
        if remaining_selected.get(card.id, 0) > 0:
            remaining_selected[card.id] -= 1
            continue
        card_data = card_data_by_id.get(card.id)
        if card_data is not None and card_data.name == name:
            return True
    return False


def _has_remaining_evolution_in_hand(
    hand: list[Any] | None,
    selected_card_ids: dict[int, int],
    *,
    evolves_from: str,
    card_data_by_id: dict[int, CardData],
) -> bool:
    remaining_selected = dict(selected_card_ids)
    for card in hand or []:
        if remaining_selected.get(card.id, 0) > 0:
            remaining_selected[card.id] -= 1
            continue
        card_data = card_data_by_id.get(card.id)
        if card_data is not None and card_data.evolvesFrom == evolves_from:
            return True
    return False


def _player(obs: Observation, owner_is_self: bool):
    if obs.current is None:
        return None

    owner_index = obs.current.yourIndex if owner_is_self else 1 - obs.current.yourIndex
    return obs.current.players[owner_index]
