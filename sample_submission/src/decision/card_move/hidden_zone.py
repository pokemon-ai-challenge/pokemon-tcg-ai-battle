from typing import Any

from cg.api import AreaType, CardData, CardType, Observation, SelectContext

from src.decision.card_move.common import analyze_card_move_option
from src.decision.fallback import choose_random_legal_action
from src.knowledge.card_cache import load_card_data
from src.knowledge.deck_profiles import (
    get_item_profile,
    get_pokemon_profile,
    get_supporter_profile,
)


def choose_hidden_zone_action(obs: Observation) -> list[int]:
    """Choose cards to move into a hidden zone without bloating the router."""
    if obs.select is None:
        raise ValueError("obs.select must not be None during hidden-zone decisions.")

    views = [
        analyze_card_move_option(obs, option_index)
        for option_index in range(len(obs.select.option))
    ]
    views = [view for view in views if view is not None]
    if not views:
        return choose_random_legal_action(obs)

    selected = choose_hidden_zone_placement_action(obs, views, _card_data_lookup())
    if selected is None:
        return choose_random_legal_action(obs)
    return selected


def choose_hidden_zone_placement_action(
    obs: Observation,
    views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> list[int] | None:
    if obs.select is None:
        raise ValueError("obs.select must not be None during deck placement decisions.")
    if not views:
        return None

    forced = _forced_full_hand_return(obs, views)
    if forced is not None:
        return forced

    scored_views: list[tuple[float, int]] = []
    for view in views:
        score = _score_deck_return_candidate(obs, view, views, card_data_by_id)
        scored_views.append((score, view.option_index))

    scored_views.sort(reverse=True)
    return _select_best_indices(
        scored_views,
        min_count=obs.select.minCount,
        max_count=obs.select.maxCount,
    )


def choose_to_deck_placement_action(
    obs: Observation,
    views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> list[int] | None:
    return choose_hidden_zone_placement_action(obs, views, card_data_by_id)


def _forced_full_hand_return(obs: Observation, views: list[Any]) -> list[int] | None:
    if obs.select is None or obs.current is None:
        return None
    if obs.select.context != SelectContext.TO_DECK_BOTTOM:
        return None
    if not views:
        return None
    if any(view.owner_is_self is not True or view.area != AreaType.HAND for view in views):
        return None

    player = obs.current.players[obs.current.yourIndex]
    hand = player.hand or []
    if len(hand) != len(views):
        return None
    if obs.select.minCount != len(views) or obs.select.maxCount != len(views):
        return None

    return [view.option_index for view in views]


def _select_best_indices(
    scored_views: list[tuple[float, int]],
    *,
    min_count: int,
    max_count: int,
) -> list[int]:
    if max_count <= 0:
        return []

    selected: list[int] = []
    for score, option_index in scored_views:
        if len(selected) < min_count:
            selected.append(option_index)
            continue
        if len(selected) >= max_count:
            break
        if score <= 0:
            break
        selected.append(option_index)

    if len(selected) < min_count:
        for _, option_index in scored_views:
            if option_index in selected:
                continue
            selected.append(option_index)
            if len(selected) >= min_count:
                break

    return selected


def _score_deck_return_candidate(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    if view.owner_is_self is True:
        return _score_self_return_candidate(obs, view, all_views, card_data_by_id)
    if view.owner_is_self is False:
        if view.area in {AreaType.DISCARD, AreaType.PRIZE, AreaType.LOOKING}:
            return -_score_opponent_return_candidate(obs, view, all_views, card_data_by_id)
        return _score_opponent_return_candidate(obs, view, all_views, card_data_by_id)
    return -1.0


def _score_self_return_candidate(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    keep_score = _self_keep_score(obs, view, all_views, card_data_by_id)

    if obs.select is None:
        return -keep_score

    if obs.select.context == SelectContext.TO_DECK:
        redraw_ok_score = _self_redraw_ok_score(view, all_views, card_data_by_id)
        if view.area in {AreaType.DISCARD, AreaType.PRIZE, AreaType.LOOKING}:
            return keep_score + redraw_ok_score
        return redraw_ok_score - keep_score

    if obs.select.context == SelectContext.TO_PRIZE:
        prize_ok_score = _self_prize_ok_score(obs, view, all_views, card_data_by_id)
        return prize_ok_score - keep_score

    bury_ok_score = _self_bury_ok_score(obs, view, all_views, card_data_by_id)
    return bury_ok_score - keep_score


def _score_opponent_return_candidate(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 1.0

    duplicate_count = _candidate_copy_count(all_views, view.card_id)
    score = 0.0

    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        score += 18
        if _opponent_active_is_energy_hungry(obs):
            score += 4
    elif card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        if card_data.evolvesFrom and _name_in_play(
            obs,
            owner_is_self=False,
            name=card_data.evolvesFrom,
            card_data_by_id=card_data_by_id,
        ):
            score += 15
        else:
            score += 10
        if profile is not None:
            score += profile.active_role_bonus * 0.45
            score += profile.bench_setup_bonus * 0.25
        if card_data.ex or card_data.megaEx:
            score += 4
    elif card_data.cardType == CardType.SUPPORTER:
        profile = get_supporter_profile(card_data.cardId)
        score += 9
        if profile is not None:
            score += profile.draw_cards * 0.6
            if profile.gust_effect:
                score += 4
            if profile.hand_reset:
                score += 3
    elif card_data.cardType == CardType.ITEM:
        profile = get_item_profile(card_data.cardId)
        score += 7
        if profile is not None:
            score += profile.searches_pokemon * 4
            score += profile.searches_basic_pokemon * 3
            score += profile.searches_basic_energy * 4
            score += profile.recovers_pokemon_from_discard * 3
            score += profile.recovers_basic_energy_from_discard * 3
            if profile.switches_own_active:
                score += 2
    else:
        score += 5

    if duplicate_count > 1:
        score -= min(duplicate_count - 1, 2) * 1.5

    return score


def _self_keep_score(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    if obs.current is None:
        return 0.0

    player = obs.current.players[obs.current.yourIndex]
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 4.0

    duplicate_count = _candidate_copy_count(all_views, card_data.cardId)
    in_play_count = _in_play_copy_count(obs, owner_is_self=True, card_id=card_data.cardId)
    score = 0.0

    if card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        if profile is not None:
            score += profile.active_role_bonus * 0.9
            score += profile.bench_setup_bonus * 0.8
            if profile.partner_card_ids and _partner_in_play(obs, profile.partner_card_ids):
                score += profile.partner_bonus * 0.45
        if card_data.ex or card_data.megaEx:
            score += 6
        if card_data.evolvesFrom:
            if _name_in_play(
                obs,
                owner_is_self=True,
                name=card_data.evolvesFrom,
                card_data_by_id=card_data_by_id,
            ):
                score += 11
            elif _name_in_hand(player.hand, card_data.evolvesFrom, card_data_by_id):
                score += 6
            else:
                score += 3
        elif card_data.basic and _ready_attacker_count(
            obs,
            owner_is_self=True,
            card_data_by_id=card_data_by_id,
        ) == 0:
            score += 4

        if profile is not None and duplicate_count + in_play_count > profile.max_useful_copies:
            score -= 5 * (duplicate_count + in_play_count - profile.max_useful_copies)
        elif duplicate_count > 1:
            score -= 2 * (duplicate_count - 1)

    elif card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        energy_count = _self_hand_energy_count(obs, all_views, card_data_by_id)
        score += 7
        if _ready_attacker_count(
            obs,
            owner_is_self=True,
            card_data_by_id=card_data_by_id,
        ) == 0:
            score += 2
        if energy_count > 2:
            score -= 3 * (energy_count - 2)

    elif card_data.cardType == CardType.SUPPORTER:
        profile = get_supporter_profile(card_data.cardId)
        score += 8
        if profile is not None:
            score += profile.draw_cards * 0.6
            if profile.gust_effect:
                score += 5
            if profile.hand_reset:
                score += 2
        if obs.current.supporterPlayed:
            score -= 3

    elif card_data.cardType == CardType.ITEM:
        profile = get_item_profile(card_data.cardId)
        score += 6
        if profile is not None:
            score += profile.searches_pokemon * 4
            score += profile.searches_basic_pokemon * 3
            score += profile.searches_basic_energy * 4
            score += profile.recovers_pokemon_from_discard * 3
            score += profile.recovers_basic_energy_from_discard * 3
            if profile.switches_own_active:
                score += 4
            score -= profile.discard_cost * 0.5

    elif card_data.cardType in {CardType.TOOL, CardType.STADIUM}:
        score += 4
    else:
        score += 5

    return score


def _self_redraw_ok_score(
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 2.0

    duplicate_count = _candidate_copy_count(all_views, card_data.cardId)
    score = 0.0

    if card_data.cardType == CardType.POKEMON:
        score += 3
        if card_data.evolvesFrom:
            score += 3
        if card_data.ex or card_data.megaEx:
            score += 2
    elif card_data.cardType in {CardType.SUPPORTER, CardType.ITEM}:
        score += 4
    elif card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        score += 1
    else:
        score += 2

    if duplicate_count > 1:
        score += min(duplicate_count - 1, 2) * 1.5

    return score


def _self_bury_ok_score(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 1.0

    duplicate_count = _candidate_copy_count(all_views, card_data.cardId)
    score = 0.0

    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        energy_count = _self_hand_energy_count(obs, all_views, card_data_by_id)
        if energy_count > 2:
            score += 7 + (energy_count - 3) * 2
        else:
            score += 1
    elif card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        if duplicate_count > 1:
            score += 5 * (duplicate_count - 1)
        if profile is None and not card_data.ex and not card_data.evolvesFrom:
            score += 3
    elif card_data.cardType == CardType.SUPPORTER:
        if obs.current is not None and obs.current.supporterPlayed:
            score += 2
    else:
        score += 2

    return score


def _self_prize_ok_score(
    obs: Observation,
    view: Any,
    all_views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> float:
    card_data = card_data_by_id.get(view.card_id) if view.card_id is not None else None
    if card_data is None:
        return 1.0

    duplicate_count = _candidate_copy_count(all_views, card_data.cardId)
    score = 0.0

    if card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}:
        energy_count = _self_hand_energy_count(obs, all_views, card_data_by_id)
        if energy_count > 2:
            score += 9 + (energy_count - 3) * 3
        elif energy_count == 2:
            score += 3
    elif card_data.cardType == CardType.POKEMON:
        profile = get_pokemon_profile(card_data.cardId)
        if duplicate_count > 1:
            score += 4 * (duplicate_count - 1)
        if profile is None and card_data.basic and not card_data.evolvesFrom:
            score += 3
        if card_data.ex or card_data.megaEx:
            score -= 4
        if card_data.evolvesFrom:
            score -= 2
    elif card_data.cardType == CardType.SUPPORTER:
        if obs.current is not None and obs.current.supporterPlayed:
            score += 1
    elif card_data.cardType == CardType.ITEM:
        score += 2
    elif card_data.cardType in {CardType.TOOL, CardType.STADIUM}:
        score += 1

    return score


def _candidate_copy_count(views: list[Any], card_id: int) -> int:
    return sum(1 for view in views if view.card_id == card_id)


def _self_hand_energy_count(
    obs: Observation,
    views: list[Any],
    card_data_by_id: dict[int, CardData],
) -> int:
    if obs.current is None:
        return 0

    hand_ids = [card.id for card in obs.current.players[obs.current.yourIndex].hand or []]
    if hand_ids:
        return sum(
            1
            for card_id in hand_ids
            if card_data_by_id.get(card_id) is not None
            and card_data_by_id[card_id].cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}
        )

    return sum(1 for view in views if view.owner_is_self is True and view.is_energy is True)


def _ready_attacker_count(
    obs: Observation,
    *,
    owner_is_self: bool,
    card_data_by_id: dict[int, CardData],
) -> int:
    if obs.current is None:
        return 0

    player = _player(obs, owner_is_self)
    if player is None:
        return 0

    count = 0
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            continue
        card_data = card_data_by_id.get(pokemon.id)
        if card_data is None or not card_data.attacks:
            continue
        profile = get_pokemon_profile(pokemon.id)
        if len(pokemon.energies) >= 2:
            count += 1
            continue
        if profile is not None and profile.active_role_bonus >= 18:
            count += 1
    return count


def _opponent_active_is_energy_hungry(obs: Observation) -> bool:
    if obs.current is None:
        return False

    player = obs.current.players[1 - obs.current.yourIndex]
    active = player.active[0] if player.active else None
    if active is None:
        return False
    return len(active.energies) <= 1


def _in_play_copy_count(obs: Observation, *, owner_is_self: bool, card_id: int) -> int:
    player = _player(obs, owner_is_self)
    if player is None:
        return 0

    counter = 0
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is not None and pokemon.id == card_id:
            counter += 1
    return counter


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


def _name_in_hand(
    hand: list[Any] | None,
    name: str,
    card_data_by_id: dict[int, CardData],
) -> bool:
    if hand is None:
        return False

    for card in hand:
        card_data = card_data_by_id.get(card.id)
        if card_data is not None and card_data.name == name:
            return True
    return False


def _partner_in_play(obs: Observation, partner_card_ids: frozenset[int]) -> bool:
    player = _player(obs, True)
    if player is None:
        return False

    for pokemon in list(player.active) + list(player.bench):
        if pokemon is not None and pokemon.id in partner_card_ids:
            return True
    return False


def _player(obs: Observation, owner_is_self: bool):
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    owner_index = your_index if owner_is_self else 1 - your_index
    return obs.current.players[owner_index]


def _card_data_lookup() -> dict[int, CardData]:
    return {card.cardId: card for card in load_card_data()}
