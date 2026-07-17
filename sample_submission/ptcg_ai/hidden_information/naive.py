"""Naive hidden-information prediction for ``cg.api.search_begin()``.

``search_begin()`` needs concrete card IDs for every hidden zone
(own deck / own face-down prizes / opponent deck / prizes / hand /
face-down Active). This module builds a consistent guess:

- Own hidden cards are derived exactly: the multiset of the 60-card deck
  list minus every own card visible in the observation. The remainder is
  split between the deck and the face-down prizes at random.
- Opponent hidden cards are unknowable, so they are filled with dummy
  cards (a Basic Energy, plus one Basic Pokemon in the deck). Phase 1
  lethal search only simulates our own turn, so their identity should
  not affect a deterministic lethal; the verification replays in
  ``lethal_simple`` guard against lines that would depend on them.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterator

from cg import api as cg_api
from cg.api import CardType, Observation


_FILLER_CACHE: dict[str, int] = {}


def _filler_ids() -> tuple[int, int]:
    """Return (basic_energy_id, basic_pokemon_id) for dummy fills."""
    if not _FILLER_CACHE:
        for card in cg_api.all_card_data():
            if "energy" not in _FILLER_CACHE and card.cardType == CardType.BASIC_ENERGY:
                _FILLER_CACHE["energy"] = card.cardId
            if "pokemon" not in _FILLER_CACHE and card.basic:
                _FILLER_CACHE["pokemon"] = card.cardId
            if "energy" in _FILLER_CACHE and "pokemon" in _FILLER_CACHE:
                break
    return _FILLER_CACHE["energy"], _FILLER_CACHE["pokemon"]


def _visible_own_card_ids(state, player_index: int) -> Iterator[int] | None:
    """Yield the card IDs of every own card visible in the observation.

    Returns None if the observation contains zones we cannot account for
    (face-down own Pokemon, cards being looked at), in which case the
    caller should give up instead of risking an inconsistent prediction.
    """
    player = state.players[player_index]
    ids: list[int] = []
    for card in player.hand or []:
        ids.append(card.id)
    for card in player.discard:
        ids.append(card.id)
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            return None  # own face-down Pokemon: cannot account for it
        ids.append(pokemon.id)
        for card in pokemon.energyCards:
            ids.append(card.id)
        for card in pokemon.tools:
            ids.append(card.id)
        for card in pokemon.preEvolution:
            ids.append(card.id)
    for card in player.prize:
        if card is not None:
            ids.append(card.id)
    for card in state.stadium:
        if card.playerIndex == player_index:
            ids.append(card.id)
    return iter(ids)


def predict_hidden(
    obs: Observation,
    full_deck: list[int],
    rng: random.Random | None = None,
) -> dict | None:
    """Build the hidden-information arguments for ``search_begin()``.

    Args:
        obs: The observation passed to the agent.
        full_deck: The 60 card IDs of our own deck (deck.csv order).
        rng: Random source for shuffling the own hidden pool.

    Returns:
        dict with keys ``your_deck``, ``your_prize``, ``opponent_deck``,
        ``opponent_prize``, ``opponent_hand``, ``opponent_active``,
        or None if a consistent prediction cannot be built.
    """
    state = obs.current
    if state is None:
        return None
    if state.looking is not None:
        # Cards mid-look may or may not be counted in deckCount; bail out.
        return None

    me = state.yourIndex
    my = state.players[me]

    visible = _visible_own_card_ids(state, me)
    if visible is None:
        return None
    unseen = Counter(full_deck)
    for card_id in visible:
        if unseen[card_id] <= 0:
            return None  # observation inconsistent with the deck list
        unseen[card_id] -= 1

    pool = [card_id for card_id, count in unseen.items() for _ in range(count)]
    facedown_prizes = sum(1 for card in my.prize if card is None)
    if len(pool) != my.deckCount + facedown_prizes:
        return None

    (rng or random).shuffle(pool)
    pool_iter = iter(pool)
    your_prize = [
        card.id if card is not None else next(pool_iter) for card in my.prize
    ]
    your_deck = list(pool_iter)

    opp = state.players[1 - me]
    energy_id, pokemon_id = _filler_ids()
    opponent_deck = [energy_id] * opp.deckCount
    if opponent_deck:
        opponent_deck[0] = pokemon_id
    opponent_prize = [
        card.id if card is not None else energy_id for card in opp.prize
    ]
    opponent_hand = [energy_id] * opp.handCount
    opponent_active: list[int] = []
    if len(opp.active) > 0 and opp.active[0] is None:
        opponent_active = [pokemon_id]

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }
