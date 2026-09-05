"""Temporary hidden-state stub for the lethal search (``search_begin()``).

This is dummy data, not a real prediction. The proper hidden-information
estimation (own deck / prizes, opponent deck / hand, ...) is being
developed separately; once it lands, callers should build the
``hidden_state`` from that module instead and this stub can be removed.
The lethal search itself never imports this module: it receives the
result from the outside via its ``hidden_state`` /
``hidden_state_factory`` context keys.

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


def _visible_own_cards(state, player_index: int) -> list[tuple[int, int]] | None:
    """Return the (card id, serial) of every own card visible in the observation.

    Returns None if the observation contains zones we cannot account for
    (face-down own Pokemon, cards being looked at), in which case the
    caller should give up instead of risking an inconsistent prediction.
    """
    player = state.players[player_index]
    cards: list[tuple[int, int]] = []
    for card in player.hand or []:
        cards.append((card.id, card.serial))
    for card in player.discard:
        cards.append((card.id, card.serial))
    for pokemon in list(player.active) + list(player.bench):
        if pokemon is None:
            return None  # own face-down Pokemon: cannot account for it
        cards.append((pokemon.id, pokemon.serial))
        for card in pokemon.energyCards:
            cards.append((card.id, card.serial))
        for card in pokemon.tools:
            cards.append((card.id, card.serial))
        for card in pokemon.preEvolution:
            cards.append((card.id, card.serial))
    for card in player.prize:
        if card is not None:
            cards.append((card.id, card.serial))
    for card in state.stadium:
        if card.playerIndex == player_index:
            cards.append((card.id, card.serial))
    return cards


def _visible_own_card_ids(state, player_index: int) -> Iterator[int] | None:
    """Backwards-compatible view of :func:`_visible_own_cards` (IDs only)."""
    cards = _visible_own_cards(state, player_index)
    if cards is None:
        return None
    return iter(card_id for card_id, _ in cards)


def build_dummy_search_state(
    obs: Observation,
    full_deck: list[int],
    rng: random.Random | None = None,
) -> dict | None:
    """Build dummy hidden-information arguments for ``search_begin()``.

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

    visible = _visible_own_cards(state, me)
    if visible is None:
        return None

    # A Trainer being resolved right now (``select.effect``) has already left
    # the hand but has not reached the discard pile yet, so it shows up in no
    # zone at all. Without counting it, the leftover pool is one card too big
    # and the consistency check below rejects the position -- which used to
    # skip the search on every mid-effect selection (deck search results,
    # bench placement, ...). Dedupe by serial: for effects that are already
    # visible somewhere (e.g. an attached Special Energy whose on-attach
    # effect is resolving) the card must not be counted twice.
    effect = obs.select.effect if obs.select is not None else None
    if effect is not None and effect.playerIndex == me:
        if all(serial != effect.serial for _, serial in visible):
            visible.append((effect.id, effect.serial))

    unseen = Counter(full_deck)
    for card_id, _ in visible:
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
