from dataclasses import dataclass
from functools import lru_cache

from cg.api import AreaType, Card, CardData, CardType, EnergyType, Observation, Pokemon

from src.knowledge.card_cache import load_card_data


@dataclass(frozen=True)
class CardMoveOptionView:
    """カード移動候補を読むための共通ビュー。"""

    option_index: int
    owner_is_self: bool | None
    area: AreaType | None
    card_id: int | None = None
    card_type: CardType | None = None
    energy_type: EnergyType | None = None
    is_pokemon: bool | None = None
    is_energy: bool | None = None
    is_basic_energy: bool | None = None
    is_special_energy: bool | None = None


def analyze_card_move_option(
    obs: Observation,
    option_index: int,
) -> CardMoveOptionView | None:
    """option を owner / area / card_id ベースで解釈する。"""
    if obs.select is None:
        return None

    option = obs.select.option[option_index]
    owner_is_self = None
    if obs.current is not None and option.playerIndex is not None:
        owner_is_self = option.playerIndex == obs.current.yourIndex

    card_id = resolve_option_card_id(obs, option_index)
    card_data_by_id = _card_data_lookup()
    card_data = card_data_by_id.get(card_id) if card_id is not None else None

    return CardMoveOptionView(
        option_index=option_index,
        owner_is_self=owner_is_self,
        area=option.area,
        card_id=card_id,
        card_type=card_data.cardType if card_data is not None else None,
        energy_type=card_data.energyType if card_data is not None else None,
        is_pokemon=card_data.cardType == CardType.POKEMON if card_data is not None else None,
        is_energy=(
            card_data.cardType in {CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY}
            if card_data is not None
            else None
        ),
        is_basic_energy=card_data.cardType == CardType.BASIC_ENERGY if card_data is not None else None,
        is_special_energy=card_data.cardType == CardType.SPECIAL_ENERGY if card_data is not None else None,
    )


def resolve_option_card_id(obs: Observation, option_index: int) -> int | None:
    """option が指す実体から card_id を取り出す。"""
    entry = resolve_option_entry(obs, option_index)
    if isinstance(entry, Card):
        return entry.id
    if isinstance(entry, Pokemon):
        return entry.id
    return None


def resolve_option_entry(obs: Observation, option_index: int) -> object | None:
    """option の area/index をたどって元の実体を返す。"""
    if obs.select is None:
        return None

    option = obs.select.option[option_index]
    if option.cardId is not None:
        return Card(
            id=option.cardId,
            serial=-1 if option.serial is None else option.serial,
            playerIndex=0 if option.playerIndex is None else option.playerIndex,
        )

    owner_index = 0
    if obs.current is not None:
        owner_index = obs.current.yourIndex
    if option.playerIndex is not None:
        owner_index = option.playerIndex

    source = resolve_option_source(obs, option.area, owner_index)
    if source is None or option.index is None:
        return None
    if option.index < 0 or option.index >= len(source):
        return None

    return source[option.index]


def resolve_option_source(
    obs: Observation,
    area: AreaType | None,
    owner_index: int,
) -> list[object] | None:
    """area ごとの元データ配列を返す。"""
    if area is None:
        return None
    if obs.current is None:
        return list(obs.select.deck) if area == AreaType.DECK and obs.select and obs.select.deck else None

    player = obs.current.players[owner_index]

    if area == AreaType.HAND:
        return player.hand
    if area == AreaType.DECK:
        return obs.select.deck
    if area == AreaType.DISCARD:
        return player.discard
    if area == AreaType.PRIZE:
        return player.prize
    if area == AreaType.ACTIVE:
        return player.active
    if area == AreaType.BENCH:
        return player.bench
    if area == AreaType.LOOKING:
        return obs.current.looking
    return None


def count_basic_energy_cards(cards: list[Card] | None, energy_type: EnergyType) -> int:
    """指定色の基本エネルギー枚数を数える。"""
    if cards is None:
        return 0

    count = 0
    for card in cards:
        card_data = _card_data_lookup().get(card.id)
        if card_data is None:
            continue
        if card_data.cardType == CardType.BASIC_ENERGY and card_data.energyType == energy_type:
            count += 1
    return count


@lru_cache(maxsize=1)
def _card_data_lookup() -> dict[int, CardData]:
    return {card.cardId: card for card in load_card_data()}
