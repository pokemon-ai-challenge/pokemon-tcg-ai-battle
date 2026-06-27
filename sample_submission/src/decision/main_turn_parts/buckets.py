from dataclasses import dataclass, field

from cg.api import CardType, Observation, OptionType

from src.knowledge.card_cache import load_card_data


@dataclass
class MainOptionBuckets:
    """メインフェーズの選択肢を大まかな用途ごとに分けて持つ。"""

    pokemon_play: list[int] = field(default_factory=list)
    supporter_play: list[int] = field(default_factory=list)
    item_play: list[int] = field(default_factory=list)
    tool_play: list[int] = field(default_factory=list)
    stadium_play: list[int] = field(default_factory=list)
    evolve: list[int] = field(default_factory=list)
    ability: list[int] = field(default_factory=list)
    attach: list[int] = field(default_factory=list)
    retreat: list[int] = field(default_factory=list)
    attack: list[int] = field(default_factory=list)
    end: list[int] = field(default_factory=list)
    discard: list[int] = field(default_factory=list)
    other: list[int] = field(default_factory=list)


def bucket_main_options(obs: Observation) -> MainOptionBuckets:
    """メインフェーズの選択肢を用途ごとの箱に分ける。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None in main phase.")

    buckets = MainOptionBuckets()
    card_data_by_id = {card.cardId: card for card in load_card_data()}

    for option_index, option in enumerate(obs.select.option):
        if option.type == OptionType.PLAY:
            classify_play_option(obs, option_index, card_data_by_id, buckets)
        elif option.type == OptionType.EVOLVE:
            buckets.evolve.append(option_index)
        elif option.type == OptionType.ABILITY:
            buckets.ability.append(option_index)
        elif option.type == OptionType.ATTACH:
            buckets.attach.append(option_index)
        elif option.type == OptionType.RETREAT:
            buckets.retreat.append(option_index)
        elif option.type == OptionType.ATTACK:
            buckets.attack.append(option_index)
        elif option.type == OptionType.END:
            buckets.end.append(option_index)
        elif option.type == OptionType.DISCARD:
            buckets.discard.append(option_index)
        else:
            buckets.other.append(option_index)

    return buckets


def classify_play_option(
    obs: Observation,
    option_index: int,
    card_data_by_id: dict[int, object],
    buckets: MainOptionBuckets,
) -> None:
    """PLAY の選択肢を手札のカード種別ごとに分ける。"""
    if obs.select is None or obs.current is None:
        buckets.other.append(option_index)
        return

    option = obs.select.option[option_index]
    your_index = obs.current.yourIndex
    hand = obs.current.players[your_index].hand
    if hand is None or option.index is None or option.index >= len(hand):
        buckets.other.append(option_index)
        return

    hand_card = hand[option.index]
    card_data = card_data_by_id.get(hand_card.id)
    if card_data is None:
        buckets.other.append(option_index)
        return

    card_type = card_data.cardType
    if card_type == CardType.POKEMON:
        buckets.pokemon_play.append(option_index)
    elif card_type == CardType.SUPPORTER:
        buckets.supporter_play.append(option_index)
    elif card_type == CardType.ITEM:
        buckets.item_play.append(option_index)
    elif card_type == CardType.TOOL:
        buckets.tool_play.append(option_index)
    elif card_type == CardType.STADIUM:
        buckets.stadium_play.append(option_index)
    else:
        buckets.other.append(option_index)
