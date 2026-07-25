"""State/Pokemon から仕様判定に必要な値を読み取るヘルパー群。

汎用のゲーム内部表現の橋渡しのみを行い、カード優先度などの戦略判断はここに書かない。
"""

from __future__ import annotations

from cg.api import AreaType, CardType, Option, Pokemon, State

from ptcg_ai.custom_agent import constants
from ptcg_ai.shared import card_cache


def own(state: State) -> "PlayerState":
    return state.players[state.yourIndex]


def opponent(state: State) -> "PlayerState":
    return state.players[1 - state.yourIndex]


def board_pokemon(state: State, player_index: int | None = None) -> list[Pokemon]:
    """指定プレイヤー（省略時は自分）の場のポケモン（バトル場+ベンチ）を返す。"""
    player = state.players[player_index if player_index is not None else state.yourIndex]
    mons: list[Pokemon] = []
    if player.active and player.active[0] is not None:
        mons.append(player.active[0])
    mons.extend(p for p in player.bench if p is not None)
    return mons


def board_ids(state: State, player_index: int | None = None) -> list[int]:
    return [p.id for p in board_pokemon(state, player_index)]


def hand_ids(state: State) -> list[int]:
    hand = own(state).hand or []
    return [card.id for card in hand]


def own_hand_count(state: State) -> int:
    return own(state).handCount


def discard_ids(state: State, player_index: int | None = None) -> list[int]:
    player = state.players[player_index if player_index is not None else state.yourIndex]
    return [card.id for card in player.discard]


def discard_pokemon_count(state: State, player_index: int | None = None) -> int:
    return sum(
        1 for card_id in discard_ids(state, player_index) if card_cache.get_card(card_id).cardType == CardType.POKEMON
    )


def opponent_hand_count(state: State) -> int:
    return opponent(state).handCount


def opponent_active(state: State) -> Pokemon | None:
    player = opponent(state)
    if player.active and player.active[0] is not None:
        return player.active[0]
    return None


def opponent_bench(state: State) -> list[Pokemon]:
    return [p for p in opponent(state).bench if p is not None]


def opponent_active_has_special_energy(state: State) -> bool:
    active = opponent_active(state)
    if active is None:
        return False
    return any(card_cache.get_card(card.id).cardType == CardType.SPECIAL_ENERGY for card in active.energyCards)


def energy_count(pokemon: Pokemon) -> int:
    return len(pokemon.energies)


def is_ex(card_id: int) -> bool:
    card = card_cache.get_card(card_id)
    return bool(card.ex or card.megaEx)


def prize_value(card_id: int) -> int:
    """きぜつさせたときに相手に与えるサイド枚数（通常1、ex2、メガex3）。"""
    card = card_cache.get_card(card_id)
    if card.megaEx:
        return 3
    if card.ex:
        return 2
    return 1


def fudin_line_charged_count(state: State) -> int:
    """自分の場にいるフーライン個体のうち、エネルギーが1個以上付いている数。"""
    return sum(1 for p in board_pokemon(state) if p.id in constants.FUDIN_LINE and energy_count(p) > 0)


def own_ids_present(state: State) -> set[int]:
    """自分の場+手札に存在するカードIDの集合（進化先の充足判定などに使う）。"""
    return set(board_ids(state)) | set(hand_ids(state))


def resolve_inplay_pokemon(option: Option, state: State) -> Pokemon | None:
    """MAINレベルの ATTACH/EVOLVE Option が指す「場のポケモン」(inPlayArea/inPlayIndex) を解決する。

    cg/api.py: OptionType.ATTACH/EVOLVE は area/index が付け外しするカード側、
    inPlayArea/inPlayIndex が対象ポケモン側を指す。
    """
    if option.inPlayArea not in (AreaType.ACTIVE, AreaType.BENCH) or option.inPlayIndex is None:
        return None
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if option.inPlayArea == AreaType.ACTIVE else player.bench
    if not (0 <= option.inPlayIndex < len(zone)):
        return None
    return zone[option.inPlayIndex]
