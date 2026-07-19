"""クラスタ④ カード移動・対象選択（共通ヘルパー）／担当B

decision.card_move 配下の各モジュールが共有するヘルパー。
Option 配列を area / playerIndex / type などでフィルタする汎用処理のみを置く。
カード名・カードIDのハードコードはしない（knowledge.profile_registry 経由で参照する）。
"""

from typing import Callable

from cg.api import AreaType, Option, OptionType, PlayerState, Pokemon, SelectData, State

# cg/api.py のコメントによると、Option.cardId が直接入るのは OptionType.SKILL のみ。
# CARD/PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/TOOL_CARD/ENERGY_CARD などは
# area/index（+ toolIndex/energyIndex）経由でしか対象カードを特定できないため、
# option.cardId を直接読むコードは書かない。必ずこの resolve_card_id を経由する。
_AREA_TO_PLAYER_ZONE: dict[AreaType, str] = {
    AreaType.HAND: "hand",
    AreaType.DISCARD: "discard",
    AreaType.PRIZE: "prize",
    AreaType.ACTIVE: "active",
    AreaType.BENCH: "bench",
}


def filter_by_area(options: list[Option], area: AreaType) -> list[Option]:
    """指定した AreaType の Option だけを抽出する。"""
    return [option for option in options if option.area == area]


def filter_own(options: list[Option], your_index: int) -> list[Option]:
    """playerIndex が自分のものである Option だけを抽出する。"""
    return [option for option in options if option.playerIndex == your_index]


def resolve_card_id(option: Option, state: State) -> int | None:
    """Option が指すカード/ポケモンの card_id (CardData.id) を特定する。

    OptionType.SKILL 以外は option.cardId が None のことが多いため、
    area/index（PLAY は area 省略・index はhand内インデックス）と、
    どうぐ/エネルギーの場合は toolIndex/energyIndex を辿って解決する。
    解決できない場合は None を返す（呼び出し側はデフォルト扱いにフォールバックすること）。
    """
    if option.cardId is not None:
        return option.cardId

    area = option.area
    if area is None and option.type == OptionType.PLAY:
        area = AreaType.HAND

    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if area is None or option.index is None or not (0 <= player_index < len(state.players)):
        return None

    zone = _zone_entries(area, state.players[player_index], state)
    if zone is None or not (0 <= option.index < len(zone)):
        return None
    target = zone[option.index]
    if target is None:
        return None

    if option.toolIndex is not None:
        tools = getattr(target, "tools", None)
        if tools is None or not (0 <= option.toolIndex < len(tools)):
            return None
        return tools[option.toolIndex].id
    if option.energyIndex is not None:
        energy_cards = getattr(target, "energyCards", None)
        if energy_cards is None or not (0 <= option.energyIndex < len(energy_cards)):
            return None
        return energy_cards[option.energyIndex].id

    return target.id


def resolve_pokemon(option: Option, state: State) -> Pokemon | None:
    """Option の area/index が指す場のポケモン（ACTIVE/BENCH）を返す。

    ダメージ対象・回復対象・交代先など、card_id ではなく hp/maxHp など Pokemon の状態
    そのものが必要な場面で使う（action_selection/handlers 配下から使われる想定）。
    ACTIVE/BENCH 以外の area、または範囲外の index の場合は None。
    """
    if option.area not in (AreaType.ACTIVE, AreaType.BENCH) or option.index is None:
        return None
    player_index = option.playerIndex if option.playerIndex is not None else state.yourIndex
    if not (0 <= player_index < len(state.players)):
        return None
    player = state.players[player_index]
    zone = player.active if option.area == AreaType.ACTIVE else player.bench
    if not (0 <= option.index < len(zone)):
        return None
    return zone[option.index]


def _zone_entries(area: AreaType, player: PlayerState, state: State) -> list | None:
    """area に対応するカード/ポケモンのリストを返す（存在しない/非公開なら None）。"""
    if area == AreaType.LOOKING:
        return state.looking
    if area == AreaType.STADIUM:
        return state.stadium
    attr = _AREA_TO_PLAYER_ZONE.get(area)
    if attr is None:
        # DECK（非公開）/ PRE_EVOLUTION・PLAYER・ENERGY・TOOL（親ポケモン側から辿るべき情報）は
        # ここでは解決しない。
        return None
    return getattr(player, attr, None)


def pick_top(select: SelectData, score_fn: Callable[[Option], float], reverse: bool = True) -> list[int]:
    """select.option を score_fn の降順（reverse=Falseなら昇順）で並べ、
    minCount 以上 maxCount 以下になるよう上位から選択肢インデックスを返す汎用ヘルパー。
    """
    ranked = sorted(range(len(select.option)), key=lambda i: score_fn(select.option[i]), reverse=reverse)
    count = max(select.minCount, min(select.maxCount, len(ranked)))
    return ranked[:count]
