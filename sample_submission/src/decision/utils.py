from cg.api import AreaType, Option, State


def get_card_id_from_option(opt: Option, state: State) -> int | None:
    """Option の area/index から card_id を取得する。

    CARD type options は area/index でカード位置を指定する。
    opt.cardId が設定されている場合はそちらを優先する。
    """
    if opt.cardId is not None:
        return opt.cardId
    if opt.area is None or opt.index is None:
        return None

    player_idx = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
    player = state.players[player_idx]

    if opt.area == AreaType.HAND:
        if player.hand and 0 <= opt.index < len(player.hand):
            return player.hand[opt.index].id
    elif opt.area == AreaType.BENCH:
        if 0 <= opt.index < len(player.bench):
            return player.bench[opt.index].id
    elif opt.area == AreaType.ACTIVE:
        if player.active and 0 <= opt.index < len(player.active):
            poke = player.active[opt.index]
            return poke.id if poke else None
    elif opt.area == AreaType.DISCARD:
        if 0 <= opt.index < len(player.discard):
            return player.discard[opt.index].id

    return None


def get_inplay_pokemon_id(opt: Option, state: State) -> int | None:
    """ATTACH/EVOLVE オプションの対象ポケモン（inPlayArea/inPlayIndex）から card_id を取得する。"""
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None

    player = state.players[state.yourIndex]

    if opt.inPlayArea == AreaType.ACTIVE:
        if player.active and 0 <= opt.inPlayIndex < len(player.active):
            poke = player.active[opt.inPlayIndex]
            return poke.id if poke else None
    elif opt.inPlayArea == AreaType.BENCH:
        if 0 <= opt.inPlayIndex < len(player.bench):
            return player.bench[opt.inPlayIndex].id

    return None
