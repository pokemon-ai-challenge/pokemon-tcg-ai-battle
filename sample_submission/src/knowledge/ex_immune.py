from __future__ import annotations
from cg.api import State

# Crustle (345): ability prevents all damage from opponent's EX Pokémon
EX_IMMUNE_CARD_IDS: frozenset[int] = frozenset({345})

_ex_card_ids: set[int] | None = None


def get_ex_card_ids() -> set[int]:
    """all_card_data() から ex ポケモンの card_id セットを返す（初回のみ構築）。"""
    global _ex_card_ids
    if _ex_card_ids is None:
        from cg.api import all_card_data
        _ex_card_ids = {c.cardId for c in all_card_data() if c.name.lower().endswith(' ex')}
    return _ex_card_ids


def is_ex_pokemon_id(card_id: int) -> bool:
    return card_id in get_ex_card_ids()


def opponent_active_blocks_ex(state: State) -> bool:
    """相手のバトルポケモンが ex 攻撃を無効化するアビリティを持つ場合 True。"""
    opp_idx = 1 - state.yourIndex
    for poke in state.players[opp_idx].active:
        if poke and poke.id in EX_IMMUNE_CARD_IDS:
            return True
    return False


def my_active_is_ex(state: State) -> bool:
    """自分のバトルポケモンが ex ポケモンの場合 True。"""
    for poke in state.players[state.yourIndex].active:
        if poke and is_ex_pokemon_id(poke.id):
            return True
    return False


def has_non_ex_on_bench(state: State) -> bool:
    """自分のベンチに非 EX ポケモンが 1 体以上いる場合 True（リトリート先の確認用）。"""
    ex_ids = get_ex_card_ids()
    for poke in state.players[state.yourIndex].bench:
        if poke and poke.id not in ex_ids:
            return True
    return False
