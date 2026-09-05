"""クラスタ② 盤面評価／担当B

Observation/State から UsageContext（グッズ/サポート/スタジアムの使用条件に渡す
盤面スナップショット）を組み立てる。「生の盤面 -> 判定に使える形」への変換という点で
board_features.py などと同じ役割。カードIDそのものはただの整数として素通しするだけで、
特定デッキの意味づけ（このIDが何か）は一切知らない。
"""

from cg.api import CardType, Pokemon, State

from ptcg_ai.shared import card_cache
from ptcg_ai.shared.profile_types import OpponentBenchStatus, UsageContext


def build_usage_context(state: State, your_index: int) -> UsageContext:
    """state と your_index から UsageContext を組み立てる。"""
    player = state.players[your_index]
    opponent = state.players[1 - your_index]

    own_active = player.active[0] if player.active else None
    own_hand_ids = [card.id for card in (player.hand or [])]
    own_bench_ids = [pokemon.id for pokemon in player.bench]
    own_discard_ids = [card.id for card in player.discard]

    opponent_active = opponent.active[0] if opponent.active else None
    opponent_bench = [
        OpponentBenchStatus(card_id=pokemon.id, hp=pokemon.hp) for pokemon in opponent.bench
    ]

    return UsageContext(
        own_hand_ids=own_hand_ids,
        own_active_id=own_active.id if own_active is not None else None,
        own_bench_ids=own_bench_ids,
        own_discard_ids=own_discard_ids,
        own_active_energy_count=len(own_active.energies) if own_active is not None else 0,
        own_discard_pokemon_count=_count_pokemon(own_discard_ids),
        opponent_active_id=opponent_active.id if opponent_active is not None else None,
        opponent_active_hp=opponent_active.hp if opponent_active is not None else None,
        opponent_bench=opponent_bench,
        opponent_active_has_special_energy=_has_special_energy(opponent_active),
        stadium_id=state.stadium[0].id if state.stadium else None,
    )


def _count_pokemon(card_ids: list[int]) -> int:
    return sum(1 for card_id in card_ids if card_cache.get_card(card_id).cardType == CardType.POKEMON)


def _has_special_energy(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return False
    return any(
        card_cache.get_card(energy_card.id).cardType == CardType.SPECIAL_ENERGY
        for energy_card in pokemon.energyCards
    )
