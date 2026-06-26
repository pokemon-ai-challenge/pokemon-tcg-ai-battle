from functools import lru_cache
from cg.api import all_card_data, all_attack, CardData, Attack


@lru_cache(maxsize=1)
def get_card_db() -> dict[int, CardData]:
    """card_id -> CardData の辞書を返す（初回のみ構築）。"""
    return {card.cardId: card for card in all_card_data()}


@lru_cache(maxsize=1)
def get_attack_db() -> dict[int, Attack]:
    """attack_id -> Attack の辞書を返す（初回のみ構築）。"""
    return {atk.attackId: atk for atk in all_attack()}
