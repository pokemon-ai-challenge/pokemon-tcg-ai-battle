from functools import lru_cache

from cg.api import all_attack, all_card_data


@lru_cache(maxsize=1)
def load_card_data():
    """Shared card database cache for heuristics modules."""
    return all_card_data()


@lru_cache(maxsize=1)
def load_attack_data():
    """Shared attack database cache for heuristics modules."""
    return all_attack()
