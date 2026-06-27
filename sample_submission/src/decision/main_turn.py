from cg.api import Observation

from src.decision.fallback import choose_random_legal_action
from src.decision.main_turn_parts.buckets import bucket_main_options
from src.decision.main_turn_parts.priorities import (
    propose_ability_action,
    propose_attack_action,
    propose_board_item_action,
    propose_draw_or_search_action,
    propose_end_action,
    propose_energy_action,
    propose_pokemon_or_evolve_action,
    propose_retreat_action,
)
from src.decision.main_turn_parts.proposals import choose_best_proposal


def choose_main_action(obs: Observation) -> list[int]:
    """メインフェーズで取る行動を、仮の重みづけで選ぶ。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None in main phase.")
    if obs.current is None:
        raise ValueError("obs.current must not be None in main phase.")

    buckets = bucket_main_options(obs)

    # 各担当の処理は「今やりたい行動候補」と重みを返す。
    proposals = [
        propose_draw_or_search_action(obs, buckets),
        propose_pokemon_or_evolve_action(obs, buckets),
        propose_board_item_action(obs, buckets),
        propose_ability_action(obs, buckets),
        propose_energy_action(obs, buckets),
        propose_retreat_action(obs, buckets),
        propose_attack_action(obs, buckets),
        propose_end_action(obs, buckets),
    ]

    best = choose_best_proposal(proposals)
    if best is not None:
        return best.action

    return choose_random_legal_action(obs)
