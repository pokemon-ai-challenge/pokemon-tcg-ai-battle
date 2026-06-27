from cg.api import Observation, OptionType

from src.decision.fallback import choose_random_legal_action
from src.knowledge.card_cache import load_attack_data


def choose_attack_action(obs: Observation) -> list[int]:
    """Choose the usable attack with the highest printed damage."""
    if obs.select is None:
        raise ValueError("obs.select must not be None in attack phase.")

    attack_damage_by_id = {
        attack.attackId: attack.damage for attack in load_attack_data()
    }

    best_option_index: int | None = None
    best_damage = -1

    for option_index, option in enumerate(obs.select.option):
        if option.type != OptionType.ATTACK or option.attackId is None:
            continue

        damage = attack_damage_by_id.get(option.attackId, -1)
        if damage > best_damage:
            best_damage = damage
            best_option_index = option_index

    if best_option_index is not None:
        return [best_option_index]

    return choose_random_legal_action(obs)
