from cg.api import Observation

from src.decision.fallback import choose_random_legal_action
from src.decision.evaluation.switch_eval import choose_best_switch_option


def choose_switch_action(obs: Observation) -> list[int]:
    """入れ替え系の選択で、前に出す価値が最も高いポケモンを選ぶ。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None during switch decisions.")

    option_indices = list(range(len(obs.select.option)))
    if not option_indices:
        return choose_random_legal_action(obs)

    # SWITCH / TO_ACTIVE では、退却評価と同じ物差しで前衛候補を比較する。
    best_option = choose_best_switch_option(obs, option_indices)
    if best_option.should_offer:
        return [best_option.option_index]

    return choose_random_legal_action(obs)
