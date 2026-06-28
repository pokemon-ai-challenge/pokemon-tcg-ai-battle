from cg.api import Observation

from src.decision.fallback import choose_random_legal_action
from src.decision.main_turn_parts.priorities.attack import choose_best_attack_option


def choose_attack_action(obs: Observation) -> list[int]:
    """ATTACK 文脈では、共通の攻撃評価をそのまま使う。"""
    if obs.select is None:
        raise ValueError("obs.select must not be None in attack phase.")

    # MAIN 中の attack 候補評価と同じ基準にそろえて、挙動をぶらさない。
    best_option_index = choose_best_attack_option(obs)
    if best_option_index is not None:
        return [best_option_index]

    # 念のため攻撃候補が取れなかったときだけ既存フォールバックへ落とす。
    return choose_random_legal_action(obs)
