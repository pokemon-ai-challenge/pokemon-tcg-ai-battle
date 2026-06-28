from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_card_move_action(obs: Observation) -> list[int]:
    """カードの移動先や移動対象を選ぶ。"""
    # TODO:
    # - TO_BENCH / TO_FIELD で、出したいポケモンや置き先を選ぶ
    # - TO_HAND / TO_DECK / TO_DECK_BOTTOM / TO_PRIZE で、戻す価値の低い札を選ぶ
    # - DISCARD で、コストとして失っても痛くない札を優先して選ぶ
    # - LOOK / NOT_MOVE で、公開情報や次ターン計画に合う選択をする
    return choose_random_legal_action(obs)
