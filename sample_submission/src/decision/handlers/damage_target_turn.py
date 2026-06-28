from cg.api import Observation

from src.decision.fallback import choose_random_legal_action


def choose_damage_target_action(obs: Observation) -> list[int]:
    """ダメージ・ダメカン・回復対象を選ぶ。"""
    # TODO:
    # - DAMAGE / DAMAGE_COUNTER 系で、きぜつを取れる相手や主力を止められる相手を優先する
    # - HEAL / REMOVE_DAMAGE_COUNTER で、自分の主力や次ターンも使いたいポケモンを守る
    # - EFFECT_TARGET で、現在の盤面計画に最も合う対象を選ぶ
    return choose_random_legal_action(obs)
