"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: DISCARD, DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD,
DISCARD_CARD_OR_ATTACHED_CARD, DISCARD_ENERGY

捨て札にするカード/エネルギー/どうぐを選ぶ。
knowledge.profile_registry.get_deck_plan().protected_card_ids（捨てたくないカードの集合）を
参照し、それらを候補から除外することを優先する。
改造ハンマー等で相手の特殊エネルギーを壊す場合は、ワザの効果を無効化する
ミストエネルギー系（get_opponent_effect_lock_energy_ids）を最優先で壊す。
「進化ペアになる手札は捨て候補から除外する」といった汎用ルールもここに置く。
"""

from cg.api import SelectData, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry

# 破壊/ディスカード対象のスコア（大きいほど優先して選ぶ）。
_SCORE_EFFECT_LOCK = 2  # ミストエネ等（相手のワザ効果無効化）＝最優先で壊す
_SCORE_NORMAL = 1
_SCORE_PROTECTED = 0  # 自分の捨てたくないカード＝最後


def choose(select: SelectData, state: State) -> list[int]:
    """捨てる/壊すカード・エネルギー・どうぐの選択肢インデックスを返す。"""
    return common.pick_top(select, lambda option: _discard_score(option, state))


def _discard_score(option, state: State) -> int:
    """優先度スコア。ミストエネ系は最優先で壊し、自分の保護カードは最後に回す。"""
    card_id = common.resolve_card_id(option, state)
    if card_id is not None and card_id in profile_registry.get_opponent_effect_lock_energy_ids():
        return _SCORE_EFFECT_LOCK
    if _is_protected(option, state):
        return _SCORE_PROTECTED
    return _SCORE_NORMAL


def _is_protected(option, state: State) -> bool:
    """get_deck_plan().protected_card_ids を参照し、捨てたくないカードかどうかを判定する。"""
    card_id = common.resolve_card_id(option, state)
    if card_id is None:
        return False
    return card_id in profile_registry.get_deck_plan().protected_card_ids
