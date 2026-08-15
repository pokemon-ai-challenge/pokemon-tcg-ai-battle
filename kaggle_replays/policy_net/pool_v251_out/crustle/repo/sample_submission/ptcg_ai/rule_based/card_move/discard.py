"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: DISCARD, DISCARD_ENERGY_CARD, DISCARD_TOOL_CARD,
DISCARD_CARD_OR_ATTACHED_CARD, DISCARD_ENERGY

捨て札にするカード/エネルギー/どうぐを選ぶ。
knowledge.profile_registry.get_deck_plan().protected_card_ids（捨てたくないカードの集合）を
参照し、それらを候補から除外することを優先する。
「進化ペアになる手札は捨て候補から除外する」といった汎用ルールもここに置く。
"""

from cg.api import SelectData, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """捨てるカード/エネルギー/どうぐの選択肢インデックスを返す。"""
    return common.pick_top(select, lambda option: 0 if _is_protected(option, state) else 1)


def _is_protected(option, state: State) -> bool:
    """get_deck_plan().protected_card_ids を参照し、捨てたくないカードかどうかを判定する。"""
    card_id = common.resolve_card_id(option, state)
    if card_id is None:
        return False
    return card_id in profile_registry.get_deck_plan().protected_card_ids
