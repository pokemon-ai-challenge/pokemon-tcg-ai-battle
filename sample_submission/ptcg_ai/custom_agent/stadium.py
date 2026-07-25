"""スタジアムの使用条件。"""

from __future__ import annotations

from cg.api import State

from ptcg_ai.custom_agent import constants


def is_usable(card_id: int, state: State) -> bool:
    if card_id == constants.BATTLE_COLOSSEUM:
        current = state.stadium[0].id if state.stadium else None
        return current != constants.BATTLE_COLOSSEUM
    return True
