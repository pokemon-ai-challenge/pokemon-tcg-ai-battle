"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: NOT_MOVE, EFFECT_TARGET

「動かさずその場に残すカード」の選択（例: 何枚か選んで残りは移動、のうち残す側）、および
汎用的な効果対象（EFFECT_TARGET）の選択。damage_target_turn.py がカバーしない
一般的な効果対象選択の受け皿としてここに置く。
"""

from cg.api import SelectData, State


def choose(select: SelectData, state: State) -> list[int]:
    """NOT_MOVE / EFFECT_TARGET の選択肢インデックスを返す。"""
    raise NotImplementedError
