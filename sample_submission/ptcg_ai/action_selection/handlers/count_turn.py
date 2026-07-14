"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: DRAW_COUNT, DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT

「何枚/いくつ選ぶか」という数値選択（SelectType.COUNT、OptionType.NUMBER）。
基本的には SelectData.minCount/maxCount の範囲で、効果を最大化する数値を選ぶ。
"""

from cg.api import Observation


def handle(obs: Observation) -> list[int]:
    """COUNT 系の選択肢から、最も効果が大きい数値を選んで返す。"""
    raise NotImplementedError
