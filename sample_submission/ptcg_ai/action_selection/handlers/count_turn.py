"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: DRAW_COUNT, DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT

「何枚/いくつ選ぶか」という数値選択（SelectType.COUNT、OptionType.NUMBER）。
基本的には SelectData.minCount/maxCount の範囲で、効果を最大化する数値を選ぶ。
"""

from cg.api import Observation

from ptcg_ai.action_selection import fallback


def handle(obs: Observation) -> list[int]:
    """COUNT 系の選択肢から、最も効果が大きい数値を選んで返す。

    DRAW_COUNT/DAMAGE_COUNTER_COUNT/REMOVE_DAMAGE_COUNTER_COUNT はいずれも
    「多いほど自分に有利」な効果なので、既定では option.number が最大のものを選ぶ。
    """
    best_index = None
    best_number = None
    for i, option in enumerate(obs.select.option):
        if option.number is None:
            continue
        if best_number is None or option.number > best_number:
            best_number = option.number
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]
