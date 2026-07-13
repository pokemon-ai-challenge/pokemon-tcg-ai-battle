"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: 他の handlers に割り当てられていない YES_NO 系 context のフォールバック
（例: 今後 SelectContext に YES_NO 系の要素が追加された場合の受け皿）。

cg/api.py の注記どおり SelectContext は競技期間中に要素が追加され得るため、router.py の
対応表に無い YES_NO type の選択肢はここに流れてくる想定。判断が難しい場合は
decision.fallback.safe_choice に委譲してよい。
"""

from cg.api import Observation

from ptcg_ai.action_selection import fallback


def handle(obs: Observation) -> list[int]:
    """未分類の YES_NO 系選択肢を処理する。判断できない場合は fallback に委譲する。"""
    return fallback.safe_choice(obs)
