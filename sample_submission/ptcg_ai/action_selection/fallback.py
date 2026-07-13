"""クラスタ① 選択振り分け／担当B

router.py が対応できない場面（未知の SelectContext、handlers 側で例外が出た場合など）の
最終フォールバック。ゲームを止めないことを最優先し、合法手の範囲で無難な選択を返す。

CLAUDE.md / cg/api.py の注記: SelectContext 等の Enum はコンペ期間中に要素が追加され得る。
このモジュールは、その「未知の context」に対する安全網として機能する。
"""

from cg.api import Observation


def safe_choice(obs: Observation) -> list[int]:
    """obs.select.minCount を満たす最小限の合法手を返す。

    先頭から minCount 個を選ぶだけの単純な実装。cg/api.py の保証
    （0 <= minCount <= maxCount <= len(option)）により、常に合法手になる。
    「何が最善か」を判断する材料が無い/信頼できない場面の最終防衛ラインなので、
    複雑な評価はせず、まず合法手を返してゲームを止めないことを優先する。
    """
    select = obs.select
    count = min(max(select.minCount, 0), len(select.option))
    return list(range(count))
