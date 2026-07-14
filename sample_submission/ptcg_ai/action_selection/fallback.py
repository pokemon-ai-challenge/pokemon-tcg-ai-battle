"""クラスタ① 選択振り分け／担当B

router.py が対応できない場面（未知の SelectContext、handlers 側で例外が出た場合など）の
最終フォールバック。ゲームを止めないことを最優先し、合法手の範囲で無難な選択を返す。

CLAUDE.md / cg/api.py の注記: SelectContext 等の Enum はコンペ期間中に要素が追加され得る。
このモジュールは、その「未知の context」に対する安全網として機能する。
"""

from cg.api import Observation


def safe_choice(obs: Observation) -> list[int]:
    """obs.select.minCount を満たす最小限の合法手を返す。

    方針の目安（実装時に決める）:
        - option を先頭から minCount 個選ぶなど、常に合法手の範囲に収まる選び方にする
        - 可能なら「効果が薄い/安全側」の選択肢を優先する
    """
    raise NotImplementedError
