import sys


def trace(turn: int, context: str, selected: int | list[int], reason: str) -> None:
    """選択内容をターン・コンテキスト・理由つきで stderr に出力する。"""
    print(f"[TRACE] turn={turn} ctx={context} sel={selected} reason={reason}", file=sys.stderr)
