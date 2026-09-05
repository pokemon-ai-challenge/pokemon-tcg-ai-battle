"""Phase13: L1/L2 の純粋ロジック(engine 非依存)。単体テスト対象。

`_l1l2_branch.py`(収集)と `_l1l2_analyze.py`(集計)が同じ規則を使うことを保証する。
"""
from __future__ import annotations

import statistics

TEACHER_TIE = 0.005


def outcome_of(terminal: int | None, me: int) -> float | None:
    """terminal result -> 自分視点の勝敗。win=1.0 / loss=0.0 / それ以外(引き分け等)=0.5。"""
    if terminal is None:
        return None
    if terminal == me:
        return 1.0
    if terminal == 1 - me:
        return 0.0
    return 0.5


def winner(delta: float | None, eps: float) -> str | None:
    """delta = S3 - Q0。eps 以内は tie(§13、eps は結果を見る前に固定)。"""
    if delta is None:
        return None
    if delta > eps:
        return "S3"
    if delta < -eps:
        return "Q0"
    return "tie"


def teacher_vote(diffs, tie: float = TEACHER_TIE, n_block: int | None = None) -> dict:
    """block ごとの (Q0手 - S3手) 差 -> Q0 支持率と安定性クラス。

    Phase11D/12 と同一規則: vote は 1 / 0.5(同点) / 0、cls は全 block 一致で stable。
    """
    n = n_block or len(diffs)
    votes = [0.5 if abs(d) < tie else (1.0 if d > 0 else 0.0) for d in diffs]
    support = statistics.mean(votes)
    signs = [0 if abs(d) < tie else (1 if d > 0 else -1) for d in diffs]
    nz = [s for s in signs if s != 0]
    if not nz:
        cls = "always_tie"
    else:
        agree = max(nz.count(1), nz.count(-1))
        cls = ("stable" if agree == n else "mostly" if agree == n - 1 else "unstable")
    return {"support_q0": support, "confidence": 2 * abs(support - 0.5),
            "mean_margin": statistics.mean(diffs), "cls": cls}


def teacher_side(support_q0: float) -> str:
    if support_q0 > 0.5:
        return "supports_Q0"
    if support_q0 < 0.5:
        return "supports_S3"
    return "near_tie"
