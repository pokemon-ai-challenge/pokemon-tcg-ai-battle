"""リスク調整Determinization（EXP-A44）のリスク集約関数。

``docs/plans/search/EXP-A44-risk-adjusted-determinization.md`` §5.4 の実装。決定化サンプル
（隠し情報を1通りに固定した局面）ごとの勝率推定値のリストを、1つのスコアへ集約する。
外部ライブラリは使わず、標準ライブラリの ``statistics`` / ``math`` のみで完結させる
（``learning/value_model.py`` の「numpy/torch 非依存」方針と同じ思想）。

``risk_determinization.py``（探索本体）から呼ばれる想定で、本体側のゲート判定・候補生成・
Value呼び出しとは独立にテストできるよう、集約ロジックだけを本モジュールへ切り出した
（要件書の phase 分割: P1 集約関数 → P3 探索層、に対応する構成上の判断。要件書は集約関数を
どのファイルに置くか明示していないため、独立コミットにしやすいこの分割を採用した）。
"""

from __future__ import annotations

import math
import statistics

# aggregate() が受け付ける mode の一覧。risk_determinization.py の config 検証でも使う。
MODES = ("mean", "mean_std", "cvar", "adaptive")

# adaptive モードの局面区分（サイド差 = 自分の残サイド - 相手の残サイド）。
# 優勢: 自分の残サイドが相手より2以上少ない（＝2枚以上多く取っている）。
_ADAPTIVE_ADVANTAGE_MARGIN = 2


def aggregate(
    values: list[float],
    mode: str = "mean",
    *,
    beta: float = 1.0,
    alpha: float = 0.3,
    prize_diff: int | None = None,
) -> float:
    """決定化サンプルごとの勝率推定 ``values`` を1つのスコアへ集約する。

    Args:
        values: 各決定化サンプルでの候補手の評価値（例: ``ValueModel.predict_win_prob_from_state``
            の出力）。空リストは呼び出し側の誤り（ゲート済みであれば起こらない想定）だが、
            安全側に倒して ``0.0`` を返す。
        mode: ``"mean"`` / ``"mean_std"`` / ``"cvar"`` / ``"adaptive"``。
        beta: ``mean_std`` の標準偏差ペナルティ係数（``mean - beta * pstdev``）。
        alpha: ``cvar`` の下側分位（下位 ``ceil(alpha * N)`` 件、最低1件）。
        prize_diff: ``adaptive`` モード専用。自分の残サイド数 − 相手の残サイド数
            （負なら自分が優勢）。``None`` の場合は互角として扱う（安全なデフォルト）。

    Returns:
        float: 集約スコア。

    Raises:
        ValueError: 未知の ``mode`` を渡した場合。
    """
    if not values:
        return 0.0
    if mode not in MODES:
        raise ValueError(f"未知の aggregate mode: {mode!r}")

    if mode == "mean":
        return _mean(values)
    if mode == "mean_std":
        return _mean_std(values, beta)
    if mode == "cvar":
        return _cvar(values, alpha)
    # mode == "adaptive"
    return _adaptive(values, alpha, prize_diff)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _mean_std(values: list[float], beta: float) -> float:
    """``mean - beta * pstdev``。``pstdev`` は母標準偏差（N=1 のときは0）。"""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean
    return mean - beta * statistics.pstdev(values)


def _lower_tail_mean(values: list[float], alpha: float) -> float:
    """昇順ソートした下位 ``ceil(alpha * N)`` 件（最低1件）の平均。"""
    n = len(values)
    tail_count = max(1, math.ceil(alpha * n))
    tail_count = min(tail_count, n)
    ordered = sorted(values)
    return statistics.fmean(ordered[:tail_count])


def _upper_tail_mean(values: list[float], alpha: float) -> float:
    """降順（上位側）の ``ceil(alpha * N)`` 件（最低1件）の平均。"""
    n = len(values)
    tail_count = max(1, math.ceil(alpha * n))
    tail_count = min(tail_count, n)
    ordered = sorted(values, reverse=True)
    return statistics.fmean(ordered[:tail_count])


def _cvar(values: list[float], alpha: float) -> float:
    return _lower_tail_mean(values, alpha)


def _adaptive(values: list[float], alpha: float, prize_diff: int | None) -> float:
    """局面適応（要件書 §5.4）。

    - 優勢（``prize_diff <= -_ADAPTIVE_ADVANTAGE_MARGIN``、自分の残サイドが相手より2以上少ない）:
      ``cvar``（下側裾を重視し、事故負けを避ける）。
    - 劣勢（``prize_diff >= _ADAPTIVE_ADVANTAGE_MARGIN``）: 上位 ``alpha`` 分位の平均
      （逆転の目がある手を評価するため、上振れを重視する）。
    - 互角（それ以外、``prize_diff is None`` を含む）: ``0.5 * mean + 0.5 * cvar``。
    """
    if prize_diff is not None and prize_diff <= -_ADAPTIVE_ADVANTAGE_MARGIN:
        return _cvar(values, alpha)
    if prize_diff is not None and prize_diff >= _ADAPTIVE_ADVANTAGE_MARGIN:
        return _upper_tail_mean(values, alpha)
    return 0.5 * _mean(values) + 0.5 * _cvar(values, alpha)
