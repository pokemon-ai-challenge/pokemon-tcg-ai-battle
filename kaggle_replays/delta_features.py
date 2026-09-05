"""Phase15 §35-1/2: Action-conditioned delta の**凍結スキーマ**と抽出。

Phase14 P6 が使った 23 次元をそのまま正式化する(推測で足さない・引かない)。
    dim 0..21 : f(s')[i] - f(s)[i]   … `_entity_extract.SUMMARY_KEYS` と同じ並び
    dim 22    : valid flag(1step 実行に成功したら 1.0、失敗なら 0 で全次元 0)

正規化はPhase14と同一の /20.0。turn だけは意味が違う(段差 0/1)ので同じ扱いのまま残す
(Phase14 の効果を再現するのが今回の Gate A であり、ここを変えると比較にならない)。

deterministic / stochastic の事前分類(§8):
    counts(手札枚数・山札・トラッシュ・サイド・ベンチ数・エネ数)は
      行動ルールから枚数が決まるため **count としては deterministic**。
      「引いたカードの中身」は stochastic だがこの 22 次元は中身を持たない。
    HP 系は自分の行動由来なら deterministic、相手の隠れ状態に依存する解決では stochastic。
    実際にどれが揺れるかは M 個の determinization 間分散で**実測**する(§18.2)。
    結果を見てから feature を選び直すことはしない(§32)。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _entity_extract import SUMMARY_KEYS  # noqa: E402

N_BASE = len(SUMMARY_KEYS)          # 22
N_DELTA = N_BASE + 1                # 23(+ valid flag)
SCALE = 20.0

# 事前分類(結果を見る前に固定)。"count" は枚数として確定、"resolve" は解決依存。
KIND = {
    "self_active_hp": "resolve", "self_active_dmg": "resolve",
    "self_active_energy": "count", "self_active_tools": "count",
    "self_bench_count": "count", "self_bench_hp_total": "resolve",
    "self_bench_energy_total": "count", "self_hand_count": "count",
    "self_deck": "count", "self_discard": "count", "self_prize": "count",
    "opp_active_hp": "resolve", "opp_active_dmg": "resolve",
    "opp_active_energy": "count", "opp_bench_count": "count",
    "opp_bench_hp_total": "resolve", "opp_deck": "count",
    "opp_discard": "count", "opp_prize": "count",
    "self_energy_board": "count", "opp_energy_board": "count", "turn": "count",
}
DETERMINISTIC_DIMS = tuple(i for i, k in enumerate(SUMMARY_KEYS) if KIND[k] == "count")


def schema() -> list[dict]:
    """§34-B 用の全次元定義。"""
    rows = [{"idx": i, "feature": k, "kind": KIND[k], "normalization": f"/{SCALE}"}
            for i, k in enumerate(SUMMARY_KEYS)]
    rows.append({"idx": N_BASE, "feature": "valid_flag", "kind": "meta",
                 "normalization": "0/1"})
    return rows


def single(before, after):
    """D1: 1 determinization の実現 delta。after が無ければ全 0 + flag 0。"""
    if after is None:
        return [0.0] * N_DELTA
    return [(a - b) / SCALE for a, b in zip(after, before)] + [1.0]


def expected(before, afters):
    """D2: M 個の determinization の平均 delta。"""
    ok = [a for a in afters if a is not None]
    if not ok:
        return [0.0] * N_DELTA
    return [statistics.mean((a[i] - before[i]) / SCALE for a in ok)
            for i in range(N_BASE)] + [1.0]


def std(before, afters):
    """D3 用の determinization 間 std(delta の不確実性)。"""
    ok = [a for a in afters if a is not None]
    if len(ok) < 2:
        return [0.0] * N_BASE
    return [statistics.pstdev([(a[i] - before[i]) / SCALE for a in ok])
            for i in range(N_BASE)]


def next_state(after, before=None):
    """D4 診断: 差分ではなく次状態そのもの(スケールだけ合わせる)。"""
    if after is None:
        return [0.0] * N_DELTA
    return [a / SCALE for a in after] + [1.0]


def expected_next_state(afters):
    ok = [a for a in afters if a is not None]
    if not ok:
        return [0.0] * N_DELTA
    return [statistics.mean(a[i] for a in ok) / SCALE for i in range(N_BASE)] + [1.0]


def deterministic_only(before, afters):
    """Ddet: 事前に count と分類した次元だけを残し、他は 0 にする。"""
    e = expected(before, afters)
    out = [0.0] * N_DELTA
    for i in DETERMINISTIC_DIMS:
        out[i] = e[i]
    out[N_BASE] = e[N_BASE]
    return out


def variance_scalar(before, afters) -> float:
    """候補ごとの determinization 間ばらつき(§18.2 の分類キー)。"""
    s = std(before, afters)
    return sum(x * x for x in s) ** 0.5


def pair_distance(d1, d2) -> float:
    """§18.1 候補間の delta 距離(候補が区別できているか)。"""
    return sum(abs(a - b) for a, b in zip(d1[:N_BASE], d2[:N_BASE]))
