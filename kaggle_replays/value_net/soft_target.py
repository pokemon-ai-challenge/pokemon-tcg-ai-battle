"""Phase12: block-support soft target(Search Distillation v0)の target/confidence/Loss。

Phase11D で確定した事実:
    大 margin pair は 95% stable、near-tie pair は 52% が always_tie。
にもかかわらず従来 pairwise Loss は両者を同じ 1 問として扱っていた。ここではその修正だけを行う。

**名称の厳密さ**: p_ij は「P(i>j) の真値」ではなく **block-support target** と呼ぶ。
独立 search block が i を支持した割合にすぎない。

vote(§5):   i>j → 1.0 / tie → 0.5 / i<j → 0.0
p_ij     =  mean(vote_A..vote_D)
confidence = 2*|p-0.5|                                            (§6)
margin_weight = clip(|mean_i - mean_j| / 0.07, 0, 1)              (§7)

margin 境界 0.01/0.03/0.07 は Phase11D 由来の固定値。test 結果を見て変更しない(§22)。
"""
from __future__ import annotations

import itertools
import statistics

TIE = 0.005          # Phase11D と同一(モデル結果を見る前に固定)
NB = 4
MARGIN_BOUNDS = (0.01, 0.03, 0.07)
MARGIN_SCALE = 0.07


def vote(d: float, tie: float = TIE) -> float:
    """1 block 分の投票。|d| < tie は同点として 0.5。"""
    if abs(d) < tie:
        return 0.5
    return 1.0 if d > 0 else 0.0


def block_support(diffs, tie: float = TIE) -> float:
    """block ごとの差分列 → block-support target p_ij ∈ [0,1]。"""
    return statistics.mean(vote(d, tie) for d in diffs)


def confidence(p: float) -> float:
    """支持率 → confidence(§6)。p=1/0→1.0, p=.75/.25→0.5, p=.5→0.0。"""
    return 2.0 * abs(p - 0.5)


def margin_weight(m: float, scale: float = MARGIN_SCALE) -> float:
    a = abs(m) / scale
    return 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)


def margin_band(m: float) -> str:
    a = abs(m)
    lo, mid, hi = MARGIN_BOUNDS
    return ("very_small" if a < lo else "small" if a < mid else
            "medium" if a < hi else "large")


def pair_class(diffs, tie: float = TIE) -> str:
    """Phase11D と同一の stable / mostly / unstable / always_tie 分類。"""
    signs = [0 if abs(d) < tie else (1 if d > 0 else -1) for d in diffs]
    nz = [s for s in signs if s != 0]
    if not nz:
        return "always_tie"
    agree = max(nz.count(1), nz.count(-1))
    n = len(diffs)
    return "stable" if agree == n else "mostly" if agree == n - 1 else "unstable"


def group_pairs(group: dict, n_blocks: int = NB, tie: float = TIE) -> list[dict]:
    """group -> 候補ペアごとの target/confidence/margin/分類。

    `blocks` は候補ごとの長さ n_blocks のスコア列。i<j の順序で 1 度だけ列挙する。
    """
    cands = group["candidates"]
    means = [statistics.mean(c["blocks"][:n_blocks]) for c in cands]
    out = []
    for i, j in itertools.combinations(range(len(cands)), 2):
        diffs = [cands[i]["blocks"][b] - cands[j]["blocks"][b] for b in range(n_blocks)]
        p = block_support(diffs, tie)
        m = means[i] - means[j]
        out.append({
            "i": i, "j": j,
            "support": p,
            "confidence": confidence(p),
            "mean_margin": m,
            "abs_margin": abs(m),
            "margin_weight": margin_weight(m),
            "margin_band": margin_band(m),
            "cls": pair_class(diffs, tie),
            "diffs": diffs,
        })
    return out


# ---------------- arm ごとの target / weight (§8) ----------------

def arm_target(pair: dict, arm: str, block_a: int = 0):
    """(target, weight) を返す。weight=0 は「その pair を学習に使わない」。

    S0: 単一 block(=従来の単一 teacher)による hard ranking。tie は除外。
    S1: 4block 多数決。4/4・3/4 のみ hard 化し、2/2 は除外。
    S2: soft block-support。全 pair。
    S3: soft block-support × confidence 重み(本命)。
    S4: S3 × sqrt(margin_weight)(S3 が有望な場合のみ)。
    """
    p = pair["support"]
    if arm == "S0":
        d = pair["diffs"][block_a]
        if abs(d) < TIE:
            return 0.0, 0.0
        return (1.0 if d > 0 else 0.0), 1.0
    if arm == "S1":
        if p >= 0.75:
            return 1.0, 1.0
        if p <= 0.25:
            return 0.0, 1.0
        return 0.5, 0.0
    if arm == "S2":
        return p, 1.0
    if arm == "S3":
        return p, pair["confidence"]
    if arm == "S4":
        return p, pair["confidence"] * (pair["margin_weight"] ** 0.5)
    raise ValueError(arm)
