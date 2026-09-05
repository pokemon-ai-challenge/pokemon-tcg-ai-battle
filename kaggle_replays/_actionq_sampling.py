"""Phase9B: MAIN decision group の層化抽出(quota ベースのオンライン受理)。

Phase9A は `max_per_game=4` で試合の**先頭から**取ったため 124/128 が序盤になった。
ここでは層ごとに quota を持ち、**quota が残っている層だけ受理**する。
序盤 quota は早く埋まるので、以降は中盤・終盤だけが受理され、先着順バイアスが消える。

層化キーは **教師生成前に分かる情報だけ**を使う(§4.6):
    turn 帯 / candidate 数帯 / architecture / action type
teacher score・spread・順位・モデル予測は使わない(「学びやすい局面」の選別禁止)。
"""
from __future__ import annotations

from collections import Counter


def turn_band(turn: int) -> str:
    if turn <= 5:
        return "early"
    if turn <= 10:
        return "middle"
    return "late"


def cand_band(n: int) -> str:
    if n <= 4:
        return "small"      # 2-4
    if n <= 7:
        return "medium"     # 5-7
    return "large"          # 8+


class StratifiedQuota:
    """ターン帯を主 quota、候補数帯・アーキタイプを従 quota として受理判定する。

    主 quota(ターン帯)は厳格に守る。従 quota は「すでに大幅超過している層」だけを
    弾くソフト制約にする(実ゲーム分布に無いものを無理に作らないため §4.3)。
    """

    def __init__(self, target: int,
                 turn_mix=(("early", 0.35), ("middle", 0.35), ("late", 0.30)),
                 cand_soft_cap: float = 0.60,
                 arch_soft_cap: float = 0.40,
                 per_game_cap: int = 12):
        self.target = target
        self.turn_quota = {k: int(target * v) for k, v in turn_mix}
        # int() の切り捨てで quota 合計が target 未満になると `total >= target` が永久に
        # 成立せず、収集完了後も max_games まで空回りする。端数は最大の層へ足す。
        short = target - sum(self.turn_quota.values())
        if short > 0:
            self.turn_quota[max(self.turn_quota, key=lambda k: self.turn_quota[k])] += short
        self.turn_count: Counter = Counter()
        self.cand_count: Counter = Counter()
        self.arch_count: Counter = Counter()
        self.cand_soft_cap = cand_soft_cap
        self.arch_soft_cap = arch_soft_cap
        self.per_game_cap = per_game_cap
        self.total = 0
        self.rejected: Counter = Counter()

    def _soft_ok(self, counter: Counter, key: str, cap: float) -> bool:
        if self.total < 50:          # 立ち上がりは制約しない(分母が小さすぎる)
            return True
        return (counter[key] + 1) / (self.total + 1) <= cap

    def accept(self, turn: int, n_cands: int, arch: str, game_count: int) -> bool:
        if self.total >= self.target:
            self.rejected["target_reached"] += 1
            return False
        if game_count >= self.per_game_cap:
            self.rejected["per_game_cap"] += 1
            return False
        tb = turn_band(turn)
        if self.turn_count[tb] >= self.turn_quota.get(tb, 0):
            self.rejected[f"turn_quota_{tb}"] += 1
            return False
        cb = cand_band(n_cands)
        if not self._soft_ok(self.cand_count, cb, self.cand_soft_cap):
            self.rejected[f"cand_soft_{cb}"] += 1
            return False
        if not self._soft_ok(self.arch_count, arch, self.arch_soft_cap):
            self.rejected[f"arch_soft_{arch}"] += 1
            return False
        return True

    def commit(self, turn: int, n_cands: int, arch: str) -> None:
        self.turn_count[turn_band(turn)] += 1
        self.cand_count[cand_band(n_cands)] += 1
        self.arch_count[arch] += 1
        self.total += 1

    def remaining_turn_quota(self) -> dict:
        return {k: v - self.turn_count[k] for k, v in self.turn_quota.items()}

    def report(self) -> dict:
        t = max(1, self.total)
        return {
            "total": self.total, "target": self.target,
            "turn_quota": dict(self.turn_quota),
            "turn_count": dict(self.turn_count),
            "turn_frac": {k: round(v / t, 4) for k, v in self.turn_count.items()},
            "cand_count": dict(self.cand_count),
            "cand_frac": {k: round(v / t, 4) for k, v in self.cand_count.items()},
            "arch_count": dict(self.arch_count),
            "arch_frac": {k: round(v / t, 4) for k, v in self.arch_count.items()},
            "rejected": dict(self.rejected.most_common(12)),
        }
