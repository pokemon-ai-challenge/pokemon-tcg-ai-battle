"""クラスタ③ メイン行動の意思決定／担当B

各カテゴリの「今やるべきか」の提案（priorities/*.py）を集め、スコアで比較して1手を選ぶ骨格。
"""

import os
from dataclasses import dataclass

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts import buckets, weights

# 下準備を攻撃より先に消化する挙動。PTCG_SETUP_BEFORE_ATTACK=0 で従来の
# 「全カテゴリを単純にスコア比較」に戻せる（A/B 比較・緊急時のロールバック用）。
SETUP_BEFORE_ATTACK = os.environ.get("PTCG_SETUP_BEFORE_ATTACK", "1") != "0"


@dataclass
class ActionProposal:
    """1つの候補行動と、その採用スコア。"""

    category: str
    select: list[int]
    score: float
    reason: str


from ptcg_ai.rule_based.main_turn_parts.priorities import ability, attack, board, draw, end_turn, energy, retreat

_PRIORITY_MODULES = (draw, board, ability, energy, retreat, attack, end_turn)

# 「無コストで自己枯渇する下準備」カテゴリ。attack は選ぶとその時点でターンが終わるため、
# これらの下準備（進化・エネルギー付与・ドロー/サーチ）は attack より先に消化しないと、
# 強い攻撃（特に KO 可能な攻撃は score=1000超）に毎回負けて、そのターン中に一度も
# 実行されないまま飛ばされてしまう（例: ノココッチに進化できるのに攻撃を優先して進化しない）。
#
# ここに含めるのは「実行すると自分の選択肢が消える／1ターンに限りがある」カテゴリ:
#   board   … 進化は手札の進化カードを消費し、展開/どうぐも play で手札から消える
#   energy  … State.energyAttached（1ターン1回）で次回以降 propose が None になる
#   draw    … サポーターは State.supporterPlayed、グッズは play で手札から消える
#   ability … このデッキの特性は「サイコドロー（進化時1回）／さかてにとる（1ターン1回）／
#             にげあしドロー（使うと自身が山札へ戻る=消える）」でいずれも枯渇する。
#             特にノココッチのにげあしドロー（進化→3ドロー→自身と付属を山札へ→再度ノコッチを
#             進化させて回すドローエンジン）は、攻撃を選ぶとターンが終わって回せなくなるため、
#             攻撃より先に消化したい中核。
# 万一「1ターンに何度も使える枯渇しない特性」が将来入っても無限ループしないよう、
# decide() 側で1ターンあたりの下準備採用回数に上限（_SETUP_ACTION_CAP）を設ける。
_SELF_DEPLETING_SETUP = ("board", "energy", "draw", "ability")

# 1ターンに下準備を採用できる上限。通常のターンは多くても10手未満なので、これを超える
# 場合は枯渇しない提案が繰り返されている異常とみなし、以降は通常のスコア比較に委ねる
# （＝いずれ attack/end が選ばれてターンが進む）。無限ループ・タイムアウトの安全弁。
_SETUP_ACTION_CAP = 24


@dataclass
class _SetupTurnBudget:
    """直近に下準備を採用したターン番号と、そのターン内での採用回数を持つ状態。

    ターンが変わると自動でリセットされる（turn != self.turn）。1プロセス内で複数ゲームを
    回す自己対戦・テストでは、ゲーム跨ぎでターン番号が偶然一致すると自己修復に頼れないため、
    新規ゲーム開始時に reset_turn_state() で明示リセットする。
    """

    turn: int | None = None
    count: int = 0

    def take(self, turn: int | None, cap: int) -> bool:
        if turn != self.turn:
            self.turn, self.count = turn, 0
        if self.count >= cap:
            return False
        self.count += 1
        return True

    def reset(self) -> None:
        self.turn, self.count = None, 0


_setup_budget = _SetupTurnBudget()


def reset_turn_state() -> None:
    """新規ゲーム開始時（rule_based_agent.agent が obs.select is None を受けた時）に呼ぶ。"""
    _setup_budget.reset()


def collect_proposals(obs: Observation) -> list[ActionProposal]:
    """各 priorities/*.py の propose() を呼び、提案を集める（None は除外する）。

    end_turn.propose は必ず提案を返すため、戻り値のリストは常に1件以上になる。
    """
    proposals: list[ActionProposal] = []
    for module in _PRIORITY_MODULES:
        proposal = module.propose(obs)
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def _total_score(proposal: ActionProposal) -> float:
    return proposal.score + weights.CATEGORY_BASE_WEIGHT.get(proposal.category, 0.0)


def decide(obs: Observation) -> list[int]:
    """collect_proposals の結果から次の1手を選ぶ。

    無コストで自己枯渇する下準備（_SELF_DEPLETING_SETUP: 進化/エネルギー/ドロー/特性）が
    残っている間は、攻撃より先にそれを消化する（攻撃はターンを終わらせてしまうため）。
    下準備が尽きたら、残り（retreat/attack/end）を weights.py の基礎重み補正込みの
    スコアで比較する。
    """
    proposals = collect_proposals(obs)

    if SETUP_BEFORE_ATTACK:
        turn = obs.current.turn if obs.current is not None else None
        setup = [p for p in proposals if p.category in _SELF_DEPLETING_SETUP]
        if setup and _setup_budget.take(turn, _SETUP_ACTION_CAP):
            return max(setup, key=_total_score).select

    best = max(proposals, key=_total_score)
    return best.select
