"""クラスタ③ メイン行動の意思決定／担当B

各カテゴリの「今やるべきか」の提案（priorities/*.py）を集め、スコアで比較して1手を選ぶ骨格。
"""

from dataclasses import dataclass

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts import buckets, weights


@dataclass
class ActionProposal:
    """1つの候補行動と、その採用スコア。"""

    category: str
    select: list[int]
    score: float
    reason: str


from ptcg_ai.rule_based.main_turn_parts.priorities import ability, attack, board, draw, end_turn, energy, retreat

_PRIORITY_MODULES = (draw, board, ability, energy, retreat, attack, end_turn)


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


def decide(obs: Observation) -> list[int]:
    """collect_proposals の結果を weights.py の基礎重みで補正し、最もスコアの高い1手を返す。"""
    proposals = collect_proposals(obs)

    def total_score(proposal: ActionProposal) -> float:
        return proposal.score + weights.CATEGORY_BASE_WEIGHT.get(proposal.category, 0.0)

    best = max(proposals, key=total_score)
    return best.select
