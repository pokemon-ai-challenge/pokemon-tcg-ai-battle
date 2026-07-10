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


def collect_proposals(obs: Observation) -> list[ActionProposal]:
    """buckets.bucketize の結果を各 priorities/*.py の propose() に渡し、提案を集める。"""
    raise NotImplementedError


def decide(obs: Observation) -> list[int]:
    """collect_proposals の結果を weights.py の基礎重みで補正し、最もスコアの高い1手を返す。"""
    raise NotImplementedError
