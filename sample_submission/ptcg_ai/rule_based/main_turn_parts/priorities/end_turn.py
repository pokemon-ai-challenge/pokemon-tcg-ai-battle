"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: end（ターン終了）

他のどのカテゴリからも有意な提案が得られなかった場合の最終手段として、ターン終了を提案する。
常に選択可能な保険的な提案であり、スコアは他カテゴリより十分低く設定する
（weights.py の基礎点で調整）。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal


def propose(obs: Observation) -> ActionProposal:
    """ターン終了の提案を返す（常に提案可能）。"""
    raise NotImplementedError
