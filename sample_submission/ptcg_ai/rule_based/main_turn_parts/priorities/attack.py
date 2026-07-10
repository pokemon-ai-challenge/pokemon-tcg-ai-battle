"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: attack（ワザを使う）

decision.evaluation.attack_features / energy_requirements を使い、使用可能なワザの中から
最も価値の高いものを提案する。相手をきぜつさせられる（can_ko）ワザは最優先とする。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal


def propose(obs: Observation) -> ActionProposal | None:
    """ワザ使用の行動を1つ提案する。使用可能なワザが無ければ None。"""
    raise NotImplementedError
