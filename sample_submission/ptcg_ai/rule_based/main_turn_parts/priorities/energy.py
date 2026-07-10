"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: energy（エネルギー付与）

今ターンのエネルギー付与を提案する。対象選定は
decision.main_turn_parts.energy_eval.best_energy_target を使う。
1ターン1回までの制限（State.energyAttached）を尊重する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal


def propose(obs: Observation) -> ActionProposal | None:
    """エネルギー付与の行動を1つ提案する。付与済み/対象なしの場合は None。"""
    raise NotImplementedError
