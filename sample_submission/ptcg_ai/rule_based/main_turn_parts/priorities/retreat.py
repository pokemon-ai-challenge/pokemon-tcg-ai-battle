"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: retreat（にげる）

バトル場のポケモンが次ターン倒されやすい（decision.evaluation.board_features）場合や、
より良いアタッカーがベンチにいる場合に、にげる/交代を提案する。
逃げ先の評価は decision.evaluation.switch_eval を使う。今ターン既に交代済みなら提案しない
（State.retreated を確認する）。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal


def propose(obs: Observation) -> ActionProposal | None:
    """にげる行動を1つ提案する。交代の必要が無ければ None。"""
    raise NotImplementedError
