"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: MAIN（SelectType.MAIN）

「今ターン、進化・エネ付与・展開・特性・撤退・攻撃・終了のうち何をするか」を選ぶ最も
中心的な場面。実際の意思決定ロジックは main_turn_parts/ 配下（クラスタ③）に委譲する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.main_turn_parts import proposals


def handle(obs: Observation) -> list[int]:
    """MAIN の選択肢（Option.type: PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/ATTACK/END）
    の中から1手を決めて返す。実処理は main_turn_parts.proposals.decide に委譲する想定。
    """
    return proposals.decide(obs)
