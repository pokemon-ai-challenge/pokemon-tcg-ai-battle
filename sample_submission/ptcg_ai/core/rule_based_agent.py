"""ルールベースAIの入口／担当B

core/agent.py が agent_type="rule_based" を選んだときに呼ばれる、ルールベースAI
そのものの入口。デッキ返却/通常ターンの分岐は core/agent.py の責務なので持たない
（ここに来る時点で通常ターンの Observation であることが前提）。

通常ターンの選択肢処理そのものは action_selection.router（SelectContextごとの
handlerへの振り分け）に委譲する。legacy_agent/ismcts_agent/hybrid_agent など
他のエージェント種別を追加する際は、同じ形（agent(obs) -> list[int]）で
core/ 配下に並べて追加し、core/agent.py 側の分岐に一行足すだけでよい。
"""

from cg.api import Observation

from ptcg_ai.action_selection import router


def agent(obs: Observation) -> list[int]:
    """通常ターンの選択肢インデックスを、router.route に委譲して返す。"""
    return router.route(obs)
