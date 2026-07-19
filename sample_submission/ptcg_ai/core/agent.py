"""クラスタ① 選択振り分け（入口）／担当B

新デッキ用 AI のロジック側エントリポイント。

責務:
    - AGENT_TYPE に応じて、対応する Agent 実装（*_agent.py の agent(obs)）へ処理を委譲する
    - Agent 実装ごとの判断ロジックはここには持たない

Kaggle 提出の入口は sample_submission/main.py の agent(obs_dict) のまま変更しない。

新しい Agent（legacy / ismcts / hybrid など）を追加する場合は、対応する
`ptcg_ai/<agent_type>/<agent_type>_agent.py` に agent(obs) を実装し、
下の分岐に import と呼び出しを1行ずつ追加する。
"""

from cg.api import Observation

AGENT_TYPE = "rule_based"


def agent(obs: Observation) -> list[int]:
    """AGENT_TYPE に応じた Agent 実装へ処理を委譲する。

    Args:
        obs: main.py の agent(obs_dict) で to_observation_class 変換済みの Observation。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    if AGENT_TYPE == "rule_based":
        from ptcg_ai.rule_based.rule_based_agent import agent as rule_based_agent

        return rule_based_agent(obs)

    raise ValueError(f"Unknown AGENT_TYPE: {AGENT_TYPE!r}")
