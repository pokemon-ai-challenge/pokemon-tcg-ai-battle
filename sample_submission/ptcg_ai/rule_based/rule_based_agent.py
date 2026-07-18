"""ルールベース Agent の入口／担当B

`core.agent` から `AGENT_TYPE == "rule_based"` のときに呼ばれる Agent 実装。

責務:
    - デッキ返却（初回選択）か通常ターンかを分岐する
    - 通常ターンは action_selection.router.route に処理を委譲する（判断ロジックは持たない）
"""

from cg.api import Observation

from ptcg_ai.action_selection import router


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or ルーター呼び出しを行う。

    Args:
        obs: core.agent から渡される Observation。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    if obs.select is None:
        return _select_deck()
    return router.route(obs)


def _select_deck() -> list[int]:
    """使用する60枚デッキのカードIDリストを返す。

    decks.active の DeckPlan（担当A差し替え対象）を参照して構築する想定。
    """
    raise NotImplementedError
