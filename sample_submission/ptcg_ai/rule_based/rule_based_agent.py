"""ルールベース Agent の入口／担当B

`core.agent` から `AGENT_TYPE == "rule_based"` のときに呼ばれる Agent 実装。

責務:
    - デッキ返却（初回選択）か通常ターンかを分岐する
    - 通常ターンは action_selection.router.route に処理を委譲する（判断ロジックは持たない）
"""

import os

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

    現状は sample_submission/deck.csv（提出物本体、CLAUDE.md記載の「使用デッキ」）を
    読むだけ。デッキ内容そのものは decks/new_deck/ 側の担当Aデータと揃っている必要があるが、
    その一致を保証するのはこの関数の責務ではない。
    """
    return read_deck_csv()


def read_deck_csv() -> list[int]:
    """deck.csv を読み、60枚のカードIDリストを返す。

    Kaggle 提出時の実行パスは /kaggle_simulations/agent/ 以下になるため、
    カレントディレクトリに deck.csv が無ければそちらにフォールバックする
    （CLAUDE.md「開発時の注意」参照）。main.py からも同じ実装を re-export して使う。
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck
