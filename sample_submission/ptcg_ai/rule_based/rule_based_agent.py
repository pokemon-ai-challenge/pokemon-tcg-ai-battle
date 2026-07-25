"""ルールベース Agent の入口／担当B

`core.agent` から `AGENT_TYPE == "rule_based"` のときに呼ばれる Agent 実装。

責務:
    - デッキ返却（初回選択）か通常ターンかを分岐する
    - 通常ターンは action_selection.selector.select_action に処理を委譲する（判断ロジックは持たない）。
      selector 側で確定リーサル探索（#57/#58）を先に試し、無ければ router.route のルールベースに落ちる
"""

import os

from cg.api import Observation

from ptcg_ai.action_selection import selector
from ptcg_ai.hidden_information import match_context
from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.rule_based.main_turn_parts import proposals

_DECK_CACHE: list[int] | None = None


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or 行動選択（selector）呼び出しを行う。

    Args:
        obs: core.agent から渡される Observation。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    # 非公開情報推定レイヤー(hidden_information)の更新。純粋な副作用追加であり、
    # 失敗しても意思決定を止めない(match_context.update内部でtry/exceptしている)。
    # router.route以降の判断ロジックには一切影響しない。
    match_context.update(obs)

    if obs.select is None:
        # 新しい試合の開始（match_context と同じ検知方法）。前試合の相手デッキ予測が
        # 次の試合に持ち越されないよう、opponent_modeling.tracker の状態を破棄する。
        opponent_tracker.reset()
        # 1プロセス内で複数ゲームを回す自己対戦・テストでのターン跨ぎ状態汚染を避けるため、
        # proposals.py の下準備採用状態も明示リセットする。
        proposals.reset_turn_state()
        return _select_deck()
    return selector.select_action(obs, _full_deck())


def _full_deck() -> list[int]:
    """deck.csv の内容をキャッシュして返す（リーサル探索の隠れ情報スタブ用）。"""
    global _DECK_CACHE
    if _DECK_CACHE is None:
        _DECK_CACHE = read_deck_csv()
    return _DECK_CACHE


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
