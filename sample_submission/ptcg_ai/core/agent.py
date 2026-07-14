"""クラスタ① 選択振り分け（入口）／担当B

新デッキ用 AI のロジック側エントリポイント。

責務:
    - デッキ返却（初回選択）か通常ターンかを分岐する
    - 通常ターンは decision.router.route に処理を委譲する（判断ロジックは持たない）

Kaggle 提出の入口は sample_submission/main.py の agent(obs_dict) のまま変更しない。
将来的には main.py の agent() 内から本モジュールの agent(obs) を呼び出す形で配線する想定
（今回はスケルトン作成のみで main.py の配線は行わない）。
"""

from cg.api import Observation

from ptcg_ai.action_selection import router


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or ルーター呼び出しを行う。

    Args:
        obs: main.py の agent(obs_dict) で to_observation_class 変換済みの Observation。

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
