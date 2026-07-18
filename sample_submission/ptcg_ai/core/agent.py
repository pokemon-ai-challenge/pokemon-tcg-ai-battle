"""AI全体の入口／担当B

新デッキ用 AI のロジック側エントリポイント。sample_submission/main.py の agent(obs_dict) から
呼ばれる、ルール判断ロジック側の唯一の入口。

責務:
    - デッキ返却（初回選択）か通常ターンかを分岐する（エージェント種別に関わらず共通）
    - 通常ターンは configs/agent.json の agent_type に応じて、対応するエージェント実装
      （rule_based_agent など、いずれも agent(obs) -> list[int] という同じ形）に委譲する
    - 個別の判断ロジックはここには持たない

エージェント種別を追加する場合:
    1. core/ 配下に "xxx_agent.py"（agent(obs) -> list[int] を持つモジュール）を追加する
    2. 下の分岐に import と if を1行ずつ足す
    3. configs/agent.json の agent_type をそのエージェント名に切り替える
"""

import os

from cg.api import Observation

from ptcg_ai.core import config, rule_based_agent

# 他のエージェント種別を追加したら、ここに import を足す。
# from ptcg_ai.core import legacy_agent
# from ptcg_ai.core import ismcts_agent
# from ptcg_ai.core import hybrid_agent


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or 各エージェントへの委譲を行う。

    Args:
        obs: main.py の agent(obs_dict) で to_observation_class 変換済みの Observation。

    Returns:
        list[int]: 初回はデッキの60枚のカードIDリスト。通常ターンは選択肢インデックスのリスト。
    """
    if obs.select is None:
        return _select_deck()
    return _select_turn_agent()(obs)


def _select_turn_agent():
    """configs/agent.json の agent_type に対応するエージェントの agent(obs) 関数を返す。

    未知の agent_type や設定読み込み失敗時は rule_based_agent にフォールバックする
    （対戦を止めないことを優先する。他のfallback系モジュールと同じ方針）。
    """
    agent_type = config.get_agent_type()

    if agent_type == "rule_based":
        return rule_based_agent.agent
    # if agent_type == "legacy":
    #     return legacy_agent.agent
    # if agent_type == "ismcts":
    #     return ismcts_agent.agent
    # if agent_type == "hybrid":
    #     return hybrid_agent.agent

    return rule_based_agent.agent


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
