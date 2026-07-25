"""カスタムルールベースAgentのエントリポイント。

`ptcg_ai.core.agent` の AGENT_TYPE == "custom_rule_based" から呼ばれる。
"""

from __future__ import annotations

import os

from cg.api import Observation, SelectData

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import lethal_search, matchup, router

_DECK_CACHE: list[int] | None = None


def read_deck_csv() -> list[int]:
    """deck.csv を読み、60枚のカードIDリストを返す。

    Kaggle 提出時の実行パスは /kaggle_simulations/agent/ 以下になるため、
    カレントディレクトリに deck.csv が無ければそちらにフォールバックする。
    ptcg_ai.rule_based.rule_based_agent.read_deck_csv と同じ実装だが、この Agent は
    旧 rule_based ツリー全体には依存しないよう、あえて重複させて自己完結させている。
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


def agent(obs: Observation) -> list[int]:
    """obs を見てデッキ返却 or 行動選択（router）呼び出しを行う。

    相手デッキ予測(matchup)・確定リーサル探索(lethal_search)は、それぞれのモジュールの
    ENABLE_* フラグがTrueのときだけ動作するトグル機能で、既定(False)では一切影響しない。
    """
    if obs.select is None:
        return _select_deck()

    matchup.update(obs)

    action = lethal_search.find_lethal_selection(obs, _full_deck())
    if action is not None and _is_valid_action(action, obs.select):
        return action

    try:
        action = router.route(obs)
    except Exception:
        action = None
    if action is not None and _is_valid_action(action, obs.select):
        return action
    return fallback.safe_choice(obs)


def _is_valid_action(action, select: SelectData) -> bool:
    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        return False
    if not (select.minCount <= len(action) <= select.maxCount):
        return False
    if len(action) != len(set(action)):
        return False
    return all(0 <= i < len(select.option) for i in action)


def _select_deck() -> list[int]:
    """deck.csv（read_deck_csv は rule_based_agent のものをそのまま再利用）の60枚を返す。"""
    global _DECK_CACHE
    if _DECK_CACHE is None:
        _DECK_CACHE = read_deck_csv()
    return _DECK_CACHE


def _full_deck() -> list[int]:
    return _select_deck()
