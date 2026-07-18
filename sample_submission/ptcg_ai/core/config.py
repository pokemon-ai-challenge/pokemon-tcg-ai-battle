"""AI全体の入口設定／担当B

configs/ 配下のJSONファイルを読み、エージェント種別などの切り替え設定を取り出す。
Kaggle提出時の実行パスは /kaggle_simulations/agent/ 以下になるため、deck.csv の読み込み
（ptcg_ai/core/agent.py の read_deck_csv）と同じ考え方で、カレントディレクトリに無ければ
そちらにフォールバックする。設定ファイルが無い/壊れている場合でも対戦を止めないよう、
既定値（rule_based）にフォールバックする。
"""

import json
import os

_AGENT_CONFIG_PATH = "configs/agent.json"
_DEFAULT_AGENT_TYPE = "rule_based"


def get_agent_type() -> str:
    """configs/agent.json の agent_type を返す。読めない/未指定なら既定値（rule_based）。"""
    config = _load_json(_AGENT_CONFIG_PATH)
    agent_type = config.get("agent_type", _DEFAULT_AGENT_TYPE)
    return agent_type if isinstance(agent_type, str) else _DEFAULT_AGENT_TYPE


def _load_json(relative_path: str) -> dict:
    """相対パス→Kaggle実行パスの順に探して読む。どちらにも無い/読めない/dict型でない場合は空dict。"""
    file_path = relative_path
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + relative_path
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
