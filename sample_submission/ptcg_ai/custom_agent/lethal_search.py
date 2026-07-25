"""確定リーサル探索（サイドが少なくなったらリーサルを考える）。ON/OFFを切り替えられるトグル。

OFF（既定）のときは呼び出し側に一切影響せず、main_turn.py の簡易リーサル判定
（今のターンの1手で倒し切れるか）のみで動く。ONにすると、既存の探索インフラ
（`ptcg_ai.search.lethal_simple` + `ptcg_ai.hidden_information.search_state_stub`。
汎用のゲーム木探索であり「デッキ固有の戦略」ではないため再利用する）を使い、
進化/エネルギー加速/複数体キルなどを組み合わせた複数手の確定リーサルを探す。
残りサイド枚数がいくつまで探索するかは configs/rule_lethal.json の
`lethal_search.max_remaining_prizes` に従う（既定2枚）。
"""

from __future__ import annotations

from cg.api import Observation

from ptcg_ai.core.config import load_config
from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state
from ptcg_ai.search import lethal_simple

# --- トグル ---
ENABLE_LETHAL_SEARCH = True

_CONFIG_CACHE: dict | None = None


def _config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


def find_lethal_selection(obs: Observation, full_deck: list[int]) -> list[int] | None:
    """確定リーサルの手筋が見つかればその最初の選択を返す。OFF/未検出ならNone。"""
    if not ENABLE_LETHAL_SEARCH:
        return None
    if obs.current is None or obs.select is None:
        return None

    lethal_config = (_config() or {}).get("lethal_search") or {}
    context = {
        "observation": obs,
        "config": lethal_config,
        "hidden_state_factory": lambda: build_dummy_search_state(obs, full_deck),
    }
    try:
        return lethal_simple.search(obs.current, obs.select.option, context)
    except Exception:
        return None
