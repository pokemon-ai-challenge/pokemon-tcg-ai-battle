"""対戦相手のアーキタイプを rough_predictor で確信を持って認識できたとき、そのアーキタイプ専用に
鍛えた個別RL重みが opponent_model_registry.json に登録されていればそのパスを返す
（無ければ None＝呼び出し側は既定の汎用重みのまま）。

design: kaggle_replays/docs/requirements-kamitsuorochi-2026-08-12.md step4。

wall_guard.py と同じ「モジュールグローバルの試合スコープ状態＋config で有効化（既定 off）」
パターンを踏襲する。デッキ予測器（21クラスNN）ではなく rough_predictor を使う理由・
ゲート条件（confident のときだけ切替／一度切替えたら試合中は戻さない／レジストリに
無いアーキタイプは汎用のまま）は要件書 step4 を参照。

このモジュール自体は「どの重みパスを使うべきか」を返すだけで、PolicyModel の差し替えは
呼び出し側（ml_policy_agent.py）の責務。
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from cg.api import Observation

from ptcg_ai.hidden_information import match_context

from . import rough_predictor

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "learning" / "opponent_model_registry.json"

# このプロセスの「今の試合」で一度確定したアーキタイプ名（rough_predictor の deck_type と同じ
# 文字列）。確定後は試合中ずっとこれを返す（プランの不連続を避けるため戻さない）。
_routed_deck_type: str | None = None


def reset() -> None:
    """新しい試合の開始を検知したら呼ぶ（wall_guard.reset_pending_target と同じ役割）。"""
    global _routed_deck_type
    _routed_deck_type = None


@lru_cache(maxsize=1)
def _load_registry() -> dict:
    with _REGISTRY_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def route(obs: Observation, config: dict | None = None) -> str | None:
    """個別RL重みの絶対パス文字列を返す。ルーティング無効／未確信／未登録なら None。

    None が返るケース（＝呼び出し側は何もせず既定の重みを使い続ける）:
      - config["opponent_model_routing"]["enabled"] が真でない（既定）
      - rough_predictor がまだ確信を持てていない（insufficient_evidence/ambiguous/no_candidate）
      - 確信は持てたが、そのアーキタイプがレジストリに未登録（＝個別RLが無い、または
        E3ゲートで有意差なしと判定された）
    """
    routing_config = (config or {}).get("opponent_model_routing") or {}
    if not routing_config.get("enabled", False):
        return None

    global _routed_deck_type
    registry = _load_registry()
    if _routed_deck_type is not None:
        weights_file = registry.get(_routed_deck_type)
        return str(_REGISTRY_PATH.parent / weights_file) if weights_file else None

    if obs.current is None:
        return None

    knowledge = match_context.get_knowledge(obs.current.yourIndex)
    result = rough_predictor.predict(obs.current, knowledge)
    if result.get("status") != "confident":
        return None

    deck_type = result.get("deck_type")
    weights_file = registry.get(deck_type)
    if not weights_file:
        return None  # 認識はできたがレジストリに個別重みが無い。確定はさせない(次ターンも再判定)

    _routed_deck_type = deck_type
    return str(_REGISTRY_PATH.parent / weights_file)
