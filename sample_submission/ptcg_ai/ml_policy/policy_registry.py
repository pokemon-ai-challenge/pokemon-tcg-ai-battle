"""アーキタイプ別 climb 専門Policy のルーティング(PolicyRegistry)。

概念:
    POLICY_PATHS = {"general": ".../policy_weights.json", "crustle": ".../policy_vs_crustle.json", ...}

- ``route_map.json``(このファイルと同じディレクトリ)に載っている(かつロード検証を通った)
  アーキタイプだけが専門Policyへルーティングされる。空/存在しなければ常に general(climb)。
- 対戦開始は必ず general。``rough_predictor.predict()`` の ``status`` が "confident" に
  なるまで切り替えない。同一予測が連続2回、または専門Policyのタイミングでのみ確定する
  (仕様: 「同一予測が連続2回出た場合、または固有anchorカードが公開された場合に確定」。
  anchorカード判定は rough_predictor の evidence 種別を流用し、他アーキと共有しやすい
  カード名(config の generic_cards)だけでは確定しないよう2回連続一致を基本ゲートにする)。
- 一度確定した route はその試合中固定(sticky)。対戦終了(obs.select is None)で全状態リセット。
- 未知/曖昧/未対応/不合格モデル/ロード失敗は general へフォールバック。
- 遅延ロード + プロセス内キャッシュ + 特徴量次元の互換性確認。
"""

from __future__ import annotations

import json
from pathlib import Path

from cg.api import Observation
from ptcg_ai.hidden_information import match_context
from ptcg_ai.learning.policy_model import PolicyModel
from ptcg_ai.opponent_modeling import rough_predictor

_HERE = Path(__file__).resolve().parent
_DEFAULT_GENERAL_WEIGHTS = _HERE.parent / "learning" / "policy_weights.json"
_ROUTE_MAP_PATH = _HERE / "route_map.json"

_EXPECTED_STATE_DIM = 166
_EXPECTED_OPTION_DIM = 65

# パスごとにロード済み PolicyModel をキャッシュ(プロセス内、対戦を跨いでも安全 = 重みは不変)。
_model_cache: dict[str, PolicyModel] = {}
_validated_route_map: dict[str, str] | None = None

# 対戦内 sticky routing state(対戦間で漏れないよう reset() で必ずクリアする)。
_state = {
    "confirmed_archetype": None,   # str | None. 一度確定したらこのまま固定。
    "last_prediction": None,       # 直前ターンの deck_type(confident時のみ)。
    "consecutive_count": 0,
}


def _load_route_map() -> dict[str, str]:
    """route_map.json をロードし、実際に読み込める(is_ready)候補だけを残す。

    パスは __file__ 基準で解決する(相対パス文字列を route_map.json に書く前提)。
    """
    global _validated_route_map
    if _validated_route_map is not None:
        return _validated_route_map
    raw: dict[str, str] = {}
    try:
        if _ROUTE_MAP_PATH.exists():
            raw = json.loads(_ROUTE_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        raw = {}
    validated: dict[str, str] = {}
    for arch, rel_path in raw.items():
        path = Path(rel_path)
        if not path.is_absolute():
            path = (_HERE / rel_path).resolve()
        try:
            pm = _get_model_by_path(str(path))
        except Exception:  # noqa: BLE001
            continue
        if pm.is_ready:
            validated[arch] = str(path)
    _validated_route_map = validated
    return validated


def _get_model_by_path(path: str) -> PolicyModel:
    model = _model_cache.get(path)
    if model is None:
        model = PolicyModel(path)
        _model_cache[path] = model
    return model


def has_routes() -> bool:
    """route_map.json に合格モデルが1件でもあれば True。空/存在しないなら False。

    ml_policy_agent._get_model はこれが False の間、旧来のモジュールグローバル `_model`
    経路をそのまま使う(既存の単一 climb 設定・テストの monkeypatch 契約と完全に後方互換)。
    """
    return bool(_load_route_map())


def reset() -> None:
    """対戦終了時に呼ぶ。sticky routing state を完全にクリアする。"""
    _state["confirmed_archetype"] = None
    _state["last_prediction"] = None
    _state["consecutive_count"] = 0


def _general_weights_path() -> str:
    return str(_DEFAULT_GENERAL_WEIGHTS)


def resolve_active_weights_path(obs: Observation) -> str:
    """このターンに使うべき重みJSONのパスを返す(general or 確定済み専門Policy)。

    PIMCのprior・Policy fallback の両方がこの関数の戻り値を使う限り、混在は起きない
    (ml_policy_agent._get_model が唯一の呼び出し口)。
    """
    if _state["confirmed_archetype"] is not None:
        route_map = _load_route_map()
        path = route_map.get(_state["confirmed_archetype"])
        if path is not None:
            return path
        # 採否後に route_map から消えた(不合格化された)場合の安全側フォールバック。
        _state["confirmed_archetype"] = None

    route_map = _load_route_map()
    if not route_map or obs.current is None:
        return _general_weights_path()

    try:
        result = rough_predictor.predict(
            obs.current, match_context.get_opponent_knowledge(obs.current.yourIndex))
    except Exception:  # noqa: BLE001
        return _general_weights_path()

    if result.get("status") != "confident":
        _state["last_prediction"] = None
        _state["consecutive_count"] = 0
        return _general_weights_path()

    deck_type = result.get("deck_type")
    if deck_type not in route_map:
        # 未対応アーキタイプ(合格モデルが無い)。確定させず general のまま。
        _state["last_prediction"] = None
        _state["consecutive_count"] = 0
        return _general_weights_path()

    if _state["last_prediction"] == deck_type:
        _state["consecutive_count"] += 1
    else:
        _state["last_prediction"] = deck_type
        _state["consecutive_count"] = 1

    if _state["consecutive_count"] >= 2:
        _state["confirmed_archetype"] = deck_type
        return route_map[deck_type]

    return _general_weights_path()


def get_model(obs: Observation | None = None, explicit_weights_path: str | None = None) -> PolicyModel:
    """model 解決の唯一の入口。

    explicit_weights_path(既存の config["policy_weights_path"] 注入点、A/B比較用)が
    あれば最優先で使う(既存の単一 climb 設定・オフライン評価スクリプトとの後方互換)。
    無ければルーターに従う。ロード失敗/特徴量次元不一致なら general へ即フォールバックする。
    """
    if explicit_weights_path is not None:
        return _get_model_by_path(explicit_weights_path)

    path = resolve_active_weights_path(obs) if obs is not None else _general_weights_path()
    try:
        model = _get_model_by_path(path)
    except Exception:  # noqa: BLE001
        model = _get_model_by_path(_general_weights_path())
        return model
    if not model.is_ready:
        return _get_model_by_path(_general_weights_path())
    return model
