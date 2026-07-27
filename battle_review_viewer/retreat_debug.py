"""feature/bayesian-agent で追加した「にげる判断の確率化」(probabilistic_ko)を
ビュアー向けにまとめる。

判定ロジックの実体は
``sample_submission/ptcg_ai/board_evaluation/board_features.likely_ko_probability_next_turn``
（新: ベイズ推定によるKO確率）と ``is_likely_ko_next_turn``（旧: 決定論的なブール判定）、
どちらを使うか・閾値は ``sample_submission/ptcg_ai/core/config.py`` の
``probabilistic_ko`` セクションにある。ここではその出力をビュアー payload の形に
薄くラップするだけ。

**トリガーAのみを表示する**（``priorities/retreat.py`` の
``_trigger_a_avoid_ko`` 相当）。トリガーB（後退での即きぜつ）は今回の変更と無関係の
別ロジックなので対象外。
"""

from __future__ import annotations

from typing import Any


def build_retreat_debug(real_state: Any) -> dict[str, Any] | None:
    """自分のアクティブ(+にげ先候補)のKO確率・旧ブール判定・トリガーA相当の判定結果をまとめる。

    real_state が None(まだ対局開始前など)なら None。それ以外の失敗は握りつぶし
    {"error": str(exc)} を返す（``value_eval_debug.py`` と同じフォールバック規則）。
    """
    if real_state is None:
        return None
    try:
        return _build(real_state)
    except Exception as exc:  # noqa: BLE001 -- 推定レイヤーが落ちてもリプレイ生成/ライブ対戦は止めない
        return {"error": str(exc)}


def _pokemon_payload(pokemon: Any, probability: float, old_boolean: bool, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "hp": pokemon.hp,
        "max_hp": pokemon.maxHp,
        "probability": probability,
        "old_boolean": old_boolean,
    }


def _build(real_state: Any) -> dict[str, Any]:
    from ptcg_ai.board_evaluation import board_features  # noqa: PLC0415 -- 遅延import(呼び出し側のsys.path設定に依存するため)
    from ptcg_ai.core.config import load_config  # noqa: PLC0415
    from ptcg_ai.rule_based.main_turn_parts import pokemon_value  # noqa: PLC0415
    from ptcg_ai.shared import card_cache  # noqa: PLC0415

    your_index = real_state.yourIndex
    player = real_state.players[your_index]
    active = player.active[0] if player.active else None
    if active is None:
        return {"has_active": False}

    cfg = (load_config() or {}).get("probabilistic_ko") or {}
    enabled = bool(cfg.get("enabled", False))
    threshold = float(cfg.get("threshold", 0.2))

    def name_of(card_id: int) -> str:
        try:
            return card_cache.get_card(card_id).name
        except KeyError:
            return str(card_id)

    active_prob = board_features.likely_ko_probability_next_turn(active, real_state, your_index)
    active_bool = board_features.is_likely_ko_next_turn(active, real_state, your_index)
    active_payload = _pokemon_payload(active, active_prob, active_bool, name_of(active.id))

    bench = list(player.bench)
    candidate = pokemon_value.best_switch_target(bench, real_state, your_index) if bench else None
    candidate_payload = None
    candidate_prob = None
    if candidate is not None:
        candidate_prob = board_features.likely_ko_probability_next_turn(candidate, real_state, your_index)
        candidate_bool = board_features.is_likely_ko_next_turn(candidate, real_state, your_index)
        candidate_payload = _pokemon_payload(candidate, candidate_prob, candidate_bool, name_of(candidate.id))

    would_retreat_new = None
    would_retreat_old = None
    if candidate_payload is not None:
        would_retreat_new = active_prob >= threshold and candidate_prob < threshold
        would_retreat_old = bool(active_bool) and not bool(candidate_payload["old_boolean"])

    return {
        "has_active": True,
        "config": {"enabled": enabled, "threshold": threshold},
        "active": active_payload,
        "candidate": candidate_payload,
        "would_retreat": {"new": would_retreat_new, "old": would_retreat_old},
    }
