"""相手デッキ予測（opponent_modeling）との連携。ON/OFFを切り替えられるようにしたトグル。

OFF（既定）のときは呼び出し側に一切影響しない。ONにすると、
`ptcg_ai.opponent_modeling.tracker`（既存の対戦相手デッキ予測。汎用インフラとして再利用）が
「確信を持てた」と判定したアーキタイプに応じて、シェイミ・改造ハンマーの優先度を調整する。
ブースト値は decks/new_deck/deck_plan.py の MATCHUP_PLANS で行われていた分析
（ベンチ狙撃技を持つアーキタイプにはシェイミを優先、単体攻撃のみのアーキタイプでは温存 等）を
そのまま踏襲したもの。
"""

from __future__ import annotations

from cg.api import Observation

from ptcg_ai.custom_agent import constants
from ptcg_ai.opponent_modeling import tracker as opponent_tracker

# --- トグル ---
ENABLE_OPPONENT_DECK_PREDICTION = False

# シェイミ(343): ベンチ狙撃技を持つアーキタイプには優先度を上げ、単体攻撃のみのアーキタイプには
# 下げる（出しても腐りやすいため他の展開を優先させる）。
_SHAYMIN_BOOST_BY_ARCHETYPE: dict[str, float] = {
    "dragapult_ex": 3.0,
    "gekkouga_ex": 3.0,
    "oliva_ex": 3.0,
    "mega_starmie_ex": 1.5,
    "marnie_grimmsnarl_ex": 1.5,
    "mega_lucario_ex": -3.0,
    "alakazam": -3.0,
    "kamitsuorochi_ex": -3.0,
    "takeruraiko_ex": -3.0,
    "ogerpon_teal_ex": -3.0,
    "shirona_garchomp_ex": -3.0,
    "toxtricity": -3.0,
    "rocket_honchkrow": -3.0,
    "crustle": -3.0,
    "mega_froslass_ex": -3.0,
    "mega_abomasnow_ex": -3.0,
    "archaludon_ex": -3.0,
}

# 改造ハンマー(1081): 特殊エネルギー依存の構築（メガルカリオex等）に対して優先度を上げる。
_ENHANCED_HAMMER_BOOST_BY_ARCHETYPE: dict[str, float] = {
    "mega_lucario_ex": 0.4,
}


def update(obs: Observation) -> None:
    """相手デッキ予測の更新（副作用のみ）。OFFなら何もしない。失敗しても意思決定を止めない。"""
    if not ENABLE_OPPONENT_DECK_PREDICTION:
        return
    if obs.current is None:
        return
    try:
        opponent_tracker.update(obs)
    except Exception:
        pass


def _current_archetype() -> str | None:
    if not ENABLE_OPPONENT_DECK_PREDICTION:
        return None
    prediction = opponent_tracker.current_prediction()
    if not prediction or prediction.get("status") != "confident":
        return None
    return prediction.get("deck_type")


def card_priority_boost(card_id: int) -> float:
    """card_id の優先度に加算するブースト値（OFF、または未確信、対象外カードなら0.0）。"""
    archetype = _current_archetype()
    if archetype is None:
        return 0.0
    if card_id == constants.SHAYMIN:
        return _SHAYMIN_BOOST_BY_ARCHETYPE.get(archetype, 0.0)
    if card_id == constants.ENHANCED_HAMMER:
        return _ENHANCED_HAMMER_BOOST_BY_ARCHETYPE.get(archetype, 0.0)
    return 0.0
