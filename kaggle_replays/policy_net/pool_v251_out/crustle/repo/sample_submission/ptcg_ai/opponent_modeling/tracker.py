"""相手デッキ予測を1試合を通じて追跡し、意思決定側から引けるようにするための薄い配線層。

opponent_knowledge.OpponentKnowledge（観測の蓄積）と rough_predictor.predict()
（アーキタイプ推定）はどちらも単体では意思決定に呼ばれていなかったため、
毎ターンの入口（action_selection.selector.select_action）から1回だけ update() を
呼ぶことで、priorities/*.py から current_matchup_plan() を引くだけで済むようにする。

rule_based_agent.py の _DECK_CACHE と同じ、モジュールレベルキャッシュのパターンに揃えている。
"""

from __future__ import annotations

from cg.api import Observation

from ptcg_ai.opponent_modeling import rough_predictor
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge
from ptcg_ai.shared import profile_registry
from ptcg_ai.shared.profile_types import MatchupPlan

# rough_predictor.predict() が確信を持てたと判断するステータス。
# これ以外（ambiguous/insufficient_evidence/no_candidate）では matchup_plans を適用しない
# （序盤の当てずっぽうな予測でカード優先度を歪めないため）。
_CONFIDENT_STATUS = "confident"

_knowledge: OpponentKnowledge | None = None
_last_prediction: dict | None = None


def update(obs: Observation) -> dict:
    """毎ターン1回呼ぶ。観測を蓄積し、最新の予測結果を返す。

    Args:
        obs: その時点の Observation（obs.current が None でないこと）。

    Returns:
        dict: rough_predictor.predict() が返す予測結果。
    """
    global _knowledge, _last_prediction
    if _knowledge is None:
        _knowledge = OpponentKnowledge()

    # 呼び出し順序固定（opponent_knowledge.py 参照）: logs → state の順で更新する。
    _knowledge.update_from_logs(obs.logs)
    _knowledge.update_from_state(obs.current)

    _last_prediction = rough_predictor.predict(obs.current, _knowledge)
    return _last_prediction


def current_prediction() -> dict | None:
    """直近の update() 結果を返す（再計算はしない）。update() 未実行なら None。"""
    return _last_prediction


def current_matchup_plan() -> MatchupPlan | None:
    """現在の予測が確信を持てるものであれば、対応する MatchupPlan を返す。

    予測が無い/確信度不足（ambiguous・insufficient_evidence・no_candidate）の場合や、
    デッキ側にそのアーキタイプ用のエントリがまだ無い場合は None を返す。
    """
    prediction = _last_prediction
    if prediction is None or prediction.get("status") != _CONFIDENT_STATUS:
        return None
    deck_type = prediction.get("deck_type")
    if not deck_type:
        return None
    return profile_registry.get_deck_plan().matchup_plans.get(deck_type)


def reset() -> None:
    """テスト用: トラッカーの状態を破棄する（通常の対戦では不要）。"""
    global _knowledge, _last_prediction
    _knowledge = None
    _last_prediction = None
