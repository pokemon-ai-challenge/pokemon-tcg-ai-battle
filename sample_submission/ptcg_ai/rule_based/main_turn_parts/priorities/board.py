"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: board（展開/進化/グッズ/スタジアム/どうぐ）

ベンチ展開・進化・場を強化するグッズ/スタジアム/どうぐの使用を提案する。
展開・進化の優先順位は knowledge.profile_registry.get_deck_plan() の
opening_priority / evolution_priority を参照する。
相手デッキ予測が確信を持てている場合は、デッキ側の対アーキタイプ加点
（MatchupPlan.card_priority_boost）も加算する。
"""

from cg.api import CardType, Observation

from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.rule_based.main_turn_parts import buckets, usage_gate
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import card_cache, profile_registry

_EVOLUTION_BASE_SCORE = 20.0
_OPENING_BASE_SCORE = 10.0
# ドローエンジンの予約ベンチ枠が埋まっていないとき、そのたね（ノコッチ）の展開に与える最優先スコア。
# 進化(_EVOLUTION_BASE_SCORE=20)より高くして「予備ノコッチのベンチ確保」を board 内で最優先にする
# （にげあしドローを毎ターン穴なく回すには次ターン用の予備ノコッチが常時必要なため）。
_RESERVED_BENCH_SCORE = 30.0


def propose(obs: Observation) -> ActionProposal | None:
    """展開/進化/グッズ/スタジアム/どうぐ系の行動を1つ提案する。該当行動が無ければ None。"""
    candidates = [i for i, option in enumerate(obs.select.option) if buckets.classify(option, obs.current) == "board"]
    candidates = [i for i in candidates if _is_option_usable(obs.select.option[i], obs.current)]
    if not candidates:
        return None

    plan = profile_registry.get_deck_plan()
    reserve_underfilled = _draw_engine_bench_underfilled(obs, plan)
    matchup_plan = opponent_tracker.current_matchup_plan()

    def priority_score(index: int) -> float:
        card_id = common.resolve_card_id(obs.select.option[index], obs.current)
        if card_id is None:
            return 0.0

        # ドローエンジンの予約ベンチ枠が不足しているなら、そのたね（ノコッチ）の展開を最優先。
        if reserve_underfilled and card_id == plan.reserved_bench_basic_id:
            return _RESERVED_BENCH_SCORE

        matchup_bonus = matchup_plan.card_priority_boost.get(card_id, 0.0) if matchup_plan is not None else 0.0

        if card_id in plan.evolution_priority:
            return _EVOLUTION_BASE_SCORE - plan.evolution_priority.index(card_id) * 0.01 + matchup_bonus
        if card_id in plan.opening_priority:
            return _OPENING_BASE_SCORE - plan.opening_priority.index(card_id) * 0.01 + matchup_bonus

        card = card_cache.get_card(card_id)
        profile = None
        if card.cardType == CardType.ITEM:
            profile = profile_registry.get_item_profile(card_id)
        elif card.cardType == CardType.TOOL:
            profile = profile_registry.get_tool_profile(card_id)
        elif card.cardType == CardType.STADIUM:
            profile = profile_registry.get_stadium_profile(card_id)
        base_score = profile.priority if profile is not None else 0.0
        return base_score + matchup_bonus

    best_index = max(candidates, key=priority_score)
    return ActionProposal(category="board", select=[best_index], score=priority_score(best_index), reason="board development")


def _draw_engine_bench_underfilled(obs: Observation, plan) -> bool:
    """ドローエンジンの予約ベンチ枠（reserved_bench_line_ids）が reserved_bench_slots 未満か。

    盤面（バトル場＋ベンチ）にいる予約ライン（ノコッチ／ノココッチ）の数が確保枠数に満たなければ
    True。展開するたね（reserved_bench_basic_id）が定義されていないデッキでは常に False。
    """
    if plan.reserved_bench_slots <= 0 or plan.reserved_bench_basic_id is None:
        return False
    state = obs.current
    player = state.players[state.yourIndex]
    on_board = [p for p in (player.active or []) if p is not None] + list(player.bench)
    line_count = sum(1 for p in on_board if p.id in plan.reserved_bench_line_ids)
    return line_count < plan.reserved_bench_slots


def _is_option_usable(option, state) -> bool:
    """usage_condition（担当Aの「今使うべきか」判定）を満たすかを見る。判定できなければ使用可扱い。"""
    card_id = common.resolve_card_id(option, state)
    if card_id is None:
        return True
    return usage_gate.is_usable(card_id, state, state.yourIndex)
