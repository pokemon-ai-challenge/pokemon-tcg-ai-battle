"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: draw（ドロー/サーチ系サポーター・グッズ）

手札のリソースが尽きていないか、探したいカード
（knowledge.profile_registry.get_deck_plan().search_priority）がまだ手札/場に無いかを見て、
優先度を判断する。
"""

from cg.api import Observation

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.rule_based.main_turn_parts import buckets, usage_gate
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import profile_registry


def propose(obs: Observation) -> ActionProposal | None:
    """ドロー/サーチ系の行動を1つ提案する。該当行動が無ければ None。"""
    candidates = [i for i, option in enumerate(obs.select.option) if buckets.classify(option, obs.current) == "draw"]
    candidates = [i for i in candidates if _is_option_usable(obs.select.option[i], obs.current)]
    if not candidates:
        return None

    plan = profile_registry.get_deck_plan()

    def priority_score(index: int) -> float:
        card_id = common.resolve_card_id(obs.select.option[index], obs.current)
        if card_id in plan.search_priority:
            # 優先度の高い（リストの先頭に近い）カードを探せるものほど高スコアにする。
            return float(len(plan.search_priority) - plan.search_priority.index(card_id))
        return 0.0

    best_index = max(candidates, key=priority_score)
    return ActionProposal(category="draw", select=[best_index], score=priority_score(best_index), reason="draw/search")


def _is_option_usable(option, state) -> bool:
    """usage_condition（担当Aの「今使うべきか」判定）を満たすかを見る。判定できなければ使用可扱い。"""
    card_id = common.resolve_card_id(option, state)
    if card_id is None:
        return True
    return usage_gate.is_usable(card_id, state, state.yourIndex)
