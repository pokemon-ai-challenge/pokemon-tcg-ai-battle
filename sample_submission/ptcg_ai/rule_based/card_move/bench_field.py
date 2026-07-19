"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_BENCH, TO_FIELD, SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON

手札や山札からベンチ/バトル場に出すカードを選ぶ。優先順位は
knowledge.profile_registry.get_deck_plan() の opening_priority / evolution_priority を参照する。
（KO後の空いたバトル場にベンチから繰り出す TO_ACTIVE は、盤面評価で交代先を選ぶ
action_selection/handlers/switch_turn.py の担当。こちらはあくまで手札/山札からの展開）。
"""

from cg.api import SelectData, State

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def choose(select: SelectData, state: State) -> list[int]:
    """ベンチ/バトル場に出すポケモンの選択肢インデックスを返す。"""
    plan = profile_registry.get_deck_plan()

    def priority_rank(option) -> int:
        card_id = common.resolve_card_id(option, state)
        if card_id in plan.opening_priority:
            return plan.opening_priority.index(card_id)
        return len(plan.opening_priority)

    return common.pick_top(select, lambda option: -priority_rank(option))
