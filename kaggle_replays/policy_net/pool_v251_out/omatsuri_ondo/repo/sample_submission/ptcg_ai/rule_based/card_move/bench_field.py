"""クラスタ④ カード移動・対象選択／担当B

対象 SelectContext: TO_BENCH, TO_FIELD, SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON

手札や山札からベンチ/バトル場に出すカードを選ぶ。優先順位は
knowledge.profile_registry.get_deck_plan() の opening_priority / evolution_priority を参照する。
（KO後の空いたバトル場にベンチから繰り出す TO_ACTIVE は、盤面評価で交代先を選ぶ
action_selection/handlers/switch_turn.py の担当。こちらはあくまで手札/山札からの展開）。

相手デッキ予測が確信を持てている場合、デッキ側の対アーキタイプ加点
（MatchupPlan.card_priority_boost）も加味するが、これは TO_BENCH / SETUP_BENCH_POKEMON
（ベンチに出す場面）にのみ適用する。TO_FIELD / SETUP_ACTIVE_POKEMON
（バトル場に出す場面）には適用しない — シェイミのようにベンチ運用前提のポケモンへの
加点が、誤ってバトル場に出す判断に流用されないようにするための線引き。
"""

from cg.api import SelectContext, SelectData, State

from ptcg_ai.opponent_modeling import tracker as opponent_tracker
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry

# card_priority_boost を適用してよいのは「ベンチに出す」場面だけ。
_BENCH_ONLY_CONTEXTS = (SelectContext.TO_BENCH, SelectContext.SETUP_BENCH_POKEMON)


def choose(select: SelectData, state: State) -> list[int]:
    """ベンチ/バトル場に出すポケモンの選択肢インデックスを返す。"""
    plan = profile_registry.get_deck_plan()
    matchup_plan = opponent_tracker.current_matchup_plan() if select.context in _BENCH_ONLY_CONTEXTS else None

    def priority_rank(option) -> int:
        card_id = common.resolve_card_id(option, state)
        if card_id in plan.opening_priority:
            return plan.opening_priority.index(card_id)
        return len(plan.opening_priority)

    def score(option) -> float:
        matchup_bonus = 0.0
        if matchup_plan is not None:
            card_id = common.resolve_card_id(option, state)
            if card_id is not None:
                matchup_bonus = matchup_plan.card_priority_boost.get(card_id, 0.0)
        return -priority_rank(option) + matchup_bonus

    return common.pick_top(select, score)
