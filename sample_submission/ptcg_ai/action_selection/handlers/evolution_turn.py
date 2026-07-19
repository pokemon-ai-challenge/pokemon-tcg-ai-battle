"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: EVOLVE, EVOLVES_FROM, EVOLVES_TO, DEVOLVE, MORE_DEVOLVE

進化・退化に関する選択。どのポケモンをどの順で進化させるかの優先順位は
knowledge.profile_registry.get_deck_plan().evolution_priority を参照する。
"""

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import profile_registry


def handle(obs: Observation) -> list[int]:
    """EVOLVE 系の選択肢から進化元・進化先の組み合わせを決めて返す。"""
    context = obs.select.context
    if context == SelectContext.MORE_DEVOLVE:
        # 既定では退化を深追いしない（進行を戻すのは基本的に不利なため）。
        return _choose_no(obs)
    if context == SelectContext.DEVOLVE:
        # 退化させる対象を選ぶ判断材料が乏しいため、安全側として先頭から選ぶ。
        return common.pick_top(obs.select, lambda option: 0.0)
    if context in (SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO, SelectContext.EVOLVE):
        return _choose_by_evolution_priority(obs)
    return fallback.safe_choice(obs)


def _choose_by_evolution_priority(obs: Observation) -> list[int]:
    plan = profile_registry.get_deck_plan()

    def score(option) -> float:
        card_id = common.resolve_card_id(option, obs.current)
        if card_id in plan.evolution_priority:
            return float(len(plan.evolution_priority) - plan.evolution_priority.index(card_id))
        return 0.0

    return common.pick_top(obs.select, score)


def _choose_no(obs: Observation) -> list[int]:
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.NO:
            return [i]
    return fallback.safe_choice(obs)
