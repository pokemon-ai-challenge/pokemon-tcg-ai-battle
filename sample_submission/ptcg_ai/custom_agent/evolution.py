"""進化の優先順位・ふしぎなアメの使用条件・EVOLVE系選択肢の処理。"""

from __future__ import annotations

from cg.api import Observation, OptionType, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import constants
from ptcg_ai.rule_based.card_move import common


def choose_main_evolve_option(select, state) -> int | None:
    """MAINの選択肢にEVOLVEがあれば必ず進化する（進化できるなら進化する）。

    複数のEVOLVEが同時に選べる場合のみ、EVOLUTION_PRIORITY
    （フーディン系列743/742→ノココッチ66）でタイブレークする。
    EVOLUTION_PRIORITYに無い進化（通常この2系列以外は発生しない想定）でも、
    進化できる機会を逃さないよう選択対象に含める。
    """
    best_index = None
    best_rank = None
    for i, option in enumerate(select.option):
        if option.type != OptionType.EVOLVE:
            continue
        card_id = common.resolve_card_id(option, state)
        rank = (
            constants.EVOLUTION_PRIORITY.index(card_id)
            if card_id in constants.EVOLUTION_PRIORITY
            else len(constants.EVOLUTION_PRIORITY)
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_index = i
    return best_index


def handle_evolution_select(obs: Observation) -> list[int]:
    """SelectContext: EVOLVE, EVOLVES_FROM, EVOLVES_TO, DEVOLVE, MORE_DEVOLVE。"""
    context = obs.select.context
    if context == SelectContext.MORE_DEVOLVE:
        return _choose_no(obs)
    if context == SelectContext.DEVOLVE:
        return common.pick_top(obs.select, lambda option: 0.0)
    if context in (SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO, SelectContext.EVOLVE):
        return _choose_by_priority(obs)
    return fallback.safe_choice(obs)


def _choose_by_priority(obs: Observation) -> list[int]:
    state = obs.current

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in constants.EVOLUTION_PRIORITY:
            return float(len(constants.EVOLUTION_PRIORITY) - constants.EVOLUTION_PRIORITY.index(card_id))
        return 0.0

    return common.pick_top(obs.select, score)


def _choose_no(obs: Observation) -> list[int]:
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.NO:
            return [i]
    return fallback.safe_choice(obs)
