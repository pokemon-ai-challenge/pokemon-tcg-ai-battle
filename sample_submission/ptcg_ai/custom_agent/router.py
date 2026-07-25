"""SelectContext -> 各ハンドラへの振り分け（薄い層。判断ロジックは持たない）。"""

from __future__ import annotations

from cg.api import Observation, SelectContext, SelectType

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import attack, energy, evolution, main_turn, misc_handlers, retreat_switch, setup

_CONTEXT_HANDLERS = {
    SelectContext.MAIN: main_turn.decide,
    SelectContext.SETUP_ACTIVE_POKEMON: setup.handle,
    SelectContext.SETUP_BENCH_POKEMON: setup.handle,
    SelectContext.IS_FIRST: setup.handle,
    SelectContext.MULLIGAN: setup.handle,
    SelectContext.SWITCH: retreat_switch.handle_switch_select,
    SelectContext.TO_ACTIVE: retreat_switch.handle_switch_select,
    SelectContext.EVOLVE: evolution.handle_evolution_select,
    SelectContext.EVOLVES_FROM: evolution.handle_evolution_select,
    SelectContext.EVOLVES_TO: evolution.handle_evolution_select,
    SelectContext.DEVOLVE: evolution.handle_evolution_select,
    SelectContext.MORE_DEVOLVE: evolution.handle_evolution_select,
    SelectContext.ATTACH_FROM: energy.handle_energy_tool_select,
    SelectContext.ATTACH_TO: energy.handle_energy_tool_select,
    SelectContext.DETACH_FROM: energy.handle_energy_tool_select,
    SelectContext.SWITCH_ENERGY_CARD: energy.handle_energy_tool_select,
    SelectContext.TO_HAND_ENERGY: energy.handle_energy_tool_select,
    SelectContext.TO_DECK_ENERGY: energy.handle_energy_tool_select,
    SelectContext.SWITCH_ENERGY: energy.handle_energy_tool_select,
    SelectContext.ATTACK: attack.handle_attack_select,
    SelectContext.DISABLE_ATTACK: attack.handle_attack_select,
    SelectContext.SKILL_ORDER: misc_handlers.handle_effect_choice,
    SelectContext.ACTIVATE: misc_handlers.handle_effect_choice,
    SelectContext.FIRST_EFFECT: misc_handlers.handle_effect_choice,
    SelectContext.COIN_HEAD: misc_handlers.handle_effect_choice,
    SelectContext.DAMAGE_COUNTER: misc_handlers.handle_damage_target,
    SelectContext.DAMAGE_COUNTER_ANY: misc_handlers.handle_damage_target,
    SelectContext.DAMAGE: misc_handlers.handle_damage_target,
    SelectContext.REMOVE_DAMAGE_COUNTER: misc_handlers.handle_damage_target,
    SelectContext.HEAL: misc_handlers.handle_damage_target,
    SelectContext.DRAW_COUNT: misc_handlers.handle_count,
    SelectContext.DAMAGE_COUNTER_COUNT: misc_handlers.handle_count,
    SelectContext.REMOVE_DAMAGE_COUNTER_COUNT: misc_handlers.handle_count,
    SelectContext.AFFECT_SPECIAL_CONDITION: misc_handlers.handle_special_condition,
    SelectContext.RECOVER_SPECIAL_CONDITION: misc_handlers.handle_special_condition,
    SelectContext.TO_BENCH: misc_handlers.handle_bench_field,
    SelectContext.TO_FIELD: misc_handlers.handle_bench_field,
    SelectContext.TO_HAND: misc_handlers.handle_hand_like,
    SelectContext.TO_DECK: misc_handlers.handle_hand_like,
    SelectContext.TO_DECK_BOTTOM: misc_handlers.handle_hand_like,
    SelectContext.DISCARD: misc_handlers.handle_discard_like,
    SelectContext.DISCARD_ENERGY_CARD: misc_handlers.handle_discard_like,
    SelectContext.DISCARD_TOOL_CARD: misc_handlers.handle_discard_like,
    SelectContext.DISCARD_ENERGY: misc_handlers.handle_discard_like,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD: misc_handlers.handle_discard_like,
    SelectContext.TO_PRIZE: misc_handlers.handle_to_prize_or_look,
    SelectContext.LOOK: misc_handlers.handle_to_prize_or_look,
    SelectContext.NOT_MOVE: misc_handlers.handle_not_move_or_effect_target,
    SelectContext.EFFECT_TARGET: misc_handlers.handle_not_move_or_effect_target,
}


def route(obs: Observation) -> list[int]:
    select = obs.select
    handler = _CONTEXT_HANDLERS.get(select.context)
    if handler is None and select.type == SelectType.YES_NO:
        # SelectContext はコンペ期間中に要素が追加され得るための安全網。
        handler = misc_handlers.handle_yes_no_fallback
    if handler is None:
        return fallback.safe_choice(obs)

    try:
        return handler(obs)
    except Exception:
        return fallback.safe_choice(obs)
