"""クラスタ① 選択振り分け／担当B

「今どの場面の選択か」（SelectContext）を判定して、対応する handlers/*.py に渡すだけの薄い層。
判断ロジック（何を選ぶべきか）はここには持たない。

SelectContext -> handlers/*.py の対応表（初期案。実装時に見直してよい）:

    main_turn.py            MAIN
    setup_turn.py            SETUP_ACTIVE_POKEMON, SETUP_BENCH_POKEMON, IS_FIRST, MULLIGAN
    switch_turn.py            SWITCH, TO_ACTIVE
    evolution_turn.py        EVOLVES_FROM, EVOLVES_TO, DEVOLVE, MORE_DEVOLVE, EVOLVE
    energy_tool_turn.py    ATTACH_FROM, ATTACH_TO, DETACH_FROM, DISCARD_ENERGY_CARD,
                            DISCARD_TOOL_CARD, SWITCH_ENERGY_CARD, DISCARD_ENERGY,
                            TO_HAND_ENERGY, TO_DECK_ENERGY, SWITCH_ENERGY
    attack_turn.py            ATTACK, DISABLE_ATTACK
    effect_choice_turn.py    SKILL_ORDER, ACTIVATE, FIRST_EFFECT, COIN_HEAD
    damage_target_turn.py    DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE,
                            REMOVE_DAMAGE_COUNTER, HEAL
    count_turn.py            DRAW_COUNT, DAMAGE_COUNTER_COUNT, REMOVE_DAMAGE_COUNTER_COUNT
    special_condition_turn.py AFFECT_SPECIAL_CONDITION, RECOVER_SPECIAL_CONDITION
    card_move_turn.py        TO_BENCH, TO_FIELD, TO_HAND, DISCARD, TO_DECK, TO_DECK_BOTTOM,
                            TO_PRIZE, NOT_MOVE, LOOK, EFFECT_TARGET,
                            DISCARD_CARD_OR_ATTACHED_CARD
    yes_no_turn.py            上記に含まれない YES_NO 系（フォールバック）。
                            SelectContext は競技期間中に要素が追加され得るため、
                            未知の YES_NO context を安全に処理する安全網として置く。

対応表に無い（未知の）SelectContext が来た場合は fallback.py に委譲する。
"""

from cg.api import Observation, SelectContext, SelectType

from ptcg_ai.action_selection import fallback
from ptcg_ai.action_selection.handlers import (
    attack_turn,
    card_move_turn,
    count_turn,
    damage_target_turn,
    effect_choice_turn,
    energy_tool_turn,
    evolution_turn,
    main_turn,
    setup_turn,
    special_condition_turn,
    switch_turn,
    yes_no_turn,
)

# 上部の対応表をそのままコードに落とし込んだもの。ここに無い SelectContext は
# route() 側で SelectType.YES_NO なら yes_no_turn、それ以外は fallback に回す
# （SelectContext はコンペ期間中に要素が追加され得るため）。
_CONTEXT_HANDLERS = {
    SelectContext.MAIN: main_turn,
    SelectContext.SETUP_ACTIVE_POKEMON: setup_turn,
    SelectContext.SETUP_BENCH_POKEMON: setup_turn,
    SelectContext.IS_FIRST: setup_turn,
    SelectContext.MULLIGAN: setup_turn,
    SelectContext.SWITCH: switch_turn,
    SelectContext.TO_ACTIVE: switch_turn,
    SelectContext.EVOLVE: evolution_turn,
    SelectContext.EVOLVES_FROM: evolution_turn,
    SelectContext.EVOLVES_TO: evolution_turn,
    SelectContext.DEVOLVE: evolution_turn,
    SelectContext.MORE_DEVOLVE: evolution_turn,
    SelectContext.ATTACH_FROM: energy_tool_turn,
    SelectContext.ATTACH_TO: energy_tool_turn,
    SelectContext.DETACH_FROM: energy_tool_turn,
    SelectContext.DISCARD_ENERGY_CARD: energy_tool_turn,
    SelectContext.DISCARD_TOOL_CARD: energy_tool_turn,
    SelectContext.SWITCH_ENERGY_CARD: energy_tool_turn,
    SelectContext.DISCARD_ENERGY: energy_tool_turn,
    SelectContext.TO_HAND_ENERGY: energy_tool_turn,
    SelectContext.TO_DECK_ENERGY: energy_tool_turn,
    SelectContext.SWITCH_ENERGY: energy_tool_turn,
    SelectContext.ATTACK: attack_turn,
    SelectContext.DISABLE_ATTACK: attack_turn,
    SelectContext.SKILL_ORDER: effect_choice_turn,
    SelectContext.ACTIVATE: effect_choice_turn,
    SelectContext.FIRST_EFFECT: effect_choice_turn,
    SelectContext.COIN_HEAD: effect_choice_turn,
    SelectContext.DAMAGE_COUNTER: damage_target_turn,
    SelectContext.DAMAGE_COUNTER_ANY: damage_target_turn,
    SelectContext.DAMAGE: damage_target_turn,
    SelectContext.REMOVE_DAMAGE_COUNTER: damage_target_turn,
    SelectContext.HEAL: damage_target_turn,
    SelectContext.DRAW_COUNT: count_turn,
    SelectContext.DAMAGE_COUNTER_COUNT: count_turn,
    SelectContext.REMOVE_DAMAGE_COUNTER_COUNT: count_turn,
    SelectContext.AFFECT_SPECIAL_CONDITION: special_condition_turn,
    SelectContext.RECOVER_SPECIAL_CONDITION: special_condition_turn,
    SelectContext.TO_BENCH: card_move_turn,
    SelectContext.TO_FIELD: card_move_turn,
    SelectContext.TO_HAND: card_move_turn,
    SelectContext.DISCARD: card_move_turn,
    SelectContext.TO_DECK: card_move_turn,
    SelectContext.TO_DECK_BOTTOM: card_move_turn,
    SelectContext.TO_PRIZE: card_move_turn,
    SelectContext.NOT_MOVE: card_move_turn,
    SelectContext.LOOK: card_move_turn,
    SelectContext.EFFECT_TARGET: card_move_turn,
    SelectContext.DISCARD_CARD_OR_ATTACHED_CARD: card_move_turn,
}


def route(obs: Observation) -> list[int]:
    """obs.select.context を見て対応する handlers/*.py の handle() を呼び出す。

    Args:
        obs: 通常ターンの Observation（obs.select is not None）。

    Returns:
        list[int]: obs.select.option に対する選択肢インデックスのリスト。
    """
    select = obs.select
    handler = _CONTEXT_HANDLERS.get(select.context)
    if handler is None and select.type == SelectType.YES_NO:
        # 対応表に無い YES_NO 系 context（将来 Enum に追加された分の安全網）。
        handler = yes_no_turn
    if handler is None:
        return fallback.safe_choice(obs)

    try:
        return handler.handle(obs)
    except Exception:
        # handlers 側の未知の例外でゲームを止めないための最終防衛ライン。
        return fallback.safe_choice(obs)
