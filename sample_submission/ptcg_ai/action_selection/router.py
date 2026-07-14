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

from cg.api import Observation, SelectContext

from ptcg_ai.action_selection import fallback


def route(obs: Observation) -> list[int]:
    """obs.select.context を見て対応する handlers/*.py の handle() を呼び出す。

    Args:
        obs: 通常ターンの Observation（obs.select is not None）。

    Returns:
        list[int]: obs.select.option に対する選択肢インデックスのリスト。
    """
    raise NotImplementedError


def _context_to_handler_name(context: SelectContext) -> str:
    """SelectContext から対応する handlers/*.py のモジュール名を引く。

    上部の対応表をそのままコードに落とし込む場所。未対応の context は
    fallback に回すため None 等を返す設計にしてもよい。
    """
    raise NotImplementedError
