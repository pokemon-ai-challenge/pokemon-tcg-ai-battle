"""にげる／交代先の選択（気絶後の後継優先順位: フーライン→ノコッチ→ノココッチ→シェイミ）。

SelectContext.SWITCH / TO_ACTIVE は「自分のバトル場を埋める」場合と、ボスの指令の効果で
「相手のベンチポケモンをバトル場に出す」場合の両方で使われるため、option.playerIndex を見て
どちらの場面かを判定する。
"""

from __future__ import annotations

from cg.api import Observation, OptionType, Pokemon, State

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import energy_requirements
from ptcg_ai.custom_agent import board_context, constants, supporters
from ptcg_ai.rule_based.card_move import common
from ptcg_ai.shared import card_cache


def _replacement_rank(card_id: int) -> int:
    if card_id in constants.KO_REPLACEMENT_PRIORITY:
        return constants.KO_REPLACEMENT_PRIORITY.index(card_id)
    return len(constants.KO_REPLACEMENT_PRIORITY)


def best_switch_target(candidates: list[Pokemon]) -> Pokemon | None:
    """気絶後の後継優先順位に沿って交代先を選ぶ。ノココッチ・シェイミは他に候補が無い場合のみ。"""
    if not candidates:
        return None
    preferred = [p for p in candidates if p.id not in constants.BENCH_ONLY]
    pool = preferred if preferred else candidates
    return min(pool, key=lambda p: _replacement_rank(p.id))


def should_retreat(state: State) -> bool:
    """にげるのはフーラインがメインに出ていないときのみ。

    フーラインがバトル場にいる間は（相手ターンでの被弾リスクがあっても）にげない
    ——フーラインの役目はバトル場に留まって攻撃することであり、気絶した場合の後継は
    KO_REPLACEMENT_PRIORITY 側の役目。
    ノコッチがバトル場にいて、手札にノココッチがある場合も、進化すれば特性
    （にげあしドロー）で自身が山札に戻るため、コストを払ってにげる必要は無い。
    それ以外（ノコッチ単独で進化先が無い、または本来ベンチ固定のノココッチ・シェイミが
    何らかの事情でバトル場にいる場合）は、フーラインなどより優れた後継がベンチにいれば
    にげる。
    """
    if state.retreated:
        return False
    own_active_slot = board_context.own(state).active
    if not own_active_slot or own_active_slot[0] is None:
        return False
    active = own_active_slot[0]

    if active.id in constants.FUDIN_LINE:
        return False
    if active.id == constants.DUNSPARCE and constants.DUDUNSPARCE in board_context.hand_ids(state):
        return False

    bench = [p for p in board_context.own(state).bench if p is not None]
    if not bench:
        return False

    active_card = card_cache.get_card(active.id)
    if not energy_requirements.can_afford_retreat(active_card.retreatCost, active.energies):
        return False

    target = best_switch_target(bench)
    if target is None:
        return False
    return _replacement_rank(target.id) < _replacement_rank(active.id)


def choose_main_retreat_option(select, state: State) -> int | None:
    """MAINの選択肢からRETREATオプションを1つ返す(にげる先自体は続くSWITCH選択で決まる)。"""
    for i, option in enumerate(select.option):
        if option.type == OptionType.RETREAT:
            return i
    return None


def handle_switch_select(obs: Observation) -> list[int]:
    """SWITCH / TO_ACTIVE: 自分のバトル場を埋める場合とボスの指令で相手を出す場合を区別する。"""
    state = obs.current
    select = obs.select
    sample = next((option for option in select.option if option.playerIndex is not None), None)

    if sample is not None and sample.playerIndex != state.yourIndex:
        target = supporters.boss_orders_target(state)
        if target is not None:
            for i, option in enumerate(select.option):
                pokemon = common.resolve_pokemon(option, state)
                if pokemon is not None and pokemon.serial == target.serial:
                    return [i]
        return fallback.safe_choice(obs)

    bench = [p for p in board_context.own(state).bench if p is not None]
    if not bench:
        return fallback.safe_choice(obs)
    target = best_switch_target(bench)
    if target is None:
        return fallback.safe_choice(obs)
    for i, option in enumerate(select.option):
        pokemon = common.resolve_pokemon(option, state)
        if pokemon is not None and pokemon.serial == target.serial:
            return [i]
    return fallback.safe_choice(obs)
