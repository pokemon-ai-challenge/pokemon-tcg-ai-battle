"""MAINターンの意思決定パイプライン（立ち回りの優先順位）。

優先順位:
    1. 確定リーサル攻撃
    2. 特性（にげあしドロー等。手札20枚以上なら見送る）
    3. 進化（フーディン系列→ノココッチ）
    4. ふしぎなアメ
    5. グッズ（なかよしポフィン→改造ハンマー→ワンダーパッチ→夜のタンカ→せいなるはい→ポケパッド）
    6. エネルギー付け（1ターン1回）
    7. にげる
    8. サポート（1ターン1枚。ボスの指令は「攻撃直前に判断する」仕様のため、進化/エネルギー付け/
       にげるがすべて終わった後のこの時点で判定し、条件成立時は最優先で使用する。不成立なら
       クセロシキのたくらみ→トウコ→ヒカリ→スイレンのお世話の順で通常のサポートを検討する）
    9. スタジアム
    10. 攻撃
    11. ターン終了

同じ1ターン内でもMAIN選択はアクション1つごとに何度も呼ばれる（このパイプラインは毎回
最初から再評価される）ため、コード上6〜7番が8番より先にあっても、ボスの指令の判定は
実際にそのターンの進化・エネルギー付け・にげるが完了した後の状態を見て行われる。
"""

from __future__ import annotations

from cg.api import Observation, OptionType, SelectData, State

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import (
    attack,
    board_context,
    constants,
    energy,
    evolution,
    items,
    retreat_switch,
    stadium,
    supporters,
)
from ptcg_ai.rule_based.card_move import common

_ABILITY_CARD_IDS = (constants.KADABRA, constants.ALAKAZAM, constants.DUDUNSPARCE)


def decide(obs: Observation) -> list[int]:
    select = obs.select
    state = obs.current

    index = _lethal_attack_index(select, state)
    if index is not None:
        return [index]

    index = _ability_index(select, state)
    if index is not None:
        return [index]

    index = evolution.choose_main_evolve_option(select, state)
    if index is not None:
        return [index]

    index = _play_index(select, state, constants.RARE_CANDY, items.is_usable)
    if index is not None:
        return [index]

    for item_id in constants.ITEM_USE_PRIORITY:
        index = _play_index(select, state, item_id, items.is_usable)
        if index is not None:
            return [index]

    if not state.energyAttached:
        index = energy.choose_main_attach_option(select, state)
        if index is not None:
            return [index]

    if not state.retreated and retreat_switch.should_retreat(state):
        index = retreat_switch.choose_main_retreat_option(select, state)
        if index is not None:
            return [index]

    if not state.supporterPlayed:
        # ボスの指令は「攻撃直前に判断する」仕様のため、進化/エネルギー付け/にげるが
        # すべて済んだこの時点の状態で判定する（最優先で温存する）。
        if supporters.boss_orders_target(state) is not None:
            index = _find_play_option(select, state, constants.BOSS_ORDERS)
            if index is not None:
                return [index]
        for supporter_id in constants.SUPPORTER_USE_PRIORITY:
            index = _play_index(select, state, supporter_id, supporters.is_usable)
            if index is not None:
                return [index]

    if not state.stadiumPlayed:
        index = _play_index(select, state, constants.BATTLE_COLOSSEUM, stadium.is_usable)
        if index is not None:
            return [index]

    index = attack.choose_main_attack_option(select, state)
    if index is not None:
        return [index]

    return _end_turn_index(obs)


def _find_play_option(select: SelectData, state: State, card_id: int) -> int | None:
    for i, option in enumerate(select.option):
        if option.type == OptionType.PLAY and common.resolve_card_id(option, state) == card_id:
            return i
    return None


def _play_index(select: SelectData, state: State, card_id: int, usable_fn) -> int | None:
    index = _find_play_option(select, state, card_id)
    if index is None:
        return None
    if not usable_fn(card_id, state):
        return None
    return index


def _lethal_attack_index(select: SelectData, state: State) -> int | None:
    lethal_attack = attack.find_lethal_attack(state)
    if lethal_attack is None:
        return None
    for i, option in enumerate(select.option):
        if option.type == OptionType.ATTACK and option.attackId == lethal_attack.attackId:
            return i
    return None


def _ability_index(select: SelectData, state: State) -> int | None:
    if board_context.own_hand_count(state) >= constants.DECK_DRAW_STOP_HAND_SIZE:
        return None
    best_index = None
    best_rank = None
    for i, option in enumerate(select.option):
        if option.type != OptionType.ABILITY:
            continue
        card_id = common.resolve_card_id(option, state)
        if card_id not in _ABILITY_CARD_IDS:
            continue
        rank = constants.POKEMON_PRIORITY.index(card_id) if card_id in constants.POKEMON_PRIORITY else 99
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_index = i
    return best_index


def _end_turn_index(obs: Observation) -> list[int]:
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.END:
            return [i]
    return fallback.safe_choice(obs)
