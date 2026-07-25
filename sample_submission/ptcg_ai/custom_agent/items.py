"""グッズの使用条件・対象選択。"""

from __future__ import annotations

from cg.api import SelectData, State

from ptcg_ai.custom_agent import board_context, constants, matchup
from ptcg_ai.rule_based.card_move import common


def _rare_candy_condition(state: State) -> bool:
    """ふしぎなアメ: バトル場がケーシィ、かつ手札にユンゲラーが無くフーディンがある場合。"""
    own_active = board_context.own(state).active
    if not own_active or own_active[0] is None or own_active[0].id != constants.CASEY:
        return False
    hand = board_context.hand_ids(state)
    return constants.KADABRA not in hand and constants.ALAKAZAM in hand


def _enhanced_hammer_condition(state: State) -> bool:
    """改造ハンマー: 相手のバトル場のポケモンに特殊エネルギーが付いている場合。"""
    return board_context.opponent_active_has_special_energy(state)


def _buddy_buddy_poffin_condition(state: State) -> bool:
    """なかよしポフィン: ケーシィ/ノコッチのいずれかが場にも手札にも無い場合。"""
    ids_present = board_context.own_ids_present(state)
    return any(cid not in ids_present for cid in (constants.CASEY, constants.DUNSPARCE))


def _night_stretcher_condition(state: State) -> bool:
    """夜のタンカ: トラッシュに回収できるポケモンが1体以上いる場合。"""
    return board_context.discard_pokemon_count(state) >= 1


def _sacred_ash_condition(state: State) -> bool:
    """せいなるはい: トラッシュにポケモンが3体以上ある場合。"""
    return board_context.discard_pokemon_count(state) >= 3


def _wondrous_patch_condition(state: State) -> bool:
    """ワンダーパッチ: 手持ちにエネルギーが無く、トラッシュに基本【超】エネルギーがある場合。"""
    hand = board_context.hand_ids(state)
    hand_has_energy = any(cid in hand for cid in constants.ENERGY_CARD_IDS)
    return not hand_has_energy and constants.BASIC_PSYCHIC_ENERGY in board_context.discard_ids(state)


def _poke_pad_condition(state: State) -> bool:
    """ポケパッド: 手札にあれば基本的に使用する（山札を消費するため手札20枚以上なら見送る）。"""
    return board_context.own_hand_count(state) < constants.DECK_DRAW_STOP_HAND_SIZE


_CONDITIONS = {
    constants.RARE_CANDY: _rare_candy_condition,
    constants.ENHANCED_HAMMER: _enhanced_hammer_condition,
    constants.BUDDY_BUDDY_POFFIN: _buddy_buddy_poffin_condition,
    constants.NIGHT_STRETCHER: _night_stretcher_condition,
    constants.SACRED_ASH: _sacred_ash_condition,
    constants.WONDROUS_PATCH: _wondrous_patch_condition,
    constants.POKE_PAD: _poke_pad_condition,
}


def is_usable(card_id: int, state: State) -> bool:
    condition = _CONDITIONS.get(card_id)
    return condition(state) if condition is not None else True


# --- 対象選択（TO_BENCH/TO_HAND/TO_DECK 等、select.effect.id で判定） ---


def choose_buddy_buddy_poffin_target(select: SelectData, state: State) -> list[int]:
    """なかよしポフィン: ケーシィ/ノコッチで場・手札に無い方を優先して1枚ずつ。
    両方すでにある場合はケーシィ2枚目以降を優先。
    """
    ids_present = board_context.own_ids_present(state)
    missing = [cid for cid in (constants.CASEY, constants.DUNSPARCE) if cid not in ids_present]
    order = missing + [cid for cid in (constants.CASEY, constants.DUNSPARCE) if cid not in missing]

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in order:
            return float(len(order) - order.index(card_id))
        return 0.0

    return common.pick_top(select, score)


def choose_night_stretcher_target(select: SelectData, state: State) -> list[int]:
    """夜のタンカ: 基本ポケモンを優先し、進化先が場・手札に無いラインを最優先する。
    ポケモンが選べない場合はエネルギーカードを優先順位順に選ぶ。
    """
    ids_present = board_context.own_ids_present(state)

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id is None:
            return 0.0
        if card_id in constants.BASIC_POKEMON_IDS:
            successor = constants.EVOLUTION_SUCCESSOR.get(card_id)
            urgent = successor is not None and successor not in ids_present
            base = constants.POKEMON_PRIORITY.index(card_id) if card_id in constants.POKEMON_PRIORITY else 99
            return (1000.0 if urgent else 500.0) - base
        if card_id in constants.ENERGY_CARD_IDS:
            return 100.0 - constants.ENERGY_CARD_IDS.index(card_id)
        return 0.0

    return common.pick_top(select, score)


def choose_sacred_ash_target(select: SelectData, state: State) -> list[int]:
    """せいなるはい: up to 5。判断材料が薄いため POKEMON_PRIORITY 順に上限まで選ぶ。"""

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in constants.POKEMON_PRIORITY:
            return float(len(constants.POKEMON_PRIORITY) - constants.POKEMON_PRIORITY.index(card_id))
        return 0.0

    return common.pick_top(select, score)


def choose_poke_pad_target(select: SelectData, state: State) -> list[int]:
    """ポケパッド: 場・手札に無いポケモンを優先し、それ以外は POKEMON_PRIORITY 順。"""
    ids_present = board_context.own_ids_present(state)

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id is None:
            return 0.0
        base = float(len(constants.POKEMON_PRIORITY) - constants.POKEMON_PRIORITY.index(card_id)) if card_id in constants.POKEMON_PRIORITY else 0.0
        missing_bonus = 1000.0 if card_id not in ids_present else 0.0
        return base + missing_bonus + matchup.card_priority_boost(card_id)

    return common.pick_top(select, score)


_TARGET_CHOOSERS = {
    constants.BUDDY_BUDDY_POFFIN: choose_buddy_buddy_poffin_target,
    constants.NIGHT_STRETCHER: choose_night_stretcher_target,
    constants.SACRED_ASH: choose_sacred_ash_target,
    constants.POKE_PAD: choose_poke_pad_target,
}


def choose_target(select: SelectData, state: State) -> list[int] | None:
    """select.effect.id を見て、対応するグッズの対象選択ロジックへ委譲する。該当が無ければNone。"""
    effect = select.effect
    if effect is None:
        return None
    chooser = _TARGET_CHOOSERS.get(effect.id)
    if chooser is None:
        return None
    return chooser(select, state)
