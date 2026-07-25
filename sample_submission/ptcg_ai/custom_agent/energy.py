"""エネルギー付けの対象・カード選択。

対象優先度: フーライン → ノコッチ → ノココッチ → シェイミ（ENERGY_REQUIRED_COUNT上限まで）。
カード優先度:
    対象がノコライン(65/66): フーライン充填数(energies>0の個体数)が2以上なら
        [リッチ, テレパス超, 基本超]、2未満なら [テレパス超, 基本超]（リッチは温存）。
    対象がフーライン/シェイミ: 常に [テレパス超, 基本超]（リッチエネルギーはノコライン専用）。
"""

from __future__ import annotations

from cg.api import AreaType, Observation, OptionType, Pokemon, SelectContext, State

from ptcg_ai.action_selection import fallback
from ptcg_ai.custom_agent import board_context, constants
from ptcg_ai.rule_based.card_move import common


def _target_rank(card_id: int) -> int:
    if card_id in constants.POKEMON_PRIORITY:
        return constants.POKEMON_PRIORITY.index(card_id)
    return len(constants.POKEMON_PRIORITY)


def card_order_for_target(target_card_id: int, state: State) -> list[int]:
    if target_card_id in constants.NOKO_LINE and board_context.fudin_line_charged_count(state) >= 2:
        return [constants.RICH_ENERGY, constants.TELEPATH_ENERGY, constants.BASIC_PSYCHIC_ENERGY]
    return [constants.TELEPATH_ENERGY, constants.BASIC_PSYCHIC_ENERGY]


def _needs_energy(pokemon: Pokemon) -> bool:
    required = constants.ENERGY_REQUIRED_COUNT.get(pokemon.id)
    if required is None:
        return False
    return board_context.energy_count(pokemon) < required


def choose_main_attach_option(select, state: State) -> int | None:
    """MAINの選択肢の中から、最も優先度の高い(対象, エネルギーカード)の組み合わせを選ぶ。

    OptionType.ATTACH は area/index が付け外しするエネルギーカード側、
    inPlayArea/inPlayIndex が対象ポケモン側を指す（cg/api.py参照）。
    """
    best_index = None
    best_key = None
    for i, option in enumerate(select.option):
        if option.type != OptionType.ATTACH:
            continue
        target = board_context.resolve_inplay_pokemon(option, state)
        card_id = common.resolve_card_id(option, state)
        if target is None or card_id is None or not _needs_energy(target):
            continue
        order = card_order_for_target(target.id, state)
        card_rank = order.index(card_id) if card_id in order else len(order)
        active_bonus = 0 if option.inPlayArea == AreaType.ACTIVE else 1
        key = (_target_rank(target.id), card_rank, active_bonus)
        if best_key is None or key < best_key:
            best_key = key
            best_index = i
    return best_index


def choose_attach_target(obs: Observation) -> list[int]:
    """ATTACH_FROM: 効果で対象ポケモンを選ぶ場面（例: ワンダーパッチ）。提示された候補の中から、
    優先度の高い(必要エネルギー未充足の)ポケモンを選ぶ。
    """
    state = obs.current
    best_index = None
    best_rank = None
    for i, option in enumerate(obs.select.option):
        pokemon = common.resolve_pokemon(option, state)
        if pokemon is None or not _needs_energy(pokemon):
            continue
        rank = _target_rank(pokemon.id)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_index = i
    if best_index is None:
        return fallback.safe_choice(obs)
    return [best_index]


def choose_attach_card(obs: Observation) -> list[int]:
    """ATTACH_TO: 効果でどのエネルギーカードを付けるか選ぶ場面。対象がまだ不明なことが多いため、
    ノコライン以外向けの既定順(テレパス超>基本超)を使う（リッチエネルギー専用効果でなければ十分）。
    """
    state = obs.current
    order = [constants.TELEPATH_ENERGY, constants.BASIC_PSYCHIC_ENERGY, constants.RICH_ENERGY]

    def score(option) -> float:
        card_id = common.resolve_card_id(option, state)
        if card_id in order:
            return float(len(order) - order.index(card_id))
        return 0.0

    return common.pick_top(obs.select, score)


def handle_energy_tool_select(obs: Observation) -> list[int]:
    """ATTACH_FROM/ATTACH_TO 以外(DETACH_FROM/SWITCH_ENERGY(_CARD)/TO_HAND_ENERGY/TO_DECK_ENERGY)。
    主力アタッカー(フーライン)のエネルギーを不用意に手放さないよう、価値の低いポケモンから選ぶ。
    """
    state = obs.current
    context = obs.select.context
    if context == SelectContext.ATTACH_FROM:
        return choose_attach_target(obs)
    if context == SelectContext.ATTACH_TO:
        return choose_attach_card(obs)

    def score(option) -> float:
        pokemon = common.resolve_pokemon(option, state)
        if pokemon is None:
            return 0.0
        return -float(len(constants.POKEMON_PRIORITY) - _target_rank(pokemon.id))

    return common.pick_top(obs.select, score)
