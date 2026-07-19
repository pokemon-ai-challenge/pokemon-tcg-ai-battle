"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: ATTACH_FROM, ATTACH_TO, DETACH_FROM, DISCARD_ENERGY_CARD,
DISCARD_TOOL_CARD, SWITCH_ENERGY_CARD, DISCARD_ENERGY, TO_HAND_ENERGY,
TO_DECK_ENERGY, SWITCH_ENERGY

エネルギー・ポケモンのどうぐの付け外し・入れ替えに関する選択。
どのポケモンに付けるべきかの評価は decision.evaluation.energy_requirements /
decision.main_turn_parts.energy_eval を参照する。
"""

from cg.api import AreaType, Observation, SelectContext

from ptcg_ai.action_selection import fallback
from ptcg_ai.board_evaluation import board_features
from ptcg_ai.rule_based.card_move import common, discard
from ptcg_ai.rule_based.main_turn_parts import energy_eval
from ptcg_ai.shared import profile_registry

# discard.choose は DISCARD 系 SelectContext（protected_card_ids を避ける判断）を
# まとめて面倒みているため、ここでもそのまま使い回す。
_DISCARD_LIKE_CONTEXTS = (
    SelectContext.DISCARD_ENERGY_CARD,
    SelectContext.DISCARD_TOOL_CARD,
    SelectContext.DISCARD_ENERGY,
)


def handle(obs: Observation) -> list[int]:
    """ATTACH/DETACH/DISCARD/SWITCH のエネルギー・どうぐ系選択肢を処理する。"""
    context = obs.select.context
    if context in _DISCARD_LIKE_CONTEXTS:
        return discard.choose(obs.select, obs.current)
    if context == SelectContext.ATTACH_FROM:
        return _choose_attach_target(obs)
    if context == SelectContext.ATTACH_TO:
        return _choose_attach_card(obs)
    # DETACH_FROM / SWITCH_ENERGY_CARD / SWITCH_ENERGY / TO_HAND_ENERGY / TO_DECK_ENERGY:
    # 場から手放す/動かす対象は、既定では最も価値の低いポケモンから選ぶ
    # （主力アタッカーのエネルギーを不用意に失わないようにする）。
    return common.pick_top(obs.select, lambda option: -_owner_attacker_score(option, obs.current))


def _choose_attach_target(obs: Observation) -> list[int]:
    """ATTACH_FROM: エネルギー/どうぐを付けるべきポケモンを選ぶ。"""
    target = energy_eval.best_energy_target(obs)
    if target is None:
        return fallback.safe_choice(obs)

    player = obs.current.players[obs.current.yourIndex]
    location = None
    for i, active in enumerate(player.active):
        if active is target:
            location = (AreaType.ACTIVE, i)
    for i, bench_pokemon in enumerate(player.bench):
        if bench_pokemon is target:
            location = (AreaType.BENCH, i)
    if location is None:
        return fallback.safe_choice(obs)
    area, index = location

    for i, option in enumerate(obs.select.option):
        if option.area == area and option.index == index:
            return [i]
    return fallback.safe_choice(obs)


def _choose_attach_card(obs: Observation) -> list[int]:
    """ATTACH_TO: 手札のどのエネルギー/どうぐを付けるか選ぶ。"""
    plan = profile_registry.get_deck_plan()

    def score(option) -> float:
        card_id = common.resolve_card_id(option, obs.current)
        if card_id in plan.energy_priority:
            return float(len(plan.energy_priority) - plan.energy_priority.index(card_id))
        return 0.0

    return common.pick_top(obs.select, score)


def _owner_attacker_score(option, state) -> float:
    """option の area/index が指すポケモンの attacker_score を返す（不明なら0.0）。"""
    pokemon = common.resolve_pokemon(option, state)
    if pokemon is None:
        return 0.0
    return board_features.attacker_score(pokemon)
