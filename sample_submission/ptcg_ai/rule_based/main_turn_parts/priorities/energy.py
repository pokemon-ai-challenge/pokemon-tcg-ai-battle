"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: energy（エネルギー付与）

今ターンのエネルギー付与を提案する。対象選定は
decision.main_turn_parts.energy_eval.best_energy_target を使う。
対象に付けるエネルギーが手札に複数種類ある場合は、
deck_plan.ENERGY_CARD_PRIORITY_RULES（profile_registry.get_energy_card_priority_rules）に
沿って優先度の高いカードを選ぶ。1ターン1回までの制限（State.energyAttached）を尊重する。
"""

from cg.api import AreaType, Observation, OptionType, Pokemon

from ptcg_ai.rule_based.card_move import common
from ptcg_ai.rule_based.main_turn_parts import energy_eval
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared.profile_types import EnergyCardContext

_ENERGY_ATTACH_SCORE = 1.0


def propose(obs: Observation) -> ActionProposal | None:
    """エネルギー付与の行動を1つ提案する。付与済み/対象なしの場合は None。"""
    if obs.current.energyAttached:
        return None

    target = energy_eval.best_energy_target(obs)
    if target is None:
        return None

    location = _locate(obs, target)
    if location is None:
        return None
    area, index = location

    matching = [
        i
        for i, option in enumerate(obs.select.option)
        if option.type == OptionType.ATTACH and option.inPlayArea == area and option.inPlayIndex == index
    ]
    if not matching:
        return None

    best_index = _select_energy_card(obs, matching, target)
    return ActionProposal(category="energy", select=[best_index], score=_ENERGY_ATTACH_SCORE, reason="attach energy")


def _select_energy_card(obs: Observation, option_indices: list[int], target: Pokemon) -> int:
    """対象に付けるエネルギーが複数候補ある場合、ENERGY_CARD_PRIORITY_RULES で最良のものを選ぶ。"""
    if len(option_indices) == 1:
        return option_indices[0]

    context = EnergyCardContext(target_card_id=target.id, target_energy_count=len(target.energies))
    order = energy_eval.resolve_energy_card_order(context)

    def rank(index: int) -> int:
        card_id = common.resolve_card_id(obs.select.option[index], obs.current)
        if card_id in order:
            return order.index(card_id)
        return len(order)  # 優先順位リストに無いものは最後扱い

    return min(option_indices, key=rank)


def _locate(obs: Observation, pokemon: Pokemon) -> tuple[AreaType, int] | None:
    """対象ポケモンが自分の場のどこ（バトル場/ベンチの何番目）にいるかを特定する。"""
    player = obs.current.players[obs.current.yourIndex]
    for i, active in enumerate(player.active):
        if active is pokemon:
            return AreaType.ACTIVE, i
    for i, bench_pokemon in enumerate(player.bench):
        if bench_pokemon is pokemon:
            return AreaType.BENCH, i
    return None
