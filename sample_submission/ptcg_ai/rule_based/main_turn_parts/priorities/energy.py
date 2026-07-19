"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: energy（エネルギー付与）

今ターンのエネルギー付与を提案する。対象選定は
decision.main_turn_parts.energy_eval.best_energy_target を使う。
1ターン1回までの制限（State.energyAttached）を尊重する。
"""

from cg.api import AreaType, Observation, OptionType, Pokemon

from ptcg_ai.rule_based.main_turn_parts import energy_eval
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal

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

    for i, option in enumerate(obs.select.option):
        if option.type != OptionType.ATTACH:
            continue
        if option.inPlayArea == area and option.inPlayIndex == index:
            return ActionProposal(category="energy", select=[i], score=_ENERGY_ATTACH_SCORE, reason="attach energy")
    return None


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
