"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: retreat（にげる）

バトル場のポケモンが次ターン倒されやすい（decision.evaluation.board_features）場合や、
より良いアタッカーがベンチにいる場合に、にげる/交代を提案する。
逃げ先の評価は main_turn_parts.pokemon_value（B の switch_eval + A の PokemonProfile）を使う。
今ターン既に交代済みなら提案しない（State.retreated を確認する）。
"""

from cg.api import Observation, OptionType

from ptcg_ai.board_evaluation import board_features
from ptcg_ai.rule_based.main_turn_parts import pokemon_value
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal

_URGENT_RETREAT_SCORE = 5.0
_IMPROVEMENT_RETREAT_SCORE = 1.0


def propose(obs: Observation) -> ActionProposal | None:
    """にげる行動を1つ提案する。交代の必要が無ければ None。"""
    state = obs.current
    if state.retreated:
        return None

    your_index = state.yourIndex
    player = state.players[your_index]
    if not player.active or player.active[0] is None:
        return None
    active = player.active[0]

    bench = list(player.bench)
    if not bench:
        return None

    best_bench_target = pokemon_value.best_switch_target(bench, state, your_index)
    if best_bench_target is None:
        return None

    needs_retreat = board_features.is_likely_ko_next_turn(active, state, your_index)
    if not needs_retreat:
        current_score = pokemon_value.active_value(active, state, your_index)
        bench_score = pokemon_value.switch_target_value(best_bench_target, state, your_index)
        if bench_score <= current_score:
            return None

    retreat_index = next((i for i, option in enumerate(obs.select.option) if option.type == OptionType.RETREAT), None)
    if retreat_index is None:
        return None

    score = _URGENT_RETREAT_SCORE if needs_retreat else _IMPROVEMENT_RETREAT_SCORE
    return ActionProposal(category="retreat", select=[retreat_index], score=score, reason="retreat")
