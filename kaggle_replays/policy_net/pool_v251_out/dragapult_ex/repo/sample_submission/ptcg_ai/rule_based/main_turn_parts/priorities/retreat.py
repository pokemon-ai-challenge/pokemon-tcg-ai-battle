"""クラスタ③ メイン行動の意思決定（カテゴリ別提案）／担当B

カテゴリ: retreat（にげる）

バトル場のポケモンが次ターン倒されやすい（decision.evaluation.board_features）場合や、
より良いアタッカーがベンチにいる場合に、にげる/交代を提案する。
逃げ先の評価は main_turn_parts.pokemon_value（B の switch_eval + A の PokemonProfile）を使う。
今ターン既に交代済みなら提案しない（State.retreated を確認する）。
"""

from cg.api import Observation, OptionType

from ptcg_ai.board_evaluation import board_features, energy_requirements
from ptcg_ai.rule_based.main_turn_parts import pokemon_value
from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal
from ptcg_ai.shared import card_cache

_URGENT_RETREAT_SCORE = 5.0
_IMPROVEMENT_RETREAT_SCORE = 1.0
# switch_target_value 側にだけ安全ボーナス（switch_eval._SAFE_BONUS=3.0）や
# KO_REPLACEMENT_PRIORITY ボーナス（最大4.0）が乗る非対称な比較になっているため、
# 僅差のスコア差だけでは撤退を提案しないようにするマージン。この値未満の差は
# 「今のバトルポケモンでも大差ない」とみなして無駄な逃げを避ける。
_IMPROVEMENT_MARGIN = 3.0


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

    active_card = card_cache.get_card(active.id)
    if not energy_requirements.can_afford_retreat(active_card.retreatCost, active.energies):
        # にげるコストを払えない（合法手として出ていても念のための防御）。
        return None

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
        if bench_score <= current_score + _IMPROVEMENT_MARGIN:
            return None

    retreat_index = next((i for i, option in enumerate(obs.select.option) if option.type == OptionType.RETREAT), None)
    if retreat_index is None:
        return None

    score = _URGENT_RETREAT_SCORE if needs_retreat else _IMPROVEMENT_RETREAT_SCORE
    return ActionProposal(category="retreat", select=[retreat_index], score=score, reason="retreat")
