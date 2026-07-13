"""クラスタ① 選択振り分け（ハンドラ）／担当B

対象 SelectContext: SWITCH, TO_ACTIVE

バトル場のポケモンを入れ替える場面（にげる後の交代、バトル場が空いたときの繰り出しなど）。
どのポケモンに交代するかの評価は main_turn_parts.pokemon_value
（B の board_evaluation.switch_eval + A の PokemonProfile）に委譲する。
priorities/retreat.py の撤退判断と同じ評価基準を使うことで、「にげるべきか」と
「にげるならどこへ」の判断が食い違わないようにしている。
"""

from cg.api import AreaType, Observation

from ptcg_ai.action_selection import fallback
from ptcg_ai.rule_based.main_turn_parts import pokemon_value


def handle(obs: Observation) -> list[int]:
    """SWITCH / TO_ACTIVE の選択肢から、pokemon_value のスコアが最も高い候補を選ぶ。"""
    state = obs.current
    your_index = state.yourIndex
    player = state.players[your_index]
    bench = list(player.bench)
    if not bench:
        return fallback.safe_choice(obs)

    target = pokemon_value.best_switch_target(bench, state, your_index)
    if target is None:
        return fallback.safe_choice(obs)

    bench_index = next((i for i, pokemon in enumerate(bench) if pokemon is target), None)
    if bench_index is None:
        return fallback.safe_choice(obs)

    for i, option in enumerate(obs.select.option):
        if option.area == AreaType.BENCH and option.index == bench_index and option.playerIndex == your_index:
            return [i]
    return fallback.safe_choice(obs)
