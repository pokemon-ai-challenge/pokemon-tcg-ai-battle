from cg.api import Observation

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.proposals import MainActionProposal
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS


def propose_pokemon_or_evolve_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """ポケモン展開や進化を候補として返す。"""
    if obs.current is None:
        return None

    your_index = obs.current.yourIndex
    player = obs.current.players[your_index]
    bench_space = player.benchMax - len(player.bench)

    if buckets.pokemon_play and bench_space > 0:
        score = MAIN_ACTION_BASE_WEIGHTS["pokemon_play"] + min(bench_space, 2) * 2
        return MainActionProposal(
            action=[buckets.pokemon_play[0]],
            score=score,
            label="pokemon_play",
        )
    if buckets.evolve:
        score = MAIN_ACTION_BASE_WEIGHTS["evolve"]
        return MainActionProposal(
            action=[buckets.evolve[0]],
            score=score,
            label="evolve",
        )
    return None


def propose_board_item_action(
    obs: Observation,
    buckets: MainOptionBuckets,
) -> MainActionProposal | None:
    """グッズ、スタジアム、どうぐで盤面を整える候補を返す。"""
    if buckets.item_play:
        return MainActionProposal(
            action=[buckets.item_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["board_item"],
            label="board_item",
        )
    if buckets.stadium_play:
        return MainActionProposal(
            action=[buckets.stadium_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["stadium"],
            label="stadium",
        )
    if buckets.tool_play:
        return MainActionProposal(
            action=[buckets.tool_play[0]],
            score=MAIN_ACTION_BASE_WEIGHTS["tool"],
            label="tool",
        )
    return None
