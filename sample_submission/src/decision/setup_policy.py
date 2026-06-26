from cg.api import Observation, OptionType
from src.decision.utils import get_card_id_from_option
from src.knowledge.deck_plan import get_deck_plan
from src.knowledge.decision_trace import trace


def choose_setup_active(obs: Observation) -> list[int]:
    """SETUP_ACTIVE_POKEMON: バトル場に出すポケモンを DeckPlan の active_priority で選ぶ。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    turn = state.turn if state else 0

    for priority_id in plan.active_priority:
        for i, opt in enumerate(options):
            if opt.type != OptionType.CARD:
                continue
            card_id = get_card_id_from_option(opt, state) if state else opt.cardId
            if card_id == priority_id:
                trace(turn, "SETUP_ACTIVE", i, f"priority_pokemon={priority_id}")
                return [i]

    trace(turn, "SETUP_ACTIVE", 0, "default_first")
    return [0]


def choose_setup_bench(obs: Observation) -> list[int]:
    """SETUP_BENCH_POKEMON: ベンチに出すポケモンを DeckPlan の bench_priority で選ぶ。"""
    plan = get_deck_plan()
    state = obs.current
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount
    turn = state.turn if state else 0

    chosen: list[int] = []
    used: set[int] = set()

    for priority_id in plan.bench_priority:
        if len(chosen) >= max_count:
            break
        for i, opt in enumerate(options):
            if i in used or opt.type != OptionType.CARD:
                continue
            card_id = get_card_id_from_option(opt, state) if state else opt.cardId
            if card_id == priority_id:
                chosen.append(i)
                used.add(i)
                break

    # 最低限の選択数を満たすために残りを補完
    if len(chosen) < min_count:
        for i in range(len(options)):
            if i not in used:
                chosen.append(i)
                used.add(i)
                if len(chosen) >= min_count:
                    break

    trace(turn, "SETUP_BENCH", chosen, f"place_{len(chosen)}_on_bench")
    return chosen
