from cg.api import Observation, SelectContext
from src.knowledge.decision_trace import trace


def _get_turn(obs: Observation) -> int:
    return obs.current.turn if obs.current else 0


def choose_default_legal(obs: Observation) -> list[int]:
    """デフォルトハンドラ: 最低限の数を先頭から選ぶ。"""
    return list(range(obs.select.minCount))


def choose_yes(obs: Observation) -> list[int]:
    """YES/NO 選択で YES を返す（インデックス 0 が常に YES）。"""
    trace(_get_turn(obs), str(obs.select.context), 0, "always_yes")
    return [0]


def choose_no(obs: Observation) -> list[int]:
    """YES/NO 選択で NO を返す（インデックス 1 が常に NO）。"""
    trace(_get_turn(obs), str(obs.select.context), 1, "always_no")
    return [1]


def choose_max_count(obs: Observation) -> list[int]:
    """COUNT 選択で最大値（最も大きい数値の選択肢）を選ぶ。"""
    options = obs.select.option
    best_i = max(range(len(options)), key=lambda i: options[i].number or 0)
    trace(_get_turn(obs), str(obs.select.context), best_i, "max_count")
    return [best_i]


def choose_action(obs: Observation) -> list[int]:
    """SelectContext に応じて適切なハンドラに振り分ける。"""
    from src.decision.setup_policy import choose_setup_active, choose_setup_bench
    from src.decision.main_policy import choose_main_action
    from src.decision.target_policy import (
        choose_attack, choose_switch, choose_to_bench, choose_to_hand,
        choose_discard, choose_attach_from, choose_attach_to,
        choose_evolves_from, choose_evolves_to, choose_evolve,
    )

    ctx = obs.select.context

    dispatch = {
        SelectContext.MAIN: choose_main_action,
        SelectContext.SETUP_ACTIVE_POKEMON: choose_setup_active,
        SelectContext.SETUP_BENCH_POKEMON: choose_setup_bench,
        SelectContext.ATTACK: choose_attack,
        SelectContext.SWITCH: choose_switch,
        SelectContext.TO_ACTIVE: choose_switch,     # 強制アクティブもSWITCHと同じ基準
        SelectContext.TO_BENCH: choose_to_bench,
        SelectContext.TO_FIELD: choose_to_bench,    # フィールドに出す = ベンチと同基準
        SelectContext.TO_HAND: choose_to_hand,
        SelectContext.DISCARD: choose_discard,
        SelectContext.EVOLVES_FROM: choose_evolves_from,
        SelectContext.EVOLVES_TO: choose_evolves_to,
        SelectContext.EVOLVE: choose_evolve,
        SelectContext.ATTACH_FROM: choose_attach_from,
        SelectContext.ATTACH_TO: choose_attach_to,
        # YES を返すコンテキスト
        SelectContext.IS_FIRST: choose_yes,
        SelectContext.COIN_HEAD: choose_yes,
        SelectContext.ACTIVATE: choose_yes,
        SelectContext.FIRST_EFFECT: choose_yes,
        SelectContext.MULLIGAN: choose_yes,       # マリガン質問が来る = Basic がないので YES
        # NO を返すコンテキスト
        SelectContext.MORE_DEVOLVE: choose_no,    # 過度な退化は避ける
        # COUNT: 最大値を選ぶ
        SelectContext.DRAW_COUNT: choose_max_count,
        SelectContext.DAMAGE_COUNTER_COUNT: choose_max_count,
        SelectContext.REMOVE_DAMAGE_COUNTER_COUNT: choose_max_count,
    }

    handler = dispatch.get(ctx, choose_default_legal)
    return handler(obs)
