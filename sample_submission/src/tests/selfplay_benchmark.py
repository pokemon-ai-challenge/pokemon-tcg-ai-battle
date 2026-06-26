"""Phase 5 vs Phase 6 自己対戦ベンチマーク。

Phase 5 = KO リスク計算なし・エネルギー上限なし・buffer_pokemon_ids なし の旧ロジック
Phase 6 = 現行コード（main.py / choose_action）

Usage:
    cd sample_submission
    python src/tests/selfplay_benchmark.py --games 30
    python src/tests/selfplay_benchmark.py --games 30 --decks dragapult_ex hydrapple_ex
"""
import sys
import os
import argparse
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cg.api import (
    Observation, to_observation_class,
    OptionType, CardType, AreaType,
)
from cg.game import battle_finish, battle_select, battle_start
from main import agent as agent_v6
from src.knowledge.meta_decks import DECK_CATALOG, tier_s_decks
from src.knowledge.deck_plan import get_deck_plan
from src.knowledge.card_database import get_card_db, get_attack_db
from src.knowledge.opponent_model import get_opponent_model, IONO_ID
from src.knowledge.matchup_policy import get_matchup_override
from src.knowledge.ex_immune import opponent_active_blocks_ex, my_active_is_ex
from src.decision.utils import get_card_id_from_option, get_inplay_pokemon_id
from src.decision.evaluator import get_evaluator
from src.decision.router import choose_action as _choose_action_v6


# ---------------------------------------------------------------------------
# Phase 5 ロジック（変更前の動作を忠実に再現）
# ---------------------------------------------------------------------------

def _v5_pick_best_attach(obs: Observation) -> int:
    """Phase 5 のエネルギー付与選択: KO リスクなし・エネルギー上限なし。"""
    plan = get_deck_plan()
    state = obs.current

    effective_priority = plan.energy_priority
    if state and opponent_active_blocks_ex(state) and plan.non_ex_energy_priority:
        effective_priority = plan.non_ex_energy_priority

    attach_options = [
        i for i, opt in enumerate(obs.select.option)
        if opt.type == OptionType.ATTACH
    ]
    best_i = attach_options[0]
    best_score = -1

    for i in attach_options:
        opt = obs.select.option[i]
        target_id = get_inplay_pokemon_id(opt, state) if state else None

        if target_id in effective_priority:
            priority_score = len(effective_priority) - effective_priority.index(target_id)
            energy_count = 0
            if state:
                player = state.players[state.yourIndex]
                if opt.inPlayArea == AreaType.ACTIVE and player.active and opt.inPlayIndex is not None:
                    poke = player.active[opt.inPlayIndex] if opt.inPlayIndex < len(player.active) else None
                    energy_count = len(poke.energies) if poke else 0
                elif opt.inPlayArea == AreaType.BENCH and opt.inPlayIndex is not None:
                    if opt.inPlayIndex < len(player.bench):
                        energy_count = len(player.bench[opt.inPlayIndex].energies)
            # Phase 5: シンプルなスコア（KO リスク・上限チェックなし）
            score = priority_score * 10 - energy_count
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def _v5_active_is_buffer(state, plan) -> bool:
    """Phase 5 のバッファ判定: 逃げエネ 0 のみ（buffer_pokemon_ids なし）。"""
    card_db = get_card_db()
    my = state.players[state.yourIndex]
    active = my.active[0] if my.active else None
    if not active:
        return False
    if active.id in plan.main_attacker_ids or active.id in plan.sub_attacker_ids:
        return False
    active_card = card_db.get(active.id)
    if not active_card or active_card.retreatCost != 0:
        return False
    for bench_poke in my.bench:
        if bench_poke.id in plan.main_attacker_ids or bench_poke.id in plan.sub_attacker_ids:
            if len(bench_poke.energies) >= 1:
                return True
    return False


def _v5_non_ex_bench_ready(state, plan) -> bool:
    if not plan.non_ex_active_priority:
        return False
    non_ex_ids = set(plan.non_ex_active_priority)
    bench = state.players[state.yourIndex].bench
    return any(poke and poke.id in non_ex_ids and len(poke.energies) >= 1 for poke in bench)


def _v5_pick_best_play(obs: Observation) -> int:
    """Phase 5 のカードプレイ選択: bench_priority 外 Pokemon のスコアが 5。"""
    plan = get_deck_plan()
    card_db = get_card_db()
    state = obs.current
    opp_model = get_opponent_model()

    play_options = [
        i for i, opt in enumerate(obs.select.option)
        if opt.type == OptionType.PLAY
    ]
    bench_count = len(state.players[state.yourIndex].bench) if state else 0
    override = get_matchup_override(opp_model.estimate_arch())
    bench_full = bench_count >= override.max_bench_size

    boss_targets = opp_model.boss_target_ids() if override.boss_rush else []
    from src.decision.main_policy import _boss_target_on_bench, BOSS_ORDERS_ID
    boss_rush_active = override.boss_rush and _boss_target_on_bench(obs, boss_targets)
    hand_quality = opp_model.estimate_hand_quality(obs)

    SCORE = {
        CardType.SUPPORTER: 25,
        CardType.ITEM: 20,
        CardType.TOOL: 15,
        CardType.STADIUM: 10,
        CardType.POKEMON: 5,
    }
    BENCH_PRIORITY_POKEMON_SCORE = 30
    BOSS_RUSH_SCORE = 40

    best_i = play_options[0]
    best_score = -1

    for i in play_options:
        opt = obs.select.option[i]
        score = 5
        if opt.index is not None and state:
            player = state.players[state.yourIndex]
            if player.hand and opt.index < len(player.hand):
                card_id = player.hand[opt.index].id
                card_data = card_db.get(card_id)
                if card_data:
                    if card_data.cardType == CardType.POKEMON:
                        if bench_full:
                            score = 0
                        elif card_id in plan.bench_priority:
                            score = BENCH_PRIORITY_POKEMON_SCORE
                        else:
                            score = 5  # Phase 5: 5 のまま（Phase 6 では 0）
                    elif (card_data.cardType == CardType.SUPPORTER
                          and boss_rush_active
                          and card_id == BOSS_ORDERS_ID):
                        score = BOSS_RUSH_SCORE
                    elif card_data.cardType == CardType.SUPPORTER and card_id == IONO_ID:
                        if hand_quality < 0:
                            score = 3
                        elif hand_quality >= 2:
                            score = 35
                        else:
                            score = 18
                    else:
                        score = SCORE.get(card_data.cardType, 5)
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def choose_main_action_v5(obs: Observation) -> list[int]:
    """Phase 5 の MAIN フェーズ行動選択。"""
    state = obs.current
    if state is None:
        return [0]

    plan = get_deck_plan()
    turn = state.turn

    classified: dict[str, list[int]] = {
        "play": [], "attach": [], "evolve": [], "ability": [],
        "discard": [], "retreat": [], "attack": [], "end": [],
    }
    for i, opt in enumerate(obs.select.option):
        if opt.type == OptionType.PLAY:       classified["play"].append(i)
        elif opt.type == OptionType.ATTACH:   classified["attach"].append(i)
        elif opt.type == OptionType.EVOLVE:   classified["evolve"].append(i)
        elif opt.type == OptionType.ABILITY:  classified["ability"].append(i)
        elif opt.type == OptionType.DISCARD:  classified["discard"].append(i)
        elif opt.type == OptionType.RETREAT:  classified["retreat"].append(i)
        elif opt.type == OptionType.ATTACK:   classified["attack"].append(i)
        elif opt.type == OptionType.END:      classified["end"].append(i)

    from src.decision.main_policy import (
        _pick_best_evolve, _pick_best_attack,
    )

    evaluator = get_evaluator()
    prefer_retreat = (
        classified["retreat"]
        and classified["attack"]
        and evaluator.should_retreat(state, plan)
    )
    if prefer_retreat:
        return [classified["retreat"][0]]

    if classified["evolve"]:
        return [_pick_best_evolve(obs, classified["evolve"])]

    if not state.energyAttached and classified["attach"]:
        return [_v5_pick_best_attach(obs)]

    if classified["play"]:
        return [_v5_pick_best_play(obs)]

    if classified["ability"]:
        return [classified["ability"][0]]

    # Phase 5 バッファ判定（逃げエネ 0 のみ）
    if classified["retreat"] and _v5_active_is_buffer(state, plan):
        return [classified["retreat"][0]]

    if (classified["retreat"]
            and opponent_active_blocks_ex(state)
            and my_active_is_ex(state)
            and _v5_non_ex_bench_ready(state, plan)):
        return [classified["retreat"][0]]

    if classified["attack"]:
        return [_pick_best_attack(obs, classified["attack"])]

    if classified["retreat"]:
        return [classified["retreat"][0]]

    if classified["end"]:
        return [classified["end"][0]]

    return [0]


def agent_v5(obs_dict: dict) -> list[int]:
    """Phase 5 エージェント。"""
    from main import read_deck_csv
    from src.decision.router import choose_action
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return _v5_read_deck()
    get_opponent_model().update(obs)

    from cg.api import SelectContext
    ctx = obs.select.context
    # MAIN フェーズのみ v5 ロジック、それ以外は共通ルーター
    if ctx == SelectContext.MAIN:
        chosen = choose_main_action_v5(obs)
    else:
        chosen = choose_action(obs)
    return chosen


_v5_deck_cache: list[int] | None = None

def _v5_read_deck() -> list[int]:
    global _v5_deck_cache
    if _v5_deck_cache is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "deck_maries_obstagoon.csv")
        with open(path) as f:
            _v5_deck_cache = [int(l) for l in f.read().strip().split("\n")[:60]]
    return _v5_deck_cache


# ---------------------------------------------------------------------------
# 自己対戦ベンチマーク
# ---------------------------------------------------------------------------

def play_one_selfplay(
    our_deck: list[int],
    v6_is_player0: bool,
) -> int:
    """v6 vs v5 の 1 試合を実行。v6 が勝ったら 1、負けたら 0 を返す。"""
    if v6_is_player0:
        p0, p1 = agent_v6, agent_v5
        d0, d1 = our_deck, our_deck
    else:
        p0, p1 = agent_v5, agent_v6
        d0, d1 = our_deck, our_deck

    obs_dict, start_data = battle_start(d0, d1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed errorType={start_data.errorType}")

    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)
            if obs.current is not None and obs.current.result != -1:
                raw_result = obs.current.result  # 0=player0 win, 1=player1 win
                if v6_is_player0:
                    return 1 if raw_result == 0 else 0  # v6=player0 が勝ったら 1
                else:
                    return 1 if raw_result == 1 else 0  # v6=player1 が勝ったら 1
            acting_player = obs.current.yourIndex if obs.current is not None else 0
            acting_agent = p0 if acting_player == 0 else p1
            action = acting_agent(obs_dict)
            obs_dict = battle_select(action)
    finally:
        battle_finish()


def _expand_recipe(recipe: list[tuple[int, int]]) -> list[int]:
    return [cid for cid, count in recipe for _ in range(count)]


def run_selfplay(
    deck_names: list[str],
    games_per_deck: int,
    our_deck: list[int],
) -> dict[str, dict]:
    """v6 vs v5 の自己対戦ベンチマーク。対戦相手デッキは使わず、自分デッキで対戦。"""
    # 実際には deck_names は参照しない（自己対戦のため）が、
    # 「どのアーキタイプ相手」の環境でテストするかを分けてもよい。
    # シンプルに全試合まとめて計測する。
    print(f"\n{'='*60}")
    print("SELFPLAY: Phase 6 (v6) vs Phase 5 (v5)")
    print(f"Games: {games_per_deck}")
    print(f"{'='*60}")

    wins = 0
    losses = 0
    errors = 0
    total = games_per_deck

    for i in range(1, total + 1):
        v6_first = (i % 2 == 1)  # 先攻後攻を交互に
        try:
            result = play_one_selfplay(our_deck, v6_is_player0=v6_first)
            if result == 1:
                wins += 1
            else:
                losses += 1
            label = "先攻" if v6_first else "後攻"
            print(f"  game {i:>3}/{total} [{label}]: v6={'WIN' if result==1 else 'LOSE'}  "
                  f"running_wr={wins/i:.0%}")
        except Exception as e:
            print(f"  game {i} ERROR: {e}")
            errors += 1

    played = wins + losses
    wr = wins / played if played else 0.0
    print(f"\n  => Phase 6 vs Phase 5: {wins}W/{losses}L  win_rate={wr:.0%}  (errors={errors})")
    print(f"{'='*60}")
    return {"v6_wins": wins, "v5_wins": losses, "errors": errors, "v6_win_rate": wr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play Phase 5 vs Phase 6 benchmark")
    parser.add_argument("--games", type=int, default=30, help="Total games to play")
    args = parser.parse_args()

    our_deck = _v5_read_deck()
    run_selfplay([], args.games, our_deck)


if __name__ == "__main__":
    main()
