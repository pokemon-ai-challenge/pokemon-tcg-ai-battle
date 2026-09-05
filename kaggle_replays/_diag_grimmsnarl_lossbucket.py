#!/usr/bin/env python3
"""読み取り専用の診断分析（コード変更なし・一時スクリプト）。

performance-analyst タスク: grimmsnarl_rule_01（マリィのオーロンゲex ルールベース）が
同デッキの模倣ポリシー(ml_policy, marnie weights)にミラーで負け越す（43.5%）原因を、
grimmsnarl 側の全 decision を計装した実対戦データで診断する。

計測方法:
  - league/run_match.py の play_match は使わず、cg.game を直接叩く同等のループを
    このスクリプト内に複製する（play_match は勝敗が決した瞬間の obs をエージェントに
    渡さずに即 return するため、LogType.RESULT の reason コード(1=サイド完投/2=山札切れ/
    3=ベンチ壊滅/4=カード効果)を取得できない。診断に生の理由コードが要るための複製）。
  - grimmsnarl 側の MAIN 決定ごとに、core.plan(plan_attack の出力)・盤面・選んだ
    option type を記録する。
  - 見逃しリーサル仮説は、(a) plan.prizes と実際にそのターン奪取したサイド枚数の差分、
    (b) 独立に書いた総当たりオラクル(_enumerate_finish を一切再利用しない別実装)で
    plan.prizes を追試し、オラクルが上回るケースを数える、の2通りで検証する。

このスクリプトは opponents/ptcg_ai/league を一切変更しない。分析専用。
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
LEAGUE_DIR = REPO_ROOT / "league"
for _p in (str(REPO_ROOT), str(SAMPLE_SUB), str(LEAGUE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(SAMPLE_SUB)

from cg.api import (
    AreaType, EnergyType, LogType, OptionType, SelectContext, to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start
import run_league
import opponents.grimmsnarl_core as core

DECK_PATH = REPO_ROOT / "kaggle_replays/meta_analysis/archetype_decks/marnie_grimmsnarl_ex/01.csv"
WEIGHTS = "sample_submission/ptcg_ai/learning/policy_weights_marnie_grimmsnarl_ex.json"
CONFIG_BASE = "ml_lethal_attackplan_v0only"
N_GAMES = 140
MAX_STEPS = 3000
CLOSE_MARGIN = 2

OUT_JSON = Path(__file__).resolve().parent / "_diag_grimmsnarl_lossbucket_results.json"


def _prize_count(cid):
    data = core.card_table.get(cid)
    if data is None:
        return 1
    return 3 if data.megaEx else 2 if data.ex else 1


def _weak_mult(cid):
    data = core.card_table.get(cid)
    if data is not None and data.weakness == EnergyType.DARKNESS:
        return 2
    return 1


def _is_tera(cid):
    data = core.card_table.get(cid)
    return bool(data is not None and data.tera)


def _brute_bins(rem, killed, bins_left, counters_left, ids):
    base = sum(_prize_count(ids[i]) for i in killed)
    if bins_left <= 0 or counters_left <= 0:
        return base
    candidates = [i for i in range(len(rem)) if i not in killed and rem[i] > 0]
    if not candidates:
        return base
    best = base
    for tgt in candidates:
        need = -(-rem[tgt] // 10)
        max_count = min(3, counters_left, need)
        for c in range(1, max_count + 1):
            rem2 = list(rem)
            rem2[tgt] -= c * 10
            killed2 = set(killed)
            if rem2[tgt] <= 0:
                killed2.add(tgt)
            val = _brute_bins(rem2, killed2, bins_left - 1, counters_left - c, ids)
            if val > best:
                best = val
    return best


def oracle_max_prizes(can_main_attack, targets, masila_bins, own_counters):
    ids = [t["id"] for t in targets]
    best = 0
    best_detail = {}
    primary_options = [None]
    if can_main_attack and targets:
        primary_options.append(0)
    for primary in primary_options:
        rem = [t["hp"] for t in targets]
        killed = set()
        if primary is not None:
            dmg = 180 * _weak_mult(ids[primary])
            rem[primary] -= dmg
            if rem[primary] <= 0:
                killed.add(primary)
        bench30_choices = [None]
        if primary is not None:
            bench30_choices += [
                i for i in range(len(targets)) if i != 0 and i not in killed and not _is_tera(ids[i])
            ]
        for b30 in bench30_choices:
            rem2 = list(rem)
            killed2 = set(killed)
            if b30 is not None:
                rem2[b30] -= 30
                if rem2[b30] <= 0:
                    killed2.add(b30)
            val = _brute_bins(rem2, killed2, masila_bins, own_counters, ids)
            if val > best:
                best = val
                best_detail = {"primary": primary, "bench30": b30}
    return best, best_detail


def build_main_record(obs, g_idx, o_idx, action):
    cur = obs.current
    my_state = cur.players[g_idx]
    op_state = cur.players[o_idx]

    field_counts = defaultdict(int)
    own_field = []
    for c in my_state.active:
        if c is not None:
            field_counts[c.id] += 1
            own_field.append(c)
    for c in my_state.bench:
        field_counts[c.id] += 1
        own_field.append(c)

    main_line_count = (
        field_counts[core.IMPIDIMP] + field_counts[core.MORGREM] + field_counts[core.GRIMMSNARL_EX]
    )
    mature = main_line_count >= 2
    no_draw = my_state.deckCount <= 8

    op_active = op_state.active[0] if op_state.active and op_state.active[0] is not None else None
    op_targets = []
    if op_active is not None:
        op_targets.append({"role": "active", "id": op_active.id, "hp": op_active.hp, "maxHp": op_active.maxHp, "serial": op_active.serial})
    for b in op_state.bench:
        op_targets.append({"role": "bench", "id": b.id, "hp": b.hp, "maxHp": b.maxHp, "serial": b.serial})

    own_field_snapshot = [
        {"id": p.id, "hp": p.hp, "maxHp": p.maxHp, "energies": len(p.energies), "serial": p.serial}
        for p in own_field
    ]

    plan = core.plan
    chosen_types = []
    chosen_attack_id = None
    opts = obs.select.option
    for idx in action:
        if 0 <= idx < len(opts):
            o = opts[idx]
            tname = o.type.name if hasattr(o.type, "name") else str(o.type)
            chosen_types.append(tname)
            if tname == "ATTACK":
                chosen_attack_id = o.attackId

    return {
        "turn": cur.turn,
        "deckCount": my_state.deckCount,
        "no_draw": no_draw,
        "hand": len(my_state.hand) if my_state.hand is not None else None,
        "my_prize": len(my_state.prize),
        "opp_prize": len(op_state.prize),
        "main_line_count": main_line_count,
        "grimmsnarl_ex_on_field": field_counts[core.GRIMMSNARL_EX],
        "mature": mature,
        "can_main_attack": bool(core.can_main_attack),
        "bench_attacker": bool(core.bench_attacker),
        "use_support": core.use_support,
        "plan_prizes": plan.prizes,
        "plan_primary": plan.primary,
        "plan_needs_boss": plan.needs_boss,
        "plan_own_counters": plan.own_counters,
        "plan_n_adrena_moves": len(plan.adrena_moves),
        "plan_bench30_target": plan.bench30_target_serial,
        "op_targets": op_targets,
        "own_field": own_field_snapshot,
        "chosen_types": chosen_types,
        "chosen_attack_id": chosen_attack_id,
    }


def play_instrumented(agent_g, agent_o, deck_g, deck_o, g_is_player0, seed, game_idx):
    random.seed(seed)
    g_idx = 0 if g_is_player0 else 1
    o_idx = 1 - g_idx
    agents = {g_idx: agent_g, o_idx: agent_o}
    decks = (deck_g, deck_o) if g_is_player0 else (deck_o, deck_g)

    game_log = {
        "game_idx": game_idx, "g_idx": g_idx, "seed": seed,
        "main_records": [], "ko_events": [], "deck_trace": [], "context_tally": {},
        "winner": None, "reason": None, "final_turn": None,
        "final_my_prize": None, "final_opp_prize": None, "error": None,
    }

    obs_dict, start_data = battle_start(decks[0], decks[1])
    if start_data.errorType != 0:
        game_log["error"] = "battle_start errorType=" + str(start_data.errorType)
        return game_log

    seen_turns = set()
    turn_context_tally = defaultdict(Counter)
    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                game_log["error"] = "current is None mid-match"
                break
            if cur.result != -1:
                game_log["winner"] = cur.result
                game_log["final_turn"] = cur.turn
                for log in (obs.logs or []):
                    if log.type == LogType.RESULT:
                        game_log["reason"] = log.reason
                game_log["final_my_prize"] = len(cur.players[g_idx].prize)
                game_log["final_opp_prize"] = len(cur.players[o_idx].prize)
                break
            if steps >= MAX_STEPS:
                game_log["error"] = "max_steps_exceeded"
                break

            for log in (obs.logs or []):
                if log.type == LogType.MOVE_CARD and log.toArea == AreaType.DISCARD and log.fromArea in (AreaType.ACTIVE, AreaType.BENCH):
                    if log.playerIndex == o_idx:
                        game_log["ko_events"].append({"turn": cur.turn, "who": "G"})
                    elif log.playerIndex == g_idx:
                        game_log["ko_events"].append({"turn": cur.turn, "who": "O"})

            turn_player = cur.yourIndex
            action = agents[turn_player](obs)

            if turn_player == g_idx and obs.select is not None:
                ctx = obs.select.context
                cname = ctx.name if hasattr(ctx, "name") else str(ctx)
                turn_context_tally[cur.turn][cname] += 1
                if ctx == SelectContext.MAIN:
                    rec = build_main_record(obs, g_idx, o_idx, action)
                    game_log["main_records"].append(rec)
                    if cur.turn not in seen_turns:
                        game_log["deck_trace"].append({"turn": cur.turn, "deckCount": cur.players[g_idx].deckCount})
                        seen_turns.add(cur.turn)

            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:
        game_log["error"] = repr(exc)
    finally:
        battle_finish()

    game_log["context_tally"] = {str(t): dict(c) for t, c in turn_context_tally.items()}
    return game_log


def main():
    t0 = time.time()
    deck_g = run_league.read_deck_csv_file(str(DECK_PATH))
    agent_g = run_league.build_agent("grimmsnarl_rule_01", None, CONFIG_BASE)
    agent_o = run_league.build_agent("ml_policy", WEIGHTS, CONFIG_BASE)

    games = []
    for i in range(N_GAMES):
        g_is_p0 = (i % 2 == 0)
        seed = 5000 + i
        gl = play_instrumented(agent_g, agent_o, deck_g, deck_g, g_is_p0, seed, i)
        games.append(gl)
        if (i + 1) % 20 == 0:
            print("  ...", i + 1, "/", N_GAMES, "games done (", round(time.time() - t0, 1), "s elapsed)", flush=True)

    elapsed = time.time() - t0
    print()
    print(N_GAMES, "games in", round(elapsed, 1), "s")

    errors = [g for g in games if g["error"] is not None]
    valid = [g for g in games if g["error"] is None]
    print("errors:", len(errors))
    for g in errors:
        print("  game", g["game_idx"], ":", g["error"])

    wins = sum(1 for g in valid if g["winner"] == g["g_idx"])
    if valid:
        print()
        print("overall (this run): grimmsnarl_rule_01", wins, "/", len(valid), "=", round(wins / len(valid), 3))

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"n_games": N_GAMES, "elapsed_sec": elapsed, "games": games}, f, ensure_ascii=False)
    print("raw data written to", OUT_JSON)


if __name__ == "__main__":
    main()
