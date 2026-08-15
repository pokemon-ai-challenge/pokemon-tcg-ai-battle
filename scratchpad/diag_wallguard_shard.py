"""Instrumented re-measurement harness for the wall_guard fix (Guard A reachability).

Rebuilt from scratch in this session: the original `scratchpad/diag_crustle_shard.py` from a
prior (crashed) session no longer exists on disk (scratchpad/ is not committed to git and this
is a fresh session). This script is a from-scratch reconstruction that follows the same
documented pattern (`kaggle_replays/docs/requirements-kamitsuorochi-2026-08-12.md` SS5-4):
replicates `league/run_league.py`'s `build_agent` / deck loading / turn-order alternation
(imported, not copy-pasted, so any future change to those helpers is picked up automatically),
and drives the match with a custom loop (like `league/run_match.py::play_match`, but this
version additionally inspects `obs.current`/`obs.logs` on *every* engine step, including the
terminal one that `play_match` swallows internally, so it can recover the RESULT log's `reason`
field and catch a same-move Mega Kangaskhan ex KO that happens on our own winning move).

Does NOT modify `league/run_match.py` or `league/run_league.py` (frozen) -- only imports them.

Per-attack "0 damage" is computed the same way `wall_guard.py` computes it: via
`board_evaluation.attack_features.resolve_damage` on the option actually chosen, not by
parsing HP_CHANGE logs. This keeps the definition of "0 damage" identical to what the guard
itself is reacting to.

Usage:
    python diag_wallguard_shard.py --deck-opp <path> --config-base abl_5_full_wallguard \
        --games 100 --workers 5 --seed-start 0 --out runs/crustle_wallguard_fixed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRATCHPAD_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRATCHPAD_DIR.parent
_LEAGUE_DIR = _ROOT_DIR / "league"
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"

for _candidate in (str(_LEAGUE_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from cg.api import Observation, LogType, OptionType, SelectType, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from run_league import build_agent, read_deck_csv_file  # noqa: E402 (frozen file, import-only)

from ptcg_ai.board_evaluation import attack_features  # noqa: E402
from ptcg_ai.search import wall_guard  # noqa: E402
from ptcg_ai.shared import card_cache  # noqa: E402

MAX_STEPS = 3000

# Cards this diagnostic hardcodes on purpose (measurement-only script, not production code):
# Crustle ("Mysterious Rock Inn") and Mega Kangaskhan ex, per
# kaggle_replays/docs/requirements-kamitsuorochi-2026-08-12.md SS5-4.
CRUSTLE_CARD_ID = 345
MEGA_KANGASKHAN_CARD_ID = 756

LOSS_REASON_LABELS = {
    1: "prizes_taken",       # side taken to 0
    2: "deckout",            # started a turn with 0 cards in deck
    3: "no_active_pokemon",  # no Pokemon left in the Active Spot
    4: "card_effect",        # a card effect ended the match
}


def _side(active, bench):
    return [active, *[p for p in (bench or []) if p is not None]]


def _resolved_damage_for_chosen_attack(state, option, attacker, opp_active):
    """Same computation `wall_guard._active_attack_damage` uses, applied to the single
    attack option the agent actually chose (not the max across all options)."""
    attack = card_cache.get_attack(option.attackId)
    opp_card = card_cache.get_card(opp_active.id)
    your_index = state.yourIndex
    me = state.players[your_index]
    opp = state.players[1 - your_index]
    own_side = _side(attacker, me.bench)
    opp_side = _side(opp_active, opp.bench)
    return attack_features.resolve_damage(
        attack, attacker, opp_card.weakness, opp_card.resistance,
        defender=opp_active, defender_side_pokemon=opp_side,
        defender_is_benched=False,
        damage_is_effect=attack_features.damage_is_effect_based(attack),
        attacker_side_pokemon=own_side, defender_active_pokemon=opp_active,
    )


def _alive_kangaskhan_serials(player_state):
    out = set()
    for mon in [player_state.active[0] if player_state.active else None, *(player_state.bench or [])]:
        if mon is not None and mon.id == MEGA_KANGASKHAN_CARD_ID:
            out.add(mon.serial)
    return out


def play_instrumented_match(agent_ours, agent_opp, deck_ours, deck_opp, ours_is_player0, seed):
    """One match. Returns a dict of raw per-game measurements (or {"error": ...})."""
    import random
    if seed is not None:
        random.seed(seed)

    if ours_is_player0:
        agent0, agent1 = agent_ours, agent_opp
        deck0, deck1 = deck_ours, deck_opp
        our_index = 0
    else:
        agent0, agent1 = agent_opp, agent_ours
        deck0, deck1 = deck_opp, deck_ours
        our_index = 1
    agents = {0: agent0, 1: agent1}

    stats = {
        "winner": None, "turns": None, "error": None,
        "our_attacks": 0, "our_zero_damage_attacks": 0, "ex_vs_wall_zero_damage": 0,
        "kangaskhan_ko_count": 0,
        # None until we've observed the real starting count (6). Pre-setup snapshots report
        # an empty prize list (prizes haven't been dealt yet) -- must not be mistaken for
        # "already at 0 prizes remaining", or every game would wrongly read prizes_taken=6.
        "min_prize_remaining": None,
        "loss_reason": None,   # only meaningful if we lost
        "win_reason": None,    # only meaningful if we won
    }

    seen_alive_kangaskhan: set[int] = set()
    counted_kangaskhan_ko: set[int] = set()
    t0 = time.time()
    steps = 0
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        stats["error"] = f"battle_start errorType={start_data.errorType}"
        return stats
    try:
        while True:
            obs: Observation = to_observation_class(obs_dict)

            # Scan every log we see (even on the terminal step, which play_match's own loop
            # never hands to an agent) for the match RESULT reason.
            for log in obs.logs:
                if log.type == LogType.RESULT:
                    stats["winner"] = log.result
                    if log.result == our_index:
                        stats["win_reason"] = log.reason
                    elif log.result in (0, 1):
                        stats["loss_reason"] = log.reason

            if obs.current is None:
                stats["error"] = "current is None mid-match"
                return stats

            # Track Mega Kangaskhan ex KOs and our remaining prize count from every snapshot
            # we see, regardless of whose turn it is (obs.current is a full, public board
            # snapshot -- only hands are hidden).
            opp_state = obs.current.players[1 - our_index]
            our_state = obs.current.players[our_index]
            n_prize = len(our_state.prize or [])
            if n_prize > 0 or stats["min_prize_remaining"] is not None:
                stats["min_prize_remaining"] = (
                    n_prize if stats["min_prize_remaining"] is None
                    else min(stats["min_prize_remaining"], n_prize)
                )
            now_alive = _alive_kangaskhan_serials(opp_state)
            vanished = (seen_alive_kangaskhan - now_alive) - counted_kangaskhan_ko
            if vanished:
                stats["kangaskhan_ko_count"] += len(vanished)
                counted_kangaskhan_ko |= vanished
            seen_alive_kangaskhan |= now_alive

            if obs.current.result != -1:
                stats["turns"] = obs.current.turn
                break
            if steps >= MAX_STEPS:
                stats["error"] = f"max_steps_exceeded({MAX_STEPS})"
                stats["turns"] = obs.current.turn
                break

            turn_player = obs.current.yourIndex
            action = agents[turn_player](obs)

            if turn_player == our_index and obs.select is not None and obs.select.type == SelectType.MAIN \
                    and len(action) == 1:
                option = obs.select.option[action[0]]
                if option.type == OptionType.ATTACK and option.attackId is not None:
                    me = obs.current.players[our_index]
                    opp = obs.current.players[1 - our_index]
                    attacker = me.active[0] if me.active else None
                    opp_active = opp.active[0] if opp.active else None
                    if attacker is not None and opp_active is not None:
                        damage = _resolved_damage_for_chosen_attack(
                            obs.current, option, attacker, opp_active,
                        )
                        stats["our_attacks"] += 1
                        if damage <= 0:
                            stats["our_zero_damage_attacks"] += 1
                            attacker_card = card_cache.get_card(attacker.id)
                            if opp_active.id == CRUSTLE_CARD_ID and (attacker_card.ex or attacker_card.megaEx):
                                stats["ex_vs_wall_zero_damage"] += 1
                        # Catch a KO that happens on this exact move (no further obs.current
                        # will show it disappearing -- the match may end right here).
                        if opp_active.id == MEGA_KANGASKHAN_CARD_ID and damage >= opp_active.hp \
                                and opp_active.serial not in counted_kangaskhan_ko:
                            stats["kangaskhan_ko_count"] += 1
                            counted_kangaskhan_ko.add(opp_active.serial)

            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:  # noqa: BLE001 - one bad game must not sink the whole shard
        stats["error"] = repr(exc)
    finally:
        battle_finish()

    stats["seconds"] = time.time() - t0
    stats["steps"] = steps
    stats["prizes_taken"] = 6 - stats["min_prize_remaining"] if stats["min_prize_remaining"] is not None else None
    return stats


_WORKER_STATE: dict = {}


def _worker_init(deck_opp_path, deck_ours_path, config_base, agent_opp_name):
    os.chdir(_SAMPLE_SUBMISSION_DIR)
    _WORKER_STATE["agent_ours"] = build_agent("ml_policy", None, config_base)
    _WORKER_STATE["agent_opp"] = build_agent(agent_opp_name, None, config_base)
    _WORKER_STATE["deck_ours"] = read_deck_csv_file(deck_ours_path)
    _WORKER_STATE["deck_opp"] = read_deck_csv_file(deck_opp_path)
    wall_guard.reset_stats()
    wall_guard.reset_pending_target()


def _worker_play_chunk(tasks: list[tuple[int, int]]) -> dict:
    """tasks: list of (index, seed). Plays all of them sequentially in this worker process,
    returns per-game records plus this worker's share of wall_guard's cumulative stats."""
    s = _WORKER_STATE
    records = []
    for index, seed in tasks:
        ours_is_player0 = (index % 2 == 0)
        rec = play_instrumented_match(
            s["agent_ours"], s["agent_opp"], s["deck_ours"], s["deck_opp"], ours_is_player0, seed,
        )
        rec["index"] = index
        rec["seed"] = seed
        records.append(rec)
    return {"records": records, "wall_guard_stats": wall_guard.get_stats()}


def run_shard(deck_opp_path, deck_ours_path, config_base, agent_opp_name, games, workers, seed_start):
    tasks = [(i, seed_start + i) for i in range(games)]
    chunks: list[list[tuple[int, int]]] = [[] for _ in range(workers)]
    for i, task in enumerate(tasks):
        chunks[i % workers].append(task)
    chunks = [c for c in chunks if c]

    all_records: list[dict] = []
    wg_totals = Counter()

    if workers <= 1:
        _worker_init(deck_opp_path, deck_ours_path, config_base, agent_opp_name)
        for chunk in chunks:
            result = _worker_play_chunk(chunk)
            all_records.extend(result["records"])
            wg_totals.update(result["wall_guard_stats"])
    else:
        initargs = (deck_opp_path, deck_ours_path, config_base, agent_opp_name)
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=initargs) as ex:
            futures = [ex.submit(_worker_play_chunk, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                result = fut.result()
                all_records.extend(result["records"])
                wg_totals.update(result["wall_guard_stats"])

    all_records.sort(key=lambda r: r["index"])
    return all_records, dict(wg_totals)


def summarize(records: list[dict]) -> dict:
    valid = [r for r in records if not r.get("error")]
    errors = [r for r in records if r.get("error")]
    # winner index bookkeeping: we stored winner as absolute player index (0/1), and
    # win_reason/loss_reason already tell us whether *we* won, so just count via those.
    n_win = sum(1 for r in valid if r.get("win_reason") is not None)
    n_loss = sum(1 for r in valid if r.get("loss_reason") is not None)
    n_draw = len(valid) - n_win - n_loss

    total_attacks = sum(r.get("our_attacks", 0) for r in valid)
    zero_attacks = sum(r.get("our_zero_damage_attacks", 0) for r in valid)
    ex_wall_zero = sum(r.get("ex_vs_wall_zero_damage", 0) for r in valid)
    prizes = [r["prizes_taken"] for r in valid if r.get("prizes_taken") is not None]
    kanga_ko = [r.get("kangaskhan_ko_count", 0) for r in valid]
    kanga_dist = Counter(("0" if k == 0 else "1" if k == 1 else "2+") for k in kanga_ko)
    loss_reasons = Counter(LOSS_REASON_LABELS.get(r["loss_reason"], f"unknown({r['loss_reason']})")
                            for r in valid if r.get("loss_reason") is not None)
    turns = [r["turns"] for r in valid if r.get("turns") is not None]

    def wilson(successes, n, z=1.959963984540054):
        if n == 0:
            return (0.0, 1.0)
        phat = successes / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = phat + z2 / (2 * n)
        margin = z * ((phat * (1 - phat) + z2 / (4 * n)) / n) ** 0.5
        return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))

    lo, hi = wilson(n_win, len(valid))
    return {
        "games_run": len(records),
        "valid_games": len(valid),
        "errors": len(errors),
        "error_details": [r["error"] for r in errors],
        "wins": n_win, "losses": n_loss, "draws": n_draw,
        "win_rate": n_win / len(valid) if valid else None,
        "wilson_95ci": [lo, hi],
        "avg_turns": sum(turns) / len(turns) if turns else None,
        "our_attacks_total": total_attacks,
        "zero_damage_attacks": zero_attacks,
        "zero_damage_rate": zero_attacks / total_attacks if total_attacks else None,
        "ex_vs_wall_zero_damage": ex_wall_zero,
        "mean_prizes_taken": sum(prizes) / len(prizes) if prizes else None,
        "kangaskhan_ko_distribution": dict(kanga_dist),
        "loss_reason_breakdown": dict(loss_reasons),
    }


def _won(r):
    return r.get("win_reason") is not None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-opp", required=True)
    parser.add_argument("--deck-ours", default=None)
    parser.add_argument("--config-base", default="abl_5_full_wallguard")
    parser.add_argument("--agent-opp", default="rule_based")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    t0 = time.time()
    records, wg_stats = run_shard(
        args.deck_opp, args.deck_ours, args.config_base, args.agent_opp,
        args.games, args.workers, args.seed_start,
    )
    summary = summarize(records)
    summary["wall_guard_stats"] = wg_stats
    summary["elapsed_seconds"] = time.time() - t0
    summary["config_base"] = args.config_base
    summary["deck_opp"] = args.deck_opp
    summary["deck_ours"] = args.deck_ours
    summary["agent_opp"] = args.agent_opp
    summary["games"] = records

    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _SCRATCHPAD_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nwritten to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
