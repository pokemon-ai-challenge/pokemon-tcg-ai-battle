"""Board-reach / win-rate measurement harness for the kamitsuorochi_ex multi-select BC change
(requirements-kamitsuorochi-2026-08-12.md "Then measure").

Reconstructed in this session: the original `kaggle_replays/diagnostics/diag_crustle_shard.py` no longer exists
on disk (kaggle_replays/diagnostics/ is not committed to git). This version is adapted from
`kaggle_replays/diagnostics/diag_wallguard_shard.py` (same pattern: replicates `league/run_league.py`'s
`build_agent` / deck loading / turn-order alternation by importing them, not copy-pasting; drives
the match with a custom loop that also inspects the terminal `obs.current`/`obs.logs`, which
`run_match.play_match` swallows internally, to recover the RESULT log's `reason` field).

Does NOT modify `league/run_match.py` or `league/run_league.py` (frozen) -- only imports them.

New in this version (vs diag_wallguard_shard.py): tracks `board_dev` -- a dict of
card_id -> first turn (obs.current.turn) that card_id was seen anywhere on OUR side
(active or bench) -- for the four kamitsuorochi_ex evolution-line cards named in the
requirements doc (Hydrapple ex=150, Meganium=710, Dipplin=93, Teal Mask Ogerpon ex=96).
This dict always has all four keys; value is None if never reached. This is the primary
metric requested (board-reach rate is far lower variance than win rate against crustle,
since the native engine's shuffle is not controlled by random.seed() -- see run_match.py
docstring and requirements doc SS "Important caveat").

Usage (5 shards x 20 games, matching the documented pattern for this harness):
    python diag_crustle_shard.py --deck-ours <path/to/kamitsuorochi 06.csv> \
        --deck-opp <path/to/crustle 06.csv> \
        --weights-ours <candidate policy_weights.json> \
        --weights-opp <path/to/policy_weights_crustle_v715b_s42.json> \
        --config-base abl_5_full --games 20 --workers 1 --seed-start 0 \
        --out runs/shard0.json
Run 5 of these with --seed-start 0/20/40/60/80 (or similar) and merge the summaries, since the
cg engine's battle state is a process-global singleton and cannot be parallelized in a single
process (see requirements doc's "Then measure" section).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
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

# Cards this diagnostic hardcodes on purpose (measurement-only script, not production code),
# per requirements-kamitsuorochi-2026-08-12.md's board-reach table:
BOARD_DEV_CARD_IDS = {
    150: "hydrapple_ex",
    710: "meganium",
    93: "dipplin",
    96: "ogerpon_ex",
}
CRUSTLE_CARD_ID = 345  # Mysterious Rock Inn (wall)
MEGA_KANGASKHAN_CARD_ID = 756  # Mega ex, 3-prize KO

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


def _our_board_card_ids(our_state) -> set[int]:
    ids: set[int] = set()
    for mon in [our_state.active[0] if our_state.active else None, *(our_state.bench or [])]:
        if mon is not None:
            ids.add(mon.id)
    return ids


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
        "board_dev": {str(cid): None for cid in BOARD_DEV_CARD_IDS},
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

            # Board-reach tracking: first turn each watched card_id appears on OUR board
            # (active or bench), regardless of whose turn it currently is (obs.current is a
            # full public board snapshot -- only hands are hidden).
            our_board_ids = _our_board_card_ids(our_state)
            for cid in BOARD_DEV_CARD_IDS:
                key = str(cid)
                if stats["board_dev"][key] is None and cid in our_board_ids:
                    stats["board_dev"][key] = obs.current.turn

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


def run_shard(deck_opp_path, deck_ours_path, weights_ours, weights_opp, a_config_base, b_config_base,
              games, seed_start):
    os.chdir(_SAMPLE_SUBMISSION_DIR)
    agent_ours = build_agent("ml_policy", weights_ours, a_config_base)
    agent_opp = build_agent("ml_policy", weights_opp, b_config_base)
    deck_ours = read_deck_csv_file(deck_ours_path)
    deck_opp = read_deck_csv_file(deck_opp_path)

    wall_guard.reset_stats()
    wall_guard.reset_pending_target()

    records = []
    for i in range(games):
        seed = seed_start + i
        ours_is_player0 = (i % 2 == 0)
        rec = play_instrumented_match(agent_ours, agent_opp, deck_ours, deck_opp, ours_is_player0, seed)
        rec["index"] = i
        rec["seed"] = seed
        records.append(rec)
    wg_stats = wall_guard.get_stats()
    return records, wg_stats


def summarize(records: list[dict]) -> dict:
    valid = [r for r in records if not r.get("error")]
    errors = [r for r in records if r.get("error")]
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

    def median(xs):
        if not xs:
            return None
        s = sorted(xs)
        m = len(s)
        return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2

    board_dev_summary = {}
    for cid in BOARD_DEV_CARD_IDS:
        key = str(cid)
        vals = [r["board_dev"][key] for r in valid if r.get("board_dev", {}).get(key) is not None]
        board_dev_summary[key] = {
            "name": BOARD_DEV_CARD_IDS[cid],
            "n_reached": len(vals),
            "n_games": len(valid),
            "reach_rate": len(vals) / len(valid) if valid else None,
            "median_turn": median(vals),
            "turns": vals,
        }

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
        "board_dev": board_dev_summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-opp", required=True)
    parser.add_argument("--deck-ours", required=True)
    parser.add_argument("--weights-ours", "--a-weights", dest="weights_ours", default=None,
                         help="ml_policy weights for our side (kamitsuorochi_ex candidate)")
    parser.add_argument("--weights-opp", "--b-weights", dest="weights_opp", default=None,
                         help="ml_policy weights for the opponent (crustle-trained BC)")
    parser.add_argument("--config-base", default="abl_5_full",
                         help="fallback config_base for both sides if --a-config-base/--b-config-base not given")
    parser.add_argument("--a-config-base", dest="a_config_base", default=None,
                         help="config_base for our (agent A) side; defaults to --config-base")
    parser.add_argument("--b-config-base", dest="b_config_base", default=None,
                         help="config_base for the opponent (agent B) side; defaults to --config-base")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1, help="unused (kept for CLI-compat with the wallguard harness); this script always runs single-process per shard -- launch multiple OS processes for parallelism")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    a_config_base = args.a_config_base or args.config_base
    b_config_base = args.b_config_base or args.config_base

    t0 = time.time()
    records, wg_stats = run_shard(
        args.deck_opp, args.deck_ours, args.weights_ours, args.weights_opp,
        a_config_base, b_config_base, args.games, args.seed_start,
    )
    summary = summarize(records)
    summary["wall_guard_stats"] = wg_stats
    summary["elapsed_seconds"] = time.time() - t0
    summary["a_config_base"] = a_config_base
    summary["b_config_base"] = b_config_base
    summary["deck_opp"] = args.deck_opp
    summary["deck_ours"] = args.deck_ours
    summary["weights_ours"] = args.weights_ours
    summary["weights_opp"] = args.weights_opp
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
