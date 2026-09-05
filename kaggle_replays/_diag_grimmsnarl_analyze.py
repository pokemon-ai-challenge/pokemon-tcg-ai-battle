#!/usr/bin/env python3
"""_diag_grimmsnarl_lossbucket.py が書いた生データを分析する(読み取り専用・分析専用)。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SUB = REPO_ROOT / "sample_submission"
for p in (str(REPO_ROOT), str(SAMPLE_SUB)):
    if p not in sys.path:
        sys.path.insert(0, p)

IN_JSON = Path(__file__).resolve().parent / "_diag_grimmsnarl_lossbucket_results.json"

CLOSE_MARGIN = 2
REASON_NAME = {1: "prize_out", 2: "deckout", 3: "bench_wipe", 4: "card_effect", None: "unknown"}


def load():
    with IN_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["games"]


def main():
    games = load()
    valid = [g for g in games if g["error"] is None]
    losses = [g for g in valid if g["winner"] != g["g_idx"]]
    wins = [g for g in valid if g["winner"] == g["g_idx"]]
    print("valid games:", len(valid), " wins:", len(wins), " losses:", len(losses))
    print("win rate:", round(len(wins) / len(valid), 3))
    print()

    buckets = Counter()
    bucket_examples = defaultdict(list)
    for g in losses:
        reason = REASON_NAME.get(g["reason"], "code" + str(g["reason"]))
        my_final = g["final_my_prize"]
        deck_trace = g["deck_trace"]
        min_deck = min((d["deckCount"] for d in deck_trace), default=None)

        if reason == "deckout":
            bucket = "deck_out"
        elif reason == "bench_wipe":
            bucket = "bench_wipe"
        elif reason == "prize_out":
            bucket = "side_race_close" if (my_final is not None and my_final <= CLOSE_MARGIN) else "side_race_blowout"
        else:
            bucket = "other_" + str(reason)
        buckets[bucket] += 1
        bucket_examples[bucket].append({
            "game_idx": g["game_idx"], "seed": g["seed"], "final_turn": g["final_turn"],
            "my_final_prize": my_final, "opp_final_prize": g["final_opp_prize"],
            "min_deckCount": min_deck, "reason_code": g["reason"],
        })

    print("=== bucket 1: loss taxonomy (reason code x final prize margin) ===")
    for b, n in buckets.most_common():
        print("  ", b, ":", n, "kenn (", round(n / len(losses) * 100, 1), "%)")
    print()
    for b, exs in bucket_examples.items():
        print("  --", b, "examples (first 3) --")
        for ex in exs[:3]:
            print("     game=", ex["game_idx"], "seed=", ex["seed"], "final_turn=", ex["final_turn"],
                  "my_prize=", ex["my_final_prize"], "opp_prize=", ex["opp_final_prize"],
                  "min_deckCount=", ex["min_deckCount"], "reason=", ex["reason_code"])
    print()

    print("=== bucket 2: deck_out loss deckCount trace ===")
    for ex in bucket_examples.get("deck_out", []):
        g = next(x for x in losses if x["game_idx"] == ex["game_idx"])
        trace = g["deck_trace"]
        turns_to_zero = [d["turn"] for d in trace if d["deckCount"] == 0]
        tail = [d["deckCount"] for d in trace[-10:]]
        print("  game=", g["game_idx"], "final_turn=", g["final_turn"],
              "deckCount tail10=", tail, "turn_hit_zero=", turns_to_zero[:1])
    print()

    def first_mature_turn(g):
        for r in g["main_records"]:
            if r["mature"]:
                return r["turn"]
        return None

    def first_grimmsnarl_ex_turn(g):
        for r in g["main_records"]:
            if r["grimmsnarl_ex_on_field"] >= 1:
                return r["turn"]
        return None

    mature_win = [t for t in (first_mature_turn(g) for g in wins) if t is not None]
    mature_loss = [t for t in (first_mature_turn(g) for g in losses) if t is not None]
    gex_win = [t for t in (first_grimmsnarl_ex_turn(g) for g in wins) if t is not None]
    gex_loss = [t for t in (first_grimmsnarl_ex_turn(g) for g in losses) if t is not None]
    never_mature_loss = sum(1 for g in losses if first_mature_turn(g) is None)
    never_gex_loss = sum(1 for g in losses if first_grimmsnarl_ex_turn(g) is None)

    def avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("=== bucket 3: tempo (mature-gate turn) ===")
    print("  wins: mature n=", len(mature_win), "avg_turn=", round(avg(mature_win), 2),
          " / grimmsnarl_ex first n=", len(gex_win), "avg_turn=", round(avg(gex_win), 2))
    print("  losses: mature n=", len(mature_loss), "avg_turn=", round(avg(mature_loss), 2),
          " / grimmsnarl_ex first n=", len(gex_loss), "avg_turn=", round(avg(gex_loss), 2))
    print("  losses never reaching mature:", never_mature_loss,
          "(", round(never_mature_loss / len(losses) * 100, 1), "%)")
    print("  losses never showing grimmsnarl_ex on field:", never_gex_loss,
          "(", round(never_gex_loss / len(losses) * 100, 1), "%)")
    print()

    print("=== missed-lethal (a): plan.prizes at last MAIN of turn vs actual prizes gained ===")
    gap_events = []
    attack_available_no_fire = []
    total_turns_checked = 0
    for g in valid:
        recs = g["main_records"]
        by_turn = defaultdict(list)
        for r in recs:
            by_turn[r["turn"]].append(r)
        turns_sorted = sorted(by_turn)
        for ti, t in enumerate(turns_sorted):
            turn_recs = by_turn[t]
            last = turn_recs[-1]
            total_turns_checked += 1
            attacked = any("ATTACK" in r["chosen_types"] for r in turn_recs)
            if last["plan_prizes"] >= 1 and not attacked:
                attack_available_no_fire.append({
                    "game_idx": g["game_idx"], "turn": t, "plan_prizes": last["plan_prizes"],
                    "can_main_attack": last["can_main_attack"], "mature": last["mature"],
                })
            if attacked and last["plan_prizes"] >= 1:
                start_prize = turn_recs[0]["my_prize"]
                next_own_turn_idx = ti + 2
                if next_own_turn_idx < len(turns_sorted):
                    next_turn = turns_sorted[next_own_turn_idx]
                    end_prize = by_turn[next_turn][0]["my_prize"]
                else:
                    end_prize = g["final_my_prize"] if g["final_my_prize"] is not None else start_prize
                actual_delta = (start_prize - end_prize) if end_prize is not None else None
                if actual_delta is not None and actual_delta < last["plan_prizes"]:
                    gap_events.append({
                        "game_idx": g["game_idx"], "turn": t,
                        "plan_prizes": last["plan_prizes"], "actual_delta": actual_delta,
                        "won": g["winner"] == g["g_idx"],
                    })
    print("  turns checked:", total_turns_checked)
    print("  [candidate A] plan_prizes>=1 but ATTACK never chosen this turn:", len(attack_available_no_fire))
    for ex in attack_available_no_fire[:10]:
        print("     game=", ex["game_idx"], "turn=", ex["turn"], "plan_prizes=", ex["plan_prizes"],
              "can_main_attack=", ex["can_main_attack"], "mature=", ex["mature"])
    print("  [candidate B] attacked with plan_prizes>=1 but actual_delta < plan_prizes:", len(gap_events))
    for ex in gap_events[:15]:
        print("     game=", ex["game_idx"], "turn=", ex["turn"], "plan_prizes=", ex["plan_prizes"],
              "actual_delta=", ex["actual_delta"], "won=", ex["won"])
    print()

    from _diag_grimmsnarl_lossbucket import oracle_max_prizes

    print("=== missed-lethal (b): independent brute-force oracle vs plan.prizes (plan_needs_boss=False only) ===")
    oracle_checked = 0
    oracle_mismatches = []
    for g in valid:
        for r in g["main_records"]:
            if r["plan_needs_boss"]:
                continue
            if not r["op_targets"]:
                continue
            oracle_checked += 1
            n_masila = sum(1 for p in r["own_field"] if p["id"] == 112 and p["energies"] >= 1)
            oracle_val, detail = oracle_max_prizes(
                r["can_main_attack"], r["op_targets"], n_masila, r["plan_own_counters"]
            )
            if oracle_val > r["plan_prizes"]:
                oracle_mismatches.append({
                    "game_idx": g["game_idx"], "turn": r["turn"],
                    "plan_prizes": r["plan_prizes"], "oracle_prizes": oracle_val,
                    "n_masila": n_masila, "own_counters": r["plan_own_counters"],
                    "can_main_attack": r["can_main_attack"], "op_targets": r["op_targets"],
                    "detail": detail,
                })
    print("  MAIN decisions checked (plan_needs_boss=False):", oracle_checked)
    print("  oracle > plan.prizes count:", len(oracle_mismatches),
          "(", round(len(oracle_mismatches) / max(1, oracle_checked) * 100, 3), "%)")
    for ex in oracle_mismatches[:10]:
        print("     game=", ex["game_idx"], "turn=", ex["turn"], "plan_prizes=", ex["plan_prizes"],
              "oracle_prizes=", ex["oracle_prizes"], "n_masila=", ex["n_masila"],
              "own_counters=", ex["own_counters"], "can_main_attack=", ex["can_main_attack"])
        print("        op_targets=", ex["op_targets"])
    print()

    print("=== adrenaline (Munkidori ability) usage ===")
    turns_with_counters_but_unused = 0
    turns_with_counters_total = 0
    for g in valid:
        recs = g["main_records"]
        by_turn = defaultdict(list)
        for r in recs:
            by_turn[r["turn"]].append(r)
        for t, turn_recs in by_turn.items():
            last = turn_recs[-1]
            if last["plan_own_counters"] > 0:
                turns_with_counters_total += 1
                ctx_tally = g["context_tally"].get(str(t), {})
                n_adrena = ctx_tally.get("REMOVE_DAMAGE_COUNTER_COUNT", 0)
                if n_adrena == 0 and last["plan_n_adrena_moves"] == 0:
                    turns_with_counters_but_unused += 1
    print("  turns with own_counters>0:", turns_with_counters_total)
    print("  of those, plan built zero adrena_moves (=unused):", turns_with_counters_but_unused,
          "(", round(turns_with_counters_but_unused / max(1, turns_with_counters_total) * 100, 1), "%)")
    print()

    print("=== ATTACH_TO(pankup) occurrences while no_draw=True ===")
    no_draw_attach_events = 0
    for g in valid:
        for t, tally in g["context_tally"].items():
            n_attach = tally.get("ATTACH_TO", 0)
            if n_attach == 0:
                continue
            recs = [r for r in g["main_records"] if str(r["turn"]) == t]
            if recs and recs[0]["no_draw"]:
                no_draw_attach_events += n_attach
    print("  total ATTACH_TO events while no_draw=True (all games):", no_draw_attach_events)
    print()

    out = {
        "win_rate": len(wins) / len(valid),
        "n_valid": len(valid), "n_losses": len(losses),
        "loss_buckets": dict(buckets),
        "loss_bucket_examples": {k: v[:10] for k, v in bucket_examples.items()},
        "tempo": {
            "mature_win_avg_turn": avg(mature_win), "mature_loss_avg_turn": avg(mature_loss),
            "gex_win_avg_turn": avg(gex_win), "gex_loss_avg_turn": avg(gex_loss),
            "never_mature_loss": never_mature_loss, "never_gex_loss": never_gex_loss,
        },
        "missed_lethal_a_attack_available_no_fire": attack_available_no_fire,
        "missed_lethal_a_gap_events": gap_events,
        "missed_lethal_b_oracle_mismatches": oracle_mismatches,
        "oracle_checked": oracle_checked,
        "adrena": {
            "turns_with_counters_total": turns_with_counters_total,
            "turns_with_counters_but_unused": turns_with_counters_but_unused,
        },
        "no_draw_attach_events": no_draw_attach_events,
    }
    out_path = Path(__file__).resolve().parent / "_diag_grimmsnarl_analyze_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("summary written to", out_path)


if __name__ == "__main__":
    main()
