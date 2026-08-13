"""同一Observation上でのPlanner OFF/ON比較(shadow)を集計する。

og_attach_shadow(bonus計算はするが実行動には反映しない)で実プレイし、
OGERPON_SHADOW_LOG に記録された各MAIN意思決定を game_id/turn/state fingerprint で
重複排除した上で、以下を報告する:
  - raw_attach_exposures / unique_attach_states / unique_attach_turns
  - selection_flipped の内訳(flip_direction別)
  - planner_raw_bonus / planner_capped_bonus / pimc_margin の分布(min/median/p90/max)
  - baseline_tempo_loss / planner_induced_direct_tempo_loss / active_unattackable_observation
  - 5件ずつの具体例(4カテゴリ)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent), str(_HERE.parent.parent / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from matchup_common import atomic_write_json, read_deck, resolve_path  # noqa: E402
from eval_agent_field import build_field, make_tasks  # noqa: E402

MAX_STEPS = 3000
TAPU_BULU = 920
_W: dict = {}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent, ogerpon_planner

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    _W["P"] = ogerpon_planner
    models = {}
    for _a, wp, _d in opponents:
        if wp not in models:
            models[wp] = PolicyModel(wp)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _state_fingerprint(P, cur, me):
    mine = cur.players[me]
    active = next((s for s in (mine.active or []) if s is not None), None)
    bench = [s for s in (mine.bench or []) if s is not None]

    def pk(p):
        return (p.id, tuple(sorted(P.energies_of(p))), p.hp) if p is not None else None

    hand_serials = tuple(sorted(getattr(c, "serial", None) for c in (mine.hand or [])))
    return (cur.turn, cur.energyAttached, pk(active),
           tuple(sorted((pk(b) for b in bench), key=lambda x: (x is None, x))), hand_serials)


def _play(task, game_id):
    from cg.api import OptionType, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    P = _W["P"]
    agent = _W["agent"]
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    agent.OGERPON_SHADOW_LOG = []
    n = 0
    err = None
    reward = 0.0
    exposures = []   # 全 raw exposure(重複あり)
    seen_states = {}  # fingerprint -> record(state重複排除)
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}"}
    try:
        while True:
            log_len_before = len(agent.OGERPON_SHADOW_LOG)
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                err = "max_steps"; break
            if cur.yourIndex == learner_index and sel is not None:
                action = agent.agent(obs, _W["config"])
                if len(agent.OGERPON_SHADOW_LOG) > log_len_before:
                    rec = agent.OGERPON_SHADOW_LOG[-1]
                    fp = _state_fingerprint(P, cur, learner_index)
                    rec2 = dict(rec)
                    rec2["game_id"] = game_id
                    rec2["turn"] = cur.turn
                    rec2["fingerprint"] = fp
                    exposures.append(rec2)
                    seen_states[fp] = rec2  # 同一stateは最後の観測で代表させる
            else:
                if sel is None or not sel.option:
                    action = []
                elif sel.maxCount == 1:
                    oi = opp_pm.select_option(obs)
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp_pm.score_options(obs)
                    nn = len(sel.option)
                    c = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:c]
                              if osc else list(range(c)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    return {"error": err, "reward": reward, "exposures": exposures,
           "unique_states": list(seen_states.values())}


def _play_wrapper(args):
    task, game_id = args
    return _play(task, game_id)


def _pctl(values, p):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, int(round(p * (len(s) - 1))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ml-config", default="og_attach_shadow")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--max-games", type=int, default=150)
    ap.add_argument("--min-unique-attach-turns", type=int, default=50)
    ap.add_argument("--min-flips", type=int, default=20)
    ap.add_argument("--seed", type=int, default=271828)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")

    all_exposures = []
    all_unique_states = []
    total_games = 0
    batch = args.games
    while total_games < args.max_games:
        tasks = make_tasks(weights, batch, args.seed + total_games)
        workdir = tempfile.mkdtemp(prefix="attachshadow_")
        Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
        with Pool(processes=args.workers, initializer=_init,
                  initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
            res = pool.map(_play_wrapper,
                          [(t, total_games + i) for i, t in enumerate(tasks)], chunksize=1)
        for r in res:
            if r.get("error") is None:
                all_exposures.extend(r["exposures"])
                all_unique_states.extend(r["unique_states"])
        total_games += batch
        # unique_attach_turns の近似(state fingerprintのgame_id+turn部分で重複排除)
        turns_seen = {(e["game_id"], e["turn"]) for e in all_exposures if e.get("bulu_energy_before") is not None}
        flips = sum(1 for e in all_exposures
                    if e.get("bulu_energy_before") is not None
                    and e.get("flip_direction", "unchanged") != "unchanged")
        print(f"[progress] games={total_games} raw_exposures={len(all_exposures)} "
              f"unique_attach_turns~={len(turns_seen)} flips={flips}", flush=True)
        if len(turns_seen) >= args.min_unique_attach_turns and flips >= args.min_flips:
            break

    # --- 集計 ---
    bulu_relevant = [e for e in all_exposures if e.get("bulu_energy_before") is not None]
    unique_turn_keys = {}
    for e in bulu_relevant:
        key = (e["game_id"], e["turn"])
        unique_turn_keys.setdefault(key, []).append(e)
    unique_states_by_fp = {}
    for e in bulu_relevant:
        unique_states_by_fp[e["fingerprint"]] = e  # 最後の観測を代表に

    # 注意: pipeline.py の既存 "selection_flipped" は「実際に返した行動 != PIMC単独の最良手」
    # (shadow_only下ではtie_eps由来のブレを含みうる)。ここで比較したいのは
    # baseline_decision(pimc_only_best)とplanner_decision(bonus込みargmax)なので、
    # sink側で計算済みの flip_direction != "unchanged" を正とする。
    def any_flip_in_turn(recs):
        return any(r.get("flip_direction", "unchanged") != "unchanged" for r in recs)

    unique_attach_turns_with_flip = sum(1 for recs in unique_turn_keys.values() if any_flip_in_turn(recs))

    flips = [e for e in unique_states_by_fp.values()
            if e.get("flip_direction", "unchanged") != "unchanged"]
    by_direction: dict[str, int] = {}
    for e in flips:
        d = e.get("flip_direction", "unknown")
        by_direction[d] = by_direction.get(d, 0) + 1

    raw_bonuses = [e["planner_raw_bonus"] for e in unique_states_by_fp.values()
                  if e.get("planner_raw_bonus") is not None]
    margins = [e["pimc_margin"] for e in unique_states_by_fp.values() if e.get("pimc_margin") is not None]
    capped = []
    for e in unique_states_by_fp.values():
        b = e.get("bonuses") or {}
        if b:
            capped.append(max(abs(v) for v in b.values()))

    baseline_tempo = sum(1 for e in unique_states_by_fp.values() if e.get("baseline_tempo_loss"))
    planner_tempo = sum(1 for e in unique_states_by_fp.values()
                        if e.get("planner_induced_direct_tempo_loss"))
    unattackable_obs = sum(1 for e in unique_states_by_fp.values()
                           if e.get("active_unattackable_observation"))

    recommend_bulu = sum(1 for e in unique_states_by_fp.values() if e.get("planner_target_is_bulu"))
    keep_active = sum(1 for e in unique_states_by_fp.values()
                      if e.get("baseline_target_is_bulu") is False
                      and e.get("planner_target_is_bulu") is False)
    reject_bulu = sum(1 for e in unique_states_by_fp.values()
                      if e.get("baseline_target_is_bulu") and not e.get("planner_target_is_bulu"))

    def stat(vals):
        if not vals:
            return {"min": None, "median": None, "p90": None, "max": None}
        return {"min": min(vals), "median": statistics.median(vals),
               "p90": _pctl(vals, 0.9), "max": max(vals)}

    examples = {
        "flipped_to_bulu": [e for e in flips if e.get("planner_target_is_bulu")][:5],
        "kept_active_rejected_bulu": [e for e in unique_states_by_fp.values()
                                      if e.get("baseline_target_is_bulu") is False
                                      and e.get("planner_target_is_bulu") is False
                                      and e.get("bulu_energy_before", 0) is not None][:5],
        "policy_pimc_vs_planner_disagreement": flips[:5],
        "bulu_3to4_completion": [e for e in unique_states_by_fp.values()
                                 if e.get("bulu_energy_before") == 3
                                 and e.get("planner_target_is_bulu")][:5],
        "bonus_but_no_flip": [e for e in unique_states_by_fp.values()
                              if (e.get("bonuses") or {}) and not e.get("selection_flipped")][:5],
    }

    out = {
        "ml_config": args.ml_config, "games_run": total_games,
        "raw_attach_exposures": len(bulu_relevant),
        "unique_attach_states": len(unique_states_by_fp),
        "unique_attach_turns": len(unique_turn_keys),
        "unique_attach_turns_with_flip": unique_attach_turns_with_flip,
        "selection_flipped_count": len(flips),
        "flip_direction_breakdown": by_direction,
        "planner_raw_bonus_stats": stat(raw_bonuses),
        "planner_capped_bonus_stats": stat(capped),
        "pimc_margin_stats": stat(margins),
        "baseline_tempo_loss": baseline_tempo,
        "planner_induced_direct_tempo_loss": planner_tempo,
        "planner_induced_reachable_attack_loss": "未計測(search_step未実装)",
        "active_unattackable_observation": unattackable_obs,
        "planner_recommended_bulu_count": recommend_bulu,
        "planner_kept_active_count": keep_active,
        "planner_rejected_bulu_count": reject_bulu,
        "examples": {k: [{kk: vv for kk, vv in e.items() if kk not in ("resolved_candidates",)}
                         for e in v] for k, v in examples.items()},
    }
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(json.dumps({k: out[k] for k in
                      ("games_run", "raw_attach_exposures", "unique_attach_states",
                       "unique_attach_turns", "selection_flipped_count",
                       "flip_direction_breakdown", "baseline_tempo_loss",
                       "planner_induced_direct_tempo_loss")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
