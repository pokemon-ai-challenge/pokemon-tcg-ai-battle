"""同一相手プール・同一先攻後攻割当でのconfig間の強さ比較(Stage1/Stage2共用)。

matchup・deck・先攻後攻を揃えた層別比較。DLL内部RNGは制御不能なため各armの試合乱数は
独立(真のpaired gameではない)。差の信頼区間は独立2標本の比率差として計算する。
25〜50試合ごとにチェックポイントを保存し、中断しても再開できる。
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
OGERPON_EX = 96
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


def _play(task):
    from cg.api import OptionType, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    P = _W["P"]
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    ev = {
        "bulu_attach_opportunity": 0, "bulu_attach_actual": 0, "bulu_completions": 0,
        "bulu_ready_promotions": 0, "bulu_attacks_total": 0, "games_bulu_attacked": 0,
        "attack_tempo_loss_events": 0, "strategy_loop_complete": 0,
    }
    unique_turns = set()
    move_times = []
    n = 0
    err = None
    illegal = False
    reward = 0.0
    prev_active_serial = None
    bulu_attacked_this_game = False
    saw_bulu_ready_active = False
    saw_bulu_attack = False
    saw_new_ogerpon_ready_after = False
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}"}
    try:
        while True:
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
                mine = cur.players[learner_index]
                active = next((s for s in (mine.active or []) if s is not None), None)
                bench = [s for s in (mine.bench or []) if s is not None]
                bulu = next((b for b in bench if b.id == TAPU_BULU), None)
                bulu_active_now = (active is not None and active.id == TAPU_BULU)
                hand_energy = [c for c in (mine.hand or []) if c.id == 1]

                if int(sel.type) == int(SelectType.MAIN):
                    if bulu is not None and hand_energy and not cur.energyAttached:
                        ev["bulu_attach_opportunity"] += 1
                        unique_turns.add(cur.turn)

                    t0 = time.perf_counter()
                    action = _W["agent"].agent(obs, _W["config"])
                    move_times.append((time.perf_counter() - t0) * 1000.0)

                    if action and 0 <= action[0] < len(sel.option):
                        opt = sel.option[action[0]]
                        ot = int(getattr(opt, "type", -1))
                        if ot == int(OptionType.ATTACH) and bulu is not None:
                            tgt = P._resolve_attach_target(cur, learner_index, opt)  # noqa: SLF001
                            if tgt is not None and tgt.id == TAPU_BULU:
                                ev["bulu_attach_actual"] += 1
                                before = len(P.energies_of(bulu))
                                if before == 3:
                                    ev["bulu_completions"] += 1
                            if tgt is not None and tgt.id == OGERPON_EX and active is not None \
                                    and not P.can_attack_now(active):
                                ev["attack_tempo_loss_events"] += 1
                        if ot == int(OptionType.ATTACK) and bulu_active_now:
                            ev["bulu_attacks_total"] += 1
                            bulu_attacked_this_game = True
                            saw_bulu_attack = True
                else:
                    t0 = time.perf_counter()
                    action = _W["agent"].agent(obs, _W["config"])
                    move_times.append((time.perf_counter() - t0) * 1000.0)

                if sel is not None:
                    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
                        illegal = True; err = f"bad type {action}"; break
                    if not (sel.minCount <= len(action) <= sel.maxCount):
                        illegal = True; err = f"bad count {action}"; break
                    if len(action) != len(set(action)):
                        illegal = True; err = f"dup {action}"; break
                    if not all(0 <= i < len(sel.option) for i in action):
                        illegal = True; err = f"oob {action}"; break

                if bulu_active_now and P.can_attack_now(active):
                    if getattr(active, "serial", None) != prev_active_serial:
                        ev["bulu_ready_promotions"] += 1
                        saw_bulu_ready_active = True
                if saw_bulu_ready_active and not bulu_active_now and active is not None \
                        and P.is_ex(active) and P.can_attack_now(active):
                    saw_new_ogerpon_ready_after = True
                if active is not None:
                    prev_active_serial = getattr(active, "serial", None)
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

    if bulu_attacked_this_game:
        ev["games_bulu_attacked"] = 1
    if saw_bulu_ready_active and saw_bulu_attack and saw_new_ogerpon_ready_after:
        ev["strategy_loop_complete"] = 1
    ev["unique_attach_turns"] = len(unique_turns)

    return {"error": err, "illegal": illegal, "reward": reward, "archetype": arch,
           "learner_index": learner_index, "events": ev,
           "move_ms_mean": (sum(move_times) / len(move_times)) if move_times else 0.0,
           "move_ms_max": max(move_times) if move_times else 0.0}


def wilson_ci(w, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (p, max(0.0, (c - m) / d), min(1.0, (c + m) / d))


def diff_ci_independent(w1, n1, w2, n2, z=1.96):
    """独立2標本の比率差 (p2 - p1) の正規近似CI。"""
    if n1 == 0 or n2 == 0:
        return (None, None, None)
    p1, p2 = w1 / n1, w2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    return (diff, diff - z * se, diff + z * se)


def run_arm(name, ml_config, deck, weights, tasks, opponents, workers, checkpoint_path,
           resume):
    done = []
    if resume and checkpoint_path.exists():
        try:
            done = json.loads(checkpoint_path.read_text(encoding="utf-8"))["results"]
        except Exception:  # noqa: BLE001
            done = []
    start_i = len(done)
    remaining = tasks[start_i:]
    if remaining:
        workdir = tempfile.mkdtemp(prefix=f"strength_{name}_")
        Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
        with Pool(processes=workers, initializer=_init,
                  initargs=(str(resolve_path(weights)), ml_config, workdir, opponents)) as pool:
            batch = 40
            for i in range(0, len(remaining), batch):
                chunk = remaining[i:i + batch]
                res = pool.map(_play, chunk, chunksize=1)
                done.extend(res)
                atomic_write_json(checkpoint_path, {"name": name, "ml_config": ml_config,
                                                    "results": done})
                print(f"  [{name}] {len(done)}/{len(tasks)} 完了", flush=True)
    return done


def summarize(name, ml_config, results, opponents):
    ok = [r for r in results if r.get("error") is None]
    illegal = sum(1 for r in results if r.get("illegal"))
    timeouts = sum(1 for r in results if r.get("error") == "max_steps")
    errors = len(results) - len(ok)
    wins = sum(1 for r in ok if r["reward"] >= 1.0)
    valid = len(ok)
    p, lo, hi = wilson_ci(wins, valid)

    by_arch = {}
    by_side = {"first": [0, 0], "second": [0, 0]}
    ev_agg = {}
    move_means = []
    for r in ok:
        a = r["archetype"]
        by_arch.setdefault(a, [0, 0])
        by_arch[a][1] += 1
        by_arch[a][0] += 1 if r["reward"] >= 1.0 else 0
        side = "first" if r["learner_index"] == 0 else "second"
        by_side[side][1] += 1
        by_side[side][0] += 1 if r["reward"] >= 1.0 else 0
        for k, v in r.get("events", {}).items():
            ev_agg[k] = ev_agg.get(k, 0) + v
        move_means.append(r.get("move_ms_mean", 0.0))

    matchup_wr = {a: (v[0] / v[1] if v[1] else None) for a, v in by_arch.items()}
    worst = min((v for v in matchup_wr.values() if v is not None), default=None)

    return {
        "name": name, "ml_config": ml_config, "games": len(results),
        "valid": valid, "errors": errors, "illegal_actions": illegal, "timeouts": timeouts,
        "wins": wins, "winrate": p, "wilson_95ci": [lo, hi],
        "matchup_winrate": matchup_wr, "worst_matchup_winrate": worst,
        "side_winrate": {k: (v[0] / v[1] if v[1] else None) for k, v in by_side.items()},
        "side_counts": by_side,
        "move_ms_mean_avg": (sum(move_means) / len(move_means)) if move_means else 0.0,
        "move_ms_max_observed": max((r.get("move_ms_max", 0.0) for r in ok), default=0.0),
        "events": ev_agg,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help='JSON: {"name":..., "ml_config":...}')
    ap.add_argument("--games", type=int, default=154)
    ap.add_argument("--seed", type=int, default=13131313)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    deck = read_deck(resolve_path(args.deck))
    opponents, weights, skipped = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)
    print(f"field: {len(opponents)} skipped={skipped} tasks={len(tasks)}", flush=True)

    arms = [json.loads(a) for a in args.arm]
    summaries = {}
    for a in arms:
        ckpt = out_dir / f"raw_{a['name']}.json"
        print(f"[{a['name']}] 開始 ({args.games}試合, resume={args.resume})", flush=True)
        results = run_arm(a["name"], a["ml_config"], deck, args.weights, tasks, opponents,
                          args.workers, ckpt, args.resume)
        s = summarize(a["name"], a["ml_config"], results, opponents)
        summaries[a["name"]] = s
        print(f"[{a['name']}] wr={s['winrate']:.3f} 95%CI={s['wilson_95ci']} "
              f"valid={s['valid']} err={s['errors']} illegal={s['illegal_actions']} "
              f"timeout={s['timeouts']} worst_matchup={s['worst_matchup_winrate']}", flush=True)

    baseline_name = arms[0]["name"]
    comparisons = {}
    for a in arms[1:]:
        n = a["name"]
        diff, lo, hi = diff_ci_independent(
            summaries[baseline_name]["wins"], summaries[baseline_name]["valid"],
            summaries[n]["wins"], summaries[n]["valid"])
        comparisons[f"{n} - {baseline_name}"] = {"diff": diff, "ci95": [lo, hi],
                                                  "note": "独立2標本の比率差(paired gameではない)"}
        print(f"[{n} - {baseline_name}] diff={diff:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]"
              if diff is not None else f"[{n} - {baseline_name}] diff=N/A", flush=True)

    out = {"games_per_arm": args.games, "seed": args.seed, "field_size": len(opponents),
          "summaries": summaries, "comparisons": comparisons}
    atomic_write_json(out_dir / "summary.json", out)
    print(f"wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
