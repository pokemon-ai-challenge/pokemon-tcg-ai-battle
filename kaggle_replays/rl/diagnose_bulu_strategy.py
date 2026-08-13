"""カプ・ブルル中継戦略の発火計測 + PIMCスコアとPlanner bonusのスケール確認。

計測するもの:
  A. shadow 記録(og_main_shadow 用): PIMCスコア範囲 / bonus 範囲 / 逆転有無
  B. 戦略ループの機会数と成立数
  C. 具体的な行動トレース(交代した例・しなかった例・PIMCと食い違った例)
"""

from __future__ import annotations

import argparse
import json
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
    _W["planner"] = ogerpon_planner
    ml_policy_agent.OGERPON_SHADOW_LOG = []      # MAIN shadow 記録を有効化
    ml_policy_agent.OGERPON_CARD_LOG = []        # CARD系(ATTACH_TO等)の発火記録
    _W["shadow"] = ml_policy_agent.OGERPON_SHADOW_LOG
    _W["cardlog"] = ml_policy_agent.OGERPON_CARD_LOG
    models = {}
    for _a, wp, _d in opponents:
        if wp not in models:
            models[wp] = PolicyModel(wp)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _play(task):
    from cg.api import OptionType, SelectContext, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    P = _W["planner"]
    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    _W["shadow"].clear()
    _W["cardlog"].clear()
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}"}

    ev = {
        "retreat_considerable": 0,     # 交代検討可能局面(削れたex + ベンチにブルル + にげる合法)
        "bulu_promoted": 0,            # 実際にカプ・ブルルがバトル場に立った回数
        "bulu_promoted_ready": 0,      # そのとき攻撃可能だった回数
        "bulu_attacks": 0,             # カプ・ブルルで攻撃した回数
        "bulu_ko_with_next_ready": 0,  # ブルル退場時に次のオーガポンが完成していた
        "no_attack_turns": 0,          # 自分のターンで攻撃しなかった回数
        "ogerpon_saved": 0,            # 削れたexをベンチへ退避できた回数
        "attach_from_reached": 0,      # 付与先選択に到達した回数
        "attach_from_with_bulu": 0,    # うちベンチにカプ・ブルルがいた回数
    }
    traces = []
    n = 0
    err = None
    reward = 0.0
    prev_active_serial = None
    bulu_was_active = False
    my_turn_attacked = set()
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
                opp = cur.players[1 - learner_index]
                active = next((s for s in (mine.active or []) if s is not None), None)
                bench = [s for s in (mine.bench or []) if s is not None]
                bulu_bench = [b for b in bench if not P.is_ex(b)]
                cfg = P.main_config(_W["config"])

                # ATTACH_FROM(付与先選択)への到達数と、ベンチにカプ・ブルルがいた回数
                if int(sel.context) == int(SelectContext.ATTACH_FROM):
                    ev["attach_from_reached"] = ev.get("attach_from_reached", 0) + 1
                    if bulu_bench:
                        ev["attach_from_with_bulu"] = ev.get("attach_from_with_bulu", 0) + 1

                # 交代検討可能局面か
                if (active is not None and P.is_ex(active) and bulu_bench
                        and int(sel.type) == int(SelectType.MAIN)
                        and any(int(getattr(o, "type", -1)) == int(OptionType.RETREAT)
                                for o in sel.option)):
                    risk, reason = P.ko_risk(active, next((s for s in (opp.active or [])
                                                           if s is not None), None), cfg)
                    if risk in (P.LIKELY_KO, P.CERTAIN_KO):
                        ev["retreat_considerable"] += 1

                # カプ・ブルルがバトル場に立った瞬間
                if active is not None and not P.is_ex(active):
                    if getattr(active, "serial", None) != prev_active_serial:
                        ev["bulu_promoted"] += 1
                        if P.can_attack_now(active):
                            ev["bulu_promoted_ready"] += 1
                        traces.append({"kind": "bulu_promoted", "turn": cur.turn,
                                       "bulu_energy": len(P.energies_of(active)),
                                       "can_attack": P.can_attack_now(active),
                                       "my_prize": len(mine.prize or []),
                                       "opp_prize": len(opp.prize or []),
                                       "next_ogerpon_ready": any(P.is_ex(b) and P.can_attack_now(b)
                                                                 for b in bench)})
                    bulu_was_active = True
                elif bulu_was_active and active is not None and P.is_ex(active):
                    ev["bulu_ko_with_next_ready"] += 1 if P.can_attack_now(active) else 0
                    bulu_was_active = False
                if active is not None:
                    prev_active_serial = getattr(active, "serial", None)

                # 攻撃したか
                if int(sel.type) == int(SelectType.MAIN):
                    chosen = _W["agent"].agent(obs, _W["config"])
                    if chosen and 0 <= chosen[0] < len(sel.option):
                        ot = int(getattr(sel.option[chosen[0]], "type", -1))
                        if ot == int(OptionType.ATTACK):
                            my_turn_attacked.add(cur.turn)
                            if active is not None and not P.is_ex(active):
                                ev["bulu_attacks"] += 1
                    action = chosen
                else:
                    action = _W["agent"].agent(obs, _W["config"])
            elif cur.yourIndex == learner_index:
                action = _W["agent"].agent(obs, _W["config"])
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

    return {"error": err, "reward": reward, "archetype": arch, "events": ev,
            "traces": traces[:6], "shadow": list(_W["shadow"])[:40],
            "cardlog": list(_W["cardlog"])[:60]}


def run_arm(name, deck, weights, ml_config, tasks, opponents, workers):
    workdir = tempfile.mkdtemp(prefix=f"bulu_{name}_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
    t0 = time.time()
    with Pool(processes=workers, initializer=_init,
              initargs=(str(resolve_path(weights)), ml_config, workdir, opponents)) as pool:
        res = pool.map(_play, tasks, chunksize=1)
    ok = [r for r in res if r.get("error") is None]
    agg = {}
    for r in ok:
        for k, v in r["events"].items():
            agg[k] = agg.get(k, 0) + v
    shadows = [s for r in ok for s in r["shadow"]]
    cards = [c for r in ok for c in r["cardlog"]]
    from cg.api import SelectContext
    card_by_ctx = {}
    for c in cards:
        try:
            name = SelectContext(c["context"]).name
        except Exception:
            name = str(c["context"])
        slot = card_by_ctx.setdefault(name, {"fired": 0, "flipped": 0})
        slot["fired"] += 1
        slot["flipped"] += 1 if c.get("flipped") else 0
    margins = [s["pimc_margin"] for s in shadows if s.get("pimc_margin") is not None]
    bonuses = [s["planner_raw_bonus"] for s in shadows if s.get("planner_raw_bonus") is not None]
    pimc_all = [v for s in shadows for v in (s.get("pimc_scores") or {}).values()]
    flips = sum(1 for s in shadows if s.get("selection_flipped"))
    return {
        "ml_config": ml_config, "valid": len(ok), "errors": len(res) - len(ok),
        "wall_seconds": time.time() - t0,
        "winrate": sum(1 for r in ok if r["reward"] >= 1.0) / (len(ok) or 1),
        "events": agg,
        "shadow_events": len(shadows),
        "pimc_score_range": [min(pimc_all), max(pimc_all)] if pimc_all else None,
        "pimc_margin_range": [min(margins), max(margins)] if margins else None,
        "pimc_margin_median": (sorted(margins)[len(margins) // 2] if margins else None),
        "planner_bonus_range": [min(bonuses), max(bonuses)] if bonuses else None,
        "selection_flipped": flips,
        "card_side": {"total_fired": len(cards), "by_context": card_by_ctx,
                      "samples": cards[:8]},
        "traces": [t for r in ok for t in r["traces"]][:25],
        "shadow_samples": shadows[:10],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=8675309)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, skipped = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)
    out = {"deck": str(resolve_path(args.deck)), "games": args.games, "arms": {}}
    for a in (json.loads(x) for x in args.arm):
        r = run_arm(a["name"], deck, args.weights, a["ml_config"], tasks, opponents, args.workers)
        out["arms"][a["name"]] = r
        e = r["events"]
        print(f"[{a['name']}] wr={r['winrate']:.3f} errors={r['errors']} "
              f"交代検討可能={e.get('retreat_considerable',0)} "
              f"ブルル昇格={e.get('bulu_promoted',0)}(攻撃可={e.get('bulu_promoted_ready',0)}) "
              f"ブルル攻撃={e.get('bulu_attacks',0)} "
              f"shadow={r['shadow_events']} flip={r['selection_flipped']} "
              f"{r['wall_seconds']:.0f}s", flush=True)
        print(f"    PIMCスコア範囲={r['pimc_score_range']} margin範囲={r['pimc_margin_range']} "
              f"bonus範囲={r['planner_bonus_range']}", flush=True)
        cs = r["card_side"]
        print(f"    CARD系発火={cs['total_fired']} 内訳={cs['by_context']}", flush=True)
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
