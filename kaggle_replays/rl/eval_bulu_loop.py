"""カプ・ブルル中継の戦略ループ(手貼り→交代→攻撃→次オーガポン準備→気絶→昇格)の完遂を計測する。

Phase 11.3 相当。試合数よりイベント数を優先する。
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

from matchup_common import atomic_write_json, read_deck  # noqa: E402
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
        "bulu_attach_opportunity": 0, "bulu_attach_actual": 0,
        "bulu_energy_0to1": 0, "bulu_energy_1to2": 0, "bulu_energy_2to3": 0,
        "bulu_energy_3to4": 0, "bulu_completions": 0,
        "bulu_ready_promotions": 0, "bulu_attacks_total": 0,
        "bulu_ko_opponent": 0, "games_bulu_attacked": 0,
        "attack_tempo_loss_events": 0,
        "bulu_died_or_lost_before_completion": 0,
        "strategy_loop_complete": 0,
        "final_my_prize": None, "final_opp_prize": None,
    }
    n = 0
    err = None
    reward = 0.0
    bulu_attacked_this_game = False
    prev_bulu_energy = None
    prev_active_serial = None
    saw_bulu_ready_active = False
    saw_bulu_attack = False
    saw_new_ogerpon_ready_after = False
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}", "events": ev}
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur, sel = obs.current, obs.select
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                ev["final_my_prize"] = len(cur.players[learner_index].prize or [])
                ev["final_opp_prize"] = len(cur.players[1 - learner_index].prize or [])
                break
            if n >= MAX_STEPS:
                err = "max_steps"; break
            if cur.yourIndex == learner_index and sel is not None:
                mine = cur.players[learner_index]
                active = next((s for s in (mine.active or []) if s is not None), None)
                bench = [s for s in (mine.bench or []) if s is not None]
                bulu = next((b for b in bench if b.id == TAPU_BULU), None)
                bulu_active_now = (active is not None and active.id == TAPU_BULU)
                hand_energy = [c for c in (mine.hand or []) if c.id == 1]  # 基本【草】

                if int(sel.type) == int(SelectType.MAIN):
                    if bulu is not None and hand_energy and not cur.energyAttached:
                        ev["bulu_attach_opportunity"] += 1

                    action = _W["agent"].agent(obs, _W["config"])
                    if action and 0 <= action[0] < len(sel.option):
                        opt = sel.option[action[0]]
                        ot = int(getattr(opt, "type", -1))
                        if ot == int(OptionType.ATTACH) and bulu is not None:
                            in_area = getattr(opt, "inPlayArea", None)
                            in_idx = getattr(opt, "inPlayIndex", None)
                            tgt = P._resolve_attach_target(cur, learner_index, opt)  # noqa: SLF001
                            if tgt is not None and tgt.id == TAPU_BULU:
                                ev["bulu_attach_actual"] += 1
                                before = len(P.energies_of(bulu))
                                key = {0: "bulu_energy_0to1", 1: "bulu_energy_1to2",
                                      2: "bulu_energy_2to3", 3: "bulu_energy_3to4"}.get(before)
                                if key:
                                    ev[key] += 1
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
                    action = _W["agent"].agent(obs, _W["config"])

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
    return {"error": err, "reward": reward, "archetype": arch, "events": ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13571113)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from matchup_common import resolve_path

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)

    out = {"games": args.games, "arms": {}}
    for a in (json.loads(x) for x in args.arm):
        name, ml_config = a["name"], a["ml_config"]
        workdir = tempfile.mkdtemp(prefix=f"bululoop_{name}_")
        Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
        t0 = time.time()
        with Pool(processes=args.workers, initializer=_init,
                  initargs=(str(resolve_path(args.weights)), ml_config, workdir, opponents)) as pool:
            res = pool.map(_play, tasks, chunksize=1)
        ok = [r for r in res if r.get("error") is None]
        agg = {}
        for r in ok:
            for k, v in r["events"].items():
                if v is None:
                    continue
                agg[k] = agg.get(k, 0) + v
        wr = sum(1 for r in ok if r["reward"] >= 1.0) / (len(ok) or 1)
        out["arms"][name] = {"ml_config": ml_config, "valid": len(ok), "errors": len(res) - len(ok),
                             "winrate": wr, "wall_seconds": time.time() - t0, "events": agg}
        print(f"[{name}] wr={wr:.3f} errors={len(res)-len(ok)} "
              f"手貼り機会={agg.get('bulu_attach_opportunity',0)} "
              f"実手貼り={agg.get('bulu_attach_actual',0)} "
              f"4エネ完成={agg.get('bulu_completions',0)} "
              f"攻撃可昇格={agg.get('bulu_ready_promotions',0)} "
              f"ブルル攻撃={agg.get('bulu_attacks_total',0)} "
              f"攻撃した試合={agg.get('games_bulu_attacked',0)} "
              f"テンポ損失={agg.get('attack_tempo_loss_events',0)} "
              f"ループ完遂={agg.get('strategy_loop_complete',0)} "
              f"{time.time()-t0:.0f}s", flush=True)
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
