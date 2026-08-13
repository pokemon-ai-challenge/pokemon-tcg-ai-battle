"""オーガポンデッキで、どの意思決定点がどの経路(リーサル/PIMC/Policy)で決まっているかを実測する。

背景: pipeline.search は SelectType.MAIN かつ maxCount==1 にしか適用されない
(pipeline.py の適用範囲)。したがって「エネルギーを誰に付けるか(ATTACH_TO)」
「気絶後に誰を前に出すか(TO_ACTIVE)」が MAIN 以外なら、leaf_eval をいくら直しても
それらの判断には構造的に届かない。ここではそれを実測で確定させる。

さらに TO_ACTIVE(昇格)の瞬間に、ベンチの各候補が「戦える状態か(技を撃つのに足りる
エネルギーが乗っているか)」を記録し、実際にどれが選ばれたかを突き合わせる。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import Counter
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

# カプ・ブルルの技コスト(ウッドハンマー ●●●● = 4)。オーガポンex は ●●● = 3。
ATTACK_COST = {920: 4, 96: 3}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent
    from ptcg_ai.shared import card_cache

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent
    _W["card_cache"] = card_cache

    # どの経路で決まったかを数えるため、production を書き換えずに wrap する。
    stats = {"lethal": 0, "pipeline": 0}
    _W["stats"] = stats
    orig_l, orig_p = ml_policy_agent._try_lethal, ml_policy_agent._try_pipeline

    def wl(obs, config=None):
        r = orig_l(obs, config=config)
        if r is not None:
            stats["lethal"] += 1
        return r

    def wp(obs, config=None):
        r = orig_p(obs, config=config)
        if r is not None:
            stats["pipeline"] += 1
        return r

    ml_policy_agent._try_lethal, ml_policy_agent._try_pipeline = wl, wp

    models = {}
    for _a, wpath, _d in opponents:
        if wpath not in models:
            models[wpath] = PolicyModel(wpath)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _is_ex(pid):
    try:
        return bool(_W["card_cache"].get_card(int(pid)).ex)
    except Exception:  # noqa: BLE001
        return False


def _play(task):
    from cg.api import SelectContext, SelectType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}"}

    st = _W["stats"]
    base = dict(st)
    ctx_counter = Counter()          # (SelectType, SelectContext) -> 回数
    pipeline_eligible = 0
    to_active_events = []            # 昇格の瞬間の記録
    n = 0
    err = None
    reward = 0.0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                err = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == learner_index else 0.0
                break
            if n >= MAX_STEPS:
                err = "max_steps"; break
            sel = obs.select
            if cur.yourIndex == learner_index:
                if sel is not None:
                    stype = sel.type.name if hasattr(sel.type, "name") else str(sel.type)
                    sctx = sel.context.name if hasattr(sel.context, "name") else str(sel.context)
                    ctx_counter[(stype, sctx)] += 1
                    if stype == "MAIN" and sel.maxCount == 1:
                        pipeline_eligible += 1
                    # 昇格(TO_ACTIVE)の瞬間: ベンチ各候補の戦闘準備状況を記録する。
                    if sctx == "TO_ACTIVE":
                        mine = cur.players[learner_index]
                        cands = []
                        for slot in (mine.bench or []):
                            if slot is None:
                                continue
                            need = ATTACK_COST.get(int(slot.id))
                            e = len(slot.energies or [])
                            cands.append({"id": int(slot.id), "ex": _is_ex(slot.id),
                                          "energy": e, "need": need,
                                          "ready": (need is not None and e >= need)})
                        to_active_events.append({
                            "my_prize": len(mine.prize or []),
                            "candidates": cands,
                            "any_nonex_ready": any(c["ready"] and not c["ex"] for c in cands),
                            "any_nonex_present": any(not c["ex"] for c in cands),
                        })
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
                    cnt = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:cnt]
                              if osc else list(range(cnt)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    return {"error": err, "reward": reward,
            "ctx": {f"{k[0]}|{k[1]}": v for k, v in ctx_counter.items()},
            "pipeline_eligible": pipeline_eligible,
            "lethal": st["lethal"] - base["lethal"],
            "pipeline": st["pipeline"] - base["pipeline"],
            "to_active_events": to_active_events}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ml-config", default="abl_5_full")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13572468)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)
    workdir = tempfile.mkdtemp(prefix="ctxdiag_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        res = pool.map(_play, tasks, chunksize=1)
    ok = [r for r in res if r.get("error") is None]
    if not ok:
        print("ALL ERRORED", res[:2]); return

    total_ctx = Counter()
    for r in ok:
        for k, v in r["ctx"].items():
            total_ctx[k] += v
    total_selects = sum(total_ctx.values())
    ev = [e for r in ok for e in r["to_active_events"]]

    ready_when_promoting = sum(1 for e in ev if e["any_nonex_ready"])
    present_when_promoting = sum(1 for e in ev if e["any_nonex_present"])

    out = {
        "ml_config": args.ml_config, "games": args.games, "valid": len(ok),
        "errors": len(res) - len(ok), "wall_seconds": time.time() - t0,
        "winrate": sum(1 for r in ok if r["reward"] >= 1.0) / len(ok),
        "total_my_selects": total_selects,
        "pipeline_eligible_selects": sum(r["pipeline_eligible"] for r in ok),
        "pipeline_eligible_share": sum(r["pipeline_eligible"] for r in ok) / (total_selects or 1),
        "pipeline_actually_returned": sum(r["pipeline"] for r in ok),
        "lethal_fired": sum(r["lethal"] for r in ok),
        "select_context_distribution": [
            {"key": k, "count": v, "share": v / (total_selects or 1)}
            for k, v in total_ctx.most_common(20)],
        "to_active_events": len(ev),
        "to_active_with_nonex_on_bench": present_when_promoting,
        "to_active_with_nonex_BATTLE_READY": ready_when_promoting,
        "to_active_ready_share_of_events":
            ready_when_promoting / (len(ev) or 1),
        "note": "TO_ACTIVE = オーガポン気絶後などの昇格選択。BATTLE_READY = 技コストを"
                "満たすエネルギーが乗った非exがベンチにいた回数。",
    }
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
