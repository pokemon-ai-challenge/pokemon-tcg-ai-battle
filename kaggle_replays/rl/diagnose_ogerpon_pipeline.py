"""オーガポンデッキで abl_5_full の探索パイプラインが実際にどう動いているかを計測する。

abl_5_full はフーディン Plan A(ポケモン19枚・進化ライン・ふしぎなアメ)向けに調整された
config で、オーガポンデッキ(ポケモン5枚・進化なし・エネルギー19枚)では前提が崩れている
可能性がある。特に:

  - pipeline.time_budget.assumed_total_selects = 400
      1手あたり予算 = 残り時間 / (400 - これまでの選択数)。実際の選択数が 400 から大きく
      外れていると、予算配分がずれる(少なすぎれば時間を余らせ、多すぎれば途中で息切れ)。
  - pipeline.top1_shortcut_prob = 0.9
      Policy の top1 確率がこれを超えると探索せず即決する。オーガポンBC の top1 精度は
      0.7116(フーディンより低い)なので、発火率が変わる。
  - lethal_search.max_remaining_prizes = 3 / pipeline.top_k / num_determinizations

ここでは production コードを一切変更せず、外側から計測する:
  - 1試合あたりの select 回数(agent が呼ばれた回数)
  - そのうち pipeline 適用対象になった回数・実際に探索が走った回数
  - lethal 探索の発火回数
  - 1手あたりの実測時間
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
from eval_agent_field import build_field  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}


def _init(weights_path, ml_config_name, workdir, opponents):
    import os
    os.environ["PTCG_AI_ML_CONFIG"] = ml_config_name
    os.chdir(workdir)
    from ptcg_ai.core.config import load_config
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent

    config = dict(load_config(ml_config_name))
    config["policy_weights_path"] = weights_path
    _W["config"] = config
    _W["agent"] = ml_policy_agent

    # production を書き換えずに計測するため、モジュール属性を wrap する。
    counters = {"lethal_fired": 0, "pipeline_returned": 0, "select_calls": 0,
                "move_ms": []}
    _W["counters"] = counters
    orig_lethal = ml_policy_agent._try_lethal
    orig_pipeline = ml_policy_agent._try_pipeline

    def wrapped_lethal(obs, config=None):
        r = orig_lethal(obs, config=config)
        if r is not None:
            counters["lethal_fired"] += 1
        return r

    def wrapped_pipeline(obs, config=None):
        r = orig_pipeline(obs, config=config)
        if r is not None:
            counters["pipeline_returned"] += 1
        return r

    ml_policy_agent._try_lethal = wrapped_lethal
    ml_policy_agent._try_pipeline = wrapped_pipeline

    models: dict[str, PolicyModel] = {}
    for _arch, wpath, _deck in opponents:
        if wpath not in models:
            models[wpath] = PolicyModel(wpath)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _play(task):
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    learner_index, seed, opp_idx = task
    random.seed(seed)
    arch, wpath, deck_o = _W["opponents"][opp_idx]
    opp_pm = _W["opp_models"][wpath]
    deck_l = read_deck(Path("deck.csv"))
    deck0, deck1 = (deck_l, deck_o) if learner_index == 0 else (deck_o, deck_l)

    c = _W["counters"]
    before = {k: (len(v) if isinstance(v, list) else v) for k, v in c.items()}

    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return {"error": f"start {sd.errorType}"}
    n = 0
    my_selects = 0
    err = None
    reward = 0.0
    move_times = []
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
                my_selects += 1
                t0 = time.perf_counter()
                action = _W["agent"].agent(obs, _W["config"])
                move_times.append((time.perf_counter() - t0) * 1000.0)
            else:
                if sel is None or not sel.option:
                    action = []
                elif sel.maxCount == 1:
                    oi = opp_pm.select_option(obs)
                    action = [oi if oi is not None else 0]
                else:
                    osc = opp_pm.score_options(obs)
                    nn = len(sel.option)
                    count = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                              if osc else list(range(count)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    return {
        "error": err, "reward": reward, "archetype": arch,
        "my_selects": my_selects, "total_steps": n,
        "lethal_fired": c["lethal_fired"] - before["lethal_fired"],
        "pipeline_returned": c["pipeline_returned"] - before["pipeline_returned"],
        "move_ms_total": sum(move_times),
        "move_ms_mean": statistics.mean(move_times) if move_times else 0.0,
        "move_ms_max": max(move_times) if move_times else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ml-config", default="abl_5_full")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=555555)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    deck = read_deck(resolve_path(args.deck))
    opponents, weights, _ = build_field(deck_gen="g2")
    rng = random.Random(args.seed)
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    def sample():
        r = rng.random()
        for i, cc in enumerate(cum):
            if r <= cc:
                return i
        return len(cum) - 1

    tasks = [(g % 2, args.seed + g, sample()) for g in range(args.games)]
    workdir = tempfile.mkdtemp(prefix="diag_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")

    with Pool(processes=args.workers, initializer=_init,
              initargs=(str(resolve_path(args.weights)), args.ml_config, workdir, opponents)) as pool:
        res = pool.map(_play, tasks, chunksize=1)

    ok = [r for r in res if r.get("error") is None]
    if not ok:
        print("ALL GAMES ERRORED:", res[:2])
        return
    sel = [r["my_selects"] for r in ok]
    lethal = [r["lethal_fired"] for r in ok]
    pipe = [r["pipeline_returned"] for r in ok]
    mv_mean = [r["move_ms_mean"] for r in ok]
    mv_max = [r["move_ms_max"] for r in ok]
    mv_tot = [r["move_ms_total"] for r in ok]

    summary = {
        "deck": str(resolve_path(args.deck)), "weights": str(resolve_path(args.weights)),
        "ml_config": args.ml_config, "games": args.games, "valid": len(ok),
        "errors": len(res) - len(ok),
        "winrate": sum(1 for r in ok if r["reward"] >= 1.0) / len(ok),
        "my_selects_per_game": {"mean": statistics.mean(sel), "median": statistics.median(sel),
                                 "min": min(sel), "max": max(sel)},
        "assumed_total_selects_in_config": 400,
        "lethal_fired_per_game": statistics.mean(lethal),
        "lethal_fire_rate": sum(lethal) / sum(sel) if sum(sel) else 0.0,
        "pipeline_returned_per_game": statistics.mean(pipe),
        "pipeline_return_rate": sum(pipe) / sum(sel) if sum(sel) else 0.0,
        "move_ms_mean_of_game_means": statistics.mean(mv_mean),
        "move_ms_max_observed": max(mv_max),
        "agent_ms_per_game": {"mean": statistics.mean(mv_tot), "max": max(mv_tot)},
        "time_budget_total_ms_in_config": 540000,
    }
    atomic_write_json(Path(resolve_path(args.output)), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
