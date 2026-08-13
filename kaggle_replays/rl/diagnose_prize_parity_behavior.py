"""サイド価値項が「狙った挙動」を実際に変えたかを直接数える。

フィールド全体の平均勝率は、該当局面が出ない大多数の試合で効果が薄まるため、
「サイドが偶数のときにカプ・ブルル(1枚ポケモン)を使わずオーガポンexだけで戦って負ける」
という報告された負け方が減ったかを直接測る。

arm ごとに1試合あたり次を記録する:
  - 1枚ポケモン(非ex)がバトル場に出たか / 出ていた選択数
  - 1枚ポケモンが最後までベンチに眠ったまま負けたか  ← 報告された負け方
  - 自分のサイドが偶数のときに1枚ポケモンがバトル場にいた選択数
  - 気絶によって相手に渡したサイドの内訳(1枚/2枚)
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

from matchup_common import atomic_write_json, read_deck, resolve_path, sha256_file  # noqa: E402
from eval_agent_field import build_field, make_tasks  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}


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
    models: dict[str, PolicyModel] = {}
    for _arch, wpath, _deck in opponents:
        if wpath not in models:
            models[wpath] = PolicyModel(wpath)
    _W["opp_models"] = models
    _W["opponents"] = opponents


def _prize_value(pokemon):
    try:
        return 2 if _W["card_cache"].get_card(int(pokemon.id)).ex else 1
    except Exception:  # noqa: BLE001
        return 1


def _play(task):
    from cg.api import to_observation_class
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

    n = 0
    err = None
    reward = 0.0
    my_selects = 0
    onep_active_selects = 0        # 1枚ポケモンがバトル場にいた自分の選択数
    onep_active_even_prize = 0     # うち自分の残サイドが偶数だったもの
    onep_ever_active = False
    onep_ever_in_play = False      # ベンチ含め場に出たか
    onep_on_bench_selects = 0
    last_my_prize = None
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
                mine = cur.players[learner_index]
                my_prize = len(mine.prize or [])
                last_my_prize = my_prize
                act_onep = any(s is not None and _prize_value(s) == 1
                               for s in (mine.active or []))
                bench_onep = any(s is not None and _prize_value(s) == 1
                                 for s in (mine.bench or []))
                if act_onep:
                    onep_ever_active = True
                    onep_ever_in_play = True
                    onep_active_selects += 1
                    if my_prize % 2 == 0:
                        onep_active_even_prize += 1
                if bench_onep:
                    onep_ever_in_play = True
                    onep_on_bench_selects += 1
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
                    count = max(sel.minCount, min(sel.maxCount, nn))
                    action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                              if osc else list(range(count)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    finally:
        battle_finish()

    lost = (err is None and reward < 1.0)
    return {
        "error": err, "reward": reward, "archetype": arch, "my_selects": my_selects,
        "onep_ever_active": onep_ever_active,
        "onep_ever_in_play": onep_ever_in_play,
        "onep_active_selects": onep_active_selects,
        "onep_active_even_prize": onep_active_even_prize,
        "onep_on_bench_selects": onep_on_bench_selects,
        # 報告された負け方: 1枚ポケモンを一度もバトル場に出さないまま負けた
        "lost_without_using_onep": bool(lost and not onep_ever_active),
        "lost_with_onep_stuck_on_bench": bool(lost and not onep_ever_active
                                              and onep_on_bench_selects > 0),
        "final_my_prize": last_my_prize,
    }


def run_arm(name, deck_path, weights_path, ml_config, tasks, opponents, workers):
    deck = read_deck(resolve_path(deck_path))
    workdir = tempfile.mkdtemp(prefix=f"behav_{name}_")
    Path(workdir, "deck.csv").write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")
    t0 = time.time()
    with Pool(processes=workers, initializer=_init,
              initargs=(str(resolve_path(weights_path)), ml_config, workdir, opponents)) as pool:
        res = pool.map(_play, tasks, chunksize=1)
    ok = [r for r in res if r.get("error") is None]
    n = len(ok) or 1
    losses = [r for r in ok if r["reward"] < 1.0]
    nl = len(losses) or 1
    total_sel = sum(r["my_selects"] for r in ok) or 1
    return {
        "ml_config": ml_config, "valid": len(ok), "errors": len(res) - len(ok),
        "wall_seconds": time.time() - t0,
        "winrate": sum(1 for r in ok if r["reward"] >= 1.0) / n,
        "games_onep_ever_active_pct": sum(1 for r in ok if r["onep_ever_active"]) / n,
        "games_onep_ever_in_play_pct": sum(1 for r in ok if r["onep_ever_in_play"]) / n,
        "onep_active_share_of_selects": sum(r["onep_active_selects"] for r in ok) / total_sel,
        "onep_active_even_prize_share": sum(r["onep_active_even_prize"] for r in ok) / total_sel,
        "losses": len(losses),
        "lost_without_using_onep_pct_of_losses":
            sum(1 for r in losses if r["lost_without_using_onep"]) / nl,
        "lost_with_onep_stuck_on_bench_pct_of_losses":
            sum(1 for r in losses if r["lost_with_onep_stuck_on_bench"]) / nl,
        "_per_game": [{"won": r["reward"] >= 1.0,
                       "onep_active": r["onep_ever_active"]} for r in ok],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", action="append", required=True,
                    help='JSON: {"name":..., "ml_config":...}')
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--seed", type=int, default=246810)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    opponents, weights, skipped = build_field(deck_gen="g2")
    tasks = make_tasks(weights, args.games, args.seed)
    print(f"field: {len(opponents)} skipped={skipped}", flush=True)

    out = {"deck": str(resolve_path(args.deck)),
           "deck_sha256": sha256_file(resolve_path(args.deck)),
           "weights": str(resolve_path(args.weights)),
           "games": args.games, "seed": args.seed, "arms": {}}
    for a in (json.loads(x) for x in args.arm):
        r = run_arm(a["name"], args.deck, args.weights, a["ml_config"], tasks, opponents,
                    args.workers)
        out["arms"][a["name"]] = {k: v for k, v in r.items() if k != "_per_game"}
        print(f"[{a['name']}] wr={r['winrate']:.4f} "
              f"1枚ポケモンをバトル場に出した試合={r['games_onep_ever_active_pct']*100:.1f}% "
              f"一度も出さずに負け={r['lost_without_using_onep_pct_of_losses']*100:.1f}%(敗戦中) "
              f"{r['wall_seconds']:.0f}s", flush=True)
    atomic_write_json(Path(resolve_path(args.output)), out)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
