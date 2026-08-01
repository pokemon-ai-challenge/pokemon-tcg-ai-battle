"""ブローアウト機構検証(ユーザー仮説): 対アグロで『勝ち筋を裏で育てられているか/防御ツールを
使えているか』。使えていない=立ち回りレバー実在。使えて轢かれる=構造(deck-speed)。

本番フルエージェント(abl_5_full, player0) vs mega_lucario 模倣(player1)を N 試合。各試合の
"最終盤面"から自分(player0)のセットアップ達成度を捉える:
  - alakazam_online : 勝ち筋 Alakazam(743) が場(active/bench)に居るか。
  - kadabra_online  : 進化途中 Kadabra(742) が居るか(部分セットアップ)。
  - bench_count     : ベンチ枚数。
  - battle_cage_up  : 防御スタジアム Battle Cage(1264) が出ているか。
  - active_id / active_energy : バトル場に何がいて何エネ乗っていたか(犠牲運用できているか)。
  - prizes_taken    : 6 - 自分のサイド残(轢かれ具合)。

勝ち/負けで上記を対比し、「敗戦では Alakazam が育っていない/Battle Cage を使えていない」なら
ユーザーの言う『犠牲+裏で育てる立ち回り』が効く余地=足場で直せる。

使い方: python kaggle_replays/_diag_local_aggro_setup.py --games 24 --opp mega_lucario_ex
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402
from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
PROD_DECK = _ROOT / "sample_submission" / "deck.csv"

ALAKAZAM, KADABRA, ABRA, DUNSPARCE, BATTLE_CAGE = 743, 742, 741, 65, 1264
MAX_STEPS = 3000


def _pids(pokemons):
    out = []
    for p in pokemons or []:
        if p is not None:
            out.append(getattr(p, "id", None))
    return out


def _capture(agent0, agent1, deck0, deck1, seed):
    import random
    random.seed(seed)
    agents = {0: agent0, 1: agent1}
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return None
    steps = 0
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                return None
            if cur.result != -1:
                me = 0
                p0 = cur.players[me]
                active_ids = _pids(p0.active)
                bench_ids = _pids(p0.bench)
                in_play = active_ids + bench_ids
                stadium_ids = [getattr(c, "cardId", getattr(c, "id", None)) for c in (cur.stadium or [])]
                active0 = p0.active[0] if p0.active and p0.active[0] is not None else None
                return {
                    "winner": cur.result,
                    "prizes_taken": 6 - (len(p0.prize) if p0.prize is not None else 6),
                    "alakazam_online": ALAKAZAM in in_play,
                    "kadabra_online": KADABRA in in_play,
                    "bench_count": len(bench_ids),
                    "battle_cage_up": BATTLE_CAGE in stadium_ids,
                    "active_id": active_ids[0] if active_ids else None,
                    "active_energy": len(getattr(active0, "energies", []) or []) if active0 else 0,
                    "turns": cur.turn,
                }
            if steps >= MAX_STEPS:
                return None
            obs_dict = battle_select(agents[cur.yourIndex](obs))
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        battle_finish()


def _summarize(rows, label):
    valid = [r for r in rows if "error" not in r]
    if not valid:
        print(f"  [{label}] no valid games"); return
    wins = [r for r in valid if r["winner"] == 0]
    losses = [r for r in valid if r["winner"] == 1]
    print(f"  [{label}] valid={len(valid)} win_rate={len(wins)/len(valid)*100:.1f}% ({len(wins)}W/{len(losses)}L)")

    def frac(subset, key):
        return (sum(1 for r in subset if r[key]) / len(subset) * 100) if subset else 0.0

    def avg(subset, key):
        return (sum(r[key] for r in subset) / len(subset)) if subset else 0.0

    for name, sub in (("WINS", wins), ("LOSSES", losses)):
        if not sub:
            continue
        print(f"    {name} n={len(sub)}: alakazam_online={frac(sub,'alakazam_online'):.0f}%  "
              f"kadabra+={frac(sub,'kadabra_online'):.0f}%  bench_avg={avg(sub,'bench_count'):.1f}  "
              f"battle_cage_up={frac(sub,'battle_cage_up'):.0f}%  active_energy_avg={avg(sub,'active_energy'):.1f}  "
              f"turns_avg={avg(sub,'turns'):.1f}")
    if losses:
        print(f"    LOSS prizes_taken dist: {dict(Counter(r['prizes_taken'] for r in losses))}")
    return {"wins": len(wins), "losses": len(losses), "valid": len(valid),
            "loss_alakazam_online_pct": frac(losses, "alakazam_online"),
            "loss_battle_cage_up_pct": frac(losses, "battle_cage_up"),
            "loss_bench_avg": avg(losses, "bench_count")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--opp", default="mega_lucario_ex")
    ap.add_argument("--out", default=str(_HERE / "_diag_local_aggro_setup_results.json"))
    args = ap.parse_args()

    os.chdir(_ROOT / "sample_submission")
    my = run_league.build_agent("ml_policy", None, "abl_5_full")
    opp = run_league.build_agent("ml_policy", str(WDIR / f"policy_weights_{args.opp}.json"), "abl_5_full")
    deck0 = run_league.read_deck_csv_file(str(PROD_DECK))
    deck1 = run_league.read_deck_csv_file(str(DECKDIR / args.opp / "01.csv"))

    print(f"=== 対アグロ setup 診断: prod vs {args.opp} ({args.games}試合) ===", flush=True)
    t0 = time.time()
    rows = [r for i in range(args.games) if (r := _capture(my, opp, deck0, deck1, seed=i)) is not None]
    summary = _summarize(rows, args.opp)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"opp": args.opp, "games": args.games, "summary": summary,
                   "rows": [r for r in rows if "error" not in r]}, f, ensure_ascii=False, indent=1)
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
