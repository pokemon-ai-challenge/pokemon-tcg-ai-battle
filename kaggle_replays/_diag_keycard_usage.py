"""②③ルール探し: strategist の key card が『手札にあるのに展開されない』率を測る(Sacred Ash と同手法)。

持続カード(展開=場にある): Shaymin(343, ベンチ保護特性), Battle Cage(1264, スタジアム)。
単発カード(展開=トラッシュにある=撃った): Boss(1182), Xerosic(1197), Sacred Ash(1129)。

各試合・各key cardで「自分のターンに手札にあった回数」と「最終的に展開されたか」を集計。
『手札に長く持っていたのに展開しなかった』カード = 模倣の単一決定の取りこぼし候補。

使い方: python kaggle_replays/_diag_keycard_usage.py --opp mega_lucario_ex --games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
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
MAX_STEPS = 3000

NAMES = {343: "Shaymin", 1264: "BattleCage", 1182: "Boss", 1197: "Xerosic", 1129: "SacredAsh"}
PERSISTENT = {343, 1264}   # 展開 = 場に存在
ONESHOT = {1182, 1197, 1129}  # 展開 = トラッシュに存在(=撃った)


def _cid(c):
    return getattr(c, "cardId", getattr(c, "id", None))


def _capture(agent0, agent1, deck0, deck1, seed):
    import random
    random.seed(seed)
    agents = {0: agent0, 1: agent1}
    obs_dict, sd = battle_start(deck0, deck1)
    if sd.errorType != 0:
        return None
    steps = 0
    in_hand_turns = defaultdict(int)  # cardId -> 自分のターンで手札にあった回数
    deployed = {}                      # cardId -> bool(展開されたか)
    seen_my_turn = set()               # そのターンに既にカウント済みか(turn番号)
    try:
        while steps < MAX_STEPS:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None or cur.result != -1:
                break
            if cur.yourIndex == 0:
                me = cur.players[0]
                # ターン先頭でのみカウント(同ターンの複数selectで多重計上しない)
                if cur.turn not in seen_my_turn:
                    seen_my_turn.add(cur.turn)
                    hand_ids = {_cid(c) for c in (me.hand or [])}
                    active_ids = {getattr(p, "id", None) for p in (me.active or []) if p}
                    bench_ids = {getattr(p, "id", None) for p in (me.bench or []) if p}
                    in_play = active_ids | bench_ids
                    stadium_ids = {_cid(c) for c in (cur.stadium or [])}
                    discard_ids = {_cid(c) for c in (me.discard or [])}
                    for k in NAMES:
                        if k in hand_ids:
                            in_hand_turns[k] += 1
                        dep = deployed.get(k, False)
                        if k in PERSISTENT:
                            dep = dep or (k in in_play) or (k in stadium_ids)
                        else:
                            dep = dep or (k in discard_ids)
                        deployed[k] = dep
                obs_dict = battle_select(agents[0](obs))
            else:
                obs_dict = battle_select(agents[1](obs))
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        battle_finish()
    return {"winner": cur.result if cur else None,
            "in_hand_turns": dict(in_hand_turns), "deployed": dict(deployed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="mega_lucario_ex")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(_HERE / f"_diag_keycard_usage_{args.opp}.json")

    os.chdir(_ROOT / "sample_submission")
    wpath = WDIR / f"policy_weights_{args.opp}.json"
    my = run_league.build_agent("ml_policy", None, "abl_5_full")
    opp = run_league.build_agent("ml_policy", str(wpath) if wpath.exists() else None, "abl_5_full")
    deck0 = run_league.read_deck_csv_file(str(PROD_DECK))
    deck1 = run_league.read_deck_csv_file(str(DECKDIR / args.opp / "01.csv"))

    print(f"=== key card 展開診断: prod vs {args.opp} ({args.games}試合) ===", flush=True)
    rows = [r for i in range(args.games) if (r := _capture(my, opp, deck0, deck1, i)) is not None and "error" not in r]
    print(f"valid games: {len(rows)}")

    agg = {}
    for k, name in NAMES.items():
        games_in_hand = [r for r in rows if r["in_hand_turns"].get(k, 0) > 0]
        n_ih = len(games_in_hand)
        n_dep = sum(1 for r in games_in_hand if r["deployed"].get(k))
        avg_hand_turns = (sum(r["in_hand_turns"].get(k, 0) for r in games_in_hand) / n_ih) if n_ih else 0
        # 手札に3ターン以上あったのに展開しなかった試合数
        held_not_dep = sum(1 for r in rows if r["in_hand_turns"].get(k, 0) >= 3 and not r["deployed"].get(k))
        dep_rate = (n_dep / n_ih * 100) if n_ih else 0.0
        agg[name] = {"games_in_hand": n_ih, "deploy_rate_pct": round(dep_rate, 1),
                     "avg_turns_in_hand": round(avg_hand_turns, 1), "held3plus_not_deployed": held_not_dep}
        print(f"  {name:<11} 手札に来た試合={n_ih:2d}/{len(rows)}  展開率={dep_rate:5.1f}%  "
              f"平均手札滞在={avg_hand_turns:.1f}t  '3t以上持って未展開'={held_not_dep}試合", flush=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"opp": args.opp, "games": len(rows), "by_card": agg}, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
