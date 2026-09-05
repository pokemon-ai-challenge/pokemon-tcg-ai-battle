"""E0: cg エンジン RNG の決定性を実測する。

`battle_start` に seed 引数は無く、`cg/game.py` `cg/api.py` に random/seed の参照も無い
= シャッフルはネイティブ DLL 内部。ならば残る可能性は:

  (a) DLL ロード時に固定 seed  -> **別プロセスで同じ呼び出し列を再生すれば同一**
  (b) 時刻等で seed          -> 再現不能

(a) なら「baseline と candidate を別プロセスで同じ game index 列に対して走らせる」だけで
common random numbers が成立し、A/B の分散を大幅に落とせる。

このスクリプトは 1 プロセスで N ゲームを固定方策で回し、勝敗列と手数列を出す。
2回起動して出力が一致するかを外側で比較する。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()


def main() -> None:
    wd = _SUB / "ptcg_ai" / "learning"
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opp", default="mega_lucario_ex")
    ap.add_argument("--pyseed", type=int, default=12345)
    ap.add_argument("--policy-only", action="store_true",
                    help="config を policy 単体(abl_2_policy_only)にして探索を外す")
    args = ap.parse_args()

    cfg_name = "abl_2_policy_only" if args.policy_only else "abl_5_full"
    climb = str(wd / "policy_weights_alakazam_rl_climb.json")
    c = agents.load_config_copy(cfg_name)
    c["policy_weights_path"] = climb
    me = agents.make_ml_policy_agent(c)
    co = agents.load_config_copy(cfg_name)
    co["policy_weights_path"] = str(wd / "policy_weights_{}.json".format(args.opp))
    opp = agents.make_ml_policy_agent(co)
    deck = runner.load_deck(_SUB / "deck.csv")
    do = runner.load_deck(_ROOT / "kaggle_replays" / "meta_analysis"
                          / "archetype_decks" / args.opp / "01.csv")

    outs, steps = [], []
    for g in range(args.games):
        random.seed(args.pyseed + g)
        first = g % 2 == 0
        r = (runner.play_game(me, opp, list(deck), list(do)) if first
             else runner.play_game(opp, me, list(do), list(deck)))
        my = 0 if first else 1
        w = getattr(r, "winner", None)
        outs.append(-1 if w is None else (1 if w == my else 0))
        steps.append(getattr(r, "steps", -1))
    sig = hashlib.sha256(json.dumps([outs, steps]).encode()).hexdigest()[:16]
    print(json.dumps({"config": cfg_name, "games": args.games, "opp": args.opp,
                      "wins": sum(1 for x in outs if x == 1),
                      "outcomes": outs, "steps": steps, "signature": sig},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
