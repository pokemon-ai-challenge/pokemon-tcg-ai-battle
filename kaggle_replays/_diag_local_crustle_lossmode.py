"""vs-crustle 縦串P0 の妥当性検証: ローカルcrustle戦で本番エージェントは『どう負けているか』。

前提の検証: 勝率33%が実戦と一致しても、負けの"機構"(山札切れ vs サイドレース)まで一致する
保証はない。deck_sustain(山切れ対策)が効かない(Δ+0.0pt)理由が「そもそもローカルの負けが
山切れではない」なら、この測定vehicle自体が実戦の deckout 敗因を再現していないことになる。

本番フルエージェント(abl_5_full, player0) vs crustle模倣(player1)を N 試合回し、自分(player0)が
負けた試合を分類する:
  - deckout    : 自分の最終 deckCount == 0(山札切れ)。
  - prize_loss : 相手のサイド残 == 0(相手が6枚取り切った=通常のサイドレース負け)。
  - other      : それ以外(バトル場ポケモン切れ等)。

使い方: python kaggle_replays/_diag_local_crustle_lossmode.py --games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def _play_capture(agent0, agent1, deck0, deck1, seed):
    """1試合を回し、終端の (winner, my_deckcount, my_prize_left, opp_prize_left, turns, sustain_fires)。

    my = player0。sustain_fires は deck_sustain がこの試合で発火した回数(モジュール計数)。
    """
    import random
    random.seed(seed)
    agents = {0: agent0, 1: agent1}
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        return None
    steps = 0
    last = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                return None
            if cur.result != -1:
                p0, p1 = cur.players[0], cur.players[1]
                return {
                    "winner": cur.result,
                    "my_deckcount": p0.deckCount,
                    "my_prize_left": len(p0.prize) if p0.prize is not None else None,
                    "opp_prize_left": len(p1.prize) if p1.prize is not None else None,
                    "turns": cur.turn,
                }
            if steps >= MAX_STEPS:
                return None
            action = agents[cur.yourIndex](obs)
            obs_dict = battle_select(action)
            steps += 1
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    finally:
        battle_finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--config", default="abl_5_full", help="自分側 config")
    ap.add_argument("--out", default=str(_HERE / "_diag_local_crustle_lossmode_results.json"))
    args = ap.parse_args()

    # full agent の read_deck_csv()/_get_deck() は cwd 相対の deck.csv を読む(無ければ Kaggle
    # パスへフォールバック)。逐次実行では chdir されないため、eval_field.py に倣い cwd を
    # sample_submission にそろえる(deck/weights は絶対で渡すので影響なし)。
    os.chdir(_ROOT / "sample_submission")

    my_agent = run_league.build_agent("ml_policy", None, args.config)
    crustle = run_league.build_agent("ml_policy", str(WDIR / "policy_weights_crustle.json"), "abl_5_full")
    deck0 = run_league.read_deck_csv_file(str(PROD_DECK))
    deck1 = run_league.read_deck_csv_file(str(DECKDIR / "crustle" / "01.csv"))

    print(f"=== ローカルcrustle 負け機構 (config={args.config}, {args.games}試合) ===", flush=True)
    t0 = time.time()
    rows = []
    for i in range(args.games):
        r = _play_capture(my_agent, crustle, deck0, deck1, seed=i)
        if r is not None:
            rows.append(r)

    valid = [r for r in rows if "error" not in r]
    my_wins = [r for r in valid if r["winner"] == 0]
    my_losses = [r for r in valid if r["winner"] == 1]

    def classify(r):
        if r["my_deckcount"] == 0:
            return "deckout"
        if r["opp_prize_left"] == 0:
            return "prize_loss"
        return "other"

    from collections import Counter
    loss_modes = Counter(classify(r) for r in my_losses)

    print(f"  valid={len(valid)}  my_wins={len(my_wins)}  my_losses={len(my_losses)}"
          f"  win_rate={len(my_wins)/len(valid)*100:.1f}%" if valid else "  no valid games", flush=True)
    print(f"  負けの機構内訳: {dict(loss_modes)}", flush=True)
    if my_losses:
        avg_deck = sum(r["my_deckcount"] for r in my_losses) / len(my_losses)
        avg_turn = sum(r["turns"] for r in my_losses) / len(my_losses)
        print(f"  負け試合の 自分の最終deckCount 平均={avg_deck:.1f}  最終turn平均={avg_turn:.1f}", flush=True)
    print(f"  elapsed={time.time()-t0:.1f}s", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config, "games": args.games,
            "valid": len(valid), "my_wins": len(my_wins), "my_losses": len(my_losses),
            "loss_modes": dict(loss_modes),
            "losses_detail": my_losses,
        }, f, ensure_ascii=False, indent=1)
    print(f"  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
