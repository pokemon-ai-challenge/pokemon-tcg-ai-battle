"""Phase19.15-R §18/§21: 独立seedでの FIELD 再評価。

train_field.py の best checkpoint は **同一 eval seed 上の最大値**で選ばれているので、
そのまま候補の実力として読むと best-of-N の選択バイアスが乗る。ここでは学習中に一度も
使っていない seed で、climb と各候補を **同一 seed / 同一相手** で評価しなおす。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "kaggle_replays" / "rl"), str(_ROOT),
           str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_field import parallel_collect_field  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
FIELD = [("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
         ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
         ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181)]


def wilson(k, n, z=1.96):
    if not n:
        return None
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True,
                    help="name=path,name=path,... 形式")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=31_415_926)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--tag", default="f1")
    args = ap.parse_args()

    deck = read_deck_csv_file(args.deck or str(_ROOT / "sample_submission" / "deck.csv"))
    opp_specs = [(str(WDIR / "policy_weights_{}.json".format(a)),
                  read_deck_csv_file(str(DECKDIR / a / "01.csv"))) for a, _ in FIELD]
    shares = [s for _, s in FIELD]

    out = {"games": args.games, "seed": args.seed, "policies": {}}
    for item in args.policies.split(","):
        name, path = item.split("=", 1)
        p = Path(path)
        if not p.is_absolute():
            p = WDIR / p.name
        t = time.perf_counter()
        _, w, v, err, per_opp = parallel_collect_field(
            str(p), opp_specs, shares, deck, args.games, args.seed,
            temperature=0.01, workers=args.workers)
        out["policies"][name] = {
            "path": str(p), "wins": w, "valid": v,
            "winrate": round(w / v, 4) if v else None, "ci95": wilson(w, v),
            "errors": err, "sec": round(time.perf_counter() - t, 1),
            "per_opponent": {FIELD[i][0]: per_opp[i] for i in range(len(FIELD))}
            if isinstance(per_opp, (list, tuple)) and len(per_opp) == len(FIELD)
            else per_opp}
        print(json.dumps({name: out["policies"][name]}, ensure_ascii=False), flush=True)

    (_HERE / "_p1915field_{}.json".format(args.tag)).write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {"winrate": v["winrate"], "ci95": v["ci95"],
                          "wins": v["wins"], "valid": v["valid"]}
                      for k, v in out["policies"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
