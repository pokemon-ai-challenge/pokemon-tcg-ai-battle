"""config-only A/B: abl_5_full(baseline) vs abl_5_full_routing(候補)。

deck / base Policy / search 設定は同一。差分は `routing` ブロックだけ
(相手アーキ予測が閾値以上なら専用重みへ差し替える)。

実ラダーの share で相手を抽選し、誤ルーティング(非crustleで発火)も監視する。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

LADDER = [("marnie_grimmsnarl_ex", 19.88), ("mega_lucario_ex", 14.86),
          ("crustle", 10.18), ("archaludon_ex", 9.63), ("dragapult_ex", 6.38),
          ("shirona_garchomp_ex", 2.44), ("rocket_mewtwo_ex", 2.31)]
MIRROR = 20.96
_S: dict = {}


def _init(cfg_a: str, cfg_b: str):
    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.ml_policy import ml_policy_agent as MA
    _S["runner"], _S["MA"] = runner, MA
    wdir = _SUB / "ptcg_ai" / "learning"
    ddir = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
    climb = str(wdir / "policy_weights_alakazam_rl_climb.json")

    def mk(cfg_name, weights=None):
        c = agents.load_config_copy(cfg_name)
        if weights:
            c["policy_weights_path"] = weights
        return agents.make_ml_policy_agent(c), c
    _S["A"] = mk(cfg_a, climb)
    _S["B"] = mk(cfg_b, climb)
    _S["deck"] = runner.load_deck(_SUB / "deck.csv")
    _S["opps"] = {a: (mk("abl_5_full", str(wdir / "policy_weights_{}.json".format(a)))[0],
                      runner.load_deck(ddir / a / "01.csv")) for a, _ in LADDER}
    _S["opps"]["MIRROR"] = (mk("abl_5_full", climb)[0], list(_S["deck"]))


def _one(task):
    gi, arch, first = task
    rn = _S["runner"]
    MA = _S["MA"]
    opp, deck_o = _S["opps"][arch]
    out = {"game": gi, "arch": arch, "first": first}
    for side in ("A", "B"):
        me, _cfg = _S[side]
        routed = Counter()
        orig = MA._compute_route

        def spy(obs, config, _o=orig, _r=routed):
            p = _o(obs, config)
            _r["fired" if p else "base"] += 1
            return p
        MA._compute_route = spy
        random.seed(9_900_000 + gi)
        t = time.perf_counter()
        try:
            r = (rn.play_game(me, opp, list(_S["deck"]), list(deck_o)) if first
                 else rn.play_game(opp, me, list(deck_o), list(_S["deck"])))
        except Exception as exc:                               # noqa: BLE001
            MA._compute_route = orig
            out[side] = {"error": type(exc).__name__}
            continue
        finally:
            MA._compute_route = orig
        my = 0 if first else 1
        w = getattr(r, "winner", None)
        tot = routed["fired"] + routed["base"]
        out[side] = {"win": None if w is None else (1.0 if w == my else 0.0),
                     "primary": str(getattr(r, "primary", None)
                                    or getattr(r, "primary_win_condition", None)),
                     "ms": round((time.perf_counter() - t) * 1000, 1),
                     "route_fire_rate": round(routed["fired"] / tot, 4) if tot else None,
                     "decisions": tot}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-a", default="abl_5_full")
    ap.add_argument("--cfg-b", default="abl_5_full_routing")
    ap.add_argument("--games", type=int, default=420)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=13_579_246)
    ap.add_argument("--tag", default="routing")
    args = ap.parse_args()
    names = [a for a, _ in LADDER] + ["MIRROR"]
    wts = [s for _, s in LADDER] + [MIRROR]
    tot = sum(wts)
    cum, acc = [], 0.0
    for s in wts:
        acc += s
        cum.append(acc / tot)
    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.games):
        r = rng.random()
        tasks.append((i, names[next(k for k, c in enumerate(cum) if r <= c)], i % 2 == 0))

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init,
              initargs=(args.cfg_a, args.cfg_b)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    el = time.perf_counter() - t0

    def agg(k):
        v = [r[k]["win"] for r in rows if r.get(k) and r[k].get("win") is not None]
        fr = [r[k]["route_fire_rate"] for r in rows
              if r.get(k) and r[k].get("route_fire_rate") is not None]
        return {"n": len(v), "winrate": round(sum(v) / len(v), 4) if v else None,
                "loss_modes": dict(Counter(r[k]["primary"] for r in rows
                                           if r.get(k) and r[k].get("win") == 0.0)),
                "mean_route_fire_rate": round(float(np.mean(fr)), 4) if fr else None,
                "mean_ms": round(float(np.mean([r[k]["ms"] for r in rows
                                                if r.get(k) and r[k].get("ms")])), 1),
                "errors": sum(1 for r in rows if r.get(k) and r[k].get("error"))}

    paired = [(r["A"]["win"], r["B"]["win"]) for r in rows
              if r.get("A") and r.get("B") and r["A"].get("win") is not None
              and r["B"].get("win") is not None]
    d = np.array([b - a for a, b in paired], float)
    rs = np.random.RandomState(0)
    m = d[rs.randint(0, len(d), (20000, len(d)))].mean(1)
    per = {}
    for a in names:
        s = [r for r in rows if r["arch"] == a and r.get("A") and r.get("B")
             and r["A"].get("win") is not None and r["B"].get("win") is not None]
        if s:
            per[a] = {"n": len(s),
                      "A": round(sum(x["A"]["win"] for x in s) / len(s), 4),
                      "B": round(sum(x["B"]["win"] for x in s) / len(s), 4),
                      "delta": round(sum(x["B"]["win"] - x["A"]["win"] for x in s) / len(s), 4),
                      "B_fire": round(float(np.mean([x["B"]["route_fire_rate"] for x in s
                                                     if x["B"].get("route_fire_rate")
                                                     is not None])), 4) if s else None}
    rep = {"tag": args.tag, "cfg_a": args.cfg_a, "cfg_b": args.cfg_b,
           "games": args.games, "elapsed_s": round(el, 1),
           "A": agg("A"), "B": agg("B"), "paired_n": len(paired),
           "delta": round(float(d.mean()), 4),
           "ci95": [round(float(np.percentile(m, 2.5)), 4),
                    round(float(np.percentile(m, 97.5)), 4)],
           "P_gt_0": round(float((m > 0).mean()), 3), "by_archetype": per}
    (_HERE / "_p1923_{}.json".format(args.tag)).write_text(
        json.dumps({"summary": rep, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
