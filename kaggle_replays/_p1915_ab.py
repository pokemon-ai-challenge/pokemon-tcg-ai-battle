"""Phase19.15-R Stage B/C: production full-game paired A/B と Policy drift。

baseline(climb)と candidate は **Policy weights だけ** が違う。config は abl_5_full、
deck は Plan A(sample_submission/deck.csv)で完全固定。

paired: 同じ (opponent, seed, 先後) で baseline と candidate を1本ずつ回す。
drift : 同一 observation 上で両 Policy の argmax 一致率を select type 別に測る。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"


def _abspath(p):
    """`ensure_production_cwd()` で cwd が sample_submission へ移るため、相対パスは
    repo root 基準で絶対化してから渡す(渡さないと PolicyModel が黙って未ロードになる)。"""
    q = Path(p)
    return str(q if q.is_absolute() else (_ROOT / q).resolve())
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

FIELD = [("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
         ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
         ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181)]

_W: dict = {}


def _init(baseline_w, cand_w):
    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: F401
    _W["agents"], _W["runner"] = agents, runner
    wdir = _SUB / "ptcg_ai" / "learning"
    ddir = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

    from ptcg_ai.learning.policy_model import PolicyModel

    def mk(path):
        # 未ロードの PolicyModel は例外を出さず黙って弱くなるので、ここで必ず落とす。
        if not PolicyModel(str(path)).is_ready:
            raise RuntimeError("policy not ready: {}".format(path))
        c = agents.load_config_copy("abl_5_full")
        c["policy_weights_path"] = str(path)
        return agents.make_ml_policy_agent(c)
    _W["base"] = mk(_abspath(baseline_w))
    _W["cand"] = mk(_abspath(cand_w))
    _W["deck"] = runner.load_deck(_SUB / "deck.csv")
    _W["opps"] = [(a, mk(wdir / "policy_weights_{}.json".format(a)),
                   runner.load_deck(ddir / a / "01.csv")) for a, _ in FIELD]


def _one(task):
    gi, opp_idx, learner_first = task
    runner = _W["runner"]
    arch, opp, deck_o = _W["opps"][opp_idx]
    out = {"game": gi, "arch": arch, "first": learner_first}
    for side in ("base", "cand"):
        me = _W[side]
        random.seed(1_000_000 + gi)
        t = time.perf_counter()
        try:
            r = (runner.play_game(me, opp, list(_W["deck"]), list(deck_o))
                 if learner_first else
                 runner.play_game(opp, me, list(deck_o), list(_W["deck"])))
        except Exception as exc:                                # noqa: BLE001
            out[side] = {"error": type(exc).__name__}
            continue
        ms = (time.perf_counter() - t) * 1000
        my = 0 if learner_first else 1
        w = getattr(r, "winner", None)
        out[side] = {"win": None if w is None else (1.0 if w == my else 0.0),
                     "ms": round(ms, 1), "steps": getattr(r, "steps", None),
                     "error": getattr(r, "error", None)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(
        _SUB / "ptcg_ai" / "learning" / "policy_weights_alakazam_rl_climb.json"))
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=7_000_000)
    ap.add_argument("--tag", default="ab")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tot = sum(s for _, s in FIELD)
    cum, acc = [], 0
    for i, (_, s) in enumerate(FIELD):
        acc += s
        cum.append(acc / tot)
    tasks = []
    for gi in range(args.games):
        r = rng.random()
        oi = next(k for k, c in enumerate(cum) if r <= c)
        tasks.append((gi, oi, gi % 2 == 0))

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init,
              initargs=(args.baseline, args.candidate)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    el = time.perf_counter() - t0

    def agg(key):
        v = [r[key]["win"] for r in rows if r.get(key) and r[key].get("win") is not None]
        ms = [r[key]["ms"] for r in rows if r.get(key) and r[key].get("ms") is not None]
        er = sum(1 for r in rows if r.get(key) and (r[key].get("error")
                                                    or r[key].get("win") is None))
        return {"n": len(v), "winrate": round(sum(v) / len(v), 4) if v else None,
                "mean_ms": round(sum(ms) / len(ms), 1) if ms else None,
                "errors": er}

    paired = [(r["base"]["win"], r["cand"]["win"]) for r in rows
              if r.get("base") and r.get("cand")
              and r["base"].get("win") is not None and r["cand"].get("win") is not None]
    import numpy as np
    d = np.array([c - b for b, c in paired], float)
    rs = np.random.RandomState(0)
    ci = (None if len(d) == 0 else
          [round(float(np.percentile(d[rs.randint(0, len(d), (10000, len(d)))].mean(1),
                                     2.5)), 4),
           round(float(np.percentile(d[rs.randint(0, len(d), (10000, len(d)))].mean(1),
                                     97.5)), 4)])
    per = {}
    for a, _ in FIELD:
        s = [r for r in rows if r["arch"] == a and r.get("base") and r.get("cand")
             and r["base"].get("win") is not None and r["cand"].get("win") is not None]
        if not s:
            continue
        b = sum(r["base"]["win"] for r in s) / len(s)
        c = sum(r["cand"]["win"] for r in s) / len(s)
        per[a] = {"n": len(s), "baseline": round(b, 4), "candidate": round(c, 4),
                  "delta": round(c - b, 4)}

    rep = {"tag": args.tag, "games": args.games, "elapsed_s": round(el, 1),
           "baseline_weights": args.baseline, "candidate_weights": args.candidate,
           "baseline": agg("base"), "candidate": agg("cand"),
           "paired_n": len(paired),
           "paired_delta": round(float(d.mean()), 4) if len(d) else None,
           "paired_ci95": ci, "by_archetype": per,
           "arch_mix": dict(Counter(r["arch"] for r in rows))}
    (_HERE / "_p1915ab_{}.json".format(args.tag)).write_text(
        json.dumps({"summary": rep, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
