"""LOOP1: deck-only paired A/B（Policy は climb 固定、abl_5_full 固定）。

Codex レビュー(#7)の指摘どおり、変更するのは **自陣の deck だけ**。同じ相手・同じ seed・
先後入替で D0(現行 Plan A) と D1(候補) を1本ずつ回す paired 比較。

終局内訳(deckout / prize_out / no_pokemon)と、Sacred Ash が手札にあったのに
使われなかった回数(#1 の検証)も併記する。
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

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

FIELD = [("mega_lucario_ex", 1257), ("archaludon_ex", 1078), ("crustle", 737),
         ("dragapult_ex", 625), ("marnie_grimmsnarl_ex", 591),
         ("rocket_mewtwo_ex", 247), ("shirona_garchomp_ex", 181)]
SACRED_ASH = 1129
_W: dict = {}


def _abs(p):
    q = Path(p)
    return q if q.is_absolute() else (_ROOT / q).resolve()


def _init(deck0, deck1, opps_wanted, mirror_deck):
    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.learning import encoder as ENC
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.ml_policy import ml_policy_agent as MA
    _W["runner"], _W["ENC"], _W["MA"] = runner, ENC, MA
    wdir = _SUB / "ptcg_ai" / "learning"
    ddir = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
    climb = str(wdir / "policy_weights_alakazam_rl_climb.json")
    if not PolicyModel(climb).is_ready:
        raise RuntimeError("climb policy not ready")

    def mk(w):
        c = agents.load_config_copy("abl_5_full")
        c["policy_weights_path"] = str(w)
        return agents.make_ml_policy_agent(c)
    _W["cfg_climb"] = climb
    _W["me"] = mk(climb)
    _W["decks"] = {"D0": runner.load_deck(_abs(deck0)),
                   "D1": runner.load_deck(_abs(deck1))}
    _W["opps"] = {}
    for a in opps_wanted:
        if a == "MIRROR":
            _W["opps"][a] = (mk(climb), runner.load_deck(_abs(mirror_deck)))
        else:
            _W["opps"][a] = (mk(wdir / "policy_weights_{}.json".format(a)),
                             runner.load_deck(ddir / a / "01.csv"))


def _probe_factory():
    """Sacred Ash が選択肢に出た回数 / 実際に打った回数を数える wrapper。"""
    inner = _W["me"]
    ENC = _W["ENC"]
    stat = Counter()

    def probe(obs):
        act = inner(obs)
        sel, st = obs.select, obs.current
        if sel is not None and st is not None and sel.option:
            try:
                ids = [ENC._resolve_card_id(o, st) for o in sel.option]
            except Exception:                                   # noqa: BLE001
                return act
            if SACRED_ASH in ids:
                stat["ash_offered"] += 1
                if act and act[0] < len(ids) and ids[act[0]] == SACRED_ASH:
                    stat["ash_played"] += 1
        return act
    return probe, stat


def _one(task):
    gi, arch, first = task
    runner = _W["runner"]
    opp, deck_o = _W["opps"][arch]
    out = {"game": gi, "arch": arch, "first": first}
    for name in ("D0", "D1"):
        deck_me = list(_W["decks"][name])
        probe, stat = _probe_factory()
        random.seed(4_000_000 + gi)
        t = time.perf_counter()
        try:
            r = (runner.play_game(probe, opp, deck_me, list(deck_o)) if first
                 else runner.play_game(opp, probe, list(deck_o), deck_me))
        except Exception as exc:                                # noqa: BLE001
            out[name] = {"error": type(exc).__name__}
            continue
        my = 0 if first else 1
        w = getattr(r, "winner", None)
        prim = getattr(r, "primary", None) or getattr(r, "primary_win_condition", None)
        out[name] = {"win": None if w is None else (1.0 if w == my else 0.0),
                     "ms": round((time.perf_counter() - t) * 1000, 1),
                     "primary": str(prim), "lost_by": (str(prim) if w is not None
                                                       and w != my else None),
                     "ash_offered": stat["ash_offered"],
                     "ash_played": stat["ash_played"],
                     "error": getattr(r, "error", None)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck0", default="sample_submission/deck.csv")
    ap.add_argument("--deck1", required=True)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=9_100_000)
    ap.add_argument("--opponents", default="crustle",
                    help="'FIELD'=メタshare抽選 / 'MIRROR' / カンマ区切りのアーキ名")
    ap.add_argument("--tag", default="d1")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.opponents == "FIELD":
        tot = sum(s for _, s in FIELD)
        cum, acc = [], 0
        for _, s in FIELD:
            acc += s
            cum.append(acc / tot)
        picks = []
        for _ in range(args.games):
            r = rng.random()
            picks.append(FIELD[next(k for k, c in enumerate(cum) if r <= c)][0])
    else:
        names = [x for x in args.opponents.split(",") if x]
        picks = [names[i % len(names)] for i in range(args.games)]
    tasks = [(i, picks[i], i % 2 == 0) for i in range(args.games)]
    wanted = sorted(set(picks))

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init,
              initargs=(args.deck0, args.deck1, wanted, args.deck0)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    el = time.perf_counter() - t0

    def agg(k):
        v = [r[k]["win"] for r in rows if r.get(k) and r[k].get("win") is not None]
        lm = Counter(r[k]["lost_by"] for r in rows
                     if r.get(k) and r[k].get("lost_by"))
        ao = sum(r[k].get("ash_offered", 0) for r in rows if r.get(k))
        apl = sum(r[k].get("ash_played", 0) for r in rows if r.get(k))
        return {"n": len(v), "winrate": round(sum(v) / len(v), 4) if v else None,
                "loss_modes": dict(lm), "ash_offered": ao, "ash_played": apl,
                "ash_play_rate": round(apl / ao, 3) if ao else None,
                "mean_ms": round(sum(r[k]["ms"] for r in rows
                                     if r.get(k) and r[k].get("ms")) / max(1, len(v)), 1),
                "errors": sum(1 for r in rows if r.get(k) and r[k].get("error"))}

    paired = [(r["D0"]["win"], r["D1"]["win"]) for r in rows
              if r.get("D0") and r.get("D1")
              and r["D0"].get("win") is not None and r["D1"].get("win") is not None]
    d = np.array([b - a for a, b in paired], float)
    rs = np.random.RandomState(0)
    m = d[rs.randint(0, len(d), (20000, len(d)))].mean(1) if len(d) else np.array([0.0])
    per = {}
    for a in wanted:
        s = [r for r in rows if r["arch"] == a and r.get("D0") and r.get("D1")
             and r["D0"].get("win") is not None and r["D1"].get("win") is not None]
        if s:
            b0 = sum(r["D0"]["win"] for r in s) / len(s)
            b1 = sum(r["D1"]["win"] for r in s) / len(s)
            per[a] = {"n": len(s), "D0": round(b0, 4), "D1": round(b1, 4),
                      "delta": round(b1 - b0, 4)}

    rep = {"tag": args.tag, "games": args.games, "opponents": args.opponents,
           "elapsed_s": round(el, 1), "deck0": args.deck0, "deck1": args.deck1,
           "D0": agg("D0"), "D1": agg("D1"), "paired_n": len(paired),
           "paired_delta": round(float(d.mean()), 4) if len(d) else None,
           "paired_ci95": [round(float(np.percentile(m, 2.5)), 4),
                           round(float(np.percentile(m, 97.5)), 4)],
           "P_gt_0": round(float((m > 0).mean()), 3), "by_archetype": per}
    (_HERE / "_p1918ab_{}.json".format(args.tag)).write_text(
        json.dumps({"summary": rep, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
