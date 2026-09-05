"""L2: ミラー戦(Alakazam 同型)での policy head-to-head。

local FIELD には Alakazam が 0% しか入っていないが、実ラダーでは 20.96% を占める最大シェア。
「local FIELD で勝った candidate が LB で落ちる」現象が、このミラー欠落で説明できるかを直接測る。

deck は両者 Plan A で固定し、**Policy だけ**を変える。先後は交互、seed は paired。

注意: Windows の spawn では `python -c` 経由だと worker が `__main__` を再 import できず
initializer を unpickle できない。必ずファイルとして実行すること。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

_S: dict = {}


def _init(pa: str, pb: str):
    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.learning.policy_model import PolicyModel
    for p in (pa, pb):
        if not PolicyModel(p).is_ready:
            raise RuntimeError("policy not ready: " + p)

    def mk(w):
        c = agents.load_config_copy("abl_5_full")
        c["policy_weights_path"] = w
        return agents.make_ml_policy_agent(c)
    _S["A"], _S["B"] = mk(pa), mk(pb)
    _S["runner"] = runner
    _S["deck"] = runner.load_deck(_SUB / "deck.csv")


def _one(task):
    gi, a_first = task
    rn, d = _S["runner"], _S["deck"]
    random.seed(5_500_000 + gi)
    t = time.perf_counter()
    try:
        r = (rn.play_game(_S["A"], _S["B"], list(d), list(d)) if a_first
             else rn.play_game(_S["B"], _S["A"], list(d), list(d)))
    except Exception as exc:                                    # noqa: BLE001
        return {"win_a": None, "err": type(exc).__name__}
    ai = 0 if a_first else 1
    w = getattr(r, "winner", None)
    return {"win_a": None if w is None else (1.0 if w == ai else 0.0),
            "ms": round((time.perf_counter() - t) * 1000, 1),
            "primary": str(getattr(r, "primary", None)
                           or getattr(r, "primary_win_condition", None)),
            "err": getattr(r, "error", None)}


def main() -> None:
    wd = _SUB / "ptcg_ai" / "learning"
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(wd / "policy_weights_alakazam_rl_p1915r1.json"),
                    help="評価したい候補 Policy")
    ap.add_argument("--b", default=str(wd / "policy_weights_alakazam_rl_climb.json"),
                    help="基準 Policy(climb)")
    ap.add_argument("--games", type=int, default=160)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tag", default="mirror")
    args = ap.parse_args()

    tasks = [(i, i % 2 == 0) for i in range(args.games)]
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init, initargs=(args.a, args.b)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    el = time.perf_counter() - t0

    v = [r["win_a"] for r in rows if r.get("win_a") is not None]
    a = np.array(v, float)
    rs = np.random.RandomState(0)
    m = a[rs.randint(0, len(a), (20000, len(a)))].mean(1)
    from collections import Counter
    rep = {"tag": args.tag, "a": args.a, "b": args.b, "games": args.games,
           "n_valid": len(v), "elapsed_s": round(el, 1),
           "a_winrate_vs_b": round(float(a.mean()), 4),
           "ci95": [round(float(np.percentile(m, 2.5)), 4),
                    round(float(np.percentile(m, 97.5)), 4)],
           "P_gt_0.5": round(float((m > 0.5).mean()), 3),
           "terminal_modes": dict(Counter(r.get("primary") for r in rows)),
           "errors": sum(1 for r in rows if r.get("err")),
           "mean_ms": round(float(np.mean([r["ms"] for r in rows if r.get("ms")])), 1)}
    (_HERE / "_p1919_{}.json".format(args.tag)).write_text(
        json.dumps({"summary": rep, "rows": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
