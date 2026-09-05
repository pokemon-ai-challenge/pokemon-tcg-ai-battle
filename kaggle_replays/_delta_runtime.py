"""Phase15 §25/§26: Action-Delta の推論コスト測定。

分けて測る:
  standalone : Delta のためだけに search_begin + 候補ごとの search_step を実行した場合
  reused     : production PIMC が**既に踏んでいる** search_step から delta を拾う場合の増分
               (= feature 抽出 + delta encoder 推論のみ)

production 予算は Phase12 で実測した 1 手あたり 1446.7ms を基準にする。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "value_net"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

import _entity_extract as EX  # noqa: E402
import delta_features as DF  # noqa: E402
import train_delta_q as TD  # noqa: E402
from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
PROD_BUDGET_MS = 1446.7            # Phase12 実測(1手あたり平均)

REC = []
_ctx = {"on": False, "me": 0, "n": 0}
OPTS = {"max_groups": 60, "m": 4}


def _measure(obs, policy_model):
    select, state = obs.select, obs.current
    me = state.yourIndex
    try:
        ps = policy_model.score_options(obs, None, None)
    except Exception:                              # noqa: BLE001
        return
    if not ps or len(ps) != len(select.option):
        return
    cand = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)[:8]
    if len(cand) < 2:
        return
    before = EX.summarize(state, me)
    rows = {"n_cands": len(cand)}

    # ---- standalone: determinization + 候補ごとの search_step + 特徴抽出 ----
    for tag, m in (("standalone_M1", 1), ("standalone_M4", OPTS["m"])):
        t0 = time.perf_counter()
        per_cand = []
        for _d in range(m):
            try:
                hs = search_adapter.to_search_begin_kwargs(
                    match_context.get_own_state(me), match_context.get_opponent_state(me),
                    obs, rng=random.Random(random.getrandbits(30)))
                root = P._begin(obs, hs)
            except Exception:                      # noqa: BLE001
                return
            try:
                for i in cand:
                    c0 = time.perf_counter()
                    try:
                        ch = P.cg_api.search_step(root.searchId, [i])
                    except ValueError:
                        continue
                    try:
                        stt = ch.observation.current
                        if stt is not None:
                            EX.summarize(stt, me)
                    finally:
                        try:
                            P.cg_api.search_release(ch.searchId)
                        except Exception:          # noqa: BLE001
                            pass
                    per_cand.append((time.perf_counter() - c0) * 1000)
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:                  # noqa: BLE001
                    pass
        try:
            P.cg_api.search_end()
        except Exception:                          # noqa: BLE001
            pass
        rows[tag + "_group_ms"] = (time.perf_counter() - t0) * 1000
        rows[tag + "_cand_ms"] = st.mean(per_cand) if per_cand else None

    # ---- reused: search_step は既存とみなし、抽出 + encoder 推論だけ ----
    afters = [[float(x) for x in before] for _ in range(OPTS["m"])]
    t0 = time.perf_counter()
    d1 = [DF.single(before, afters[0]) for _ in cand]
    rows["reuse_extract_ms"] = (time.perf_counter() - t0) * 1000
    model = TD.build_model("D1", 166, 65, 64)
    b = {"state": torch.zeros(1, 166), "opt": torch.zeros(1, len(cand), 65),
         "card": torch.zeros(1, len(cand), dtype=torch.long),
         "mask": torch.ones(1, len(cand), dtype=torch.bool),
         "delta": torch.tensor([d1], dtype=torch.float32)}
    with torch.no_grad():
        t0 = time.perf_counter()
        model(b)
        rows["encoder_ms"] = (time.perf_counter() - t0) * 1000
    REC.append(rows)
    _ctx["n"] += 1


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        if (_ctx["on"] and obs.current is not None and obs.current.yourIndex == _ctx["me"]
                and obs.select is not None and obs.select.type == SelectType.MAIN
                and obs.select.option and len(obs.select.option) >= 2
                and _ctx["n"] < OPTS["max_groups"]):
            try:
                _measure(obs, ml_policy_agent._get_model(config))
            except Exception as exc:               # noqa: BLE001
                print("[err]", exc, file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=60)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--out", default=str(_HERE / "value_net" / "phase15_runtime.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)
    OPTS["max_groups"] = args.groups
    _install()

    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
    climb = agents.make_ml_policy_agent(cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = ["mega_lucario_ex", "crustle", "dragapult_ex"]
    for gi in range(args.games):
        if _ctx["n"] >= args.groups:
            break
        arch = opps[gi % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        _ctx.update(on=True, me=0)
        runner.play_game(climb, opp, deck_c, deck_o)
        _ctx["on"] = False

    def band(n):
        return "2-4" if n <= 4 else "5-7" if n <= 7 else "8+"

    def stats(vals):
        if not vals:
            return None
        v = sorted(vals)
        return {"mean": round(st.mean(v), 3), "p50": round(v[len(v) // 2], 3),
                "p90": round(v[int(0.9 * (len(v) - 1))], 3), "n": len(v)}

    out = {"n_groups": len(REC), "prod_budget_ms_per_decision": PROD_BUDGET_MS,
           "m_dets": OPTS["m"], "overall": {}, "by_cand_band": {}}
    for k in ("standalone_M1_group_ms", "standalone_M4_group_ms", "standalone_M1_cand_ms",
              "standalone_M4_cand_ms", "reuse_extract_ms", "encoder_ms"):
        out["overall"][k] = stats([r[k] for r in REC if r.get(k) is not None])
    by = defaultdict(list)
    for r in REC:
        by[band(r["n_cands"])].append(r)
    for b, rs in sorted(by.items()):
        out["by_cand_band"][b] = {
            "groups": len(rs),
            "standalone_M1_group_ms": stats([r["standalone_M1_group_ms"] for r in rs]),
            "standalone_M4_group_ms": stats([r["standalone_M4_group_ms"] for r in rs]),
            "reuse_ms": stats([r["reuse_extract_ms"] + r["encoder_ms"] for r in rs])}
    for k in ("standalone_M1_group_ms", "standalone_M4_group_ms"):
        s = out["overall"][k]
        if s:
            out["overall"][k]["pct_of_budget"] = round(100 * s["mean"] / PROD_BUDGET_MS, 2)
    re_ = out["overall"]["reuse_extract_ms"]
    en = out["overall"]["encoder_ms"]
    if re_ and en:
        out["overall"]["reuse_total_ms"] = round(re_["mean"] + en["mean"], 3)
        out["overall"]["reuse_pct_of_budget"] = round(
            100 * (re_["mean"] + en["mean"]) / PROD_BUDGET_MS, 3)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
