"""Phase5.1: 本番時間予算の再現と select 数分布の測定(production 非改変)。

production の `_dynamic_pipeline_time_limit_ms` は次の式(実コードのまま):

    base = pipeline_config["time_limit_ms"]                      # abl_5_full なら 400
    if not time_budget or _match_start_perf is None: return base # ← 400ms fallback
    elapsed_ms       = (perf_counter() - _match_start_perf) * 1000
    remaining_ms     = total_ms - elapsed_ms
    if remaining_ms <= 0: return min_ms
    remaining_selects = max(1, assumed_total_selects - _selects_seen)
    per_move         = remaining_ms / remaining_selects
    return max(min_ms, min(max_ms, per_move))                    # ← max_ms=2000 が効く

ローカル harness は `obs.select is None`(デッキ選択)経路を通らないため `_match_start_perf` が
None のままで **常に 400ms fallback** になる。本スクリプトは

  1. 1試合1エージェントあたりの実 select 数分布を測る(= assumed_total_selects=400 の妥当性)
  2. 各 select で production が実際に割り当てる budget を、同じ式で再現して記録する
  3. `_replicate_budget()` が production 関数と数値一致することを確認する(パリティ)

production コードは import して呼ぶだけ。出力: _diag_budget_parity_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")


def replicate_budget(time_budget: dict, base_ms: float,
                     elapsed_ms: float, selects_seen: int) -> dict:
    """production と同じ式で budget を再現し、途中の値も返す(報告用に分解する)。

    `raw` = clamp 前 / `clamped` = min/max 適用後。production は clamped を pipeline に渡す。
    """
    total_ms = float(time_budget["total_ms"])
    min_ms = float(time_budget.get("min_ms", 50))
    max_ms = float(time_budget.get("max_ms", 2000))
    assumed_total = int(time_budget.get("assumed_total_selects", 400))
    remaining_ms = total_ms - elapsed_ms
    if remaining_ms <= 0:
        return {"raw": None, "clamped": min_ms, "remaining_ms": remaining_ms,
                "remaining_selects": None, "exhausted": True}
    remaining_selects = max(1, assumed_total - selects_seen)
    raw = remaining_ms / remaining_selects
    return {"raw": raw, "clamped": max(min_ms, min(max_ms, raw)),
            "remaining_ms": remaining_ms, "remaining_selects": remaining_selects,
            "exhausted": False}


ROWS: list[dict] = []          # select ごと
GAMES: list[dict] = []         # 試合ごと
_ctx = {"game": -1, "me": 0, "recording": False,
        "start": 0.0, "n_selects": 0}


def _install(time_budget: dict, base_ms: float) -> None:
    """自陣営の select ごとに、production が割り当てるはずの budget を再現記録する。"""
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not (_ctx["recording"] and obs.current is not None
                and obs.current.yourIndex == _ctx["me"]):
            return orig(obs, config=config)
        idx = _ctx["n_selects"]
        elapsed_ms = (time.perf_counter() - _ctx["start"]) * 1000.0
        b = replicate_budget(time_budget, base_ms, elapsed_ms, idx)
        t0 = time.perf_counter()
        try:
            return orig(obs, config=config)
        finally:
            consumed = (time.perf_counter() - t0) * 1000.0
            _ctx["n_selects"] += 1
            ROWS.append({
                "game": _ctx["game"], "select_index": idx,
                "elapsed_ms": round(elapsed_ms, 1),
                "remaining_ms": round(b["remaining_ms"], 1),
                "remaining_selects": b["remaining_selects"],
                "raw_budget_ms": round(b["raw"], 1) if b["raw"] is not None else None,
                "assigned_budget_ms": round(b["clamped"], 1),
                "consumed_ms": round(consumed, 1),
                "n_options": len(obs.select.option) if obs.select and obs.select.option else 0,
            })

    ml_policy_agent._select_action = select_action


def _pct(v, q):
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--config", default="climb_v15_leaf_a05")
    ap.add_argument("--out", default="_diag_budget_parity_results.json")
    args = ap.parse_args()

    cfg = agents.load_config_copy(args.config)
    cfg["policy_weights_path"] = CLIMB
    pl = cfg["pipeline"]
    time_budget = pl["time_budget"]
    base_ms = float(pl.get("time_limit_ms", 400))

    # --- パリティ: 再現式が production 関数と一致するか(疑似時刻を注入して比較) ---
    parity = []
    for elapsed_ms, seen in ((0.0, 0), (25000.0, 60), (100000.0, 150),
                             (500000.0, 380), (600000.0, 399)):
        ml_policy_agent._match_start_perf = time.perf_counter() - elapsed_ms / 1000.0
        ml_policy_agent._selects_seen = seen
        prod = ml_policy_agent._dynamic_pipeline_time_limit_ms(pl)
        mine = replicate_budget(time_budget, base_ms, elapsed_ms, seen)["clamped"]
        parity.append({"elapsed_ms": elapsed_ms, "selects_seen": seen,
                       "production_ms": round(prod, 1), "replica_ms": round(mine, 1),
                       "match": abs(prod - mine) < 1.0})
    # fallback 条件(match_start 未設定 = ローカル harness の現状)
    ml_policy_agent._match_start_perf = None
    ml_policy_agent._selects_seen = 0
    fallback_ms = ml_policy_agent._dynamic_pipeline_time_limit_ms(pl)

    _install(time_budget, base_ms)
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("climb_baseline")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    for g in range(args.games):
        p0 = (g % 2 == 0)
        _ctx.update(game=g, me=0 if p0 else 1, recording=True,
                    start=time.perf_counter(), n_selects=0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        GAMES.append({"game": g, "n_selects": _ctx["n_selects"],
                      "wall_ms": round((time.perf_counter() - _ctx["start"]) * 1000, 1)})
        print(f"  game {g+1}/{args.games} selects={_ctx['n_selects']}",
              file=sys.stderr, flush=True)

    sel = [x["n_selects"] for x in GAMES]
    assigned = [r["assigned_budget_ms"] for r in ROWS]
    raw = [r["raw_budget_ms"] for r in ROWS if r["raw_budget_ms"] is not None]
    consumed = [r["consumed_ms"] for r in ROWS]
    n = len(assigned)

    out = {
        "config": args.config,
        "time_budget": time_budget,
        "base_time_limit_ms": base_ms,
        "local_fallback_ms_when_match_start_unset": fallback_ms,
        "parity_check": parity,
        "parity_all_match": all(p["match"] for p in parity),
        "games": args.games,
        "select_count_per_agent_per_game": {
            "mean": round(statistics.mean(sel), 2), "median": statistics.median(sel),
            "p10": _pct(sel, 0.10), "p25": _pct(sel, 0.25),
            "p75": _pct(sel, 0.75), "p90": _pct(sel, 0.90),
            "min": min(sel), "max": max(sel),
            "assumed_total_selects": time_budget["assumed_total_selects"],
        },
        "assigned_budget_ms": {
            "mean": round(statistics.mean(assigned), 1), "median": _pct(assigned, 0.5),
            "p10": _pct(assigned, 0.10), "p25": _pct(assigned, 0.25),
            "p75": _pct(assigned, 0.75), "p90": _pct(assigned, 0.90),
            "max": round(max(assigned), 1),
            "frac_le_400": round(sum(1 for x in assigned if x <= 400) / n, 4),
            "frac_ge_1000": round(sum(1 for x in assigned if x >= 1000) / n, 4),
            "frac_ge_1350": round(sum(1 for x in assigned if x >= 1350) / n, 4),
            "frac_at_2000_cap": round(sum(1 for x in assigned if x >= 1999.9) / n, 4),
            "frac_ge_6000": round(sum(1 for x in assigned if x >= 6000) / n, 4),
        },
        "raw_budget_ms": {
            "mean": round(statistics.mean(raw), 1), "max": round(max(raw), 1),
            "frac_above_max_cap": round(sum(1 for x in raw if x > time_budget["max_ms"]) / len(raw), 4),
        },
        "consumed_ms_under_local_400ms": {
            "mean": round(statistics.mean(consumed), 1), "p90": _pct(consumed, 0.90),
            "max": round(max(consumed), 1),
        },
        "budget_if_divided_by_actual_selects": {
            "total_540000_over_mean_selects": round(540000 / statistics.mean(sel), 1),
            "total_600000_over_mean_selects": round(600000 / statistics.mean(sel), 1),
            "note": "max_ms のクランプ前の理論値。実際は max_ms=2000 で頭打ち。",
        },
        "n_select_rows": n,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "parity_check"},
                     ensure_ascii=False, indent=2))
    print("parity:", json.dumps(parity, ensure_ascii=False))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    out["rows"] = ROWS
    out["games_detail"] = GAMES
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
