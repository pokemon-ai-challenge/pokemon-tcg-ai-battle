"""climb(abl_5_full) の意思決定カバレッジ診断。

「探索がそもそも走っているか」を実ゲームで測る。仮説B/Dの前提確認であり、
top-k を増やす/ターン探索を足す前に **効く余地があるか** を先に確かめるためのもの。

測るもの(1 select = 1行):
  - select 種別(SelectType 名)/ maxCount / 選択肢数 / turn
  - pipeline の適用対象か(MAIN かつ maxCount==1 かつ自ターン)
  - lethal_simple が発火したか
  - pipeline.search が実際に決定化探索まで到達したか
    (= `_select_candidate_indices` が呼ばれたか。top1_shortcut で抜けた場合は未到達)
  - 探索候補数
  - select レイテンシ(ms)

production コードは import して呼ぶだけ(monkeypatch は本プロセス内のみ)。
出力: kaggle_replays/_diag_climb_coverage_results.json
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
from ptcg_ai.search import pipeline  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

# 収集バッファ(本プロセス内・エージェント側=climb のみ記録)
ROWS: list[dict] = []
_cur: dict = {}
_recording = False


def _sel_type_name(value) -> str:
    """SelectType を名前へ。cg の enum は int を返すことがあるので値からも引く。"""
    name = getattr(value, "name", None)
    if name:
        return name
    try:
        from cg.api import SelectType

        return SelectType(int(value)).name
    except Exception:  # noqa: BLE001
        return str(value)


def _install_probes() -> None:
    """production の関数を包んで観測点を作る(挙動は変えない)。"""
    orig_select_action = ml_policy_agent._select_action
    orig_try_lethal = ml_policy_agent._try_lethal
    orig_pipeline_search = pipeline.search
    orig_cand = pipeline._select_candidate_indices

    def select_action(obs, config=None):
        global _cur
        if not _recording:
            return orig_select_action(obs, config=config)
        sel = obs.select
        state = obs.current
        _cur = {
            "sel_type": _sel_type_name(sel.type) if sel is not None else None,
            "max_count": getattr(sel, "maxCount", None),
            "min_count": getattr(sel, "minCount", None),
            "n_options": len(sel.option) if sel is not None and sel.option else 0,
            "turn": getattr(state, "turn", None) if state is not None else None,
            "lethal": False,
            "pipeline_called": False,
            "pipeline_searched": False,
            "pipeline_returned": False,
            "n_candidates": None,
        }
        t0 = time.perf_counter()
        try:
            return orig_select_action(obs, config=config)
        finally:
            _cur["ms"] = (time.perf_counter() - t0) * 1000.0
            ROWS.append(_cur)

    def try_lethal(obs, config=None):
        out = orig_try_lethal(obs, config=config)
        if _recording and out is not None:
            _cur["lethal"] = True
        return out

    def pipeline_search(state, legal_actions, context):
        if _recording:
            _cur["pipeline_called"] = True
        out = orig_pipeline_search(state, legal_actions, context)
        if _recording and out is not None:
            _cur["pipeline_returned"] = True
        return out

    def cand(select, ranked, config, probs=None):
        out = orig_cand(select, ranked, config, probs)
        if _recording:
            _cur["pipeline_searched"] = True
            _cur["n_candidates"] = len(out)
        return out

    ml_policy_agent._select_action = select_action
    ml_policy_agent._try_lethal = try_lethal
    pipeline.search = pipeline_search
    pipeline._select_candidate_indices = cand


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--config", default="abl_5_full")
    ap.add_argument("--out", default=str(_HERE / "_diag_climb_coverage_results.json"))
    args = ap.parse_args()

    _install_probes()

    cfg_climb = agents.load_config_copy(args.config)
    cfg_climb["policy_weights_path"] = CLIMB_WEIGHTS
    climb = agents.make_ml_policy_agent(cfg_climb)

    cfg_opp = agents.load_config_copy("abl_5_full")
    cfg_opp["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_opp)

    deck_climb = runner.load_deck(_SUB / "deck.csv")
    deck_opp = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    global _recording
    wins = 0
    played = 0
    game_ms: list[float] = []
    t_start = time.perf_counter()

    for g in range(args.games):
        # climb を先手/後手交互に置き、climb 側の select だけ記録する。
        climb_is_p0 = (g % 2 == 0)
        gt0 = time.perf_counter()
        if climb_is_p0:
            _recording = True
            res = runner.play_game(climb, opp, deck_climb, deck_opp)
        else:
            _recording = True
            res = runner.play_game(opp, climb, deck_opp, deck_climb)
        _recording = False
        game_ms.append((time.perf_counter() - gt0) * 1000.0)
        if res.error:
            print(f"[warn] game {g}: {res.error}", file=sys.stderr)
            continue
        played += 1
        me = 0 if climb_is_p0 else 1
        if res.winner == me:
            wins += 1
        print(f"  game {g+1}/{args.games} winner={res.winner} climb={'p0' if climb_is_p0 else 'p1'}",
              file=sys.stderr)

    elapsed_min = (time.perf_counter() - t_start) / 60.0

    # ---- 集計 ----
    # 注: 記録は climb 側の select だけでなく、同一プロセスで動く相手側 select も
    # _select_action を通るため混ざる。両者とも ml_policy なので「パイプラインの
    # 構造的カバレッジ」を測る目的には影響しないが、勝率は climb 視点のみ。
    total = len(ROWS)
    by_type = Counter(r["sel_type"] for r in ROWS)
    eligible = [r for r in ROWS if r["sel_type"] == "MAIN" and r["max_count"] == 1]
    lethal_rows = [r for r in ROWS if r["lethal"]]
    called = [r for r in ROWS if r["pipeline_called"]]
    searched = [r for r in ROWS if r["pipeline_searched"]]
    returned = [r for r in ROWS if r["pipeline_returned"]]
    lat = [r["ms"] for r in ROWS if "ms" in r]

    out = {
        "agent": "climb_baseline",
        "config": args.config,
        "weights": Path(CLIMB_WEIGHTS).name,
        "deck": "sample_submission/deck.csv (Plan A alakazam)",
        "opponent": args.opponent,
        "games_requested": args.games,
        "games_played": played,
        "climb_wins": wins,
        "win_rate": (wins / played) if played else None,
        "elapsed_min": round(elapsed_min, 2),
        "games_per_min": round(played / elapsed_min, 3) if elapsed_min > 0 else None,
        "selects": {
            "total": total,
            "by_select_type": dict(by_type.most_common()),
            "eligible_main_max1": len(eligible),
            "eligible_frac": round(len(eligible) / total, 4) if total else None,
            "lethal_fired": len(lethal_rows),
            "lethal_frac_of_all": round(len(lethal_rows) / total, 4) if total else None,
            "pipeline_called": len(called),
            "pipeline_actually_searched": len(searched),
            "pipeline_returned_action": len(returned),
            "search_frac_of_all_selects": round(len(searched) / total, 4) if total else None,
            "search_frac_of_eligible": round(len(searched) / len(eligible), 4) if eligible else None,
            "top1_shortcut_or_bail_frac_of_called": (
                round(1 - len(searched) / len(called), 4) if called else None
            ),
            "mean_candidates_when_searched": (
                round(statistics.mean([r["n_candidates"] for r in searched if r["n_candidates"]]), 3)
                if searched else None
            ),
        },
        "latency_ms": {
            "mean": round(statistics.mean(lat), 2) if lat else None,
            "p50": round(_pct(lat, 0.50), 2),
            "p90": round(_pct(lat, 0.90), 2),
            "p95": round(_pct(lat, 0.95), 2),
            "p99": round(_pct(lat, 0.99), 2),
            "max": round(max(lat), 2) if lat else None,
        },
        "by_type_detail": {
            t: {
                "n": sum(1 for r in ROWS if r["sel_type"] == t),
                "frac_of_all": round(sum(1 for r in ROWS if r["sel_type"] == t) / total, 4),
                "searched": sum(1 for r in ROWS if r["sel_type"] == t and r["pipeline_searched"]),
                "mean_n_options": round(
                    statistics.mean([r["n_options"] for r in ROWS if r["sel_type"] == t]), 2
                ),
            }
            for t in by_type
        },
        "n_options_dist": {
            "mean_all": round(statistics.mean([r["n_options"] for r in ROWS]), 2) if ROWS else None,
            "mean_eligible": (
                round(statistics.mean([r["n_options"] for r in eligible]), 2) if eligible else None
            ),
        },
    }

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
