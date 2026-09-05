"""Phase9A: MAIN counterfactual 教師の K 感度診断(教師ゲート)。

Q0 は既に教師自己一致 0.652 の 97%(0.6332)へ到達しており、**物差しが飽和**している。
モデルを大きくする前に「K を増やせば教師の再現性が上がるのか」を確かめる。

設計:
  - 各 decision group で **K=24 まで一度だけ**生成し、先頭を切り出して K=6/12/24 を
    **入れ子**で比較する(K ごとに別々の rollout を引くと K 差と乱数差が混ざる)。
  - 自己一致は **重複しない** A/B 集合で測る(K=24 なら A=先頭12 / B=次の12)。
  - subset 別(信頼度・候補数・ターン帯・アーキタイプ)にも出す。

出力: _main_teacher_k_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

GROUPS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0, "opp": ""}
OPTS = {"every": 2, "Kmax": 24, "max_cands": 6, "time_ms": 90000, "max_per_game": 4}
K_LEVELS = (6, 12, 24)


def _pairwise(a, b):
    tot = ok = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if b[i] == b[j]:
                continue
            tot += 1
            d = a[i] - a[j]
            ok += 0.5 if d == 0 else (1.0 if d * (b[i] - b[j]) > 0 else 0.0)
    return (ok / tot) if tot else None


def _rank(v):
    n = len(v)
    o = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[o[j + 1]] == v[o[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[o[k]] = avg
        i = j + 1
    return r


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return (num / (da * db)) if da > 0 and db > 0 else None


def _kendall(a, b):
    con = dis = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    return (con - dis) / (con + dis) if (con + dis) else None


def _self_at_k(vals: list[list[float]], K: int):
    """先頭 K サンプルを A/B(重複なし)へ割って自己一致を測る。"""
    if min(len(v) for v in vals) < K:
        return None
    h = K // 2
    A = [statistics.mean(v[:h]) for v in vals]
    B = [statistics.mean(v[h:K]) for v in vals]          # A と重複しない
    allv = [statistics.mean(v[:K]) for v in vals]
    std = statistics.mean([statistics.pstdev(v[:K]) for v in vals])
    spread = max(allv) - min(allv)
    srt = sorted(allv, reverse=True)
    return {
        "pairwise": _pairwise(A, B),
        "spearman": _spearman(A, B),
        "kendall": _kendall(A, B),
        "top1_agree": 1.0 if A.index(max(A)) == B.index(max(B)) else 0.0,
        "top2_agree": 1.0 if set(sorted(range(len(A)), key=lambda i: A[i], reverse=True)[:2])
        == set(sorted(range(len(B)), key=lambda i: B[i], reverse=True)[:2]) else 0.0,
        "spread": spread, "std": std,
        "spread_over_std": (spread / std) if std > 0 else None,
        "top2_margin": (srt[0] - srt[1]) if len(srt) > 1 else 0.0,
    }


def _probe(obs, model):
    select = obs.select
    state = obs.current
    me = state.yourIndex
    try:
        scores = model.score_options(obs, None, None)
    except Exception:  # noqa: BLE001
        return None
    if not scores or len(scores) != len(select.option):
        return None
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cand = ranked[: OPTS["max_cands"]]
    if len(cand) < 2:
        return None

    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0
    vals: dict[int, list[float]] = {i: [] for i in cand}

    t0 = time.perf_counter()
    for _ in range(OPTS["Kmax"]):
        if time.perf_counter() > deadline:
            break
        hs = factory()
        if hs is None:
            continue
        try:
            root = P._begin(obs, hs)
        except Exception:  # noqa: BLE001
            continue
        try:
            for i in cand:
                if time.perf_counter() > deadline:
                    break
                try:
                    child = P.cg_api.search_step(root.searchId, [i])
                except ValueError:
                    continue
                try:
                    v = P._rollout_and_eval(
                        child, me, {"opponent_depth": 1, "max_rollout_steps": 40},
                        deadline, ev, model)
                    if v is not None:
                        vals[i].append(v)
                finally:
                    try:
                        P.cg_api.search_release(child.searchId)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            try:
                P.cg_api.search_release(root.searchId)
            except Exception:  # noqa: BLE001
                pass
    try:
        P.cg_api.search_end()
    except Exception:  # noqa: BLE001
        pass

    kmin = min((len(v) for v in vals.values()), default=0)
    if kmin < 6:
        return None
    series = [vals[i][:kmin] for i in cand]
    rec = {"game": _ctx["game"], "turn": int(getattr(state, "turn", 0) or 0),
           "opponent": _ctx["opp"], "n_cands": len(cand), "k_achieved": kmin,
           "gen_sec": round(time.perf_counter() - t0, 1), "by_k": {}}
    for K in K_LEVELS:
        r = _self_at_k(series, K)
        if r:
            rec["by_k"][str(K)] = r
    return rec if rec["by_k"] else None


def _install():
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.MAIN and sel.maxCount == 1
                and len(sel.option) >= 2):
            _ctx["seen"] += 1
            if (_ctx["seen"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    r = _probe(obs, ml_policy_agent._get_model(config))
                    if r:
                        GROUPS.append(r)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def _agg(rows, K, key):
    v = [r["by_k"][str(K)][key] for r in rows
         if str(K) in r["by_k"] and r["by_k"][str(K)].get(key) is not None]
    return round(statistics.mean(v), 4) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex")
    ap.add_argument("--out", default="_main_teacher_k_results.json")
    args = ap.parse_args()

    _install()
    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for g in range(args.games):
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{arch}.json")
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (g % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, probed=0, opp=arch)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        print(f"  game {g+1}/{args.games} vs {arch} groups={len(GROUPS)}",
              file=sys.stderr, flush=True)

    full = [r for r in GROUPS if "24" in r["by_k"]]
    out = {"note": "K は入れ子(K=24を一度生成し先頭を切り出す)。A/Bは重複しない集合。",
           "budget_regime": "research_accuracy",
           "settings": dict(OPTS), "games": args.games,
           "n_groups": len(GROUPS), "n_groups_with_K24": len(full),
           "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
           "mean_gen_sec_per_group": round(
               statistics.mean([r["gen_sec"] for r in GROUPS]), 1) if GROUPS else None,
           "by_k": {}, "subsets": {}}

    for K in K_LEVELS:
        rows = [r for r in GROUPS if str(K) in r["by_k"]]
        out["by_k"][str(K)] = {
            "groups": len(rows),
            "pairwise": _agg(rows, K, "pairwise"),
            "spearman": _agg(rows, K, "spearman"),
            "kendall": _agg(rows, K, "kendall"),
            "top1_agree": _agg(rows, K, "top1_agree"),
            "top2_agree": _agg(rows, K, "top2_agree"),
            "spread": _agg(rows, K, "spread"),
            "std": _agg(rows, K, "std"),
            "spread_over_std": _agg(rows, K, "spread_over_std"),
            "top2_margin": _agg(rows, K, "top2_margin"),
        }

    def subset(name, pred):
        rows = [r for r in full if pred(r)]
        if len(rows) < 5:
            return
        out["subsets"][name] = {
            "n": len(rows),
            **{f"K{K}_pairwise": _agg(rows, K, "pairwise") for K in K_LEVELS}}

    hi = [r for r in full if (r["by_k"]["6"].get("pairwise") or 0) >= 0.75]
    lo = [r for r in full if (r["by_k"]["6"].get("pairwise") or 0) < 0.75]
    subset("high_confidence(K6 pairwise>=0.75)", lambda r: r in hi)
    subset("low_confidence(K6 pairwise<0.75)", lambda r: r in lo)
    subset("candidates_2_4", lambda r: r["n_cands"] <= 4)
    subset("candidates_5plus", lambda r: r["n_cands"] >= 5)
    subset("turn_early(1-5)", lambda r: r["turn"] <= 5)
    subset("turn_mid(6-10)", lambda r: 6 <= r["turn"] <= 10)
    subset("turn_late(11+)", lambda r: r["turn"] >= 11)
    for a in opps:
        subset(f"arch_{a}", lambda r, a=a: r["opponent"] == a)

    # ゲート判定
    k24 = out["by_k"].get("24", {})
    k6 = out["by_k"].get("6", {})
    pw24, pw6 = k24.get("pairwise") or 0, k6.get("pairwise") or 0
    out["gate"] = {
        "strong_PASS": bool(pw24 >= 0.72 and (k24.get("spearman") or 0) >= 0.55
                            and (k24.get("spread_over_std") or 0) > 1.0
                            and (k24.get("top1_agree") or 0) >= 0.60),
        "weak_PASS": bool(pw24 >= 0.70 and (pw24 - pw6) >= 0.04),
        "k24_pairwise": pw24, "k6_pairwise": pw6,
        "delta_k6_to_k24": round(pw24 - pw6, 4),
        "monotone_in_k": bool(
            (out["by_k"].get("12", {}).get("pairwise") or 0) >= pw6
            and pw24 >= (out["by_k"].get("12", {}).get("pairwise") or 0)),
    }
    out["gate"]["VERDICT"] = ("STRONG_PASS" if out["gate"]["strong_PASS"]
                              else "WEAK_PASS" if out["gate"]["weak_PASS"] else "FAIL")
    out["detail"] = GROUPS
    print(json.dumps({k: v for k, v in out.items() if k != "detail"},
                     ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
