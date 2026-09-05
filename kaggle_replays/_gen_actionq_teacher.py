"""Phase6.5 §4.3: counterfactual 教師の**信頼性**測定 + 固定教師セット生成。

Phase6 で V0p も V1 も「ターン終端 rollout 教師」に対して pairwise < 0.5 / Spearman 負だった。
モデル側の失敗と決めつける前に、**教師が自分自身と一致するのか**を測る。
教師が自己一致しないなら、どんなモデルもそれに合わせられない(= 上限が 0.5 付近)。

各 decision group(CARD / MAIN)について、候補ごとに K 回**独立な決定化**で
「候補を適用 → 自ターン終端まで Policy 貪欲 → leaf 評価」を行い、
K 回を前半/後半の2群に分けて **群間の順位一致**を測る:

  - top-1 一致率(群Aの1位 == 群Bの1位)
  - pairwise 一致率
  - Spearman
  - 候補ごとの std / margin / 95%CI

これが教師の**再現性上限**。モデルの pairwise はこの上限を超えられない。

出力: _actionq_teacher.json(信頼性レポート + 固定教師セット)
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
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0}
OPTS = {"every": 2, "K": 6, "max_cands": 6, "time_ms": 20000, "max_per_game": 5}


def _rank(v):
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
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


def _pairwise(pred, teach):
    tot = ok = 0.0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            if teach[i] == teach[j]:
                continue
            tot += 1
            d = pred[i] - pred[j]
            ok += 0.5 if d == 0 else (1.0 if d * (teach[i] - teach[j]) > 0 else 0.0)
    return (ok / tot) if tot else None


def _probe(obs, model, sel_name) -> dict | None:
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

    for _ in range(OPTS["K"]):
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

    # 全候補が同数の評価を持つところまで切り揃える(不公平な平均を避ける)
    k = min((len(v) for v in vals.values()), default=0)
    if k < 4:                       # 群分割に最低4回(2+2)必要
        return None
    keep = list(cand)
    half = k // 2
    a_vals = [statistics.mean(vals[i][:half]) for i in keep]
    b_vals = [statistics.mean(vals[i][half:half * 2]) for i in keep]
    all_vals = [statistics.mean(vals[i][:k]) for i in keep]
    stds = [statistics.pstdev(vals[i][:k]) for i in keep]

    best_a = a_vals.index(max(a_vals))
    best_b = b_vals.index(max(b_vals))
    srt = sorted(all_vals, reverse=True)
    margin = (srt[0] - srt[1]) if len(srt) > 1 else 0.0

    return {
        "game": _ctx["game"], "turn": int(getattr(state, "turn", 0) or 0),
        "select_type": sel_name,
        "n_cands": len(keep),
        "k_evals": k,
        "candidate_policy_ranks": [ranked.index(i) for i in keep],
        "teacher_mean": [round(v, 6) for v in all_vals],
        "teacher_std": [round(v, 6) for v in stds],
        "teacher_spread": round(max(all_vals) - min(all_vals), 6),
        "teacher_top1_margin": round(margin, 6),
        "splitA_mean": [round(v, 6) for v in a_vals],
        "splitB_mean": [round(v, 6) for v in b_vals],
        "self_top1_agree": best_a == best_b,
        "self_pairwise": _pairwise(a_vals, b_vals),
        "self_spearman": _spearman(a_vals, b_vals),
        "mean_std": round(statistics.mean(stds), 6),
    }


def _install() -> None:
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type in (SelectType.CARD, SelectType.MAIN)
                and sel.maxCount == 1 and len(sel.option) >= 2):
            _ctx["seen"] += 1
            if (_ctx["seen"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    nm = "CARD" if sel.type == SelectType.CARD else "MAIN"
                    r = _probe(obs, ml_policy_agent._get_model(config), nm)
                    if r is not None:
                        GROUPS.append(r)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def _summ(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(v), 4) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--K", type=int, default=OPTS["K"])
    ap.add_argument("--out", default="_actionq_teacher.json")
    args = ap.parse_args()
    OPTS["K"] = args.K

    _install()
    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("climb_baseline")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    t0 = time.perf_counter()
    for g in range(args.games):
        p0 = (g % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, probed=0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        print(f"  game {g+1}/{args.games} groups={len(GROUPS)}", file=sys.stderr, flush=True)

    out = {"note": ("counterfactual 教師の自己一致(再現性上限)。K回の独立決定化を2群に割り、"
                    "群間の順位一致を測る。モデルの pairwise はこの上限を超えられない。"),
           "budget_regime": "research_accuracy",
           "settings": dict(OPTS), "games": args.games,
           "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
           "n_groups": len(GROUPS)}
    for nm in ("CARD", "MAIN"):
        rows = [r for r in GROUPS if r["select_type"] == nm]
        hi = [r for r in rows if r["teacher_top1_margin"] >= 0.02]
        out[nm] = {
            "groups": len(rows),
            "mean_cands": round(statistics.mean([r["n_cands"] for r in rows]), 2) if rows else None,
            "mean_k": round(statistics.mean([r["k_evals"] for r in rows]), 2) if rows else None,
            "TEACHER_SELF_top1_agree": round(
                sum(1 for r in rows if r["self_top1_agree"]) / len(rows), 4) if rows else None,
            "TEACHER_SELF_pairwise": _summ(rows, "self_pairwise"),
            "TEACHER_SELF_spearman": _summ(rows, "self_spearman"),
            "mean_teacher_spread": _summ(rows, "teacher_spread"),
            "mean_teacher_std": _summ(rows, "mean_std"),
            "mean_top1_margin": _summ(rows, "teacher_top1_margin"),
            "high_margin_rate(>=0.02)": round(len(hi) / len(rows), 4) if rows else None,
            "high_margin_SELF_top1_agree": round(
                sum(1 for r in hi if r["self_top1_agree"]) / len(hi), 4) if hi else None,
            "high_margin_SELF_pairwise": _summ(hi, "self_pairwise"),
        }
    out["groups_detail"] = GROUPS
    print(json.dumps({k: v for k, v in out.items() if k != "groups_detail"},
                     ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
