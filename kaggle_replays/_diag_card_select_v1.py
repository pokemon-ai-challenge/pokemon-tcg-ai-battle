"""Phase6C §9.1/9.2: CARD select 候補を V0p / V1 が区別できるかのテスト。

各 CARD select で、候補カードを1手だけ適用した直後の状態を

  - handcrafted leaf(現行本番の葉評価)
  - V0p(state166 のみ)
  - V1(state166 + カード集合)

で評価し、候補間の spread と順位を比べる。**暫定教師**として
「その候補を選んでから自ターン終端まで Policy 貪欲で進めた末端評価」を使う
(§9.2。真の最適手ではないが、即時 leaf よりは候補の良し悪しを反映する)。

対戦挙動は変更しない(観測のみ)。出力: _diag_card_select_v1_results.json
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
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "value_net"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import SelectType  # noqa: E402
from hand_value_infer import HandValue  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_VDIR = _ROOT / "kaggle_replays" / "value_net"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

ROWS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0}
OPTS = {"every": 3, "worlds": 3, "max_cands": 8, "time_ms": 8000, "max_per_game": 6}
MODELS: dict = {}


def _spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None

    def rank(v):
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

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math_sqrt(sum((x - ma) ** 2 for x in ra))
    db = math_sqrt(sum((y - mb) ** 2 for y in rb))
    return (num / (da * db)) if da > 0 and db > 0 else None


def math_sqrt(x):
    return x ** 0.5


def _pairwise_acc(pred: list[float], teacher: list[float]) -> float | None:
    """教師が差をつけているペアで、予測が同じ向きに差をつけられた割合。

    予測が同値(タイ)のペアは **0.5 の部分点**にする。そうしないと
    「候補間で全く差が出ない評価器」(= handcrafted)が 0.0 になり、
    ランダム(0.5)より悪いという誤った印象になる。実際には情報ゼロ = 0.5 が正しい基準。
    """
    n = len(pred)
    tot = 0.0
    ok = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if teacher[i] == teacher[j]:
                continue
            tot += 1
            d = (pred[i] - pred[j])
            if d == 0:
                ok += 0.5
            elif d * (teacher[i] - teacher[j]) > 0:
                ok += 1
    return (ok / tot) if tot else None


def _probe(obs, model) -> dict | None:
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

    hand_ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0

    hc: dict[int, list[float]] = {i: [] for i in cand}
    v0: dict[int, list[float]] = {i: [] for i in cand}
    v1: dict[int, list[float]] = {i: [] for i in cand}
    tur: dict[int, list[float]] = {i: [] for i in cand}

    try:
        for _ in range(OPTS["worlds"]):
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
                        cst = child.observation.current
                        if cst is None:
                            continue
                        hc[i].append(hand_ev.evaluate(cst, me))
                        v0[i].append(MODELS["v0p"].predict_from_state(cst, me))
                        v1[i].append(MODELS["v1"].predict_from_state(cst, me))
                        v = P._rollout_and_eval(
                            child, me, {"opponent_depth": 1, "max_rollout_steps": 40},
                            deadline, hand_ev, model)
                        if v is not None:
                            tur[i].append(v)
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
    finally:
        try:
            P.cg_api.search_end()
        except Exception:  # noqa: BLE001
            pass

    keep = [i for i in cand if hc[i] and v0[i] and v1[i] and tur[i]]
    if len(keep) < 2:
        return None
    m = lambda d, i: statistics.mean(d[i])  # noqa: E731
    hcv = [m(hc, i) for i in keep]
    v0v = [m(v0, i) for i in keep]
    v1v = [m(v1, i) for i in keep]
    tv = [m(tur, i) for i in keep]
    policy_rank = [ranked.index(i) for i in keep]
    top1 = keep[policy_rank.index(min(policy_rank))]
    best = lambda vals: keep[vals.index(max(vals))]  # noqa: E731

    return {
        "game": _ctx["game"], "turn": int(getattr(state, "turn", 0) or 0),
        "n_cands": len(keep),
        "spread_handcrafted": round(max(hcv) - min(hcv), 6),
        "spread_v0p": round(max(v0v) - min(v0v), 6),
        "spread_v1": round(max(v1v) - min(v1v), 6),
        "spread_turn_teacher": round(max(tv) - min(tv), 6),
        "teacher_best_is_policy_top1": best(tv) == top1,
        "v0p_top1_matches_teacher": best(v0v) == best(tv),
        "v1_top1_matches_teacher": best(v1v) == best(tv),
        "hc_top1_matches_teacher": best(hcv) == best(tv),
        "v0p_pairwise": _pairwise_acc(v0v, tv),
        "v1_pairwise": _pairwise_acc(v1v, tv),
        "hc_pairwise": _pairwise_acc(hcv, tv),
        "v0p_spearman": _spearman(v0v, tv),
        "v1_spearman": _spearman(v1v, tv),
    }


def _install() -> None:
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.CARD and len(sel.option) >= 2):
            _ctx["seen"] += 1
            if (_ctx["seen"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    r = _probe(obs, ml_policy_agent._get_model(config))
                    if r is not None:
                        ROWS.append(r)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def _agg(key, rows):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(v), 4) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--out", default="_diag_card_select_v1_results.json")
    args = ap.parse_args()

    MODELS["v0p"] = HandValue(_VDIR / "hand_value_v0p.pt")
    MODELS["v1"] = HandValue(_VDIR / "hand_value_v1.pt")
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
        print(f"  game {g+1}/{args.games} probes={_ctx['probed']} total={len(ROWS)}",
              file=sys.stderr, flush=True)

    n = len(ROWS)
    out = {
        "note": ("CARD候補を1手適用した直後の状態を各評価器で採点。教師はターン終端rollout"
                 "(暫定・真の最適ではない)。budget_regime=research_accuracy。"),
        "games": args.games, "n_probes": n,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "mean_candidates": round(statistics.mean([r["n_cands"] for r in ROWS]), 2) if n else None,
        "spread": {
            "handcrafted": _agg("spread_handcrafted", ROWS),
            "v0p": _agg("spread_v0p", ROWS),
            "v1": _agg("spread_v1", ROWS),
            "turn_teacher": _agg("spread_turn_teacher", ROWS),
            "frac_v1_spread_gt_0.01": round(
                sum(1 for r in ROWS if r["spread_v1"] > 0.01) / n, 4) if n else None,
            "frac_v0p_spread_gt_0.01": round(
                sum(1 for r in ROWS if r["spread_v0p"] > 0.01) / n, 4) if n else None,
        },
        "agreement_with_turn_teacher": {
            "policy_top1": round(
                sum(1 for r in ROWS if r["teacher_best_is_policy_top1"]) / n, 4) if n else None,
            "handcrafted_top1": round(
                sum(1 for r in ROWS if r["hc_top1_matches_teacher"]) / n, 4) if n else None,
            "v0p_top1": round(
                sum(1 for r in ROWS if r["v0p_top1_matches_teacher"]) / n, 4) if n else None,
            "v1_top1": round(
                sum(1 for r in ROWS if r["v1_top1_matches_teacher"]) / n, 4) if n else None,
        },
        "pairwise_accuracy": {
            "handcrafted": _agg("hc_pairwise", ROWS),
            "v0p": _agg("v0p_pairwise", ROWS),
            "v1": _agg("v1_pairwise", ROWS),
        },
        "spearman": {"v0p": _agg("v0p_spearman", ROWS), "v1": _agg("v1_spearman", ROWS)},
        "rows": ROWS,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
