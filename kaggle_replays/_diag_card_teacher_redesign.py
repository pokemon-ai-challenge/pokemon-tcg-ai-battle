"""Phase7 Track B: CARD counterfactual 教師の再設計比較(T0〜T5)。

CARD 教師は現状 自己pairwise 0.560(≒偶然)/ std > spread でノイズが信号を上回る。
モデルを作る前に **教師自身の再現性**を上げられるかを比較する。

設計上の要点: **同じ rollout を使い回して leaf 種別だけを変える**。
rollout をやり直すと leaf 差と rollout ノイズが交絡するため、
1回の rollout で到達した末端 **state** を保存し、handcrafted / V1 / blend の3評価器で採点する。
同様に K=24 まで回して前半を切り出せば K=6/12/24 を**入れ子**で比較できる。

  T0 : handcrafted leaf, horizon=自分の次ターン開始(opponent_depth=1), K=6
  T1 : V1(hand-aware) leaf, 同 horizon, K=6
  T2 : blend(0.5*V1 + 0.5*handcrafted), 同 horizon, K=6
  T3 : handcrafted leaf, horizon=opponent_depth=2(より長い地平), K=6
  T5 : handcrafted leaf, 同 horizon, K=12 / K=24

  T4(CRN強化)は cg のシャッフル/相手乱数ストリームが API 非公開のため
  **本フェーズでは実施不可**。現状の共有範囲(root 決定化を候補間で共有)を明記して報告する。

出力: _card_teacher_redesign.json
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

from cg import api as cg_api  # noqa: E402
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

GROUPS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0, "opp": ""}
OPTS = {"every": 2, "Kmax": 24, "max_cands": 5, "time_ms": 150000, "max_per_game": 3}
MODELS: dict = {}


def _rollout_to_state(node, me, opponent_depth, max_steps, deadline, policy_model):
    """`pipeline._rollout_and_eval` と同じ停止条件で、**末端 state** を返す。

    これにより 1 回の rollout を複数の leaf 評価器で採点でき、
    leaf 差と rollout ノイズを交絡させずに比較できる。
    """
    prev_actor = me
    opp_turns = 0
    for _ in range(max_steps):
        if time.perf_counter() > deadline:
            break
        obs = node.observation
        state = obs.current
        if state is None:
            return None
        if state.result != -1:
            return state
        actor = state.yourIndex
        if prev_actor == me and actor != me:
            opp_turns += 1
        if actor == me and prev_actor != me and opp_turns >= opponent_depth:
            return state
        if obs.select is None or not obs.select.option:
            return state
        sel = P._greedy_selection(policy_model, obs)
        if not sel:
            return state
        try:
            node = cg_api.search_step(node.searchId, sel)
        except ValueError:
            return state
        prev_actor = actor
    st = node.observation.current
    return st


def _collect(obs, model, depth, K, deadline):
    """候補ごとに K 回 rollout し、末端 state を3評価器で採点した値列を返す。"""
    select = obs.select
    state = obs.current
    me = state.yourIndex
    scores = model.score_options(obs, None, None)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cand = ranked[: OPTS["max_cands"]]
    hc = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    v1 = MODELS["v1"]
    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    out = {i: {"hc": [], "v1": [], "blend": []} for i in cand}

    for _ in range(K):
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
                    child = cg_api.search_step(root.searchId, [i])
                except ValueError:
                    continue
                try:
                    st = _rollout_to_state(child, me, depth, 40, deadline, model)
                    if st is None:
                        continue
                    a = hc.evaluate(st, me)
                    b = v1.predict_from_state(st, me)
                    out[i]["hc"].append(a)
                    out[i]["v1"].append(b)
                    out[i]["blend"].append(0.5 * a + 0.5 * b)
                finally:
                    try:
                        cg_api.search_release(child.searchId)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            try:
                cg_api.search_release(root.searchId)
            except Exception:  # noqa: BLE001
                pass
    try:
        cg_api.search_end()
    except Exception:  # noqa: BLE001
        pass
    return cand, out


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


def _self_consistency(vals: list[list[float]]):
    """候補ごとの値列(長さK)から split-half の自己一致を出す。"""
    k = min(len(v) for v in vals)
    if k < 4:
        return None
    h = k // 2
    A = [statistics.mean(v[:h]) for v in vals]
    B = [statistics.mean(v[h:h * 2]) for v in vals]
    allv = [statistics.mean(v[:k]) for v in vals]
    std = statistics.mean([statistics.pstdev(v[:k]) for v in vals])
    spread = max(allv) - min(allv)
    return {
        "top1_agree": (A.index(max(A)) == B.index(max(B))),
        "pairwise": _pairwise(A, B),
        "spearman": _spearman(A, B),
        "spread": spread, "std": std,
        "spread_over_std": (spread / std) if std > 0 else None,
        "k": k,
    }


def _probe(obs, model):
    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0
    # depth=1 で Kmax まで(T0/T1/T2/T5 を入れ子で作る)
    cand, d1 = _collect(obs, model, 1, OPTS["Kmax"], deadline)
    if not cand:
        return None
    # depth=2(より長い地平)は K=6
    _, d2 = _collect(obs, model, 2, 6, deadline)

    rec = {"game": _ctx["game"], "turn": int(getattr(obs.current, "turn", 0) or 0),
           "opponent": _ctx["opp"], "n_cands": len(cand), "configs": {}}

    def add(name, vals):
        sc = _self_consistency(vals)
        if sc:
            rec["configs"][name] = sc

    for K in (6, 12, 24):
        v = [d1[i]["hc"][:K] for i in cand]
        if min(len(x) for x in v) >= max(4, K // 2):
            add(f"T0_hc_d1_K{K}" if K == 6 else f"T5_hc_d1_K{K}", v)
    for key, nm in (("v1", "T1_v1leaf_d1_K6"), ("blend", "T2_blend50_d1_K6")):
        v = [d1[i][key][:6] for i in cand]
        if min(len(x) for x in v) >= 4:
            add(nm, v)
    v = [d2[i]["hc"][:6] for i in cand] if d2 else []
    if v and min(len(x) for x in v) >= 4:
        add("T3_hc_d2_K6", v)
    return rec if rec["configs"] else None


def _install():
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.CARD and sel.maxCount == 1
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--opponents", default="mega_lucario_ex,dragapult_ex,crustle")
    ap.add_argument("--out", default="_card_teacher_redesign.json")
    args = ap.parse_args()

    MODELS["v1"] = HandValue(_VDIR / "hand_value_v1.pt")
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

    names = sorted({n for r in GROUPS for n in r["configs"]})
    summary = {}
    GATE = {"pairwise": 0.70, "spearman": 0.50, "top1": 0.55}
    for n in names:
        rows = [r["configs"][n] for r in GROUPS if n in r["configs"]]
        if not rows:
            continue
        f = lambda k: [x[k] for x in rows if x.get(k) is not None]  # noqa: E731
        pw = f("pairwise")
        sp = f("spearman")
        summary[n] = {
            "groups": len(rows),
            "self_pairwise": round(statistics.mean(pw), 4) if pw else None,
            "self_spearman": round(statistics.mean(sp), 4) if sp else None,
            "self_top1_agree": round(statistics.mean(
                [1.0 if x["top1_agree"] else 0.0 for x in rows]), 4),
            "mean_spread": round(statistics.mean(f("spread")), 5),
            "mean_std": round(statistics.mean(f("std")), 5),
            "spread_over_std": round(statistics.mean(f("spread_over_std")), 3)
            if f("spread_over_std") else None,
            "mean_k": round(statistics.mean(f("k")), 1),
        }
        s = summary[n]
        summary[n]["GATE_PASS"] = bool(
            (s["self_pairwise"] or 0) >= GATE["pairwise"]
            and (s["self_spearman"] or 0) >= GATE["spearman"]
            and (s["self_top1_agree"] or 0) >= GATE["top1"]
            and (s["mean_spread"] or 0) > (s["mean_std"] or 1))

    out = {"note": ("CARD教師の自己再現性比較。同一 rollout を使い回して leaf 種別だけ変え、"
                    "K は入れ子で切り出す(rollout ノイズと交絡させない)。"),
           "budget_regime": "research_accuracy",
           "T4_CRN": ("未実施: cg のシャッフル/相手乱数ストリームが API 非公開のため、"
                      "root 決定化の候補間共有を超える CRN は本フェーズでは実装不可。"),
           "gate": GATE, "games": args.games, "n_groups": len(GROUPS),
           "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
           "configs": summary, "detail": GROUPS}
    print(json.dumps({k: v for k, v in out.items() if k != "detail"},
                     ensure_ascii=False, indent=2))
    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _HERE / dest.name
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
