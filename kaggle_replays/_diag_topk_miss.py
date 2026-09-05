"""Policy top-k 見落とし率(climb v1.5 診断 §5.1 / 仮説B)。

現行 pipeline は Policy 上位 `top_k=4` だけを探索候補にする。**探索的に最良の手が top-4 の外に
どれだけあるか** を実戦局面で測る。ここが小さければ「top-k を増やす」投資は無意味であり、
大きければ dynamic top-k / beam search の前提が立つ。

方法:
  climb を実際に対戦させ、探索適用対象(MAIN・maxCount==1・選択肢が十分ある)の decision を
  一定間隔でサンプルし、その局面で **全選択肢(最大 max_options 件)** を、本番より多い
  決定化数・緩い時間予算で評価する(= 通常より強い評価器)。得られた search-Q の argmax が
  Policy 何位だったかを記録する。

  注: 本番と同じ評価器(handcrafted leaf・opponent_depth=1)を使い、**候補数と決定化数と予算だけ**
  を増やす。つまりここで測るのは「同じ探索を全選択肢に広げたら1位が変わるか」であって、
  より賢い評価器を使った場合の真の最善手ではない。top-k 拡張の上限効果の推定値。

出力: _diag_topk_miss_results.json
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

from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

PROBES: list[dict] = []
_ctx = {"game": -1, "recording": False, "eligible_seen": 0, "probed": 0}
OPTS = {"every": 6, "max_options": 16, "worlds": 12, "time_ms": 8000,
        "min_options": 5, "max_per_game": 8}


def _probe(obs, model) -> dict | None:
    """この局面で全選択肢を深く評価し、search-Q argmax の Policy 順位を返す。"""
    select = obs.select
    state = obs.current
    me = state.yourIndex
    try:
        scores = model.score_options(obs, None, None)
    except Exception:  # noqa: BLE001
        return None
    if not scores or len(scores) != len(select.option):
        return None

    probs = P._softmax(scores)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    cand = ranked[: OPTS["max_options"]]

    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs,
    )
    evaluator = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    cfg = {"opponent_depth": 1, "max_rollout_steps": 40}
    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0

    agg: dict[int, list[float]] = {i: [] for i in cand}
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
                    s = P._evaluate_candidate(root, [i], me, cfg, deadline, evaluator, model)
                    if s is not None:
                        agg[i].append(s)
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

    # 全候補が同じ回数評価された分だけを使う(世界数が候補ごとに違うと平均が不公平になる)。
    n_common = min((len(v) for v in agg.values()), default=0)
    if n_common < 3:
        return None
    q = {i: statistics.mean(agg[i][:n_common]) for i in cand}

    best_i = max(q, key=lambda i: q[i])
    policy_rank = ranked.index(best_i) + 1          # 1-origin
    top1_i = ranked[0]
    return {
        "game": _ctx["game"],
        "turn": int(getattr(state, "turn", 0) or 0),
        "n_options": len(select.option),
        "n_candidates_probed": len(cand),
        "worlds_used": n_common,
        "best_policy_rank": policy_rank,
        "best_q": round(q[best_i], 5),
        "top1_q": round(q[top1_i], 5),
        "q_gap_best_minus_top1": round(q[best_i] - q[top1_i], 5),
        "top1_prob": round(probs[top1_i], 5),
        "best_prob": round(probs[best_i], 5),
        "q_spread": round(max(q.values()) - min(q.values()), 5),
    }


def _install_probe() -> None:
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"]:
            return orig(obs, config=config)
        sel = obs.select
        eligible = (
            sel is not None and obs.current is not None
            and sel.type == SelectType.MAIN and sel.maxCount == 1
            and sel.option is not None and len(sel.option) >= OPTS["min_options"]
        )
        if eligible:
            _ctx["eligible_seen"] += 1
            if (_ctx["eligible_seen"] % OPTS["every"] == 0
                    and _ctx["probed"] < OPTS["max_per_game"]):
                try:
                    model = ml_policy_agent._get_model(config)
                    row = _probe(obs, model)
                    if row is not None:
                        PROBES.append(row)
                        _ctx["probed"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--every", type=int, default=OPTS["every"])
    ap.add_argument("--worlds", type=int, default=OPTS["worlds"])
    ap.add_argument("--max-options", type=int, default=OPTS["max_options"])
    ap.add_argument("--max-per-game", type=int, default=OPTS["max_per_game"])
    ap.add_argument("--out", default=str(_HERE / "_diag_topk_miss_results.json"))
    args = ap.parse_args()
    OPTS.update(every=args.every, worlds=args.worlds, max_options=args.max_options,
                max_per_game=args.max_per_game)

    _install_probe()
    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB_WEIGHTS
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("abl_5_full")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    t0 = time.perf_counter()
    for g in range(args.games):
        climb_p0 = (g % 2 == 0)
        _ctx.update(game=g, recording=True, probed=0)
        res = (runner.play_game(climb, opp, deck_c, deck_o) if climb_p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        print(f"  game {g+1}/{args.games} probes={_ctx['probed']} total={len(PROBES)}",
              file=sys.stderr)
        if res.error:
            print(f"[warn] {res.error}", file=sys.stderr)

    n = len(PROBES)
    ranks = [p["best_policy_rank"] for p in PROBES]
    dist = Counter(ranks)
    cum = {}
    for k in (1, 2, 4, 8, 12, 16):
        cum[f"best_in_top{k}"] = round(sum(1 for r in ranks if r <= k) / n, 4) if n else None
    outside4 = [p for p in PROBES if p["best_policy_rank"] > 4]

    out = {
        "note": "本番と同じ評価器で候補数・決定化数・予算だけ増やしたときの最良手の Policy 順位。",
        "settings": dict(OPTS),
        "games": args.games,
        "opponent": args.opponent,
        "n_probes": n,
        "elapsed_min": round((time.perf_counter() - t0) / 60.0, 2),
        "rank_distribution": {str(k): v for k, v in sorted(dist.items())},
        "cumulative": cum,
        "miss_rate_outside_top4": round(len(outside4) / n, 4) if n else None,
        "q_gap_when_outside_top4": {
            "mean": round(statistics.mean([p["q_gap_best_minus_top1"] for p in outside4]), 5)
            if outside4 else None,
            "median": round(statistics.median([p["q_gap_best_minus_top1"] for p in outside4]), 5)
            if outside4 else None,
            "max": round(max([p["q_gap_best_minus_top1"] for p in outside4]), 5)
            if outside4 else None,
            "n_gap_above_tie_eps_0.02": sum(
                1 for p in outside4 if p["q_gap_best_minus_top1"] > 0.02
            ) if outside4 else 0,
        },
        "q_gap_all": {
            "mean": round(statistics.mean([p["q_gap_best_minus_top1"] for p in PROBES]), 5)
            if n else None,
            "frac_gap_above_tie_eps": round(
                sum(1 for p in PROBES if p["q_gap_best_minus_top1"] > 0.02) / n, 4) if n else None,
        },
        "mean_n_options": round(statistics.mean([p["n_options"] for p in PROBES]), 2) if n else None,
        "probes": PROBES,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "probes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
