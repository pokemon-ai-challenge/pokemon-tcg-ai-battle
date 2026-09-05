"""Phase7 Track A: MAIN Action-Q 学習用 decision group データセット生成。

`_gen_actionq_teacher.py` は教師の**信頼性測定**専用だった。本スクリプトはそれに加えて
**学習に必要な特徴量も保存**する:

  group: state166 / 手札・トラッシュ・相手公開カードID / turn / select種別 / 勝敗
  candidate: option65 / action card id / policy score・rank / 実選択フラグ
             teacher_mean / teacher_std / teacher_k / splitA・splitB(自己一致用)

教師は「候補を適用 → 自ターン終端まで Policy 貪欲 → handcrafted leaf」を K 回独立決定化。
**同一 decision group は必ず同じ split**(group単位で分ける)。勝敗は試合終了後に埋める。

隠れ情報は保存しない(相手手札・山札は触らない)。budget_regime=research_accuracy。
出力: _actionq_dataset_<tag>.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
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
from hand_value_infer import visible_card_sets  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

GROUPS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "probed": 0, "seen": 0, "opp": ""}
OPTS = {"every": 1, "K": 6, "max_cands": 6, "time_ms": 20000, "max_per_game": 10,
        "select": "MAIN"}


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

    k = min((len(v) for v in vals.values()), default=0)
    if k < 4:
        return None
    half = k // 2

    # --- 学習用特徴量(production と同じ encoder を使う) ---
    try:
        state_feat = encoder.encode_state_from_state(state)
        opt_rows = encoder.encode_options_from_state(state, select)
        card_ids = encoder.encode_option_card_ids(state, select)
    except Exception:  # noqa: BLE001
        return None
    hand, disc, opp_vis = visible_card_sets(state, me)

    cands = []
    for i in cand:
        v = vals[i][:k]
        cands.append({
            "option_index": i,
            "option_feat": [round(float(x), 6) for x in opt_rows[i]],
            "action_card_id": int(card_ids[i]) if card_ids[i] is not None else -1,
            "option_type": int(getattr(select.option[i].type, "value",
                                       select.option[i].type)),
            "policy_score": round(float(scores[i]), 6),
            "policy_rank": ranked.index(i),
            "teacher_mean": round(statistics.mean(v), 6),
            "teacher_std": round(statistics.pstdev(v), 6),
            "teacher_k": k,
            "teacher_splitA": round(statistics.mean(v[:half]), 6),
            "teacher_splitB": round(statistics.mean(v[half:half * 2]), 6),
        })

    return {
        "group_id": f"g{_ctx['game']}_s{_ctx['seen']}",
        "game": _ctx["game"], "player": me, "turn": int(getattr(state, "turn", 0) or 0),
        "opponent_archetype": _ctx["opp"],
        "select_type": sel_name,
        "state_feat": [round(float(x), 6) for x in state_feat],
        "hand_ids": hand, "discard_ids": disc, "opp_visible_ids": opp_vis,
        "candidates": cands,
        "outcome": None,        # 試合終了後に埋める
    }


def _install() -> None:
    orig = ml_policy_agent._select_action
    want = OPTS["select"]

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        ok_type = (sel is not None and (
            (want == "MAIN" and sel.type == SelectType.MAIN)
            or (want == "CARD" and sel.type == SelectType.CARD)
            or (want == "BOTH" and sel.type in (SelectType.MAIN, SelectType.CARD))))
        if (ok_type and obs.current.yourIndex == _ctx["me"] and sel.option
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
        # 実際に選ばれた行動を記録(Logged Return / Policy Aux 用)
        out = orig(obs, config=config)
        if (GROUPS and _ctx["recording"] and out
                and GROUPS[-1].get("selected_option") is None
                and GROUPS[-1]["game"] == _ctx["game"]):
            GROUPS[-1]["selected_option"] = int(out[0]) if out else None
        return out

    ml_policy_agent._select_action = select_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--opponents", default="mega_lucario_ex,dragapult_ex,crustle")
    ap.add_argument("--select", default="MAIN", choices=["MAIN", "CARD", "BOTH"])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-per-game", type=int, default=10)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    OPTS.update(K=args.K, every=args.every, max_per_game=args.max_per_game,
                select=args.select)

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
        start = len(GROUPS)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, probed=0, opp=arch)
        res = (runner.play_game(climb, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        won = None if res.error else (1 if res.winner == _ctx["me"] else 0)
        for r in GROUPS[start:]:
            r["outcome"] = won
        print(f"  game {g+1}/{args.games} vs {arch} groups={len(GROUPS)} won={won}",
              file=sys.stderr, flush=True)

    GROUPS[:] = [r for r in GROUPS if r["outcome"] is not None]
    out_path = _HERE / f"_actionq_dataset_{args.tag}.jsonl.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "tag": args.tag, "select": args.select, "games": args.games,
        "opponents": opps, "K": args.K,
        "budget_regime": "research_accuracy",
        "n_groups": len(GROUPS),
        "n_candidates": sum(len(r["candidates"]) for r in GROUPS),
        "mean_cands": round(statistics.mean([len(r["candidates"]) for r in GROUPS]), 2)
        if GROUPS else None,
        "by_archetype": {a: sum(1 for r in GROUPS if r["opponent_archetype"] == a)
                         for a in opps},
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (_HERE / f"_actionq_dataset_{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
