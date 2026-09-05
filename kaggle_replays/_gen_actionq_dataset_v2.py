"""Phase9B: 層化 MAIN decision group 生成(K=6 固定 / test は独立教師A/B)。

Phase9A の結論に従い **K は 6 のまま**(K を増やしても教師再現性は上がらない)。
代わりに `_actionq_sampling.StratifiedQuota` でターン帯・候補数帯・アーキタイプを
層化し、先着順バイアス(序盤偏重)を解消する。

test split の group だけは **独立した K=6 ブロックを2つ**(teacher_A / teacher_B)作る。
  teacher_A : 信頼度層の定義・自己一致の片側(診断用)
  teacher_B : モデル順位評価(正式教師)
A と B のサンプルは重複させない(別々の rollout ブロック)。これで
「同じ教師値で層を定義し、その層で性能が高いと主張する」循環評価を避ける。

split は **試合単位**。試合を先に train/val/test へ割り当ててから生成する
(生成後に割り直すと test だけ A/B が無い、といった不整合が起きるため)。

出力: _actionq_v2_<tag>.jsonl.gz / _actionq_v2_<tag>_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "value_net"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
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

K = 6
GROUPS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "split": 0}
QUOTA: StratifiedQuota | None = None
OPTS = {"time_ms": 30000, "max_cands": 8}


def split_of_game(game_idx: int) -> int:
    """**試合単位** split(0=train / 1=val / 2=test)。64/16/20。

    test を 20% にしているのは、総数1500に対し test>=250 を満たすため。
    worker 間で game index は重複しないので、並列でも同じ割当になる。
    """
    h = int(hashlib.md5(f"v2game{game_idx}".encode()).hexdigest(), 16) % 100
    return 0 if h < 64 else 1 if h < 80 else 2


def _rollout_block(obs, model, cand, me, deadline):
    """K 回の独立決定化で候補ごとの値列を返す(1ブロック)。"""
    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    factory = lambda: search_adapter.to_search_begin_kwargs(  # noqa: E731
        match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
    vals: dict[int, list[float]] = {i: [] for i in cand}
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
    return vals


def _block_stats(vals, cand):
    k = min((len(vals[i]) for i in cand), default=0)
    if k < 4:
        return None, 0
    out = []
    for i in cand:
        v = vals[i][:k]
        h = k // 2
        out.append({"mean": round(statistics.mean(v), 6),
                    "std": round(statistics.pstdev(v), 6),
                    "k": k,
                    "half_a": round(statistics.mean(v[:h]), 6),
                    "half_b": round(statistics.mean(v[h:k]), 6)})
    return out, k


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

    deadline = time.perf_counter() + OPTS["time_ms"] / 1000.0
    is_test = _ctx["split"] == 2
    a_vals = _rollout_block(obs, model, cand, me, deadline)
    a_stats, ka = _block_stats(a_vals, cand)
    if a_stats is None:
        return None
    b_stats = None
    if is_test:
        # 独立ブロック(別 rollout。A のサンプルは再利用しない)
        b_vals = _rollout_block(obs, model, cand, me,
                                time.perf_counter() + OPTS["time_ms"] / 1000.0)
        b_stats, kb = _block_stats(b_vals, cand)
        if b_stats is None:
            return None

    try:
        state_feat = encoder.encode_state_from_state(state)
        opt_rows = encoder.encode_options_from_state(state, select)
        card_ids = encoder.encode_option_card_ids(state, select)
    except Exception:  # noqa: BLE001
        return None
    hand, disc, opp_vis = visible_card_sets(state, me)

    cands = []
    for n, i in enumerate(cand):
        o = select.option[i]
        cands.append({
            "option_index": i,
            "option_feat": [round(float(x), 6) for x in opt_rows[i]],
            "action_card_id": int(card_ids[i]) if card_ids[i] is not None else -1,
            "option_type": int(getattr(o.type, "value", o.type)),
            "policy_score": round(float(scores[i]), 6),
            "policy_rank": ranked.index(i),
            # train/val は teacher_mean を使う。test は teacher_B を正式教師にする。
            "teacher_mean": a_stats[n]["mean"] if not is_test else b_stats[n]["mean"],
            "teacher_std": a_stats[n]["std"] if not is_test else b_stats[n]["std"],
            "teacher_k": a_stats[n]["k"],
            "teacher_A": a_stats[n],
            "teacher_B": b_stats[n] if is_test else None,
        })
    return {
        "group_id": f"v2g{_ctx['game']}_{_ctx['in_game']}",
        "game": _ctx["game"], "player": me, "turn": int(getattr(state, "turn", 0) or 0),
        "turn_band": turn_band(int(getattr(state, "turn", 0) or 0)),
        "cand_band": cand_band(len(cand)),
        "opponent_archetype": _ctx["opp"],
        "split": _ctx["split"],
        "select_type": "MAIN",
        "state_feat": [round(float(x), 6) for x in state_feat],
        "hand_ids": hand, "discard_ids": disc, "opp_visible_ids": opp_vis,
        "candidates": cands,
        "outcome": None,
        "selected_option": None,
    }


def _install():
    orig = ml_policy_agent._select_action

    def select_action(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        made = None
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.MAIN and sel.maxCount == 1
                and len(sel.option) >= 2):
            turn = int(getattr(obs.current, "turn", 0) or 0)
            n_c = min(len(sel.option), OPTS["max_cands"])
            if QUOTA.accept(turn, n_c, _ctx["opp"], _ctx["in_game"]):
                try:
                    r = _probe(obs, ml_policy_agent._get_model(config))
                    if r is not None:
                        GROUPS.append(r)
                        QUOTA.commit(turn, len(r["candidates"]), _ctx["opp"])
                        _ctx["in_game"] += 1
                        made = r
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe-err] {exc}", file=sys.stderr)
        out = orig(obs, config=config)
        if made is not None and out:
            made["selected_option"] = int(out[0])
        return out

    ml_policy_agent._select_action = select_action


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--max-games", type=int, default=400)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--per-game-cap", type=int, default=12)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()
    # 並列時: worker ごとに target を等分し、game index を stride で割り当てる。
    # game index が重複しないので split_of_game の割当も worker 間で一貫する。
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap,
                            cand_soft_cap=0.45)
    _install()
    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    games_played = 0
    for gi in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        g = args.worker_id + gi * args.num_workers      # 大域的に一意な game index
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{arch}.json")
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (g % 2 == 0)
        start = len(GROUPS)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0,
                    opp=arch, split=split_of_game(g))
        res = (runner.play_game(climb, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        games_played += 1
        won = None if res.error else (1 if res.winner == _ctx["me"] else 0)
        for r in GROUPS[start:]:
            r["outcome"] = won
        if gi % 5 == 0 or QUOTA.total >= args.target:
            rq = QUOTA.remaining_turn_quota()
            print(f"  [w{args.worker_id}] game#{g} vs {arch} total={QUOTA.total}/{args.target} "
                  f"remain={rq} elapsed={int((time.perf_counter()-t0)/60)}min",
                  file=sys.stderr, flush=True)

    GROUPS[:] = [r for r in GROUPS if r["outcome"] is not None]
    out_path = _HERE / f"_actionq_{args.tag}.jsonl.gz"
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_split = Counter(r["split"] for r in GROUPS)
    summary = {
        "tag": args.tag, "K": K, "budget_regime": "research_accuracy",
        "games_played": games_played, "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "n_groups": len(GROUPS),
        "n_candidates": sum(len(r["candidates"]) for r in GROUPS),
        "split_counts": {"train": by_split[0], "val": by_split[1], "test": by_split[2]},
        "turn_band": dict(Counter(r["turn_band"] for r in GROUPS)),
        "cand_band": dict(Counter(r["cand_band"] for r in GROUPS)),
        "archetype": dict(Counter(r["opponent_archetype"] for r in GROUPS)),
        "option_type": dict(Counter(c["option_type"] for r in GROUPS
                                    for c in r["candidates"]).most_common()),
        "mean_cands": round(statistics.mean([len(r["candidates"]) for r in GROUPS]), 2)
        if GROUPS else None,
        "test_has_teacher_B": sum(1 for r in GROUPS if r["split"] == 2
                                  and r["candidates"][0]["teacher_B"] is not None),
        "quota_report": QUOTA.report(),
        "dataset_sha": hashlib.sha256(out_path.read_bytes()).hexdigest()[:16],
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (_HERE / f"_actionq_{args.tag}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
