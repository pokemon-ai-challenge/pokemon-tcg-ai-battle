"""Phase11D: 独立 K=6 教師ブロック A/B/C/D を **生成時に同時取得**する診断セット。

既存 `_actionq_v2.jsonl.gz` には生 observation が保存されていないため、
既存 group に対して後からブロックを追加生成できない(検証済み)。
よって新規に局面を取り、その場で 4 ブロックを引く。

各ブロックは独立した決定化・rollout(サンプル重複なし)。
K を 12/24 に増やす実験ではなく、**独立な K=6 推定量を4つ**得るのが目的。

stratum は診断用に3等分で割り当てる(元の 64/16/20 split とは別物。
教師ノイズの層別比較に使うだけで、モデル学習には使わない)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
K, NBLOCK = 6, 4
GAME_OFFSET = 20000          # 既存 v2 の game index と衝突させない

GROUPS: list[dict] = []
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "stratum": 0}
QUOTA = None
OPTS = {"time_ms": 60000, "max_cands": 8}


def _block(obs, model, cand, me, deadline):
    """1ブロック = 独立した K 回の決定化 + rollout。"""
    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})

    def factory():
        return search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me), obs)

    vals = {i: [] for i in cand}
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


def _probe(obs, model):
    select, state = obs.select, obs.current
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

    blocks = []
    for _b in range(NBLOCK):
        vals = _block(obs, model, cand, me,
                      time.perf_counter() + OPTS["time_ms"] / 1000.0)
        k = min((len(vals[i]) for i in cand), default=0)
        if k < K:                 # K未達のブロックがあれば group ごと捨てる(§17 中止条件)
            return None
        blocks.append([statistics.mean(vals[i][:K]) for i in cand])

    try:
        sf = encoder.encode_state_from_state(state)
        orow = encoder.encode_options_from_state(state, select)
        cids = encoder.encode_option_card_ids(state, select)
    except Exception:  # noqa: BLE001
        return None

    cands = []
    for n, i in enumerate(cand):
        o = select.option[i]
        cands.append({
            "option_index": i,
            "option_feat": [round(float(x), 6) for x in orow[i]],
            "action_card_id": int(cids[i]) if cids[i] is not None else -1,
            "option_type": int(getattr(o.type, "value", o.type)),
            "policy_score": round(float(scores[i]), 6),
            "policy_rank": ranked.index(i),
            "blocks": [round(blocks[b][n], 6) for b in range(NBLOCK)],
        })
    return {
        "group_id": "blk{}_{}".format(_ctx["game"], _ctx["in_game"]),
        "game": _ctx["game"], "turn": int(getattr(state, "turn", 0) or 0),
        "turn_band": turn_band(int(getattr(state, "turn", 0) or 0)),
        "cand_band": cand_band(len(cand)),
        "arch": _ctx["opp"], "stratum": _ctx["stratum"],
        "n_cands": len(cand),
        "state_feat": [round(float(x), 6) for x in sf],
        "candidates": cands,
    }


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        if not _ctx["recording"] or obs.current is None:
            return orig(obs, config=config)
        sel = obs.select
        if (obs.current.yourIndex == _ctx["me"] and sel is not None and sel.option
                and sel.type == SelectType.MAIN and sel.maxCount == 1
                and len(sel.option) >= 2):
            t = int(getattr(obs.current, "turn", 0) or 0)
            n_c = min(len(sel.option), OPTS["max_cands"])
            if QUOTA.accept(t, n_c, _ctx["opp"], _ctx["in_game"]):
                try:
                    r = _probe(obs, ml_policy_agent._get_model(config))
                    if r:
                        GROUPS.append(r)
                        QUOTA.commit(t, len(r["candidates"]), _ctx["opp"])
                        _ctx["in_game"] += 1
                except Exception as exc:  # noqa: BLE001
                    print("[probe-err] {}".format(exc), file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=384)
    ap.add_argument("--max-games", type=int, default=200)
    ap.add_argument("--per-game-cap", type=int, default=8)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="blocks")
    ap.add_argument("--offset", type=int, default=GAME_OFFSET)
    args = ap.parse_args()
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
    for gi in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch,
                    stratum=g % 3)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        if gi % 5 == 0:
            print("  [w{}] game#{} total={}/{} {}min".format(
                args.worker_id, g, QUOTA.total, args.target,
                int((time.perf_counter() - t0) / 60)), file=sys.stderr, flush=True)

    out = _HERE / "_teacher_blocks_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "tag": args.tag, "K": K, "blocks": NBLOCK, "n_groups": len(GROUPS),
        "n_candidates": sum(len(r["candidates"]) for r in GROUPS),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "turn_band": dict(Counter(r["turn_band"] for r in GROUPS)),
        "cand_band": dict(Counter(r["cand_band"] for r in GROUPS)),
        "arch": dict(Counter(r["arch"] for r in GROUPS)),
        "stratum": dict(Counter(r["stratum"] for r in GROUPS)),
        "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
