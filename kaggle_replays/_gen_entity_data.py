"""Phase14: raw entity 付きデータ収集(collision audit / recall audit / probe を1回で賄う)。

1 回のプレイから 3 つを同時に取る:

  (a) STATES : 全 MAIN 決定の state166 + raw entity + policy/Q0 順位(教師なし・安価)
               -> §6 State Collision Audit, §8 recall の分母
  (b) PROBES : quota で選んだ部分集合に、Phase11D/12 と同一仕様の教師
               (4 block x K=6, handcrafted leaf, policy top-8)+ action delta
               -> §10 Feature-family Oracle Probe の学習データ
  (c) WIDE   : PROBES の局面で policy rank 8..11 を 1 block x K=6 だけ追加評価
               -> 「良い手が top-8 の外にあるか」= §8/§9 recall

情報リーク禁止(§27): 相手の手札中身・山札順・determinization の真値・未来は保存しない。
history は「自分が実際に選んだ行動」だけを積む(相手の内部情報は使わない)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "value_net"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

import _entity_extract as EX  # noqa: E402
from _actionq_sampling import StratifiedQuota, cand_band, turn_band  # noqa: E402
from action_q import ActionQNet  # noqa: E402
from cg.api import SelectType  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
_FROZEN = _HERE / "value_net" / "frozen"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

TOP_MAIN, TOP_WIDE = 8, 12          # 教師 top-8 / recall 用に rank 8..11 を追加
NB, K, TIE = 4, 6, 0.005            # Phase11D/12 と同一仕様
HIST_N = 8

STATES: list[dict] = []
PROBES: list[dict] = []
STATS = Counter()
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "first": True,
        "hist": deque(maxlen=HIST_N), "state_rows": []}
QUOTA = None
MODELS: dict = {}
OPTS = {"max_states": 10 ** 9}


def load_q(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ActionQNet(ck["state_dim"], ck["option_dim"], use_cards=False,
                   use_action_card=ck.get("use_action_card", True))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], dtype=np.float32),
            "std": np.asarray(ck["std"], dtype=np.float32)}


@torch.no_grad()
def q_scores(b, sf, orows, cids, otypes):
    st = torch.from_numpy(((np.asarray(sf, dtype=np.float32) - b["mean"]) / b["std"])[None, :])
    of = torch.from_numpy(np.asarray(orows, dtype=np.float32)[None, ...])
    ci = torch.tensor([[max(0, c + 1) if c >= 0 else 0 for c in cids]], dtype=torch.long)
    ot = torch.tensor([[min(t, 63) for t in otypes]], dtype=torch.long)
    mk = torch.ones(1, len(orows), dtype=torch.bool)
    z = torch.zeros((1, 1), dtype=torch.long)
    q, _ = b["model"](st, [z, z, z], of, ci, ot, mk)
    return q[0].tolist()


def _teacher(obs, me, cand, policy_model, n_block, deltas_out=None):
    """cand(選択肢 index 列)について n_block x K の教師値。Phase11D/12 と同一設定。"""
    ev = leaf_eval_module.build_evaluator({"kind": "handcrafted"})
    cfg = {"opponent_depth": 1, "max_rollout_steps": 40}
    blocks = []
    for b in range(n_block):
        vals = {i: [] for i in cand}
        for _k in range(K):
            try:
                hs = search_adapter.to_search_begin_kwargs(
                    match_context.get_own_state(me), match_context.get_opponent_state(me), obs)
            except Exception:                      # noqa: BLE001
                continue
            if hs is None:
                continue
            try:
                root = P._begin(obs, hs)
            except Exception:                      # noqa: BLE001
                continue
            try:
                dl = time.perf_counter() + 30.0
                for i in cand:
                    try:
                        ch = P.cg_api.search_step(root.searchId, [i])
                    except ValueError:
                        continue
                    try:
                        if deltas_out is not None and i not in deltas_out:
                            stt = ch.observation.current
                            if stt is not None:
                                deltas_out[i] = EX.summarize(stt, me)
                        v = P._rollout_and_eval(ch, me, cfg, dl, ev, policy_model)
                        if v is not None:
                            vals[i].append(v)
                    finally:
                        try:
                            P.cg_api.search_release(ch.searchId)
                        except Exception:          # noqa: BLE001
                            pass
            finally:
                try:
                    P.cg_api.search_release(root.searchId)
                except Exception:                  # noqa: BLE001
                    pass
        if min((len(vals[i]) for i in cand), default=0) < K:
            try:
                P.cg_api.search_end()
            except Exception:                      # noqa: BLE001
                pass
            return None
        blocks.append([statistics.mean(vals[i][:K]) for i in cand])
    try:
        P.cg_api.search_end()
    except Exception:                              # noqa: BLE001
        pass
    return blocks


def _probe_state(obs, policy_model):
    select, state = obs.select, obs.current
    me = state.yourIndex
    try:
        ps = policy_model.score_options(obs, None, None)
    except Exception:                              # noqa: BLE001
        return
    if not ps or len(ps) != len(select.option):
        return
    ranked = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)
    cand = ranked[:TOP_MAIN]
    if len(cand) < 2:
        return
    try:
        sf = encoder.encode_state_from_state(state)
        orow = encoder.encode_options_from_state(state, select)
        cids = encoder.encode_option_card_ids(state, select)
        ent = EX.extract(state, me)
    except Exception:                              # noqa: BLE001
        STATS["encode_fail"] += 1
        return

    def pack(idxs):
        return ([[round(float(x), 6) for x in orow[i]] for i in idxs],
                [int(cids[i]) if cids[i] is not None else -1 for i in idxs],
                [int(getattr(select.option[i].type, "value", select.option[i].type))
                 for i in idxs])
    orows, cid, otp = pack(cand)
    q0 = q_scores(MODELS["Q0"], sf, orows, cid, otp)

    t = int(getattr(state, "turn", 0) or 0)
    row = {
        "sid": "s{}_{}".format(_ctx["game"], len(_ctx["state_rows"])),
        "game": _ctx["game"], "turn": t, "turn_band": turn_band(t),
        "arch": _ctx["opp"], "me_first": _ctx["first"],
        "n_legal": len(select.option), "n_cands": len(cand),
        "state_feat": [round(float(x), 6) for x in sf],
        "entity": ent,
        "history": list(_ctx["hist"]),
        "policy_top": cand, "policy_scores": [round(float(ps[i]), 6) for i in cand],
        "q0_scores": [round(v, 6) for v in q0],
        "cand_card_ids": cid, "cand_types": otp,
    }
    STATES.append(row)
    _ctx["state_rows"].append(row)
    STATS["states"] += 1

    if not QUOTA.accept(t, len(cand), _ctx["opp"], _ctx["in_game"]):
        return
    deltas: dict = {}
    blocks = _teacher(obs, me, cand, policy_model, NB, deltas_out=deltas)
    if blocks is None:
        STATS["teacher_fail"] += 1
        return
    wide = ranked[TOP_MAIN:TOP_WIDE]
    wide_blocks = _teacher(obs, me, wide, policy_model, 1) if wide else []
    if wide and wide_blocks is None:
        wide, wide_blocks = [], []
        STATS["wide_fail"] += 1

    pr = {
        "group_id": row["sid"], "game": _ctx["game"], "turn": t,
        "turn_band": row["turn_band"], "cand_band": cand_band(len(cand)),
        "arch": _ctx["opp"], "me_first": _ctx["first"], "n_legal": len(select.option),
        "state_feat": row["state_feat"], "entity": ent, "history": list(_ctx["hist"]),
        "candidates": [{
            "option_index": cand[n], "option_feat": orows[n], "action_card_id": cid[n],
            "option_type": otp[n], "policy_score": row["policy_scores"][n],
            "policy_rank": n,
            "blocks": [round(blocks[b][n], 6) for b in range(NB)],
            "after": [round(x, 4) for x in deltas[cand[n]]] if cand[n] in deltas else None,
        } for n in range(len(cand))],
        "before": [round(x, 4) for x in EX.summarize(state, me)],
        "wide": [{"option_index": wide[n], "policy_rank": TOP_MAIN + n,
                  "action_card_id": int(cids[wide[n]]) if cids[wide[n]] is not None else -1,
                  "block": round(wide_blocks[0][n], 6)} for n in range(len(wide))]
                if wide_blocks else [],
    }
    PROBES.append(pr)
    QUOTA.commit(t, len(cand), _ctx["opp"])
    _ctx["in_game"] += 1
    STATS["probes"] += 1


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        rec = (_ctx["recording"] and obs.current is not None
               and obs.current.yourIndex == _ctx["me"])
        sel = obs.select
        if (rec and sel is not None and sel.option and sel.type == SelectType.MAIN
                and sel.maxCount == 1 and len(sel.option) >= 2
                and STATS["states"] < OPTS["max_states"]):
            try:
                _probe_state(obs, ml_policy_agent._get_model(config))
            except Exception as exc:               # noqa: BLE001
                STATS["probe_err"] += 1
                print("[err] {}".format(exc), file=sys.stderr)
        out = orig(obs, config=config)
        # history: 自分が実際に選んだ行動だけを積む(§16)
        if rec and sel is not None and sel.option and out:
            try:
                i = out[0]
                o = sel.option[i]
                cid = encoder.encode_option_card_ids(obs.current, sel)
                _ctx["hist"].append({
                    "type": int(getattr(o.type, "value", o.type)),
                    "sel_type": int(getattr(sel.type, "value", sel.type)),
                    "card_id": int(cid[i]) if cid and cid[i] is not None else -1,
                    "turn": int(getattr(obs.current, "turn", 0) or 0)})
            except Exception:                      # noqa: BLE001
                pass
        return out

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-target", type=int, default=600)
    ap.add_argument("--max-states", type=int, default=20000)
    ap.add_argument("--max-games", type=int, default=90)
    ap.add_argument("--per-game-cap", type=int, default=6)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=80000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="ent")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.probe_target = max(1, args.probe_target // args.num_workers)
        args.max_states = max(1, args.max_states // args.num_workers)
    OPTS["max_states"] = args.max_states

    global QUOTA
    QUOTA = StratifiedQuota(args.probe_target, per_game_cap=args.per_game_cap,
                            cand_soft_cap=0.45)
    MODELS["Q0"] = load_q(_FROZEN / "Q0-expanded.pt")
    _install()

    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(cfg)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for gi in range(args.max_games):
        if QUOTA.total >= args.probe_target and STATS["states"] >= args.max_states:
            break
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch,
                    first=p0, state_rows=[])
        _ctx["hist"].clear()
        res = (runner.play_game(climb, opp, deck_c, deck_o) if p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        # 試合結果を、その試合の全 state へ後埋め(collision audit の outcome 比較用)
        won = None
        try:
            w = getattr(res, "winner", None)
            if w is None:
                w = getattr(res, "result_value", None)
            if w is not None and int(w) != -1:
                won = 1.0 if int(w) == _ctx["me"] else 0.0
        except Exception:                          # noqa: BLE001
            pass
        if won is None:
            STATS["outcome_missing"] += 1
        for r in _ctx["state_rows"]:
            r["game_outcome"] = won
        if gi % 5 == 0:
            print("  [w{}] game#{} states={} probes={}/{} {}min".format(
                args.worker_id, g, STATS["states"], QUOTA.total, args.probe_target,
                int((time.perf_counter() - t0) / 60)), file=sys.stderr, flush=True)

    for name, rows in (("states", STATES), ("probes", PROBES)):
        out = _HERE / "_ent_{}_{}.jsonl.gz".format(name, args.tag)
        with gzip.open(out, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "tag": args.tag, "n_states": len(STATES), "n_probes": len(PROBES),
        "stats": dict(STATS), "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "turn_band": dict(Counter(r["turn_band"] for r in PROBES)),
        "arch": dict(Counter(r["arch"] for r in PROBES))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
