"""Phase19.9: Turn-End Scorer(TES)評価データ。

TES(s,a) = 候補 a を強制 -> **現在の自分ターンの残り**を Frozen continuation policy で進める
           -> ターン終端 C1b の Frozen Value(途中決着なら terminal outcome)

同じ rollout から Immediate Value(IV = C1 の Value)も取れるので、
「行動直後で止める vs ターン終端まで進める」の因果比較(§12/§21)が同一 trajectory で可能。

H1 reference は **独立 seed family**(§16)で別途生成する。
continuation policy の中で TES を呼ばない(§30 再帰禁止)。
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import time
from collections import Counter
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
import entity_tokens as ET  # noqa: E402
import train_lh as L  # noqa: E402
import train_transition as TT  # noqa: E402
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
_META_DIR = _ROOT / "kaggle_replays" / "meta_analysis"
_FROZEN = _HERE / "value_net" / "frozen"
_CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")
# --field-preset の choices。"july7" は本ファイル既存の --opponents(現行挙動)専用で、
# train_league.FIELD_PRESETS には無いキーなので別扱いする(_c0_headroom.py の
# _FIELD_PRESET_CHOICES と同じ流儀)。
_FIELD_PRESET_CHOICES = ("july7", "g2", "mix")

TOP_K = 8
GROUPS: list[dict] = []
STATS = Counter()
TIMES: list[float] = []
_ctx = {"game": -1, "recording": False, "me": 0, "in_game": 0, "opp": "", "first": True}
QUOTA = None
MODELS: dict = {}
EVALS: dict = {}
OPTS = {"m": 8, "max_steps": 300, "tag": "x", "min_turn": 0}


def _resolve_path(raw: str) -> Path:
    """CLI で渡された相対パスを repo ルート基準で解決する(_c0_headroom.py と同じ手口)。

    `agents.ensure_production_cwd()` が import 時点で cwd を `sample_submission/` に
    変えているため、呼び出し側の cwd を基準にした素朴な相対パス解決はできない。
    絶対パス・既存の相対パス(cwd=sample_submission基準で解決できる場合)はそのまま使う。
    """
    path = Path(raw)
    if path.is_absolute() or path.exists():
        return path
    return _ROOT / path


def _resolve_field(preset: str, opponents_arg: str) -> list[tuple[str, float, str, str]]:
    """--field-preset を (arch, share, 重みファイルgen接尾辞, デッキdir名) のリストに解決する。

    "july7"(既定)は --opponents(既定6アーキ・接尾辞なし・archetype_decks、現行挙動)を
    そのまま使い、現行挙動を一切変えない。"g2"/"mix" は kaggle_replays/rl/train_league.py の
    FIELD_PRESETS を都度 import して使う(kaggle_replays/_c0_headroom.py の _resolve_field
    と同じ手口)。train_league.py 自体は変更しない。実行系(torch 等)が重いため、july7 では
    import しない。
    """
    if preset == "july7":
        opps = [o for o in opponents_arg.split(",") if o]
        return [(arch, 1.0, "", "archetype_decks") for arch in opps]
    _rl_dir = _ROOT / "kaggle_replays" / "rl"
    _league_dir = _ROOT / "league"
    for _p in (str(_rl_dir), str(_league_dir)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import train_league  # noqa: E402 - g2/mix 指定時のみの遅延import

    return list(train_league.FIELD_PRESETS[preset])


def _pick_opponent(
    field: list[tuple[str, float, str, str]], game_id: int, rng: random.Random,
) -> tuple[str, str, Path]:
    """フィールド定義から相手アーキを1体選び、(アーキ名, 重みパス, デッキパス) を返す。

    share が全て一様(="july7"の現行挙動)なら従来どおり game_id の巡回で決定論的に選ぶ
    (`opps[game_id % len(opps)]` と完全一致)。share にばらつきがあれば
    kaggle_replays/_c0_headroom.py の _pick_opponent と同じ累積share の重み付き抽出で選ぶ。
    """
    shares = [share for _arch, share, _gen, _deckdir in field]
    if len(set(shares)) <= 1:
        idx = game_id % len(field)
    else:
        total = sum(shares)
        draw = rng.random() * total
        acc = 0.0
        idx = len(field) - 1
        for i, share in enumerate(shares):
            acc += share
            if draw <= acc:
                idx = i
                break
    arch, _share, gen, deckdir = field[idx]
    weights_path = str(_WDIR / f"policy_weights_{arch}{gen}.json")
    deck_path = _META_DIR / deckdir / arch / "01.csv"
    return arch, weights_path, deck_path


def _load_pool(path, mode):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TT.PoolArm(ck["option_dim"], mode=mode)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], np.float32),
            "std": np.asarray(ck["std"], np.float32), "arm": "TD1"}


def _load_actionq(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ActionQNet(ck["state_dim"], ck["option_dim"], use_cards=False,
                   use_action_card=ck.get("use_action_card", True))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return {"model": m, "mean": np.asarray(ck["mean"], np.float32),
            "std": np.asarray(ck["std"], np.float32)}


@torch.no_grad()
def q0_scores(b, sf, orows, cids, otypes):
    st = torch.from_numpy(((np.asarray(sf, np.float32) - b["mean"]) / b["std"])[None, :])
    z = torch.zeros((1, 1), dtype=torch.long)
    q, _ = b["model"](st, [z, z, z],
                      torch.from_numpy(np.asarray(orows, np.float32)[None, ...]),
                      torch.tensor([[max(0, c) if 0 < c < 2048 else 0 for c in cids]],
                                   dtype=torch.long),
                      torch.tensor([[min(t, 63) for t in otypes]], dtype=torch.long),
                      torch.ones(1, len(orows), dtype=torch.bool))
    return q[0].tolist()


@torch.no_grad()
def entityq_scores(b, orows, cids, ent_before, ent_afters):
    tb = ET.tokenize(ent_before)
    pb = TT._pack([tb], [ET.relation_matrix(tb)])
    at, ar = [], []
    for ae in ent_afters:
        t = ET.tokenize(ae) if ae is not None else tb
        at.append(t)
        ar.append(ET.relation_matrix(t) if ae is not None else ET.relation_matrix(tb))
    pa = TT._pack(at, ar)
    C = len(orows)
    batch = {"opt": torch.from_numpy(np.asarray(orows, np.float32)[None, ...]),
             "card": torch.tensor([[c if 0 < c < ET.VOCAB else 0 for c in cids]],
                                  dtype=torch.long),
             "mask": torch.ones(1, C, dtype=torch.bool), "before": pb,
             "after": {k: v.view(1, C, *v.shape[1:]) for k, v in pa.items()}}
    return b["model"](batch)[0].tolist()


def rollout(obs, me, action, seed, policy_model, stop):
    """stop='C1b' で自分の現ターン終了まで / 'C4' で自分の次ターン終了まで。

    戻り値に C1(行動直後)の Value も入れる(IV 用)。
    """
    ev = EVALS["value"]
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(seed))
        root = P._begin(obs, hs)
    except Exception:                              # noqa: BLE001
        STATS["begin_fail"] += 1
        return None
    child = None
    try:
        try:
            child = P.cg_api.search_step(root.searchId, [action])
        except ValueError:
            STATS["illegal"] += 1
            return None
        node = child
        s0 = node.observation.current
        iv = None
        if s0 is not None:
            iv = (1.0 if s0.result == me else 0.0) if s0.result != -1 \
                else float(ev.evaluate(s0, me))
        need = 1 if stop == "C1b" else 2
        me_to_opp = 0
        prev = me
        steps = 0
        n_actions = 0
        val = None
        while steps < OPTS["max_steps"]:
            o = node.observation
            s = o.current
            if s is None:
                break
            if s.result != -1:
                val = 1.0 if s.result == me else (0.0 if s.result == 1 - me else 0.5)
                break
            actor = s.yourIndex
            if prev == me and actor != me:
                me_to_opp += 1
                if me_to_opp >= need:
                    val = float(ev.evaluate(s, me))
                    break
            if o.select is None or not o.select.option:
                break
            sel = P._greedy_selection(policy_model, o)
            if not sel:
                break
            if actor == me and me_to_opp == 0:
                n_actions += 1          # forced action 後、同一ターン内の後続行動数
            try:
                node = P.cg_api.search_step(node.searchId, sel)
            except ValueError:
                break
            prev = actor
            steps += 1
        if val is None:
            s = node.observation.current
            val = float(ev.evaluate(s, me)) if s is not None else None
        return {"v": val, "iv": iv, "steps": steps, "plan_len": n_actions}
    finally:
        for sid in (child.searchId if child is not None else None, root.searchId):
            if sid is not None:
                try:
                    P.cg_api.search_release(sid)
                except Exception:                  # noqa: BLE001
                    pass
        try:
            P.cg_api.search_end()
        except Exception:                          # noqa: BLE001
            pass


def _probe(obs, policy_model):
    select, state = obs.select, obs.current
    me = state.yourIndex
    try:
        ps = policy_model.score_options(obs, None, None)
    except Exception:                              # noqa: BLE001
        return
    if not ps or len(ps) != len(select.option):
        return
    ranked = sorted(range(len(ps)), key=lambda i: ps[i], reverse=True)
    cand = ranked[:TOP_K]
    if len(cand) < 2:
        return
    t = int(getattr(state, "turn", 0) or 0)
    if t < OPTS.get("min_turn", 0):
        STATS["min_turn_skip"] += 1
        return
    if not QUOTA.accept(t, len(cand), _ctx["opp"], _ctx["in_game"]):
        return
    try:
        sf = encoder.encode_state_from_state(state)
        orow = encoder.encode_options_from_state(state, select)
        cids = encoder.encode_option_card_ids(state, select)
        ent = EX.extract(state, me)
    except Exception:                              # noqa: BLE001
        return
    orows = [[round(float(x), 6) for x in orow[i]] for i in cand]
    cid = [int(cids[i]) if cids[i] is not None else -1 for i in cand]
    otp = [int(getattr(select.option[i].type, "value", select.option[i].type)) for i in cand]
    base = (hash((_ctx["game"], _ctx["in_game"], t)) & 0x3FFFFF) * 1000

    # ---- TES block A / B(独立 seed family)。同じ rollout から IV も取る ----
    tA, tB = {}, {}
    t0 = time.perf_counter()
    for i in cand:
        a = [rollout(obs, me, i, base + m, policy_model, "C1b") for m in range(OPTS["m"])]
        b = [rollout(obs, me, i, base + 100000 + m, policy_model, "C1b")
             for m in range(OPTS["m"])]
        a = [x for x in a if x and x["v"] is not None]
        b = [x for x in b if x and x["v"] is not None]
        if len(a) < OPTS["m"] or len(b) < OPTS["m"]:
            STATS["tes_fail"] += 1
            return
        tA[i], tB[i] = a, b
    tes_ms = (time.perf_counter() - t0) * 1000
    TIMES.append(tes_ms)

    # ---- H1 reference(さらに別 seed family。§16)----
    h1 = {}
    for i in cand:
        r = [rollout(obs, me, i, base + 500000 + m, policy_model, "C4")
             for m in range(OPTS["m"])]
        r = [x for x in r if x and x["v"] is not None]
        if len(r) < OPTS["m"] // 2:
            STATS["h1_fail"] += 1
            return
        h1[i] = r
    h1b = {}
    for i in cand:
        r = [rollout(obs, me, i, base + 800000 + m, policy_model, "C4")
             for m in range(OPTS["m"])]
        r = [x for x in r if x and x["v"] is not None]
        h1b[i] = r

    # ---- 学習済み scorer(after entity は 1 決定化から)----
    ent_afters = []
    try:
        hs = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me),
            obs, rng=random.Random(base + 999))
        root = P._begin(obs, hs)
        for i in cand:
            try:
                ch = P.cg_api.search_step(root.searchId, [i])
            except ValueError:
                ent_afters.append(None)
                continue
            try:
                stt = ch.observation.current
                ent_afters.append(EX.extract(stt, me) if stt is not None else None)
            finally:
                try:
                    P.cg_api.search_release(ch.searchId)
                except Exception:                  # noqa: BLE001
                    pass
        P.cg_api.search_release(root.searchId)
        P.cg_api.search_end()
    except Exception:                              # noqa: BLE001
        ent_afters = [None] * len(cand)

    g_tmp = {"state_feat": [float(x) for x in sf], "entity": ent,
             "candidates": [{"option_feat": orows[n], "action_card_id": cid[n],
                             "option_type": otp[n]} for n in range(len(cand))],
             "_yA": [0.0] * len(cand)}
    g_tmp["_tok"] = ET.tokenize(ent)
    g_tmp["_rel"] = ET.relation_matrix(g_tmp["_tok"])
    td1 = L.score_group(MODELS["TD1"]["model"], g_tmp, MODELS["TD1"]["mean"],
                        MODELS["TD1"]["std"], "TD1")
    q0 = q0_scores(MODELS["Q0"], sf, orows, cid, otp)
    eq = entityq_scores(MODELS["EQ"], orows, cid, ent, ent_afters)

    GROUPS.append({
        "group_id": "te{}_{}".format(_ctx["game"], _ctx["in_game"]),
        "game": _ctx["game"], "turn": t, "turn_band": turn_band(t),
        "cand_band": cand_band(len(cand)), "arch": _ctx["opp"],
        "me_first": _ctx["first"], "m": OPTS["m"], "tes_ms": round(tes_ms, 1),
        "state_feat": [round(float(x), 6) for x in sf], "entity": ent,
        "candidates": [{
            "option_index": cand[n], "option_feat": orows[n], "action_card_id": cid[n],
            "option_type": otp[n], "policy_rank": n,
            "q0": round(q0[n], 6), "entityq": round(eq[n], 6), "td1": round(td1[n], 6),
            "tes_A": [round(x["v"], 6) for x in tA[cand[n]]],
            "tes_B": [round(x["v"], 6) for x in tB[cand[n]]],
            "iv": [round(x["iv"], 6) for x in tA[cand[n]] if x["iv"] is not None],
            "plan_len": [x["plan_len"] for x in tA[cand[n]]],
            "h1_a": [round(x["v"], 6) for x in h1[cand[n]]],
            "h1_b": [round(x["v"], 6) for x in h1b.get(cand[n], [])],
        } for n in range(len(cand))]})
    QUOTA.commit(t, len(cand), _ctx["opp"])
    _ctx["in_game"] += 1
    STATS["groups"] += 1
    if len(GROUPS) % 10 == 0:
        _flush()


def _flush():
    out = _HERE / "_tes_{}.jsonl.gz".format(OPTS["tag"])
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in GROUPS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    tmp.replace(out)


def _install():
    orig = ml_policy_agent._select_action

    def sa(obs, config=None):
        if (_ctx["recording"] and obs.current is not None
                and obs.current.yourIndex == _ctx["me"] and obs.select is not None
                and obs.select.option and obs.select.type == SelectType.MAIN
                and obs.select.maxCount == 1 and len(obs.select.option) >= 2
                and QUOTA.total < QUOTA.target):
            try:
                _probe(obs, ml_policy_agent._get_model(config))
            except Exception as exc:               # noqa: BLE001
                STATS["probe_err"] += 1
                print("[err]", exc, file=sys.stderr)
        return orig(obs, config=config)

    ml_policy_agent._select_action = sa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=350)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--per-game-cap", type=int, default=3)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=1500000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex",
                    help="--field-preset july7(既定)のときだけ使う相手アーキ巡回リスト(現行挙動)。"
                         "g2/mixでは無視され、train_league.FIELD_PRESETSのshareで選ばれる。")
    ap.add_argument("--tag", default="w0",
                    help="出力ファイル識別子(_tes_<tag>.jsonl.gz)。並列workerや別デッキで走らせる"
                         "ときは既存ファイルと衝突しない値を明示すること(例: w0/w1.../oger_w0)。")
    ap.add_argument("--min-turn", type=int, default=0,
                    help="このターン未満のrootは収集しない(turn帯均衡サンプリング用。既定0=従来挙動)。")
    ap.add_argument("--deck", default=None,
                    help="自分側デッキCSVパス(省略時=sample_submission/deck.csv、現行挙動維持)。")
    ap.add_argument("--weights", default=None,
                    help="自分側 policy 重みJSONパス(省略時=climb重み、現行挙動維持)。"
                         "continuation policy(自ターンを進める Frozen policy)にも同じ重みが使われる"
                         "(_get_model(config) 経由で _install() の wrapper が同一 config を渡すため)。")
    ap.add_argument("--field-preset", default="july7", choices=_FIELD_PRESET_CHOICES,
                    help="相手フィールド。july7=--opponentsの巡回(既定・現行挙動維持)、"
                         "g2/mix=kaggle_replays/rl/train_league.FIELD_PRESETSをshare比例抽出で使う。")
    args = ap.parse_args()
    if args.num_workers > 1:
        args.target = max(1, args.target // args.num_workers)
    OPTS.update(m=args.m, tag=args.tag, min_turn=args.min_turn)
    torch.set_num_threads(1)
    L.HSTAR[0] = "H1"

    global QUOTA
    QUOTA = StratifiedQuota(args.target, per_game_cap=args.per_game_cap, cand_soft_cap=0.45)
    EVALS["value"] = leaf_eval_module.build_evaluator({"kind": "value"})
    MODELS["TD1"] = _load_pool(_FROZEN / "TD1_N2499.pt", "current")
    MODELS["EQ"] = _load_pool(_FROZEN / "EntityQ_EPool.pt", "transition")
    MODELS["Q0"] = _load_actionq(_FROZEN / "Q0-expanded.pt")
    _install()

    own_weights = str(_resolve_path(args.weights)) if args.weights else _CLIMB_WEIGHTS
    cfg = agents.load_config_copy("climb_baseline")
    cfg["policy_weights_path"] = own_weights
    climb = agents.make_ml_policy_agent(cfg)
    own_deck_path = _resolve_path(args.deck) if args.deck else (_SUB / "deck.csv")
    deck_c = runner.load_deck(own_deck_path)
    field = _resolve_field(args.field_preset, args.opponents)
    field_rng = random.Random(0xC0FFEE ^ args.worker_id)

    out_path = _HERE / "_tes_{}.jsonl.gz".format(args.tag)
    if out_path.exists():
        print("[warn] output already exists and will be overwritten by flush: {}".format(out_path),
              file=sys.stderr)

    t0 = time.perf_counter()
    for gi in range(args.max_games):
        if QUOTA.total >= args.target:
            break
        g = args.offset + args.worker_id + gi * args.num_workers
        arch, opp_weights_path, opp_deck_path = _pick_opponent(field, g, field_rng)
        cfg_o = agents.load_config_copy("climb_baseline")
        cfg_o["policy_weights_path"] = opp_weights_path
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(opp_deck_path)
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, recording=True, me=0 if p0 else 1, in_game=0, opp=arch, first=p0)
        (runner.play_game(climb, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        if gi % 3 == 0:
            print("  [w{}] game#{} groups={}/{} {}min".format(
                args.worker_id, g, QUOTA.total, args.target,
                int((time.perf_counter() - t0) / 60)), file=sys.stderr, flush=True)

    _flush()
    print(json.dumps({"tag": args.tag, "n_groups": len(GROUPS), "stats": dict(STATS),
                      "tes_ms_mean": round(statistics.mean(TIMES), 1) if TIMES else None,
                      "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
                      "arch": dict(Counter(r["arch"] for r in GROUPS)),
                      "turn_band": dict(Counter(r["turn_band"] for r in GROUPS))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
