"""Phase19.14 Stage A: CARD decision census(subtype / genuine-choice / chain)。

production(abl_5_full + climb)をそのまま回し、`SelectType.CARD` の decision を全件記録する。
subtype は名前を先に決め打ちせず **engine の `SelectContext` を source of truth** にする(§7)。

genuine-choice 判定(§8):
  C0 trivial   : 候補1個以下、または minCount==len(option)(選ぶ余地なし)
  C1 dup-equiv : 重複除去後の実効候補が1個
                 - 手札/山札/トラッシュ/サイド等の「束」領域は同じ cardId なら等価
                 - 場(ACTIVE/BENCH/TOOL/ENERGY 等)は個体差があるので (area,index,...) で区別
  C2 genuine   : 実効候補が2個以上
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from cg.api import AreaType, SelectContext, SelectType  # noqa: E402
from ptcg_ai.learning import encoder as ENC  # noqa: E402
from ptcg_ai.ml_policy import ml_policy_agent as MA  # noqa: E402
from ptcg_ai.search import pipeline as P  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

# 「束」= 同じ cardId なら実質同一とみなせる領域(順序も個体差も見えない)
PILE_AREAS = {int(AreaType.HAND), int(AreaType.DECK), int(AreaType.DISCARD),
              int(AreaType.PRIZE), int(AreaType.LOOKING)}

ROWS: list[dict] = []
STATS: Counter = Counter()
_ctx: dict = {"game": 0, "arch": "?", "me": 0, "first": True, "rec": False,
              "idx": 0, "chain": 0, "chain_pos": 0, "prev_main": None}


def opt_key(o, state) -> tuple:
    """候補の同一性キー。

    `Option.cardId` は多くの場合 None なので、production と同じ
    `encoder._resolve_card_id`(area/index/toolIndex/energyIndex を辿る)で解決する。
    束領域(山/手札/トラッシュ/サイド/LOOKING)は card 同一なら等価。
    場(ACTIVE/BENCH 等)は個体差があるので位置まで含める(= 等価判定は保守的に狭く取り、
    genuine を過小評価しない)。
    """
    cid = ENC._resolve_card_id(o, state)
    a = int(o.area) if o.area is not None else -1
    if a in PILE_AREAS:
        return (int(o.type), a, cid)
    return (int(o.type), a, o.index, o.playerIndex, o.toolIndex, o.energyIndex, cid)


def classify(sel, state) -> tuple[str, int, list]:
    """§8 + 実データで判明した4類目。

    C3_unobservable: 候補は2個以上あるが **Observation から候補の識別情報が一切得られない**。
    実データではデッキサーチ(area=DECK)がこれで、engine は
    `area=DECK / index=<山札内index>` しか返さず cardId も `state.looking` も無い。
    候補は実際には別のカードなので terminal は変わりうるが、
    **合法な観測だけを使う限りどの方策・探索も区別できない**。
    """
    n = len(sel.option or [])
    ids = [ENC._resolve_card_id(o, state) for o in (sel.option or [])]
    if n <= 1 or sel.minCount >= n:
        return "C0_trivial", n, ids
    if all(i is None for i in ids):
        return "C3_unobservable", n, ids
    eff = len({opt_key(o, state) for o in sel.option})
    return ("C1_dup_equivalent" if eff <= 1 else "C2_genuine"), eff, ids


def make_probe(inner, cfg_full, policy_model):
    def probe(obs):
        sel = obs.select
        if sel is None or not _ctx["rec"]:
            return inner(obs)
        stt = obs.current
        is_card = (sel.type == SelectType.CARD)
        if sel.type == SelectType.MAIN:
            _ctx["chain"] += 1
            _ctx["chain_pos"] = 0
        t = time.perf_counter()
        act = inner(obs)
        ms = (time.perf_counter() - t) * 1000
        _ctx["idx"] += 1
        if sel.type == SelectType.MAIN and act:
            o = sel.option[act[0]] if act[0] < len(sel.option) else None
            _ctx["prev_main"] = {"type": int(o.type), "cardId": o.cardId} if o else None
        if not is_card or stt is None:
            STATS["non_card"] += 1
            return act
        _ctx["chain_pos"] += 1
        cls, eff, res_ids = classify(sel, stt)
        probs = scores = None
        try:
            deadline = time.perf_counter() + 5.0
            sc = policy_model.score_options(
                obs, MA._model_hidden_state_factory(obs, cfg_full), deadline)
            if sc and len(sc) == len(sel.option):
                scores = [round(float(x), 5) for x in sc]
                probs = P._softmax(sc)
        except Exception:                                     # noqa: BLE001
            STATS["policy_score_fail"] += 1
        sp = sorted(probs, reverse=True) if probs else None
        ROWS.append({
            "game": _ctx["game"], "arch": _ctx["arch"], "me": _ctx["me"],
            "first": _ctx["first"], "idx": _ctx["idx"],
            "turn": int(getattr(stt, "turn", -1)),
            "context": int(sel.context) if sel.context is not None else -1,
            "context_name": SelectContext(int(sel.context)).name
            if sel.context is not None and int(sel.context) in
            SelectContext._value2member_map_ else str(sel.context),
            "n_opt": len(sel.option), "min": sel.minCount, "max": sel.maxCount,
            "eff_opt": eff, "cls": cls,
            "card_ids": res_ids,
            "areas": [int(o.area) if o.area is not None else None for o in sel.option],
            "policy_scores": scores,
            "top1_prob": round(sp[0], 5) if sp else None,
            "margin": round(sp[0] - sp[1], 5) if sp and len(sp) > 1 else None,
            "selected": act[0] if act else None, "n_selected": len(act) if act else 0,
            "chain": _ctx["chain"], "chain_pos": _ctx["chain_pos"],
            "prev_main": _ctx["prev_main"], "ms": round(ms, 3)})
        STATS[cls] += 1
        return act
    return probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--offset", type=int, default=4100000)
    ap.add_argument("--opponents",
                    default="mega_lucario_ex,dragapult_ex,crustle,marnie_grimmsnarl_ex,"
                            "archaludon_ex,shirona_garchomp_ex")
    ap.add_argument("--tag", default="a0")
    args = ap.parse_args()

    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB
    climb = agents.make_ml_policy_agent(copy.deepcopy(cfg))
    policy_model = MA._get_model({"policy_weights_path": CLIMB})
    probe = make_probe(climb, cfg, policy_model)
    deck_c = runner.load_deck(_SUB / "deck.csv")
    opps = [o for o in args.opponents.split(",") if o]

    t0 = time.perf_counter()
    for gi in range(args.games):
        g = args.offset + args.worker_id + gi * args.num_workers
        arch = opps[g % len(opps)]
        cfg_o = agents.load_config_copy("abl_5_full")
        cfg_o["policy_weights_path"] = str(_WDIR / "policy_weights_{}.json".format(arch))
        opp = agents.make_ml_policy_agent(cfg_o)
        deck_o = runner.load_deck(_DECKDIR / arch / "01.csv")
        p0 = (gi % 2 == 0)
        _ctx.update(game=g, arch=arch, me=0 if p0 else 1, first=p0, rec=True,
                    idx=0, chain=0, chain_pos=0, prev_main=None)
        (runner.play_game(probe, opp, deck_c, deck_o) if p0
         else runner.play_game(opp, probe, deck_o, deck_c))
        _ctx["rec"] = False
        STATS["games"] += 1
        if gi % 5 == 0:
            print("  [w{}] game#{} card={} {}min".format(
                args.worker_id, g, len(ROWS), int((time.perf_counter() - t0) / 60)),
                file=sys.stderr, flush=True)

    out = _HERE / "_p1914_{}.jsonl.gz".format(args.tag)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in ROWS:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    print(json.dumps({"tag": args.tag, "card_rows": len(ROWS), "stats": dict(STATS)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
