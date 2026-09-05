"""Discovery Round1 の判別 probe: 序盤の board 未展開は **デッキ起因か方策起因か**。

敗因分析(_p1921)で、no_pokemon 敗北(全敗北の27.7%)は
  bench が turn3 で 1.39(勝ち 2.46)、盤面エネが 0.7 前後で頭打ち、平均75手で終局
という「立ち上がらないまま負ける」型だと分かった。

ここで切り分ける:
  A) デッキ起因 = 展開札(Buddy-Buddy Poffin 等)や basic がそもそも手札に来ていない
  B) 方策起因   = 手札にあるのに序盤に打っていない

turn<=4 の自分の decision で、
  - 手札にある展開札(Poffin/Rare Candy/Night Stretcher/Poké Pad)の枚数
  - それらが選択肢として提示された回数 / 実際に選ばれた回数
  - 手札の basic ポケモン枚数、ベンチ空き
を記録し、最終的に「その試合が no_pokemon 負けだったか」と突き合わせる。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

LADDER = [("marnie_grimmsnarl_ex", 19.88), ("mega_lucario_ex", 14.86),
          ("crustle", 10.18), ("archaludon_ex", 9.63), ("dragapult_ex", 6.38),
          ("shirona_garchomp_ex", 2.44), ("rocket_mewtwo_ex", 2.31)]
MIRROR_SHARE = 20.96
DEV = {1086: "poffin", 1079: "rare_candy", 1097: "night_stretcher", 1152: "poke_pad"}
BASICS = {741, 65, 343}          # Abra / Dunsparce / Shaymin
_S: dict = {}


def _init(policy_path: str):
    import agents
    import runner
    agents.ensure_production_cwd()
    _S["runner"] = runner
    _S["policy"] = policy_path

    def mk(w):
        c = agents.load_config_copy("abl_5_full")
        c["policy_weights_path"] = w
        return agents.make_ml_policy_agent(c)
    _S["mk"] = mk
    wdir = _SUB / "ptcg_ai" / "learning"
    ddir = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
    _S["deck"] = runner.load_deck(_SUB / "deck.csv")
    _S["opps"] = {a: (mk(str(wdir / "policy_weights_{}.json".format(a))),
                      runner.load_deck(ddir / a / "01.csv")) for a, _ in LADDER}
    _S["opps"]["MIRROR"] = (mk(policy_path), list(_S["deck"]))


def _one(task):
    from cg.api import SelectType
    from ptcg_ai.learning import encoder as ENC
    gi, arch, first = task
    rn = _S["runner"]
    opp, deck_o = _S["opps"][arch]
    st_ = Counter()
    hand_seen = {k: 0 for k in DEV}
    turns_seen = set()

    def probe(obs):
        a = _S["me"](obs)
        s, sel = obs.current, obs.select
        if s is None or sel is None:
            return a
        me = s.yourIndex
        t = int(getattr(s, "turn", 99))
        if t <= 4:
            mine = s.players[me]
            hand = mine.hand or []
            if t not in turns_seen:
                turns_seen.add(t)
                st_["basics_in_hand_t%d" % t] = sum(1 for c in hand if c.id in BASICS)
                st_["bench_t%d" % t] = sum(1 for x in (mine.bench or []) if x is not None)
                for cid, nm in DEV.items():
                    n = sum(1 for c in hand if c.id == cid)
                    if n:
                        hand_seen[cid] = max(hand_seen[cid], n)
                        st_["hand_" + nm] += 1
            if sel.type == SelectType.MAIN and sel.option:
                try:
                    ids = [ENC._resolve_card_id(o, s) for o in sel.option]
                except Exception:                              # noqa: BLE001
                    return a
                for cid, nm in DEV.items():
                    if cid in ids:
                        st_["offered_" + nm] += 1
                        if a and a[0] < len(ids) and ids[a[0]] == cid:
                            st_["played_" + nm] += 1
        return a

    _S["me"] = _S["mk"](_S["policy"])
    random.seed(8_800_000 + gi)
    try:
        r = (rn.play_game(probe, opp, list(_S["deck"]), list(deck_o)) if first
             else rn.play_game(opp, probe, list(deck_o), list(_S["deck"])))
    except Exception as exc:                                   # noqa: BLE001
        return {"game": gi, "arch": arch, "error": type(exc).__name__}
    my = 0 if first else 1
    w = getattr(r, "winner", None)
    return {"game": gi, "arch": arch, "first": first,
            "win": None if w is None else (1.0 if w == my else 0.0),
            "primary": str(getattr(r, "primary", None)
                           or getattr(r, "primary_win_condition", None)),
            "steps": getattr(r, "steps", None), "early": dict(st_)}


def main() -> None:
    wd = _SUB / "ptcg_ai" / "learning"
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(wd / "policy_weights_alakazam_rl_climb.json"))
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=55_667_788)
    ap.add_argument("--tag", default="open")
    args = ap.parse_args()
    names = [a for a, _ in LADDER] + ["MIRROR"]
    wts = [s for _, s in LADDER] + [MIRROR_SHARE]
    tot = sum(wts)
    cum, acc = [], 0.0
    for s in wts:
        acc += s
        cum.append(acc / tot)
    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.games):
        r = rng.random()
        tasks.append((i, names[next(k for k, c in enumerate(cum) if r <= c)], i % 2 == 0))
    with Pool(args.workers, initializer=_init, initargs=(args.policy,)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    (_HERE / "_p1922_{}.json".format(args.tag)).write_text(
        json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    ok = [r for r in rows if r.get("win") is not None]
    print(json.dumps({"games": len(ok),
                      "winrate": round(sum(r["win"] for r in ok) / len(ok), 4),
                      "loss_modes": dict(Counter(r["primary"] for r in ok
                                                 if r["win"] == 0.0))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
