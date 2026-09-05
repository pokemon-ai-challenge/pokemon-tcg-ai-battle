"""Discovery Round 1 §23-§25: current BEST_KAGGLE(climb)の敗因を機械的に分類する。

新仮説は「面白そう」ではなく **高頻度 × 回復可能 × production 観測可能** な敗因から作る。
そのための素材を集める。production(abl_5_full + climb + Plan A)をそのまま回し、
実ラダーのメタシェアで相手を抽選する。

1試合につき記録するもの:
  終局      : primary / flags / prize margin / 手数
  自陣の軌跡: ターンごとの (残サイド, 相手残サイド, 自山, 相手山, ベンチ数, 場のエネ総数,
              手札枚数, アクティブHP割合)
  行動要約  : 攻撃した回数 / 進化した回数 / エネ付与回数 / サポート使用回数

これで「いつ・何が原因で負けたか」を後から分類できる。
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

# 実ラダーのメタシェア(2026-08-01 スクレイプ)。local FIELD の share ではなくこちらを使う。
LADDER = [("marnie_grimmsnarl_ex", 19.88), ("mega_lucario_ex", 14.86),
          ("crustle", 10.18), ("archaludon_ex", 9.63), ("dragapult_ex", 6.38),
          ("shirona_garchomp_ex", 2.44), ("rocket_mewtwo_ex", 2.31)]
MIRROR_SHARE = 20.96   # Alakazam ミラー
_S: dict = {}


def _init(policy_path: str):
    import agents
    import runner
    agents.ensure_production_cwd()
    from ptcg_ai.learning.policy_model import PolicyModel
    if not PolicyModel(policy_path).is_ready:
        raise RuntimeError("policy not ready")

    def mk(w):
        c = agents.load_config_copy("abl_5_full")
        c["policy_weights_path"] = w
        return agents.make_ml_policy_agent(c)
    wdir = _SUB / "ptcg_ai" / "learning"
    ddir = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
    _S["runner"] = runner
    _S["me_policy"] = policy_path
    _S["mk"] = mk
    _S["deck"] = runner.load_deck(_SUB / "deck.csv")
    _S["opps"] = {a: (mk(str(wdir / "policy_weights_{}.json".format(a))),
                      runner.load_deck(ddir / a / "01.csv")) for a, _ in LADDER}
    _S["opps"]["MIRROR"] = (mk(policy_path), list(_S["deck"]))


def _summ(state, me):
    try:
        mine, opp = state.players[me], state.players[1 - me]
        act = (mine.active or [None])[0]
        hp = (act.hp / act.maxHp) if act and act.maxHp else None
        ene = 0
        for slot in list(mine.active or []) + list(mine.bench or []):
            if slot is not None:
                ene += len(slot.energies or [])
        return {"t": int(getattr(state, "turn", -1)),
                "pz": len(mine.prize or []), "opz": len(opp.prize or []),
                "dk": int(getattr(mine, "deckCount", -1)),
                "odk": int(getattr(opp, "deckCount", -1)),
                "bn": sum(1 for s in (mine.bench or []) if s is not None),
                "en": ene, "hd": int(getattr(mine, "handCount", 0) or 0),
                "hp": round(hp, 3) if hp is not None else None}
    except Exception:                                          # noqa: BLE001
        return None


def _one(task):
    from cg.api import SelectType, OptionType
    gi, arch, first = task
    rn = _S["runner"]
    opp, deck_o = _S["opps"][arch]
    trace, acts = [], Counter()
    seen_turn = {"t": -1}

    def probe(obs):
        a = _S["me"](obs)
        st, sel = obs.current, obs.select
        if st is not None:
            me = st.yourIndex
            t = int(getattr(st, "turn", -1))
            if t != seen_turn["t"]:
                seen_turn["t"] = t
                s = _summ(st, me)
                if s:
                    trace.append(s)
            if sel is not None and sel.type == SelectType.MAIN and a and sel.option:
                try:
                    ot = sel.option[a[0]].type
                    acts[str(int(ot))] += 1
                except Exception:                              # noqa: BLE001
                    pass
        return a

    _S["me"] = _S["mk"](_S["me_policy"])
    random.seed(7_700_000 + gi)
    t0 = time.perf_counter()
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
            "flags": list(getattr(r, "flags", []) or []),
            "steps": getattr(r, "steps", None),
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "trace": trace, "main_actions": dict(acts)}


def main() -> None:
    wd = _SUB / "ptcg_ai" / "learning"
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(wd / "policy_weights_alakazam_rl_climb.json"))
    ap.add_argument("--games", type=int, default=420)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=11_223_344)
    ap.add_argument("--tag", default="climb")
    args = ap.parse_args()

    pool_names = [a for a, _ in LADDER] + ["MIRROR"]
    weights = [s for _, s in LADDER] + [MIRROR_SHARE]
    tot = sum(weights)
    cum, acc = [], 0.0
    for s in weights:
        acc += s
        cum.append(acc / tot)
    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.games):
        r = rng.random()
        tasks.append((i, pool_names[next(k for k, c in enumerate(cum) if r <= c)],
                      i % 2 == 0))

    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init, initargs=(args.policy,)) as pool:
        rows = pool.map(_one, tasks, chunksize=1)
    ok = [r for r in rows if r.get("win") is not None]
    out = {"policy": args.policy, "games": args.games, "valid": len(ok),
           "elapsed_s": round(time.perf_counter() - t0, 1),
           "winrate": round(sum(r["win"] for r in ok) / len(ok), 4) if ok else None,
           "by_arch": {}, "loss_primary": dict(Counter(
               r["primary"] for r in ok if r["win"] == 0.0))}
    for a in pool_names:
        s = [r for r in ok if r["arch"] == a]
        if s:
            out["by_arch"][a] = {"n": len(s),
                                 "wr": round(sum(x["win"] for x in s) / len(s), 4),
                                 "loss_primary": dict(Counter(
                                     x["primary"] for x in s if x["win"] == 0.0))}
    (_HERE / "_p1921_{}.json".format(args.tag)).write_text(
        json.dumps({"summary": out, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
