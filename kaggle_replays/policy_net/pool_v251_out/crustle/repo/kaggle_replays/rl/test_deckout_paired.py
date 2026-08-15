"""デッキ切れ判定の近似と厳密判定を、同一試合内で対応ありで突き合わせる。

test_deckout_detect.py は近似側(collect_pool 経由)と厳密側(blunder_metrics 経由)を
**別々の試合セット**で集計するため、cg の乱数が seed で制御できない以上、率の差が
近似の誤差なのか試合のばらつきなのか区別できない(実際、敗北数が 49 対 59 とずれた)。

本スクリプトは1つのゲームループの中で両方を記録する:
  - 厳密: 試合終了時に負けた側の deckCount == 0 か
  - 近似: 学習側の最後の判断時点の self_deck_count が threshold 以下か
これで混同行列が作れる。
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pools  # noqa: E402

MAX_STEPS = 3000
_W: dict = {}


def _init(learner_w, opp_specs, deck_l):
    from ptcg_ai.learning.policy_model import PolicyModel
    from ptcg_ai.learning import encoder

    pm = PolicyModel(learner_w)
    if not pm.is_ready:
        raise RuntimeError(f"learner not ready: {learner_w}")
    _W["pm"] = pm
    _W["opps"] = []
    for name, w, deck in opp_specs:
        o = PolicyModel(w)
        if not o.is_ready:
            raise RuntimeError(f"opponent not ready: {name} {w}")
        _W["opps"].append((name, o, deck))
    _W["deck_l"] = deck_l
    _W["idx"] = encoder.FEATURE_NAMES.index("self_deck_count")


def _play(task):
    """1試合。戻り: (相手名, 学習側が負けたか, 厳密デッキ切れ, 最後の判断時点の自山札枚数)"""
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from ptcg_ai.learning import encoder

    opp_idx, learner_index, _seed = task
    pm = _W["pm"]
    name, opp, deck_o = _W["opps"][opp_idx]
    deck0, deck1 = (_W["deck_l"], deck_o) if learner_index == 0 else (deck_o, _W["deck_l"])

    last_deck = None
    lost = strict = False
    obs_dict, start = battle_start(deck0, deck1)
    if start.errorType != 0:
        return None
    try:
        for _ in range(MAX_STEPS):
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                return None
            if cur.result != -1:
                lost = cur.result != learner_index and cur.result in (0, 1)
                # 厳密: 負けた側の山札が 0
                if lost:
                    strict = cur.players[learner_index].deckCount == 0
                break
            sel = obs.select
            mine = cur.yourIndex == learner_index
            if mine and sel is not None and sel.option and sel.maxCount == 1:
                last_deck = encoder.encode_state_from_state(cur)[_W["idx"]]
            model = pm if mine else opp
            if sel is None or not sel.option:
                action = []
            elif sel.maxCount == 1:
                i = model.select_option(obs)
                action = [i if i is not None else 0]
            else:
                sc = model.score_options(obs)
                n = len(sel.option)
                c = max(sel.minCount, min(sel.maxCount, n))
                action = (sorted(range(n), key=lambda k: sc[k], reverse=True)[:c]
                          if sc else list(range(c)))
            obs_dict = battle_select(action)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        battle_finish()
    return (name, lost, strict, last_deck)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="alakazam")
    ap.add_argument("--opponents", default=pools.DEFAULT_POOL if hasattr(pools, "DEFAULT_POOL")
                    else "alakazam,crustle,marnie_grimmsnarl_ex,archaludon_ex")
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    names = [s.strip() for s in args.opponents.split(",") if s.strip()]
    opp_specs = pools.build_opponents(names)
    lw, ldeck_csv = pools.resolve_learner(args.learner)
    from run_league import read_deck_csv_file
    deck_l = read_deck_csv_file(ldeck_csv)

    k = len(opp_specs)
    per = (args.games // k) // 2 * 2
    tasks = [(i, g % 2, i * 100000 + g) for i in range(k) for g in range(per)]

    with Pool(args.workers, initializer=_init, initargs=(lw, opp_specs, deck_l)) as pool:
        res = [r for r in pool.map(_play, tasks) if r is not None]

    losses = [r for r in res if r[1]]
    print(f"有効 {len(res)} 試合 / 学習側の敗北 {len(losses)}")
    strict_n = sum(1 for r in losses if r[2])
    print(f"厳密デッキ切れ: {strict_n}/{len(losses)} = {strict_n/max(1,len(losses)):.3f}")
    print()
    print(f"{'閾値':>4} {'近似':>12} {'真陽性':>7} {'偽陽性':>7} {'偽陰性':>7} {'適合率':>7} {'再現率':>7}")
    for th in (0, 1, 2, 3):
        tp = sum(1 for r in losses if r[2] and r[3] is not None and r[3] <= th)
        fp = sum(1 for r in losses if not r[2] and r[3] is not None and r[3] <= th)
        fn = strict_n - tp
        approx = tp + fp
        prec = tp / approx if approx else float("nan")
        rec = tp / strict_n if strict_n else float("nan")
        print(f"{th:>4} {approx:>5}/{len(losses):<6} {tp:>7} {fp:>7} {fn:>7} {prec:>7.3f} {rec:>7.3f}")


if __name__ == "__main__":
    main()
