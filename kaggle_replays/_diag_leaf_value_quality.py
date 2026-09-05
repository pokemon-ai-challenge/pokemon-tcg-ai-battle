"""葉評価器の品質比較: handcrafted vs learned value(climb v1.5 診断 §5.2)。

「学習 Value を本番探索の葉に入れる価値があるか」を、ゲームA/Bを回す前に**オフラインで**判定する。

方法:
  climb(abl_5_full 相当)を実際に対戦させ、**climb 自身の実局面**(自分の手番の decision state)で
  両評価器のスコアを記録し、その試合の**実際の勝敗**をラベルにして予測性能を比較する。
    - AUC(順位付け能力。探索は葉の大小関係しか使わないのでこれが主指標)
    - Brier / LogLoss(確率としての当たり具合)
    - ターン帯別 AUC(序盤は情報が無く当たらないのが正常。中終盤で効くかを見る)
    - キャリブレーション曲線(10分位)

方法上の注意(結論を書くときに落とさないこと):
  探索が実際に評価するのは「決定化された世界での数手先の葉」であって、ここで測る実局面とは
  分布が違う。実局面での優劣は葉での優劣を**保証しない**。ここは安価な事前スクリーニングであり、
  採否の最終判断は実ゲーム A/B(configs/climb_v15_leaf_a*.json)で行う。

production コードは import して呼ぶだけ。出力: _diag_leaf_value_quality_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "kaggle_replays" / "measurement"))

import agents  # noqa: E402
import runner  # noqa: E402

agents.ensure_production_cwd()

from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402
from ptcg_ai.search import leaf_eval as leaf_eval_module  # noqa: E402

_SUB = _ROOT / "sample_submission"
_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
CLIMB_WEIGHTS = str(_WDIR / "policy_weights_alakazam_rl_climb.json")

ROWS: list[dict] = []
_ctx = {"game": -1, "recording": False, "climb_index": 0}


def _install_probe() -> None:
    """climb の手番の decision state で両評価器のスコアを記録する(挙動は変えない)。"""
    orig = ml_policy_agent._select_action
    hand_ev = leaf_eval_module.HandcraftedEvaluator()
    val_ev = leaf_eval_module.ValueModelEvaluator()

    def select_action(obs, config=None):
        # climb 自身の手番の局面だけを記録する。両陣営が同一プロセスで _select_action を
        # 通るため、フィルタしないと **相手視点の勝率に climb の勝敗ラベルを付ける** ことになり
        # データの約半分が反転して AUC が 0.5 付近に潰れる(実測で確認した失敗)。
        if (_ctx["recording"] and obs.current is not None
                and obs.current.yourIndex == _ctx["climb_index"]):
            st = obs.current
            me = st.yourIndex
            try:
                ROWS.append({
                    "game": _ctx["game"],
                    "turn": int(getattr(st, "turn", 0) or 0),
                    "handcrafted": float(hand_ev.evaluate(st, me)),
                    "learned": float(val_ev.evaluate(st, me)),
                })
            except Exception:  # noqa: BLE001 - 診断が対戦を止めない
                pass
        return orig(obs, config=config)

    ml_policy_agent._select_action = select_action


# ---- 指標 ----

def _auc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney U による ROC-AUC(同値は 0.5 カウント)。"""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    n_p, n_n = len(pos), len(neg)
    return (rank_pos - n_p * (n_p + 1) / 2.0) / (n_p * n_n)


def _brier(scores, labels):
    return statistics.mean([(s - y) ** 2 for s, y in zip(scores, labels)])


def _logloss(scores, labels, eps=1e-12):
    return statistics.mean([
        -(y * math.log(max(s, eps)) + (1 - y) * math.log(max(1 - s, eps)))
        for s, y in zip(scores, labels)
    ])


def _calibration(scores, labels, bins=10):
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(s, y) for s, y in zip(scores, labels) if (lo <= s < hi or (b == bins - 1 and s == 1.0))]
        if not sel:
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
            continue
        out.append({
            "bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
            "mean_pred": round(statistics.mean([s for s, _ in sel]), 4),
            "actual": round(statistics.mean([y for _, y in sel]), 4),
        })
    return out


def _metrics(rows, key):
    s = [r[key] for r in rows]
    y = [r["label"] for r in rows]
    return {
        "n": len(rows),
        "auc": round(_auc(s, y), 4) if _auc(s, y) is not None else None,
        "brier": round(_brier(s, y), 4),
        "logloss": round(_logloss(s, y), 4),
        "mean_pred": round(statistics.mean(s), 4),
        "base_rate": round(statistics.mean(y), 4),
        "calibration": _calibration(s, y),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opponent", default="mega_lucario_ex")
    ap.add_argument("--out", default=str(_HERE / "_diag_leaf_value_quality_results.json"))
    args = ap.parse_args()

    _install_probe()

    cfg = agents.load_config_copy("abl_5_full")
    cfg["policy_weights_path"] = CLIMB_WEIGHTS
    climb = agents.make_ml_policy_agent(cfg)
    cfg_o = agents.load_config_copy("abl_5_full")
    cfg_o["policy_weights_path"] = str(_WDIR / f"policy_weights_{args.opponent}.json")
    opp = agents.make_ml_policy_agent(cfg_o)

    deck_c = runner.load_deck(_SUB / "deck.csv")
    deck_o = runner.load_deck(_DECKDIR / args.opponent / "01.csv")

    outcomes: dict[int, int] = {}
    for g in range(args.games):
        climb_p0 = (g % 2 == 0)
        _ctx["game"] = g
        _ctx["climb_index"] = 0 if climb_p0 else 1
        _ctx["recording"] = True
        res = (runner.play_game(climb, opp, deck_c, deck_o) if climb_p0
               else runner.play_game(opp, climb, deck_o, deck_c))
        _ctx["recording"] = False
        if res.error:
            print(f"[warn] game {g}: {res.error}", file=sys.stderr)
            outcomes[g] = -1
            continue
        me = 0 if climb_p0 else 1
        outcomes[g] = 1 if res.winner == me else 0
        print(f"  game {g+1}/{args.games} climb_won={outcomes[g]}", file=sys.stderr)

    rows = [dict(r, label=outcomes.get(r["game"], -1)) for r in ROWS]
    rows = [r for r in rows if r["label"] in (0, 1)]

    bands = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 10 ** 9)]
    out = {
        "note": "実局面での予測性能。探索の葉(決定化された未来)とは分布が違う点に注意。",
        "games": args.games,
        "opponent": args.opponent,
        "n_positions": len(rows),
        "climb_win_rate": round(statistics.mean([v for v in outcomes.values() if v >= 0]), 4)
        if outcomes else None,
        "overall": {
            "handcrafted": _metrics(rows, "handcrafted"),
            "learned": _metrics(rows, "learned"),
        },
        "by_turn_band": {
            name: {
                "n": len([r for r in rows if lo <= r["turn"] <= hi]),
                "handcrafted_auc": (
                    round(_auc([r["handcrafted"] for r in rows if lo <= r["turn"] <= hi],
                               [r["label"] for r in rows if lo <= r["turn"] <= hi]) or 0.0, 4)
                    if len([r for r in rows if lo <= r["turn"] <= hi]) > 10 else None
                ),
                "learned_auc": (
                    round(_auc([r["learned"] for r in rows if lo <= r["turn"] <= hi],
                               [r["label"] for r in rows if lo <= r["turn"] <= hi]) or 0.0, 4)
                    if len([r for r in rows if lo <= r["turn"] <= hi]) > 10 else None
                ),
            }
            for name, lo, hi in bands
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "overall"}, ensure_ascii=False, indent=2))
    for k in ("handcrafted", "learned"):
        m = out["overall"][k]
        print(f"{k:12s} AUC={m['auc']} Brier={m['brier']} LogLoss={m['logloss']} n={m['n']}")


if __name__ == "__main__":
    main()
