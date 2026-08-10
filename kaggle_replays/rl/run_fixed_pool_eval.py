"""T1(Transformer student)を、D1で使った固定相手8体プールに対して独立に評価する
(T1単独対戦評価 item5)。既存``evaluate_pool()``は変更しない(並存する新規スクリプト)。

D1(``select_teacher.py``)と同じ相手プール・同じデッキで、v40教師のときと同様に
T1を独立に評価する。**T1とv40は互いに対戦しない**(D1のv40の結果は
``results/teacher_selection.json``に既にあるので、それと比較する)。

主判定: 全体のT1勝率 − v40勝率の95%CI(下限が-5%を上回るか)。
補助指標: 相手ごとの勝率差(個別には非劣性判定に使わない、崩壊確認のみ)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission"),
          str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
        / "stage100k_teacher_t1_seed1" / "best.pt")
_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_LEARNER_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-per-opponent", type=int, default=125)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=str(_CKPT))
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint)

    import common as C
    import t1_live_agent as la
    import t1_eval as te
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    opponents = run_cfg["opponents"]

    model, vocab, profile_name, _ = la.load_t1_for_inference(ckpt_path, device="cpu")
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))

    # T1自身の状態エンコード専用(相手のprofileに関わらず常にfuudin_v4/v40を使う。
    # 相手(例: weights=Noneのalakazamは166次元profile)のprofileと食い違ってよい
    # ——相手のPolicyModelは相手自身の行動選択にしか使わないため)。
    t1_state_pm = PolicyModel(str(_TEACHER))
    if not t1_state_pm.is_ready:
        raise SystemExit(f"T1状態エンコード用のv40を読み込めない: {_TEACHER}")

    per_opponent = []
    tot_wins, tot_valid = 0, 0
    offset = 0
    t0 = time.time()
    for opp in opponents:
        weights = C.resolve_opponent_weights(_RUN_DIR, opp.get("weights"))
        deck_o_path = C.resolve_deck(opp.get("deck") or run_cfg["opponent_deck"])
        pm = PolicyModel(str(weights) if weights else None)
        if not pm.is_ready:
            raise SystemExit(f"相手モデルを読み込めない: {opp['id']}")
        deck_o = read_deck_csv_file(str(deck_o_path))

        result = te.run_head_to_head(model, vocab, t1_state_pm, pm, deck_t1, deck_o,
                                     args.games_per_opponent, seed0=args.seed0 + offset,
                                     profile_name=profile_name, device="cpu")
        offset += args.games_per_opponent
        tot_wins += result["t1_wins"]; tot_valid += result["n_valid"]
        per_opponent.append({
            "id": opp["id"], "wins": result["t1_wins"], "valid": result["n_valid"],
            "errors": result["n_errors"], "draws": result["n_draws"],
            "illegal_actions": result["illegal_actions_total"],
            "winrate": result["t1_winrate"],
        })
        print(f"{opp['id']}: T1勝率={result['t1_winrate']:.3f} "
             f"({result['t1_wins']}/{result['n_valid']}) errors={result['n_errors']}", flush=True)

    elapsed = time.time() - t0
    overall_winrate = tot_wins / tot_valid if tot_valid else float("nan")
    out = {
        "checkpoint": str(ckpt_path), "games_per_opponent": args.games_per_opponent,
        "seconds": round(elapsed, 1),
        "overall_wins": tot_wins, "overall_valid": tot_valid, "overall_winrate": overall_winrate,
        "per_opponent": per_opponent,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{elapsed:.0f}s] 全体T1勝率={overall_winrate:.4f}({tot_wins}/{tot_valid})")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
