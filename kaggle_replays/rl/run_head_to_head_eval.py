"""T1(Transformer student)とv40教師の直接対戦head-to-head評価
(T1単独対戦評価 item4)。既存MLP経路・evaluate_pool()は変更しない。

両者とも明示的argmax(温度サンプリングなし)。T1の座席(先攻/後攻)を試合indexの
偶奇で均等化する(§9.1.1と同じ方式)。draw/timeout/errorを分けて報告する。
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

_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json"
_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_CKPT = (_ROOT / "kaggle_replays" / "rl" / "runs" / "distill_v40" / "train"
        / "stage100k_teacher_t1_seed1" / "best.pt")


def wilson_lo(w: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return float("nan")
    p = w / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (center - margin) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=str(_CKPT))
    args = ap.parse_args()
    ckpt_path = Path(args.checkpoint)

    import t1_live_agent as la
    import t1_eval as te
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    pm = PolicyModel(str(_TEACHER))
    if not pm.is_ready:
        raise SystemExit(f"教師を読み込めない: {_TEACHER}")
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    model, vocab, ckpt_profile, _ = la.load_t1_for_inference(ckpt_path, device="cpu")
    assert ckpt_profile == profile_name

    deck = read_deck_csv_file(str(_DECK))
    t0 = time.time()
    result = te.run_head_to_head(model, vocab, pm, pm, deck, deck, args.games, seed0=args.seed0,
                                 profile_name=profile_name, device="cpu")
    elapsed = time.time() - t0

    wr = result["t1_winrate"]
    ci_lo = wilson_lo(result["t1_wins"], result["n_valid"]) if result["n_valid"] else float("nan")
    print(f"[{elapsed:.0f}s] games={result['n_games']} valid={result['n_valid']} "
         f"errors={result['n_errors']} draws={result['n_draws']}")
    print(f"T1勝率={wr:.4f}({result['t1_wins']}/{result['n_valid']})  95%CI下限(Wilson)={ci_lo:.4f}")
    print(f"T1先攻={result['t1_first_games']} 後攻={result['t1_second_games']}")
    print(f"illegal_actions_total={result['illegal_actions_total']}")
    if result["error_types"]:
        print(f"エラー内訳: {result['error_types']}")

    out = {k: v for k, v in result.items() if k != "results"}
    out["winrate_ci95_lo"] = ci_lo
    out["seconds"] = round(elapsed, 1)
    out["checkpoint"] = str(ckpt_path)
    out["teacher"] = str(_TEACHER)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
