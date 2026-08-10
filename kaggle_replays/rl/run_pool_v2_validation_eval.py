"""pool_v2_validation評価(item6)。``pool_v2_validation_manifest.json``で事前固定した6相手に対し、
checkpoint結果を見る前に固定した条件(相手順・seed・座席)で評価する。
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
_LEARNER_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"
_MANIFEST = _HERE / "pool_v2_validation_manifest.json"


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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import t1_live_agent as la
    import t1_eval as te
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    games_per_opp = manifest["games_per_opponent"]
    seed0 = manifest["eval_seed0"]

    # load_t1_for_inferenceがPPO checkpoint形式(feature_profile等のメタ情報を持たない)を
    # 自動判別し、蒸留checkpointからメタ情報を借りて処理する(t1_live_agent.py参照)。
    model, vocab, profile_name, _ = la.load_t1_for_inference(Path(args.checkpoint), device="cpu")
    t1_state_pm = PolicyModel(str(_TEACHER))
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))

    per_opponent = []
    tot_w = tot_v = 0
    offset = 0
    t0 = time.time()
    for opp in manifest["opponents"]:
        pm = PolicyModel(opp["weights"])  # null -> production既定
        deck_o = read_deck_csv_file(str(_ROOT / opp["deck"]) if not Path(opp["deck"]).is_absolute()
                                    else opp["deck"])
        result = te.run_head_to_head(model, vocab, t1_state_pm, pm, deck_t1, deck_o,
                                     games_per_opp, seed0=seed0 + offset,
                                     profile_name=profile_name, device="cpu")
        offset += games_per_opp
        tot_w += result["t1_wins"]; tot_v += result["n_valid"]
        lo = wilson_lo(result["t1_wins"], result["n_valid"])
        per_opponent.append({"id": opp["id"], "wins": result["t1_wins"], "valid": result["n_valid"],
                             "errors": result["n_errors"], "draws": result["n_draws"],
                             "illegal_actions": result["illegal_actions_total"],
                             "winrate": result["t1_winrate"], "winrate_ci95_lo": lo})
        print(f"{opp['id']}: {result['t1_winrate']:.3f} ({result['t1_wins']}/{result['n_valid']}) "
             f"CIlo={lo:.3f} err={result['n_errors']} illegal={result['illegal_actions_total']}",
             flush=True)

    elapsed = time.time() - t0
    overall_winrate = tot_w / tot_v if tot_v else float("nan")
    winrates = [o["winrate"] for o in per_opponent]
    worst = min(per_opponent, key=lambda o: o["winrate"])
    out = {
        "label": args.label, "checkpoint": args.checkpoint, "seconds": round(elapsed, 1),
        "overall_wins": tot_w, "overall_valid": tot_v, "overall_winrate": overall_winrate,
        "overall_winrate_ci95_lo": wilson_lo(tot_w, tot_v),
        "per_opponent": per_opponent,
        "worst_case_opponent": worst["id"], "worst_case_winrate": worst["winrate"],
        "winrate_variance_across_opponents": (sum((w - overall_winrate) ** 2 for w in winrates) / len(winrates)
                                              if winrates else float("nan")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{elapsed:.0f}s] 全体={overall_winrate:.4f}({tot_w}/{tot_v}) worst={worst['id']}={worst['winrate']:.3f}")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
