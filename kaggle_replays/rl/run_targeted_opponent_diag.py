"""rocket_mewtwo_exとalakazamに絞った、seed間比較用の追加診断(各150試合)。
run_fixed_pool_eval.pyと同じロジックを、対象opponentだけに絞って実行する。
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
_RUN_DIR = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1"
_RUN_JSON = _RUN_DIR / "run.json"
_TARGET_IDS = ("rocket_mewtwo_ex", "alakazam")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games-per-opponent", type=int, default=150)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import common as C
    import t1_live_agent as la
    import t1_eval as te
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    run_cfg = json.loads(_RUN_JSON.read_text(encoding="utf-8"))
    opponents = [o for o in run_cfg["opponents"] if o["id"] in _TARGET_IDS]

    model, vocab, profile_name, _ = la.load_t1_for_inference(Path(args.checkpoint), device="cpu")
    t1_state_pm = PolicyModel(str(_TEACHER))
    deck_t1 = read_deck_csv_file(str(_LEARNER_DECK))

    per_opponent = []
    t0 = time.time()
    offset = 0
    for opp in opponents:
        weights = C.resolve_opponent_weights(_RUN_DIR, opp.get("weights"))
        deck_o_path = C.resolve_deck(opp.get("deck") or run_cfg["opponent_deck"])
        pm = PolicyModel(str(weights) if weights else None)
        deck_o = read_deck_csv_file(str(deck_o_path))
        result = te.run_head_to_head(model, vocab, t1_state_pm, pm, deck_t1, deck_o,
                                     args.games_per_opponent, seed0=args.seed0 + offset,
                                     profile_name=profile_name, device="cpu")
        offset += args.games_per_opponent
        per_opponent.append({"id": opp["id"], "wins": result["t1_wins"], "valid": result["n_valid"],
                             "errors": result["n_errors"], "winrate": result["t1_winrate"]})
        print(f"{opp['id']}: {result['t1_winrate']:.3f} ({result['t1_wins']}/{result['n_valid']}) "
             f"errors={result['n_errors']}", flush=True)

    out = {"checkpoint": args.checkpoint, "seconds": round(time.time() - t0, 1),
          "per_opponent": per_opponent}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
