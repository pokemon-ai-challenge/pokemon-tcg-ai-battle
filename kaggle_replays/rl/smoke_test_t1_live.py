"""T1ライブ推論ラッパーの20試合smoke test(T1単独対戦評価 item3)。

crash・timeout・illegal actionが0件であることと、推論時間を確認する診断スクリプト
(pytestの回帰テストではなく、手動実行して結果を目視確認するためのもの)。
"""

from __future__ import annotations

import sys
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


def main(n_games: int = 20) -> None:
    import t1_live_agent as la
    import t1_eval as te
    from ptcg_ai.learning.policy_model import PolicyModel
    from run_league import read_deck_csv_file

    pm = PolicyModel(str(_TEACHER))
    if not pm.is_ready:
        print(f"教師を読み込めない: {_TEACHER}"); return
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    model, vocab, ckpt_profile, _ = la.load_t1_for_inference(_CKPT, device="cpu")
    assert ckpt_profile == profile_name

    deck = read_deck_csv_file(str(_DECK))
    result = te.run_head_to_head(model, vocab, pm, pm, deck, deck, n_games, seed0=90000,
                                 profile_name=profile_name, device="cpu")

    print(f"試合数={result['n_games']} valid={result['n_valid']} errors={result['n_errors']} "
         f"draws={result['n_draws']}")
    print(f"T1勝率={result['t1_winrate']:.3f} ({result['t1_wins']}/{result['n_valid']})  "
         f"T1先攻={result['t1_first_games']} 後攻={result['t1_second_games']}")
    print(f"illegal_actions_total={result['illegal_actions_total']}")
    print(f"推論回数={result['n_inferences']}  平均推論時間={result['inference_time_mean']*1000:.2f}ms  "
         f"最大={result['inference_time_max']*1000:.2f}ms")
    if result["error_types"]:
        print(f"エラー内訳: {result['error_types']}")

    assert result["n_errors"] == 0, "crash/timeoutが発生した"
    assert result["illegal_actions_total"] == 0, "illegal actionが発生した"
    print("OK: crash/timeout/illegal actionは0件。")


if __name__ == "__main__":
    main()
