"""collect_pool.parallel_collect_pool の健全性 + 正当性検証。

検証1: 相手が均等に当たる(各相手 games == 12)
検証2: 相手ごとに席が均等(seat0 == seat1 == 6) -- 過去の相手/席同期バグの回帰テスト
検証3: 学習側の decision のみ記録されている(logprob 再計算による実測確認、最重要)
検証4: 相手によって結果が違う(勝率を並べるだけ、有意差は主張しない)
検証5: エラーが 0
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_pool import parallel_collect_pool  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

N_GAMES = 48
WORKERS = 8
TEMPERATURE = 0.05
SEED0 = 7000


def _softmax_logprob(scores, temperature, chosen_idx):
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    probs = [e / z for e in exps]
    return math.log(max(probs[chosen_idx], 1e-12))


def main():
    deck_l = read_deck_csv_file(str(DECKDIR / "dragapult_ex" / "01.csv"))
    weights_l = str(WDIR / "policy_weights_dragapult_ex.json")

    opponents = [
        ("alakazam", None, read_deck_csv_file(str(DECKDIR / "alakazam" / "01.csv"))),
        ("crustle", str(WDIR / "policy_weights_crustle.json"),
         read_deck_csv_file(str(DECKDIR / "crustle" / "01.csv"))),
        ("marnie_grimmsnarl_ex", str(WDIR / "policy_weights_marnie_grimmsnarl_ex.json"),
         read_deck_csv_file(str(DECKDIR / "marnie_grimmsnarl_ex" / "01.csv"))),
        ("archaludon_ex", str(WDIR / "policy_weights_archaludon_ex.json"),
         read_deck_csv_file(str(DECKDIR / "archaludon_ex" / "01.csv"))),
    ]

    t0 = time.time()
    trajs, stats = parallel_collect_pool(
        weights_l, opponents, deck_l, n_games=N_GAMES, seed0=SEED0,
        temperature=TEMPERATURE, workers=WORKERS)
    dt = time.time() - t0

    print(f"収集時間: {dt:.1f}s  n_games={N_GAMES} workers={WORKERS} per={stats['per']} "
          f"dropped_games={stats['dropped_games']}")
    print(f"total: {stats['total']}")

    # ------------------------------------------------------------------
    # 検証1: 相手が均等に当たる
    # ------------------------------------------------------------------
    print("\n--- 検証1: 相手ごとの games ---")
    for name, bucket in stats["per_opponent"].items():
        print(f"  {name}: games={bucket['games']}")
        assert bucket["games"] == 12, f"{name} games={bucket['games']} != 12"
    print("検証1 PASS: 全相手 games == 12")

    # ------------------------------------------------------------------
    # 検証2: 相手ごとに席が均等(過去の同期バグの回帰テスト)
    # ------------------------------------------------------------------
    print("\n--- 検証2: 相手ごとの seat0/seat1 ---")
    for name, bucket in stats["per_opponent"].items():
        print(f"  {name}: seat0={bucket['seat0']} seat1={bucket['seat1']}")
        assert bucket["seat0"] == 6, f"{name} seat0={bucket['seat0']} != 6"
        assert bucket["seat1"] == 6, f"{name} seat1={bucket['seat1']} != 6"
    print("検証2 PASS: 全相手 seat0 == seat1 == 6")

    # ------------------------------------------------------------------
    # 検証3: 学習側の decision のみ記録されている(logprob 再計算による実測確認)
    # ------------------------------------------------------------------
    print("\n--- 検証3: logprob 再計算による記録内容の検証 ---")
    from ptcg_ai.learning.policy_model import PolicyModel

    pm_learner = PolicyModel(weights_l)
    assert pm_learner.is_ready, "learner PolicyModel 未ロード"
    pm_alakazam = PolicyModel(None)  # production alakazam(デフォルトパス)
    assert pm_alakazam.is_ready, "alakazam PolicyModel 未ロード"

    total_steps = 0
    match_learner = 0
    mismatch_learner = 0
    compared_vs_opp = 0
    match_learner_mismatch_opp = 0

    for r in trajs:
        is_alakazam = r["opponent"] == "alakazam"
        for s in r["steps"]:
            total_steps += 1
            sf = s["state_feat"]
            of = s["option_feats"]
            ci = s["card_ids"]
            chosen = s["chosen_idx"]
            recorded_logp = s["logprob"]

            learner_scores = [pm_learner._forward(sf, of[i], ci[i]) for i in range(len(of))]
            learner_logp = _softmax_logprob(learner_scores, TEMPERATURE, chosen)
            diff_learner = abs(learner_logp - recorded_logp)
            if diff_learner < 1e-6:
                match_learner += 1
            else:
                mismatch_learner += 1
                print(f"  MISMATCH vs learner: opponent={r['opponent']} "
                      f"recorded={recorded_logp:.6f} recomputed={learner_logp:.6f} diff={diff_learner:.2e}")

            if is_alakazam:
                compared_vs_opp += 1
                opp_scores = [pm_alakazam._forward(sf, of[i], ci[i]) for i in range(len(of))]
                opp_logp = _softmax_logprob(opp_scores, TEMPERATURE, chosen)
                diff_opp = abs(opp_logp - recorded_logp)
                if diff_learner < 1e-6 and diff_opp >= 1e-6:
                    match_learner_mismatch_opp += 1

    print(f"  total_steps={total_steps}")
    print(f"  match_learner={match_learner} mismatch_learner={mismatch_learner}")
    assert total_steps > 0, "steps=0"
    assert mismatch_learner == 0, f"mismatch_learner={mismatch_learner} (0 であるべき)"
    match_rate = match_learner / total_steps
    print(f"  learner 一致率: {match_learner}/{total_steps} = {match_rate:.4f}")

    print(f"  比較実行件数(vs alakazam, opponent==alakazamのstepのみ): compared_vs_opp={compared_vs_opp}")
    assert compared_vs_opp > 0, "比較が一件も実行されていない(比較対象 None のまま、という過去の誤報告パターン)"
    diverge_rate = match_learner_mismatch_opp / compared_vs_opp
    print(f"  learnerと一致しopponent(alakazam)とは不一致だったstep: "
          f"{match_learner_mismatch_opp}/{compared_vs_opp} = {diverge_rate:.4f}")
    assert diverge_rate >= 0.30, (
        f"diverge_rate={diverge_rate:.4f} < 0.30 (learner/opponentの出力が近すぎて検証が空振りしている疑い)"
    )
    print("検証3 PASS: 記録された全stepがlearnerのlogprobと一致、かつopponentとは十分な割合で不一致")

    # ------------------------------------------------------------------
    # 検証4: 相手によって結果が違う(数値を並べるだけ、有意差は主張しない)
    # ------------------------------------------------------------------
    print("\n--- 検証4: 相手ごとの勝率(12試合ずつ、統計的検定はしない) ---")
    for name, bucket in stats["per_opponent"].items():
        wr = bucket["wins"] / bucket["valid"] if bucket["valid"] else float("nan")
        print(f"  {name}: {bucket['wins']}/{bucket['valid']} = {wr:.3f}")

    # ------------------------------------------------------------------
    # 検証5: エラーが 0
    # ------------------------------------------------------------------
    print("\n--- 検証5: エラー件数 ---")
    print(f"  total errors={stats['total']['errors']}")
    assert stats["total"]["errors"] == 0, f"errors={stats['total']['errors']}"
    print("検証5 PASS: errors == 0")

    print(f"\n収集にかかった実時間(48試合): {dt:.1f}s")
    print("全検証 PASS")


if __name__ == "__main__":
    main()
