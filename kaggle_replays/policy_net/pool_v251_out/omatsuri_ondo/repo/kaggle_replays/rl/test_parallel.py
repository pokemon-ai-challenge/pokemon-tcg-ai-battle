"""並列収集の健全性 + 正当性検証。

low temperature(≒argmax)で並列収集した勝率が、単一プロセスの素policy baseline(~18-20%)と
一致することを確認する。並列パス全体(cg + encoder + pure-Python 方策)の妥当性チェック。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission"), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_parallel import parallel_collect  # noqa: E402
from run_league import read_deck_csv_file  # noqa: E402

WDIR = _ROOT / "sample_submission" / "ptcg_ai" / "learning"
DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"


def main():
    deck_l = read_deck_csv_file(str(DECKDIR / "dragapult_ex" / "01.csv"))
    deck_o = read_deck_csv_file(str(DECKDIR / "alakazam" / "01.csv"))
    weights = str(WDIR / "policy_weights_dragapult_ex.json")

    t0 = time.time()
    trajs, wins, valid, errors = parallel_collect(
        weights, None, deck_l, deck_o, n_games=24, seed0=5000,
        temperature=0.05, workers=4)
    dt = time.time() - t0
    wr = wins / valid if valid else float("nan")
    total_steps = sum(len(t["steps"]) for t in trajs)
    print(f"並列収集 24games/4workers: {dt:.1f}s  勝率(≒argmax) {wins}/{valid}={wr:.3f} "
          f"errors={errors} trajs={len(trajs)} steps={total_steps}")
    assert valid >= 20, f"valid={valid} 少なすぎ"
    assert errors == 0, f"errors={errors}"
    assert total_steps > 0, "steps=0"
    # baseline 18-20% 近傍(24試合なので広めに許容)
    assert 0.05 <= wr <= 0.45, f"勝率が baseline から乖離: {wr}"
    print("並列収集 PASS: 単一プロセス baseline と整合、全コア収集動作")


if __name__ == "__main__":
    main()
