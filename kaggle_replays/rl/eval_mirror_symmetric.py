#!/usr/bin/env python3
"""2つの方策を役割バイアスなしで直接比較するミラー戦評価。

**背景（2026-08-11 発見）**: `eval_diagnostics.py` 経由のミラー戦（`collect_parallel._play_one`
を使う）は "learner役"（`--learner-weights` 側）が温度サンプリング、"opponent役"
（`--opponents` 側）が純粋greedyという非対称な実装（元はRL軌跡収集用の設計であり、A/B比較用
ではなかった）。処置群を常にlearner役に固定して評価した結果、役割バイアスだけでElo -56相当
（`marnie_grimmsnarl_ex` T1, seed=42, n=400 での実測）が混入していたことが判明した。
詳細は `kaggle_replays/docs/imitation/design-transformer-representation-2026-08-08.md` §8.3。

`collect_parallel.py` はチーム既存資産のため変更しない。本モジュールは新規に、
両陣営とも `PolicyModel.select_option()` / `score_options()`（greedy、サンプリングなし）に
統一したゲームループを提供し、モデルの中身だけを比較できるようにする。

使い方（ローカル、直列、workers省略時または1）:
    python eval_mirror_symmetric.py --model-a policy_weights_treatment.json \
        --model-b policy_weights_control.json --archetype marnie_grimmsnarl_ex --games 400

使い方（Kaggle等、multiprocessing.Pool並列。ローカル(Windows)のサンドボックス環境では
    Pool生成が WinError 5 (ハンドル複製の権限拒否) でデッドロックすることが確認されているため、
    workers>=2 はKaggle(Linux)側での実行を前提とする）:
    python eval_mirror_symmetric.py --model-a ... --model-b ... --archetype ... \
        --games 3500 --workers 4
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAX_STEPS = 4000

_W: dict = {}


def _init_worker(weights_a: str, weights_b: str, deck: list) -> None:
    from ptcg_ai.learning.policy_model import PolicyModel

    pm_a = PolicyModel(weights_a)
    pm_b = PolicyModel(weights_b)
    if not pm_a.is_ready:
        raise RuntimeError(f"model_a policy not ready (path?): {weights_a}")
    if not pm_b.is_ready:
        raise RuntimeError(f"model_b policy not ready (path?): {weights_b}")
    _W["pm_a"] = pm_a
    _W["pm_b"] = pm_b
    _W["deck"] = deck


def _play_one(task) -> dict:
    """task = (a_index, seed)。a_index がモデルAの席(0 or 1)。両陣営とも greedy で決定する。

    ``collect_parallel._play_one`` と同じ cg 呼び出し方(battle_start/battle_select/
    battle_finish)を踏襲しつつ、意思決定ロジックだけを対称にしたもの。
    """
    from cg.api import LogType, to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    a_index, seed = task
    pm_a = _W["pm_a"]
    pm_b = _W["pm_b"]
    deck = _W["deck"]

    obs_dict, start_data = battle_start(deck, deck)
    if start_data.errorType != 0:
        return {"reward": 0.0, "error": f"start {start_data.errorType}",
                "went_first": None, "final_turn": None}
    n = 0
    reward = 0.0
    error = None
    went_first = None
    final_turn = None
    try:
        while True:
            obs = to_observation_class(obs_dict)
            cur = obs.current
            if cur is None:
                error = "current None"; break
            if cur.result != -1:
                reward = 1.0 if cur.result == a_index else 0.0
                went_first = (cur.firstPlayer == a_index) if cur.firstPlayer is not None else None
                final_turn = cur.turn
                break
            if n >= MAX_STEPS:
                error = "max_steps"; break
            select = obs.select
            model = pm_a if cur.yourIndex == a_index else pm_b
            if select is None or not select.option:
                action = []
            elif select.maxCount == 1:
                oi = model.select_option(obs)
                action = [oi if oi is not None else 0]
            else:
                osc = model.score_options(obs)
                nn = len(select.option)
                count = max(select.minCount, min(select.maxCount, nn))
                action = (sorted(range(nn), key=lambda i: osc[i], reverse=True)[:count]
                          if osc else list(range(count)))
            obs_dict = battle_select(action)
            n += 1
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    finally:
        battle_finish()
    return {"reward": reward, "error": error, "went_first": went_first, "final_turn": final_turn}


def run(weights_a: str, weights_b: str, deck: list, n_games: int, workers: int, seed0: int = 10000):
    tasks = [(g % 2, seed0 + g) for g in range(n_games)]
    if workers <= 1:
        _init_worker(weights_a, weights_b, deck)
        results = [_play_one(t) for t in tasks]
    else:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(weights_a, weights_b, deck)) as pool:
            results = pool.map(_play_one, tasks, chunksize=max(1, n_games // (workers * 4)))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="モデルAの重みJSON(絶対パス推奨)")
    ap.add_argument("--model-b", required=True, help="モデルBの重みJSON(絶対パス推奨)")
    ap.add_argument("--archetype", required=True, help="デッキのアーキタイプ名(ミラー戦)")
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=1,
                    help="1(既定)=直列(ローカルWindowsサンドボックスでも動作確認済み)。"
                         "2以上はmultiprocessing.Pool(Kaggle等Linux推奨)")
    ap.add_argument("--seed0", type=int, default=10000)
    ap.add_argument(
        "--deck-csv", default=None,
        help="ミラー戦に使うデッキCSV(既定: archetype_decks/<archetype>/01.csv)。"
             "本番の提出デッキで測りたいときに指定する"
             "(alakazam をメインデッキに据えた 2026-08-11 以降は 06.csv = sample_submission/deck.csv)",
    )
    args = ap.parse_args()

    from run_league import read_deck_csv_file
    deck_csv = args.deck_csv or str(
        _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / args.archetype / "01.csv"
    )
    deck = read_deck_csv_file(deck_csv)

    print(f"model_a={args.model_a}")
    print(f"model_b={args.model_b}")
    print(f"archetype={args.archetype} deck_csv={deck_csv} games={args.games} workers={args.workers}\n")

    results = run(args.model_a, args.model_b, deck, args.games, args.workers, args.seed0)

    ok = [r for r in results if r.get("error") is None]
    print(f"有効試合 {len(ok)}/{len(results)}")
    errors = len(results) - len(ok)
    if errors:
        print(f"errors={errors}")
    if not ok:
        return

    wins_a = sum(1 for r in ok if r["reward"] >= 1.0)
    print(f"\nA勝率 {wins_a/len(ok):.4f} ({wins_a}/{len(ok)})")

    first = [r for r in ok if r.get("went_first") is True]
    second = [r for r in ok if r.get("went_first") is False]
    if first and second:
        wr_f = sum(1 for r in first if r["reward"] >= 1.0) / len(first)
        wr_s = sum(1 for r in second if r["reward"] >= 1.0) / len(second)
        print(f"  A先攻 {wr_f:.4f} (n={len(first)})  A後攻 {wr_s:.4f} (n={len(second)})")


if __name__ == "__main__":
    main()
