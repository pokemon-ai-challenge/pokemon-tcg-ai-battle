#!/usr/bin/env python3
"""リモート実行バンドルが動くかを確認する。本番実行の前に必ず通すこと。

`package_for_remote.py` がこのファイルをバンドル直下に `smoke_test.py` として同梱する。
バンドル直下から実行される前提でパスを組む（`ROOT = このファイルの親`）。

確認するのは以下。どれか1つでも落ちたら本番を回してはいけない。

  1. ネイティブライブラリ(libcg.so / cg.dll)がロードでき、対局が最後まで進むか
  2. 学習済み重みが読めるか
  3. 自己対戦のログ収集(マルチプロセス)が動くか
  4. 利用可能なCPU数と1試合あたりの実測時間

## ``if __name__ == "__main__"`` ガードは必須

Windows のプロセス生成方式(spawn)では子プロセスがメインモジュールを再importする。
ガードが無いとスモークテスト全体が子プロセスでも走り、``BrokenProcessPool`` で落ちる。
Linux(fork)では起きないため「外部では動くがローカルでは落ちる」という紛らわしい
壊れ方をする。実際にこれを踏んだので消さないこと。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SS = ROOT / "sample_submission"


def main() -> None:
    if not SS.is_dir():
        raise SystemExit(
            f"sample_submission が見つかりません: {SS}\n"
            "このスクリプトはバンドルを展開した直下から実行してください。"
        )
    sys.path.insert(0, str(SS))
    sys.path.insert(0, str(ROOT / "kaggle_replays" / "policy_prior"))
    os.chdir(SS)

    print(f"Python {sys.version.split()[0]} / CPU {os.cpu_count()} 論理コア")

    # --- 1. ネイティブライブラリと対局 -------------------------------------
    t0 = time.time()
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start
    from main import agent, read_deck_csv

    deck = read_deck_csv()
    obs_dict, start = battle_start(deck, deck)
    assert start.errorType == 0, f"battle_start errorType={start.errorType}"
    steps = 0
    while True:
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            break
        obs_dict = battle_select(agent(obs_dict))
        steps += 1
        if steps > 2000:
            break
    battle_finish()
    print(f"[OK] 対局が完走: {steps}手 / {time.time() - t0:.1f}秒")

    # --- 2. 重み -----------------------------------------------------------
    from ptcg_ai.learning import policy_model

    assert policy_model.PolicyModel().is_ready, "汎用モデルが読めません"
    print("[OK] 汎用モデルを読めました")

    weights_dir = ROOT / "kaggle_replays" / "policy_prior" / "output"
    missing = [
        n for n in (1, 2, 3, 4)
        if not policy_model.PolicyModel(
            weights_dir / f"policy_weights_archetype{n}.json"
        ).is_ready
    ]
    if missing:
        raise SystemExit(
            f"アーキタイプ専用モデルが読めません: {missing}\n"
            "これが無いと相手プールがルールベースになり、評価の意味が変わります。"
        )
    print("[OK] アーキタイプ専用モデル4種すべて読めました")

    # --- 3. 自己対戦の収集（最も環境依存が大きい） --------------------------
    from rl.selfplay import collect_selfplay

    games = 4
    t0 = time.time()
    episodes = collect_selfplay(
        weights_path=SS / "ptcg_ai" / "learning" / "policy_weights.json",
        games=games,
        temperature=1.0,
        workers=2,
        seed=1,
    )
    elapsed = time.time() - t0
    n_decisions = sum(len(ep["decisions"]) for ep in episodes)
    assert episodes and n_decisions > 0, "自己対戦から decision が1件も取れていません"
    print(f"[OK] 自己対戦: エピソード{len(episodes)}件 / decision {n_decisions} / {elapsed:.1f}秒")
    print(f"     1試合あたり {elapsed / games:.2f}秒（ローカル2プロセスでの参考値）")

    workers = max(1, (os.cpu_count() or 2) - 1)
    print("\n全て成功。本番実行に進めます。")
    print(f"推奨 --selfplay-workers / --eval-workers: {workers}")


if __name__ == "__main__":
    main()
