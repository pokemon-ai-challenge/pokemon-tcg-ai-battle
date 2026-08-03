#!/usr/bin/env python3
"""fetch_top_episodes_fast.py の AIMD レート制御を、Kaggle API を一切呼ばずに検証する。

対象: RateController / download_with_retry / run_download_pool。
`_common.download_replay` を `unittest.mock.patch.object` でモックに差し替え、
実際の kaggle CLI は一度も起動しない。

3ケース:
  1. 常に成功するモック -> リクエスト間隔が --min-interval 付近まで縮むこと
  2. 一定間隔で429を返すモック -> 間隔が広がり、それでもリトライで全件成功すること
     (429はリトライ回数を消費しない設計の確認)
  3. 常に429を返すモック -> --abort-after-429 に達した時点で中止し、
     未着手分が "aborted" として扱われること

pytest ではなく、kaggle_replays/rl/test_*.py と同様に直接実行するスクリプト形式。
    python test_fetch_adaptive.py
各ケースの実測値(間隔の推移・件数・所要時間・中止の発火)を print する。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest import mock

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import fetch_top_episodes_fast as fte  # noqa: E402


def _rate_limited_error(msg: str = "Too Many Requests (429)") -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(returncode=1, cmd=["kaggle"], stderr=msg)


def scenario_always_success() -> None:
    print("\n=== ケース1: 常に成功するモック ===")
    abort_event = threading.Event()
    controller = fte.RateController(
        initial_interval=0.2,
        min_interval=0.01,
        max_interval=1.0,
        success_streak_target=3,
        cooldown_seconds=0.05,
        abort_after_429=1000,
        abort_event=abort_event,
    )

    def mock_download_replay(episode_id, out_dir):
        return None

    episode_ids = [str(i) for i in range(1, 91)]
    interval_trace: list[float] = []

    def on_result(eid, status, err):
        interval_trace.append(controller.snapshot()["interval"])

    with mock.patch.object(fte, "download_replay", new=mock_download_replay):
        t0 = time.monotonic()
        fte.run_download_pool(
            episode_ids, Path("."), controller, abort_event,
            concurrency=4, max_retries=3, rate_limit_backoff=[0.05, 0.1, 0.2],
            on_result=on_result,
        )
        dt = time.monotonic() - t0

    snap = controller.snapshot()
    print(f"  episodes={len(episode_ids)} concurrency=4 所要={dt:.2f}s")
    print(f"  間隔推移: 開始=0.2000s -> 先頭5件={[round(x, 4) for x in interval_trace[:5]]}")
    print(f"           末尾5件={[round(x, 4) for x in interval_trace[-5:]]}")
    print(f"  最終間隔={snap['interval']:.4f}s (min_interval=0.01s)")
    print(f"  累計 成功={snap['success']} 429={snap['rate_limited']} その他失敗={snap['other_failed']}")
    assert snap["success"] == len(episode_ids)
    assert snap["rate_limited"] == 0
    assert snap["other_failed"] == 0
    assert snap["interval"] <= 0.02, f"間隔が下限近くまで縮んでいない: {snap['interval']}"
    assert not abort_event.is_set()
    print("  PASS: 429/失敗なしで間隔が下限付近まで縮んだ")


def scenario_periodic_429() -> None:
    print("\n=== ケース2: 5回に1回429を返すモック ===")
    abort_event = threading.Event()
    controller = fte.RateController(
        initial_interval=0.05,
        min_interval=0.01,
        max_interval=0.3,
        success_streak_target=3,
        cooldown_seconds=0.02,
        abort_after_429=1000,  # このケースでは中止させない
        abort_event=abort_event,
    )
    counter = {"n": 0}
    lock = threading.Lock()
    period = 6

    def mock_download_replay(episode_id, out_dir):
        with lock:
            counter["n"] += 1
            n = counter["n"]
        if n % period == 0:
            raise _rate_limited_error()
        return None

    episode_ids = [str(i) for i in range(1, 31)]
    interval_trace: list[float] = []
    statuses: list[str] = []

    def on_result(eid, status, err):
        statuses.append(status)
        interval_trace.append(controller.snapshot()["interval"])

    with mock.patch.object(fte, "download_replay", new=mock_download_replay):
        t0 = time.monotonic()
        fte.run_download_pool(
            episode_ids, Path("."), controller, abort_event,
            concurrency=3, max_retries=3, rate_limit_backoff=[0.02, 0.03, 0.05],
            on_result=on_result,
        )
        dt = time.monotonic() - t0

    snap = controller.snapshot()
    status_counts = {s: statuses.count(s) for s in set(statuses)}
    print(f"  episodes={len(episode_ids)} concurrency=3 所要={dt:.2f}s 実際のAPI呼び出し回数={counter['n']}")
    print(f"  間隔推移: 最小={min(interval_trace):.4f}s 最大={max(interval_trace):.4f}s 最終={snap['interval']:.4f}s")
    print(f"  累計 成功={snap['success']} 429={snap['rate_limited']} その他失敗={snap['other_failed']}")
    print(f"  ステータス内訳: {status_counts}")
    assert snap["rate_limited"] > 0, "429が一度も記録されていない"
    assert snap["success"] == len(episode_ids), "429を受けてもリトライして最終的に全件成功するはず"
    assert status_counts.get("success", 0) == len(episode_ids)
    assert max(interval_trace) > 0.09, "429を受けても間隔が広がっていない"
    assert not abort_event.is_set(), "abort_after_429=1000 なのに中止してしまっている"
    print("  PASS: 429のたびに間隔が広がり、それでも(リトライ回数を消費せず)全件成功")


def scenario_always_429_aborts() -> None:
    print("\n=== ケース3: 常に429を返すモック(中止するはず) ===")
    abort_event = threading.Event()
    abort_after_429 = 5
    controller = fte.RateController(
        initial_interval=0.05,
        min_interval=0.01,
        max_interval=1.0,
        success_streak_target=5,
        cooldown_seconds=0.03,
        abort_after_429=abort_after_429,
        abort_event=abort_event,
    )
    call_count = {"n": 0}
    lock = threading.Lock()

    def mock_download_replay(episode_id, out_dir):
        with lock:
            call_count["n"] += 1
        raise _rate_limited_error("429 Client Error: Too Many Requests")

    episode_ids = [str(i) for i in range(1, 51)]
    statuses: list[str] = []

    def on_result(eid, status, err):
        statuses.append(status)

    with mock.patch.object(fte, "download_replay", new=mock_download_replay):
        t0 = time.monotonic()
        fte.run_download_pool(
            episode_ids, Path("."), controller, abort_event,
            concurrency=3, max_retries=3, rate_limit_backoff=[0.02, 0.03, 0.05],
            on_result=on_result,
        )
        dt = time.monotonic() - t0

    snap = controller.snapshot()
    n_success = statuses.count("success")
    n_failed = statuses.count("failed")
    n_aborted = statuses.count("aborted")
    print(f"  episodes={len(episode_ids)} concurrency=3 所要={dt:.2f}s 実際のAPI呼び出し回数={call_count['n']}")
    print(f"  累計429={snap['rate_limited']} (--abort-after-429={abort_after_429})")
    print(f"  ステータス内訳: success={n_success} failed={n_failed} aborted={n_aborted}")
    print(f"  abort_event.is_set()={abort_event.is_set()}")
    assert abort_event.is_set(), "abort_after_429 に達したのに中止フラグが立っていない"
    assert n_success == 0, "常に429なので成功は0件のはず"
    assert n_aborted > 0, "中止後、未着手のエピソードが aborted として扱われていない"
    assert snap["rate_limited"] >= abort_after_429
    assert dt < 5.0, f"小さいbackoff/cooldownを使っているのに遅すぎる(実装ミスの疑い): {dt:.2f}s"
    print(f"  PASS: 累計429={snap['rate_limited']}件({dt:.2f}s)で中止、{n_aborted}件が未着手のまま終了")


def main() -> None:
    scenario_always_success()
    scenario_periodic_429()
    scenario_always_429_aborts()
    print(
        "\n全ケース PASS。download_replay はモックに差し替え済みのため、"
        "kaggle CLI / Kaggle API は一度も呼び出していません。"
    )


if __name__ == "__main__":
    main()
