#!/usr/bin/env python3
"""上位20・全エピソード取得のレート制限リフィル・ドライバ(2026-07-30 run 専用)。

fetch_top_episodes.py --max-episodes 0 は Kaggle の replay DL レート制限
(推定 ~数千件/時)に到達すると 429 を量産する。このドライバは冪等な再取得を
1時間ごとに自動で回し、「429ゼロで完走(step5『完了:』到達)」したら自動停止する。

- 既存のダウンロード済みリプレイはローカルで dest.exists() スキップされる(API非消費)。
- master index は各パスの step5 で未indexぶんだけ追記される(重複排除)。
- 途中経過は refill_status.json に随時書き出す(状況確認を安価にするため)。
- 実際の replay DL 失敗(429/接続断)が残っている限りクールダウンして再試行する。

停止条件:
  - あるパスが step5『完了:』に到達し、そのパスの replay DL 失敗が 0 件 → DONE。
  - もしくは MAX_PASSES に到達 → 手動確認へ委ねる(そこまでの取得分は保存済み)。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
KAGGLE_REPLAYS = RUN_DIR.parent.parent  # .../kaggle_replays
FETCH = KAGGLE_REPLAYS / "fetch_top_episodes.py"

REPLAYS = RUN_DIR / "replays"
LBH = RUN_DIR / "index" / "leaderboard_history"
MASTER = RUN_DIR / "index" / "episodes_master.jsonl"
STATUS = RUN_DIR / "refill_status.json"

SLEEP_BETWEEN_CALLS = "1.0"   # fetch の --sleep(バースト緩和)
COOLDOWN_SEC = 3600           # レート制限リセット待ち(毎時)
INITIAL_COOLDOWN_SEC = 3600   # 直前に429到達したため最初に必ず1時間待つ
ABORT_AFTER_FAILS = 60        # 1パスでこの件数の DL失敗を観測したらそのパスを打ち切る
POLL_SEC = 20                 # パス監視のポーリング間隔
MAX_PASSES = 8


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_replays() -> int:
    return sum(1 for _ in REPLAYS.glob("episode-*-replay.json"))


def count_master_rows() -> int:
    if not MASTER.exists():
        return 0
    with MASTER.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_status(**kw) -> None:
    base = {
        "updated_at": now(),
        "downloaded_total": count_replays(),
        "master_rows": count_master_rows(),
    }
    base.update(kw)
    STATUS.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


_FAIL_RE = re.compile(r"リプレイ取得に失敗")
_DONE_RE = re.compile(r"^完了:", re.MULTILINE)


def run_pass(pass_no: int) -> tuple[bool, int]:
    """1パス実行。(completed_clean, n_dl_fail) を返す。

    completed_clean = step5『完了:』に到達し DL失敗が0件だった。
    途中で DL失敗が ABORT_AFTER_FAILS を超えたらプロセスを kill して打ち切る。
    """
    pass_log = RUN_DIR / f"refill_pass_{pass_no}.log"
    cmd = [
        sys.executable, str(FETCH),
        "--top", "20", "--max-episodes", "0", "--sleep", SLEEP_BETWEEN_CALLS,
        "--out-dir", str(REPLAYS),
        "--leaderboard-history-dir", str(LBH),
        "--master-index-path", str(MASTER),
    ]
    with pass_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        aborted = False
        while proc.poll() is None:
            time.sleep(POLL_SEC)
            try:
                text = pass_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            n_fail = len(_FAIL_RE.findall(text))
            write_status(
                phase=f"pass{pass_no}:running",
                pass_no=pass_no,
                last_pass_fails=n_fail,
                note="rate-limited, will abort this pass and cool down" if n_fail >= ABORT_AFTER_FAILS else "downloading",
            )
            if n_fail >= ABORT_AFTER_FAILS:
                proc.kill()
                aborted = True
                break
        proc.wait()

    text = pass_log.read_text(encoding="utf-8", errors="replace")
    n_fail = len(_FAIL_RE.findall(text))
    completed = bool(_DONE_RE.search(text)) and not aborted
    completed_clean = completed and n_fail == 0
    return completed_clean, n_fail


def main() -> None:
    write_status(phase="cooldown:initial", pass_no=0, note=f"waiting {INITIAL_COOLDOWN_SEC}s for rate-limit reset")
    time.sleep(INITIAL_COOLDOWN_SEC)

    for pass_no in range(1, MAX_PASSES + 1):
        write_status(phase=f"pass{pass_no}:start", pass_no=pass_no)
        completed_clean, n_fail = run_pass(pass_no)
        if completed_clean:
            write_status(phase="done", pass_no=pass_no, last_pass_fails=0, done=True,
                         note="clean completion: all top-20 episodes downloaded")
            print(f"[refill] DONE at pass {pass_no}: {count_replays()} replays, master {count_master_rows()} rows")
            return
        write_status(phase=f"pass{pass_no}:cooldown", pass_no=pass_no, last_pass_fails=n_fail, done=False,
                     note=f"{n_fail} DL失敗残 -> {COOLDOWN_SEC}s クールダウン後に次パス")
        if pass_no < MAX_PASSES:
            time.sleep(COOLDOWN_SEC)

    write_status(phase="stopped:max_passes", pass_no=MAX_PASSES, done=False,
                 note="MAX_PASSES到達。取得分は保存済み。残りは手動確認へ")
    print(f"[refill] stopped after {MAX_PASSES} passes: {count_replays()} replays, master {count_master_rows()} rows")


if __name__ == "__main__":
    main()
