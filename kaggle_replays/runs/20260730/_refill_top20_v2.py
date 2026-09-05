#!/usr/bin/env python3
"""上位20・全エピソード取得のリフィル(probe-gated版・2026-07-30 run 専用)。

前版(_refill_top20.py)は毎時フルパスを回したが、429を量産して制限窓を
温め続けた可能性があり、8時間/8パスで1件も回復しなかった。replay DL の
クォータは毎時ではなく日次~24時間窓と判明。

本版は「叩かずに待つ」:
  - 未取得エピソードを1件だけ静かにプローブ(1リクエスト/インターバル)。
  - 429 の間はただ待つ(制限窓を温めない)。
  - 200 が返った=クォータ回復 と判断した瞬間にだけ、クリーンな
    フルパス(fetch_top_episodes --max-episodes 0)を1回実行する。
    このパスが 429 ゼロで step5『完了:』に到達したら全件取得完了 → 自動停止。
    残量がクォータより多く途中で再度枯渇したら、待って再プローブする。

冪等性: 既存ダウンロードはローカルskip。master は最終パスの step5 で
全件(タイムスタンプ付き)index される。
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
KAGGLE_REPLAYS = RUN_DIR.parent.parent
FETCH = KAGGLE_REPLAYS / "fetch_top_episodes.py"

REPLAYS = RUN_DIR / "replays"
LBH = RUN_DIR / "index" / "leaderboard_history"
MASTER = RUN_DIR / "index" / "episodes_master.jsonl"
STATUS = RUN_DIR / "refill_v2_status.json"

SLEEP_BETWEEN_CALLS = "1.0"
PROBE_INTERVAL_SEC = 1800     # 429の間は30分おきに1件だけプローブ
POST_PARTIAL_WAIT_SEC = 21600  # フルパスが途中でクォータ枯渇したら6h待って再プローブ
PASS_ABORT_FAILS = 200        # フルパス中にこの件数失敗したらクォータ再枯渇とみなし打ち切り
POLL_SEC = 20
MAX_HOURS = 30                # 全体の安全上限

_FAIL_RE = re.compile(r"リプレイ取得に失敗")
_DONE_RE = re.compile(r"^完了:", re.MULTILINE)
_FAILID_RE = re.compile(r"episode (\d+) のリプレイ取得に失敗")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def on_disk_ids() -> set[str]:
    return {p.stem.split("-")[1] for p in REPLAYS.glob("episode-*-replay.json")}


def count_replays() -> int:
    return sum(1 for _ in REPLAYS.glob("episode-*-replay.json"))


def count_master_rows() -> int:
    if not MASTER.exists():
        return 0
    with MASTER.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_status(**kw) -> None:
    base = {"updated_at": now(), "downloaded_total": count_replays(), "master_rows": count_master_rows()}
    base.update(kw)
    STATUS.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def build_probe_pool() -> list[str]:
    """過去ログの『取得失敗』episode id から、今ディスクに無いものを集める。"""
    have = on_disk_ids()
    ids: list[str] = []
    seen: set[str] = set()
    for log in sorted(RUN_DIR.glob("*.log")):
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _FAILID_RE.finditer(text):
            eid = m.group(1)
            if eid not in seen and eid not in have:
                seen.add(eid)
                ids.append(eid)
    return ids


_PROBE_SCRIPT = RUN_DIR / "_probe_replay.py"


def probe_one(eid: str) -> str:
    """1件だけ replay を取得試行。'success' / 'ratelimited' / 'unavailable' / 'error' を返す。

    driver から kaggle.exe を直接 spawn すると WinError 4551(アプリ制御ポリシー)で
    落ちる環境があるため、python→kaggle の入れ子で実績のある _probe_replay.py を
    subprocess で呼ぶ。spawn 失敗(OSError)でもドライバは落とさず 'error' を返す。
    """
    dest = REPLAYS / f"episode-{eid}-replay.json"
    if dest.exists():
        return "success"
    try:
        r = subprocess.run(
            [sys.executable, str(_PROBE_SCRIPT), eid],
            capture_output=True, encoding="utf-8", errors="replace", timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "error"
    if dest.exists():
        return "success"
    token = (r.stdout or "").strip().splitlines()[-1].strip().upper() if (r.stdout or "").strip() else "ERROR"
    return {"SUCCESS": "success", "RATELIMITED": "ratelimited", "UNAVAILABLE": "unavailable"}.get(token, "error")


def run_full_pass(pass_no: int) -> tuple[bool, int]:
    pass_log = RUN_DIR / f"refill_v2_pass_{pass_no}.log"
    cmd = [
        sys.executable, str(FETCH),
        "--top", "20", "--max-episodes", "0", "--sleep", SLEEP_BETWEEN_CALLS,
        "--out-dir", str(REPLAYS),
        "--leaderboard-history-dir", str(LBH),
        "--master-index-path", str(MASTER),
    ]
    with pass_log.open("w", encoding="utf-8") as lf:
        try:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        except OSError as e:
            lf.write(f"[driver] full-pass spawn failed: {e!r}\n")
            return False, 0
        aborted = False
        while proc.poll() is None:
            time.sleep(POLL_SEC)
            try:
                text = pass_log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            n_fail = len(_FAIL_RE.findall(text))
            write_status(phase=f"fullpass{pass_no}:running", pass_no=pass_no,
                         last_pass_fails=n_fail,
                         note="quota re-exhausted, aborting pass" if n_fail >= PASS_ABORT_FAILS else "downloading remaining")
            if n_fail >= PASS_ABORT_FAILS:
                proc.kill()
                aborted = True
                break
        proc.wait()
    text = pass_log.read_text(encoding="utf-8", errors="replace")
    n_fail = len(_FAIL_RE.findall(text))
    completed_clean = bool(_DONE_RE.search(text)) and not aborted and n_fail == 0
    return completed_clean, n_fail


def main() -> None:
    start = time.monotonic()
    deadline = start + MAX_HOURS * 3600
    pool = build_probe_pool()
    write_status(phase="probing", pass_no=0, probe_pool=len(pool),
                 note="waiting for quota reset; gentle 1-req probes every 30min")
    if not pool:
        write_status(phase="done", pass_no=0, done=True, note="未取得プールが空(既に全件?)。フルパスで最終確認")
        # 念のため1パス
        run_full_pass(1)
        return

    probe_idx = 0
    full_pass_no = 0
    while time.monotonic() < deadline:
        # 1インターバルにつき最大3件までプローブ(404/403の死にidをスキップするため)
        result = "ratelimited"
        probed_id = None
        for _ in range(3):
            if probe_idx >= len(pool):
                pool = build_probe_pool()  # 途中でディスク更新があれば作り直す
                probe_idx = 0
                if not pool:
                    result = "success"  # 未取得なし=完了扱い
                    break
            probed_id = pool[probe_idx]
            probe_idx += 1
            result = probe_one(probed_id)
            if result in ("success", "ratelimited"):
                break
            # unavailable/error は次のidへ
        write_status(phase="probing", pass_no=full_pass_no, probe_id=probed_id,
                     probe_result=result, probe_pool=len(pool),
                     elapsed_hours=round((time.monotonic() - start) / 3600, 2),
                     note="429継続中、待機" if result == "ratelimited" else f"probe={result}")

        if result == "success":
            full_pass_no += 1
            write_status(phase=f"fullpass{full_pass_no}:start", pass_no=full_pass_no,
                         note="クォータ回復を検知。フルパスで残りを取得")
            completed_clean, n_fail = run_full_pass(full_pass_no)
            if completed_clean:
                write_status(phase="done", pass_no=full_pass_no, done=True, last_pass_fails=0,
                             note="clean completion: 上位20の全エピソード取得完了")
                print(f"[refill_v2] DONE at fullpass {full_pass_no}: {count_replays()} replays, master {count_master_rows()} rows")
                return
            # 途中でクォータ再枯渇 → 待って再プローブ
            write_status(phase="partial:wait", pass_no=full_pass_no, last_pass_fails=n_fail, done=False,
                         note=f"部分取得({n_fail}件失敗残)。{POST_PARTIAL_WAIT_SEC}s待って再プローブ")
            pool = build_probe_pool()
            probe_idx = 0
            time.sleep(POST_PARTIAL_WAIT_SEC)
        else:
            time.sleep(PROBE_INTERVAL_SEC)

    write_status(phase="stopped:max_hours", pass_no=full_pass_no, done=False,
                 note="MAX_HOURS到達。取得分は保存済み。")
    print(f"[refill_v2] stopped after {MAX_HOURS}h: {count_replays()} replays, master {count_master_rows()} rows")


if __name__ == "__main__":
    main()
