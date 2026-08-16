"""投入済み head-to-head カーネルを完了までポーリングし、回収→マージまで自動で行う。

``push_kernel_deck.py --status/--fetch/--merge`` を長時間ぶん回すためのラッパ。
数時間かかる測定を張り付いて見ていられないので、進捗をログファイルに追記しながら待つ。

使い方:
    python kaggle_replays/kaggle_run/watch_deck_h2h.py --tag d1 \
        --log C:/tmp/ptcg_deck_h2h_kernels/_watch_d1.log \
        --save-to kaggle_replays/_kaggle_deck_h2h_results.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import push_kernel_deck as pk  # noqa: E402


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def status_of(ref: str) -> str:
    """kaggle CLI を直接叩く(pk.kernel_status は毎回コマンドを print するのでログが荒れる)。"""
    proc = subprocess.run([pk.kaggle_cmd(), "kernels", "status", ref],
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = ((proc.stdout or "") + (proc.stderr or "")).lower()
    for state in ("complete", "error", "cancelacknowledged", "running", "queued"):
        if state in text:
            return state
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1")
    ap.add_argument("--work", default=str(pk.DEFAULT_WORK))
    ap.add_argument("--interval", type=float, default=600.0, help="ポーリング間隔(秒)")
    ap.add_argument("--max-hours", type=float, default=13.0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--save-to", default="kaggle_replays/_kaggle_deck_h2h_results.json")
    args = ap.parse_args()

    work = Path(args.work).resolve()
    record = pk.load_record(work, args.tag)
    refs = [k["ref"] for k in record["kernels"]]
    log_path = Path(args.log) if args.log else (work / f"_watch_{args.tag}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log(log_path, f"watch start tag={args.tag} kernels={len(refs)} "
                  f"games_per_shard={record['games_per_shard']} arms={record['arms']}")

    deadline = time.time() + args.max_hours * 3600
    terminal = {"complete", "error", "cancelacknowledged"}
    while True:
        states = {ref: status_of(ref) for ref in refs}
        log(log_path, " ".join(f"{r.split('/')[-1]}={s}" for r, s in states.items()))
        if all(s in terminal for s in states.values()):
            log(log_path, "all kernels reached a terminal state")
            break
        if time.time() > deadline:
            log(log_path, "watch timeout reached; fetching whatever is available")
            break
        time.sleep(args.interval)

    # 回収 → マージ(1本でも失敗していても、完了ぶんだけで集計できる)
    ns = pk.parse_args(["--fetch", "--merge", "--tag", args.tag, "--work", str(work),
                        "--save-to", args.save_to])
    try:
        pk.cmd_fetch(ns)
    except SystemExit as exc:  # noqa: PERF203
        log(log_path, f"fetch failed: {exc}")
    try:
        pk.cmd_merge(ns)
        log(log_path, f"merged -> {args.save_to}")
    except SystemExit as exc:
        log(log_path, f"merge failed: {exc}")


if __name__ == "__main__":
    main()
