#!/usr/bin/env python3
"""上位20・全エピソード取得を in-process Kaggle API で完遂する(2026-07-30 run 専用)。

背景: kaggle.exe ランチャがアプリ制御ポリシーでブロックされ(WinError 4551 /
Permission denied)、CLI 経由(subprocess)が全滅した。kaggle Python API は
in-process(HTTPリクエスト)なら動作するため、fetch_top_episodes.py 相当の
処理を API 直呼びで再実装する。

処理:
  1. 現在の上位20チームをリーダーボードAPIで取得(snapshotも保存)。
  2. 各チームの最新提出(1件)→そのエピソード一覧(COMPLETEDのみ, createTime付き)。
  3. まだディスクに無いエピソードだけ replay をDL(429が連続したらクォータ枯渇と
     みなし打ち切り=グラインドしない)。
  4. ディスク上の全リプレイを episodes_master.jsonl へ index(タイムスタンプ付き)。

冪等: 既存DLは dest.exists() でskip。index は既存episode_idをskip。何度でも再実行可。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
KAGGLE_REPLAYS = RUN_DIR.parent.parent
sys.path.insert(0, str(KAGGLE_REPLAYS))
from _common import (  # noqa: E402
    append_master_rows, build_master_rows, load_existing_episode_ids,
)

import kaggle  # noqa: E402

COMPETITION = "pokemon-tcg-ai-battle"
TOP = 20
SUBS_PER_TEAM = 1
REPLAYS = RUN_DIR / "replays"
LBH = RUN_DIR / "index" / "leaderboard_history"
MASTER = RUN_DIR / "index" / "episodes_master.jsonl"
STATUS = RUN_DIR / "refill_inproc_status.json"
FAILED_LOG = RUN_DIR / "refill_inproc_failed.json"

DL_SLEEP = 0.4                 # replay DL 間隔
CONSECUTIVE_429_STOP = 20      # 連続429がこの数に達したらクォータ枯渇とみなし打ち切り


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(**kw) -> None:
    base = {
        "updated_at": now(),
        "downloaded_total": sum(1 for _ in REPLAYS.glob("episode-*-replay.json")),
    }
    base.update(kw)
    STATUS.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def is_rate_limit(exc: Exception) -> bool:
    m = str(exc)
    return "429" in m or "Too Many Requests" in m


def main() -> None:
    api = kaggle.api
    api.authenticate()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-inproc"
    fetched_at = now()
    write_status(phase="leaderboard", run_id=run_id, note="上位20をAPI取得中")

    # 1. leaderboard
    lb_rows = api.competition_leaderboard_view(COMPETITION, page_size=TOP) or []
    lb = [r.to_dict() for r in lb_rows if r is not None][:TOP]
    name_to_context = {
        row["teamName"]: {"team_id": row["teamId"], "rank": i + 1, "score": float(row["score"])}
        for i, row in enumerate(lb)
    }
    LBH.mkdir(parents=True, exist_ok=True)
    (LBH / f"leaderboard-{run_id}.json").write_text(
        json.dumps({
            "competition": COMPETITION, "run_id": run_id, "fetched_at": fetched_at,
            "top_n": TOP, "source": "in-process-api",
            "leaderboard": [{"rank": i + 1, **row} for i, row in enumerate(lb)],
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[1/4] leaderboard 上位{len(lb)}チーム (run_id={run_id})")
    for i, row in enumerate(lb, 1):
        print(f"  #{i} {row['teamName']} (teamId={row['teamId']}, score={row['score']})")

    # 2. submissions per team
    print("[2/4] 各チームの最新提出IDを取得")
    submission_ids: list[int] = []
    for row in lb:
        try:
            subs = [s.to_dict() for s in (api.competition_team_submissions(row["teamId"]) or [])]
            submission_ids.extend(s["id"] for s in subs[:SUBS_PER_TEAM])
        except Exception as e:  # noqa: BLE001
            print(f"  警告: team {row['teamId']} ({row['teamName']}) 提出取得失敗: {e!r}", file=sys.stderr)
        time.sleep(0.2)

    # 3. episodes per submission
    print(f"[3/4] エピソード一覧を取得({len(submission_ids)}提出)")
    episode_meta: dict[str, dict] = {}
    for sid in submission_ids:
        try:
            eps = [e.to_dict() for e in (api.competition_list_episodes(int(sid)) or [])]
        except Exception as e:  # noqa: BLE001
            print(f"  警告: submission {sid} エピソード一覧取得失敗: {e!r}", file=sys.stderr)
            continue
        for e in eps:
            if "COMPLETED" in str(e.get("state", "")):
                episode_meta[str(e["id"])] = {
                    "id": e["id"], "createTime": e.get("createTime"), "endTime": e.get("endTime"),
                }
        time.sleep(0.2)

    all_ids = sorted(episode_meta.keys(), key=int)
    missing = [eid for eid in all_ids if not (REPLAYS / f"episode-{eid}-replay.json").exists()]
    print(f"  上位20の総エピソード: {len(all_ids)} / うち未取得: {len(missing)}")
    write_status(phase="downloading", run_id=run_id, total_episodes=len(all_ids),
                 missing_at_start=len(missing), downloaded_this_run=0, note="未取得replayをDL中")

    # 4. download missing
    downloaded = 0
    rate_limited = 0
    unavailable = 0
    consec_429 = 0
    failed_ids: list[str] = []
    stopped_by_quota = False
    for i, eid in enumerate(missing, 1):
        try:
            api.competition_episode_replay(int(eid), path=str(REPLAYS), quiet=True)
            if (REPLAYS / f"episode-{eid}-replay.json").exists():
                downloaded += 1
                consec_429 = 0
            else:
                unavailable += 1
                failed_ids.append(eid)
        except Exception as e:  # noqa: BLE001
            if is_rate_limit(e):
                rate_limited += 1
                consec_429 += 1
                failed_ids.append(eid)
                if consec_429 >= CONSECUTIVE_429_STOP:
                    stopped_by_quota = True
                    print(f"  連続{consec_429}件の429 → クォータ枯渇とみなし打ち切り(残り{len(missing)-i}件)")
                    break
            else:
                unavailable += 1
                consec_429 = 0
                failed_ids.append(eid)
        if i % 25 == 0 or i == len(missing):
            write_status(phase="downloading", run_id=run_id, total_episodes=len(all_ids),
                         missing_at_start=len(missing), processed=i, downloaded_this_run=downloaded,
                         rate_limited=rate_limited, unavailable=unavailable,
                         note=f"{i}/{len(missing)} 処理")
            print(f"  ({i}/{len(missing)}) downloaded={downloaded} 429={rate_limited} unavailable={unavailable}")
        time.sleep(DL_SLEEP)

    FAILED_LOG.write_text(json.dumps({
        "run_id": run_id, "stopped_by_quota": stopped_by_quota,
        "rate_limited": rate_limited, "unavailable": unavailable,
        "failed_ids": failed_ids,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. index all on-disk replays
    print("[4/4] episodes_master.jsonl を更新(ディスク全リプレイ)")
    already = load_existing_episode_ids(MASTER)
    rows = build_master_rows(
        run_id=run_id, fetched_at=fetched_at, competition=COMPETITION,
        replays_dir=REPLAYS, episode_meta=episode_meta,
        name_to_context=name_to_context, already_indexed=already,
    )
    append_master_rows(MASTER, rows)

    total_replays = sum(1 for _ in REPLAYS.glob("episode-*-replay.json"))
    total_master = sum(1 for line in MASTER.open(encoding="utf-8") if line.strip())
    done = (not stopped_by_quota) and len(missing) - len(failed_ids) >= 0 and rate_limited == 0
    write_status(phase="done" if not stopped_by_quota else "stopped_quota",
                 run_id=run_id, total_episodes=len(all_ids), downloaded_this_run=downloaded,
                 rate_limited=rate_limited, unavailable=unavailable,
                 total_replays_on_disk=total_replays, master_rows=total_master,
                 indexed_new=len(rows), done=(not stopped_by_quota),
                 note="完了" if not stopped_by_quota else "クォータ枯渇で一部未取得。窓回復後に再実行で継続")
    print(f"完了: 今回DL={downloaded}, 429={rate_limited}, unavailable={unavailable}, "
          f"master追記={len(rows)}, disk replays={total_replays}, master rows={total_master}, "
          f"stopped_by_quota={stopped_by_quota}")


if __name__ == "__main__":
    main()
