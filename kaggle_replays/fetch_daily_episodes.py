#!/usr/bin/env python3
"""運営が日次で公開している Kaggle Dataset からエピソードリプレイを取得する。

背景:
  `kaggle competitions replay` (GetEpisodeReplay API) はアカウント単位で
  429 に制限されており、いつ解除されるか分からない。一方で運営は日次の
  エピソード集合を Kaggle Dataset として公開しており、そちらは
  `kaggle datasets` 系 API 経由で取得できる(制限されていないことを実測済み)。

  - インデックス: kaggle/pokemon-tcg-ai-battle-episodes-index の manifest.csv に
    日付ごとの daily_dataset_slug が載っている。
  - 各日の Dataset (kaggle/pokemon-tcg-ai-battle-episodes-<date>) は
    <episode_id>.json 単位のファイル群で、1日あたり数千件ある。
  - このスクリプトは、日付の新しい方から順にファイル一覧を列挙し、未取得の
    エピソードを並列ダウンロードし、`episode-<id>-replay.json` にリネームして
    replays/ に配置し、episodes_master.jsonl に逐次インデックスする。

このスクリプトは `kaggle competitions replay` を一度も呼ばない。

使い方:
  # まずは dry-run で何件取得予定か確認
  python fetch_daily_episodes.py --max-episodes 20 --dry-run

  # 実際に取得
  python fetch_daily_episodes.py --max-episodes 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    load_existing_episode_ids,
    run_kaggle_json_with_page_token,
)
from repair_master_index import find_latest_leaderboard, load_name_to_context  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_DATASET_REF = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DOWNLOAD_RETRY_DELAYS = [0, 2, 4, 8]  # 最初は待たずに1回、その後2s/4s/8sで最大3回リトライ


class AbortRun(Exception):
    """429検知やAPI異常など、全体を即座に中止すべき状況で送出する。"""


def list_dataset_files(dataset_ref: str) -> list[dict]:
    """`kaggle datasets files` をページングしてファイル一覧全件を返す。"""
    entries: list[dict] = []
    page_token: str | None = None
    while True:
        args = ["datasets", "files", dataset_ref, "--page-size", "200"]
        if page_token:
            args += ["--page-token", page_token]
        try:
            obj, page_token = run_kaggle_json_with_page_token(args)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "") + (e.stdout or "")
            if _looks_like_429(stderr):
                raise AbortRun(f"datasets files で429を検知: {dataset_ref}\n{stderr[:500]}")
            raise
        entries.extend(obj)
        if not page_token:
            break
    return entries


def _looks_like_429(text: str) -> bool:
    lowered = text.lower()
    return "429" in text or "too many requests" in lowered


def load_or_fetch_manifest(index_dir: Path) -> Path:
    manifest_path = index_dir / "manifest.csv"
    if manifest_path.exists():
        print(f"manifest.csv は既に手元にあります: {manifest_path}")
        return manifest_path
    index_dir.mkdir(parents=True, exist_ok=True)
    print(f"manifest.csv を取得します: {INDEX_DATASET_REF}")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download", INDEX_DATASET_REF,
            "-f", "manifest.csv", "-p", str(index_dir), "-q",
        ],
        capture_output=True, encoding="utf-8",
    )
    if result.returncode != 0:
        combined = (result.stdout or "") + (result.stderr or "")
        if _looks_like_429(combined):
            raise AbortRun(f"manifest.csv 取得で429を検知\n{combined[:500]}")
        raise RuntimeError(f"manifest.csv の取得に失敗しました:\n{combined}")
    if not manifest_path.exists():
        raise RuntimeError(f"manifest.csv のダウンロードが完了しませんでした: {manifest_path}")
    return manifest_path


def load_manifest_rows(manifest_path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["date"]] = row
    return rows


def existing_episode_ids_in_replays(out_dir: Path) -> set[str]:
    ids = set()
    if not out_dir.exists():
        return ids
    for p in out_dir.glob("episode-*-replay.json"):
        parts = p.stem.split("-")
        if len(parts) >= 2:
            ids.add(parts[1])
    return ids


def get_day_episode_ids(
    date: str,
    slug: str,
    daily_file_cache_dir: Path,
) -> tuple[list[str], bool, int]:
    """(その日の episode_id 一覧, キャッシュヒットしたか, ファイル総数) を返す。"""
    dataset_ref = f"kaggle/{slug}"
    cache_path = daily_file_cache_dir / f"{slug}.json"
    if cache_path.exists():
        entries = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_hit = True
    else:
        entries = list_dataset_files(dataset_ref)
        daily_file_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        cache_hit = False
    ids = [e["name"][:-5] for e in entries if e.get("name", "").endswith(".json")]
    return ids, cache_hit, len(entries)


def download_one(
    episode_id: str,
    dataset_ref: str,
    out_dir: Path,
    min_free_gb: float,
    abort_event: threading.Event,
) -> tuple[str, str, str | None]:
    """1エピソードをダウンロードしてリネームする。(episode_id, status, info) を返す。

    status: "ok" | "aborted" | "low_disk" | "429" | "failed"
    """
    if abort_event.is_set():
        return episode_id, "aborted", None

    usage = shutil.disk_usage(str(out_dir))
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        abort_event.set()
        return episode_id, "low_disk", f"{free_gb:.2f}"

    raw_path = out_dir / f"{episode_id}.json"
    dest_path = out_dir / f"episode-{episode_id}-replay.json"
    last_err: str | None = None

    for delay in DOWNLOAD_RETRY_DELAYS:
        if delay:
            time.sleep(delay)
        if abort_event.is_set():
            return episode_id, "aborted", None
        # 前回の失敗で中途半端なファイルが残っていると "up to date" 判定で
        # 再ダウンロードがスキップされることがあるため、事前に消しておく。
        if raw_path.exists():
            raw_path.unlink()
        result = subprocess.run(
            [
                "kaggle", "datasets", "download", dataset_ref,
                "-f", f"{episode_id}.json", "-p", str(out_dir), "-q",
            ],
            capture_output=True, encoding="utf-8",
        )
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0 and raw_path.exists():
            raw_path.replace(dest_path)
            return episode_id, "ok", None
        if _looks_like_429(combined):
            abort_event.set()
            return episode_id, "429", combined[:300]
        last_err = combined[:300] or f"returncode={result.returncode}"

    return episode_id, "failed", last_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-episodes", type=int, required=True, help="新規取得する上限件数")
    parser.add_argument(
        "--days", type=str, default=None,
        help="対象日をカンマ区切りで明示 (例: 2026-07-30,2026-07-29)。"
        "省略時は manifest の新しい方から全日程を対象にする",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="デフォルト: kaggle_replays/replays (スクリプト基準)",
    )
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--index-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true", help="何件取得するかを表示するだけ")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (SCRIPT_DIR / "replays")
    index_dir = SCRIPT_DIR / "index"
    daily_file_cache_dir = index_dir / "daily_file_cache"
    leaderboard_history_dir = index_dir / "leaderboard_history"
    master_index_path = index_dir / "episodes_master.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest_path = load_or_fetch_manifest(index_dir)
    except AbortRun as e:
        print(f"[ABORT] {e}")
        return 1
    manifest_rows = load_manifest_rows(manifest_path)
    print(f"manifest.csv: {len(manifest_rows)}日分")

    if args.days:
        days_order = [d.strip() for d in args.days.split(",") if d.strip()]
        missing = [d for d in days_order if d not in manifest_rows]
        if missing:
            print(f"[WARN] manifest に無い日付をスキップします: {missing}")
            days_order = [d for d in days_order if d in manifest_rows]
    else:
        days_order = sorted(manifest_rows.keys(), reverse=True)

    print(f"対象候補日 (新しい順): {days_order[:5]}{' ...' if len(days_order) > 5 else ''} "
          f"(計{len(days_order)}日)")

    existing_ids = existing_episode_ids_in_replays(out_dir)
    print(f"手元に既にあるエピソード: {len(existing_ids)}件 ({out_dir})")

    selected: list[tuple[str, str, str]] = []  # (episode_id, dataset_ref, date)
    selected_ids: set[str] = set()
    remaining = args.max_episodes
    per_day_report = []

    for date in days_order:
        if remaining <= 0:
            break
        row = manifest_rows[date]
        slug = row["daily_dataset_slug"]
        dataset_ref = f"kaggle/{slug}"
        try:
            ids, cache_hit, total_files = get_day_episode_ids(date, slug, daily_file_cache_dir)
        except AbortRun as e:
            print(f"[ABORT] {e}")
            return 1
        new_ids = [i for i in ids if i not in existing_ids and i not in selected_ids]
        take = new_ids[:remaining]
        for eid in take:
            selected.append((eid, dataset_ref, date))
            selected_ids.add(eid)
        remaining -= len(take)
        per_day_report.append((date, cache_hit, total_files, len(new_ids), len(take)))
        cache_note = "cache-hit" if cache_hit else "listed+cached"
        print(
            f"  {date}: {cache_note} total={total_files} new_available={len(new_ids)} "
            f"selected={len(take)} (残り目標 {remaining})"
        )

    print(f"選択されたエピソード数: {len(selected)} / 目標 {args.max_episodes}")

    if args.dry_run:
        print("[dry-run] ダウンロードは行いません。")
        for date, cache_hit, total_files, new_avail, took in per_day_report:
            print(f"  {date}: 新規候補={new_avail} 選択={took}")
        return 0

    if not selected:
        print("取得対象がありません。終了します。")
        return 0

    leaderboard_path = find_latest_leaderboard(leaderboard_history_dir)
    name_to_context, snapshot = load_name_to_context(leaderboard_path)
    print(
        f"リーダーボードスナップショット: {leaderboard_path.name} "
        f"(run_id={snapshot['run_id']} 収録チーム数={len(name_to_context)})"
    )

    already_indexed = load_existing_episode_ids(master_index_path)
    index_before = len(already_indexed)
    print(f"既存インデックス行数: {index_before}")

    abort_event = threading.Event()
    success_count = 0
    fail_count = 0
    aborted_count = 0
    pending_batch: list[str] = []
    start_time = time.time()

    def flush_index() -> None:
        nonlocal pending_batch, already_indexed
        if not pending_batch:
            return
        rows = build_master_rows(
            run_id=snapshot["run_id"],
            fetched_at=snapshot["fetched_at"],
            competition=snapshot["competition"],
            replays_dir=out_dir,
            episode_meta={},
            name_to_context=name_to_context,
            already_indexed=already_indexed,
            episode_ids=pending_batch,
        )
        append_master_rows(master_index_path, rows)
        already_indexed.update(r["episode_id"] for r in rows)
        print(f"  [index] +{len(rows)}行 追記 (現在 {len(already_indexed)}件)")
        pending_batch = []

    print(f"ダウンロード開始: {len(selected)}件 concurrency={args.concurrency}")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(download_one, eid, ref, out_dir, args.min_free_gb, abort_event): (eid, ref, date)
            for eid, ref, date in selected
        }
        try:
            for future in as_completed(futures):
                eid, ref, date = futures[future]
                _, status, info = future.result()
                if status == "ok":
                    success_count += 1
                    pending_batch.append(eid)
                elif status == "aborted":
                    aborted_count += 1
                else:
                    fail_count += 1
                    if status == "429":
                        print(f"[ABORT] episode {eid} で429を検知しました。全体を中止します。\n{info}")
                    elif status == "low_disk":
                        print(f"[ABORT] 空き容量が閾値を下回りました ({info}GB < {args.min_free_gb}GB)。全体を中止します。")
                    else:
                        print(f"[WARN] episode {eid} 取得失敗: {info}")

                done = success_count + fail_count + aborted_count
                elapsed_min = max((time.time() - start_time) / 60, 1e-9)
                rate = success_count / elapsed_min
                free_gb = shutil.disk_usage(str(out_dir)).free / (1024 ** 3)
                print(
                    f"[progress] {success_count}/{args.max_episodes} success "
                    f"(fail={fail_count} aborted={aborted_count} done={done}/{len(selected)}) "
                    f"rate={rate:.1f}件/分 free={free_gb:.1f}GB"
                )

                if len(pending_batch) >= args.index_every:
                    flush_index()

                if abort_event.is_set():
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    flush_index()

    elapsed_min = max((time.time() - start_time) / 60, 1e-9)
    rate = success_count / elapsed_min
    index_after = len(load_existing_episode_ids(master_index_path))
    free_gb = shutil.disk_usage(str(out_dir)).free / (1024 ** 3)

    print("=" * 60)
    print(f"完了: success={success_count} fail={fail_count} aborted={aborted_count} "
          f"(選択数={len(selected)})")
    print(f"実測レート: {rate:.2f}件/分")
    print(f"インデックス行数: {index_before} -> {index_after}")
    print(f"最終空き容量: {free_gb:.1f}GB")

    if abort_event.is_set():
        print("[ABORT] 429 または空き容量不足により中止しました。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
