#!/usr/bin/env python3
"""fetch_top_episodes.py と同じことを、並列ダウンロード + インデックス逐次書き込みでやる。

fetch_top_episodes.py の実測は 14.6件/分(1件あたり4.1秒)で、内訳は
`_common.download_replay` がエピソード1件ごとに `kaggle` CLI をサブプロセス起動して
いること(起動コストだけで1〜2秒)。リプレイ本体は1件4.4MB程度でダウンロード自体は
1秒未満のはずなので、ダウンロードを並列化すれば頭打ちの主因(起動コスト)を隠せる。

また fetch_top_episodes.py はマスターインデックスを最後([5/5])にまとめて書くため、
大量ダウンロードの途中で中断するとインデックスが更新されない問題があった
(repair_master_index.py 参照)。このスクリプトは `--index-every` 件ダウンロードする
ごとに逐次追記することで、中断しても失われるのは直近 `--index-every` 件未満に抑える。

事前準備・データ保存方針は fetch_top_episodes.py と同じ(README.md 参照)。

レート制御(AIMD):
  固定並列度だけでは「並列が速すぎて429が出る」「一度429に入ると全体が止まる」
  の両方に弱いため、プロセス全体で共有する「リクエスト間隔」を持ち、結果に応じて
  適応的に調整する(詳細は各引数のヘルプを参照)。429は他の失敗と区別し、専用の
  長いバックオフ・クールダウンを適用する。累計429件数が閾値を超えたら全体を
  中止する(半分捨てながら走り続けるのを防ぐため)。

エピソード一覧のキャッシュ:
  `--top` が大きいと[2/5]〜[3/5](各チームの提出ID・エピソード一覧の取得)だけで
  数百回の逐次API呼び出しが発生し、実行時間の大半を占める。この一覧を
  `index/episode_list_cache/<run_id>.json` に保存しておき、`--episode-list PATH`
  で渡すと[2/5]〜[3/5]を丸ごとスキップしてダウンロードから再開できる。

使い方:
  python fetch_top_episodes_fast.py --top 200 --max-episodes 300 --concurrency 2
  python fetch_top_episodes_fast.py --episode-list index/episode_list_cache/xxx.json --max-episodes 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import (  # noqa: E402
    append_master_rows,
    build_master_rows,
    download_replay,
    fetch_episodes,
    load_existing_episode_ids,
)
# fetch_top_episodes.py 自体は変更しない。リーダーボード/提出ID取得ロジックを
# コピーせず、モジュールとして import して再利用する
# (import しても main() は `if __name__ == "__main__"` 節の中なので実行されない)。
from fetch_top_episodes import fetch_leaderboard, fetch_team_submission_ids  # noqa: E402

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # 他の失敗(429以外): 2s, 4s, 8s


def _is_rate_limited(stderr: str) -> bool:
    """kaggle CLI の stderr が 429 (Too Many Requests) を示しているか判定する。"""
    return "429" in stderr or "too many requests" in stderr.lower()


def _interruptible_sleep(seconds: float, abort_event: threading.Event, step: float = 0.5) -> None:
    """abort_event が立ったら即座に切り上げる sleep。"""
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    while not abort_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


class RateController:
    """プロセス全体で共有する AIMD レート制御。

    - 成功が `success_streak_target` 件連続したら間隔を10%短縮(加算的増加)。
    - 429を1件でも受けたら間隔を2倍(乗算的減少)にし、`cooldown_seconds` 秒だけ
      全ワーカーを停止させる(クールダウンは絶対時刻で管理するので、複数ワーカーが
      同時に429を受けても単純加算で伸び続けたりはしない)。
    - 累計429件数が `abort_after_429` に達したら abort_event をセットする
      (0以下なら無効)。
    """

    def __init__(
        self,
        *,
        initial_interval: float,
        min_interval: float,
        max_interval: float,
        success_streak_target: int,
        cooldown_seconds: float,
        abort_after_429: int,
        abort_event: threading.Event,
    ) -> None:
        self._lock = threading.Lock()
        self.interval = min(max(initial_interval, min_interval), max_interval)
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.success_streak_target = max(success_streak_target, 1)
        self.cooldown_seconds = cooldown_seconds
        self.abort_after_429 = abort_after_429
        self._success_streak = 0
        self._cooldown_until = 0.0
        self._abort_event = abort_event
        self.stats = {"success": 0, "rate_limited": 0, "other_failed": 0}

    def wait_before_request(self) -> None:
        while True:
            if self._abort_event.is_set():
                return
            with self._lock:
                remaining_cooldown = self._cooldown_until - time.monotonic()
            if remaining_cooldown > 0:
                _interruptible_sleep(remaining_cooldown, self._abort_event)
                continue
            break
        with self._lock:
            interval = self.interval
        _interruptible_sleep(interval, self._abort_event)

    def report_success(self) -> None:
        with self._lock:
            self.stats["success"] += 1
            self._success_streak += 1
            if self._success_streak >= self.success_streak_target:
                self.interval = max(self.min_interval, self.interval * 0.9)
                self._success_streak = 0

    def report_rate_limited(self) -> bool:
        """429を記録する。累計が閾値に達して abort_event を立てた場合 True を返す。"""
        with self._lock:
            self.stats["rate_limited"] += 1
            self.interval = min(self.max_interval, self.interval * 2.0)
            self._success_streak = 0
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + self.cooldown_seconds)
            triggered = False
            if self.abort_after_429 > 0 and self.stats["rate_limited"] >= self.abort_after_429:
                if not self._abort_event.is_set():
                    self._abort_event.set()
                    triggered = True
        return triggered

    def report_other_failed(self) -> None:
        with self._lock:
            self.stats["other_failed"] += 1
            self._success_streak = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {"interval": self.interval, **self.stats}


def download_with_retry(
    episode_id: str,
    out_dir: Path,
    controller: RateController,
    abort_event: threading.Event,
    rate_limit_backoff: list[float],
    max_retries: int,
) -> tuple[str, str, str | None]:
    """(episode_id, status, error_message) を返す。status は "success"/"failed"/"aborted"。

    429は他の失敗と区別し、リトライ回数(max_retries)を消費しない(サーバ都合の
    ため、こちらの問題として3回で諦める対象にはしない)。429を受けるたびに
    `rate_limit_backoff` の対応する待ち時間(超過分は最後の値を使い続ける)だけ
    待ってから再試行する。abort_event が立ったら即座に打ち切る。
    """
    other_attempt = 0
    rate_limit_attempt = 0
    while True:
        if abort_event.is_set():
            return episode_id, "aborted", None
        controller.wait_before_request()
        if abort_event.is_set():
            return episode_id, "aborted", None
        try:
            download_replay(int(episode_id), out_dir)
            controller.report_success()
            return episode_id, "success", None
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or str(e)).strip()
            if _is_rate_limited(stderr):
                controller.report_rate_limited()
                if abort_event.is_set():
                    return episode_id, "aborted", stderr
                rate_limit_attempt += 1
                backoff = rate_limit_backoff[min(rate_limit_attempt - 1, len(rate_limit_backoff) - 1)]
                _interruptible_sleep(backoff, abort_event)
                continue
            controller.report_other_failed()
            other_attempt += 1
            if other_attempt >= max_retries:
                return episode_id, "failed", stderr
            _interruptible_sleep(_BACKOFF_BASE * (2 ** (other_attempt - 1)), abort_event)
    # unreachable


def run_download_pool(
    episode_id_list: list[str],
    out_dir: Path,
    controller: RateController,
    abort_event: threading.Event,
    concurrency: int,
    max_retries: int,
    rate_limit_backoff: list[float],
    on_result,
) -> None:
    """episode_id_list を並列ダウンロードし、完了ごとに on_result(episode_id, status, err) を呼ぶ。

    abort_event が立った時点でまだ着手していないタスクは即座に "aborted" として
    on_result に渡される(ThreadPoolExecutor 自体には投げ込み済みだが、実際の
    ダウンロード処理はまだ何もしていない状態で戻ってくる)。
    """
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                download_with_retry, eid, out_dir, controller, abort_event, rate_limit_backoff, max_retries,
            ): eid
            for eid in episode_id_list
        }
        for future in as_completed(futures):
            eid, status, err = future.result()
            on_result(eid, status, err)


def save_episode_list_cache(
    cache_path: Path,
    *,
    run_id: str,
    fetched_at: str,
    leaderboard_snapshot_path: Path,
    episode_ids: list[str],
    episode_meta: dict[str, dict],
    name_to_context: dict[str, dict],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "fetched_at": fetched_at,
                "leaderboard_snapshot_path": str(leaderboard_snapshot_path),
                "episode_ids": episode_ids,
                "episode_meta": episode_meta,
                "name_to_context": name_to_context,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_episode_list_cache(cache_path: Path) -> dict:
    with cache_path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--top", type=int, default=20, help="上位何チームを対象にするか")
    parser.add_argument(
        "--submissions-per-team", type=int, default=1,
        help="各チームにつき何個の提出(新しい順)を辿るか",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=300,
        help="ダウンロードするエピソード数の上限(無制限にしたい場合は非常に大きい値か0以下を指定)",
    )
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "replays"))
    parser.add_argument(
        "--leaderboard-history-dir",
        default=str(Path(__file__).parent / "index" / "leaderboard_history"),
    )
    parser.add_argument(
        "--master-index-path",
        default=str(Path(__file__).parent / "index" / "episodes_master.jsonl"),
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="[2/5]/[3/5](リーダーボード/提出ID/エピソード一覧取得、逐次)の呼び出し間隔(秒)。"
        "ダウンロード本体の間隔は AIMD (--initial-interval 等)が制御するのでここでは使わない",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="並列ダウンロード数(既定2)。Kaggle側のレート制限に何度も引っかかった実績があるため"
        "控えめにしてある。実際の間隔は AIMD が調整する。8を超えると警告",
    )
    parser.add_argument(
        "--index-every", type=int, default=25,
        help="この件数ダウンロードするごとにマスターインデックスへ逐次追記する",
    )
    parser.add_argument(
        "--aimd-success-streak", type=int, default=20,
        help="この件数連続で成功したらリクエスト間隔を10%%短縮する(AIMDの加算的増加)",
    )
    parser.add_argument(
        "--min-interval", type=float, default=0.2, help="リクエスト間隔の下限(秒)",
    )
    parser.add_argument(
        "--max-interval", type=float, default=30.0, help="リクエスト間隔の上限(秒)",
    )
    parser.add_argument(
        "--initial-interval", type=float, default=1.0,
        help="開始時点のリクエスト間隔(秒)。--min-interval〜--max-interval にクランプされる",
    )
    parser.add_argument(
        "--cooldown", type=float, default=60.0,
        help="429を1件でも受けたら全ワーカーをこの秒数だけ停止する",
    )
    parser.add_argument(
        "--rate-limit-backoff", default="60,120,300",
        help="429を受けた際、同じエピソードを再試行するまでの待ち時間(秒)をカンマ区切りで指定。"
        "リトライ回数がリストの長さを超えたら最後の値を使い続ける。この待ち時間はリトライ回数"
        "(--index-every とは別の、失敗3回までのリトライ)を消費しない",
    )
    parser.add_argument(
        "--abort-after-429", type=int, default=30,
        help="429の累計件数がこれに達したら全体を中止する(半分捨てながら走り続けるのを防ぐ)。"
        "0以下で無効化",
    )
    parser.add_argument(
        "--episode-list", default=None,
        help="save_episode_list_cache で保存された JSON のパス。指定すると[2/5]〜[3/5]"
        "(各チームの提出ID・エピソード一覧の逐次取得)を丸ごとスキップしてダウンロードから再開する",
    )
    parser.add_argument(
        "--episode-list-cache-dir",
        default=str(Path(__file__).parent / "index" / "episode_list_cache"),
        help="[1/5]〜[3/5]で解決したエピソード一覧を保存するキャッシュディレクトリ"
        "(--episode-list を指定した場合は書き込まない)",
    )
    args = parser.parse_args()

    if args.concurrency > 8:
        print(
            f"警告: --concurrency={args.concurrency} は8を超えています。"
            f"Kaggle側のレート制限に触れる可能性があるので推奨しません。",
            file=sys.stderr,
        )
    if args.concurrency < 1:
        parser.error("--concurrency は1以上を指定してください")

    try:
        rate_limit_backoff = [float(x) for x in args.rate_limit_backoff.split(",") if x.strip()]
    except ValueError:
        parser.error(f"--rate-limit-backoff の形式が不正です: {args.rate_limit_backoff!r}")
    if not rate_limit_backoff:
        parser.error("--rate-limit-backoff は少なくとも1つの値を指定してください")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_history_dir = Path(args.leaderboard_history_dir)
    leaderboard_history_dir.mkdir(parents=True, exist_ok=True)
    master_index_path = Path(args.master_index_path)
    master_index_path.parent.mkdir(parents=True, exist_ok=True)
    episode_list_cache_dir = Path(args.episode_list_cache_dir)

    abort_event = threading.Event()
    controller = RateController(
        initial_interval=args.initial_interval,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        success_streak_target=args.aimd_success_streak,
        cooldown_seconds=args.cooldown,
        abort_after_429=args.abort_after_429,
        abort_event=abort_event,
    )

    if args.episode_list:
        cache_path = Path(args.episode_list)
        print(f"[1/5]〜[3/5] episode-list キャッシュから読み込み: {cache_path}")
        cache = load_episode_list_cache(cache_path)
        run_id = cache["run_id"]
        fetched_at = cache["fetched_at"]
        leaderboard_snapshot_path = Path(cache["leaderboard_snapshot_path"])
        episode_meta: dict[str, dict] = cache["episode_meta"]
        name_to_context: dict[str, dict] = cache["name_to_context"]
        all_episode_ids: list[str] = cache["episode_ids"]
        print(
            f"  run_id={run_id} (発見時), 発見済みエピソード{len(all_episode_ids)}件, "
            f"leaderboard_snapshot={leaderboard_snapshot_path}"
        )
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-fast"
        fetched_at = datetime.now(timezone.utc).isoformat()

        print(f"[1/5] リーダーボード取得: {args.competition} 上位{args.top}チーム (run_id={run_id})")
        leaderboard = fetch_leaderboard(args.competition, args.top)
        for rank, row in enumerate(leaderboard, 1):
            print(f"  #{rank} {row['teamName']} (teamId={row['teamId']}, score={row['score']})")

        leaderboard_snapshot_path = leaderboard_history_dir / f"leaderboard-{run_id}.json"
        leaderboard_snapshot_path.write_text(
            json.dumps(
                {
                    "competition": args.competition,
                    "run_id": run_id,
                    "fetched_at": fetched_at,
                    "top_n": args.top,
                    "leaderboard": [
                        {"rank": rank, **row} for rank, row in enumerate(leaderboard, 1)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        name_to_context = {
            row["teamName"]: {"team_id": row["teamId"], "rank": rank, "score": float(row["score"])}
            for rank, row in enumerate(leaderboard, 1)
        }

        print(f"[2/5] 各チームの提出IDを取得(チームごと最新{args.submissions_per_team}件、逐次)")
        submission_ids: list[int] = []
        for row in leaderboard:
            team_id = row["teamId"]
            try:
                subs = fetch_team_submission_ids(team_id)
            except subprocess.CalledProcessError as e:
                print(f"  警告: team {team_id} ({row['teamName']}) の提出取得に失敗: {e.stderr}", file=sys.stderr)
                continue
            submission_ids.extend(subs[: args.submissions_per_team])
            if args.sleep > 0:
                time.sleep(args.sleep)

        print(f"[3/5] エピソード一覧を取得({len(submission_ids)}件の提出から、逐次)")
        episode_meta = {}
        for sub_id in submission_ids:
            try:
                episodes = fetch_episodes(sub_id)
            except subprocess.CalledProcessError as e:
                print(f"  警告: submission {sub_id} のエピソード取得に失敗: {e.stderr}", file=sys.stderr)
                continue
            for ep in episodes:
                episode_meta[str(ep["id"])] = ep
            if args.sleep > 0:
                time.sleep(args.sleep)

        all_episode_ids = sorted(episode_meta.keys(), key=int)

        cache_path = episode_list_cache_dir / f"{run_id}.json"
        save_episode_list_cache(
            cache_path,
            run_id=run_id,
            fetched_at=fetched_at,
            leaderboard_snapshot_path=leaderboard_snapshot_path,
            episode_ids=all_episode_ids,
            episode_meta=episode_meta,
            name_to_context=name_to_context,
        )
        print(f"  エピソード一覧をキャッシュに保存: {cache_path} (次回は --episode-list で再利用可能)")

    episode_id_list = all_episode_ids
    if args.max_episodes is not None and args.max_episodes > 0:
        episode_id_list = episode_id_list[-args.max_episodes:]

    print(
        f"[4/5] リプレイ{len(episode_id_list)}件を並列ダウンロード "
        f"(concurrency={args.concurrency}, index-every={args.index_every}, "
        f"initial-interval={controller.interval:.2f}s) -> {out_dir}"
    )

    already_indexed = load_existing_episode_ids(master_index_path)
    downloaded: list[str] = []
    failed: list[tuple[str, str]] = []
    aborted: list[str] = []
    n_skipped_existing = 0
    n_done = 0
    total = len(episode_id_list)
    start_time = time.monotonic()
    pending_batch: list[str] = []

    def flush_batch(batch: list[str]) -> None:
        if not batch:
            return
        new_rows = build_master_rows(
            run_id=run_id,
            fetched_at=fetched_at,
            competition=args.competition,
            replays_dir=out_dir,
            episode_meta=episode_meta,
            name_to_context=name_to_context,
            already_indexed=already_indexed,
            episode_ids=batch,
        )
        if new_rows:
            append_master_rows(master_index_path, new_rows)
            for r in new_rows:
                already_indexed.add(r["episode_id"])
        print(f"    -> マスターインデックスに{len(new_rows)}行を逐次追記(バッチ{len(batch)}件分)")

    to_submit: list[str] = []
    for eid in episode_id_list:
        dest = out_dir / f"episode-{eid}-replay.json"
        if dest.exists():
            n_skipped_existing += 1
            downloaded.append(eid)
            pending_batch.append(eid)
            n_done += 1
            continue
        to_submit.append(eid)

    def on_result(eid: str, status: str, err: str | None) -> None:
        nonlocal n_done, pending_batch
        n_done += 1
        if status == "success":
            downloaded.append(eid)
            pending_batch.append(eid)
        elif status == "aborted":
            aborted.append(eid)
        else:
            failed.append((eid, err or "unknown error"))
            print(f"  警告: episode {eid} のリプレイ取得に失敗(3回リトライ後): {err}", file=sys.stderr)

        elapsed_min = max((time.monotonic() - start_time) / 60.0, 1e-9)
        rate = n_done / elapsed_min
        snap = controller.snapshot()
        print(
            f"  ({n_done}/{total}) episode {eid} {status.upper()} | 実測レート={rate:.1f}件/分 | "
            f"間隔={snap['interval']:.2f}s | 累計 成功={snap['success']} 429={snap['rate_limited']} "
            f"その他失敗={snap['other_failed']}"
        )

        if len(pending_batch) >= args.index_every:
            flush_batch(pending_batch)
            pending_batch = []

    run_download_pool(
        to_submit,
        out_dir,
        controller,
        abort_event,
        args.concurrency,
        _MAX_RETRIES,
        rate_limit_backoff,
        on_result,
    )

    flush_batch(pending_batch)
    pending_batch = []

    elapsed_min = max((time.monotonic() - start_time) / 60.0, 1e-9)

    if abort_event.is_set():
        snap = controller.snapshot()
        print(
            f"[中止] 429の累計件数が --abort-after-429={args.abort_after_429} に達したため中止しました。\n"
            f"  取得済み(このバッチ): 成功{len(downloaded)}件、失敗{len(failed)}件、"
            f"未着手のまま中止{len(aborted)}件\n"
            f"  マスターインデックスへの追記: 完了済み(--index-every 未満の端数は次段の[5/5]でも再走査)"
        )
    else:
        print(
            f"  ダウンロード完了: 成功{len(downloaded)}件(うち既取得済みスキップ{n_skipped_existing}件)、"
            f"失敗{len(failed)}件、所要{elapsed_min:.2f}分、実測レート={len(downloaded) / elapsed_min:.1f}件/分"
        )

    print(f"[5/5] 取りこぼしが無いか全体スキャンでマスターインデックスを更新 -> {master_index_path}")
    # 逐次追記はバッチ単位なので取りこぼしが無いはずだが、念のため replays_dir 全体を
    # 再走査して漏れを追記する(load_existing_episode_ids をディスクから読み直すことで
    # 上の逐次追記の結果も正しく already_indexed に反映される)。中止した場合もここは
    # 必ず実行し、実際にダウンロードできた分の取りこぼしが無いようにする。
    already_indexed = load_existing_episode_ids(master_index_path)
    final_rows = build_master_rows(
        run_id=run_id,
        fetched_at=fetched_at,
        competition=args.competition,
        replays_dir=out_dir,
        episode_meta=episode_meta,
        name_to_context=name_to_context,
        already_indexed=already_indexed,
    )
    append_master_rows(master_index_path, final_rows)
    if final_rows:
        print(f"  最終スキャンで{len(final_rows)}行を追加で追記しました(取りこぼし分)")
    else:
        print("  最終スキャンでの追加追記はありませんでした(逐次追記で全て反映済み)")

    snap = controller.snapshot()
    print(
        f"完了: リプレイ{len(downloaded)}件を保存(失敗{len(failed)}件、中止による未着手{len(aborted)}件)、"
        f"実測レート={len(downloaded) / elapsed_min:.1f}件/分、最終間隔={snap['interval']:.2f}s、"
        f"リーダーボードスナップショットを {leaderboard_snapshot_path} に保存しました"
    )
    if abort_event.is_set():
        print(
            "中止フラグが立っています。残りのエピソードは未取得です。"
            "エピソード一覧はキャッシュ済みなので、間隔を広げるかクールダウンを伸ばしてから "
            "--episode-list で再開してください。",
            file=sys.stderr,
        )
    if failed:
        print("失敗したエピソード一覧:")
        for eid, err in failed:
            print(f"  episode {eid}: {err}")


if __name__ == "__main__":
    main()
