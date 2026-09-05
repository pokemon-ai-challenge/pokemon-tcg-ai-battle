"""fetch_top_episodes.py / fetch_my_episodes.py で共有するヘルパー。"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

KAGGLE = "kaggle"

# --- kaggle API のレート制限(429)対策 -------------------------------------
# 実測: 逐次(1呼び出し約1.5秒 + sleep 0.3 = 約0.55 req/s)では一度も429にならないが、
# 8スレッドで team-submissions を叩くと200件すべてが 429 Too Many Requests になった。
# そこで「全スレッド共有の最小呼び出し間隔」と「429時の指数バックオフ再試行」を入れる。
# min_interval=0(既定)なら従来と完全に同じ挙動。
_RATE_LOCK = threading.Lock()
_RATE_STATE = {"next_slot": 0.0, "min_interval": 0.0, "cooldown_until": 0.0, "consecutive_429": 0}
_RETRY_STATS = {"429": 0, "cooldown_seconds": 0}

# 429 が出たときに「全スレッドまとめて」止まる時間。クォータ(GetEpisodeReplay は
# 1600件ほど連続で落とすと枯渇する)は接続単位ではなくアカウント単位なので、
# 1スレッドだけ待たせても他が叩き続ける限り回復しない。連続回数で伸ばす。
_COOLDOWN_STEPS = (60.0, 180.0, 420.0, 900.0, 900.0)


def set_min_interval(seconds: float) -> None:
    """kaggle CLI 呼び出しの最小間隔(秒)を設定する。並列取得時に使う。"""
    _RATE_STATE["min_interval"] = max(0.0, float(seconds))


def get_retry_stats() -> dict[str, int]:
    """429で再試行した回数などの累計を返す(実行末尾のサマリ表示用)。"""
    return dict(_RETRY_STATS)


def _rate_gate() -> None:
    """全スレッド共有のトークンスロット。min_interval 秒に1回だけ通し、
    クールダウン中(直近に429を食らった)は全スレッドをまとめて待たせる。"""
    min_interval = _RATE_STATE["min_interval"]
    while True:
        with _RATE_LOCK:
            now = time.monotonic()
            remaining_cooldown = _RATE_STATE["cooldown_until"] - now
            if remaining_cooldown <= 0:
                slot = max(now, _RATE_STATE["next_slot"])
                _RATE_STATE["next_slot"] = slot + min_interval
                wait = slot - now
                break
        # クールダウンはロックを持たずに待つ(他スレッドも同じ地点で足並みを揃える)
        time.sleep(min(remaining_cooldown, 5.0))
    if wait > 0:
        time.sleep(wait)


def _enter_cooldown() -> float:
    """429を受けたので全スレッド共通のクールダウンに入る。戻り値は待つ秒数。"""
    with _RATE_LOCK:
        _RETRY_STATS["429"] += 1
        _RATE_STATE["consecutive_429"] += 1
        step = _COOLDOWN_STEPS[min(_RATE_STATE["consecutive_429"], len(_COOLDOWN_STEPS)) - 1]
        pause = step + random.uniform(0, 5.0)
        now = time.monotonic()
        already = max(0.0, _RATE_STATE["cooldown_until"] - now)
        if pause > already:
            _RATE_STATE["cooldown_until"] = now + pause
            _RETRY_STATS["cooldown_seconds"] += int(pause - already)
            return pause
        return already


def run_kaggle_cli(args: list[str], retries: int = 8) -> subprocess.CompletedProcess:
    """kaggle CLI を1回呼ぶ。429 のときは全スレッド共通のクールダウンを挟んで再試行する。

    429以外の失敗は従来どおり CalledProcessError を送出する(呼び出し側が
    「このチームだけスキップして続行」等の判断をしているため握り潰さない)。
    """
    last: subprocess.CompletedProcess | None = None
    for attempt in range(retries + 1):
        _rate_gate()
        last = subprocess.run([KAGGLE, *args], capture_output=True, encoding="utf-8")
        if last.returncode == 0:
            with _RATE_LOCK:
                _RATE_STATE["consecutive_429"] = 0
            return last
        combined = (last.stdout or "") + (last.stderr or "")
        if "429" in combined and attempt < retries:
            pause = _enter_cooldown()
            print(
                f"  [rate-limit] 429を受信。全スレッドを{pause:.0f}秒クールダウンします"
                f"(通算{_RETRY_STATS['429']}回、試行{attempt + 1}/{retries})",
                flush=True,
            )
            continue
        break
    assert last is not None
    raise subprocess.CalledProcessError(
        last.returncode, [KAGGLE, *args], output=last.stdout, stderr=last.stderr
    )


def run_kaggle_json(args: list[str]):
    obj, _token = run_kaggle_json_with_page_token(args)
    return obj


def run_kaggle_json_with_page_token(args: list[str]) -> tuple[object, str | None]:
    """`--format json` の出力をパースし、あわせて "Next Page Token = ..." 行があれば返す。

    `kaggle competitions leaderboard -s` 等、ページネーション対応のサブコマンドは
    JSON本体の前に "Next Page Token = <token>" という非JSON行を出力する
    (--page-token に渡すことで次ページを取得できる)。トークンが出力されない
    (最終ページ等)場合は None を返す。
    """
    result = run_kaggle_cli([*args, "--format", "json"])
    # 一部のサブコマンドは JSON の前後に "Next Page Token = ..." や
    # 使い方ヒント等の非JSON行を stdout に混ぜて出力するため、
    # JSON開始位置から raw_decode して末尾の余計な文字列は無視する。
    stdout = result.stdout
    token = None
    m = re.match(r"Next Page Token = (\S+)", stdout)
    if m:
        token = m.group(1)
    start = min((i for i in (stdout.find("["), stdout.find("{")) if i != -1), default=-1)
    if start == -1:
        raise ValueError(f"kaggle CLI の出力からJSONを検出できませんでした: {stdout!r}")
    obj, _ = json.JSONDecoder().raw_decode(stdout[start:])
    return obj, token


def fetch_episodes(submission_id) -> list[dict]:
    """指定した提出に紐づく完了済みエピソードのメタ情報(id/createTime/endTime)を返す。"""
    rows = run_kaggle_json(["competitions", "episodes", str(submission_id)])
    return [row for row in rows if row.get("state") == "EpisodeState.COMPLETED"]


def download_replay(episode_id: int, out_dir: Path) -> Path:
    dest = out_dir / f"episode-{episode_id}-replay.json"
    if dest.exists():
        return dest
    run_kaggle_cli(["competitions", "replay", str(episode_id), "-p", str(out_dir), "-q"])
    return dest


def download_replays_parallel(
    episode_ids: list[str],
    out_dir: Path,
    workers: int,
    progress_every: int = 25,
    label: str = "",
) -> tuple[list[str], list[str]]:
    """リプレイを並列ダウンロードする。戻り値は (成功した episode_id, 失敗した episode_id)。

    1件あたりの実測は約5秒(kaggle CLI の起動 1〜2秒 + 転送 4MB強)で、ほぼ全部が
    I/O待ちのため GIL の影響は小さく、スレッドプールでほぼ線形に短縮できる。
    数千件を逐次で回すと数時間かかるので、大量取得時は workers>1 を指定する。

    既に out_dir にあるファイルは download_replay() 側の dest.exists() でスキップ
    されるため、中断→再実行しても二重取得にはならない。
    """
    if workers <= 1:
        raise ValueError("workers は2以上を指定してください(逐次版は download_replay を直接使う)")

    done_lock = threading.Lock()
    state = {"done": 0}
    total = len(episode_ids)
    ok: list[str] = []
    failed: list[str] = []

    def _one(eid: str) -> tuple[str, bool]:
        try:
            download_replay(int(eid), out_dir)
            success = True
        except subprocess.CalledProcessError as e:
            print(f"  警告: episode {eid} のリプレイ取得に失敗: {e.stderr}", file=sys.stderr)
            success = False
        with done_lock:
            state["done"] += 1
            if progress_every and state["done"] % progress_every == 0:
                print(f"  {label}{state['done']}/{total} 件完了", flush=True)
        return eid, success

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, eid) for eid in episode_ids]
        for fut in as_completed(futures):
            eid, success = fut.result()
            (ok if success else failed).append(eid)

    return ok, failed


def load_existing_episode_ids(master_path: Path) -> set[str]:
    if not master_path.exists():
        return set()
    ids: set[str] = set()
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["episode_id"])
    return ids


def count_episodes_by_team(master_path: Path) -> dict[int, int]:
    """episodes_master.jsonl から team_id ごとの累積エピソード件数を数える。

    深層取得(fetch_deep_decks.py)で「そのチームは既に episodes-per-team 件以上
    プールにあるからAPI呼び出し自体をスキップする」高速化に使う。team_id が
    null の行はカウントしない。
    """
    counts: dict[int, int] = {}
    if not master_path.exists():
        return counts
    with master_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for player in row.get("players", []):
                team_id = player.get("team_id")
                if team_id is None:
                    continue
                counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def build_master_rows(
    run_id: str,
    fetched_at: str,
    competition: str,
    replays_dir: Path,
    episode_meta: dict[str, dict],
    name_to_context: dict[str, dict],
    already_indexed: set[str],
    episode_ids: list[str] | None = None,
) -> list[dict]:
    """未インデックスのリプレイから episodes_master.jsonl の行を作る。

    name_to_context: チーム名 -> {"team_id", "rank", "score"}。
    対応表に存在しないチーム(取得範囲外だった相手)は team_id/rank/score が null になる。

    episode_ids を指定した場合はその一覧だけを対象にする(replays_dir 全体を
    毎回スキャンするコストを避けたい呼び出し元向け)。省略時は従来どおり
    replays_dir 内の episode-*-replay.json を全走査する。
    """
    if episode_ids is not None:
        candidates = [
            (eid, replays_dir / f"episode-{eid}-replay.json") for eid in episode_ids
        ]
    else:
        candidates = [
            (replay_path.stem.split("-")[1], replay_path)
            for replay_path in sorted(replays_dir.glob("episode-*-replay.json"))
        ]
    rows = []
    for episode_id, replay_path in candidates:
        if episode_id in already_indexed:
            continue
        if not replay_path.exists():
            continue
        with replay_path.open(encoding="utf-8") as f:
            replay = json.load(f)
        team_names = replay.get("info", {}).get("TeamNames", [None, None])
        meta = episode_meta.get(episode_id, {})
        players = []
        for player_index, team_name in enumerate(team_names):
            ctx = name_to_context.get(team_name, {})
            players.append(
                {
                    "player_index": player_index,
                    "team_name": team_name,
                    "team_id": ctx.get("team_id"),
                    "rank_at_fetch": ctx.get("rank"),
                    "leaderboard_score_at_fetch": ctx.get("score"),
                }
            )
        rows.append(
            {
                "episode_id": episode_id,
                "competition": competition,
                "episode_create_time": meta.get("createTime"),
                "episode_end_time": meta.get("endTime"),
                "discovered_run_id": run_id,
                "discovered_at": fetched_at,
                "players": players,
            }
        )
    return rows


def append_master_rows(master_index_path: Path, rows: list[dict]) -> None:
    with master_index_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
