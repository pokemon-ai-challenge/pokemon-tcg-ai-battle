"""手動ダウンロードした日次Dataset の zip から、リプレイを取り込む。

Kaggle の API がレート制限に入ったため、日次Dataset をブラウザで落として取り込む経路。

zip の中身は ``<episode_id>.json`` だが、**我々の既存コード(``_common.py`` の
``build_master_rows`` 等)は ``episode-<id>-replay.json`` を前提にしている**ので、
展開時にリネームする。ここを間違えると後段が全部壊れる。

順位は ``index/leaderboard_history/`` の最新スナップショットとチーム名で結合して付ける
(``repair_master_index.py`` と同じ方式)。zip 側には順位情報が無いため。

1件あたり約4.9MB。空き容量を見ながら ``--max-files`` で刻んで取り込むこと。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import append_master_rows, build_master_rows, load_existing_episode_ids  # noqa: E402
from repair_master_index import find_latest_leaderboard, load_name_to_context  # noqa: E402

REPLAYS = _HERE / "replays"
MASTER = _HERE / "index" / "episodes_master.jsonl"


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+", help="日次Dataset の zip パス")
    ap.add_argument("--max-files", type=int, default=6000, help="取り込む上限(合計)")
    ap.add_argument("--min-free-gb", type=float, default=15.0,
                    help="空き容量がこれを下回ったら中止する")
    ap.add_argument("--index-every", type=int, default=200)
    ap.add_argument("--replays-dir", default=str(REPLAYS))
    ap.add_argument("--master-index-path", default=str(MASTER))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.replays_dir)
    out.mkdir(parents=True, exist_ok=True)
    master = Path(args.master_index_path)

    have = {p.name.split("-")[1] for p in out.glob("episode-*-replay.json")}
    print(f"手元 {len(have)} 件 / 空き {free_gb(out):.1f} GB")

    lb = find_latest_leaderboard(_HERE / "index" / "leaderboard_history")
    name_to_context, snapshot = load_name_to_context(lb)
    run_id, fetched_at = snapshot["run_id"], snapshot["fetched_at"]
    print(f"順位の出所: {Path(lb).name} ({len(name_to_context)} チーム)")

    todo: list[tuple[Path, str, str]] = []  # (zip, 内部名, episode_id)
    for zp in args.zips:
        zpath = Path(zp)
        if not zpath.exists():
            raise SystemExit(f"zip が無い: {zpath}")
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if not n.endswith(".json"):
                    continue
                eid = Path(n).stem
                if eid in have:
                    continue
                todo.append((zpath, n, eid))
    print(f"取り込み候補 {len(todo)} 件（既存は除外済み）")
    todo = todo[: args.max_files]
    print(f"今回取り込む {len(todo)} 件 ≈ {len(todo) * 4.9 / 1000:.1f} GB")

    if args.dry_run:
        return

    already = load_existing_episode_ids(master)
    batch: list[str] = []
    done = 0
    handles: dict[Path, zipfile.ZipFile] = {}

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        rows = build_master_rows(
            run_id=run_id, fetched_at=fetched_at, competition="pokemon-tcg-ai-battle",
            replays_dir=out, episode_meta={}, name_to_context=name_to_context,
            already_indexed=already, episode_ids=batch,
        )
        append_master_rows(master, rows)
        already.update(r["episode_id"] for r in rows)
        print(f"  [index] +{len(rows)}行")
        batch = []

    try:
        for zpath, inner, eid in todo:
            if free_gb(out) < args.min_free_gb:
                print(f"[中止] 空き容量が {args.min_free_gb} GB を下回った")
                break
            z = handles.get(zpath)
            if z is None:
                z = handles[zpath] = zipfile.ZipFile(zpath)
            dest = out / f"episode-{eid}-replay.json"
            with z.open(inner) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)
            batch.append(eid)
            done += 1
            if len(batch) >= args.index_every:
                flush()
            if done % 200 == 0:
                print(f"  {done}/{len(todo)} 件  空き {free_gb(out):.1f} GB")
        flush()
    finally:
        for z in handles.values():
            z.close()

    print(f"完了: {done} 件を取り込み / 手元 {len(list(out.glob('episode-*-replay.json')))} 件 "
          f"/ 空き {free_gb(out):.1f} GB")


if __name__ == "__main__":
    main()
