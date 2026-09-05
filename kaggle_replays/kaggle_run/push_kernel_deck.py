"""デッキ head-to-head カーネルの投入・監視・回収・マージ(Kaggle CLI ラッパ)。

``push_kernel.py``(ドラパルト測定用)の派生。走らせる本体が ``deck_h2h.py`` に変わり、
シャードのパラメタが ``--arms``(候補デッキ名)と ``--games-per-shard``(1群あたり)になる。

Kaggle は複数カーネルを並列に実行できるので、**1カーネル = 1シャード**(= seed 範囲違い)に
分けて投げ、回収後にマージする。各シャードは全群・先後半々の完結したミニ実験なので、
単純に連結するだけで正しく合算できる。

**提出(kaggle competitions submit)は一切行わない。** Notebook / Dataset の push だけ。

使い方:
    # 0) データセットを作る/更新する
    python kaggle_replays/kaggle_run/push_kernel_deck.py --push-dataset --dataset-dir C:/tmp/ptcg_deck_h2h

    # 1) 投入(5シャード × 各群1200試合 = 群あたり6000試合)
    python kaggle_replays/kaggle_run/push_kernel_deck.py --push --tag d1 --shards 5 \
        --arms aceburn,bulu,baseline --games-per-shard 1200

    # 2) 監視 → 3) 回収 → 4) マージ
    python kaggle_replays/kaggle_run/push_kernel_deck.py --status --tag d1
    python kaggle_replays/kaggle_run/push_kernel_deck.py --fetch  --tag d1
    python kaggle_replays/kaggle_run/push_kernel_deck.py --merge  --tag d1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import deck_h2h  # noqa: E402 - 集計ロジックを共有する(stdlib のみに依存)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATASET_ID = "shogonagashima/ptcg-deck-h2h"
DATASET_SLUG = DATASET_ID.split("/")[1]
USERNAME = DATASET_ID.split("/")[0]
BLOB_NAME = "repo_deckh2h_blob.dat"
DEFAULT_WORK = Path(tempfile.gettempdir()) / "ptcg_deck_h2h_kernels"

# Kaggle CPU カーネルの上限は12時間。集計まで到達させるための安全マージン(11時間)。
DEFAULT_MAX_SECONDS = 39600

# 生成するカーネル本体。**あえて ASCII のみ**(日本語を入れると Kaggle 側で落ちる/化ける)。
KERNEL_TEMPLATE = '''"""Kaggle CPU: deck head-to-head (shard {shard}/{shards}, tag={tag}).

Extracts the repo blob from the attached dataset and runs
kaggle_replays/kaggle_run/deck_h2h.py. Measurement only; no training, no internet,
no competition submission. ASCII-only on purpose.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

TAG = "{tag}"
SHARD = {shard}
ARMS = "{arms}"
BASELINE = "{baseline}"
GAMES = {games}
WORKERS = {workers}
SEED_BASE = {seed_base}
MAX_SECONDS = {max_seconds}
CONFIG = "{config}"
WEIGHTS = "{weights}"

blobs = glob.glob("/kaggle/input/**/{blob}", recursive=True)
assert blobs, "{blob} not found in /kaggle/input"
ROOT = "/kaggle/working/repo"
if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT)
with tarfile.open(blobs[0], "r:gz") as t:
    t.extractall(ROOT)
print("[setup] extracted %s -> %s" % (blobs[0], ROOT), flush=True)
print("[setup] cpu_count=%s python=%s" % (os.cpu_count(), sys.version.split()[0]), flush=True)

out = "/kaggle/working/h2h_%s_s%d.jsonl" % (TAG, SHARD)
cmd = [sys.executable, ROOT + "/kaggle_replays/kaggle_run/deck_h2h.py",
       "--repo-root", ROOT,
       "--arms", ARMS,
       "--baseline", BASELINE,
       "--games", str(GAMES),
       "--workers", str(WORKERS),
       "--seed-base", str(SEED_BASE),
       "--config", CONFIG,
       "--weights", WEIGHTS,
       "--max-seconds", str(MAX_SECONDS),
       "--progress-every", "50",
       "--out", out]
print("[run] " + " ".join(cmd), flush=True)
t0 = time.time()
r = subprocess.run(cmd, cwd=ROOT)
print("[done] returncode=%d elapsed=%.0fs" % (r.returncode, time.time() - t0), flush=True)

summary = out.replace(".jsonl", ".summary.json")
if os.path.exists(summary):
    print("[summary] " + open(summary).read()[:6000], flush=True)
else:
    print("[summary] MISSING (the run did not finish)", flush=True)

# Whatever stays in /kaggle/working becomes the kernel output; drop the extracted repo.
shutil.rmtree(ROOT, ignore_errors=True)
'''


def kaggle_cmd() -> str:
    exe = shutil.which("kaggle")
    if not exe:
        raise SystemExit("kaggle CLI が見つかりません(pip install kaggle / PATH を確認)")
    return exe


def run_kaggle(args: list[str], check: bool = True,
               cwd: Path | None = None) -> subprocess.CompletedProcess:
    # 安全弁: このスクリプトからコンペ提出を絶対に行わない。
    if args and args[0] == "competitions" and "submit" in args:
        raise SystemExit("提出は禁止されています(このスクリプトは Notebook/Dataset の push のみ)")
    cmd = [kaggle_cmd(), *args]
    print("$ " + " ".join(cmd) + (f"   (cwd={cwd})" if cwd else ""), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(cwd) if cwd else None)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.strip(), flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle コマンドが失敗しました (rc={proc.returncode})")
    return proc


def slug(tag: str, shard: int) -> str:
    return f"ptcg-h2h-{tag}-s{shard}"


def run_record_path(work: Path, tag: str) -> Path:
    return work / f"_run_{tag}.json"


def push_dataset(dataset_dir: Path) -> None:
    """データセットが未作成なら create、あれば version で更新する。

    ハマり所: Kaggle CLI は ``-p <絶対パス>`` を渡すとアップロードキャッシュのパスを
    作らずに落ちる。**データセットディレクトリを cwd にして ``-p`` を付けずに実行する**。
    """
    if not (dataset_dir / "dataset-metadata.json").exists():
        raise SystemExit(f"dataset-metadata.json がありません: {dataset_dir}"
                         " (先に build_dataset_deck.py を実行)")
    listed = run_kaggle(["datasets", "list", "--mine", "-s", DATASET_SLUG], check=False)
    exists = DATASET_ID in (listed.stdout or "")
    if exists:
        run_kaggle(["datasets", "version",
                    "-m", f"update {time.strftime('%Y-%m-%d %H:%M:%S')}", "--dir-mode", "skip"],
                   cwd=dataset_dir)
    else:
        run_kaggle(["datasets", "create", "--dir-mode", "skip"], cwd=dataset_dir)
    print("[dataset] 反映には数十秒かかることがある(カーネル投入前に ready を確認すること)")


def write_kernel(work: Path, tag: str, shard: int, shards: int, arms: str, baseline: str,
                 games: int, workers: int, seed_base: int, max_seconds: int,
                 config: str, weights: str) -> Path:
    """1シャードぶんのカーネルディレクトリ(script + metadata)を作る。"""
    name = slug(tag, shard)
    kdir = work / name
    kdir.mkdir(parents=True, exist_ok=True)
    script = KERNEL_TEMPLATE.format(
        tag=tag, shard=shard, shards=shards, arms=arms, baseline=baseline, games=games,
        workers=workers, seed_base=seed_base, max_seconds=max_seconds, config=config,
        weights=weights, blob=BLOB_NAME)
    (kdir / f"{name}.py").write_text(script, encoding="ascii")
    meta = {
        "id": f"{USERNAME}/{name}",
        "title": name,
        "code_file": f"{name}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [DATASET_ID],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return kdir


def cmd_push(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    refs = []
    for shard in range(args.shards):
        seed_base = args.seed_base + shard * 1_000_000  # シャード間で seed を必ず分離する
        kdir = write_kernel(work, args.tag, shard, args.shards, args.arms, args.baseline,
                            args.games_per_shard, args.workers, seed_base,
                            args.max_seconds, args.config, args.weights)
        ref = f"{USERNAME}/{slug(args.tag, shard)}"
        refs.append({"ref": ref, "shard": shard, "seed_base": seed_base, "dir": str(kdir)})
        if args.dry_run:
            print(f"[dry-run] would push {ref} ({kdir})")
            continue
        run_kaggle(["kernels", "push", "-p", str(kdir)])
    record = {
        "tag": args.tag,
        "arms": args.arms,
        "baseline": args.baseline,
        "games_per_shard": args.games_per_shard,
        "shards": args.shards,
        "workers": args.workers,
        "config": args.config,
        "weights": args.weights,
        "max_seconds": args.max_seconds,
        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kernels": refs,
    }
    run_record_path(work, args.tag).write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    n_arms = len([a for a in args.arms.split(",") if a.strip()])
    total = args.shards * args.games_per_shard
    print(f"\n投入: {len(refs)} カーネル / 1群あたり合計 {total} 試合 "
          f"({args.games_per_shard} × {args.shards} シャード) × {n_arms} 群")
    print(f"記録: {run_record_path(work, args.tag)}")


def load_record(work: Path, tag: str) -> dict:
    path = run_record_path(work, tag)
    if not path.exists():
        raise SystemExit(f"実行記録がありません: {path} (先に --push)")
    return json.loads(path.read_text(encoding="utf-8"))


def kernel_status(ref: str) -> str:
    proc = run_kaggle(["kernels", "status", ref], check=False)
    text = ((proc.stdout or "") + (proc.stderr or "")).lower()
    for state in ("complete", "error", "cancelacknowledged", "running", "queued"):
        if state in text:
            return state
    return "unknown"


def cmd_status(args: argparse.Namespace) -> None:
    record = load_record(Path(args.work).resolve(), args.tag)
    deadline = time.time() + args.poll_seconds
    while True:
        states = {k["ref"]: kernel_status(k["ref"]) for k in record["kernels"]}
        done = all(s in ("complete", "error", "cancelacknowledged") for s in states.values())
        print(f"[{time.strftime('%H:%M:%S')}] " +
              " ".join(f"{r.split('/')[-1]}={s}" for r, s in states.items()), flush=True)
        if done or args.poll_seconds <= 0 or time.time() > deadline:
            break
        time.sleep(args.poll_interval)


def cmd_fetch(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    record = load_record(work, args.tag)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (work / f"out_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for kernel in record["kernels"]:
        dest = out_dir / kernel["ref"].split("/")[-1]
        dest.mkdir(parents=True, exist_ok=True)
        run_kaggle(["kernels", "output", kernel["ref"], "-p", str(dest)], check=False)
    files = sorted(out_dir.rglob("*.jsonl"))
    print(f"\n回収: {len(files)} JSONL -> {out_dir}")
    for f in files:
        print(f"  {f} ({f.stat().st_size / 1e6:.2f} MB)")


def cmd_merge(args: argparse.Namespace) -> None:
    work = Path(args.work).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (work / f"out_{args.tag}")
    files = sorted(out_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"JSONL が見つかりません: {out_dir}")
    records: list[dict] = []
    metas: list[dict] = []
    for path in files:
        recs, ms = deck_h2h.load_records(path)
        print(f"  {path.name}: {len(recs)} games")
        records.extend(recs)
        metas.extend(ms)
    # シャード間で seed が重複していないことを確認する(重複=同一試合の二重計上)。
    keys = [(r["arm"], r["seed"]) for r in records]
    if len(set(keys)) != len(keys):
        print(f"[warn] 重複した (arm, seed) が {len(keys) - len(set(keys))} 件あります"
              "(同じシャードを二重に回収した可能性)")
    summary = deck_h2h.aggregate(records, {"merged_from": [f.name for f in files],
                                           "shard_metas": metas})
    deck_h2h.print_summary(summary)
    dest = out_dir / f"merged_{args.tag}.summary.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmerged summary -> {dest}")
    if args.save_to:
        target = Path(args.save_to)
        if not target.is_absolute():
            target = _ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"repo results -> {target}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push-dataset", action="store_true", help="データセットを作成/更新する")
    ap.add_argument("--push", action="store_true", help="カーネルを投入する")
    ap.add_argument("--status", action="store_true", help="ステータスをポーリングする")
    ap.add_argument("--fetch", action="store_true", help="出力を回収する")
    ap.add_argument("--merge", action="store_true", help="回収済み JSONL をマージして集計する")

    ap.add_argument("--tag", default="d1", help="実験の識別子(カーネル slug に入る)")
    ap.add_argument("--shards", type=int, default=5, help="分割するカーネル本数")
    ap.add_argument("--games-per-shard", type=int, default=1200,
                    help="1シャード・1群あたりの試合数(合計 = これ × --shards)")
    ap.add_argument("--arms", default="aceburn,bulu,baseline",
                    help="候補デッキ名(baseline を入れると self-mirror 対照群になる)")
    ap.add_argument("--baseline", default=deck_h2h.BASELINE, help="相手デッキ名")
    ap.add_argument("--workers", type=int, default=4, help="カーネル内の並列数(Kaggle CPU は4コア)")
    ap.add_argument("--seed-base", type=int, default=1300000, help="シャード0の seed 基点")
    ap.add_argument("--config", default=deck_h2h.CONFIG, help="両陣営の config")
    ap.add_argument("--weights", default=deck_h2h.WEIGHTS, help="両陣営の重み(リポジトリ相対)")
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                    help=f"カーネル内の測定時間上限(default: {DEFAULT_MAX_SECONDS}s = 11h)")

    ap.add_argument("--dataset-dir", default=str(Path(tempfile.gettempdir()) / "ptcg_deck_h2h"),
                    help="build_dataset_deck.py の出力ディレクトリ")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="カーネル生成/回収の作業ディレクトリ")
    ap.add_argument("--out-dir", default=None, help="回収先(default: <work>/out_<tag>)")
    ap.add_argument("--save-to", default=None,
                    help="--merge の結果をリポジトリ内にも保存する(例: "
                         "kaggle_replays/_kaggle_deck_h2h_results.json)")
    ap.add_argument("--poll-seconds", type=float, default=0.0,
                    help="--status で完了までポーリングする最大秒数(0=1回だけ見る)")
    ap.add_argument("--poll-interval", type=float, default=60.0, help="ポーリング間隔(秒)")
    ap.add_argument("--dry-run", action="store_true", help="--push でカーネルを生成するだけ")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not any([args.push_dataset, args.push, args.status, args.fetch, args.merge]):
        raise SystemExit("--push-dataset / --push / --status / --fetch / --merge のどれかを指定")
    if args.push_dataset:
        push_dataset(Path(args.dataset_dir).resolve())
    if args.push:
        cmd_push(args)
    if args.status:
        cmd_status(args)
    if args.fetch:
        cmd_fetch(args)
    if args.merge:
        cmd_merge(args)


if __name__ == "__main__":
    main()
