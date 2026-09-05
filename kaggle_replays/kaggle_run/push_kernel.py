"""ドラパルト測定カーネルの投入・監視・回収・マージ(Kaggle CLI ラッパ)。

Kaggle は複数カーネルを並列に実行できるので、**1カーネル = 1シャード**(= 同じ実験の
seed 範囲違い)に分けて投げ、回収後にマージする。各シャードは11アーキ・先後半々の
完結したミニ実験なので、単純に連結するだけで正しく合算できる。

過去に動いた構成(``shogonagashima/ptcg-rl-*`` カーネル)を踏襲:
  kernel_type=script / language=python / enable_internet=false / enable_gpu=false /
  dataset_sources=[ptcg-dragapult-eval] / is_private=true。
  カーネル本体は ``/kaggle/input/**/repo_drapa_blob.dat`` を glob → ``/kaggle/working/repo``
  に展開 → ``subprocess`` で測定スクリプトを実行 → 出力を ``/kaggle/working`` に残す。

使い方:
    # 0) データセットを作る/更新する(build_dataset.py が作ったディレクトリを渡す)
    python kaggle_replays/kaggle_run/push_kernel.py --push-dataset --dataset-dir C:/tmp/ptcg_dragapult_eval

    # 1) 投入(2シャード × 各群200試合)
    python kaggle_replays/kaggle_run/push_kernel.py --push --tag t1 --shards 2 --games-per-shard 200

    # 2) 監視 → 3) 回収 → 4) マージ
    python kaggle_replays/kaggle_run/push_kernel.py --status --tag t1
    python kaggle_replays/kaggle_run/push_kernel.py --fetch  --tag t1
    python kaggle_replays/kaggle_run/push_kernel.py --merge  --tag t1

``--tag`` は実験の識別子(カーネル slug と実行記録のキー)。同じ tag で ``--push`` を
やり直すとカーネルの新バージョンとして上書きされる。
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

import drapa_ablation  # noqa: E402 - 集計ロジックを共有する(stdlib のみに依存)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATASET_ID = "shogonagashima/ptcg-dragapult-eval"
USERNAME = DATASET_ID.split("/")[0]
BLOB_NAME = "repo_drapa_blob.dat"
DEFAULT_WORK = Path(tempfile.gettempdir()) / "ptcg_dragapult_kernels"

# Kaggle CPU カーネルの上限は12時間。測定を打ち切って集計まで到達させるための安全マージン。
DEFAULT_MAX_SECONDS = 39600  # 11時間

# 生成するカーネル本体。**あえて ASCII のみ**にしている(Kaggle 側のログ/エディタで
# 文字化けした過去の事例を避けるため。日本語の説明はこのファイル側に置く)。
KERNEL_TEMPLATE = '''"""Kaggle CPU: dragapult piloting ablation (shard {shard}/{shards}, tag={tag}).

Extracts the repo blob from the attached dataset and runs
kaggle_replays/kaggle_run/drapa_ablation.py. Measurement only; no training, no internet.
ASCII-only on purpose.
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
GROUPS = "{groups}"
GAMES = {games}
WORKERS = {workers}
SEED_BASE = {seed_base}
MAX_SECONDS = {max_seconds}
HORIZON = {horizon}
CONFIG = "{config}"

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

out = "/kaggle/working/drapa_%s_s%d.jsonl" % (TAG, SHARD)
cmd = [sys.executable, ROOT + "/kaggle_replays/kaggle_run/drapa_ablation.py",
       "--repo-root", ROOT,
       "--groups", GROUPS,
       "--games", str(GAMES),
       "--workers", str(WORKERS),
       "--seed-base", str(SEED_BASE),
       "--horizon", str(HORIZON),
       "--config", CONFIG,
       "--max-seconds", str(MAX_SECONDS),
       "--progress-every", "10",
       "--out", out]
print("[run] " + " ".join(cmd), flush=True)
t0 = time.time()
r = subprocess.run(cmd, cwd=ROOT)
print("[done] returncode=%d elapsed=%.0fs" % (r.returncode, time.time() - t0), flush=True)

summary = out.replace(".jsonl", ".summary.json")
if os.path.exists(summary):
    print("[summary] " + open(summary).read()[:4000], flush=True)
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
    return f"ptcg-drapa-{tag}-s{shard}"


def run_record_path(work: Path, tag: str) -> Path:
    return work / f"_run_{tag}.json"


def push_dataset(dataset_dir: Path) -> None:
    """データセットが未作成なら create、あれば version で更新する。

    ハマり所: Kaggle CLI 2.2.3 は ``-p <絶対パス>`` を渡すとアップロードキャッシュの
    パス(``%TEMP%/.kaggle/uploads/C_/tmp/...``)を作らずに落ちる
    (``[Errno 2] No such file or directory``)。**データセットディレクトリを cwd にして
    ``-p`` を付けずに実行する**と回避できるので、こちらで固定する。
    """
    if not (dataset_dir / "dataset-metadata.json").exists():
        raise SystemExit(f"dataset-metadata.json がありません: {dataset_dir}"
                         " (先に build_dataset.py を実行)")
    listed = run_kaggle(["datasets", "list", "--mine", "-s", "ptcg-dragapult-eval"], check=False)
    exists = DATASET_ID in (listed.stdout or "")
    if exists:
        run_kaggle(["datasets", "version",
                    "-m", f"update {time.strftime('%Y-%m-%d %H:%M:%S')}", "--dir-mode", "skip"],
                   cwd=dataset_dir)
    else:
        run_kaggle(["datasets", "create", "--dir-mode", "skip"], cwd=dataset_dir)
    print("[dataset] 反映には数十秒かかることがある(カーネル投入前に Kaggle 側で "
          "ready になっているか確認すること)")


def write_kernel(work: Path, tag: str, shard: int, shards: int, groups: str, games: int,
                 workers: int, seed_base: int, max_seconds: int, horizon: int,
                 config: str) -> Path:
    """1シャードぶんのカーネルディレクトリ(script + metadata)を作る。"""
    name = slug(tag, shard)
    kdir = work / name
    kdir.mkdir(parents=True, exist_ok=True)
    script = KERNEL_TEMPLATE.format(
        tag=tag, shard=shard, shards=shards, groups=groups, games=games, workers=workers,
        seed_base=seed_base, max_seconds=max_seconds, horizon=horizon, config=config,
        blob=BLOB_NAME)
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
        kdir = write_kernel(work, args.tag, shard, args.shards, args.groups,
                            args.games_per_shard, args.workers, seed_base,
                            args.max_seconds, args.horizon, args.config)
        ref = f"{USERNAME}/{slug(args.tag, shard)}"
        refs.append({"ref": ref, "shard": shard, "seed_base": seed_base, "dir": str(kdir)})
        if args.dry_run:
            print(f"[dry-run] would push {ref} ({kdir})")
            continue
        run_kaggle(["kernels", "push", "-p", str(kdir)])
    record = {
        "tag": args.tag,
        "groups": args.groups,
        "games_per_shard": args.games_per_shard,
        "shards": args.shards,
        "workers": args.workers,
        "config": args.config,
        "horizon": args.horizon,
        "max_seconds": args.max_seconds,
        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "kernels": refs,
    }
    run_record_path(work, args.tag).write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    total = args.shards * args.games_per_shard
    print(f"\n投入: {len(refs)} カーネル / 1群あたり合計 {total} 試合 "
          f"({args.games_per_shard} × {args.shards} シャード)")
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
        recs, ms = drapa_ablation.load_records(path)
        print(f"  {path.name}: {len(recs)} games")
        records.extend(recs)
        metas.extend(ms)
    # シャード間で seed が重複していないことを確認する(重複=同一試合の二重計上)。
    keys = [(r["group"], r["arch"], r["seed"]) for r in records]
    if len(set(keys)) != len(keys):
        print(f"[warn] 重複した (group, arch, seed) が {len(keys) - len(set(keys))} 件あります"
              "(同じシャードを二重に回収した可能性)")
    summary = drapa_ablation.aggregate(records, {"merged_from": [f.name for f in files],
                                                 "shard_metas": metas})
    drapa_ablation.print_summary(summary)
    dest = out_dir / f"merged_{args.tag}.summary.json"
    dest.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmerged summary -> {dest}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push-dataset", action="store_true", help="データセットを作成/更新する")
    ap.add_argument("--push", action="store_true", help="カーネルを投入する")
    ap.add_argument("--status", action="store_true", help="ステータスをポーリングする")
    ap.add_argument("--fetch", action="store_true", help="出力を回収する")
    ap.add_argument("--merge", action="store_true", help="回収済み JSONL をマージして集計する")

    ap.add_argument("--tag", default="t1", help="実験の識別子(カーネル slug に入る)")
    ap.add_argument("--shards", type=int, default=2, help="分割するカーネル本数")
    ap.add_argument("--games-per-shard", type=int, default=200,
                    help="1シャード・1群あたりの試合数(合計 = これ × --shards)")
    ap.add_argument("--groups", default="rule,plan", help="drapa_ablation.py に渡す群")
    ap.add_argument("--workers", type=int, default=4, help="カーネル内の並列数(Kaggle CPU は4コア)")
    ap.add_argument("--seed-base", type=int, default=970000, help="シャード0の seed 基点")
    ap.add_argument("--horizon", type=int, default=2, help="DRAGAPULT_PLANNER_HORIZON")
    ap.add_argument("--config", default=drapa_ablation.CONFIG, help="相手 ml_policy の config")
    ap.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_SECONDS,
                    help=f"カーネル内の測定時間上限(default: {DEFAULT_MAX_SECONDS}s = 11h)")

    ap.add_argument("--dataset-dir", default=str(Path(tempfile.gettempdir()) / "ptcg_dragapult_eval"),
                    help="build_dataset.py の出力ディレクトリ")
    ap.add_argument("--work", default=str(DEFAULT_WORK), help="カーネル生成/回収の作業ディレクトリ")
    ap.add_argument("--out-dir", default=None, help="回収先(default: <work>/out_<tag>)")
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
