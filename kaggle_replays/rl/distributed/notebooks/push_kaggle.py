"""Kaggle 側の worker を CLI だけで動かす(ブラウザ不要)。

    python push_kaggle.py --run-dir ../../runs/<run_id>

やること:
  1. ptcg_repo.zip(コード一式)と run_<id>_v<N>.zip(設定+モデル)を作る
  2. 非公開 Dataset として上げる(2回目以降は新バージョンとして更新)
  3. kaggle_worker.ipynb を Notebook として push して実行させる
  4. 終わるまで待って、出力(kaggle.npz)を <run_dir>/shards/v<N>/ へ回収する

事前に一度だけ認証が必要:
    kaggle auth login          # ブラウザで OAuth
  または https://www.kaggle.com/settings/api で token を作って ~/.kaggle/kaggle.json に置く
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

DATASET_SLUG = "ptcg-distributed-selfplay"
KERNEL_SLUG = "ptcg-worker-kaggle"


def kaggle(*args, check=True, capture=True):
    # Kaggle CLI は UTF-8 で出力するが、Windows の既定は cp932。明示しないと
    # 日本語を含む出力(ノートブック名など)で復号に失敗する。
    r = subprocess.run(["kaggle", *args], capture_output=capture, text=True,
                       encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        print(r.stdout); print(r.stderr)
        raise SystemExit(f"kaggle {' '.join(args)} が失敗")
    return r


def detect_username(explicit: str | None) -> str:
    """ユーザー名だけを取る。OAuth ログイン(kaggle auth login)だと kaggle.json は作られず
    credentials.json になるので、資格情報ファイルは読まずに config view から拾う。"""
    if explicit:
        return explicit
    r = kaggle("config", "view", check=False)
    for line in (r.stdout or "").splitlines():
        if "username" in line.lower():
            name = line.split(":", 1)[-1].strip()
            if name and name.lower() != "none":
                return name
    raise SystemExit(
        "Kaggle のユーザー名が取れない。`kaggle auth login` を済ませるか --username で渡す。")


def build_payload(run_dir: Path, gen: int, stage: Path) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    repo_zip = stage / "ptcg_repo.zip"
    # コミット済みのファイル一式。
    subprocess.run(["git", "archive", "--format=zip", "HEAD", "-o", str(repo_zip)],
                   cwd=C.REPO_ROOT, check=True)
    # distributed/ 自体がまだ未コミットだと git archive に入らない。worker が動かないので、
    # 作業ツリーから明示的に足す(コミット済みなら同じ内容で上書きされるだけ)。
    dist = C.REPO_ROOT / "kaggle_replays" / "rl" / "distributed"
    added = 0
    with zipfile.ZipFile(repo_zip, "a", zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for p in dist.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or ".kaggle_stage" in p.parts:
                continue
            arc = p.relative_to(C.REPO_ROOT).as_posix()
            if arc not in have:
                z.write(p, arc)
                added += 1
    if added:
        print(f"  distributed/ から未コミットの {added} ファイルを追加")
    run_id = C.load_run(run_dir)["run_id"]
    for old in stage.glob("run_*.zip"):
        old.unlink()
    with zipfile.ZipFile(stage / f"run_{run_id}_v{gen}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(C.run_json_path(run_dir), "run/run.json")
        z.write(C.model_path(run_dir, gen), f"run/models/model_v{gen}.json")
    print(f"  payload: {repo_zip.stat().st_size/1e6:.1f} MB + run v{gen}")


def write_kernel_notebook(src: Path, dest: Path, workers: int, start_method: str) -> None:
    """本番収集セルの --workers / --start-method を、今回指定した値に差し替えて書き出す。"""
    doc = json.loads(src.read_text(encoding="utf-8"))
    patched = 0
    for cell in doc["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            if "worker.py --run-dir" in line:
                cell["source"][i] = line.replace("--workers 4", f"--workers {workers}")
                patched += 1
            elif "--start-method spawn" in line and "worker.py" not in line:
                cell["source"][i] = line.replace("--start-method spawn",
                                                 f"--start-method {start_method}")
                patched += 1
    if patched < 2:
        raise SystemExit(
            f"kaggle_worker.ipynb の収集セルを書き換えられなかった(patched={patched})。"
            "ノートブックを編集したなら push_kaggle.py 側も合わせる。")
    # ensure_ascii=True で書く。Kaggle CLI は push するファイルをシステム既定の文字コード
    # (Windows では cp932)で読むため、日本語がそのまま入っていると復号に失敗する。
    # JSON のエスケープにしておけばファイルは純 ASCII になり、表示内容は変わらない。
    dest.write_text(json.dumps(doc, ensure_ascii=True, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--username", default=None)
    ap.add_argument("--workers", type=int, default=4, help="Kaggle 側の並列プロセス数")
    ap.add_argument("--start-method", default="spawn", choices=["spawn", "fork", "default"])
    ap.add_argument("--timeout", type=int, default=3600, help="完了待ちの上限(秒)")
    ap.add_argument("--dataset-wait", type=int, default=90,
                    help="Dataset の新バージョンが反映されるまでの待ち時間(秒)")
    ap.add_argument("--no-wait", action="store_true", help="push だけして待たない")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run = C.load_run(run_dir)
    gen = run["generation"]
    if "kaggle" not in run["workers"]:
        raise SystemExit(f"run.json の workers に 'kaggle' が無い: {run['workers']}")
    user = detect_username(args.username)
    stage = run_dir / ".kaggle_stage"
    # Dataset に上げるのは data/ の中身だけ。kernel/ や out/ を同じ階層に置くと
    # それらまで Dataset に含まれてしまうので、ディレクトリを分ける。
    ddir = stage / "data"

    print(f"run={run['run_id']} v{gen} -> Kaggle({user})")
    build_payload(run_dir, gen, ddir)

    # --- Dataset(無ければ作成、あればバージョン追加) ---
    meta = {"title": "ptcg distributed selfplay",
            "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "unknown"}]}
    (ddir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    exists = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).returncode == 0
    if exists:
        print("  Dataset を更新(新バージョン)")
        kaggle("datasets", "version", "-p", str(ddir), "-m", f"v{gen}")
    else:
        print("  Dataset を新規作成(非公開)")
        kaggle("datasets", "create", "-p", str(ddir))
    # アップロード直後は処理が終わっておらず、そのまま Notebook を投げると1つ前の版が
    # 添付されて中身が古いまま実行される。`datasets status` は版ごとの状態を返さないので
    # 待つしかない。ノートブック側でも必要ファイルの有無を検査しているので、待ちが
    # 足りなければそこで明示的に落ちる。
    print(f"  Dataset の反映を待つ({args.dataset_wait}s)", flush=True)
    time.sleep(args.dataset_wait)

    # --- Notebook を push ---
    kdir = stage / "kernel"
    kdir.mkdir(exist_ok=True)
    write_kernel_notebook(HERE / "kaggle_worker.ipynb", kdir / "kaggle_worker.ipynb",
                          args.workers, args.start_method)
    kmeta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "ptcg worker kaggle",
        "code_file": "kaggle_worker.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print("  Notebook を push して実行開始")
    kaggle("kernels", "push", "-p", str(kdir))

    if args.no_wait:
        print(f"\nhttps://www.kaggle.com/code/{user}/{KERNEL_SLUG} で進行を確認できる。")
        return

    # --- 完了待ち ---
    t0 = time.time()
    last = ""
    while time.time() - t0 < args.timeout:
        r = kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}", check=False)
        out = (r.stdout or "").strip()
        if out != last:
            print(f"  [{time.time()-t0:5.0f}s] {out}", flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise SystemExit(f"Kaggle 側で失敗: {out}\n"
                             f"  https://www.kaggle.com/code/{user}/{KERNEL_SLUG} でログを確認する。")
        time.sleep(20)
    else:
        raise SystemExit("待ち時間の上限に達した。--no-wait で投げっぱなしにして後で回収する。")

    # --- 出力を回収 ---
    dest = C.shard_dir(run_dir, gen)
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(stage / "out"))
    got = list((stage / "out").glob("*.npz"))
    if not got:
        raise SystemExit(f"出力に .npz が無い: {stage/'out'}")
    for p in got:
        shutil.copyfile(p, dest / p.name)
        print(f"  回収: {dest / p.name} ({p.stat().st_size/1e6:.1f} MB)")
    print(f"\n完了。learner.py --run-dir {run_dir} --status で揃い具合を確認できる。")


if __name__ == "__main__":
    main()
