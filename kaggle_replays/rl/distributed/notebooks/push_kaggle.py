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
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

# Dataset を2つに分ける。コード側(約10MB)はめったに変わらないので、変わったときだけ
# 上げる。毎世代上げるのは run 側(約1MB)だけになり、新バージョンの反映が速くなる
# = 「古い版が添付されたまま実行される」窓が小さくなる。
REPO_DATASET_SLUG = "ptcg-repo"
RUN_DATASET_SLUG = "ptcg-run"


def kernel_slug(worker_id: str) -> str:
    """worker ごとに別の Notebook にする。同じ slug を使い回すと、2台目の push が
    1台目を上書きして走っている実行を潰してしまう。

    Kaggle の slug は英数字とハイフンのみ受け付けるので、それ以外は '-' に落とす。
    """
    safe = "".join(c if c.isalnum() else "-" for c in worker_id.lower()).strip("-")
    return f"ptcg-worker-{safe or 'kaggle'}"


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


def upload_dataset(user: str, slug: str, title: str, ddir: Path, msg: str) -> None:
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "unknown"}]}
    (ddir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    exists = kaggle("datasets", "status", f"{user}/{slug}", check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(ddir), "-m", msg)
    else:
        kaggle("datasets", "create", "-p", str(ddir))


def build_repo_zip(stage: Path) -> str:
    """コード一式の zip を作り、その sha256 を返す。"""
    stage.mkdir(parents=True, exist_ok=True)
    repo_zip = stage / "ptcg_repo.zip"
    # コミット済みのファイル一式。
    subprocess.run(["git", "archive", "--format=zip", "HEAD", "-o", str(repo_zip)],
                   cwd=C.REPO_ROOT, check=True)
    # git archive はコミット済みの版しか出さない。そのままだと
    #   ・コミット後にコードを直しても Kaggle には古い版が届く(黙って古いコードが動く)
    #   ・新しく作ったファイルが届かない(実行時に「見つからない」で落ちる)
    # が起きる。どちらもログに異常が出ないまま結果だけ壊れる質の悪い失敗で、実際に
    # 3回踏んだ。個別にパスを足す方式は漏れるので、**作業ツリーで変更したものは全部
    # 上書きする**。対象は `git status` が挙げるもの(変更済み + 未追跡・無視対象外)。
    changed = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
        cwd=C.REPO_ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    overlay: list[str] = []
    for entry in changed.stdout.split("\0"):
        if len(entry) < 4:
            continue
        rel = entry[3:]
        if entry[:2] == "D " or entry[:2] == " D":
            continue                                  # 削除済みは送らない
        p = C.REPO_ROOT / rel
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        # 学習の成果物(runs/)と提出物の作業コピーは送る必要がない
        if rel.startswith(("kaggle_replays/rl/runs/", "sample_submission/models/",
                           "viewer/", "battle_review_viewer/")):
            continue
        overlay.append(rel)

    tmp_zip = stage / "ptcg_repo.rebuild.zip"
    over = set(overlay)
    with zipfile.ZipFile(repo_zip) as src, \
            zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if item.filename not in over:
                dst.writestr(item, src.read(item.filename))
        for rel in sorted(over):
            dst.write(C.REPO_ROOT / rel, rel)
    tmp_zip.replace(repo_zip)
    print(f"  作業ツリーの変更を上書き: {len(overlay)} ファイル")
    return C.sha256_file(repo_zip)


def build_run_zip(run_dir: Path, gen: int, stage: Path, full_cycle: bool = False) -> None:
    """その世代の設定とモデルだけの小さい zip。毎世代作り直す。

    full_cycle のときは Kaggle 側で PPO 更新まで行うので、critic / optimizer の状態と
    履歴も一緒に送る(送らないと毎世代 critic が初期化され、学習が進まなくなる)。
    """
    stage.mkdir(parents=True, exist_ok=True)
    run_id = C.load_run(run_dir)["run_id"]
    for old in stage.glob("run_*.zip"):
        old.unlink()
    run = C.load_run(run_dir)
    with zipfile.ZipFile(stage / f"run_{run_id}_v{gen}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(C.run_json_path(run_dir), "run/run.json")
        z.write(C.model_path(run_dir, gen), f"run/models/model_v{gen}.json")
        # 自己対戦で過去世代を相手にする場合、その重みも送らないと Kaggle 側で
        # 「相手の重みが見つからない」で落ちる。相手リストが参照するものを全部入れる。
        for opp in run.get("opponents", []):
            spec = opp.get("weights")
            if not spec or not spec.startswith("models/"):
                continue
            src = run_dir / spec
            if not src.is_file():
                raise SystemExit(f"相手 {opp['id']} の重みが無い: {src}")
            arc = f"run/{spec}"
            if arc not in z.namelist():
                z.write(src, arc)
        if full_cycle:
            st = C.trainer_state_path(run_dir, gen)
            if st.exists():
                z.write(st, f"run/state/trainer_v{gen}.pt")
            hist = C.history_path(run_dir)
            if hist.exists():
                z.write(hist, "run/history.jsonl")


def write_kernel_notebook(src: Path, dest: Path, workers: int, start_method: str,
                          gen: int, worker_id: str = "kaggle") -> None:
    """収集セルの --workers / --start-method と、期待する世代を埋めて書き出す。

    世代を埋めるのは、Dataset の古い版が添付されたまま走るのを Notebook 側で検出させるため。
    ファイルの有無だけでは「中身が1世代古い」ことに気づけない。
    """
    doc = json.loads(src.read_text(encoding="utf-8"))
    patched = 0
    for cell in doc["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            # 変数で受ける形(kaggle_cycle)
            if line.startswith("EXPECTED_GEN = "):
                cell["source"][i] = f"EXPECTED_GEN = {gen}\n"
                patched += 1
            elif line.startswith("WORKERS = "):
                cell["source"][i] = f"WORKERS = {workers}            # push_kaggle.py が書き換える\n"
                patched += 1
            elif line.startswith("START_METHOD = "):
                cell["source"][i] = (f"START_METHOD = '{start_method}' "
                                     "# push_kaggle.py が書き換える\n")
                patched += 1
            # シェル行に直接書いてある形(kaggle_worker)
            elif "worker.py --run-dir" in line:
                # worker 名は run.json の workers に登録した名前と一致させる必要がある
                # (learner がシャードの提出者を名前で照合するため)。出力ファイル名も
                # それに合わせる。ここがずれると「未提出の worker がある」で止まる。
                cell["source"][i] = (line.replace("--workers 4", f"--workers {workers}")
                                         .replace("--worker-id kaggle",
                                                  f"--worker-id {worker_id}")
                                         .replace("kaggle.npz", f"{worker_id}.npz"))
                patched += 1
            elif "--start-method spawn" in line and "worker.py" not in line:
                # kaggle_worker.ipynb ではシェル行が2行に折り返されており、出力先の
                # ファイル名はこちら(継続行)に書かれている。
                cell["source"][i] = (line.replace("--start-method spawn",
                                                  f"--start-method {start_method}")
                                         .replace("kaggle.npz", f"{worker_id}.npz"))
                patched += 1
    if patched < 3:
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
    ap.add_argument("--full-cycle", action="store_true",
                    help="収集だけでなく PPO 更新と評価も Kaggle 側で行い、次世代モデルまで"
                         "作らせる。手元の PC は zip の受け渡しだけになる")
    ap.add_argument("--worker-id", default="kaggle",
                    help="run.json の workers に登録した名前。Notebook はこの名前ごとに"
                         "別のものになるので、複数台を同時に走らせられる")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="Dataset のアップロードを飛ばす。2台目以降で使う"
                         "(1台目が上げた同じ世代の Dataset をそのまま使う)")
    ap.add_argument("--fetch-only", action="store_true",
                    help="push せず、その worker の実行完了を待って出力だけ回収する")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run = C.load_run(run_dir)
    gen = run["generation"]
    wid = args.worker_id
    if wid not in run["workers"]:
        raise SystemExit(f"run.json の workers に '{wid}' が無い: {run['workers']}")
    user = detect_username(args.username)
    kslug = kernel_slug(wid)
    stage = run_dir / ".kaggle_stage"
    # Dataset に上げるのはそれぞれのディレクトリの中身だけ。kernel/ や out/ を同じ階層に
    # 置くとそれらまで Dataset に含まれてしまうので、分けておく。
    # Dataset は全 worker で共通(同じ世代の同じモデルを配る)なので1か所のまま。
    # Notebook と回収先だけ worker ごとに分ける(同時に走らせても混ざらないように)。
    repo_dir = stage / "repo"
    rdir = stage / "rundata"

    print(f"run={run['run_id']} v{gen} worker={wid} -> Kaggle({user})")

    if args.fetch_only:
        wait_and_fetch(user, kslug, wid, run_dir, run, gen, stage, args)
        return

    # Kaggle 側は終わったのに反映だけ失敗した場合、もう一度計算させるのは無駄。
    # 手元に残っている結果が次の世代のものなら、それを適用して終わる。
    if args.full_cycle:
        cached = stage / "out" / "result.zip"
        if cached.exists():
            try:
                with zipfile.ZipFile(cached) as z:
                    ok = json.loads(z.read("run.json").decode("utf-8"))["generation"] == gen + 1
            except Exception:
                ok = False
            if ok:
                print("  前回の結果が未反映のまま残っている。計算せずに反映する。")
                new_gen = apply_result(run_dir, gen, cached)
                print(f"  v{new_gen} を反映")
                return

    if args.skip_dataset:
        # 2台目以降。1台目が同じ世代の Dataset を上げ終えている前提で、そこは触らない。
        # 同じ Dataset に複数のプロセスから同時に版を作ると、どの版が添付されるか
        # 分からなくなる(doc の「古い設定のまま学習が進む」を自分で作ることになる)。
        print("  Dataset は 1台目が上げたものを使う(アップロードを省略)")
    else:
        # --- コード側 Dataset(中身が変わったときだけ上げる) ---
        sha = build_repo_zip(repo_dir)
        sha_file = stage / "repo_sha.txt"
        known = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() else ""
        repo_exists = kaggle("datasets", "status",
                             f"{user}/{REPO_DATASET_SLUG}", check=False).returncode == 0
        if sha != known or not repo_exists:
            print(f"  コード Dataset を更新("
                  f"{(repo_dir/'ptcg_repo.zip').stat().st_size/1e6:.1f} MB)")
            upload_dataset(user, REPO_DATASET_SLUG, "ptcg repo", repo_dir, sha[:12])
            sha_file.write_text(sha, encoding="utf-8")
            repo_uploaded = True
        else:
            print("  コード Dataset は変更なし(アップロードを省略)")
            repo_uploaded = False

        # --- run 側 Dataset(毎世代。小さいので反映が速い) ---
        build_run_zip(run_dir, gen, rdir, full_cycle=args.full_cycle)
        print(f"  run Dataset を更新(v{gen}, "
              f"{sum(p.stat().st_size for p in rdir.glob('run_*.zip'))/1e6:.1f} MB)")
        upload_dataset(user, RUN_DATASET_SLUG, "ptcg run", rdir, f"v{gen}")

        wait = args.dataset_wait * (2 if repo_uploaded else 1)
        print(f"  Dataset の反映を待つ({wait}s)", flush=True)
        time.sleep(wait)

    # --- Notebook を push ---
    kdir = stage / f"kernel_{wid}"
    kdir.mkdir(parents=True, exist_ok=True)
    nb_src = HERE / ("kaggle_cycle.ipynb" if args.full_cycle else "kaggle_worker.ipynb")
    write_kernel_notebook(nb_src, kdir / nb_src.name, args.workers, args.start_method,
                          gen, worker_id=wid)
    kmeta = {
        "id": f"{user}/{kslug}",
        "title": f"ptcg worker {wid}",
        "code_file": nb_src.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{user}/{REPO_DATASET_SLUG}", f"{user}/{RUN_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print("  Notebook を push して実行開始")
    kaggle("kernels", "push", "-p", str(kdir))

    if args.no_wait:
        print(f"\nhttps://www.kaggle.com/code/{user}/{kslug} で進行を確認できる。")
        print(f"  完了後に回収する: --worker-id {wid} --fetch-only")
        return

    wait_and_fetch(user, kslug, wid, run_dir, run, gen, stage, args)


def wait_and_fetch(user: str, kslug: str, wid: str, run_dir: Path, run: dict,
                   gen: int, stage: Path, args) -> None:
    """その worker の Notebook の完了を待ち、出力を回収する。"""
    # --- 完了待ち ---
    t0 = time.time()
    last = ""
    while time.time() - t0 < args.timeout:
        r = kaggle("kernels", "status", f"{user}/{kslug}", check=False)
        out = (r.stdout or "").strip()
        if out != last:
            print(f"  [{wid}] [{time.time()-t0:5.0f}s] {out}", flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise SystemExit(f"Kaggle 側で失敗({wid}): {out}\n"
                             f"  https://www.kaggle.com/code/{user}/{kslug} でログを確認する。")
        time.sleep(20)
    else:
        raise SystemExit("待ち時間の上限に達した。--no-wait で投げっぱなしにして後で回収する。")

    # --- 出力を回収 ---
    dest = C.shard_dir(run_dir, gen)
    dest.mkdir(parents=True, exist_ok=True)
    # `kaggle kernels output` は既に同名のファイルがあると取得を飛ばす。前の世代の出力が
    # 残っていると、それをそのまま回収してしまう(中身が1世代古いのに気づけない)ので、
    # 毎回まっさらにしてから落とす。worker ごとに分けるのは、同時に走らせたときに
    # 互いの出力を消し合わないため。
    outdir = stage / f"out_{wid}"
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{kslug}", "-p", str(outdir))

    if args.full_cycle:
        res = outdir / "result.zip"
        if not res.exists():
            raise SystemExit(f"result.zip が無い: {outdir}\n"
                             "  Kaggle 側のログを確認する。")
        new_gen = apply_result(run_dir, gen, res)
        rec = {}
        hist = C.history_path(run_dir)
        if hist.exists():
            for line in hist.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
        print(f"\nv{new_gen} を反映: {C.model_path(run_dir, new_gen)}")
        if rec.get("eval_pool_winrate") is not None:
            print(f"  プール平均 {rec['eval_pool_winrate']:.3f} / "
                  f"固定相手 {rec.get('eval_winrate', float('nan')):.3f} / "
                  f"更新{rec.get('grad_steps')}回 KL {rec.get('approx_kl', 0):.4f}")
        print(f"  次: 同じコマンドをもう一度実行すれば v{new_gen} の世代が回る。")
        return

    got = list(outdir.glob("*.npz"))
    if not got:
        raise SystemExit(f"出力に .npz が無い: {outdir}")
    for p in got:
        # 回収したものが本当にこの世代のものかを、置く前に確かめる。
        _, meta = C.read_shard(p)
        if meta.get("generation") != gen:
            raise SystemExit(
                f"回収した {p.name} の世代が違う(v{meta.get('generation')} != v{gen})。"
                "Kaggle 側の出力が更新されていない可能性がある。")
        shutil.copyfile(p, dest / p.name)
        print(f"  回収: {dest / p.name} ({p.stat().st_size/1e6:.1f} MB) v{meta['generation']}")


def apply_result(run_dir: Path, gen: int, result_zip: Path) -> int:
    """Kaggle 側で作られた次世代の状態を、手元の run へ反映する(full-cycle 用)。

    中身を検証してから置く。run.json は最後に不可分に差し替えるので、途中で落ちても
    「世代だけ進んで実体が無い」状態にはならない。
    """
    with zipfile.ZipFile(result_zip) as z:
        names = z.namelist()
        new_run = json.loads(z.read("run.json").decode("utf-8"))
        new_gen = new_run["generation"]
        if new_gen != gen + 1:
            raise SystemExit(f"戻ってきた run.json の世代がおかしい(v{new_gen} != v{gen+1})")
        model_name = f"models/model_v{new_gen}.json"
        if model_name not in names:
            raise SystemExit(f"結果に {model_name} が無い")
        json.loads(z.read(model_name).decode("utf-8"))     # 壊れていないか

        for n in names:
            if n == "run.json":
                continue                                    # 最後に置く
            dest = run_dir / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
        tmp = C.run_json_path(run_dir).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(new_run, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, C.run_json_path(run_dir))
    return new_gen
    print(f"\n完了。learner.py --run-dir {run_dir} --status で揃い具合を確認できる。")


if __name__ == "__main__":
    main()
