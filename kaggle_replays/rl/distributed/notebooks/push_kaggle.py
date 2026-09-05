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

DEFAULT_DATASET_SLUG = "ptcg-distributed-selfplay"
DEFAULT_KERNEL_SLUG = "ptcg-worker-kaggle"

# distributed/ 配下は丸ごと作業ツリーから追加する(既存挙動)ので、この配下は
# _classify_status_lines の ship/skip 判定に回さず別扱いにする。
DIST_PREFIX = "kaggle_replays/rl/distributed/"
# git archive でも既に読み取り専用として扱っている(絶対に変更しない)ディレクトリ。
# 内容が変わることはない前提だが、万一 git 上で動きがあっても overlay 対象には含めない。
_NEVER_SHIP_PREFIXES = ("data/", "sample_submission/cg/")
# 作業ツリーを上書きコピーする対象の拡張子。バイナリ・巨大データ(リプレイ dump, .npz
# 特徴量, 想定外形式のチェックポイント等)を誤って混入させないため .py/.json に限定する。
_SHIPPABLE_EXTS = {".py", ".json"}
# 単一ファイルの上限。policy_weights_*.json は通常 1MB 未満。巨大ファイルが git status に
# 紛れて overlay されるのを防ぐガード。表示メッセージと単位を揃えるため 10進 MB で定義する。
_MAX_SHIP_BYTES = 20_000_000


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


def _parse_status_line(line: str) -> str:
    """`git status --porcelain` の1行からパスだけを取り出す(先頭2文字のステータスコードを
    除く)。rename (`OLD -> NEW`) は現在のパス(NEW)を返す。引用符付き(非ASCII文字を含む
    パス)は簡易的に外側の `"` だけ剥がす。"""
    return line[3:].replace("\\", "/").split(" -> ")[-1].strip('"')


def _dirty_status_lines(repo_root: Path) -> list[str]:
    """sample_submission/kaggle_replays/league 配下の未コミット変更を1行1ファイルで返す。
    `--untracked-files=all` を付け、未追跡ディレクトリもまとめず個々のファイルとして展開する
    (そうしないと拡張子・サイズでの判定がディレクトリ単位になってしまう)。"""
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                        "sample_submission", "kaggle_replays", "league"],
                       cwd=repo_root, capture_output=True, text=True, check=True)
    return [line for line in r.stdout.splitlines() if line.strip()]


def _classify_dirty_file(repo_root: Path, path: str) -> tuple[str, str]:
    """1ファイルを ship(作業ツリー版を上書きして送る) / skip(HEAD版のまま送られる) に分類する。
    戻り値は (decision, reason)。decision=="ship" のとき reason は空文字。"""
    if path.startswith(_NEVER_SHIP_PREFIXES):
        return "skip", "読み取り専用ディレクトリ(data/ または sample_submission/cg/)"
    full = repo_root / path
    if not full.is_file():
        # 削除済み、またはシンボリックリンク等。overlay できる作業ツリー版が無い。
        return "skip", "作業ツリーに実ファイルが無い(削除された、またはファイルではない)"
    ext = full.suffix.lower()
    if ext not in _SHIPPABLE_EXTS:
        return "skip", f"拡張子 '{ext or '(なし)'}' は対象外(.py/.json のみ overlay する)"
    size = full.stat().st_size
    if size > _MAX_SHIP_BYTES:
        return "skip", f"サイズ超過({size/1e6:.1f}MB > {_MAX_SHIP_BYTES/1e6:.0f}MB 上限)"
    return "ship", ""


def _classify_status_lines(repo_root: Path, lines: list[str]) -> dict[str, list]:
    """dirty な行を3種に分ける。
      - "dist":  kaggle_replays/rl/distributed/ 配下。build_payload が別途丸ごと作業ツリー
                 から追加するので、ここでは ship/skip どちらにも数えない。
      - "ship":  overlay で作業ツリー版を書き込むので、Kaggle には正しい版が届く。
      - "skip":  [(path, reason), ...]。overlay されないので、git archive HEAD の
                 (古い可能性がある)版がそのまま届く。
    """
    out: dict[str, list] = {"dist": [], "ship": [], "skip": []}
    for line in lines:
        path = _parse_status_line(line)
        if not path:
            continue
        if path.startswith(DIST_PREFIX):
            out["dist"].append(path)
            continue
        decision, reason = _classify_dirty_file(repo_root, path)
        if decision == "ship":
            out["ship"].append(path)
        else:
            out["skip"].append((path, reason))
    return out


def _check_dirty_shipped_files(allow_dirty: bool) -> dict[str, list]:
    """`git archive HEAD` は**コミット済みの中身**しか含めない。distributed/ 配下は
    build_payload が作業ツリーから明示的に追加する。それ以外の未コミット変更のうち
    .py/.json かつ 20MB 以下のものは build_payload が同様に作業ツリー版で上書きするので
    問題ないが(cls["ship"])、それ以外(バイナリ・巨大ファイル・削除など)は
    overlay できず HEAD コミット版が黙って Kaggle に送られる(cls["skip"])。

    2026-08-13 に実際に踏んだ事故: sample_submission/ptcg_ai/learning/policy_weights.json を
    166→715次元へ更新したが未コミットのままだった。ローカルでは作業ツリー版
    (715次元、is_ready=True)で検証していたが、push_kaggle.py が送ったのは HEAD コミット版
    (166次元)。Kaggle 側で PolicyModel.is_ready=False になり、initializer が例外を出し続けて
    multiprocessing.Pool が子プロセスを無限に再spawnし、1800秒タイムアウトまで
    「ハングしているように見える」形で失敗した(実際はエラーが即座に起きていたが、
    Pool がそれを親プロセスへ伝えなかった)。この関数と build_payload の overlay 拡張により、
    今なら policy_weights.json 自体は ship 側に分類されて正しく送られるが、他の
    .py/.json 以外の未コミット変更については引き続き検出・警告する。

    cls["skip"] が空でなければデフォルトで中断する(--allow-dirty で警告のみにできる)。
    呼び出し側(build_payload)が同じ分類結果を overlay にも使うので、git status は
    ここで1回だけ実行する。
    """
    lines = _dirty_status_lines(C.REPO_ROOT)
    cls = _classify_status_lines(C.REPO_ROOT, lines)
    if cls["skip"]:
        msg = ("未コミットの変更のうち、作業ツリー版を Kaggle へ送れないものがある"
               "(overlay 対象外なので git archive HEAD の版が送られる):\n" +
               "\n".join(f"  {p}  -- {reason}" for p, reason in cls["skip"]) +
               "\n\nコミットしてから push_kaggle.py を実行するか、意図的なら --allow-dirty で無視する。")
        if cls["ship"]:
            msg += ("\n\n(以下は作業ツリー版が overlay で正しく上書きされるので問題ない: " +
                    ", ".join(cls["ship"]) + ")")
        if allow_dirty:
            print("[push_kaggle] 警告(--allow-dirty で続行): " + msg.replace("\n", "\n  "))
        else:
            raise SystemExit("[push_kaggle] " + msg)
    elif cls["ship"]:
        print("[push_kaggle] 未コミットの変更があるが、作業ツリー版が overlay で送られるので"
              "問題ない: " + ", ".join(cls["ship"]))
    return cls


def build_payload(run_dir: Path, gen: int, stage: Path, allow_dirty: bool = False) -> None:
    cls = _check_dirty_shipped_files(allow_dirty)
    stage.mkdir(parents=True, exist_ok=True)
    repo_zip = stage / "ptcg_repo.zip"
    archive_zip = stage / "_ptcg_repo_archive.zip"  # git archive の生成物。作り直した後に消す一時ファイル
    # コミット済みのファイル一式。
    subprocess.run(["git", "archive", "--format=zip", "HEAD", "-o", str(archive_zip)],
                   cwd=C.REPO_ROOT, check=True)

    # distributed/ 配下は常に作業ツリーから明示的に足す。新規(未コミット)ファイルは
    # そもそも git archive に入らないので必須。追跡済みで中身だけ変更されているファイル
    # (push_kaggle.py 自身を含む)や、distributed/ 以外で未コミットだが .py/.json かつ
    # サイズ上限以下のファイル(cls["ship"])も同様に作業ツリー版で上書きする必要がある。
    #
    # 2026-08-13 に判明: 旧実装は git archive の zip に対して `zipfile.ZipFile(..., "a")` で
    # 同じ arcname を追記していた(zip は同名エントリを複数持てる)。ローカルの CPython
    # zipfile は read/open/extractall のいずれも「最後に書かれたエントリ」を返すため
    # ローカル検証では正しく上書きされて見えたが、実際に Kaggle へ push したところ
    # policy_weights.json が古い(git archive 側の) HEAD 版のまま送られる事故が起きた
    # (sha256 が working tree 版 52bb5330... ではなく HEAD 版 735dd38a... のまま Kaggle 側の
    # ログに現れた)。Kaggle の Dataset 取り込み/展開パイプラインは CPython の zipfile と
    # 同じ「後勝ち」を保証しないらしい(zip 仕様上そもそも重複 arcname の扱いは未規定で、
    # ツール依存)。そのため同名エントリを複数持つ zip を作ること自体をやめ、
    # git archive の中身から overlay 対象の arcname を除いた上で1回だけ書き込む
    # (物理的に同名エントリが1つしか存在しない zip にする)。
    dist = C.REPO_ROOT / "kaggle_replays" / "rl" / "distributed"
    overlay_paths: dict[str, Path] = {}
    for p in dist.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or ".kaggle_stage" in p.parts:
            continue
        overlay_paths[p.relative_to(C.REPO_ROOT).as_posix()] = p
    for rel in cls["ship"]:
        overlay_paths[rel] = C.REPO_ROOT / rel

    kept = 0
    replaced = 0
    added = 0
    with zipfile.ZipFile(archive_zip, "r") as zin, \
         zipfile.ZipFile(repo_zip, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename in overlay_paths:
                replaced += 1   # git archive 側の(HEAD の)版は書かず、下で作業ツリー版を書く
                continue
            zout.writestr(info, zin.read(info.filename))
            kept += 1
        archive_names = set(zin.namelist())
        for arc, p in overlay_paths.items():
            zout.write(p, arc)
            if arc not in archive_names:
                added += 1
    archive_zip.unlink()

    if cls["ship"]:
        print(f"  未コミットの変更を作業ツリー版で上書き: {len(cls['ship'])} ファイル "
              f"({', '.join(cls['ship'])})")
    if cls["skip"]:
        print(f"  [警告] 以下 {len(cls['skip'])} 件は未コミットだが overlay 対象外 "
              "(HEAD コミット版のまま送られる):")
        for p, reason in cls["skip"]:
            print(f"    {p} -- {reason}")
    print(f"  distributed/ 等を作業ツリー版で反映: 新規 {added} / 上書き {replaced} 件"
          f"(git archive からそのまま維持 {kept} 件)")

    # 重複 arcname が無いことをその場で確認する(zip 仕様上ツール依存の挙動になりうるので、
    # 送る前に必ず検査する)。
    names = zipfile.ZipFile(repo_zip).namelist()
    assert len(names) == len(set(names)), "ptcg_repo.zip に重複 arcname が残っている(バグ)"
    run_id = C.load_run(run_dir)["run_id"]
    for old in stage.glob("run_*.zip"):
        old.unlink()
    with zipfile.ZipFile(stage / f"run_{run_id}_v{gen}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(C.run_json_path(run_dir), "run/run.json")
        z.write(C.model_path(run_dir, gen), f"run/models/model_v{gen}.json")
        # 対戦相手の重み(init_run.py --opponent-weights がローカルの実ファイルを指していた
        # 場合に run_dir/opponents/ へコピーされる)。無ければ何もしない(後方互換)。
        opp_dir = run_dir / "opponents"
        if opp_dir.is_dir():
            for p in sorted(opp_dir.rglob("*")):
                if p.is_file():
                    z.write(p, f"run/opponents/{p.relative_to(opp_dir).as_posix()}")
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
    ap.add_argument("--allow-dirty", action="store_true",
                    help="sample_submission/kaggle_replays/league 配下に未コミットの変更が"
                         "あっても続行する(既定は中断。git archive は HEAD しか含まないため、"
                         "作業ツリーとの食い違いに気づかず古い版を送ってしまう事故を防ぐ)")
    ap.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG,
                    help="Kaggle Dataset のslug(既定 %(default)s)。push_train_pool.py と同じ"
                         "流儀: 複数の run を Kaggle 上で並行して走らせたいときは、run ごとに"
                         "別の --dataset-slug/--kernel-slug を明示的に渡す(既定のままだと"
                         "全 run が同じ Dataset/Kernel を取り合って競合する)。")
    ap.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG,
                    help="Kaggle Kernel(Notebook) のslug(既定 %(default)s)。--dataset-slug 参照。")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run = C.load_run(run_dir)
    gen = run["generation"]
    if "kaggle" not in run["workers"]:
        raise SystemExit(f"run.json の workers に 'kaggle' が無い: {run['workers']}")
    user = detect_username(args.username)
    dataset_slug = args.dataset_slug
    kernel_slug = args.kernel_slug
    stage = run_dir / ".kaggle_stage"
    # Dataset に上げるのは data/ の中身だけ。kernel/ や out/ を同じ階層に置くと
    # それらまで Dataset に含まれてしまうので、ディレクトリを分ける。
    ddir = stage / "data"

    print(f"run={run['run_id']} v{gen} -> Kaggle({user})")
    build_payload(run_dir, gen, ddir, allow_dirty=args.allow_dirty)

    # --- Dataset(無ければ作成、あればバージョン追加) ---
    meta = {"title": dataset_slug.replace("-", " "),
            "id": f"{user}/{dataset_slug}",
            "licenses": [{"name": "unknown"}]}
    (ddir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    exists = kaggle("datasets", "status", f"{user}/{dataset_slug}", check=False).returncode == 0
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
        "id": f"{user}/{kernel_slug}",
        "title": kernel_slug.replace("-", " "),
        "code_file": "kaggle_worker.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{user}/{dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(kmeta, indent=2), encoding="utf-8")
    print("  Notebook を push して実行開始")
    kaggle("kernels", "push", "-p", str(kdir))

    if args.no_wait:
        print(f"\nhttps://www.kaggle.com/code/{user}/{kernel_slug} で進行を確認できる。")
        return

    # --- 完了待ち ---
    t0 = time.time()
    last = ""
    while time.time() - t0 < args.timeout:
        r = kaggle("kernels", "status", f"{user}/{kernel_slug}", check=False)
        out = (r.stdout or "").strip()
        if out != last:
            print(f"  [{time.time()-t0:5.0f}s] {out}", flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise SystemExit(f"Kaggle 側で失敗: {out}\n"
                             f"  https://www.kaggle.com/code/{user}/{kernel_slug} でログを確認する。")
        time.sleep(20)
    else:
        raise SystemExit("待ち時間の上限に達した。--no-wait で投げっぱなしにして後で回収する。")

    # --- 出力を回収 ---
    # 2026-08-14 に実際に踏んだ事故: `.kaggle_stage/out/` を前回の回収結果のまま残しておくと、
    # `kaggle kernels output` がこのバージョンの kaggle.npz を新規ダウンロードしなかった場合
    # (セッション切断とは無関係に、割り込みゼロの正常終了でも再現した)でも
    # glob("*.npz") は前回分(古い世代のシャード)をそのまま拾ってしまい、
    # 「今回のgenに見えるが中身は前回のもの」が黙って shards/v<gen>/ にコピーされていた
    # (learner.py の世代ハッシュ照合が弾くので学習に混ざる実害は無かったが、
    # 気づかず繰り返すと無限にrejectされ続けるだけで学習が進まない)。
    # 対策: ダウンロード前に out/ を毎回空にし、ダウンロード直後に世代・モデルハッシュ等を
    # ここで検査してから shards/v<gen>/ へコピーする(learner.py と同じ common.check_shard を
    # 再利用し、二重に実装しない)。
    out_dir = stage / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    dest = C.shard_dir(run_dir, gen)
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{kernel_slug}", "-p", str(out_dir))
    got = list(out_dir.glob("*.npz"))
    if not got:
        raise SystemExit(f"出力に .npz が無い: {out_dir}")
    model_sha = C.sha256_file(C.model_path(run_dir, gen))
    for p in got:
        _, meta = C.read_shard(p)
        try:
            C.check_shard(meta, run, gen, model_sha)
        except C.ShardRejected as exc:
            raise SystemExit(
                f"[push_kaggle] ダウンロードした {p.name} が今回の世代のものではない: {exc}\n"
                "  Kaggle側の出力がまだ古いバージョンのままの可能性がある。"
                "再度 push_kaggle.py を実行して撮り直すこと(このrunは進んでいない)。"
            ) from exc
        shutil.copyfile(p, dest / p.name)
        print(f"  回収: {dest / p.name} ({p.stat().st_size/1e6:.1f} MB)")
    print(f"\n完了。learner.py --run-dir {run_dir} --status で揃い具合を確認できる。")


if __name__ == "__main__":
    main()
