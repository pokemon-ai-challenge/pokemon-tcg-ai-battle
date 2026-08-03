"""train_pool.py の PPO 学習を Kaggle Notebook 上で走らせる。

    python push_train_pool.py --train-args "--learner alakazam --train-opponents alakazam --iters 60 --tag k60" --dry-run

やること(--dry-run 無しの場合):
  1. 作業ツリーから明示リストでコードとデッキと重みを ptcg_bundle.zip にまとめる
     (git archive は使わない。train_pool.py / pools.py / 専門家の重みが未コミットのため)
  2. 非公開 Dataset として上げる(2回目以降は新バージョン)
  3. train_pool.py を実行する Notebook を生成して push する
  4. 完了まで待って、出力(学習済み重み・ログ)を回収する

--dry-run はバンドルと Notebook JSON とメタデータを作るだけで、Kaggle には一切送らない。

事前に一度だけ認証が必要:
    kaggle auth login
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE = kaggle_replays/rl/distributed/notebooks
REPO_ROOT = HERE.parent.parent.parent.parent
RL_DIR = REPO_ROOT / "kaggle_replays" / "rl"
LEARNING_DIR = REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning"

DEFAULT_DATASET_SLUG = "ptcg-train-pool"
DEFAULT_KERNEL_SLUG = "ptcg-train-pool-run"

# バンドルに含めるトップレベルのディレクトリ(REPO_ROOT からの相対パス)。
BUNDLE_ROOTS = [
    Path("sample_submission") / "cg",
    Path("sample_submission") / "ptcg_ai",
    Path("league"),
    Path("kaggle_replays") / "rl",
    Path("kaggle_replays") / "meta_analysis" / "archetype_decks",
]


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


# ---------------------------------------------------------------- バンドル
def available_weight_names() -> list[str]:
    return sorted(p.name for p in LEARNING_DIR.glob("policy_weights*.json"))


def resolve_weights_filter(spec: str | None) -> set[str]:
    all_names = available_weight_names()
    if not spec:
        return set(all_names)
    requested = {x.strip() for x in spec.split(",") if x.strip()}
    missing = requested - set(all_names)
    if missing:
        raise SystemExit(
            f"--weights に存在しない重みが指定された: {sorted(missing)}\n"
            f"  存在する重み: {all_names}")
    return requested


def _current_dataset_version(ref: str):
    """Dataset の現行バージョン番号。取得できなければ None(その場合は版チェックを諦める)。"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        owner = ref.split("/")[0]
        for d in api.dataset_list(mine=True, search=ref.split("/")[1]):
            if getattr(d, "ref", None) == ref:
                return getattr(d, "current_version_number", None)
        del owner
    except Exception as e:  # 認証形式の違いなどで落ちても push 自体は続けたい
        print(f"    (バージョン取得に失敗、版チェックは省略: {e})")
    return None


def _excluded(rel_parts: tuple[str, ...], name: str) -> bool:
    if "__pycache__" in rel_parts or ".pytest_cache" in rel_parts or ".git" in rel_parts:
        return True
    if name.endswith(".pyc"):
        return True
    if name.startswith("_tmp_policy_") and name.endswith(".json"):
        return True
    if name.startswith("_train_") and name.endswith(".log"):
        return True
    return False


def iter_bundle_files(weights_filter: set[str]):
    """(絶対パス, REPO_ROOT からの相対 arcname) を列挙する。"""
    seen: set[str] = set()
    for root_rel in BUNDLE_ROOTS:
        root = REPO_ROOT / root_rel
        if not root.exists():
            raise SystemExit(f"バンドル対象が存在しない: {root}")
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(REPO_ROOT)
            rel_parts = rel.parts
            if _excluded(rel_parts, p.name):
                continue
            # kaggle_replays/rl の直下だけ: distributed/ と runs/ を除外(distributed は
            # 分散自己対戦ワーカー専用で本バンドルには不要、runs は実行時生成物)。
            if root_rel == Path("kaggle_replays") / "rl":
                rel_in_rl = p.relative_to(root)
                if rel_in_rl.parts[0] in ("distributed", "runs"):
                    continue
            # 重みファイルは --weights で絞る。
            if p.parent == LEARNING_DIR and p.name.startswith("policy_weights") and p.suffix == ".json":
                if p.name not in weights_filter:
                    continue
            arc = rel.as_posix()
            if arc in seen:
                continue
            seen.add(arc)
            yield p, arc


def build_bundle_zip(dest: Path, weights_filter: set[str]) -> tuple[int, int, int]:
    """ptcg_bundle.zip を作る。戻り値: (ファイル数, 展開後合計バイト数, zip サイズ)。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    n = 0
    total = 0
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in iter_bundle_files(weights_filter):
            z.write(p, arc)
            n += 1
            total += p.stat().st_size
    return n, total, dest.stat().st_size


# ---------------------------------------------------------------- Notebook 生成
def _code_cell(lines: list[str]) -> dict:
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def _md_cell(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def build_notebook(dataset_slug: str, final_args: list[str], script: str = "train_pool.py") -> dict:
    train_args_repr = json.dumps(final_args)  # Python リテラルとしてそのままセルに埋め込む
    script_repr = json.dumps(script)

    cell_intro = _md_cell(
        "# train_pool.py — Kaggle 実行\n\n"
        "PPO で相手プールに対して学習する。**非公開 Notebook・非公開 Dataset。GPU 不要・"
        "インターネット不要。**\n\n"
        f"入力: Dataset `{dataset_slug}`(`ptcg_bundle.zip`)\n"
        "出力: `/kaggle/working/` 直下の学習済み重み JSON と `_train_pool_*.log`\n\n"
        "Save & Run All にすれば、ブラウザを閉じても裏で最後まで走る。"
    )

    cell_extract = _code_cell([
        "import glob, os, shutil, sys, zipfile",
        "",
        "print('--- /kaggle/input ---')",
        "for p in sorted(glob.glob('/kaggle/input/**', recursive=True))[:60]:",
        "    print(' ', p)",
        "",
        "REPO = '/kaggle/working/repo'",
        "os.makedirs(REPO, exist_ok=True)",
        "",
        "# Kaggle は Dataset にアップロードした zip を**既定で展開する**ので、",
        "# /kaggle/input には zip ではなくディレクトリ木が置かれることがある。",
        "# どちらの形でも動くようにする(実際に zip 前提で書いて失敗した)。",
        "zips = glob.glob('/kaggle/input/**/ptcg_bundle.zip', recursive=True)",
        "if zips:",
        "    print('zip から展開:', zips[0])",
        "    zipfile.ZipFile(zips[0]).extractall(REPO)",
        "else:",
        "    marks = glob.glob('/kaggle/input/**/sample_submission/cg/libcg.so', recursive=True)",
        "    if not marks:",
        "        raise SystemExit('ptcg_bundle.zip も展開済みツリーも /kaggle/input 以下に無い。'",
        "                         'Dataset が Notebook に添付され、処理が完了しているか確認する。')",
        "    src = os.path.dirname(os.path.dirname(os.path.dirname(marks[0])))",
        "    print('展開済みツリーから複製:', src)",
        "    # /kaggle/input は読み取り専用。train_pool.py は重みを書き出すので writable な場所に複製する。",
        "    shutil.copytree(src, REPO, dirs_exist_ok=True)",
        "    os.chmod(os.path.join(REPO, 'sample_submission', 'cg', 'libcg.so'), 0o755)",
        "",
        "for p in (REPO, os.path.join(REPO, 'sample_submission'), os.path.join(REPO, 'league'),",
        "          os.path.join(REPO, 'kaggle_replays', 'rl')):",
        "    if p not in sys.path:",
        "        sys.path.insert(0, p)",
        "",
        f"SCRIPT = {script_repr}",
        "need = ['sample_submission/cg/libcg.so',",
        "        'sample_submission/ptcg_ai/learning/policy_model.py',",
        "        'league/run_league.py',",
        "        'kaggle_replays/rl/' + SCRIPT,",
        "        'kaggle_replays/meta_analysis/archetype_decks/alakazam/01.csv']",
        "missing = [n for n in need if not os.path.exists(os.path.join(REPO, n))]",
        "print('\\n--- 必要ファイルの確認 ---')",
        "for n in need:",
        "    print(('  OK  ' if n not in missing else '  なし '), n)",
        "if missing:",
        "    raise SystemExit(f'Dataset の中身が足りない(古い版が添付された可能性): {missing}')",
    ])

    cell_env = _code_cell([
        "import ctypes, os, platform, sys",
        "import torch",
        "",
        "print('python       :', sys.version)",
        "print('platform     :', platform.platform())",
        "print('cpu_count    :', os.cpu_count())",
        "print('torch        :', torch.__version__)",
        "print('torch.cuda   :', torch.cuda.is_available())",
        "",
        "so_path = os.path.join(REPO, 'sample_submission', 'cg', 'libcg.so')",
        "try:",
        "    ctypes.CDLL(so_path)",
        "    print('libcg.so     : OK (', so_path, ')')",
        "except OSError as e:",
        "    raise SystemExit(f'libcg.so を読み込めない: {e}')",
    ])

    cell_run = _code_cell([
        "import os, subprocess, sys, time",
        "",
        f"SCRIPT = {script_repr}",
        f"TRAIN_ARGS = {train_args_repr}",
        "",
        "cwd = os.path.join(REPO, 'kaggle_replays', 'rl')",
        "cmd = [sys.executable, '-u', SCRIPT] + TRAIN_ARGS",
        "print('$', ' '.join(cmd), flush=True)",
        "",
        "env = dict(os.environ)",
        "env['PYTHONUNBUFFERED'] = '1'",
        "t0 = time.time()",
        "proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,",
        "                        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)",
        "for line in proc.stdout:",
        "    print(line, end='', flush=True)",
        "ret = proc.wait()",
        "print(f'\\n[train_pool.py exit={ret}] {time.time()-t0:.0f}s elapsed', flush=True)",
        "if ret != 0:",
        "    raise SystemExit(f'{SCRIPT} が失敗(exit {ret})')",
    ])

    cell_collect = _code_cell([
        "import glob, os, shutil",
        "",
        "learning_dir = os.path.join(REPO, 'sample_submission', 'ptcg_ai', 'learning')",
        "rl_dir = os.path.join(REPO, 'kaggle_replays', 'rl')",
        "",
        "out_files = (glob.glob(os.path.join(learning_dir, 'policy_weights_*_pool_*.json')) +",
        "             glob.glob(os.path.join(rl_dir, '_train_pool_*.log')) +",
        "             # train_league.py(相互鍛錬ループ)は league_runs/<tag>/ 以下に世代ごとの",
        "             # 重み・state.json・history.json を書く。ここを拾わないと結果が回収できない。",
        "             glob.glob(os.path.join(rl_dir, 'league_runs', '**', '*.json'), recursive=True))",
        "if not out_files:",
        "    print('回収対象なし(出力ファイルが見つからない)')",
        "for p in out_files:",
        "    if os.path.getmtime(p) < t0:",
        "        continue  # 今回の実行で作られたものだけ回収する",
        "    # league_runs/ 以下は gen0/ gen1/ ... に同名ファイル(policy_weights_<arch>.json)が",
        "    # 並ぶので、basename だけにすると世代同士が衝突して上書きされる。相対パスを平坦化して保つ。",
        "    if 'league_runs' in p.replace(os.sep, '/').split('/'):",
        "        dest = os.path.join('/kaggle/working',",
        "                            os.path.relpath(p, rl_dir).replace(os.sep, '_').replace('/', '_'))",
        "    else:",
        "        dest = os.path.join('/kaggle/working', os.path.basename(p))",
        "    shutil.copyfile(p, dest)",
        "    print('回収:', dest, f'({os.path.getsize(dest)/1e6:.2f} MB)')",
        "",
        "print('\\n--- /kaggle/working ---')",
        "for p in sorted(os.listdir('/kaggle/working')):",
        "    print(' ', p)",
    ])

    return {
        "cells": [cell_intro, cell_extract, cell_env, cell_run, cell_collect],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook_json(doc: dict, dest: Path) -> None:
    # ensure_ascii=True で書く。Kaggle CLI は push するファイルをシステム既定の文字コード
    # (Windows では cp932)で読むため、日本語がそのまま入っていると復号に失敗する。
    # JSON のエスケープにしておけばファイルは純 ASCII になり、表示内容は変わらない。
    dest.write_text(json.dumps(doc, ensure_ascii=True, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- CLI
def resolve_final_args(train_args: str, workers: int) -> list[str]:
    parts = shlex.split(train_args)
    if any(p == "--workers" or p.startswith("--workers=") for p in parts):
        print(f"[WARN] --train-args に --workers が含まれているため、そちらを優先する"
              f"(--workers {workers} は無視)")
        return parts
    return parts + ["--workers", str(workers)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-args", required=True,
                    help="train_pool.py にそのまま渡す引数(1つの文字列)。"
                         "例: \"--learner alakazam --train-opponents alakazam --iters 60 --tag k60\"")
    ap.add_argument("--workers", type=int, default=4, help="Kaggle 側のワーカー数(既定4)")
    ap.add_argument("--dataset-slug", default=DEFAULT_DATASET_SLUG)
    ap.add_argument("--kernel-slug", default=DEFAULT_KERNEL_SLUG)
    ap.add_argument("--username", default=None)
    ap.add_argument("--script", default="train_pool.py",
                     help="kaggle_replays/rl/ 配下の実行スクリプト名。既定 train_pool.py。"
                          "相互鍛錬ループを回すなら train_league.py を指定する。")
    ap.add_argument("--weights", default=None,
                    help="カンマ区切りの重みファイル basename。省略時は "
                         "sample_submission/ptcg_ai/learning/policy_weights*.json 全部。")
    ap.add_argument("--dry-run", action="store_true",
                    help="バンドル・Notebook JSON・メタデータを作るだけで Kaggle へは送らない")
    ap.add_argument("--out-dir", default=str(HERE / ".stage_train_pool"))
    ap.add_argument("--dataset-wait", type=int, default=90,
                    help="Dataset の新バージョンが反映されるまでの待ち時間(秒)")
    ap.add_argument("--timeout", type=int, default=3 * 3600, help="完了待ちの上限(秒)")
    ap.add_argument("--no-wait", action="store_true", help="push だけして待たない")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    data_dir = out_dir / "data"
    kernel_dir = out_dir / "kernel"

    weights_filter = resolve_weights_filter(args.weights)
    final_args = resolve_final_args(args.train_args, args.workers)

    print(f"train_pool.py 引数(最終形): {final_args}")
    print(f"重み({len(weights_filter)}/{len(available_weight_names())}): {sorted(weights_filter)}")

    zip_path = data_dir / "ptcg_bundle.zip"
    n, total, zsize = build_bundle_zip(zip_path, weights_filter)
    print(f"バンドル: {n} ファイル, 展開後 {total/1e6:.1f} MB -> zip {zsize/1e6:.1f} MB ({zip_path})")

    user = detect_username(args.username)

    dataset_meta = {
        "title": "ptcg train pool",
        "id": f"{user}/{args.dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (data_dir / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2), encoding="utf-8")

    kernel_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = f"{args.kernel_slug}.ipynb"
    notebook_doc = build_notebook(args.dataset_slug, final_args, args.script)
    write_notebook_json(notebook_doc, kernel_dir / notebook_name)

    kernel_meta = {
        "id": f"{user}/{args.kernel_slug}",
        # title は固定にしないこと。Kaggle は Notebook を新規作成するとき id ではなく
        # **title を slug 化**して ref を決める。title を固定にすると --kernel-slug を
        # 変えても同じ Notebook を上書きしてしまう。実際に踏んだ:
        # --kernel-slug ptcg-train-alakazam-k60 で送ったのに ref は
        # koshin953/ptcg-train-pool-run になった(既定 title "ptcg train pool run" 由来)。
        "title": args.kernel_slug.replace("-", " "),
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{user}/{args.dataset_slug}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")

    print(f"dataset-metadata: {data_dir / 'dataset-metadata.json'} (isPrivate={dataset_meta['isPrivate']})")
    print(f"kernel-metadata : {kernel_dir / 'kernel-metadata.json'} (is_private={kernel_meta['is_private']})")
    print(f"notebook        : {kernel_dir / notebook_name}")

    if args.dry_run:
        print("\n--dry-run のため Kaggle へは何も送らない。")
        return

    # ------------------------------------------------------------ ここから先は実送信
    print(f"\nrun={user}/{args.kernel_slug} へ送信")

    # アップロード前のバージョン番号を控える。status の "ready" は「何らかの版が
    # 使える」を意味するだけで、今上げた版が反映されたことは保証しない。実際に踏んだ:
    # ready を見て push したが Notebook には旧版が添付され、新規追加した
    # train_league.py が無くてガードに弾かれた。番号が上がったことで確認する。
    prev_version = _current_dataset_version(f"{user}/{args.dataset_slug}")
    print(f"  更新前の Dataset バージョン: {prev_version}")

    exists = kaggle("datasets", "status", f"{user}/{args.dataset_slug}", check=False).returncode == 0
    if exists:
        print("  Dataset を更新(新バージョン)")
        kaggle("datasets", "version", "-p", str(data_dir), "-m", "update")
    else:
        print("  Dataset を新規作成(非公開)")
        kaggle("datasets", "create", "-p", str(data_dir))
    # 固定秒数の sleep で済ませてはいけない。Dataset の処理が終わる前に Notebook を push すると
    # /kaggle/input が空のまま実行され、原因の分かりにくい失敗になる。実際に踏んだ:
    # 90s 待って push したが /kaggle/input/ は空で、cell1 が「zip が見つからない」で落ちた。
    # 処理完了を実際に確認してから進む。
    print(f"  Dataset の処理完了を待つ(最大 {args.dataset_wait}s)", flush=True)
    t_ds = time.time()
    ready = False
    while time.time() - t_ds < args.dataset_wait:
        r = kaggle("datasets", "status", f"{user}/{args.dataset_slug}", check=False)
        st = (r.stdout or "").strip().lower()
        if "ready" in st:
            cur = _current_dataset_version(f"{user}/{args.dataset_slug}")
            if prev_version is not None and cur is not None and cur <= prev_version:
                # ready だが版が上がっていない = まだ旧版。ここで push すると旧版を掴む。
                print(f"    ready だが版は {cur} のまま({prev_version} から未更新)。待機継続",
                      flush=True)
                time.sleep(10)
                continue
            ready = True
            print(f"    ready / バージョン {prev_version} -> {cur} ({time.time() - t_ds:.0f}s)")
            break
        if "error" in st:
            raise SystemExit(f"Dataset の処理が失敗した: {r.stdout}")
        time.sleep(10)
    if not ready:
        raise SystemExit(
            f"Dataset が {args.dataset_wait}s 以内に ready にならなかった。"
            f"--dataset-wait を伸ばすか、https://www.kaggle.com/datasets/{user}/{args.dataset_slug} を確認する。")

    print("  Notebook を push して実行開始")
    kaggle("kernels", "push", "-p", str(kernel_dir))

    if args.no_wait:
        print(f"\nhttps://www.kaggle.com/code/{user}/{args.kernel_slug} で進行を確認できる。")
        return

    t0 = time.time()
    last = ""
    while time.time() - t0 < args.timeout:
        r = kaggle("kernels", "status", f"{user}/{args.kernel_slug}", check=False)
        out = (r.stdout or "").strip()
        if out != last:
            print(f"  [{time.time()-t0:5.0f}s] {out}", flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            break
        if "error" in low or "cancel" in low:
            raise SystemExit(f"Kaggle 側で失敗: {out}\n"
                             f"  https://www.kaggle.com/code/{user}/{args.kernel_slug} でログを確認する。")
        time.sleep(20)
    else:
        raise SystemExit("待ち時間の上限に達した。--no-wait で投げっぱなしにして後で回収する。")

    dest = out_dir / "out"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{user}/{args.kernel_slug}", "-p", str(dest))
    print(f"\n完了。回収先: {dest}")


if __name__ == "__main__":
    main()
