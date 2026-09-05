"""デッキ head-to-head 測定を Kaggle Notebook で回すためのデータセット(コード同梱 blob)を作る。

``build_dataset.py``(ドラパルト測定用)の派生。違いは同梱物だけ:

* 相手は「各アーキタイプの模倣エージェント」ではなく **同じエージェントが持つ別デッキ** なので、
  アーキタイプ別の重み11本もアーキタイプ別デッキも要らない。代わりに
  ``kaggle_replays/deck_search/candidates_wall/*.csv`` と mixogerpon 重み1本を入れる。
* ``opponents/`` は使わない(両陣営とも ml_policy)。

踏襲する運用(過去に動いた形):

* リポジトリのサブセットを **単一の tar.gz** に固めて1ファイルだけアップロードする。
  拡張子を ``.dat`` にするのは、Kaggle が ``.tar.gz`` を自動展開してディレクトリ構造を
  壊すのを避けるため。
* カーネル側は ``/kaggle/input/**/repo_deckh2h_blob.dat`` を glob で見つけて
  ``/kaggle/working/repo`` に展開し、そこから通常のスクリプトとして実行する。

使い方:
    python kaggle_replays/kaggle_run/build_dataset_deck.py --dry-run
    python kaggle_replays/kaggle_run/build_dataset_deck.py --out C:/tmp/ptcg_deck_h2h --verify-game
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATASET_ID = "shogonagashima/ptcg-deck-h2h"
DATASET_TITLE = "ptcg-deck-h2h"
BLOB_NAME = "repo_deckh2h_blob.dat"

# 同梱する重み。両陣営とも mixogerpon 1本だけ使う。policy_weights.json は
# config が policy_weights_path を持たない経路のフォールバックとして入れておく。
NEEDED_WEIGHTS = [
    "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json",
    "policy_weights.json",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git"}


def _iter_files(base: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
                continue
            out.append(path)
    return out


def collect(root: Path) -> list[tuple[Path, str]]:
    """``(実ファイル, tar 内のパス)`` のリストを組み立てる。"""
    items: list[tuple[Path, str]] = []

    def add(path: Path) -> None:
        if not path.exists():
            raise SystemExit(f"必要なファイルが見つかりません: {path}")
        items.append((path, path.relative_to(root).as_posix()))

    def add_many(paths: list[Path]) -> None:
        for p in paths:
            add(p)

    ss = root / "sample_submission"

    # cg エンジン(libcg.so が無いと Linux で動かない)
    add_many(_iter_files(ss / "cg", ["**/*.py", "**/*.so", "**/*.dll"]))
    if not (ss / "cg" / "libcg.so").exists():
        raise SystemExit("libcg.so が見つかりません(Linux カーネルで動かせません)")

    # ptcg_ai: .py は全部、.json は「重み以外」(opponent_modeling の予測器等)+ 必要な重みだけ
    add_many(_iter_files(ss / "ptcg_ai", ["**/*.py"]))
    for path in _iter_files(ss / "ptcg_ai", ["**/*.json"]):
        if path.name.startswith("policy_weights"):
            continue
        add(path)
    wdir = ss / "ptcg_ai" / "learning"
    for name in NEEDED_WEIGHTS:
        add(wdir / name)

    # config と既定デッキ(deck.csv は「宣言が無い経路」のフォールバックとして読まれる)
    add_many(_iter_files(ss / "configs", ["*.json"]))
    add(ss / "deck.csv")

    # decks/ パッケージ。ptcg_ai/shared/profile_registry.py が `from decks import active` を
    # トップレベル import するため、欠けると ml_policy が import できずに即死する
    # (提出 tarball で同じ事故があった。memory: 提出tarballにdecks/必須)。
    add_many(_iter_files(ss / "decks", ["**/*.py"]))

    # 対戦基盤(run_match._begin_match_state のデッキ注入が入った版であること!
    #  これが無いと deck.csv 以外を持つ側の探索が黙って死ぬ)
    add_many(_iter_files(root / "league", ["*.py"]))

    # 候補デッキ(1枚差し替えの60枚CSV)
    add_many(_iter_files(root / "kaggle_replays" / "deck_search" / "candidates_wall", ["*.csv"]))

    # 測定スクリプト本体
    add(root / "kaggle_replays" / "kaggle_run" / "deck_h2h.py")

    # 同じファイルを2回入れない
    seen: set[str] = set()
    unique: list[tuple[Path, str]] = []
    for path, arc in items:
        if arc in seen:
            continue
        seen.add(arc)
        unique.append((path, arc))
    return unique


def group_of(arcname: str) -> str:
    if arcname.startswith("sample_submission/ptcg_ai/learning/policy_weights"):
        return "sample_submission/ptcg_ai/learning/policy_weights_*"
    return arcname.rsplit("/", 1)[0] if "/" in arcname else "."


def report(items: list[tuple[Path, str]]) -> int:
    sizes: dict[str, list[int]] = {}
    total = 0
    for path, arc in items:
        size = path.stat().st_size
        total += size
        sizes.setdefault(group_of(arc), []).append(size)
    print(f"{'group':45s} {'files':>6s} {'MB':>8s}")
    for group, values in sorted(sizes.items(), key=lambda kv: -sum(kv[1])):
        print(f"{group:45s} {len(values):6d} {sum(values) / 1e6:8.2f}")
    print(f"{'TOTAL (uncompressed)':45s} {len(items):6d} {total / 1e6:8.2f}")
    return total


def git_provenance(root: Path, items: list[tuple[Path, str]]) -> dict:
    """どのコミット + どのローカル変更を固めたのかを記録する(blob は作業ツリーの中身)。"""
    import subprocess

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    dirty_all = {line[3:].strip().strip('"') for line in git("status", "--porcelain").splitlines()}
    packaged = {arc for _p, arc in items}
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "packaged_files_dirty_vs_head": sorted(packaged & dirty_all),
    }


def build(items: list[tuple[Path, str]], out_dir: Path, root: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = out_dir / BLOB_NAME
    with tarfile.open(blob, "w:gz") as tar:
        for path, arc in items:
            tar.add(path, arcname=arc)
    meta = {
        "title": DATASET_TITLE,
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (out_dir / "dataset-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "blob": BLOB_NAME,
        "blob_bytes": blob.stat().st_size,
        "blob_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **git_provenance(root, items),
        "files": [arc for _p, arc in items],
    }
    (out_dir / "_blob_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return blob


# blob を空ディレクトリに展開したときに必ず存在すべきファイル。
MUST_HAVE = [
    "sample_submission/cg/libcg.so",
    "sample_submission/cg/api.py",
    "sample_submission/decks/active.py",   # profile_registry が import する。欠けると即死
    "league/run_match.py",
    "league/run_league.py",
    "sample_submission/configs/abl_5_full_og_r13.json",
    "sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json",
    "kaggle_replays/deck_search/candidates_wall/aceburn.csv",
    "kaggle_replays/deck_search/candidates_wall/baseline.csv",
    "kaggle_replays/deck_search/candidates_wall/bulu.csv",
    "kaggle_replays/kaggle_run/deck_h2h.py",
]

IMPORT_CHECK = [
    "cg.api", "run_match", "run_league",
    "ptcg_ai.ml_policy.ml_policy_agent",
    "ptcg_ai.hidden_information.match_context",
    "ptcg_ai.search.pipeline",
]


def verify(blob: Path, run_game: bool = False) -> None:
    """blob を空ディレクトリに展開し、必須ファイル・import・(任意で)実対戦を確認する。"""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(blob) as tar:
            names = set(tar.getnames())
            tar.extractall(tmp)
        missing = [m for m in MUST_HAVE if m not in names]
        if missing:
            raise SystemExit(f"blob に必須ファイルがありません: {missing}")
        so = Path(tmp) / "sample_submission" / "cg" / "libcg.so"
        if so.stat().st_size < 100_000:
            raise SystemExit(f"libcg.so が壊れています: {so.stat().st_size} bytes")
        print(f"[verify] OK: {len(MUST_HAVE)} 必須ファイルを確認、libcg.so も健全")

        # run_match のデッキ注入(set_own_deck_override)が入っているか。無い版を固めると
        # deck.csv 以外を持つ側の探索が黙って無効化され、A/B が意味を失う。
        rm = (Path(tmp) / "league" / "run_match.py").read_text(encoding="utf-8")
        if "ml_policy_agent.set_own_deck_override" not in rm:
            raise SystemExit("[verify] league/run_match.py にデッキ注入が入っていません"
                             "(古い版を固めています。探索が死ぬので中止)")
        print("[verify] OK: run_match のデッキ注入(ml_policy_agent.set_own_deck_override)を確認")

        root = Path(tmp)
        code = (
            "import importlib, os, sys\n"
            f"root = r'{root}'\n"
            "sys.path[:0] = [root, root + '/league', root + '/sample_submission']\n"
            "os.chdir(root + '/sample_submission')\n"
            f"for m in {IMPORT_CHECK!r}:\n"
            "    importlib.import_module(m)\n"
            "print('IMPORT_OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if "IMPORT_OK" not in (proc.stdout or ""):
            raise SystemExit("[verify] import に失敗しました(同梱漏れの可能性):\n"
                             + (proc.stdout or "") + (proc.stderr or ""))
        print(f"[verify] OK: {len(IMPORT_CHECK)} モジュールを展開先から import 成功")

        if run_game:
            out = root / "_verify.jsonl"
            cmd = [sys.executable, str(root / "kaggle_replays" / "kaggle_run" / "deck_h2h.py"),
                   "--repo-root", str(root), "--arms", "aceburn", "--games", "2",
                   "--workers", "1", "--seed-base", "999000", "--progress-every", "1",
                   "--out", str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")
            if proc.returncode != 0 or not out.exists():
                raise SystemExit("[verify] 対戦スモークに失敗:\n"
                                 + (proc.stdout or "")[-3000:] + (proc.stderr or "")[-3000:])
            lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            games = [r for r in lines if "arm" in r]
            errors = [r for r in games if r.get("error")]
            if not games:
                raise SystemExit("[verify] 1試合も走っていません:\n" + (proc.stdout or "")[-3000:])
            if errors:
                raise SystemExit(f"[verify] エラー試合があります: {errors[:2]}")
            print(f"[verify] OK: 展開先だけで {len(games)} 試合を完走(エラー0)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "ptcg_deck_h2h"),
                    help="ステージングディレクトリ")
    ap.add_argument("--repo-root", default=str(_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="サイズ内訳を表示するだけ")
    ap.add_argument("--clean", action="store_true", help="出力先を作り直す")
    ap.add_argument("--verify-game", action="store_true",
                    help="展開先だけで実際に2試合走らせて検証する(1分ほどかかる)")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    items = collect(root)
    total = report(items)

    if args.dry_run:
        print("\n(dry-run: 何も書き出していません)")
        return

    out_dir = Path(args.out).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    blob = build(items, out_dir, root)
    verify(blob, run_game=args.verify_game)
    ratio = blob.stat().st_size / total if total else 0.0
    print(f"\nblob: {blob} ({blob.stat().st_size / 1e6:.2f} MB, 圧縮率 {ratio:.2f})")
    print(f"metadata: {out_dir / 'dataset-metadata.json'} (id={DATASET_ID}, private)")
    print("\n次の手順:")
    print(f"  python {_HERE / 'push_kernel_deck.py'} --push-dataset --dataset-dir {out_dir}")
    print("  # 手で叩く場合は **必ずそのディレクトリを cwd にして -p を付けない**"
          "(Kaggle CLI 2.2.3 は -p 絶対パスで落ちる)")


if __name__ == "__main__":
    main()
