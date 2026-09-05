"""ドラパルト測定を Kaggle Notebook で回すためのデータセット(コード同梱 blob)を作る。

過去に動いた運用(``shogonagashima/ptcg-rl-gpu`` + ``shogonagashima/ptcg-rl-*`` カーネル)を
踏襲する:

* リポジトリのサブセットを **単一の tar.gz** に固めて 1 ファイルだけアップロードする。
  拡張子を ``.dat`` にするのは、Kaggle がアップロード時に ``.tar.gz`` を自動展開して
  ディレクトリ構造を壊すのを避けるため(過去の ``repo_gpu_blob.dat`` と同じ理由)。
* カーネル側は ``/kaggle/input/**/repo_drapa_blob.dat`` を glob で見つけて
  ``/kaggle/working/repo`` に展開し、そこから通常のスクリプトとして実行する。

同梱するのは測定に必要な最小限だけ:

* ``sample_submission/cg/``(**libcg.so 必須**。Linux では ctypes が自動で選ぶ)
* ``sample_submission/ptcg_ai/``(``.py`` 全部 + 測定に要る重み JSON だけ。
  ``learning/policy_weights_*.json`` は 89 個 33MB あるのでフィールド11アーキ分に絞る)
* ``sample_submission/configs/`` / ``sample_submission/deck.csv``
* ``league/`` / ``opponents/``(ドラパルト2種 + デッキCSV)
* ``kaggle_replays/meta_analysis/archetype_decks{,_g2}/<arch>/01.csv``(必要アーキのみ)
* ``kaggle_replays/_drapa_ceiling.py``(測定ロジックの流用元。参照用)
* ``kaggle_replays/kaggle_run/drapa_ablation.py``(本体)

使い方:
    python kaggle_replays/kaggle_run/build_dataset.py --dry-run          # サイズだけ確認
    python kaggle_replays/kaggle_run/build_dataset.py --out C:/tmp/ds    # 実際に作る
    kaggle datasets create -p C:/tmp/ds --dir-mode skip                  # 初回
    kaggle datasets version -p C:/tmp/ds -m "update"                     # 2回目以降
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

DATASET_ID = "shogonagashima/ptcg-dragapult-eval"
DATASET_TITLE = "ptcg-dragapult-eval"
BLOB_NAME = "repo_drapa_blob.dat"

# drapa_ablation.FIELD と同じ11アーキ(重み・デッキの絞り込みに使う)。
# import 依存を作らずここに写しているので、FIELD を変えたらこちらも直すこと。
FIELD: list[tuple[str, str]] = [
    ("marnie_grimmsnarl_ex", "_g2"), ("alakazam", "_g2"),
    ("mega_lucario_ex", ""), ("dragapult_ex", "_g2"),
    ("mega_froslass_ex", "_g2"), ("ogerpon_teal_ex", "_g2"),
    ("archaludon_ex", "_g2"), ("crustle", "_g2"),
    ("shirona_garchomp_ex", "_g2"), ("omatsuri_ondo", "_g2"),
    ("rocket_mewtwo_ex", "_g2"),
]
_DECKDIR_OF = {"": "archetype_decks", "_g2": "archetype_decks_g2"}

# フィールド以外に同梱する重み。policy_weights.json は config が
# policy_weights_path を持たない経路のフォールバックとして入れておく(387KB)。
EXTRA_WEIGHTS = ["policy_weights.json"]

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

    # ptcg_ai: .py は全部、.json は「重み以外」+「必要な重みだけ」
    add_many(_iter_files(ss / "ptcg_ai", ["**/*.py"]))
    for path in _iter_files(ss / "ptcg_ai", ["**/*.json"]):
        if path.name.startswith("policy_weights"):
            continue
        add(path)
    wdir = ss / "ptcg_ai" / "learning"
    for arch, gen in FIELD:
        add(wdir / f"policy_weights_{arch}{gen}.json")
    for name in EXTRA_WEIGHTS:
        add(wdir / name)

    # config と既定デッキ(match_context のフォールバックが deck.csv を読む)
    add_many(_iter_files(ss / "configs", ["*.json"]))
    add(ss / "deck.csv")

    # decks/ パッケージ。ptcg_ai/shared/profile_registry.py が `from decks import active` を
    # トップレベル import するため、欠けると ml_policy が import できずに即死する
    # (提出 tarball で同じ事故があった。memory: 提出tarballにdecks/必須)。
    add_many(_iter_files(ss / "decks", ["**/*.py"]))

    # 対戦基盤
    add_many(_iter_files(root / "league", ["*.py"]))

    # ドラパルト側エージェント(対照群 rule と実験群 plan)+ その前提デッキ
    opp = root / "opponents"
    add(opp / "__init__.py")
    add(opp / "dragapult_rule_agent.py")
    add(opp / "dragapult_planning_agent.py")
    add(opp / "dragapult_ex_deck.csv")

    # 相手デッキ(各アーキの代表構築 01.csv)
    for arch, gen in FIELD:
        add(root / "kaggle_replays" / "meta_analysis" / _DECKDIR_OF[gen] / arch / "01.csv")

    # 測定スクリプト本体 + 流用元(参照用)
    add(root / "kaggle_replays" / "kaggle_run" / "drapa_ablation.py")
    add(root / "kaggle_replays" / "_drapa_ceiling.py")

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
    """サイズ内訳の表示単位(= 親ディレクトリ。重みだけは1グループにまとめる)。"""
    if arcname.startswith("sample_submission/ptcg_ai/learning/policy_weights"):
        return "sample_submission/ptcg_ai/learning/policy_weights_*"
    parent = arcname.rsplit("/", 1)[0] if "/" in arcname else "."
    return parent


def report(items: list[tuple[Path, str]]) -> int:
    """内訳サイズを表示し、合計バイト数を返す。"""
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
    """どのコミット + どのローカル変更を固めたのかを記録する。

    blob は **HEAD ではなく作業ツリーの中身** を固める(いま測りたいのは手元のコード)。
    後から「あの測定は何を動かしたのか」を追えるよう、HEAD と、同梱ファイルのうち
    HEAD から変更されているものを残す。
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 - git が無くてもパッケージングは続行する
            return ""

    head = git("rev-parse", "HEAD")
    dirty_all = {line[3:].strip().strip('"') for line in git("status", "--porcelain").splitlines()}
    packaged = {arc for _p, arc in items}
    return {
        "git_head": head,
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
    "sample_submission/decks/active.py",  # profile_registry が import する。欠けると即死
    "league/run_match.py",
    "league/run_league.py",
    "opponents/dragapult_rule_agent.py",
    "opponents/dragapult_planning_agent.py",
    "opponents/dragapult_ex_deck.csv",
    "sample_submission/ptcg_ai/search/damage_counter_plan.py",
    "sample_submission/configs/abl_5_full.json",
    "kaggle_replays/kaggle_run/drapa_ablation.py",
]

# 展開先で実際に import できることを確かめるモジュール。**Kaggle に投げる前に**
# 「同梱漏れによる ModuleNotFoundError」を潰すための検査(実際に decks/ 漏れで1本落とした)。
IMPORT_CHECK = [
    "cg.api", "run_match", "run_league",
    "ptcg_ai.ml_policy.ml_policy_agent",
    "ptcg_ai.search.damage_counter_plan",
    "opponents.dragapult_rule_agent",
    "opponents.dragapult_planning_agent",
]


def verify(blob: Path, run_game: bool = False) -> None:
    """blob を空ディレクトリに展開し、ファイルの有無 + import 可否(+任意で1試合)を確認する。"""
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
            cmd = [sys.executable, str(root / "kaggle_replays" / "kaggle_run" / "drapa_ablation.py"),
                   "--repo-root", str(root), "--groups", "plan", "--games", "2", "--workers", "1",
                   "--only-archs", "crustle", "--seed-base", "999000", "--out", str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")
            if proc.returncode != 0 or not out.exists():
                raise SystemExit("[verify] 1試合スモークに失敗:\n"
                                 + (proc.stdout or "")[-3000:] + (proc.stderr or "")[-3000:])
            games = sum(1 for line in out.read_text(encoding="utf-8").splitlines()
                        if '"group"' in line)
            print(f"[verify] OK: 展開先だけで {games} 試合を完走")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "ptcg_dragapult_eval"),
                    help="ステージングディレクトリ(default: OS の一時ディレクトリ配下)")
    ap.add_argument("--repo-root", default=str(_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="サイズ内訳を表示するだけ")
    ap.add_argument("--clean", action="store_true", help="出力先を作り直す")
    ap.add_argument("--verify-game", action="store_true",
                    help="展開先だけで実際に2試合走らせて検証する(数十秒かかる)")
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
    print("\n次の手順(push_kernel.py に任せるのが確実):")
    print(f"  python {_HERE / 'push_kernel.py'} --push-dataset --dataset-dir {out_dir}")
    print("  # 手で叩く場合は **必ずそのディレクトリを cwd にして -p を付けない**。")
    print("  # Kaggle CLI 2.2.3 は -p に絶対パスを渡すとアップロードキャッシュのパスを")
    print("  # 作らずに [Errno 2] で落ちる。")
    print(f"  cd {out_dir} && kaggle datasets create --dir-mode skip     # 初回のみ")
    print(f"  cd {out_dir} && kaggle datasets version -m \"update\" --dir-mode skip  # 2回目以降")


if __name__ == "__main__":
    main()
