"""BC(模倣学習)の train.py + eval_diagnostics.py を Kaggle Notebook 上で走らせる。

打点計算の配線変更(2026-08-06)の効果測定用。シード1本につき次の2ステップを
1つの Notebook の中で直列に実行する:

    1. python train.py --features <features>.npz --out-weights policy_weights_wired_s<SEED>.json \
           --seed <SEED> --metrics-out metrics_wired_s<SEED>.json
    2. python eval_diagnostics.py --learner marnie_grimmsnarl_ex \
           --learner-weights policy_weights_wired_s<SEED>.json --opponents <POOL8> \
           --games 1600 --workers 4

--seeds で指定した各シードごとに**別々の Notebook**を生成する(並行実行するため)。

    python push_bc_train.py --dry-run

やること(--dry-run 無しの場合、seed ごとに):
  1. 作業ツリーから明示リストでコード一式を ptcg_bc_bundle.zip にまとめ、
     非公開 Dataset(既定 ptcg-bc-train)として上げる(git archive は使わない。
     未コミットのファイルを含める必要があるため)。コードとは別に、features.npz を
     専用の非公開 Dataset(既定 ptcg-bc-features)として上げる(何度も更新するコードと
     固定サイズの大きい特徴量ファイルを分離し、毎回90MBを上げ直さずに済ませるため)。
  2. train.py -> eval_diagnostics.py を直列実行する Notebook をシードごとに生成して push する。
  3. 完了まで待って、出力(学習済み重み・offline指標・両ステップの標準出力ログ)を回収する。

--dry-run はバンドル・Notebook JSON・メタデータを作り、**クリーンルーム検証**(zip をリポジトリ外の
一時ディレクトリへ展開し、そこから極小設定で train.py -> eval_diagnostics.py を実際に動かして
確認する)まで行う。Kaggle には一切送らない。

事前に一度だけ認証が必要:
    kaggle auth login
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# HERE = kaggle_replays/policy_net/distributed
REPO_ROOT = HERE.parent.parent.parent
POLICY_NET_DIR = REPO_ROOT / "kaggle_replays" / "policy_net"
RL_DIR = REPO_ROOT / "kaggle_replays" / "rl"
LEARNING_DIR = REPO_ROOT / "sample_submission" / "ptcg_ai" / "learning"

DEFAULT_CODE_DATASET_SLUG = "ptcg-bc-train"
DEFAULT_FEATURES_DATASET_SLUG = "ptcg-bc-features"
DEFAULT_KERNEL_SLUG_PREFIX = "ptcg-bc-train"

DEFAULT_SEEDS = "42,1,2"
DEFAULT_LEARNER = "marnie_grimmsnarl_ex"
DEFAULT_FEATURES_FILENAME = "features_marnie_grimmsnarl_ex_wired.npz"
DEFAULT_GAMES = 1600
DEFAULT_EVAL_WORKERS = 4

# クリーンルーム検証(--dry-run 内で実行)で使うスタンドイン特徴量。本命の
# features_marnie_grimmsnarl_ex_wired.npz はまだ生成中で存在しないことがあるため、
# 既に存在する中で最小の features_*.npz を代わりに使う(train.py の配線検証が目的で、
# 学習品質は見ない)。
CLEANROOM_STANDIN_FEATURES = "features_omatsuri_ondo.npz"

# バンドルに含めるトップレベルのディレクトリ(REPO_ROOT からの相対パス)。
# push_train_pool.py の BUNDLE_ROOTS を踏襲し、BC 学習に要る kaggle_replays/policy_net を追加した
# (RL 用バンドルには入っていない)。
BUNDLE_ROOTS = [
    Path("sample_submission") / "cg",
    Path("sample_submission") / "ptcg_ai",
    Path("league"),
    Path("kaggle_replays") / "rl",
    Path("kaggle_replays") / "policy_net",
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


# ---------------------------------------------------------------- POOL8 の重み特定
def pool8_weight_names(opponents_csv: str) -> list[str]:
    """--opponents に列挙された学習側名(pools.LEARNER_REGISTRY のキー)が必要とする
    policy_weights*.json の basename を返す。

    pools.py の LEARNER_REGISTRY をそのまま読む(名前を手で列挙してハードコードすると、
    レジストリが変わったときに黙って古いままになるため)。resolve_learner() は
    weights_file が None のとき None を返すが、その場合 PolicyModel(None) は
    sample_submission/ptcg_ai/learning/policy_weights.json(本番の既定重み)を読みに行く
    (policy_model.py の _DEFAULT_WEIGHTS_FILENAME)。alakazam がこれに該当する
    ("alakazam": (None, "alakazam"))ので、policy_weights.json を明示的に含める。
    """
    sys.path.insert(0, str(RL_DIR))
    import pools  # noqa: E402  (RL_DIR 直下、cg 非依存なので import は軽い)

    names = [t.strip() for t in opponents_csv.split(",") if t.strip()]
    out: set[str] = set()
    for name in names:
        if name not in pools.LEARNER_REGISTRY:
            available = ", ".join(sorted(pools.LEARNER_REGISTRY))
            raise SystemExit(f"未知の学習側名: {name!r}. 利用可能な名前: {available}")
        weights_file, _arch = pools.LEARNER_REGISTRY[name]
        out.add(weights_file if weights_file else "policy_weights.json")
    return sorted(out)


def resolve_weights_filter(spec: str | None, opponents_csv: str) -> set[str]:
    if spec:
        requested = {x.strip() for x in spec.split(",") if x.strip()}
    else:
        requested = set(pool8_weight_names(opponents_csv))
    missing = [n for n in requested if not (LEARNING_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"--weights (または POOL8 由来)に存在しない重みが指定された: {sorted(missing)}\n"
            f"  探索先: {LEARNING_DIR}")
    return requested


# ---------------------------------------------------------------- バンドル
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


# バンドルの展開後合計サイズの既定上限。README(push_train_pool.py 側)によると RL 用バンドルは
# 「約20MB → zip 8MB」。BC 用は policy_net/ のコードと POOL8 の重み(8ファイル、計約3MB)が
# 増えるだけなので、数十MBに収まるはず。これを大きく超えたら経路ミス(ステージング先を
# バンドル対象に含めてしまった等)を強く疑うべき、というハードガード。
DEFAULT_MAX_BUNDLE_BYTES = 300 * 1024 * 1024  # 300MB


def iter_bundle_files(weights_filter: set[str], out_dir: Path):
    """(絶対パス, REPO_ROOT からの相対 arcname) を列挙する。

    out_dir: このディレクトリ配下にあるファイルは、BUNDLE_ROOTS のどこに位置しようとも
    常に除外する(ステージング出力先がバンドル対象の中に入っている場合の自己参照防止。
    --out-dir の既定値は BUNDLE_ROOTS に含まれない場所を選んでいるが、呼び出し側が
    別の場所を指定した場合の防御として、ここでも確実に弾く)。
    """
    out_dir = out_dir.resolve()
    seen: set[str] = set()
    for root_rel in BUNDLE_ROOTS:
        root = REPO_ROOT / root_rel
        if not root.exists():
            raise SystemExit(f"バンドル対象が存在しない: {root}")
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            # out_dir 配下(ステージング出力先。zip 自身や dataset-metadata.json 等)は
            # 常に除外する。p は REPO_ROOT 配下の絶対パスなので resolve() 済みとみなせる。
            try:
                p.relative_to(out_dir)
                continue
            except ValueError:
                pass
            rel = p.relative_to(REPO_ROOT)
            rel_parts = rel.parts
            if _excluded(rel_parts, p.name):
                continue
            # kaggle_replays/rl の直下だけ: distributed/(分散自己対戦ワーカー専用) と
            # runs/(実行時生成物)と kaggle_out/(過去の評価実行の回収物、数百MB規模)を除外。
            # 本バンドルが要るのは eval_diagnostics.py / pools.py / collect_pool.py など
            # rl/ 直下のユーティリティだけ。
            if root_rel == Path("kaggle_replays") / "rl":
                rel_in_root = p.relative_to(root)
                if rel_in_root.parts[0] in ("distributed", "runs", "kaggle_out", "league_runs"):
                    continue
            # kaggle_replays/policy_net の直下: *.npz(特徴量、別 Dataset で配る)、
            # scale_* で始まるファイル/ディレクトリ(scale_data/scale_features/scale_logs/
            # scale_metrics、実験用の大容量スクラッチ)、*.log、distributed/(本 push スクリプト
            # 自身のホーム。rl 側の distributed/ 除外と対称にする。既定の --out-dir はこの下に
            # あるため、除外し忘れると自分自身(生成中の zip)をバンドルに取り込んで再帰的に
            # 膨らむ。実際に一度 18GB まで暴走した)を除外。
            if root_rel == Path("kaggle_replays") / "policy_net":
                rel_in_root = p.relative_to(root)
                if rel_in_root.parts[0] in ("scale_data", "scale_features", "scale_logs", "scale_metrics",
                                            "distributed"):
                    continue
                if rel_in_root.parts[0].startswith("scale_"):
                    continue
                if p.suffix == ".npz" or p.suffix == ".log":
                    continue
            # 重みファイルは POOL8 由来(既定)または --weights で絞る。
            if p.parent == LEARNING_DIR and p.name.startswith("policy_weights") and p.suffix == ".json":
                if p.name not in weights_filter:
                    continue
            arc = rel.as_posix()
            if arc in seen:
                continue
            seen.add(arc)
            yield p, arc


def build_bundle_zip(dest: Path, weights_filter: set[str], out_dir: Path,
                     max_bytes: int = DEFAULT_MAX_BUNDLE_BYTES) -> tuple[int, int, int]:
    """ptcg_bc_bundle.zip を作る。戻り値: (ファイル数, 展開後合計バイト数, zip サイズ)。

    累積の展開後サイズが max_bytes を超えた時点で即座に中断する(ハードガード。経路ミスで
    巨大なディレクトリを巻き込んだ場合に、ディスクを埋め尽くす前に気付けるようにする。
    実際に BUNDLE_ROOTS の除外漏れでステージング先を自己参照し、zip が18GBまで膨らんで
    ディスクを埋めかけた事故があった)。中断時はそれまでに積んだファイルをサイズ降順
    上位20件とともに表示する。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    n = 0
    total = 0
    collected: list[tuple[int, str]] = []  # (size, arc) 降順表示用
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in iter_bundle_files(weights_filter, out_dir):
            size = p.stat().st_size
            if total + size > max_bytes:
                collected.sort(reverse=True)
                print(f"\n[ABORT] バンドルの展開後サイズが上限 {max_bytes/1e6:.0f}MB を超えた"
                      f"(現在 {total/1e6:.1f}MB + 追加しようとした {arc} が {size/1e6:.1f}MB)。",
                      file=sys.stderr)
                print(f"  それまでに積んだ {len(collected)} ファイルのうちサイズ降順の上位20件:",
                      file=sys.stderr)
                for sz, a in collected[:20]:
                    print(f"    {sz/1e6:8.2f} MB  {a}", file=sys.stderr)
                raise SystemExit(
                    "バンドルが暴走している疑い。BUNDLE_ROOTS の除外設定・--out-dir の位置を確認する。")
            z.write(p, arc)
            n += 1
            total += size
            collected.append((size, arc))
    return n, total, dest.stat().st_size


def _current_dataset_version(ref: str):
    """Dataset の現行バージョン番号。取得できなければ None(その場合は版チェックを諦める)。

    push_train_pool.py の同名関数をそのまま流用(push_train_pool.py 自体は変更しない方針
    なので、ロジックをここへコピーしている)。
    """
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


# ---------------------------------------------------------------- クリーンルーム検証
def run_cleanroom_check(zip_path: Path) -> None:
    """zip をリポジトリ外の一時ディレクトリへ展開し、そこから train.py -> eval_diagnostics.py を
    極小設定で実際に実行して確認する。

    過去に(push_train_pool.py の運用で)この検証で「デッキが入っていない」
    「__main__ ガード欠落で BrokenProcessPool」「glob が特定の重みを除外」の3件を実際に
    見つけている。今回は特に、train.py の --out-weights(相対パス、実行時 cwd 基準で解決)を
    eval_diagnostics.py の --learner-weights(相対パスなら pools.WDIR 基準で解決される。
    train.py の出力先とは基準が違う!)へそのまま渡すと壊れるという、本バンドル固有の
    配線を検証する目的もある。本番の Notebook では --learner-weights に絶対パスを渡すことで
    これを回避しているが、その絶対パス受け渡し自体をここで実地確認する。
    """
    print("\n=== クリーンルーム検証 ===")
    tmp = Path(tempfile.mkdtemp(prefix="ptcg_bc_cleanroom_"))
    print(f"  展開先(リポジトリ外): {tmp}")
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)

        libcg = tmp / "sample_submission" / "cg" / "libcg.so"
        if libcg.exists():
            try:
                libcg.chmod(0o755)
            except OSError:
                pass

        cr_policy_net = tmp / "kaggle_replays" / "policy_net"
        cr_rl = tmp / "kaggle_replays" / "rl"
        need = [
            tmp / "sample_submission" / "cg" / "libcg.so",
            tmp / "sample_submission" / "ptcg_ai" / "learning" / "policy_model.py",
            tmp / "league" / "run_league.py",
            cr_policy_net / "train.py",
            cr_rl / "eval_diagnostics.py",
            cr_rl / "pools.py",
            tmp / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json",
            tmp / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam" / "01.csv",
        ]
        missing = [str(p.relative_to(tmp)) for p in need if not p.exists()]
        print("  --- 必要ファイルの確認 ---")
        for p in need:
            print(f"    {'OK  ' if p.exists() else 'なし'} {p.relative_to(tmp)}")
        if missing:
            raise SystemExit(f"クリーンルーム検証: バンドルにファイルが足りない: {missing}")

        # 特徴量は常に小さいスタンドインを使う(本命の features_*_wired.npz が既に90MB級で
        # 存在していても、あえて使わない。クリーンルーム検証の目的は「バンドルだけで
        # train.py -> eval_diagnostics.py の配線が動くか」であって学習品質の測定ではないため、
        # 90MB の allow_pickle=True 読み込みで時間を溶かす必要が無い)。
        standin_src = POLICY_NET_DIR / CLEANROOM_STANDIN_FEATURES
        standin_name = CLEANROOM_STANDIN_FEATURES
        print(f"  特徴量: スタンドイン {standin_name} を使用"
              f"(配線検証が目的で学習品質は見ない。本命の features_*_wired.npz は使わない)")
        if not standin_src.exists():
            raise SystemExit(f"クリーンルーム検証: スタンドイン特徴量が無い: {standin_src}")
        shutil.copyfile(standin_src, cr_policy_net / standin_name)

        cr_weights_out = "_cleanroom_policy_weights.json"
        cr_metrics_out = "_cleanroom_metrics.json"
        train_cmd = [
            sys.executable, "-u", "train.py",
            "--features", standin_name,
            "--limit", "500",
            "--max-epochs", "1",
            "--out-weights", cr_weights_out,
            "--seed", "42",
            "--metrics-out", cr_metrics_out,
        ]
        print(f"\n  $ {' '.join(train_cmd)}   (cwd={cr_policy_net})")
        r1 = subprocess.run(train_cmd, cwd=cr_policy_net, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        tail1 = "\n".join((r1.stdout or "").splitlines()[-25:])
        print(tail1)
        if r1.stderr:
            print("  --- stderr(末尾) ---")
            print("\n".join(r1.stderr.splitlines()[-25:]))
        if r1.returncode != 0:
            raise SystemExit(f"クリーンルーム検証: train.py が失敗(exit {r1.returncode})")
        out_weights_path = cr_policy_net / cr_weights_out
        if not out_weights_path.exists():
            raise SystemExit(f"クリーンルーム検証: train.py が重みJSONを書き出さなかった: {out_weights_path}")
        print(f"  train.py OK -> {out_weights_path} ({out_weights_path.stat().st_size/1e3:.1f} KB)")

        # eval_diagnostics.py 側: --learner-weights に絶対パスを渡す(本番 Notebook と同じ配線)。
        eval_cmd = [
            sys.executable, "-u", "eval_diagnostics.py",
            "--learner", "alakazam",
            "--learner-weights", str(out_weights_path.resolve()),
            "--opponents", "alakazam",
            "--games", "4",
            "--workers", "1",
        ]
        print(f"\n  $ {' '.join(eval_cmd)}   (cwd={cr_rl})")
        r2 = subprocess.run(eval_cmd, cwd=cr_rl, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        tail2 = "\n".join((r2.stdout or "").splitlines()[-30:])
        print(tail2)
        if r2.stderr:
            print("  --- stderr(末尾) ---")
            print("\n".join(r2.stderr.splitlines()[-25:]))
        if r2.returncode != 0:
            raise SystemExit(f"クリーンルーム検証: eval_diagnostics.py が失敗(exit {r2.returncode})")
        if "有効試合 4/4" not in r2.stdout and "有効試合" not in r2.stdout:
            raise SystemExit("クリーンルーム検証: eval_diagnostics.py の出力に想定した集計行が無い")

        print("\n  クリーンルーム検証: PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- Notebook 生成
def _code_cell(lines: list[str]) -> dict:
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def _md_cell(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def build_notebook(code_dataset_slug: str, features_dataset_slug: str, features_filename: str,
                    seed: int, learner: str, opponents: str, games: int, eval_workers: int) -> dict:
    out_weights_name = f"policy_weights_wired_s{seed}.json"
    metrics_name = f"metrics_wired_s{seed}.json"
    train_log_name = f"train_wired_s{seed}.log"
    eval_log_name = f"eval_wired_s{seed}.log"

    cell_intro = _md_cell(
        f"# BC 学習 + 評価(打点計算配線変更、seed={seed})— Kaggle 実行\n\n"
        "1. `train.py` で `features_marnie_grimmsnarl_ex_wired.npz` から模倣学習(BC)する\n"
        "2. `eval_diagnostics.py` で POOL8 相手に評価する\n\n"
        "**非公開 Notebook・非公開 Dataset x2。GPU 不要・インターネット不要。**\n\n"
        f"入力: Dataset `{code_dataset_slug}`(コード一式)・`{features_dataset_slug}`"
        f"(`{features_filename}`)\n"
        "出力: `/kaggle/working/` 直下に重み・offline指標・両ステップの標準出力ログ\n\n"
        "Save & Run All にすれば、ブラウザを閉じても裏で最後まで走る(想定: 学習 5時間強 + "
        "評価 16分、4コア)。"
    )

    cell_extract = _code_cell([
        "import glob, os, shutil, sys, zipfile",
        "",
        "print('--- /kaggle/input ---')",
        "for p in sorted(glob.glob('/kaggle/input/**', recursive=True))[:80]:",
        "    print(' ', p)",
        "",
        "REPO = '/kaggle/working/repo'",
        "os.makedirs(REPO, exist_ok=True)",
        "",
        "# コード Dataset: Kaggle は Dataset にアップロードした zip を**既定で展開する**ので、",
        "# /kaggle/input には zip ではなくディレクトリ木が置かれることがある。",
        "# どちらの形でも動くようにする(push_train_pool.py で zip 前提で書いて失敗した実績があるため)。",
        "zips = glob.glob('/kaggle/input/**/ptcg_bc_bundle.zip', recursive=True)",
        "if zips:",
        "    print('コード: zip から展開:', zips[0])",
        "    zipfile.ZipFile(zips[0]).extractall(REPO)",
        "else:",
        "    marks = glob.glob('/kaggle/input/**/sample_submission/cg/libcg.so', recursive=True)",
        "    if not marks:",
        "        raise SystemExit('ptcg_bc_bundle.zip も展開済みツリーも /kaggle/input 以下に無い。'",
        "                         'コード Dataset が Notebook に添付され、処理が完了しているか確認する。')",
        "    src = os.path.dirname(os.path.dirname(os.path.dirname(marks[0])))",
        "    print('コード: 展開済みツリーから複製:', src)",
        "    # /kaggle/input は読み取り専用。train.py は重みを書き出すので writable な場所に複製する。",
        "    shutil.copytree(src, REPO, dirs_exist_ok=True)",
        "    os.chmod(os.path.join(REPO, 'sample_submission', 'cg', 'libcg.so'), 0o755)",
        "",
        f"FEATURES_NAME = {features_filename!r}",
        "feat_hits = glob.glob(f'/kaggle/input/**/{FEATURES_NAME}', recursive=True)",
        "if not feat_hits:",
        "    raise SystemExit(f'特徴量 Dataset に {FEATURES_NAME} が見つからない。'",
        "                     '特徴量 Dataset が Notebook に添付されているか確認する。')",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "dst_features = os.path.join(policy_net_dir, FEATURES_NAME)",
        "print('特徴量:', feat_hits[0], '->', dst_features)",
        "shutil.copyfile(feat_hits[0], dst_features)",
        "",
        "for p in (REPO, os.path.join(REPO, 'sample_submission'), os.path.join(REPO, 'league'),",
        "          os.path.join(REPO, 'kaggle_replays', 'rl')):",
        "    if p not in sys.path:",
        "        sys.path.insert(0, p)",
        "",
        "need = ['sample_submission/cg/libcg.so',",
        "        'sample_submission/ptcg_ai/learning/policy_model.py',",
        "        'sample_submission/ptcg_ai/learning/policy_weights.json',",
        "        'league/run_league.py',",
        "        'kaggle_replays/policy_net/train.py',",
        "        'kaggle_replays/rl/eval_diagnostics.py',",
        "        'kaggle_replays/rl/pools.py',",
        "        'kaggle_replays/meta_analysis/archetype_decks/alakazam/01.csv']",
        "missing = [n for n in need if not os.path.exists(os.path.join(REPO, n))]",
        "print('\\n--- 必要ファイルの確認 ---')",
        "for n in need:",
        "    print(('  OK  ' if n not in missing else '  なし '), n)",
        "if missing:",
        "    raise SystemExit(f'コード Dataset の中身が足りない(古い版が添付された可能性): {missing}')",
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
        f"SEED = {seed}",
        f"LEARNER = {learner!r}",
        f"OPPONENTS = {opponents!r}",
        f"GAMES = {games}",
        f"EVAL_WORKERS = {eval_workers}",
        f"OUT_WEIGHTS_NAME = {out_weights_name!r}",
        f"METRICS_NAME = {metrics_name!r}",
        f"TRAIN_LOG_NAME = {train_log_name!r}",
        f"EVAL_LOG_NAME = {eval_log_name!r}",
        "",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "rl_dir = os.path.join(REPO, 'kaggle_replays', 'rl')",
        "",
        "def run_streamed(cmd, cwd, log_path):",
        "    # 標準出力を画面(セル出力)と /kaggle/working のログファイル両方に残す。",
        "    # train.py / eval_diagnostics.py 自体はログファイルを書かないため、ここで tee する。",
        "    print('$', ' '.join(cmd), flush=True)",
        "    env = dict(os.environ)",
        "    env['PYTHONUNBUFFERED'] = '1'",
        "    t0 = time.time()",
        "    with open(log_path, 'w', encoding='utf-8') as lf:",
        "        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,",
        "                                stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)",
        "        for line in proc.stdout:",
        "            print(line, end='', flush=True)",
        "            lf.write(line)",
        "    ret = proc.wait()",
        "    elapsed = time.time() - t0",
        "    print(f'\\n[exit={ret}] {elapsed:.0f}s elapsed', flush=True)",
        "    return ret",
        "",
        "t_run0 = time.time()",
        "",
        "print('=== 1/2: train.py (BC学習) ===', flush=True)",
        "train_cmd = [sys.executable, '-u', 'train.py',",
        f"             '--features', {features_filename!r},",
        "             '--out-weights', OUT_WEIGHTS_NAME,",
        "             '--seed', str(SEED),",
        "             '--metrics-out', METRICS_NAME]",
        "train_log_path = os.path.join('/kaggle/working', TRAIN_LOG_NAME)",
        "ret = run_streamed(train_cmd, policy_net_dir, train_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'train.py が失敗(exit {ret})')",
        "",
        "# eval_diagnostics.py の --learner-weights は相対パスだと pools.WDIR",
        "# (sample_submission/ptcg_ai/learning/)基準で解決される。train.py の出力先",
        "# (policy_net_dir、cwd基準)とは基準が違うため、必ず絶対パスで渡す。",
        "learner_weights_abs = os.path.join(policy_net_dir, OUT_WEIGHTS_NAME)",
        "if not os.path.exists(learner_weights_abs):",
        "    raise SystemExit(f'train.py の出力が見つからない: {learner_weights_abs}')",
        "",
        "print('\\n=== 2/2: eval_diagnostics.py (POOL8評価) ===', flush=True)",
        "eval_cmd = [sys.executable, '-u', 'eval_diagnostics.py',",
        "            '--learner', LEARNER,",
        "            '--learner-weights', learner_weights_abs,",
        "            '--opponents', OPPONENTS,",
        "            '--games', str(GAMES),",
        "            '--workers', str(EVAL_WORKERS)]",
        "eval_log_path = os.path.join('/kaggle/working', EVAL_LOG_NAME)",
        "ret = run_streamed(eval_cmd, rl_dir, eval_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'eval_diagnostics.py が失敗(exit {ret})')",
        "",
        "print(f'\\n[全体完了] {time.time()-t_run0:.0f}s elapsed', flush=True)",
    ])

    cell_collect = _code_cell([
        "import os, shutil",
        "",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "candidates = [",
        "    os.path.join(policy_net_dir, OUT_WEIGHTS_NAME),",
        "    os.path.join(policy_net_dir, METRICS_NAME),",
        "]",
        "for p in candidates:",
        "    if os.path.exists(p):",
        "        dest = os.path.join('/kaggle/working', os.path.basename(p))",
        "        if os.path.abspath(dest) != os.path.abspath(p):",
        "            shutil.copyfile(p, dest)",
        "        print('回収:', dest, f'({os.path.getsize(dest)/1e6:.3f} MB)')",
        "    else:",
        "        print('回収対象なし(見つからない):', p)",
        "",
        "print('\\n--- /kaggle/working ---')",
        "for p in sorted(os.listdir('/kaggle/working')):",
        "    full = os.path.join('/kaggle/working', p)",
        "    if os.path.isfile(full):",
        "        print(f'  {p}  ({os.path.getsize(full)/1e6:.3f} MB)')",
        "    else:",
        "        print(f'  {p}/')",
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
    dest.write_text(json.dumps(doc, ensure_ascii=True, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- Dataset の push
def push_dataset(data_dir: Path, slug: str, user: str, dataset_wait: int) -> None:
    """Dataset を新規作成 or 新バージョンとして更新し、新版が実際に反映されるまで待つ。

    push_train_pool.py で実際に踏んだ2つの罠への対策を踏襲する:
      - status="ready" は「何らかの版が使える」を意味するだけで、今上げた版が反映された
        ことは保証しない -> currentVersionNumber をポーリングして版が上がったことを確認する。
      - 固定秒数の sleep で済ませない -> ready になるまで待つ。
    """
    ref = f"{user}/{slug}"
    prev_version = _current_dataset_version(ref)
    print(f"  [{slug}] 更新前の Dataset バージョン: {prev_version}")

    exists = kaggle("datasets", "status", ref, check=False).returncode == 0
    if exists:
        print(f"  [{slug}] Dataset を更新(新バージョン)")
        kaggle("datasets", "version", "-p", str(data_dir), "-m", "update")
    else:
        print(f"  [{slug}] Dataset を新規作成(非公開)")
        kaggle("datasets", "create", "-p", str(data_dir))

    print(f"  [{slug}] Dataset の処理完了を待つ(最大 {dataset_wait}s)", flush=True)
    t_ds = time.time()
    while time.time() - t_ds < dataset_wait:
        r = kaggle("datasets", "status", ref, check=False)
        st = (r.stdout or "").strip().lower()
        if "ready" in st:
            cur = _current_dataset_version(ref)
            if prev_version is not None and cur is not None and cur <= prev_version:
                print(f"    ready だが版は {cur} のまま({prev_version} から未更新)。待機継続",
                      flush=True)
                time.sleep(10)
                continue
            print(f"    ready / バージョン {prev_version} -> {cur} ({time.time() - t_ds:.0f}s)")
            return
        if "error" in st:
            raise SystemExit(f"Dataset の処理が失敗した: {r.stdout}")
        time.sleep(10)
    raise SystemExit(
        f"Dataset {ref} が {dataset_wait}s 以内に ready にならなかった。"
        f"--dataset-wait を伸ばすか、https://www.kaggle.com/datasets/{ref} を確認する。")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=DEFAULT_SEEDS,
                    help=f"カンマ区切りのシード。既定 {DEFAULT_SEEDS}。シードごとに別 Notebook を作る。")
    ap.add_argument("--learner", default=DEFAULT_LEARNER)
    ap.add_argument("--features-filename", default=DEFAULT_FEATURES_FILENAME,
                    help="kaggle_replays/policy_net/ 配下の特徴量ファイル名(コード zip には含めない)")
    ap.add_argument("--opponents", default=None,
                    help="評価の相手プール(カンマ区切り、pools.LEARNER_REGISTRY のキー)。"
                         "省略時は pools.POOL8。")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--eval-workers", type=int, default=DEFAULT_EVAL_WORKERS,
                    help="eval_diagnostics.py の --workers(Kaggle 側、既定4コア)")
    ap.add_argument("--code-dataset-slug", default=DEFAULT_CODE_DATASET_SLUG)
    ap.add_argument("--features-dataset-slug", default=DEFAULT_FEATURES_DATASET_SLUG)
    ap.add_argument("--kernel-slug-prefix", default=DEFAULT_KERNEL_SLUG_PREFIX,
                    help="Notebook slug は '<prefix>-s<seed>' になる。")
    ap.add_argument("--username", default=None)
    ap.add_argument("--weights", default=None,
                    help="カンマ区切りの重みファイル basename。省略時は --opponents(既定 POOL8)"
                         "が必要とする重みを pools.LEARNER_REGISTRY から自動算出する。")
    ap.add_argument("--dry-run", action="store_true",
                    help="バンドル・Notebook JSON・メタデータを作り、クリーンルーム検証まで行う。"
                         "Kaggle へは一切送らない。")
    ap.add_argument("--skip-cleanroom", action="store_true",
                    help="--dry-run のクリーンルーム検証をスキップする(デバッグ用)")
    ap.add_argument("--out-dir", default=str(HERE / ".stage_bc_train"))
    ap.add_argument("--max-bundle-mb", type=float, default=DEFAULT_MAX_BUNDLE_BYTES / 1e6,
                    help=f"コードバンドルの展開後合計サイズの上限(MB、既定"
                         f"{DEFAULT_MAX_BUNDLE_BYTES/1e6:.0f})。超えたら即中断する暴走防止ガード。")
    ap.add_argument("--dataset-wait", type=int, default=600,
                    help="Dataset の新バージョンが反映されるまでの待ち時間(秒)。"
                         "特徴量 Dataset は90MB前後あるため既定を長めにしている。")
    ap.add_argument("--timeout", type=int, default=7 * 3600,
                    help="Notebook 完了待ちの上限(秒、既定7時間 = 学習5時間強+評価16分+余裕)")
    ap.add_argument("--no-wait", action="store_true", help="push だけして待たない")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    code_data_dir = out_dir / "code_data"

    sys.path.insert(0, str(RL_DIR))
    import pools  # noqa: E402
    opponents_csv = args.opponents or pools.POOL8

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds が空")

    weights_filter = resolve_weights_filter(args.weights, opponents_csv)
    print(f"学習側: {args.learner}")
    print(f"評価opponents: {opponents_csv}")
    print(f"シード: {seeds}")
    print(f"POOL8由来の重み({len(weights_filter)}件): {sorted(weights_filter)}")

    # ------------------------------------------------------------ コード Dataset
    zip_path = code_data_dir / "ptcg_bc_bundle.zip"
    max_bundle_bytes = int(args.max_bundle_mb * 1e6)
    n, total, zsize = build_bundle_zip(zip_path, weights_filter, out_dir, max_bundle_bytes)
    print(f"\nコードバンドル: {n} ファイル, 展開後 {total/1e6:.1f} MB -> zip {zsize/1e6:.1f} MB ({zip_path})")

    user = detect_username(args.username)

    code_dataset_meta = {
        "title": "ptcg bc train",
        "id": f"{user}/{args.code_dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (code_data_dir / "dataset-metadata.json").write_text(
        json.dumps(code_dataset_meta, indent=2), encoding="utf-8")
    print(f"code dataset-metadata: {code_data_dir / 'dataset-metadata.json'}")

    # ------------------------------------------------------------ 特徴量 Dataset
    features_local_path = POLICY_NET_DIR / args.features_filename
    features_data_dir = out_dir / "features_data"
    features_data_dir.mkdir(parents=True, exist_ok=True)
    if features_local_path.exists():
        dst = features_data_dir / features_local_path.name
        if dst.exists():
            dst.unlink()
        shutil.copyfile(features_local_path, dst)
        print(f"\n特徴量: {features_local_path} ({features_local_path.stat().st_size/1e6:.1f} MB) "
              f"-> {dst}")
    else:
        print(f"\n[WARN] 特徴量ファイルが未生成: {features_local_path}")
        print("       (--dry-run のクリーンルーム検証はスタンドインで代用する。"
              "実際に push する前にこのファイルを作ること)")

    features_dataset_meta = {
        "title": "ptcg bc features",
        "id": f"{user}/{args.features_dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (features_data_dir / "dataset-metadata.json").write_text(
        json.dumps(features_dataset_meta, indent=2), encoding="utf-8")
    print(f"features dataset-metadata: {features_data_dir / 'dataset-metadata.json'}")

    # ------------------------------------------------------------ Notebook (seedごと)
    kernel_dirs: dict[int, Path] = {}
    kernel_slugs: dict[int, str] = {}
    for seed in seeds:
        kernel_slug = f"{args.kernel_slug_prefix}-s{seed}"
        kernel_dir = out_dir / f"kernel_s{seed}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = f"{kernel_slug}.ipynb"
        notebook_doc = build_notebook(
            args.code_dataset_slug, args.features_dataset_slug, args.features_filename,
            seed, args.learner, opponents_csv, args.games, args.eval_workers,
        )
        write_notebook_json(notebook_doc, kernel_dir / notebook_name)

        kernel_meta = {
            "id": f"{user}/{kernel_slug}",
            # title は固定にしないこと。Kaggle は Notebook を新規作成するとき id ではなく
            # title を slug 化して ref を決める(push_train_pool.py で実際に踏んだ罠)。
            "title": kernel_slug.replace("-", " "),
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [f"{user}/{args.code_dataset_slug}", f"{user}/{args.features_dataset_slug}"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2), encoding="utf-8")

        print(f"\nseed={seed}  kernel={user}/{kernel_slug}")
        print(f"  notebook: {kernel_dir / notebook_name}")
        print(f"  kernel-metadata: {kernel_dir / 'kernel-metadata.json'}")
        kernel_dirs[seed] = kernel_dir
        kernel_slugs[seed] = kernel_slug

    if args.dry_run:
        if not args.skip_cleanroom:
            run_cleanroom_check(zip_path)
        else:
            print("\n--skip-cleanroom のためクリーンルーム検証を省略した。")
        print("\n--dry-run のため Kaggle へは何も送らない。")
        return

    # ------------------------------------------------------------ ここから先は実送信
    print(f"\nコード Dataset を push: {user}/{args.code_dataset_slug}")
    push_dataset(code_data_dir, args.code_dataset_slug, user, args.dataset_wait)

    if not features_local_path.exists():
        raise SystemExit(
            f"特徴量ファイルが無いため push できない: {features_local_path}\n"
            "(先に build_features.py 等で生成すること)")
    print(f"\n特徴量 Dataset を push: {user}/{args.features_dataset_slug}")
    push_dataset(features_data_dir, args.features_dataset_slug, user, args.dataset_wait)

    for seed in seeds:
        kernel_slug = kernel_slugs[seed]
        kernel_dir = kernel_dirs[seed]
        print(f"\nseed={seed}: Notebook {user}/{kernel_slug} を push して実行開始")
        kaggle("kernels", "push", "-p", str(kernel_dir))

    if args.no_wait:
        print("\n進行状況の確認:")
        for seed in seeds:
            print(f"  seed={seed}: https://www.kaggle.com/code/{user}/{kernel_slugs[seed]}")
        return

    t0 = time.time()
    last: dict[int, str] = {s: "" for s in seeds}
    pending = set(seeds)
    while pending and time.time() - t0 < args.timeout:
        for seed in list(pending):
            kernel_slug = kernel_slugs[seed]
            r = kaggle("kernels", "status", f"{user}/{kernel_slug}", check=False)
            out = (r.stdout or "").strip()
            if out != last[seed]:
                print(f"  [{time.time()-t0:5.0f}s] seed={seed}: {out}", flush=True)
                last[seed] = out
            low = out.lower()
            if "complete" in low:
                pending.discard(seed)
            elif "error" in low or "cancel" in low:
                raise SystemExit(
                    f"seed={seed}: Kaggle 側で失敗: {out}\n"
                    f"  https://www.kaggle.com/code/{user}/{kernel_slug} でログを確認する。")
        if pending:
            time.sleep(20)
    if pending:
        raise SystemExit(
            f"待ち時間の上限に達した(未完了: {sorted(pending)})。"
            f"--no-wait で投げっぱなしにして後で回収する。")

    for seed in seeds:
        kernel_slug = kernel_slugs[seed]
        dest = out_dir / f"out_s{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{user}/{kernel_slug}", "-p", str(dest))
        print(f"seed={seed} 完了。回収先: {dest}")


if __name__ == "__main__":
    main()
