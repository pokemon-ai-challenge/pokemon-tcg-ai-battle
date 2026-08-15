"""BC(模倣学習)の処置群/対照群を学習し、ミラー戦で評価する Notebook を Kaggle 上で走らせる。

design-transformer-representation-2026-08-08.md §8.1 の評価手順を自動化したもの。
T1(card_id カウント特徴)・T2(--use-board-set)・Negative Control(--shuffle-card-ids)は
いずれも「同じ features.npz を共有し、train.py への追加引数だけが違う2モデルを学習して
直接対戦させる」という同じ構造なので、1本のスクリプトで済ませる。

**既存の対戦相手プール(POOL8)の重みは251次元で、715次元エンコーダでは
`PolicyModel.is_ready=False` になり対戦相手として機能しない(design書 §8.1 で確認済み)。**
そのため本スクリプトは POOL8 とは対戦させず、**処置群 vs 対照群のミラー戦**で評価する。

シード1本につき次の3ステップを1つの Notebook の中で直列に実行する:

    1. python train.py --features <features>.npz --out-weights policy_weights_treatment_s<SEED>.json \
           --seed <SEED> [--metrics-out ...] <TREATMENT_EXTRA_ARGS>
    2. python train.py --features <features>.npz --out-weights policy_weights_control_s<SEED>.json \
           --seed <SEED> [--metrics-out ...] <CONTROL_EXTRA_ARGS>
    3. python eval_mirror_symmetric.py \
           --model-a <policy_weights_treatment_s<SEED>.json の絶対パス> \
           --model-b <policy_weights_control_s<SEED>.json の絶対パス> \
           --archetype <archetype> --games <GAMES> --workers <EVAL_WORKERS>
       (2026-08-11 発見: 旧 eval_diagnostics.py 経由のミラー戦は "learner役"(--learner-weights側)
        が温度サンプリング・"opponent役"(--opponents側)が純粋greedyという非対称な実装で、
        処置群を常にlearner役に固定していたため役割バイアスだけでElo -56相当が混入していた。
        eval_mirror_symmetric.py は両陣営とも greedy に統一した対称評価を行う。詳細は
        design-transformer-representation-2026-08-08.md §8.3。重みパスを直接受け取るため、
        旧方式で必要だった --extra-registry 用の registry JSON 生成は不要になった)

--seeds で指定した各シードごとに**別々の Notebook**を生成する(並行実行するため)。

使用例(T1: フルT1 vs T1をablateした対照群。値が "-" で始まる引数は "--control-args=..." の
"=" 記法で渡すこと。スペース区切り("--control-args value")だと argparse が値の中身次第で
"expected one argument" になることがある。実測: --control-args は動いたが --treatment-args は
落ちた。原因の切り分けはしていないため、両方とも "=" 記法を既定の書き方にする):
    python push_bc_ab_eval.py \
        --learner-archetype marnie_grimmsnarl_ex \
        --features-filename features_marnie_grimmsnarl_ex_t1.npz \
        --control-args="--ablate-features self_board_card_slot,self_discard_card_slot,opp_board_card_slot,opp_discard_card_slot" \
        --tag t1 --seeds 42,1,2 --games 3500 --dry-run

使用例(T2: --use-board-set 有り vs 無し。--treatment-args の値が "-" で始まる場合、
argparse に空文字列やオプションと誤認識されないよう "--treatment-args=..." の
"=" 記法で渡すこと(スペース区切りだと "expected one argument" で落ちる)):
    python push_bc_ab_eval.py \
        --learner-archetype marnie_grimmsnarl_ex \
        --features-filename features_marnie_grimmsnarl_ex_t1.npz \
        --treatment-args="--use-board-set" \
        --tag t2 --seeds 42,1,2 --games 3500 --dry-run

使用例(Negative Control: 本物の card_id vs シャッフル版。features は別々に作る必要がある。
--control-features-filename で対照群専用の features.npz を指定できる):
    python push_bc_ab_eval.py \
        --learner-archetype marnie_grimmsnarl_ex \
        --features-filename features_marnie_grimmsnarl_ex_t1.npz \
        --control-features-filename features_marnie_grimmsnarl_ex_t1_shuffled.npz \
        --tag negctl --seeds 42,1,2 --games 3500 --dry-run

    python push_bc_ab_eval.py --dry-run   # 既定(t1)の内容を確認するだけ

やること(--dry-run 無しの場合、seed ごとに):
  1. 作業ツリーから明示リストでコード一式を ptcg_bc_ab_bundle.zip にまとめ、
     非公開 Dataset(既定 ptcg-bc-ab-eval)として上げる。features.npz は専用の非公開 Dataset
     (既定 ptcg-bc-ab-features)として上げる(push_bc_train.py と同じ分離)。
  2. train.py(処置群)-> train.py(対照群)-> eval_mirror_symmetric.py(対称ミラー戦)を直列実行する
     Notebook をシードごとに生成して push する。
  3. 完了まで待って、出力(処置群/対照群の重み・offline指標・3ステップの標準出力ログ)を回収する。

--dry-run はバンドル・Notebook JSON・メタデータを作り、**クリーンルーム検証**(zip をリポジトリ外の
一時ディレクトリへ展開し、そこから極小設定で train.py x2 -> eval_mirror_symmetric.py を実際に動かして
確認する)まで行う。Kaggle には一切送らない。

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

DEFAULT_CODE_DATASET_SLUG = "ptcg-bc-ab-eval"
DEFAULT_FEATURES_DATASET_SLUG = "ptcg-bc-ab-features"
DEFAULT_KERNEL_SLUG_PREFIX = "ptcg-bc-ab"

DEFAULT_SEEDS = "42,1,2"
DEFAULT_LEARNER_ARCHETYPE = "marnie_grimmsnarl_ex"
DEFAULT_FEATURES_FILENAME = "features_marnie_grimmsnarl_ex_t1.npz"
DEFAULT_TAG = "t1"
DEFAULT_GAMES = 3500  # design書 §4 原則7: δ_min=0.03 なら約3,500試合/群
DEFAULT_EVAL_WORKERS = 4

# クリーンルーム検証(--dry-run 内で実行)で使うスタンドイン特徴量。本命の features はまだ
# 生成中/巨大で存在しないことがあるため、715次元エンコーダ(T1完了後、board_card_ids付き)で
# 作った小さい専用スタンドインを使う(train.py x2 -> eval_mirror_symmetric.py の配線検証が目的で、
# 学習品質は見ない)。旧 features_omatsuri_ondo.npz は166次元・board_card_ids無しの
# 古いエンコーダ産で使えない(2026-08-09確認)。
CLEANROOM_STANDIN_FEATURES = "features_omatsuri_ondo_v715_standin.npz"

# バンドルに含めるトップレベルのディレクトリ(REPO_ROOT からの相対パス)。
# push_bc_train.py の BUNDLE_ROOTS と同じ(POOL8 の重みは今回一切使わないので含めない)。
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


# バンドルの展開後合計サイズの既定上限(push_bc_train.py と同じハードガード)。
DEFAULT_MAX_BUNDLE_BYTES = 300 * 1024 * 1024  # 300MB


def iter_bundle_files(out_dir: Path):
    """(絶対パス, REPO_ROOT からの相対 arcname) を列挙する。

    push_bc_pool_v251.py と同様、今回の学習は教師データからの模倣であり POOL8 の重みは
    一切使わない(対戦相手は同じ Kaggle 実行内で学習した対照群)ため、
    policy_weights*.json は常に除外する(--weights による絞り込み自体が不要)。
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
            try:
                p.relative_to(out_dir)
                continue
            except ValueError:
                pass
            rel = p.relative_to(REPO_ROOT)
            rel_parts = rel.parts
            if _excluded(rel_parts, p.name):
                continue
            if root_rel == Path("kaggle_replays") / "rl":
                rel_in_root = p.relative_to(root)
                if rel_in_root.parts[0] in ("distributed", "runs", "kaggle_out", "league_runs"):
                    continue
            if root_rel == Path("kaggle_replays") / "policy_net":
                rel_in_root = p.relative_to(root)
                if rel_in_root.parts[0] in ("scale_data", "scale_features", "scale_logs", "scale_metrics",
                                            "distributed"):
                    continue
                if rel_in_root.parts[0].startswith("scale_"):
                    continue
                if p.suffix == ".npz" or p.suffix == ".log":
                    continue
            # 重みファイルは常に除外(今回の学習では一切使わない。対照群は同じ実行内で学習する)。
            if p.name.startswith("policy_weights") and p.suffix == ".json":
                continue
            arc = rel.as_posix()
            if arc in seen:
                continue
            seen.add(arc)
            yield p, arc


def build_bundle_zip(dest: Path, out_dir: Path, max_bytes: int = DEFAULT_MAX_BUNDLE_BYTES) -> tuple[int, int, int]:
    """ptcg_bc_ab_bundle.zip を作る。戻り値: (ファイル数, 展開後合計バイト数, zip サイズ)。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    n = 0
    total = 0
    collected: list[tuple[int, str]] = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in iter_bundle_files(out_dir):
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


# ---------------------------------------------------------------- クリーンルーム検証
def run_cleanroom_check(zip_path: Path, learner_archetype: str,
                        treatment_extra: list[str], control_extra: list[str]) -> None:
    """zip をリポジトリ外の一時ディレクトリへ展開し、そこから
    train.py(処置群) -> train.py(対照群) -> eval_mirror_symmetric.py(対称ミラー戦)を極小設定で
    実際に実行して確認する。

    push_bc_train.py の教訓(重みパスは絶対パスで渡す必要がある。相対パスだと
    実行時の cwd 基準で解決され、train.py の出力先とズレる)をそのまま踏襲し、
    eval_mirror_symmetric.py の --model-a / --model-b にも絶対パスを渡す。
    """
    print("\n=== クリーンルーム検証 ===")
    tmp = Path(tempfile.mkdtemp(prefix="ptcg_bc_ab_cleanroom_"))
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
            cr_rl / "eval_mirror_symmetric.py",
            tmp / "kaggle_replays" / "meta_analysis" / "archetype_decks" / learner_archetype / "01.csv",
        ]
        missing = [str(p.relative_to(tmp)) for p in need if not p.exists()]
        print("  --- 必要ファイルの確認 ---")
        for p in need:
            print(f"    {'OK  ' if p.exists() else 'なし'} {p.relative_to(tmp)}")
        if missing:
            raise SystemExit(f"クリーンルーム検証: バンドルにファイルが足りない: {missing}")

        standin_src = POLICY_NET_DIR / CLEANROOM_STANDIN_FEATURES
        standin_name = CLEANROOM_STANDIN_FEATURES
        print(f"  特徴量: スタンドイン {standin_name} を使用(配線検証が目的で学習品質は見ない)")
        if not standin_src.exists():
            raise SystemExit(f"クリーンルーム検証: スタンドイン特徴量が無い: {standin_src}")
        shutil.copyfile(standin_src, cr_policy_net / standin_name)

        def run_train(out_weights_name: str, extra: list[str]) -> Path:
            cmd = [
                sys.executable, "-u", "train.py",
                "--features", standin_name,
                "--limit", "500",
                "--max-epochs", "1",
                "--out-weights", out_weights_name,
                "--seed", "42",
            ] + extra
            print(f"\n  $ {' '.join(cmd)}   (cwd={cr_policy_net})")
            r = subprocess.run(cmd, cwd=cr_policy_net, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            tail = "\n".join((r.stdout or "").splitlines()[-25:])
            print(tail)
            if r.stderr:
                print("  --- stderr(末尾) ---")
                print("\n".join(r.stderr.splitlines()[-25:]))
            if r.returncode != 0:
                raise SystemExit(f"クリーンルーム検証: train.py({out_weights_name}) が失敗(exit {r.returncode})")
            out_path = cr_policy_net / out_weights_name
            if not out_path.exists():
                raise SystemExit(f"クリーンルーム検証: train.py が重みJSONを書き出さなかった: {out_path}")
            print(f"  train.py OK -> {out_path} ({out_path.stat().st_size/1e3:.1f} KB)")
            return out_path

        treatment_path = run_train("_cleanroom_treatment.json", treatment_extra)
        control_path = run_train("_cleanroom_control.json", control_extra)

        # eval_mirror_symmetric.py は重みパスを --model-a / --model-b で直接受け取るため、
        # 旧方式(--extra-registry で対照群を動的登録)は不要。ここでも絶対パスを渡す教訓を踏襲する。
        eval_cmd = [
            sys.executable, "-u", "eval_mirror_symmetric.py",
            "--model-a", str(treatment_path.resolve()),
            "--model-b", str(control_path.resolve()),
            "--archetype", learner_archetype,
            "--games", "4",
            "--workers", "1",  # ローカル(Windows)は WinError 5 でデッドロックするため必ず1
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
            raise SystemExit(f"クリーンルーム検証: eval_mirror_symmetric.py が失敗(exit {r2.returncode})")
        if "有効試合" not in r2.stdout or "A勝率" not in r2.stdout:
            raise SystemExit(
                "クリーンルーム検証: eval_mirror_symmetric.py の出力に想定した集計行"
                "(有効試合 / A勝率)が無い")

        print("\n  クリーンルーム検証: PASS(処置群学習 -> 対照群学習 -> 対称ミラー戦評価まで配線確認済み)")
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
                   control_features_filename: str, seed: int, learner_archetype: str,
                   treatment_extra: list[str], control_extra: list[str], tag: str,
                   games: int, eval_workers: int) -> dict:
    out_treatment_name = f"policy_weights_{tag}_treatment_s{seed}.json"
    out_control_name = f"policy_weights_{tag}_control_s{seed}.json"
    metrics_treatment_name = f"metrics_{tag}_treatment_s{seed}.json"
    metrics_control_name = f"metrics_{tag}_control_s{seed}.json"
    train_treatment_log = f"train_{tag}_treatment_s{seed}.log"
    train_control_log = f"train_{tag}_control_s{seed}.log"
    eval_log_name = f"eval_{tag}_s{seed}.log"
    same_features = control_features_filename == features_filename

    cell_intro = _md_cell(
        f"# BC A/B評価: 処置群 vs 対照群(tag={tag}, seed={seed})— Kaggle 実行\n\n"
        f"design-transformer-representation-2026-08-08.md §8.1 の評価手順。\n\n"
        f"1. `train.py` で処置群を学習(`{features_filename}`"
        f"{' + ' + ' '.join(treatment_extra) if treatment_extra else '（追加引数なし）'}）\n"
        f"2. `train.py` で対照群を学習(`{control_features_filename}`"
        f"{' + ' + ' '.join(control_extra) if control_extra else '（追加引数なし）'}）\n"
        "3. `eval_mirror_symmetric.py` で処置群 vs 対照群のミラー戦を評価する"
        "(両陣営とも greedy な `PolicyModel.select_option()`/`score_options()` に統一した対称評価。"
        "POOL8は251次元で使えないため、既存の対戦相手プールではなく直接対戦させる。"
        "旧 `eval_diagnostics.py` 経由のミラー戦は learner役=温度サンプリング/opponent役=greedy"
        "という役割非対称バグがあり、処置群を常にlearner役に固定していたため役割バイアスだけで"
        "Elo -56相当が混入していた[design書§8.3]）\n\n"
        "**非公開 Notebook・非公開 Dataset x2。GPU 不要・インターネット不要。**\n\n"
        f"入力: Dataset `{code_dataset_slug}`(コード一式)・`{features_dataset_slug}`"
        f"(`{features_filename}`" + ("" if same_features else f" と `{control_features_filename}`") + ")\n"
        "出力: `/kaggle/working/` 直下に処置群/対照群の重み・offline指標・3ステップの標準出力ログ\n\n"
        "Save & Run All にすれば、ブラウザを閉じても裏で最後まで走る。"
    )

    features_names_literal = repr(sorted({features_filename, control_features_filename}))

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
        "# どちらの形でも動くようにする。",
        "zips = glob.glob('/kaggle/input/**/ptcg_bc_ab_bundle.zip', recursive=True)",
        "if zips:",
        "    print('コード: zip から展開:', zips[0])",
        "    zipfile.ZipFile(zips[0]).extractall(REPO)",
        "else:",
        "    marks = glob.glob('/kaggle/input/**/sample_submission/cg/libcg.so', recursive=True)",
        "    if not marks:",
        "        raise SystemExit('ptcg_bc_ab_bundle.zip も展開済みツリーも /kaggle/input 以下に無い。'",
        "                         'コード Dataset が Notebook に添付され、処理が完了しているか確認する。')",
        "    src = os.path.dirname(os.path.dirname(os.path.dirname(marks[0])))",
        "    print('コード: 展開済みツリーから複製:', src)",
        "    shutil.copytree(src, REPO, dirs_exist_ok=True)",
        "    os.chmod(os.path.join(REPO, 'sample_submission', 'cg', 'libcg.so'), 0o755)",
        "",
        f"FEATURES_NAMES = {features_names_literal}",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "for fname in FEATURES_NAMES:",
        "    hits = glob.glob(f'/kaggle/input/**/{fname}', recursive=True)",
        "    if not hits:",
        "        raise SystemExit(f'特徴量 Dataset に {fname} が見つからない。'",
        "                         '特徴量 Dataset が Notebook に添付されているか確認する。')",
        "    dst = os.path.join(policy_net_dir, fname)",
        "    print('特徴量:', hits[0], '->', dst)",
        "    shutil.copyfile(hits[0], dst)",
        "",
        "for p in (REPO, os.path.join(REPO, 'sample_submission'), os.path.join(REPO, 'league'),",
        "          os.path.join(REPO, 'kaggle_replays', 'rl')):",
        "    if p not in sys.path:",
        "        sys.path.insert(0, p)",
        "",
        "need = ['sample_submission/cg/libcg.so',",
        "        'sample_submission/ptcg_ai/learning/policy_model.py',",
        "        'league/run_league.py',",
        "        'kaggle_replays/policy_net/train.py',",
        "        'kaggle_replays/rl/eval_mirror_symmetric.py',",
        f"        'kaggle_replays/meta_analysis/archetype_decks/{learner_archetype}/01.csv']",
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
        f"LEARNER_ARCHETYPE = {learner_archetype!r}",
        f"FEATURES_NAME = {features_filename!r}",
        f"CONTROL_FEATURES_NAME = {control_features_filename!r}",
        f"TREATMENT_EXTRA = {treatment_extra!r}",
        f"CONTROL_EXTRA = {control_extra!r}",
        f"GAMES = {games}",
        f"EVAL_WORKERS = {eval_workers}",
        f"OUT_TREATMENT_NAME = {out_treatment_name!r}",
        f"OUT_CONTROL_NAME = {out_control_name!r}",
        f"METRICS_TREATMENT_NAME = {metrics_treatment_name!r}",
        f"METRICS_CONTROL_NAME = {metrics_control_name!r}",
        f"TRAIN_TREATMENT_LOG = {train_treatment_log!r}",
        f"TRAIN_CONTROL_LOG = {train_control_log!r}",
        f"EVAL_LOG_NAME = {eval_log_name!r}",
        "",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "rl_dir = os.path.join(REPO, 'kaggle_replays', 'rl')",
        "",
        "def run_streamed(cmd, cwd, log_path):",
        "    # 標準出力を画面(セル出力)と /kaggle/working のログファイル両方に残す。",
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
        "print('=== 1/3: train.py (処置群) ===', flush=True)",
        "train_treatment_cmd = ([sys.executable, '-u', 'train.py',",
        "             '--features', FEATURES_NAME,",
        "             '--out-weights', OUT_TREATMENT_NAME,",
        "             '--seed', str(SEED),",
        "             '--metrics-out', METRICS_TREATMENT_NAME] + TREATMENT_EXTRA)",
        "train_treatment_log_path = os.path.join('/kaggle/working', TRAIN_TREATMENT_LOG)",
        "ret = run_streamed(train_treatment_cmd, policy_net_dir, train_treatment_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'train.py(処置群)が失敗(exit {ret})')",
        "",
        "print('\\n=== 2/3: train.py (対照群) ===', flush=True)",
        "train_control_cmd = ([sys.executable, '-u', 'train.py',",
        "             '--features', CONTROL_FEATURES_NAME,",
        "             '--out-weights', OUT_CONTROL_NAME,",
        "             '--seed', str(SEED),",
        "             '--metrics-out', METRICS_CONTROL_NAME] + CONTROL_EXTRA)",
        "train_control_log_path = os.path.join('/kaggle/working', TRAIN_CONTROL_LOG)",
        "ret = run_streamed(train_control_cmd, policy_net_dir, train_control_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'train.py(対照群)が失敗(exit {ret})')",
        "",
        "# eval_mirror_symmetric.py の --model-a / --model-b の重みパスは、相対パスだと",
        "# eval_mirror_symmetric.py 自身の実行時 cwd(rl_dir)基準で解決される。train.py の出力先",
        "# (policy_net_dir)とは基準が違うため、必ず絶対パスで渡す",
        "# (push_bc_train.py 以来の教訓: 相対パスだと壊れる)。",
        "treatment_weights_abs = os.path.join(policy_net_dir, OUT_TREATMENT_NAME)",
        "control_weights_abs = os.path.join(policy_net_dir, OUT_CONTROL_NAME)",
        "for p in (treatment_weights_abs, control_weights_abs):",
        "    if not os.path.exists(p):",
        "        raise SystemExit(f'train.py の出力が見つからない: {p}')",
        "",
        "print('\\n=== 3/3: eval_mirror_symmetric.py (処置群 vs 対照群 対称ミラー戦) ===', flush=True)",
        "# eval_diagnostics.py 経由(旧方式)は役割非対称バグがあったため使わない。",
        "# eval_mirror_symmetric.py は両陣営とも greedy に統一した対称評価で、重みパスを",
        "# --model-a / --model-b で直接受け取るため --extra-registry 用の registry JSON は不要。",
        "eval_cmd = [sys.executable, '-u', 'eval_mirror_symmetric.py',",
        "            '--model-a', treatment_weights_abs,",
        "            '--model-b', control_weights_abs,",
        "            '--archetype', LEARNER_ARCHETYPE,",
        "            '--games', str(GAMES),",
        "            '--workers', str(EVAL_WORKERS)]",
        "eval_log_path = os.path.join('/kaggle/working', EVAL_LOG_NAME)",
        "ret = run_streamed(eval_cmd, rl_dir, eval_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'eval_mirror_symmetric.py が失敗(exit {ret})')",
        "",
        "print(f'\\n[全体完了] {time.time()-t_run0:.0f}s elapsed', flush=True)",
    ])

    cell_collect = _code_cell([
        "import os, shutil",
        "",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "candidates = [",
        "    os.path.join(policy_net_dir, OUT_TREATMENT_NAME),",
        "    os.path.join(policy_net_dir, OUT_CONTROL_NAME),",
        "    os.path.join(policy_net_dir, METRICS_TREATMENT_NAME),",
        "    os.path.join(policy_net_dir, METRICS_CONTROL_NAME),",
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
    """Dataset を新規作成 or 新バージョンとして更新し、新版が実際に反映されるまで待つ。"""
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
                    help=f"カンマ区切りのシード。既定 {DEFAULT_SEEDS}。シードごとに別 Notebook を作る"
                         "(design書 §4 原則7)。")
    ap.add_argument("--learner-archetype", default=DEFAULT_LEARNER_ARCHETYPE,
                    help="pools.LEARNER_REGISTRY のキー(デッキCSV解決用)。処置群・対照群とも"
                         "同じデッキで学習する想定。")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="実験タグ(t1/t2/negctl等)。出力ファイル名・kernel-slugに使う。")
    ap.add_argument("--features-filename", default=DEFAULT_FEATURES_FILENAME,
                    help="kaggle_replays/policy_net/ 配下の特徴量ファイル名(処置群用、"
                         "--control-features-filename 省略時は対照群にも使う)。")
    ap.add_argument("--control-features-filename", default=None,
                    help="対照群専用の特徴量ファイル名(省略時は --features-filename と同じものを使う。"
                         "Negative Control(--shuffle-card-ids)のように、処置群と対照群で"
                         "features.npz 自体が異なる場合に指定する)。")
    ap.add_argument("--treatment-args", default="",
                    help="train.py に渡す処置群固有の追加引数(1つの文字列、shlexで分解)。"
                         "例: '--use-board-set'。既定は空(追加引数なし=フル特徴量のまま)。")
    ap.add_argument("--control-args", default="",
                    help="train.py に渡す対照群固有の追加引数。例: "
                         "'--ablate-features self_board_card_slot,self_discard_card_slot,"
                         "opp_board_card_slot,opp_discard_card_slot'。既定は空。")
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES,
                    help=f"ミラー戦の試合数(既定 {DEFAULT_GAMES}、design書 §4原則7のδ_min=0.03基準)。")
    ap.add_argument("--eval-workers", type=int, default=DEFAULT_EVAL_WORKERS,
                    help="eval_mirror_symmetric.py の --workers(Kaggle 側、既定4コア。"
                         "ローカルWindowsのクリーンルーム検証では multiprocessing.Pool が"
                         "WinError 5 でデッドロックするため常に1を使う)")
    ap.add_argument("--code-dataset-slug", default=DEFAULT_CODE_DATASET_SLUG)
    ap.add_argument("--features-dataset-slug", default=DEFAULT_FEATURES_DATASET_SLUG)
    ap.add_argument("--kernel-slug-prefix", default=DEFAULT_KERNEL_SLUG_PREFIX,
                    help="Notebook slug は '<prefix>-<tag>-s<seed>' になる。")
    ap.add_argument("--username", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="バンドル・Notebook JSON・メタデータを作り、クリーンルーム検証まで行う。"
                         "Kaggle へは一切送らない。")
    ap.add_argument("--skip-cleanroom", action="store_true",
                    help="--dry-run のクリーンルーム検証をスキップする(デバッグ用)")
    ap.add_argument("--out-dir", default=None,
                    help="既定は distributed/.stage_bc_ab_<tag>")
    ap.add_argument("--max-bundle-mb", type=float, default=DEFAULT_MAX_BUNDLE_BYTES / 1e6,
                    help=f"コードバンドルの展開後合計サイズの上限(MB、既定"
                         f"{DEFAULT_MAX_BUNDLE_BYTES/1e6:.0f})。超えたら即中断する暴走防止ガード。")
    ap.add_argument("--dataset-wait", type=int, default=600,
                    help="Dataset の新バージョンが反映されるまでの待ち時間(秒)。")
    ap.add_argument("--timeout", type=int, default=8 * 3600,
                    help="Notebook 完了待ちの上限(秒、既定8時間 = 学習x2 + 評価 + 余裕)")
    ap.add_argument("--no-wait", action="store_true", help="push だけして待たない")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (HERE / f".stage_bc_ab_{args.tag}").resolve()
    code_data_dir = out_dir / "code_data"
    features_data_dir = out_dir / "features_data"

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds が空")

    treatment_extra = shlex.split(args.treatment_args) if args.treatment_args else []
    control_extra = shlex.split(args.control_args) if args.control_args else []
    control_features_filename = args.control_features_filename or args.features_filename

    print(f"tag: {args.tag}")
    print(f"学習アーキタイプ: {args.learner_archetype}")
    print(f"features(処置群): {args.features_filename}")
    print(f"features(対照群): {control_features_filename}"
          f"{'(処置群と共有)' if control_features_filename == args.features_filename else ''}")
    print(f"treatment_extra_args: {treatment_extra}")
    print(f"control_extra_args: {control_extra}")
    print(f"シード: {seeds}")
    print(f"games: {args.games}")

    # ------------------------------------------------------------ コード Dataset
    zip_path = code_data_dir / "ptcg_bc_ab_bundle.zip"
    max_bundle_bytes = int(args.max_bundle_mb * 1e6)
    n, total, zsize = build_bundle_zip(zip_path, out_dir, max_bundle_bytes)
    print(f"\nコードバンドル: {n} ファイル, 展開後 {total/1e6:.1f} MB -> zip {zsize/1e6:.1f} MB ({zip_path})")

    user = detect_username(args.username)

    code_dataset_meta = {
        "title": "ptcg bc ab eval",
        "id": f"{user}/{args.code_dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (code_data_dir / "dataset-metadata.json").write_text(
        json.dumps(code_dataset_meta, indent=2), encoding="utf-8")
    print(f"code dataset-metadata: {code_data_dir / 'dataset-metadata.json'}")

    # ------------------------------------------------------------ 特徴量 Dataset
    features_data_dir.mkdir(parents=True, exist_ok=True)
    features_needed = sorted({args.features_filename, control_features_filename})
    missing_features = []
    print("\n特徴量:")
    for fname in features_needed:
        src = POLICY_NET_DIR / fname
        if src.exists():
            dst = features_data_dir / fname
            if dst.exists():
                dst.unlink()
            shutil.copyfile(src, dst)
            print(f"  OK  {fname}  ({src.stat().st_size/1e6:.1f} MB)")
        else:
            missing_features.append(fname)
            print(f"  [WARN] 未生成: {src}")
    if missing_features and not args.dry_run:
        raise SystemExit(f"特徴量ファイルが無いため push できない: {missing_features}\n"
                         "(先に build_features.py で生成すること)")

    features_dataset_meta = {
        "title": "ptcg bc ab features",
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
        kernel_slug = f"{args.kernel_slug_prefix}-{args.tag}-s{seed}"
        kernel_dir = out_dir / f"kernel_s{seed}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = f"{kernel_slug}.ipynb"
        notebook_doc = build_notebook(
            args.code_dataset_slug, args.features_dataset_slug, args.features_filename,
            control_features_filename, seed, args.learner_archetype,
            treatment_extra, control_extra, args.tag, args.games, args.eval_workers,
        )
        write_notebook_json(notebook_doc, kernel_dir / notebook_name)

        kernel_meta = {
            "id": f"{user}/{kernel_slug}",
            # title は固定にしないこと。Kaggle は Notebook を新規作成するとき id ではなく
            # title を slug 化して ref を決める(push_train_pool.py 系で実際に踏んだ罠)。
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
        kernel_dirs[seed] = kernel_dir
        kernel_slugs[seed] = kernel_slug

    if args.dry_run:
        if not args.skip_cleanroom:
            run_cleanroom_check(zip_path, args.learner_archetype, treatment_extra, control_extra)
        else:
            print("\n--skip-cleanroom のためクリーンルーム検証を省略した。")
        print("\n--dry-run のため Kaggle へは何も送らない。")
        return

    # ------------------------------------------------------------ ここから先は実送信
    print(f"\nコード Dataset を push: {user}/{args.code_dataset_slug}")
    push_dataset(code_data_dir, args.code_dataset_slug, user, args.dataset_wait)

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
            f"待ち時間の上限に達した(未完了 seed: {sorted(pending)})。"
            f"--no-wait で投げっぱなしにして後で回収する。")

    for seed in seeds:
        kernel_slug = kernel_slugs[seed]
        dest = out_dir / f"out_s{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{user}/{kernel_slug}", "-p", str(dest))
        print(f"seed={seed} 完了。回収先: {dest}")


if __name__ == "__main__":
    main()
