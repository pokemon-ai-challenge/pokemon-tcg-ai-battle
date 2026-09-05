"""251次元エンコーダ(state 166->251)対応の POOL8 BC 再学習を Kaggle Notebook 上で走らせる。

背景(2026-08-07): 状態特徴のエンコーダを 166 -> 251次元に拡張した(C2: 手札card_id +56 /
C7: 先攻後攻条件付け +9 / C3: 相手公開情報 +15 / C6: クロック +3 / C8: 山札速度 +2)。次元が
変わったため、既存の重み31個すべてが PolicyModel._load() の次元ガードで弾かれ、POOL8 の
相手プールが全部フォールバック状態になっている。本スクリプトは POOL8 の8アーキタイプそれぞれを
251次元特徴量で BC 再学習し、この状態を復旧する。

push_bc_train.py(打点計算の配線変更の効果測定用、1学習対象・複数シード・train+eval の
2ステップ)とは構成が異なる:
  - 本スクリプトは「8学習対象(アーキタイプ違い)・各1シード(42固定)」。
  - 測定目的ではなく次元変更で壊れた POOL8 の復旧が目的なので、**評価はしない**。
    Notebook は train.py の実行だけで終える(train.py -> eval_diagnostics.py の2段ではない)。
  - push_bc_train.py 自体は変更しない(RL側の push_train_pool.py と同様、複数の push
    スクリプトが共存する設計を踏襲する)。

各アーキタイプごとに Notebook を1本作る:
  python train.py --features features_<arch>_v251.npz \
      --out-weights policy_weights_<arch>_v251.json --seed 42 --metrics-out metrics_<arch>_v251.json

Dataset構成:
  - コード Dataset: 1つ(8 Notebook 共通)。push_bc_train.py の BUNDLE_ROOTS と除外ルールを
    そのまま踏襲する(git archive は使わない。未コミットのファイルを含める必要があるため)。
    POOL8対戦の相手重み(policy_weights*.json)は今回の学習では使わない(学習は教師データ
    からの模倣であり対戦相手は不要)ため、コード Dataset には一切含めない。
  - 特徴量 Dataset: 8つの .npz を1つの Dataset にまとめる(既定スラグ ptcg-bc-pool-v251-features)。
  - 重み Dataset: 無し。

Kaggle 側の同時実行上限は5(`Maximum batch CPU session count of 5 reached` を実際に踏んでいる)。
8本を一度に push すると6本目以降が失敗するため、--wave-size(既定5)ごとに波を分けて push し、
各波の完了を待ってから次の波を push する。push 関数(push_kernel_verified)は kaggle CLI の
戻り値だけでなく、push 直後の kernels status を短時間ポーリングして早期の error/cancel を検出する
(「push して実行開始」と表示だけして実際には失敗していた事故が別スクリプト系列で過去にあったため)。

    python push_bc_pool_v251.py --dry-run

--dry-run はバンドル・8つの Notebook・Dataset メタデータを作り、クリーンルーム検証(zip を
リポジトリ外に展開し、train.py --limit 500 --max-epochs 1 のような極小設定で実際に動くことを
確認する)まで行う。8アーキタイプのうち1つ(既定 crustle。POOL8の中では marnie に次ぐ規模だが
実データそのままで検証できるサイズ感)だけ検証すれば十分なため、他は省略する。Kaggle には
一切送らない。

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

DEFAULT_CODE_DATASET_SLUG = "ptcg-bc-pool-v251-code"
DEFAULT_FEATURES_DATASET_SLUG = "ptcg-bc-pool-v251-features"
DEFAULT_KERNEL_SLUG_PREFIX = "ptcg-bc-pool-v251"

# pools.POOL8 と同じ8アーキタイプ(順序も揃える)。動的に pools.py から読んでもよいが、
# 本スクリプトは「今日ビルド済みの8つの v251 npz」に固定して動くことを優先し、ここでは
# 明示リストにする(pools.POOL8 の並びが将来変わっても本スクリプトの対象は変わらないように)。
POOL8_ARCHETYPES = [
    "alakazam", "crustle", "marnie_grimmsnarl_ex", "rocket_mewtwo_ex",
    "omatsuri_ondo", "shirona_garchomp_ex", "ogerpon_teal_ex", "dragapult_ex",
]

DEFAULT_SEED = 42
DEFAULT_WAVE_SIZE = 5  # Kaggle の同時実行上限(実測)

# クリーンルーム検証で使う実アーキタイプ。8個全部やると冗長なので1つだけ。crustle は
# POOL8 の中で2番目に小さい実データ(features_crustle_v251.npz, 約10MB)で、スタンドインでは
# なく実データそのまま --limit 500 --max-epochs 1 で動かす(配線検証が目的)。
DEFAULT_CLEANROOM_ARCHETYPE = "crustle"

# バンドルに含めるトップレベルのディレクトリ(REPO_ROOT からの相対パス)。
# push_bc_train.py の BUNDLE_ROOTS をそのまま踏襲する。
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


# バンドルの展開後合計サイズの既定上限。push_bc_train.py と同じ考え方
# (経路ミスで巨大ディレクトリを巻き込んだ場合にディスクを埋め尽くす前に気付くためのハードガード)。
DEFAULT_MAX_BUNDLE_BYTES = 300 * 1024 * 1024  # 300MB


def iter_bundle_files(out_dir: Path):
    """(絶対パス, REPO_ROOT からの相対 arcname) を列挙する。

    out_dir: このディレクトリ配下にあるファイルは、BUNDLE_ROOTS のどこに位置しようとも
    常に除外する(ステージング出力先がバンドル対象の中に入っている場合の自己参照防止)。

    policy_weights*.json(LEARNING_DIR 直下)は本スクリプトでは**常に除外**する。今回の
    学習は教師データからの模倣であり対戦相手の重みを一切使わない(評価もしない)ため、
    push_bc_train.py と違って --weights による絞り込みが不要(何も含めない)。
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
                # ctl_eval/ と wired_eval/ は push_bc_train.py の過去の Kaggle 実行結果を
                # `kaggle kernels output -p <dest>` で回収した先(s<seed>/repo/... に展開済み
                # ツリーがまるごと入っている)。--dry-run で実際にバンドルしてみたところ、
                # 合計 1.4GB(807MB+608MB)あり、しかも repo/ の中に過去のバンドルzipの
                # 展開結果(kaggle_replays/policy_net/wired_eval/... を含む)が入れ子で入っていて
                # DEFAULT_MAX_BUNDLE_BYTES(300MB)ガードに即座に引っかかった。scale_* /
                # distributed/ と同様、学習の実行には不要な実行時生成物なので除外する。
                if rel_in_root.parts[0] in ("scale_data", "scale_features", "scale_logs", "scale_metrics",
                                            "distributed", "ctl_eval", "wired_eval"):
                    continue
                if rel_in_root.parts[0].startswith("scale_"):
                    continue
                if p.suffix == ".npz" or p.suffix == ".log":
                    continue
            # 重みファイルは常に除外(今回の学習では一切使わない)。push_bc_train.py の
            # 元の条件(p.parent == LEARNING_DIR)は LEARNING_DIR 直下しか見ないため、
            # bc_backup_*/ サブディレクトリや kaggle_replays/policy_net/ 直下の実験用重み
            # (policy_weights_configA.json 等)を取りこぼす。それらも含めて名前パターンだけで
            # 判定し、置き場所を問わず policy_weights*.json を全部除外する(実測で
            # 6MB 分の取りこぼしを確認したため、意図通り「重み一切なし」にするには
            # ディレクトリ限定を外す必要があった)。
            if p.name.startswith("policy_weights") and p.suffix == ".json":
                continue
            arc = rel.as_posix()
            if arc in seen:
                continue
            seen.add(arc)
            yield p, arc


def build_bundle_zip(dest: Path, out_dir: Path, max_bytes: int = DEFAULT_MAX_BUNDLE_BYTES) -> tuple[int, int, int]:
    """ptcg_bc_pool_v251_bundle.zip を作る。戻り値: (ファイル数, 展開後合計バイト数, zip サイズ)。

    累積の展開後サイズが max_bytes を超えた時点で即座に中断する(push_bc_train.py と同じ
    ハードガード。実際に BUNDLE_ROOTS の除外漏れでステージング先を自己参照し、zip が
    18GBまで膨らんでディスクを埋めかけた事故が過去にある)。
    """
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
def run_cleanroom_check(zip_path: Path, archetype: str) -> None:
    """zip をリポジトリ外の一時ディレクトリへ展開し、そこから train.py を極小設定で実際に
    実行して確認する。今回は評価をしないため eval_diagnostics.py の検証は不要(push_bc_train.py
    と違い train.py 単体の配線確認で足りる)。

    ``archetype`` の実データ(スタンドインではなく features_<archetype>_v251.npz そのもの)を
    --limit 500 --max-epochs 1 で回す。実データを使う理由: hand_card_vocab が npz ごとに
    埋め込まれており、これが正しく読めて meta に書き出されることも本検証で確認したいため
    (スタンドインの流用だと別アーキタイプの vocab を見てしまい確認にならない)。
    """
    print("\n=== クリーンルーム検証 ===")
    tmp = Path(tempfile.mkdtemp(prefix="ptcg_bc_pool_v251_cleanroom_"))
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
        need = [
            tmp / "sample_submission" / "cg" / "libcg.so",
            tmp / "sample_submission" / "ptcg_ai" / "learning" / "encoder.py",
            cr_policy_net / "train.py",
        ]
        missing = [str(p.relative_to(tmp)) for p in need if not p.exists()]
        print("  --- 必要ファイルの確認 ---")
        for p in need:
            print(f"    {'OK  ' if p.exists() else 'なし'} {p.relative_to(tmp)}")
        if missing:
            raise SystemExit(f"クリーンルーム検証: バンドルにファイルが足りない: {missing}")

        features_name = f"features_{archetype}_v251.npz"
        standin_src = POLICY_NET_DIR / features_name
        print(f"  特徴量: 実データ {features_name} を使用({archetype}, 配線+hand_card_vocab確認)")
        if not standin_src.exists():
            raise SystemExit(f"クリーンルーム検証: 特徴量が無い: {standin_src}")
        shutil.copyfile(standin_src, cr_policy_net / features_name)

        cr_weights_out = "_cleanroom_policy_weights.json"
        cr_metrics_out = "_cleanroom_metrics.json"
        # --limit は 500 だと test split が 0 件になりうる(split は行の並び順ではなく
        # md5(episode_id)%100 で決まるため、先頭500行が偏って1つも test に落ちない実測ケースが
        # crustle_v251 で実際に発生し、train.py の test 指標 print が
        # TypeError: unsupported format string passed to NoneType.__format__ で落ちた)。
        # --limit 1000 なら crustle_v251 で test=26件を確保できることを実測済み。
        train_cmd = [
            sys.executable, "-u", "train.py",
            "--features", features_name,
            "--limit", "1000",
            "--max-epochs", "1",
            "--out-weights", cr_weights_out,
            "--seed", str(DEFAULT_SEED),
            "--metrics-out", cr_metrics_out,
        ]
        print(f"\n  $ {' '.join(train_cmd)}   (cwd={cr_policy_net})")
        r1 = subprocess.run(train_cmd, cwd=cr_policy_net, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        tail1 = "\n".join((r1.stdout or "").splitlines()[-30:])
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

        # 次元(251)と hand_card_vocab が正しく書き出されたことを確認する(本タスクの目的そのもの)。
        weights_doc = json.loads(out_weights_path.read_text(encoding="utf-8"))
        state_dim = weights_doc["meta"]["state_feature_count"]
        vocab = weights_doc["meta"].get("hand_card_vocab", [])
        print(f"  meta.state_feature_count={state_dim} (期待 251)")
        print(f"  meta.hand_card_vocab={len(vocab)}種")
        if state_dim != 251:
            raise SystemExit(f"クリーンルーム検証: state_feature_count が251でない: {state_dim}")
        if not vocab:
            raise SystemExit("クリーンルーム検証: hand_card_vocab が空(--deck-csv 由来の語彙が書き出されていない)")

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


def build_notebook(code_dataset_slug: str, features_dataset_slug: str, archetype: str, seed: int) -> dict:
    features_filename = f"features_{archetype}_v251.npz"
    out_weights_name = f"policy_weights_{archetype}_v251.json"
    metrics_name = f"metrics_{archetype}_v251.json"
    train_log_name = f"train_{archetype}_v251.log"

    cell_intro = _md_cell(
        f"# BC 学習(POOL8 v251 復旧, {archetype}, seed={seed})— Kaggle 実行\n\n"
        f"`train.py` で `{features_filename}`(251次元エンコーダ)から模倣学習(BC)する。\n\n"
        "**評価はしない**(次元変更で壊れた POOL8 の復旧が目的で、A/B測定ではないため。"
        "学習1ステップのみで Notebook を終える)。\n\n"
        "**非公開 Notebook・非公開 Dataset x2。GPU 不要・インターネット不要。**\n\n"
        f"入力: Dataset `{code_dataset_slug}`(コード一式)・`{features_dataset_slug}`"
        f"(`{features_filename}` ほか POOL8 8アーキタイプ分)\n"
        "出力: `/kaggle/working/` 直下に重み・offline指標・標準出力ログ\n\n"
        "Save & Run All にすれば、ブラウザを閉じても裏で最後まで走る。"
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
        "# どちらの形でも動くようにする。",
        "zips = glob.glob('/kaggle/input/**/ptcg_bc_pool_v251_bundle.zip', recursive=True)",
        "if zips:",
        "    print('コード: zip から展開:', zips[0])",
        "    zipfile.ZipFile(zips[0]).extractall(REPO)",
        "else:",
        "    marks = glob.glob('/kaggle/input/**/sample_submission/cg/libcg.so', recursive=True)",
        "    if not marks:",
        "        raise SystemExit('ptcg_bc_pool_v251_bundle.zip も展開済みツリーも /kaggle/input 以下に無い。'",
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
        "for p in (REPO, os.path.join(REPO, 'sample_submission')):",
        "    if p not in sys.path:",
        "        sys.path.insert(0, p)",
        "",
        "need = ['sample_submission/cg/libcg.so',",
        "        'sample_submission/ptcg_ai/learning/encoder.py',",
        "        'kaggle_replays/policy_net/train.py']",
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
        f"ARCHETYPE = {archetype!r}",
        f"FEATURES_NAME = {features_filename!r}",
        f"OUT_WEIGHTS_NAME = {out_weights_name!r}",
        f"METRICS_NAME = {metrics_name!r}",
        f"TRAIN_LOG_NAME = {train_log_name!r}",
        "",
        "policy_net_dir = os.path.join(REPO, 'kaggle_replays', 'policy_net')",
        "",
        "def run_streamed(cmd, cwd, log_path):",
        "    # 標準出力を画面(セル出力)と /kaggle/working のログファイル両方に残す。",
        "    # train.py 自体はログファイルを書かないため、ここで tee する。",
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
        f"print('=== train.py (BC学習, {archetype}) ===', flush=True)",
        "train_cmd = [sys.executable, '-u', 'train.py',",
        "             '--features', FEATURES_NAME,",
        "             '--out-weights', OUT_WEIGHTS_NAME,",
        "             '--seed', str(SEED),",
        "             '--metrics-out', METRICS_NAME]",
        "train_log_path = os.path.join('/kaggle/working', TRAIN_LOG_NAME)",
        "ret = run_streamed(train_cmd, policy_net_dir, train_log_path)",
        "if ret != 0:",
        "    raise SystemExit(f'train.py が失敗(exit {ret})')",
        "",
        "print(f'\\n[完了] {time.time()-t_run0:.0f}s elapsed', flush=True)",
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


def kernel_slug_for(prefix: str, archetype: str) -> str:
    # Kaggle の kernel/dataset スラグは英数字とハイフンのみ許容(アンダースコア不可)。
    # アーキタイプ名はファイル名としてはアンダースコア入りだが、slug はハイフンに変える。
    return f"{prefix}-{archetype.replace('_', '-')}"


# ---------------------------------------------------------------- Dataset の push
def push_dataset(data_dir: Path, slug: str, user: str, dataset_wait: int) -> None:
    """Dataset を新規作成 or 新バージョンとして更新し、新版が実際に反映されるまで待つ。

    push_train_pool.py / push_bc_train.py で実際に踏んだ2つの罠への対策を踏襲する:
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


# ---------------------------------------------------------------- Notebook の push(波構成)
def push_kernel_verified(kernel_dir: Path, user: str, kernel_slug: str,
                         verify_wait: int = 60, verify_poll: int = 5) -> None:
    """kernels push し、戻り値を検証したうえで push 直後の status を短時間ポーリングし、
    早期の error/cancel を検出する。

    kaggle CLI の exit code は「push コマンド自体が受理されたか」しか保証しない。実際の
    実行開始や失敗(Kaggle 側の同時実行上限超過など)は少し遅れて `kernels status` に現れる。
    「push して実行開始」と表示するだけで実際には失敗していた事故が別スクリプト系列
    (push_train_pool.py 系)で過去にあったため、ここで明示的に確認する。
    """
    print(f"  push: {user}/{kernel_slug}")
    r = kaggle("kernels", "push", "-p", str(kernel_dir), check=False)
    if r.stdout:
        print(f"    {r.stdout.strip()}")
    if r.stderr:
        print(f"    [stderr] {r.stderr.strip()}")
    if r.returncode != 0:
        raise SystemExit(
            f"[PUSH FAILED] {kernel_slug}: kaggle kernels push が exit {r.returncode} で失敗\n"
            f"  stdout: {r.stdout}\n  stderr: {r.stderr}")

    t0 = time.time()
    last = ""
    while time.time() - t0 < verify_wait:
        r2 = kaggle("kernels", "status", f"{user}/{kernel_slug}", check=False)
        out = (r2.stdout or "").strip()
        if out != last:
            last = out
        low = out.lower()
        if "error" in low or "cancel" in low:
            raise SystemExit(
                f"[PUSH FAILED] {kernel_slug}: push直後に status がエラー: {out}\n"
                f"  同時実行上限({DEFAULT_WAVE_SIZE})超過の可能性が高い。--wave-size を確認する。\n"
                f"  https://www.kaggle.com/code/{user}/{kernel_slug}")
        if "running" in low or "queued" in low or "complete" in low:
            print(f"    確認OK: {out}")
            return
        time.sleep(verify_poll)
    print(f"    [WARN] {verify_wait}s 以内に running/queued を確認できなかった"
          f"(最後の status: {last!r})。手動確認推奨: https://www.kaggle.com/code/{user}/{kernel_slug}")


def wait_for_kernels(names: list[str], kernel_slugs: dict[str, str], user: str, timeout: int) -> None:
    """指定した archetype 群の Notebook が全て complete になるまで待つ。error/cancel は即座に失敗させる。"""
    t0 = time.time()
    last: dict[str, str] = {n: "" for n in names}
    pending = set(names)
    while pending and time.time() - t0 < timeout:
        for name in list(pending):
            kernel_slug = kernel_slugs[name]
            r = kaggle("kernels", "status", f"{user}/{kernel_slug}", check=False)
            out = (r.stdout or "").strip()
            if out != last[name]:
                print(f"  [{time.time()-t0:5.0f}s] {name}: {out}", flush=True)
                last[name] = out
            low = out.lower()
            if "complete" in low:
                pending.discard(name)
            elif "error" in low or "cancel" in low:
                raise SystemExit(
                    f"{name}: Kaggle 側で失敗: {out}\n"
                    f"  https://www.kaggle.com/code/{user}/{kernel_slug} でログを確認する。")
        if pending:
            time.sleep(20)
    if pending:
        raise SystemExit(
            f"待ち時間の上限に達した(未完了: {sorted(pending)})。"
            f"--no-wait で投げっぱなしにして後で回収する。")


def push_all_notebooks(archetypes: list[str], kernel_dirs: dict[str, Path], kernel_slugs: dict[str, str],
                       user: str, wave_size: int, timeout: int, no_wait: bool) -> None:
    """5本(既定)ずつ波に分けて push する。各波を push したら、次の波を push する前に
    その波の完了を待つ(同時実行上限対策)。最後の波は --no-wait なら待たない。
    """
    waves = [archetypes[i:i + wave_size] for i in range(0, len(archetypes), wave_size)]
    print(f"\n{len(archetypes)} 本の Notebook を {len(waves)} 波(wave-size={wave_size})で push する:")
    for wi, wave in enumerate(waves, 1):
        print(f"  wave {wi}: {wave}")

    for wi, wave in enumerate(waves, 1):
        print(f"\n=== wave {wi}/{len(waves)}: push {wave} ===")
        for name in wave:
            push_kernel_verified(kernel_dirs[name], user, kernel_slugs[name])

        is_last_wave = wi == len(waves)
        if not is_last_wave:
            print(f"\n  次の波を push する前に、この波({wave})の完了を待つ(同時実行上限{wave_size}対策)")
            wait_for_kernels(wave, kernel_slugs, user, timeout)
        elif not no_wait:
            wait_for_kernels(wave, kernel_slugs, user, timeout)
        else:
            print("\n  --no-wait のため最後の波は待たずに投げっぱなしにする。")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archetypes", default=",".join(POOL8_ARCHETYPES),
                    help=f"カンマ区切りのアーキタイプ名。既定は POOL8 全8種: {','.join(POOL8_ARCHETYPES)}")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"学習の乱数シード(既定 {DEFAULT_SEED}、8アーキタイプ全てに同一シードを使う。"
                         "次元変更の復旧が目的でありシード分散の測定はしないため)")
    ap.add_argument("--wave-size", type=int, default=DEFAULT_WAVE_SIZE,
                    help=f"同時に push する Notebook 数の上限(既定 {DEFAULT_WAVE_SIZE}。"
                         "Kaggle の同時実行上限に合わせる)")
    ap.add_argument("--code-dataset-slug", default=DEFAULT_CODE_DATASET_SLUG)
    ap.add_argument("--features-dataset-slug", default=DEFAULT_FEATURES_DATASET_SLUG)
    ap.add_argument("--kernel-slug-prefix", default=DEFAULT_KERNEL_SLUG_PREFIX,
                    help="Notebook slug は '<prefix>-<archetype(ハイフン区切り)>' になる。")
    ap.add_argument("--username", default=None)
    ap.add_argument("--cleanroom-archetype", default=DEFAULT_CLEANROOM_ARCHETYPE,
                    help=f"--dry-run のクリーンルーム検証で使う1アーキタイプ(既定 {DEFAULT_CLEANROOM_ARCHETYPE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="バンドル・Notebook JSON・メタデータを作り、クリーンルーム検証まで行う。"
                         "Kaggle へは一切送らない。")
    ap.add_argument("--skip-cleanroom", action="store_true",
                    help="--dry-run のクリーンルーム検証をスキップする(デバッグ用)")
    ap.add_argument("--out-dir", default=str(HERE / ".stage_bc_pool_v251"))
    ap.add_argument("--max-bundle-mb", type=float, default=DEFAULT_MAX_BUNDLE_BYTES / 1e6,
                    help=f"コードバンドルの展開後合計サイズの上限(MB、既定"
                         f"{DEFAULT_MAX_BUNDLE_BYTES/1e6:.0f})。超えたら即中断する暴走防止ガード。")
    ap.add_argument("--dataset-wait", type=int, default=900,
                    help="Dataset の新バージョンが反映されるまでの待ち時間(秒)。特徴量 Dataset は"
                         "8ファイル合計 約150MB あるため既定を長めにしている。")
    ap.add_argument("--timeout", type=int, default=6 * 3600,
                    help="各波の Notebook 完了待ちの上限(秒、既定6時間)")
    ap.add_argument("--no-wait", action="store_true",
                    help="最後の波を push した後は待たない(それより前の波は同時実行上限対策で"
                         "必ず待つ)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    code_data_dir = out_dir / "code_data"
    features_data_dir = out_dir / "features_data"

    archetypes = [a.strip() for a in args.archetypes.split(",") if a.strip()]
    if not archetypes:
        raise SystemExit("--archetypes が空")
    unknown = [a for a in archetypes if a not in POOL8_ARCHETYPES]
    if unknown:
        raise SystemExit(f"POOL8_ARCHETYPES に無いアーキタイプが指定された: {unknown}\n"
                         f"  利用可能: {POOL8_ARCHETYPES}")
    if args.cleanroom_archetype not in archetypes:
        print(f"[WARN] --cleanroom-archetype {args.cleanroom_archetype!r} は --archetypes に含まれていない"
              f"(検証自体は実行するが、push 対象には含まれない)")

    print(f"対象アーキタイプ({len(archetypes)}): {archetypes}")
    print(f"シード: {args.seed}(全アーキタイプ共通)")
    print(f"wave-size: {args.wave_size}")

    # ------------------------------------------------------------ コード Dataset
    zip_path = code_data_dir / "ptcg_bc_pool_v251_bundle.zip"
    max_bundle_bytes = int(args.max_bundle_mb * 1e6)
    n, total, zsize = build_bundle_zip(zip_path, out_dir, max_bundle_bytes)
    print(f"\nコードバンドル: {n} ファイル, 展開後 {total/1e6:.1f} MB -> zip {zsize/1e6:.1f} MB ({zip_path})")

    user = detect_username(args.username)

    code_dataset_meta = {
        "title": "ptcg bc pool v251 code",
        "id": f"{user}/{args.code_dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (code_data_dir / "dataset-metadata.json").write_text(
        json.dumps(code_dataset_meta, indent=2), encoding="utf-8")
    print(f"code dataset-metadata: {code_data_dir / 'dataset-metadata.json'}")

    # ------------------------------------------------------------ 特徴量 Dataset(8つ1本化)
    features_data_dir.mkdir(parents=True, exist_ok=True)
    missing_features = []
    print("\n特徴量(POOL8, v251):")
    for arch in POOL8_ARCHETYPES:  # 8つ全部を Dataset に含める(--archetypes で絞っても Dataset は共通)
        features_local_path = POLICY_NET_DIR / f"features_{arch}_v251.npz"
        if features_local_path.exists():
            dst = features_data_dir / features_local_path.name
            if dst.exists():
                dst.unlink()
            shutil.copyfile(features_local_path, dst)
            print(f"  OK  {features_local_path.name}  ({features_local_path.stat().st_size/1e6:.1f} MB)")
        else:
            missing_features.append(arch)
            print(f"  [WARN] 未生成: {features_local_path}")
    if missing_features and not args.dry_run:
        raise SystemExit(f"特徴量ファイルが無いため push できない: {missing_features}")

    features_dataset_meta = {
        "title": "ptcg bc pool v251 features",
        "id": f"{user}/{args.features_dataset_slug}",
        "licenses": [{"name": "unknown"}],
        "isPrivate": True,
    }
    (features_data_dir / "dataset-metadata.json").write_text(
        json.dumps(features_dataset_meta, indent=2), encoding="utf-8")
    print(f"features dataset-metadata: {features_data_dir / 'dataset-metadata.json'}")

    # ------------------------------------------------------------ Notebook(アーキタイプごと)
    kernel_dirs: dict[str, Path] = {}
    kernel_slugs: dict[str, str] = {}
    print("\n--- 生成した Notebook 一覧(学習対象 <-> 入力特徴量ファイルの対応) ---")
    print(f"  {'archetype':25s} {'kernel_slug':38s} {'features_filename'}")
    for arch in archetypes:
        kernel_slug = kernel_slug_for(args.kernel_slug_prefix, arch)
        kernel_dir = out_dir / f"kernel_{arch}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = f"{kernel_slug}.ipynb"
        notebook_doc = build_notebook(args.code_dataset_slug, args.features_dataset_slug, arch, args.seed)
        write_notebook_json(notebook_doc, kernel_dir / notebook_name)

        kernel_meta = {
            "id": f"{user}/{kernel_slug}",
            # title は固定にしないこと。Kaggle は Notebook を新規作成するとき id ではなく
            # title を slug 化して ref を決める(push_train_pool.py / push_bc_train.py で
            # 実際に踏んだ罠)。
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

        print(f"  {arch:25s} {kernel_slug:38s} features_{arch}_v251.npz")
        kernel_dirs[arch] = kernel_dir
        kernel_slugs[arch] = kernel_slug

    if args.dry_run:
        if not args.skip_cleanroom:
            run_cleanroom_check(zip_path, args.cleanroom_archetype)
        else:
            print("\n--skip-cleanroom のためクリーンルーム検証を省略した。")
        print("\n--dry-run のため Kaggle へは何も送らない。")
        return

    # ------------------------------------------------------------ ここから先は実送信
    print(f"\nコード Dataset を push: {user}/{args.code_dataset_slug}")
    push_dataset(code_data_dir, args.code_dataset_slug, user, args.dataset_wait)

    print(f"\n特徴量 Dataset を push: {user}/{args.features_dataset_slug}")
    push_dataset(features_data_dir, args.features_dataset_slug, user, args.dataset_wait)

    push_all_notebooks(archetypes, kernel_dirs, kernel_slugs, user, args.wave_size, args.timeout, args.no_wait)

    print("\n進行状況の確認:")
    for arch in archetypes:
        print(f"  {arch}: https://www.kaggle.com/code/{user}/{kernel_slugs[arch]}")

    if args.no_wait:
        return

    for arch in archetypes:
        kernel_slug = kernel_slugs[arch]
        dest = out_dir / f"out_{arch}"
        dest.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", f"{user}/{kernel_slug}", "-p", str(dest))
        print(f"{arch} 完了。回収先: {dest}")


if __name__ == "__main__":
    main()
