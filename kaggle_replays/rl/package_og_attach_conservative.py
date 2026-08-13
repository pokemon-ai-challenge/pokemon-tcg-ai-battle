"""og_attach_conservative(カプ・ブルル手貼り補正、保守設定)の提出tarballを作る。

過去の `_p1915_package.py` は climb の凍結tarballを土台に weights/deck だけを差し替える
ものだったが、それでは今回追加した ogerpon_planner.py の finish_only/attach 系コードも
configs/og_attach_conservative.json も含まれない(climbの凍結tarballはこのコード追加より
前の状態)。そのため本スクリプトは **現在の作業ツリー**(sample_submission/)から
cg/ptcg_ai/configs/decks/main.py をまるごとコピーして土台にする。

ml_config の選択は `PTCG_AI_ML_CONFIG` 環境変数(既定 "abl_5_full")で行われるが、
Kaggle実行環境ではこちらから環境変数を注入できない。そこで **tarball内の
configs/abl_5_full.json の中身だけを og_attach_conservative.json の中身に差し替える**
(weights/deck.csvの名前だけ差し替える既存の慣習と同じパターン。リポジトリ本体の
abl_5_full.json は一切変更しない)。
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sample_submission"
DECK_SRC = ROOT / "kaggle_replays/meta_analysis/archetype_decks_g2/ogerpon_teal_ex/01.csv"
WEIGHTS_SRC = SAMPLE / "ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"
CONFIG_SRC = SAMPLE / "configs/og_attach_conservative.json"
OUT_TAR = ROOT / "build_ready/submission_og_attach_conservative_20260813.tar.gz"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stage = ROOT / "build_ready" / "_stage_og_attach_conservative"
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    for name in ("cg", "ptcg_ai", "configs", "decks"):
        src = SAMPLE / name
        if src.exists():
            shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(SAMPLE / "main.py", stage / "main.py")
    shutil.copy(DECK_SRC, stage / "deck.csv")
    shutil.copy(WEIGHTS_SRC, stage / "ptcg_ai" / "learning" / "policy_weights.json")
    # ml_config の実体差し替え(env var注入ができないKaggle実行環境向け)。
    # og_attach_conservative.json の中身を、既定で読まれる abl_5_full.json の位置へ置く。
    shutil.copy(CONFIG_SRC, stage / "configs" / "abl_5_full.json")

    if OUT_TAR.exists():
        OUT_TAR.unlink()
    with tarfile.open(OUT_TAR, "w:gz") as tf:
        for name in ("main.py", "deck.csv", "cg", "configs", "decks", "ptcg_ai"):
            p = stage / name
            if p.exists():
                tf.add(p, arcname=name)
    shutil.rmtree(stage, ignore_errors=True)

    print("tarball:", OUT_TAR)
    print("size:", OUT_TAR.stat().st_size, "bytes")
    print("sha256:", sha256_file(OUT_TAR))
    print("deck src sha256:", sha256_file(DECK_SRC))
    print("weights src sha256:", sha256_file(WEIGHTS_SRC))
    print("config src (og_attach_conservative.json) sha256:", sha256_file(CONFIG_SRC))


if __name__ == "__main__":
    main()
