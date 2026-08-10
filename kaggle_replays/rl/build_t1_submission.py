"""T1(Transformer)のKaggle提出バンドルを作る(main.py+deck.csv+cg/+ptcg_ai/、torch不要)。
既存の sample_submission/ (production MLPの提出)は一切変更しない。別ディレクトリへ
コピーして構築する。
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_SAMPLE = _ROOT / "sample_submission"
_ALAKAZAM_MORIOKA_DECK = (_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                         / "alakazam_morioka" / "01.csv")
# [2026-08-09] 既定デッキを一旦alakazam_morioka(T1/PPO checkpointの学習・評価に
# 使ったデッキ)に戻した。morioka_v2(実対戦リプレイ由来)への切り替えは
# kaggle_replays/meta_analysis/archetype_decks/morioka_v2/01.csv に保存済みで、
# `--deck`で指定すれば使える(memory: morioka-v2-default-deck.md参照)。
_DEFAULT_DECK = _ALAKAZAM_MORIOKA_DECK
_V40_TEACHER = (_ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v40.json")

_MAIN_PY = '''import os
import sys

# On Kaggle, main.py is executed via exec() from /kaggle_simulations/agent/
# (no __file__), so make sure the agent directory is importable.
for _path in ("/kaggle_simulations/agent", os.getcwd()):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from cg.api import Observation, to_observation_class
from ptcg_ai.t1_policy.t1_policy_agent import agent as _t1_agent

__all__ = ["agent"]


def agent(obs_dict: dict) -> list[int]:
    """T1(盤面Transformer)方策の提出エージェント(NumPy推論、torch不要)。"""
    obs: Observation = to_observation_class(obs_dict)
    return _t1_agent(obs)
'''


def _copy_tree(src: Path, dst: Path, exclude_dirnames=("__pycache__", "t1_weights")) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*exclude_dirnames))


def build(name: str, npz_path: Path, out_dir: Path, deck_path: Path = _DEFAULT_DECK) -> Path:
    bundle_dir = out_dir / name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    _copy_tree(_SAMPLE / "cg", bundle_dir / "cg")
    _copy_tree(_SAMPLE / "ptcg_ai", bundle_dir / "ptcg_ai")

    (bundle_dir / "main.py").write_text(_MAIN_PY, encoding="utf-8")
    shutil.copy(deck_path, bundle_dir / "deck.csv")
    # [バグ修正] t1_policy_agent.py の _get_deck() は _HERE(=ptcg_ai/t1_policy/自身の
    # ディレクトリ)からdeck.csvを探すため、バンドル直下だけでなくここにも置く必要がある
    # (これを忘れてKaggleの実提出でFileNotFoundError->Validation Episode failedになった)。
    shutil.copy(deck_path, bundle_dir / "ptcg_ai" / "t1_policy" / "deck.csv")
    (bundle_dir / "ptcg_ai" / "t1_policy" / "weights.npz").write_bytes(npz_path.read_bytes())
    shutil.copy(_V40_TEACHER, bundle_dir / "ptcg_ai" / "t1_policy" / "v40_teacher.json")

    tar_path = out_dir / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for item in bundle_dir.iterdir():
            tar.add(item, arcname=item.name)
    print(f"ビルド完了: {bundle_dir}")
    print(f"tar.gz: {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
    return bundle_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", default=str(_ROOT / "kaggle_replays" / "rl" / "submission_builds"))
    ap.add_argument("--deck", default=str(_DEFAULT_DECK))
    args = ap.parse_args()
    build(args.name, Path(args.npz), Path(args.out_dir), Path(args.deck))


if __name__ == "__main__":
    main()
