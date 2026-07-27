#!/usr/bin/env python3
"""強化学習を外部（Kaggle Notebooks 等）で回すための最小バンドルを作る。

## なぜ外部で回すのか

1世代（自己対戦1,100試合 + 評価1,200試合）に約1.5時間かかる。10世代なら15時間。
ローカルで回すとその間 PC を占有し、起動し続ける必要がある。

## なぜ小さく済むのか

**強化学習はリプレイ本体（1.3GB）を必要としない。** 自己対戦でデータを作るため、
必要なのはコードと学習済み重みだけ。合計8MB程度に収まる。

## 環境の前提（確認済み）

``sample_submission/cg/libcg.so`` は **x86-64 Linux の ELF 共有ライブラリ**であり、
Kaggle Notebooks / Colab / 一般的な Linux VM で動く見込み。
``cg/sim.py`` が OS を見て ``cg.dll`` と ``libcg.so`` を自動選択する。

ただし**実際に動くかは未検証**である。バンドルには動作確認用の
``smoke_test.py`` を同梱するので、本番実行の前に必ずそれを通すこと。

## 使い方

    python policy_prior/rl/package_for_remote.py --out rl_bundle.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_RL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RL_DIR.parents[2]

# バンドルに含めるもの。リポジトリ構造をそのまま保つ（パスがそれに依存しているため）
# `sample_submission/decks` は ptcg_ai から間接的に import される
# (profile_registry -> decks.active)。バンドルから漏らすと import 時点で落ちる。
# 実際にスモークテストで検出した欠落なので、依存を減らす方向で削らないこと。
_DIRS = [
    "sample_submission/cg",
    "sample_submission/ptcg_ai",
    "sample_submission/configs",
    "sample_submission/decks",
    "kaggle_replays/policy_prior/rl",
]
_FILES = [
    "sample_submission/main.py",
    "sample_submission/deck.csv",
    "sample_submission/tests/__init__.py",
    "sample_submission/tests/local_sim/__init__.py",
    "sample_submission/tests/local_sim/parallel_eval.py",
    "sample_submission/tests/local_sim/opponent_decks.json",
]
_GLOBS = [
    "kaggle_replays/policy_prior/output/policy_weights_archetype*.json",
]
_EXCLUDE_SUFFIX = {".pyc", ".pyo"}
_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache"}


README = """# RL リモート実行バンドル

強化学習の自己対戦をローカル以外（Kaggle Notebooks 等）で回すための最小構成。
**リプレイ本体（1.3GB）は含まれない。** 強化学習は自己対戦でデータを作るため不要。

## 手順

### 1. 動作確認（必ず最初に）

```
python smoke_test.py
```

ネイティブライブラリのロード・対局の完走・重みの読み込み・自己対戦の収集を確認する。
**1つでも落ちたら本番を回さないこと。**

### 2. 本番実行

```
cd kaggle_replays
python policy_prior/rl/generation.py \\
    --generations 10 \\
    --games-per-gen 1100 \\
    --eval-games 300 \\
    --selfplay-workers <smoke_test が出した推奨値> \\
    --eval-workers <同上> \\
    --out-dir policy_prior/output/rl
```

### 3. 結果の回収

`kaggle_replays/policy_prior/output/rl/` 配下:

| ファイル | 内容 |
|---|---|
| `history.json` | 世代ごとのプール勝率と採否 |
| `genNNN.json` | 各世代で採用された重み |
| `genNNN_candidate.json` | 採否に関わらず更新直後の重み |

**`history.json` が最重要。** 世代ごとにプール勝率が上がっているかを見る。

## 規模の目安（ローカル実測、6プロセス）

| | 実測 |
|---|---|
| 自己対戦 | 2.9秒/試合 |
| 評価 | 1.31秒/試合 |
| 1世代（1,100 + 1,200試合） | **約1.5時間** |

コア数が違えば比例して変わる。Kaggle Notebooks は約4コアなので、6コアのローカルより
遅くなる可能性が高い。**外部実行の利点は速度ではなく「PCを占有しないこと」である。**

## 注意

- 現在の基準値: 凍結プールに対する勝率 **51.0%（±2.8pt、1,200試合）**
- 評価の試合数が少ないと信頼区間が広がり判定できない。相手1体あたり300試合（合計1,200）を推奨
- 相手プールは凍結する。学習中に更新すると、勝率が上がったのか相手が弱くなったのか区別できない
"""


def collect_paths() -> list[Path]:
    paths: list[Path] = []
    for d in _DIRS:
        base = _REPO_ROOT / d
        if not base.is_dir():
            raise SystemExit(f"見つかりません: {base}")
        for p in base.rglob("*"):
            if p.is_file() and p.suffix not in _EXCLUDE_SUFFIX \
                    and not any(part in _EXCLUDE_DIRS for part in p.parts):
                paths.append(p)
    for f in _FILES:
        p = _REPO_ROOT / f
        if p.is_file():
            paths.append(p)
        else:
            print(f"  ! 任意ファイルが無いのでスキップ: {f}")
    for g in _GLOBS:
        found = sorted(_REPO_ROOT.glob(g))
        if not found:
            print(f"  ! 一致なし: {g}")
            print("    アーキタイプ専用モデルが無いと相手プールがルールベースになります。")
            print("    再生成: python policy_prior/train.py --archetype archetypeN "
                  "--out policy_prior/output/policy_weights_archetypeN.json")
        paths.extend(found)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "rl_bundle.zip")
    args = ap.parse_args()

    paths = collect_paths()
    total = sum(p.stat().st_size for p in paths)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(_REPO_ROOT).as_posix())
        # スモークテストは実ファイルから入れる（文字列に埋め込むと多重エスケープで壊れる。
        # 実際に f-string を壊して SyntaxError を出したので戻さないこと）
        z.write(_RL_DIR / "remote_smoke_test.py", "smoke_test.py")
        z.writestr("README.md", README)

    print(f"\n{len(paths):,} ファイル / 元サイズ {total/1e6:.1f}MB")
    print(f"-> {args.out} ({args.out.stat().st_size/1e6:.1f}MB)")
    print("\n展開後、まず `python smoke_test.py` を通すこと。")


if __name__ == "__main__":
    main()
