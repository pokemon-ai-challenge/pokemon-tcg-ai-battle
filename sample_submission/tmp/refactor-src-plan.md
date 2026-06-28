# `sample_submission/src` 安全リファクタリング計画

このドキュメントは AI が**自動で順番に実行**できるように設計したチェックリスト形式の作業手順です。
各フェーズは独立して安全に適用でき、**フェーズごとにテストで検証 → コミット**します。

- 対象: `sample_submission/src/` 配下のファイル/フォルダ整理
- 種別: **純粋なファイル移動 + import パス書き換えのみ**。ロジック・スコアリング・挙動は一切変更しない。
- 厳守ルール（CLAUDE.md）: `cg/` と `data/` は触らない。今回の作業は `src/` 内のみ。

---

## 0. 前提・安全装置（最初に1回だけ）

### 0-1. ベースライン記録
このリファクタ開始時点のテスト結果は次の通り（**これが合格ラインの基準**）:

```
62 passed, 1 failed
```

- 唯一の失敗 `src/tests/test_ability_priority.py::test_offers_ability_when_real_deck_ability_option_exists`
  は `obs=None` を渡している既存テスト側の問題（`AttributeError: 'NoneType' object has no attribute 'current'`）で、
  **構成とは無関係の既知失敗**。リファクタで直す対象ではない。
- 各フェーズ後のゲート: **「`62 passed, 1 failed` のまま、かつ失敗が上記の同一テストだけ」**であること。
  新たな失敗・収集エラー(ImportError 等)が1件でも出たら、そのフェーズは失敗。直すかロールバックする。

テスト実行コマンド（作業ディレクトリは `sample_submission/`）:

```bash
cd sample_submission
python -m pytest src/tests -q
```

### 0-2. チェック
- [ ] `git status` がクリーン（または現在の作業が commit 済み）であることを確認した
- [ ] 上記コマンドでベースライン `62 passed, 1 failed` を再現できた
- [ ] 作業ブランチにいる（例: `feature/router-selection-contexts`）。不安なら専用ブランチを切る

### 0-3. 共通の書き換えルール（全フェーズ共通・重要）
ファイルを移動/改名したら、**そのモジュールを参照している全ファイルの import 行**を新パスへ書き換える。
書き換えは次の2パターンに分かれる:

1. **`from src.decision.X import name`** 形式（関数/クラスを直接 import）
   → モジュールパス部分だけ新パスに置換。例:
   `from src.decision.card_move_common import analyze_card_move_option`
   → `from src.decision.card_move.common import analyze_card_move_option`

2. **`from src.decision import X`** 形式（モジュールを名前空間として import し、後で `X.func()` と使う）
   → **エイリアスを付けて呼び出し側を変えない**。例:
   `from src.decision import card_move_common`（本文中で `card_move_common.foo()`）
   → `from src.decision.card_move import common as card_move_common`

   ※ このエイリアス方式により、**関数本体の呼び出し箇所は一切変更不要**になる。これが安全性の肝。

`__pycache__` は無視してよい（ビルド/テストで再生成される）。

---

## 現状の問題（diagnosis）

`src/decision/` がフラットに約30ファイル並び、役割の異なるものが混在している:

- **ルーター**: `router.py`
- **コンテキスト別ハンドラ (`*_turn.py`)**: `main_turn`, `attack_turn`, `setup_turn`, `switch_turn`,
  `evolution_turn`, `energy_tool_turn`, `effect_choice_turn`, `damage_target_turn`, `count_turn`,
  `special_condition_turn`, `yes_no_turn`, `card_move_turn`
- **card_move ファミリ（巨大・散乱）**: `card_move_common`, `card_move_bench_field`, `card_move_discard`,
  `card_move_hand_like`, `card_move_hidden_zone`, `card_move_not_move_or_look`, `card_move_to_deck`,
  `card_move_to_hand_eval`
- **共有 eval / leaf**: `switch_eval`, `fallback`
- **既に整理済みのサブパッケージ**: `evaluation/`, `main_turn_parts/`

→ 「ハンドラ」「card_move 分割」「共有 eval」が同じ階層に平積みなのが散らかりの主因。
サブパッケージ化して責務ごとに分ければ見通しが良くなる。

---

## 目標構成（target）

```
src/decision/
├── __init__.py
├── router.py                  # 据え置き（ディスパッチ入口）
├── fallback.py                # 据え置き（共有 leaf）
├── handlers/                  # ★新規: SelectContext 別の入口（*_turn）
│   ├── __init__.py
│   ├── main_turn.py
│   ├── attack_turn.py
│   ├── setup_turn.py
│   ├── switch_turn.py
│   ├── evolution_turn.py
│   ├── energy_tool_turn.py
│   ├── effect_choice_turn.py
│   ├── damage_target_turn.py
│   ├── count_turn.py
│   ├── special_condition_turn.py
│   ├── yes_no_turn.py
│   └── card_move_turn.py
├── card_move/                 # ★新規: card_move_* ファミリ（接頭辞を落とす）
│   ├── __init__.py
│   ├── common.py
│   ├── bench_field.py
│   ├── discard.py
│   ├── hand_like.py
│   ├── hidden_zone.py
│   ├── not_move_or_look.py
│   ├── to_deck.py
│   └── to_hand_eval.py
├── evaluation/               # 既存（+ フェーズ3で switch_eval を受け入れ）
│   ├── attack_features.py
│   ├── board_features.py
│   ├── energy_requirements.py
│   ├── attack_profiles.py
│   └── switch_eval.py        # ★フェーズ3で移動
└── main_turn_parts/          # 既存（据え置き。整理は任意フェーズ4）
```

`knowledge/` と `tests/` の配置は変えない（テストは import パスのみ追従修正）。

---

## フェーズ1: `card_move/` パッケージ化（移動 + 改名）

最もファイル数が多く塊として独立しているので最初に切り出す。

### 1-1. パッケージ作成と移動
- [ ] `src/decision/card_move/__init__.py` を空ファイルで作成
- [ ] 次の8ファイルを `git mv` で移動＆改名（接頭辞 `card_move_` を削除）:

| 移動元 | 移動先 |
| --- | --- |
| `decision/card_move_common.py` | `decision/card_move/common.py` |
| `decision/card_move_bench_field.py` | `decision/card_move/bench_field.py` |
| `decision/card_move_discard.py` | `decision/card_move/discard.py` |
| `decision/card_move_hand_like.py` | `decision/card_move/hand_like.py` |
| `decision/card_move_hidden_zone.py` | `decision/card_move/hidden_zone.py` |
| `decision/card_move_not_move_or_look.py` | `decision/card_move/not_move_or_look.py` |
| `decision/card_move_to_deck.py` | `decision/card_move/to_deck.py` |
| `decision/card_move_to_hand_eval.py` | `decision/card_move/to_hand_eval.py` |

`git mv` 例:
```bash
cd sample_submission/src/decision
git mv card_move_common.py        card_move/common.py
git mv card_move_bench_field.py   card_move/bench_field.py
git mv card_move_discard.py       card_move/discard.py
git mv card_move_hand_like.py     card_move/hand_like.py
git mv card_move_hidden_zone.py   card_move/hidden_zone.py
git mv card_move_not_move_or_look.py card_move/not_move_or_look.py
git mv card_move_to_deck.py       card_move/to_deck.py
git mv card_move_to_hand_eval.py  card_move/to_hand_eval.py
```
（`git mv` で `card_move/` が存在しない場合、先に `__init__.py` を作るか `mkdir card_move` する）

### 1-2. import 書き換え（モジュールパス対応表）
次の旧→新でプロジェクト全体（`src/` 配下）を置換する。

| 旧モジュールパス | 新モジュールパス |
| --- | --- |
| `src.decision.card_move_common` | `src.decision.card_move.common` |
| `src.decision.card_move_bench_field` | `src.decision.card_move.bench_field` |
| `src.decision.card_move_discard` | `src.decision.card_move.discard` |
| `src.decision.card_move_hand_like` | `src.decision.card_move.hand_like` |
| `src.decision.card_move_hidden_zone` | `src.decision.card_move.hidden_zone` |
| `src.decision.card_move_not_move_or_look` | `src.decision.card_move.not_move_or_look` |
| `src.decision.card_move_to_deck` | `src.decision.card_move.to_deck` |
| `src.decision.card_move_to_hand_eval` | `src.decision.card_move.to_hand_eval` |

書き換えが必要なファイルと注意点（grep で確認済みの参照箇所）:

- [ ] `decision/card_move/bench_field.py`
  - `from src.decision import card_move_common` → `from src.decision.card_move import common as card_move_common`（**エイリアス必須**: 本文で `card_move_common.X` を使用）
  - `from src.decision.card_move_common import (...)` → `from src.decision.card_move.common import (...)`
- [ ] `decision/card_move/discard.py`
  - `from src.decision.card_move_common import (...)` → `from src.decision.card_move.common import (...)`
- [ ] `decision/card_move/hand_like.py`
  - `from src.decision import card_move_common` → `from src.decision.card_move import common as card_move_common`（**エイリアス必須**）
  - `from src.decision.card_move_common import (...)` → `from src.decision.card_move.common import (...)`
  - `from src.decision.card_move_to_hand_eval import choose_to_hand_action` → `from src.decision.card_move.to_hand_eval import choose_to_hand_action`
- [ ] `decision/card_move/hidden_zone.py`
  - `from src.decision.card_move_common import analyze_card_move_option` → `from src.decision.card_move.common import analyze_card_move_option`
- [ ] `decision/card_move/to_deck.py`
  - `from src.decision.card_move_common import analyze_card_move_option` → `from src.decision.card_move.common import analyze_card_move_option`
- [ ] `decision/card_move_turn.py`（このフェーズではまだ `decision/` 直下。フェーズ2で移動）
  - 下記5行のモジュールパスを `card_move.*` に置換:
    - `from src.decision.card_move_bench_field import choose_bench_or_field_action` → `...card_move.bench_field...`
    - `from src.decision.card_move_discard import choose_discard_action as choose_discard_action_impl` → `...card_move.discard...`
    - `from src.decision.card_move_hand_like import choose_hand_like_action` → `...card_move.hand_like...`
    - `from src.decision.card_move_hidden_zone import choose_hidden_zone_action` → `...card_move.hidden_zone...`
    - `from src.decision.card_move_not_move_or_look import (...)` → `...card_move.not_move_or_look...`
- [ ] `tests/test_card_move_turn.py`
  - `from src.decision import card_move_bench_field` → `from src.decision.card_move import bench_field as card_move_bench_field`
  - `from src.decision import card_move_common` → `from src.decision.card_move import common as card_move_common`
  - `from src.decision import card_move_discard` → `from src.decision.card_move import discard as card_move_discard`
  - `from src.decision import card_move_hand_like` → `from src.decision.card_move import hand_like as card_move_hand_like`
  - `from src.decision import card_move_not_move_or_look` → `from src.decision.card_move import not_move_or_look as card_move_not_move_or_look`
  - `from src.decision import card_move_to_hand_eval` → `from src.decision.card_move import to_hand_eval as card_move_to_hand_eval`
  - `from src.decision import card_move_hidden_zone` → `from src.decision.card_move import hidden_zone as card_move_hidden_zone`
  - （`from src.decision import card_move_turn` はフェーズ2で対応。今は据え置きで動く）
  - ※ いずれも**エイリアスで旧名を維持** → テスト本文の `card_move_xxx.func()` は変更不要

### 1-3. 取りこぼし確認
- [ ] `src/` 全体を grep して、旧パス `src.decision.card_move_(common|bench_field|discard|hand_like|hidden_zone|not_move_or_look|to_deck|to_hand_eval)` が**0件**であること

```bash
cd sample_submission
grep -rn "src\.decision\.card_move_\(common\|bench_field\|discard\|hand_like\|hidden_zone\|not_move_or_look\|to_deck\|to_hand_eval\)" src --include='*.py'
```

### 1-4. ゲート
- [ ] `python -m pytest src/tests -q` が `62 passed, 1 failed`（既知の同一テストのみ失敗）
- [ ] `python -c "import sys; sys.path.insert(0,'.'); import src.decision.router"` がエラーなく通る（import 健全性）
- [ ] コミット: `refactor: group card_move modules into card_move package`

---

## フェーズ2: `handlers/` パッケージ化（移動のみ・改名なし）

`*_turn.py` 系の12ハンドラを `handlers/` へ集約。
これらは「下位（fallback / main_turn_parts / card_move / switch_eval）」を import するだけなので、
**移動してもハンドラ内部の import は変更不要**。書き換えが要るのは `router.py` とテストの「ハンドラを参照する側」だけ。

### 2-1. パッケージ作成と移動
- [ ] `src/decision/handlers/__init__.py` を空ファイルで作成
- [ ] 次の12ファイルを `git mv`（改名なし）:

```bash
cd sample_submission/src/decision
for f in main_turn attack_turn setup_turn switch_turn evolution_turn energy_tool_turn \
         effect_choice_turn damage_target_turn count_turn special_condition_turn \
         yes_no_turn card_move_turn; do
  git mv "$f.py" "handlers/$f.py"
done
```

### 2-2. import 書き換え
旧→新パス: `src.decision.<name>` → `src.decision.handlers.<name>`（上記12モジュール）。

- [ ] `decision/router.py`: 12個の `from src.decision.<x>_turn import ...` を `from src.decision.handlers.<x>_turn import ...` に置換
  （`main_turn`, `attack_turn`, `setup_turn`, `switch_turn`, `evolution_turn`, `energy_tool_turn`,
  `effect_choice_turn`, `damage_target_turn`, `count_turn`, `special_condition_turn`,
  `yes_no_turn`, `card_move_turn` の12行）
- [ ] `tests/test_attack_priority.py`: `from src.decision.attack_turn import choose_attack_action` → `from src.decision.handlers.attack_turn import choose_attack_action`
- [ ] `tests/test_main_turn_integration.py`: `from src.decision.main_turn import choose_main_action` → `from src.decision.handlers.main_turn import choose_main_action`
- [ ] `tests/test_card_move_turn.py`: `from src.decision import card_move_turn` → `from src.decision.handlers import card_move_turn`
- [ ] 移動した各ハンドラ内部の `from src.decision.fallback ...` / `from src.decision.switch_eval ...` /
  `from src.decision.main_turn_parts...` / `from src.decision.card_move...` は**そのままで正しい**（変更しない）

### 2-3. 取りこぼし確認
- [ ] grep で旧ハンドラパスが0件:
```bash
cd sample_submission
grep -rn "src\.decision\.\(main_turn\|attack_turn\|setup_turn\|switch_turn\|evolution_turn\|energy_tool_turn\|effect_choice_turn\|damage_target_turn\|count_turn\|special_condition_turn\|yes_no_turn\|card_move_turn\)\b" src --include='*.py' | grep -v "main_turn_parts" | grep -v "handlers\."
```
（`main_turn_parts` への参照はフェーズ2対象外なので除外。ヒット0が目標）

### 2-4. ゲート
- [ ] `python -m pytest src/tests -q` が `62 passed, 1 failed`（既知の同一テストのみ）
- [ ] `python -c "import sys; sys.path.insert(0,'.'); import src.agent; import src.decision.router"` がエラーなく通る
- [ ] コミット: `refactor: move context handlers into handlers package`

---

## フェーズ3（任意）: `switch_eval.py` を `evaluation/` へ

`switch_eval.py` は純粋な評価モジュール（`attack_features`/`board_features`/`energy_requirements` と同列）。
`evaluation/` に置くのが自然。

### 3-1. 移動
- [ ] `git mv src/decision/switch_eval.py src/decision/evaluation/switch_eval.py`

### 3-2. import 書き換え（`src.decision.switch_eval` → `src.decision.evaluation.switch_eval`）
- [ ] `decision/handlers/switch_turn.py`: `from src.decision.switch_eval import choose_best_switch_option` → `...evaluation.switch_eval...`
- [ ] `decision/handlers/evolution_turn.py`: `from src.decision.switch_eval import resolve_option_target` → `...evaluation.switch_eval...`
- [ ] `decision/main_turn_parts/priorities/retreat.py`: `from src.decision.switch_eval import choose_best_retreat_option` → `...evaluation.switch_eval...`
- [ ] `tests/test_retreat_priority.py`: `from src.decision import switch_eval` → `from src.decision.evaluation import switch_eval`（エイリアス不要、名前一致）
- [ ] `tests/test_main_turn_integration.py`: `from src.decision import switch_eval` → `from src.decision.evaluation import switch_eval`
- [ ] `switch_eval.py` 自身の import（`evaluation.attack_features` 等）は相対的に近くなるが、現状の絶対 import のままで正しい（変更不要）

### 3-3. 取りこぼし確認
- [ ] `grep -rn "src\.decision\.switch_eval" sample_submission/src --include='*.py'` が0件

### 3-4. ゲート
- [ ] `python -m pytest src/tests -q` が `62 passed, 1 failed`
- [ ] コミット: `refactor: move switch_eval into evaluation package`

---

## フェーズ4（任意・ストレッチ）: `main_turn_parts` の整合

`main_turn.py`（フェーズ2で `handlers/` へ移動済み）と `main_turn_parts/` の関係を `card_move/` と同様に揃えたい場合のみ。
**内部 import 数が多く最もリスクが高い**ので、フェーズ1〜3が安定してから単独で行う。今は無理に着手しない。

候補案（どちらか。要判断なので AI は自動では実行せず、ユーザー確認を取る）:
- 案A: `main_turn_parts/` を `handlers/main_turn_parts/` へ移動し、`main_turn.py` と物理的に隣接させる
- 案B: 現状維持（既に内部は整理済みのため、無理に動かさない）

- [ ] （着手する場合）ユーザーに案A/Bを確認してから別途詳細手順を作成

---

## 完了後の最終確認

- [ ] `python -m pytest src/tests -q` → `62 passed, 1 failed`（既知失敗のみ）
- [ ] `cd sample_submission && python build_submission.py` が成功し、`submission.tar.gz` が生成される
      （`submission_manifest.txt` は `src` をディレクトリごと含むため、サブパッケージ追加は自動で取り込まれる。手当て不要）
- [ ] `router-evaluation-map.md` 内の「関連ファイル構成」ツリーを新パス（`handlers/`・`card_move/`・`evaluation/switch_eval.py`）に更新
- [ ] 最終的な `src/decision/` 直下が `router.py` / `fallback.py` / `__init__.py` と
      `handlers/` `card_move/` `evaluation/` `main_turn_parts/` のサブパッケージのみになっている

---

## ロールバック

各フェーズはコミット単位。問題が出たら:

```bash
git reset --hard HEAD      # 直近フェーズの未コミット変更を破棄
# もしくは
git revert <そのフェーズのコミット>
```

`git mv` を使っているので履歴は追跡可能。フェーズ単位で戻せる。

---

## メモ（AI 実行者向け）

- すべて**絶対 import**（`src.decision....`）で統一されているため、書き換えは機械的な文字列置換で済む。相対 import は導入しない（既存スタイルに合わせる）。
- `from src.decision import X`（名前空間 import）だけは**必ずエイリアス**で旧名を維持し、呼び出し側を触らない。
- ロジック・数値・関数シグネチャは**絶対に変更しない**。差分は import 行とファイル位置のみになるはず。
- 既知失敗テスト1件を「直そう」としないこと（スコープ外）。
</content>
</invoke>
