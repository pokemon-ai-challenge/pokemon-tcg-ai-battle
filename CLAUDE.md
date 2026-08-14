# CLAUDE.md — Pokemon TCG AI Battle

Kaggle コンペティション [Pokemon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) および [Challenge Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy) 向けの AI エージェント開発リポジトリ。

---

## フォルダ構成

```
pokemon-tcg-ai-battle/
├── CLAUDE.md                  # このファイル
├── data/                      # コンペ提供データ（変更不可）
├── sample_submission/         # 提出コードのベース
│   ├── main.py                # ← エージェント実装（ここを編集）
│   ├── deck.csv               # ← 使用デッキ（ここを編集）
│   ├── models/climb_lb823/    # ← LB 823.5 を出したモデルの凍結一式（下記）
│   └── cg/                    # ← ゲームエンジン（変更禁止）
└── cardlist_referenced/       # カードリスト参照ツール（提出と無関係）
```

---

## 最高スコアモデル = **climb**（Kaggle publicScore **823.5** / 提出 55186283）

現時点の自己ベスト構成。**「一番良かったモデル」を探しているならこれ。**

| 何を | どこに |
|---|---|
| 同定情報・構成・使用デッキ・実行方法 | [sample_submission/models/climb_lb823/README.md](sample_submission/models/climb_lb823/README.md) |
| 機械可読メタ（sha256・ハイパラ・提出時コードの全ハッシュ） | [sample_submission/models/climb_lb823/MANIFEST.json](sample_submission/models/climb_lb823/MANIFEST.json) |
| 方策の重み本体 | `sample_submission/ptcg_ai/learning/policy_weights_alakazam_rl_climb.json`（sha256 `395b0248…`） |
| アルゴリズム詳細 | [sample_submission/docs/models/climb-823-model.md](sample_submission/docs/models/climb-823-model.md) |

- 構成: **Plan A アラカザム(フーディン)デッキ × `abl_5_full`（確定リーサル探索 → PIMC 前読み → Policy fallback）× climb 方策（BC 模倣 → field self-play PPO）**。
- **注意**: 既定の `ptcg_ai/learning/policy_weights.json` は **BC 模倣重みであって climb ではない**（sha256 `735dd38a…`）。ローカル評価で climb を測るときは重みパスを明示すること。

### 挑戦中の候補（2026-08、LB未収束）

**「オーガポンのやつ」「ハンマー4枚のやつ」を探しているならここ** →
[sample_submission/models/gen2_candidates/README.md](sample_submission/models/gen2_candidates/README.md)
（機械可読版 [MANIFEST.json](sample_submission/models/gen2_candidates/MANIFEST.json)）

2026-08 取得データ(gen2)の模倣学習とアーキタイプ別上位デッキで climb の改善を試みた4候補。
`mixogerpon`(オーガポン, LB最高874.1) / `mixhammer`(フーディン ハンマー4枚, 最高839.0) /
`mixclimb` / `g2climb`。**どれも未収束のため champion は climb のまま**。

判明した最重要事実: **効いているのはデッキであって RL ではない**（3候補ともデッキ変更は有意、
リーグRLの上乗せは判定不能）。また **同一提出の LB スコアが1時間で30点以上動く**ので、
1回の読みで優劣を判断してはいけない（実測記録 `kaggle_replays/_lb_history.tsv`）。

---

## 絶対に守るルール

- **`cg/` フォルダは変更しない。** コンペ提供のゲームエンジン（`cg.dll` / `libcg.so`）を含む。編集すると提出時に動作しなくなる。
- **`data/` フォルダは変更しない。** コンペ提供データ。参照専用。
- **`cardlist_referenced/` の変更は提出に影響しない。** ローカルツールなので自由に触ってよい。

---

## エージェントのインターフェース

提出物は `sample_submission/main.py` の `agent()` 関数。シグネチャは固定。

```python
def agent(obs_dict: dict) -> list[int]:
```

### 戻り値のルール

| タイミング | `obs.select` | 返すもの |
|-----------|-------------|---------|
| 初回（デッキ選択） | `None` | 60 枚のカード ID リスト |
| 通常ターン | `SelectData` | 選択肢のインデックスリスト |

通常ターンの制約：
- 各要素は `0 以上 len(obs.select.option) 未満`
- リスト長は `obs.select.minCount 以上 obs.select.maxCount 以下`
- 重複禁止

---

## デッキルール

- 合計 60 枚
- 同名カードは最大 4 枚（ACE SPEC カードは 1 枚まで）
- 基本エネルギーは枚数制限なし
- 必ず Basic ポケモンが 1 枚以上必要
- 使用可能なカード ID は `data/EN_Card_Data.csv` / `data/JP_Card_Data.csv` を参照

---

## 主要クラス（`cg/api.py`）

### `Observation`
エージェントが受け取る全情報。

| フィールド | 型 | 内容 |
|-----------|-----|------|
| `select` | `SelectData \| None` | 選択情報（初回は None） |
| `logs` | `list[Log]` | 前回選択以降のイベント履歴 |
| `current` | `State \| None` | 現在の盤面状態 |

### `State`
盤面全体のスナップショット。

- `turn` — ターン数（1 = 先攻1ターン目、2 = 後攻1ターン目、...）
- `yourIndex` — 自分のプレイヤーインデックス（0 or 1）
- `players[0/1]` — 各プレイヤーの状態（`PlayerState`）
- `supporterPlayed` / `energyAttached` / `retreated` — 今ターンの制限フラグ

### `PlayerState`
- `active` — バトルポケモン（`list[Pokemon | None]`、伏せ中は None）
- `bench` — ベンチポケモン
- `hand` — 手札（自分のみ参照可能、相手は `None`）
- `deckCount` — デッキ残枚数
- `prize` — サイドカード

### `SelectData`
- `type` (`SelectType`) / `context` (`SelectContext`) — 何を選ばせているか
- `option` — 選択肢の配列（`list[Option]`）
- `minCount` / `maxCount` — 選択する個数の範囲

### `CardData`（`all_card_data()` で取得）
カードの静的情報。`cardId` で `Observation` 内のカードと紐づける。

---

## サーチ API（MCTS などに使う）

`cg/api.py` にはゲームツリー探索用の関数が含まれる。

```python
from cg.api import search_begin, search_step, search_end, search_release

# 探索開始（エージェントの obs をそのまま渡す）
state = search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)

# 1手進める
next_state = search_step(state.searchId, [選択インデックス, ...])

# 探索終了（メモリ解放）
search_end()
```

`search_begin` には相手の非公開情報（デッキ・手札・伏せポケモン）を予測して渡す必要がある。

---

## 開発時の注意

- Python 3.12+ を想定（型ヒントに `list[X]` / `X | Y` 構文を使用）
- `cg/` は `ctypes` 経由でネイティブライブラリを呼ぶ。Windows では `cg.dll`、Linux では `libcg.so` を自動選択
- Kaggle 提出時のパスは `/kaggle_simulations/agent/` 以下になる（`main.py` の `read_deck_csv()` 参照）
- `Enum` クラス（`SelectContext` 等）はコンペ期間中に要素が追加される可能性がある（`api.py` のコメント参照）
