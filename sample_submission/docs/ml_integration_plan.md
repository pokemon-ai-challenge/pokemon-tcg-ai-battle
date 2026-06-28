# ML 統合 実装方針（現行コードからの移行プラン）

`docs/ml_architecture_design.md`（目標アーキテクチャの概念設計）を、**現行の `sample_submission` 実装を壊さずに**
段階導入するための作業用プランです。設計書が「何を作るか」なら、本書は「いまのコードから**どう移行するか**」を扱います。

- 作成日: 2026-06-28
- 対象ブランチ: `feature/router-selection-contexts`
- 関連: [`ml_architecture_design.md`](ml_architecture_design.md) / [`ai-development-roadmap.md`](ai-development-roadmap.md) / [`code-structure-map.md`](code-structure-map.md)

---

## 0. 前提（現状の正確な把握）

設計書は当初「Flat MCS + ランダムロールアウト」を既存実装と仮定していたが、**実コードにはそれが無い**。

| 項目 | 現状 |
|------|------|
| 行動選択 | `router.py → main_turn.py → priorities/*.py → choose_best_proposal`（**1 手読み**） |
| 探索（MCTS / Flat MCS） | **無し** |
| ロールアウト | **無し** |
| 全局面 `evaluate(state)->float` | **無し**（`evaluation/` は行動単位の特徴量ヘルパのみ） |
| 推定器（相手デッキ / 自山札） | **無し**（`cg.api.search_begin/step` は未使用） |
| デッキ知識 `meta_decks.py` | **整備済み**（採用率付き `MetaDeckEntry`、Tier1〜2 を網羅） |

→ **移行の基本原則:** 既存のルールベース proposal 方式は**消さずに温存**し、その上に推定器・評価関数・探索を
**追加**する。探索は当面フラグで ON/OFF できる形にし、既存挙動を回帰テストの基準に保つ。

---

## 1. 番号体系の統一（最初に合意すべきこと）

現在、3 つの別軸の番号が混在していて「Phase B」がどれを指すか曖昧。下表で対応を固定する。

| 本プラン | 設計書 (`ml_architecture_design.md`) | ロードマップ (`ai-development-roadmap.md`) | 進捗メモ（メモリ） |
|----------|----------|----------|----------|
| **Stage 1** 評価・推定基盤 | Phase A | Level 3（盤面評価）の一部 | Phase 5 系の延長 |
| **Stage 2** 探索（ISMCTS） | Phase B | Level 4（複数手探索） | 未着手 |
| **Stage 3** 学習（NN / 模倣学習） | Phase C | Level 5（自己改善） | 未着手 |

> 以降、本書では設計書に合わせて **Phase A / B / C** で呼ぶ。チーム内の口頭でも Phase A/B/C に統一する。

---

## 2. 移行の全体像

```
[現状] Observation → router → main_turn → proposals → choose_best_proposal → action
                                                  │
                          ┌───────────────────────┘ ここを壊さず据え置き
                          ▼
[Phase A] + 情報推論レイヤー（推定器, runtime_state で永続化）
          + 全局面 evaluate(state)（探索無しでも proposal の補助スコアに使える）
          + TimeManager
                          │
                          ▼
[Phase B] + search/ パッケージ（ISMCTS）
          main_turn に「探索ON→ISMCTS / OFF→従来proposal」のフラグ分岐を1点だけ追加
          ロールアウト方策＝既存 priorities を再利用
                          │
                          ▼
[Phase C] evaluate() / policy を NN に置換（推論のみCPUで軽量）
```

---

## 3. Phase A：壊さずに「追加」する（新規ファイル中心）

### 3-1. 追加ファイル一覧

| ファイル（新規） | 役割 | 既存資産の再利用 |
|------------------|------|------------------|
| `src/knowledge/opponent_estimator.py` | 相手デッキのベイズ推定 | `meta_decks.get_all_meta_decks()` の `key_cards` / `card_inclusion_rate` |
| `src/knowledge/self_state_estimator.py` | 残り山札・サイド推定（超幾何） | `deck.csv` レシピ + `obs.logs` 追跡 |
| `src/decision/evaluation/position_eval.py` | 全局面 `evaluate(state)->float` | `board_features.py`（`attacker_priority_score`, `is_likely_knocked_out_next_turn`） |
| `src/decision/time_manager.py` | 残り時間からターン予算を算出 | — |
| `src/decision/runtime_state.py` | 推定器をターン跨ぎで保持するシングルトン | — |

### 3-2. 関数シグネチャ案（既存と衝突しない形）

```python
# src/knowledge/opponent_estimator.py
class OpponentDeckEstimator:
    def __init__(self, meta: dict[str, MetaDeckEntry] | None = None) -> None: ...
    def update_from_logs(self, obs: Observation) -> None: ...   # 公開カードでベイズ更新
    def posterior(self) -> dict[str, float]: ...                # アーキタイプ事後分布
    def sample_opponent_deck(self, deck_count: int) -> list[int]: ...   # 決定化用（Phase B）
    def sample_opponent_hand(self, hand_count: int) -> list[int]: ...   # 決定化用（Phase B）
    def get_max_damage(self) -> int: ...                        # evaluate() 用
    def get_boss_probability(self) -> float: ...                # evaluate() 用

# src/knowledge/self_state_estimator.py
class SelfStateEstimator:
    def __init__(self, my_deck_recipe: list[int]) -> None: ...
    def update(self, obs: Observation) -> None: ...
    def probability_in_deck(self, card_id: int) -> float: ...
    def key_card_reachable(self, card_id: int, draw_count: int) -> float: ...
    def sample_side_cards(self) -> list[int]: ...               # 決定化用（Phase B）

# src/decision/evaluation/position_eval.py
def evaluate(state, my_idx: int, opp_est: OpponentDeckEstimator) -> float: ...
def evolution_readiness_score(player_state) -> float: ...

# src/decision/time_manager.py
class TimeManager:
    def budget_for_this_turn(self, turn: int, remaining_prizes: int) -> float: ...
    def should_stop(self, budget: float, turn_start: float) -> bool: ...

# src/decision/runtime_state.py
def get_runtime() -> "Runtime": ...   # モジュールレベル singleton を返す
class Runtime:
    opponent_estimator: OpponentDeckEstimator
    self_estimator: SelfStateEstimator | None
    time_manager: TimeManager
    def on_new_turn(self, obs: Observation) -> None: ...   # ターン頭で推定器を更新
```

### 3-3. 既存ファイルへの変更（最小限）

- `src/agent.py` または `src/decision/router.py`：ターン開始時に `get_runtime().on_new_turn(obs)` を 1 回呼ぶ。
  - 初回デッキ選択（`obs.select is None`）時に `SelfStateEstimator(deck)` を初期化。
- `src/decision/main_turn_parts/proposals.py` または `weights.py`（任意）：
  `evaluate()` を proposal の補助スコアに足したい場合のみ。**Phase A では足さなくてもよい**
  （evaluate() の単体テストを先に固め、proposal への反映は回帰を見ながら段階的に）。

> **重要（状態永続化）:** `agent()` は毎選択ごとにステートレスに呼ばれるが、Kaggle ではプロセスが
> 対戦中ずっと生存するため、`runtime_state.py` のモジュールレベル singleton で推定器の状態を保持できる。
> これにより既存の関数シグネチャ（`obs` だけ受け取る形）を一切変えずに済む。

### 3-4. Phase A の完了条件

- `evaluate(state)` が固定盤面で妥当な符号・大小関係を返す（単体テスト）。
- `OpponentDeckEstimator` が公開カード列から正しい事後分布に収束する（既知レシピで検証）。
- 既存対戦の挙動・勝率が**回帰しない**（推定器/評価は計算するが行動は従来通りでもよい）。

---

## 4. Phase B：ISMCTS 導入（唯一の破壊ポイント）

### 4-1. 追加ファイル一覧

```
src/decision/search/            ← 新規パッケージ
├─ __init__.py
├─ ismcts.py          探索本体（UCB1 選択・展開・バックプロパゲーション）
├─ determinize.py     OpponentDeckEstimator / SelfStateEstimator で非公開情報を埋める
├─ rollout.py         Heuristic Rollout（既存 priorities/*.py を呼ぶだけ）
└─ prune.py           B1 Heuristic top-k 事前枝刈り
```

### 4-2. 破壊ポイントは 1 箇所

- [`src/decision/main_turn.py`](../src/decision/main_turn.py) の `choose_best_proposal`（1 手読み）を、
  **フラグ分岐**で探索に切り替えられるようにする。
  ```python
  if USE_SEARCH and time_manager.has_budget():
      return run_ismcts(obs, runtime)     # 新規
  best = choose_best_proposal(proposals)  # 既存（フォールバック兼デフォルト）
  ```
- 既存の proposal 群は**削除しない**。理由: (1) フォールバック、(2) ロールアウト方策として再利用、
  (3) 探索の事前枝刈り（prior）として再利用。

### 4-3. 決定化（Determinization）で `search_begin` に渡すもの

`cg.api.search_begin`（[`cg/api.py`](../cg/api.py) 参照）は相手の非公開情報を**全部 Card ID で埋める**必要がある。
ここが Phase A の推定器を前提とする理由。

| 引数 | 埋め方 |
|------|--------|
| `your_deck` | `SelfStateEstimator` の残り山札（`obs.select.deck != None` の場合は不要） |
| `your_prize` | `SelfStateEstimator.sample_side_cards()` |
| `opponent_deck` | `OpponentDeckEstimator.sample_opponent_deck()`（最確アーキタイプから） |
| `opponent_prize` | 同上からサンプリング |
| `opponent_hand` | `OpponentDeckEstimator.sample_opponent_hand()` |
| `opponent_active` | 伏せポケモンがある場合のみ。最確の Basic を推定 |

> **要確認:** `search_step` が返すのは `SearchState`（`Observation.current` の `State` とは別型）。
> `evaluate()` をどちらの型でも動くように引数を抽象化するか、`SearchState→評価用ビュー`の薄い変換を 1 枚かませる。

### 4-4. Phase B の完了条件

- 探索 ON が探索 OFF（Phase A 版）に対し固定ベンチで明確に勝ち越す。
- `TimeManager` によりタイムアウトが発生しない（10 分制限内）。
- 決定化が「サイド落ち / 公開済みカード」を二重に使わない（整合性テスト）。

---

## 5. Phase C：学習（研究枠）

- `evaluate()` → Value Head、proposal/prune → Policy Head に置換。
- 教師データは ISMCTS の出力（模倣学習）。CPU 推論できる軽量 NN を目標。
- デッキ分類器（Transformer）で Tech カード対応。
- 詳細は設計書 §5-2 / §9 を参照。本リポジトリのコード構造には新規 `src/model/` を足す程度で、
  Phase A/B の境界（router / main_turn）はそのまま使える。

---

## 6. 実装順序と依存グラフ

```
meta_decks.py（既存）
      │
      ▼
OpponentDeckEstimator ─┐
SelfStateEstimator ────┤
TimeManager ───────────┤  ← Phase A（並行可・相互依存なし）
position_eval.evaluate ┘
      │（runtime_state で束ねる）
      ▼
ISMCTS（determinize は推定器に依存） ← Phase B
      │
      ▼
NN 置換 ← Phase C
```

- Phase A の 4 コンポーネントは相互依存が薄く、**並行作業しやすい**（担当分けに向く）。
- **Phase B は Phase A の推定器が前提**（決定化に必須）。順序を飛ばせない。

---

## 7. 提出・実行制約（忘れがちな注意）

- 提出アーカイブに **`src/` を必ず同梱**する（`main.py` が `src.agent` を import するため）。
  `build_submission.py` / `submission_manifest.txt` に新規ファイルが含まれるか確認。
- `cg/` は変更禁止。推定・探索はすべて `src/` 側に置く。
- Kaggle RAM 制約：`search_begin` の多重実行・状態の貯め込みに注意（`search_end` / `search_release` を確実に）。
- 有意差検証は最低 400〜1200 試合（設計書 §6）。`local_test.py` / `local_test_advanced.py` をベンチに使う。

---

## 8. API 整合チェックリスト（実装前に潰す）

設計書の疑似コードと実 API（`cg/api.py`）の差分。`evaluate()` 実装時に必ず確認。

- [ ] HP は `Pokemon.hp` / `Pokemon.maxHp`（`hp_remaining` は存在しない）
- [ ] 場のエネルギー数は `len(Pokemon.energies)`
- [ ] 進化段階は `Pokemon.preEvolution` または `CardData.stage1/stage2`
- [ ] `prize` は残りサイド → 取得済み = `6 - len(prize)`
- [ ] `PlayerState.hand` は相手側 `None`（自分のみ参照可）。`handCount` で枚数は取れる
- [ ] 評価対象の型：`State`（観測）と `SearchState`（探索）を統一 or 変換層を用意
- [ ] `active` は要素数 0 or 1 の `list[Pokemon | None]`（伏せ中 None）

---

## 9. 次アクション

1. 本書 §1 の番号体系をチームで合意（Phase A/B/C に統一）。
2. Phase A の 4 ファイルを担当割り当て（相互依存が薄いので並行可）。
3. `position_eval.evaluate()` の単体テストを `src/tests/` に追加（固定盤面で符号検証）。
4. Phase A 完了 → Phase B（ISMCTS）の `determinize.py` から着手。
