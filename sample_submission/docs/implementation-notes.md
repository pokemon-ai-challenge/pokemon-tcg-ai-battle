# 実装ノート: 進め方・分かったこと・課題・改善点

## 実装の進め方

### フェーズ 1: 既存コードの把握

`main_minimal_snapshot.py` に既存のルールベース実装があることを確認した。
主な問題:
- `choose_main_action` の `for` ループ内で早期 `return` → 分類が完了する前に返してしまう
- Enum を数値で比較している（`option.type == 7` など）
- MAIN 以外の SelectContext にほぼ対応していない

### フェーズ 2: API の理解

`cg/api.py` を精読し、以下を確認:

- `search_begin()`: 現在の `Observation` と相手の推測情報を渡してゲームツリー探索を開始
- `search_step(search_id, select)`: 1ステップ進める。戻り値は次の `SearchState`
- `search_release(search_id)`: 特定状態のメモリを解放
- `search_end()`: すべての探索メモリを解放

**重要な発見**: `search_begin()` は `agent_observation.search_begin_input` を使用する。
これは `obs_dict` から変換した `Observation` にのみ存在し、`search_step` の戻り値の `observation` には `None`。

### フェーズ 3: MC探索の実装

#### 設計判断

**なぜ Flat MC を選んだか:**

- Full MCTS（UCT木）は実装が複雑で、手番の切り替わりや相手のターンの扱いが難しい
- 本コンペでは各ターンに複数の選択フェーズがある（エネ付け先選択 → バトルポケモン選択 など）
- `search_step` は「1つの SelectData を解決する」粒度 → 木のノードが増えすぎる
- Flat MC は実装がシンプルで、ロールアウト数を増やすだけで強化できる

**時間予算:**

Kaggle 環境でのターン時間制限が不明なため 2.5秒 に設定した。
ローカルでの実測では 2秒 で 50〜200 ロールアウト程度（デッキ・盤面による）。

---

## 分かったこと

### SelectContext の多様性

`api.py` の `SelectContext` 定義を見ると、48 種類以上の選択コンテキストがある。
現状の実装では主要なもの（MAIN, SETUP_*）しかヒューリスティックを持っていない。

### search_begin の制約

```python
elif len(your_deck) < state.players[your_index].deckCount:
    raise ValueError("your_deck does not match the number of cards in your deck.")
```

`your_deck` の長さは少なくとも `deckCount` と一致する必要がある。
自分のデッキ（60枚）をそのまま渡しても数が合う（60 >= deckCount）。

### search_step でのメモリ管理

`search_step` を呼ぶたびに新しい `searchId` が生成される。
使い終わった状態は `search_release` で解放しないとメモリが枯渇する可能性がある。
ロールアウト終了後に全中間状態を解放するパターンで対処した。

### 相手情報予測の精度

相手のデッキ・手札を推測して渡しても、シミュレーション精度は大きく変わらない。
理由: ロールアウトがランダムであるため、相手の「最善手」は考慮されない。
重要なのは「おおよそ有効なカードリスト」を渡すことで、無効なカードIDエラーを防ぐこと。

---

## フェーズ 4: 特性（Ability）への対応

### 問題の発見

ユーザーから「この AI はポケモンの特性を使っているか？」という指摘を受けて確認したところ、
**特性をほとんど活用できていなかった**ことが判明した。

具体的な欠陥:
1. ヒューリスティックは `if abi: return [abi[0]]` で「特性があれば先頭を無条件発動」するだけ
2. **評価関数がダメカン蓄積を一切評価していない** → MC 探索が特性の効果を報酬として認識できない
3. `DAMAGE_COUNTER` / `DAMAGE_COUNTER_ANY`（ダメカン配置先選択）に専用処理がない
   → Munkidori / Chi-Yu の特性を発動しても配置先がランダムだった

これはドラパルト ex デッキにとって致命的。
このデッキの勝ち筋は「Dragapult ex の攻撃 + 特性によるダメカン蓄積でベンチも削る」ため、
特性が機能しないと本来の半分以下の制圧力しか出ない。

### 解決策

**① 評価関数にダメカン進捗を追加（最重要）**

```python
# 相手ポケモンへのダメカン蓄積を報酬化（比重 18%）
damage = (total_damage_progress(opp) - total_damage_progress(me)) * 0.18
```

`_damage_progress()` は「ポケモンがきぜつにどれだけ近いか × サイド価値」を返す。
ex ポケモン（サイド 2 枚）に乗ったダメカンを高く評価する。
これにより **MC 探索が「特性でダメカンを乗せた局面 = 高評価」を学習する**。

**② ダメカン配置先のターゲティング `_rank_damage_targets()`**

```text
優先順位:
  1. きぜつさせられる相手ポケモン（特に ex ポケモン → サイド 2 枚）
  2. すでにダメカンが乗った ex ポケモン
  3. HP が低い相手ポケモン
```

**③ MAIN フェーズで特性を攻撃より先に使う**

```python
if abi:
    return [_best_ability(obs, abi)]   # 攻撃より前に配置
```

ダメカンを先に乗せてから攻撃することで「攻撃で確定 KO」を狙う（ダメカン先積み戦術）。
特性の優先順位は Munkidori > Chi-Yu > Fezandipiti ex。

**④ コンテキスト網羅の拡充**

`DAMAGE_COUNTER`, `DAMAGE_COUNTER_ANY`, `DAMAGE_COUNTER_COUNT`, `EFFECT_TARGET`,
`REMOVE_DAMAGE_COUNTER`, `HEAL`, `ACTIVATE`, `COIN_HEAD`, `TO_HAND` など
主要コンテキストに専用ヒューリスティックを追加した（3 種 → 15 種以上）。

### 効果の検証

`diag_ability.py`（提出非対象）で 1 ゲーム自己対戦し、
MAIN で特性を選んだ回数・ダメカン配置コンテキストの出現回数を計測した。

**計測結果（自己対戦 1 ゲーム、探索 0.4 秒/手）:**

| 指標 | 回数 |
|------|------|
| MAIN で特性(ABILITY)を選んだ | **29 回** |
| ダメカン配置コンテキスト出現 | **12 回**（context 13 = DAMAGE_COUNTER） |

修正前は特性をほぼ使えていなかったため、明確に活用されるようになった。

**勝率検証（`diag_winrate.py`、対ランダム 10 戦、探索 0.5 秒/手）:**

- **9 勝 1 敗（90%）** — 特性対応後も強さは維持（むしろ向上）。
- 本番設定（3 秒/手）ではさらにロールアウト数が増え精度が上がる。

---

## 課題

### 1. 相手ターンの扱い

Flat MC では相手ターンもランダム行動でシミュレートする。
実際の相手は最善手を打つため、ロールアウトの精度が低い。

**対策候補:**
- 相手ターンには「最大ダメージを与える」ヒューリスティックを使う
- 相手の行動分布をモデル化する（困難）

### 2. 選択コンテキストの網羅

48+ 種類の SelectContext に対して、現在ヒューリスティックが存在するのは 3種類のみ。
MC 探索が失敗した際のフォールバックが弱い。

**対策候補:**
- 各コンテキストに専用ヒューリスティックを追加
- `ai-development-roadmap.md` の Level 0 チェックリストを実施

### 3. 評価関数の精度

現在の評価関数は「サイド差分」を最重要指標にしているが:
- サイドを1枚取られるポケモンと2枚取られる ex ポケモンの区別がない
- 手札の質（描いたカードの価値）を評価していない
- 山札残枚数が0に近い危機を検知していない

### 4. search_begin の失敗

相手の予測が不正確な場合（例: 裏向きバトルポケモンが基本ポケモンでない）、
`search_begin` が `ValueError` を投げて MC 探索が丸ごとスキップされる。

**対策候補:**
- エラーを細かく捕捉して原因ごとにリトライ
- 失敗した予測を記憶して次回から修正

### 5. 時間制限の不確かさ

Kaggle の実際のターン制限時間が不明。
2.5秒 が長すぎる場合はタイムアウトになる可能性がある。

**対策候補:**
- `MCTS_TIME_BUDGET_SEC` を小さくして安全マージンを確保
- 1回のロールアウト時間を計測してダイナミックに調整

---

## 改善した点（既存コードとの比較）

| 項目 | 旧コード (main_minimal_snapshot.py) | 新コード (main.py) |
|------|--------------------------------------|---------------------|
| 探索 | なし（ルールのみ） | Flat Monte Carlo Search |
| 評価 | なし | サイド・HP・ベンチを考慮した評価関数 |
| ループバグ | `for` 内で早期 `return` | 分類後に優先順位判断 |
| Enum 比較 | 数値 (`option.type == 7`) | `OptionType.ATTACH` などを使用 |
| 攻撃選択 | 先頭の攻撃 | 最大ダメージ / きぜつ優先 |
| メモリ管理 | なし | ロールアウト後に `search_release` |
| エラー処理 | なし | MC失敗時にヒューリスティックへフォールバック |
| 特性活用 | 先頭を無条件発動 | ダメカン評価 + ターゲティング + 攻撃前発動 |
| コンテキスト対応 | 3 種 | 15 種以上（ダメカン・回復・YESNO 等） |

---

## 動作確認結果

| テスト | 結果 |
|--------|------|
| `local_test.py` (1ゲーム) | 正常完走、MC探索動作 |
| `local_test_advanced.py --games 3 --opponent random` | 3勝0敗 |

### 発見・修正したバグ

**`_predict_opp_deck` のリスト不足バグ:**

`_OPP_SEEN` に登録されたカードIDを `_MY_DECK` からフィルタ除去していたため、
同じカードIDが多数ある場合（例: 同名エネルギー4枚）に filler が不足し、
`opponent_deck does not match the number of cards in opponent's deck.` エラーが発生した。

修正: フィルタ除去を廃止し、`base + shuffled(_MY_DECK)` で補完する方式に変更。
足りない場合は `_MY_DECK` を繰り返して補完（ `while len(combined) < count` ）。

---

## 実行方法

```bash
cd sample_submission

# 1ゲーム動作確認（自分 vs 自分）
python local_test.py

# 複数ゲーム（ランダム相手と10回対戦）
python local_test_advanced.py --games 10 --opponent random --verbose
```

---

## ファイル構成

```
sample_submission/
├─ main.py              # エージェント本体（Flat MC Search + Heuristic）
├─ deck.csv             # 使用デッキ（60枚のカードID）
├─ cg/                  # ゲームエンジン（変更禁止）
├─ local_test.py        # 動作確認スクリプト
├─ local_test_advanced.py  # 複数ゲーム対戦スクリプト
└─ docs/
    ├─ rl-design.md         # RL設計思想
    ├─ implementation-notes.md  # このファイル
    └─ ai-development-roadmap.md  # 開発ロードマップ
```
