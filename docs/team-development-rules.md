# GitHub チーム開発ルール

ポケモンカードゲームAIコンペで、複数人で強いAIを開発するためのルール。

最初から完璧な設計を固定するのではなく、次の5つを重視する。

1. **付け替え可能なAIモジュールを作る**
2. **複数のAI構成を config で簡単に比較できるようにする**
3. **失敗した機能をすぐ外せるようにする**
4. **常に提出可能な安定版を残す**
5. **感覚ではなく、勝率・処理時間・実験記録で判断する**

一言でまとめると:

> 強いAIを一発で作るのではなく、弱くても動くAIに「強い可能性のある部品」を付け替えて比較する。
> AIの種類はブランチではなく config で管理する。強さは対戦結果で判断する。

## 目次

1. [モジュールと共通インターフェース](#1-モジュールと共通インターフェース)
2. [config ルール](#2-config-ルール)
3. [ブランチ運用](#3-ブランチ運用)
4. [開発フロー](#4-開発フロー)
5. [Pull Request ルール](#5-pull-request-ルール)
6. [Issue ルール](#6-issue-ルール)
7. [実験結果の記録ルール](#7-実験結果の記録ルール)
8. [fallback ルール](#8-fallback-ルール)
9. [処理時間ルール](#9-処理時間ルール)
10. [やってはいけないこと](#10-やってはいけないこと)

---

## 1. モジュールと共通インターフェース

各担当者は「完成したAI全体」ではなく、**AIの一部として差し替え可能なモジュール**を作る。
モジュールは `sample_submission/ptcg_ai/` 配下の役割別パッケージに置く。

```text
sample_submission/
├── main.py                    # 提出エントリポイント（ptcg_ai を呼ぶだけ）
├── configs/                   # AI構成（→ 2. config ルール）
├── results/                   # 実験結果（→ 7. 実験結果の記録ルール）
└── ptcg_ai/
    ├── action_selection/      # 行動選択の入口（モジュールの呼び出し・fallback・合法手確認）
    ├── rule_based/ learning/  # policy 系（ルールベース・強化学習）
    ├── search/                # 探索系（例: lethal_simple）
    ├── opponent_modeling/     # 相手デッキ予測系
    ├── board_evaluation/      # 盤面評価系
    ├── hidden_information/    # 非公開情報の予測・推定
    ├── state_view/            # 状態の変換・特徴量化
    └── core/ shared/          # config 読み込みなどの共通処理
```

### 共通インターフェース

同じ種類のモジュールは同じ関数形式で実装する。これにより **config を書き換えるだけで付け替えられる**。

| 種類 | シグネチャ | 返り値 |
|---|---|---|
| policy 系 | `choose_action(state, legal_actions, context)` | 選択する行動 |
| deck predictor 系 | `predict(state, context)` | 予測結果 |
| lethal search 系 | `search(state, legal_actions, context)` | 行動 or `None`（見つからない場合） |
| evaluator 系 | `evaluate(state, action, context)` | スコア |

- `context` には元の `Observation` や config の該当セクションなど、追加情報を dict で渡す。
- 行動は **カードIDではなく `obs.select.option` に対するインデックスのリスト**で返す。

---

## 2. config ルール

AIの構成（どのモジュールをどのパラメータで使うか）は `sample_submission/configs/*.json` に保存する。
**1ファイル = 1つのAI構成**。ファイル名は構成が分かる名前にする。

```text
configs/
  rule_only.json
  rule_lethal.json
  aggressive_lethal.json
  rl_full.json
```

### config の形式

各モジュール種別のキーは**オブジェクト形式**で書く。`enabled`（ON/OFF）、`module`（使用モジュール名）、およびそのモジュールのパラメータを含める。

```json
{
  "name": "rule_lethal",
  "lethal_search": {
    "enabled": true,
    "module": "lethal_simple",
    "max_remaining_prizes": 2,
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000
  }
}
```

- `"lethal_search": "lethal_simple"` のような**文字列だけの指定は使わない**（パラメータの置き場所がなくなり、実験で閾値比較ができないため）。
- パラメータをモジュール内にハードコードせず config に出すことで、`experiment/*` ブランチでは **config の差し替えだけ**で条件比較ができる。

### config 追加時のチェック

- [ ] そのconfigで実行できる
- [ ] 使用している `module` が存在する
- [ ] 既存configを壊していない
- [ ] 何を狙った構成なのかPR本文に書いた

---

## 3. ブランチ運用

| ブランチ | 役割 | 壊れてよいか | 寿命 |
|---|---|---|---|
| `master` | **提出可能な安定版**。いつでも提出候補にできる状態を保つ | ダメ | 長い |
| `integration` | **モジュールを統合して共有する場所**。使える部品を集める | 基本ダメ | 長い |
| `feature/*` | 各機能（モジュール）の開発 | 多少OK | 機能完成まで |
| `experiment/*` | AI構成・条件の検証 | OK | 短い（実験終了まで） |

```text
feature/*  ──PR──▶  integration  ──branch──▶  experiment/*
                        ▲                          │
                        └──── 良かったものだけPR ──┘
                        │
                        └──安定確認後PR──▶  master
```

### master に入れる条件

`integration` で以下を確認してから PR を出す。

- ローカルで提出形式の実行ができる
- 主要configが落ちない・時間制限を超えない
- 既存の安定版と対戦比較して大きく弱くなっていない
- 実験結果が `results/` に記録されている

### integration に入れるもの / 入れないもの

| 入れるもの | 入れないもの |
|---|---|
| 共通インターフェースに従ったモジュール | 勝率検証中の雑な仮コード |
| fallback が用意された処理 | 特定AIだけの一時的なチューニング |
| 既存configを壊さない変更・新config候補 | 他のconfigを壊す変更 |
| 動作確認済み・整理済みの実験結果 | 大量のデバッグprint・根拠のないパラメータ変更 |

`integration` は常に「各モジュールがimportできる」「`main.py` が落ちない」「configを切り替えれば各AIを動かせる」状態を保つ。

### experiment ブランチ

`integration` に存在するモジュールを組み合わせて、**どのAI構成が強いかを検証する**一時ブランチ。
config の組み合わせ・閾値調整・使用条件の検証・勝率/処理時間の比較、一時的なログ出力までは自由にやってよい。

検証テーマの例:

```text
experiment/lethal-trigger-condition   リーサル探索を「常に / サイド2枚以下 / 終盤のみ」で比較
experiment/rl-with-deck-predictor     デッキ予測器の有無で勝率が変わるか比較
experiment/mcts-endgame-only          MCTSを終盤だけ使う構成の検証
```

実験後、`integration` に戻すのは**良かったものだけ**: 有効だったconfig、整理済みの実装、実験結果のMarkdown、妥当性が確認されたパラメータ。
雑な検証コード・一時的なprint・失敗した組み合わせは戻さない。

---

## 4. 開発フロー

```bash
# 1. integration から feature を切ってモジュールを作る
git checkout integration && git pull
git checkout -b feature/lethal-simple
# 実装 → 動作確認 → integration へ PR

# 2. integration から experiment を切って構成を検証する
git checkout integration && git pull
git checkout -b experiment/lethal-trigger-condition
# config を変えて対戦実験 → 結果を results/ に保存

# 3. 良かったものだけ integration へ PR で戻す
# 4. integration で安定を確認したら master へ PR
```

feature から integration に PR を出す前のチェック:

- [ ] 担当モジュール単体で動作確認済み（importエラーなし）
- [ ] 既存configを壊していない
- [ ] 関数の入力・出力が共通インターフェースに合っている
- [ ] 失敗時のfallbackがある
- [ ] 使い方がREADMEまたはPR本文に書かれている

---

## 5. Pull Request ルール

### タイトル

変更内容が分かるプレフィックス付きで書く。

```text
[feature] add lethal_simple search module
[experiment] compare aggressive_lethal config
[fix] prevent rl_policy timeout
[config] add rule_lethal config
[docs] update team rules
```

### 本文テンプレート

```md
## 概要
何を追加・変更したか

## 変更内容
- 変更点1
- 変更点2

## 確認したこと
- [ ] main.py が実行できる
- [ ] 既存configが壊れていない
- [ ] 新しいconfigで実行できる
- [ ] 時間制限を大きく超えていない

## 懸念点
- まだ不安な点・今後改善したい点
```

### レビュー観点

- 既存configが壊れていないか / `main.py` が実行できるか
- インターフェースがそろっているか / fallbackがあるか
- `legal_actions` 外・重複ありの行動を返さないか
- 処理時間が極端に重くないか / configでON/OFFできるか
- 実験結果が必要な変更では `results/` に記録されているか

---

## 6. Issue ルール

タスクはGitHub Issuesで管理し、**大きすぎない単位**に分ける。

```text
悪い例: 強いAIを作る
良い例: lethal_simple のインターフェースを作る
        rule_basic と lethal_simple を組み合わせるconfigを作る
        rl_policy のfallbackを実装する
```

### テンプレート

```md
## 目的
このIssueで達成したいこと

## 作るもの
- 作るファイル / 作る関数 / 追加するconfig

## 完了条件
- [ ] 実装が完了している
- [ ] ローカルで動作確認している
- [ ] 既存configを壊していない
- [ ] 必要なら results/ に記録している

## メモ
補足・懸念点
```

---

## 7. 実験結果の記録ルール

AI同士の比較結果は `sample_submission/results/` に `YYYY-MM-DD_実験名.md` で保存する。

```text
results/2026-07-06_aggressive_lethal.md
results/2026-07-17_lethal_search_trigger.md
```

### テンプレート

```md
# 実験結果

## 実験名
aggressive_lethal の検証

## 比較したAI
- AI_A: rule_only
- AI_B: aggressive_lethal

## 条件
- 対戦数 / 使用環境 / 使用ブランチ / 使用config

## 結果
- 勝利数・勝率
- 平均行動時間・最大行動時間
- エラー落ち回数

## 分かったこと
- 事実ベースの気づき

## 次にやること
- 次のアクション
```

---

## 8. fallback ルール

新しいモジュールは、失敗したときに**必ず既存の安定処理へ戻れる**ようにする。

```python
def choose_action(state, legal_actions, context):
    try:
        action = rl_policy(state, legal_actions, context)
        if action in legal_actions:
            return action
    except Exception:
        pass
    return rule_basic.choose_action(state, legal_actions, context)  # fallback
```

fallback が必要な場面の例:

- RLモデルが行動を返せない / evaluator が異常なスコアを返す
- デッキ予測器が失敗する / リーサル探索が時間切れになる
- `legal_actions` に存在しない行動を返してしまった

fallback 後に返す行動も、`minCount <= len(action) <= maxCount`・インデックス範囲内・重複なしを満たすことを確認する。

---

## 9. 処理時間ルール

強さだけでなく処理時間も重要。変更時は必ず確認する。

- 平均行動時間 / 最大行動時間 / 時間切れ回数 / 重い処理が発生する条件

重くなりやすい処理（MCTS・深いリーサル探索・RL推論・複雑なデッキ予測・全行動候補の詳細評価）は、**常時実行せず条件付きで使う**。

```text
例: リーサル探索はサイド残り2枚以下のときだけ使う
    MCTSは終盤だけ使う
    RLは候補行動を絞ってから使う
```

上限値（時間・深度・ノード数など）は config に出し、計測結果を基に調整する。

---

## 10. やってはいけないこと

- `master` に直接pushする
- `integration` に壊れる実験コードを直接入れる
- configで切り替えられないAI構成を作る
- 既存AIを壊す変更を無断で入れる
- 実験結果を残さずに「強い気がする」で判断する
- `legal_actions` に存在しない行動を返す
- 重い処理を常時実行する
- デバッグprintを大量に残したままマージする
