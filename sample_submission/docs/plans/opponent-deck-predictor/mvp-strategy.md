# 相手デッキ予測器 MVP 方針

対象: `sample_submission/ptcg_ai/opponent_modeling/rough_predictor.py`  
関連 config: `sample_submission/ptcg_ai/opponent_modeling/rough_predictor.json`

## 1. この Issue で目指すこと

この Issue では、相手の公開盤面と観測済みカード情報から「相手が大まかに何デッキか」を返す
**ルールベースの予測器 MVP** を作る。

目標は高精度な分類器ではなく、まず以下を安定してできる状態にすること。

- 公開情報だけで主要アーキタイプを数個の候補に絞れる
- 判定根拠を `evidence` として説明できる
- 確信が持てない局面では無理に決め打ちせず `unknown` を返せる
- 後から特徴量追加や重み調整をしやすい

## 2. 責務の切り分け

### `rough_predictor.py` の責務

- `state` と `opponent_knowledge` から予測用の観測カード集合を作る
- `rough_predictor.json` を読み込んで各デッキタイプをスコアリングする
- 最終的な `deck_type`, `confidence`, `evidence`, `candidates` を返す

### `rough_predictor.json` の責務

- 候補デッキ一覧
- role ごとの共通スコア
- デッキごとの特徴カード
- `unknown` 判定のしきい値
- confidence 計算に使う基準値

### `opponent_knowledge` の責務

- 過去に公開された相手カードを保持する
- 現在盤面にないカードも「観測済み」として予測器に渡せるようにする

この分離により、予測器本体は「観測カードを config に照らして加点する」処理に集中できる。

## 3. MVP の入力範囲

MVP では、まず**相手の公開情報のみ**を使う。

- バトル場
- ベンチ
- トラッシュ
- 公開されている進化ライン
- 公開されている付与エネルギー
- `opponent_knowledge` に保存された観測済みカード

今回の対象外:

- 相手の手札の中身
- 山札の中身の推定
- サイド落ち推定
- MCTS で使う hidden information の生成
- 学習ベースの分類器

## 4. 返り値方針

返り値は常に同じ形の dict にする。

```python
{
    "deck_type": "charizard_ex",
    "display_name": "リザードンex",
    "confidence": 0.72,
    "score": 12.4,
    "evidence": [...],
    "candidates": [...],
}
```

判定不能時は次を返す。

```python
{
    "deck_type": "unknown",
    "display_name": "unknown",
    "confidence": 0.0,
    "score": 0.0,
    "evidence": [],
    "candidates": [],
}
```

### `evidence` の基本方針

- 1 件ごとに「どのカードが」「どの zone で」「どの role として」効いたかを残す
- 人が読んで納得できる説明を優先する
- 汎用カードは原則 `evidence` に出しても強い根拠扱いしない

## 5. スコアリング方針

予測は「カードごとの役割」と「その重み」の足し算で始める。

### role の優先度

- `anchor`: 主役カード。最も強い根拠
- `evolution_base`: 進化元。中程度の根拠
- `evolution_middle`: 進化途中。中程度の根拠
- `dedicated_support`: そのデッキでよく使う専用サポート
- `engine_card`: エンジン札。小から中程度の根拠
- `energy`: 補助的な根拠
- `generic`: 加点しない、または極小

### 基本ルール

- 主軸カードが見えたら大きく加点する
- 進化ラインが見えたら段階的に加点する
- 専用サポートやエンジン札が見えたら補助加点する
- エネルギーは補助情報としてのみ使う
- 汎用カードはデッキ分類の決め手にしない

### 追加ルール

- 同じカードを何度も見ても過剰加点しない
- `state` と `opponent_knowledge` の両方にある同一卡は重複加点しない
- 進化ラインが複数そろったときだけ入る `combo_bonus` を許す
- zone による重み差は入れてもよいが、MVP では小さく保つ

## 6. config 設計方針

`rough_predictor.json` は「role 共通設定」と「デッキ別特徴量」を分ける。

イメージ:

```json
{
  "role_weights": {
    "anchor": 10.0,
    "evolution_base": 4.0,
    "evolution_middle": 5.0,
    "dedicated_support": 3.0,
    "engine_card": 2.0,
    "energy": 1.0,
    "generic": 0.0
  },
  "unknown_threshold": {
    "min_score": 6.0,
    "min_gap": 2.0
  },
  "archetypes": [
    {
      "deck_type": "charizard_ex",
      "display_name": "リザードンex",
      "cards": [
        {"name": "リザードンex", "role": "anchor", "reason": "デッキの主軸カード"},
        {"name": "ヒトカゲ", "role": "evolution_base", "reason": "進化元"},
        {"name": "リザード", "role": "evolution_middle", "reason": "進化ライン"}
      ]
    }
  ]
}
```

### config で大事にすること

- 初期版は card name ベースで始める
- 必要なら後から `card_id` を併記できる形にする
- role の意味は全デッキで共通にする
- 汎用カード一覧を別枠で持てるようにする
- 重みはコードに埋め込まず config で調整可能にする

## 7. 初期対応する主要デッキ

MVP ではまず少数の主要アーキタイプに絞る。

- `mega_lucario_ex`
- `alakazam`
- `dragapult_ex`
- `team_rocket_honchkrow`
- `hydrapple`

表示名の想定:

- メガルカリオex
- フーディン
- ドラパルトex
- ロケット団のドンカラス
- カミツオロチ

理由:

- `pdf_card_editor` で URL から取得しやすい実デッキを題材に確認しやすい
- 主軸カードや進化ライン、周辺カードから rough に当てられるかを見やすい
- config の書き方を固める題材に向いている
- 後から候補デッキを増やしやすい

## 8. 予測ロジックの実装順

1. `state` から相手公開カードを集める
2. `opponent_knowledge` があれば観測済みカードを追加する
3. 各デッキタイプについて role ごとに加点する
4. 根拠カードを `evidence` として保存する
5. スコア上位を `candidates` として並べる
6. しきい値未満、または 1 位と 2 位が近すぎるなら `unknown` を返す

この順序なら、MVP 実装後に以下を足しやすい。

- zone multiplier
- combo bonus
- card_id 厳密照合
- variant 判定

## 9. `unknown` を重視する

この予測器は、外したまま強く信じるよりも、
**分からないときに `unknown` を返せること**を重視する。

そのために最低限入れるべき判定は次の 2 つ。

- 1 位のスコアが最低ライン未満なら `unknown`
- 1 位と 2 位のスコア差が小さければ `unknown`

## 10. 動作確認方針

最低限、主要デッキごとにサンプル盤面を作り、期待する `deck_type` が返ることを確認する。

特に MVP では、`pdf_card_editor` で URL から取得できる実デッキを使って、
次の 5 系統を rough に判定できるかを確認対象にする。

- メガルカリオex
- フーディン
- ドラパルトex
- ロケット団のドンカラス
- カミツオロチ

確認観点:

- 主軸カードのみ見えたケース
- 進化元しか見えていないケース
- 専用サポートだけ少し見えているケース
- 複数候補が競るケース
- 汎用カードしか見えていないケース
- `opponent_knowledge` あり / なし
- URL 由来の実デッキを見たときに、そのデッキらしい候補が上位に来るか

ここではまだ完全な精度は求めない。
まずは「見えている主軸や進化ラインから、そのデッキらしい候補を上位に出せること」を合格ラインにする。

必要に応じて確認結果を `sample_submission/results/` に記録する。

## 11. 実装時の注意

- `cg/` は変更しない
- 既存の config 読み込みや他モジュールを壊さない
- 予測器は単独で import できるようにする
- 返り値形式は将来の上位ロジックから読みやすい形を維持する
- 決め打ちロジックを増やしすぎず、config 追加で拡張できる構造にする

## 12. この方針での MVP 完了イメージ

MVP 完了時点では、以下ができていれば十分。

- 主要デッキを rough に分類できる
- 根拠カードを説明付きで返せる
- 不確かな局面では `unknown` を返せる
- 新しいデッキタイプを config 追加で増やせる

高精度化や hidden information 推定は、その次の段階で扱う。
