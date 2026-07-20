# 相手デッキ分布推定・教師あり学習 実装方針

`rough_predictor` の手作業スコアを壊さずに、相手デッキ候補を
「メガルカリオex 70%、シロナのガブリアスex 20%、other 10%」のような
合計 100% の分布として扱うための方針メモ。

## 1. 結論

- 既存の `rough_predictor.py` / `rough_predictor.json` は変更を最小限にし、
  **説明可能なルールベース推定器**として残す。
- 合計 100% の候補分布は、別モジュール `deck_distribution.py` で作る。
- 教師あり学習モデルは、さらに別モジュール `learned_predictor.py` と
  `tools/opponent_modeling/` 配下の学習スクリプトで育てる。
- 強化学習ではなく、まずは **教師あり分類 + 確率キャリブレーション** として作る。
  強化学習は、推定結果を使ってプレイ方針まで学習する段階で検討する。

## 2. 既存実装を壊さない理由

現在の `rough_predictor` は、公開情報から各アーキタイプにスコアを付け、
`normalized_score = score / confident_score` と `prediction_decision` のしきい値で
`confident` / `unknown` を決める。

この仕組みは以下の価値があるため、学習モデルに置き換えず残す。

- 根拠カードと加点理由を `evidence` として説明できる。
- 新デッキ追加時に `rough_predictor.json` だけで素早く調整できる。
- 学習データが少ないデッキでも最低限の推定ができる。
- 学習モデルが未配置・低信頼・エラーのときのフォールバックに使える。

そのため、確率分布や教師あり学習は `rough_predictor` の中へ直接混ぜず、
横に足す。

## 3. 推奨ファイル構成

提出物に含める推論コード:

```text
sample_submission/ptcg_ai/opponent_modeling/
├── opponent_knowledge.py
├── rough_predictor.py
├── rough_predictor.json
├── deck_distribution.py
└── learned_predictor.py
```

提出物に直接必要ない学習・評価コード:

```text
tools/opponent_modeling/
├── build_training_dataset.py
├── train_deck_classifier.py
└── evaluate_deck_classifier.py
```

ドキュメント:

```text
sample_submission/docs/plans/opponent-deck-predictor/
├── plan.md
├── rough-predictor-score-tuning.md
└── deck-distribution-and-learned-predictor-plan.md
```

## 4. モジュール責務

### `rough_predictor.py`

手作業 config に基づく候補採点を担当する。

- 入力: `State`, `OpponentKnowledge | dict | None`
- 出力: 既存の `deck_type`, `status`, `score`, `normalized_score`, `match_rate`,
  `evidence`, `candidates`
- 役割: 説明可能な根拠付き候補リストを作る
- 変更方針: 返却互換性を壊さない。大きな責務追加はしない。

### `deck_distribution.py`

`rough_predictor` の候補スコアを、合計 100% の候補分布に変換する薄い層。

- 入力: `rough_predictor.predict()` の結果
- 出力: `deck_distribution`
- 役割: `normalized_score` / `match_rate` を確率風の分布へ変換し、
  足りない確信度を `other` に逃がす
- 注意: 初期段階の `probability` は「推定確率」ではなく
  「ルールスコア由来の候補配分」と明記する。

### `learned_predictor.py`

学習済みモデルを使って、盤面時点ごとの相手デッキ分布を返す。

- 入力: `State`, `OpponentKnowledge | dict | None`
- 出力: `deck_distribution`
- 役割: 教師あり分類モデルの推論だけを担当する
- 注意: 学習処理や評価処理は入れない。提出時に重くならないようにする。

### `tools/opponent_modeling/`

教師あり学習のためのローカル用ツール群。

- リプレイやシミュレーション結果から学習データを作る。
- 特徴量と正解ラベルを保存する。
- モデルを学習・評価し、提出物側から読める軽量な成果物にする。

## 5. 返却スキーマ案

既存の `rough_predictor` 結果に、別キーとして `deck_distribution` を足す。

```python
{
    "source": "rough_distribution",
    "top_candidate": "mega_lucario_ex",
    "deck_distribution": [
        {
            "deck_type": "mega_lucario_ex",
            "display_name": "メガルカリオex",
            "probability": 0.70,
            "basis": "rough_score"
        },
        {
            "deck_type": "cynthias_garchomp_ex",
            "display_name": "シロナのガブリアスex",
            "probability": 0.20,
            "basis": "rough_score"
        },
        {
            "deck_type": "other",
            "display_name": "other",
            "probability": 0.10,
            "basis": "unexplained_mass"
        }
    ]
}
```

将来、教師ありモデルを使う場合:

```python
{
    "source": "learned_predictor",
    "top_candidate": "mega_lucario_ex",
    "deck_distribution": [
        {"deck_type": "mega_lucario_ex", "display_name": "メガルカリオex", "probability": 0.64},
        {"deck_type": "cynthias_garchomp_ex", "display_name": "シロナのガブリアスex", "probability": 0.24},
        {"deck_type": "other", "display_name": "other", "probability": 0.12}
    ],
    "fallback_used": false
}
```

## 6. 分布化の初期実装方針

最初は ML を使わず、`rough_predictor` の `candidates` から分布を作る。

1. `candidates` の `normalized_score` を取り出す。
2. スコアが 0 以下の候補は除外する。
3. 温度付き softmax などで候補間の比率を作る。
4. 最有力候補の `normalized_score` が低い場合は、残りを `other` に寄せる。
5. 合計が必ず 1.0 になるように丸め誤差を最後に補正する。

例:

```text
候補 normalized_score:
- メガルカリオex: 1.10
- シロナのガブリアスex: 0.55
- メガスターミーex: 0.20

分布:
- メガルカリオex: 0.62
- シロナのガブリアスex: 0.22
- メガスターミーex: 0.06
- other: 0.10
```

この段階では、`probability` は実測キャリブレーション済み確率ではない。
UI やログでは `estimated_probability` や `distribution` のように表現し、
「70% と言ったら本当に 70% 当たる」という意味にしない。

## 7. 教師あり学習の方針

教師あり学習では、各試合の途中時点をサンプルにする。

- 特徴量: その時点までに見えた相手カード、エネルギー色、公開ゾーン、ターン数など
- 正解ラベル: 実際の相手デッキ archetype
- 目的: `P(deck_type | observed_public_information)` を出す

最初のモデル候補:

- ロジスティック回帰
- Naive Bayes
- RandomForest / LightGBM 系

ニューラルネットや強化学習は初期段階では優先しない。
カード出現の sparse feature なので、軽い分類器の方が説明・検証・提出時の安定性で扱いやすい。

## 8. 学習データの作り方

1試合から複数サンプルを作る。

```text
turn 1: まだ見えているカードが少ない → other 多め
turn 3: 進化元とエネルギーが見えた → 候補が数個に絞られる
turn 5: 主軸カードが見えた → 対象デッキの確率が上がる
```

保存するデータ案:

```json
{
  "battle_id": "sample-001",
  "turn": 3,
  "opponent_deck_type": "mega_lucario_ex",
  "features": {
    "observed_card_ids": {"678": 1, "677": 1},
    "observed_cards": {"メガルカリオex": 1, "リオル": 1},
    "energy_types": [4],
    "zone_cards": {
      "active": ["リオル"],
      "bench": ["メガルカリオex"]
    }
  }
}
```

データ生成は `tools/opponent_modeling/build_training_dataset.py` に置き、
提出物側の `sample_submission/` には推論に必要な軽量成果物だけを入れる。

## 9. 確率キャリブレーション

学習モデルの出力をそのまま確率として扱うと、過信しやすい。
そのため、評価データで以下を確認する。

- `70%` と出した候補が、実際に約 70% 当たっているか
- 序盤で過剰に 1 候補へ寄りすぎていないか
- `other` が必要な場面で十分に残っているか

必要に応じて temperature scaling や isotonic calibration を使う。
この段階を通した値だけを「確率」と呼び、通していない値は「候補配分」と呼ぶ。

## 10. 曖昧デッキの扱い

スターミーデッキ / ユキメノコデッキのように、
1つの構築が複数の呼び方を持つ場合は、単一ラベルへ無理に押し込まない。

推奨は、排他的な `deck_distribution` と非排他的な `tags` を併用すること。

```python
{
    "deck_distribution": [
        {"deck_type": "starmie_froslass", "display_name": "スターミー/ユキメノコ", "probability": 0.58},
        {"deck_type": "pure_starmie", "display_name": "スターミー", "probability": 0.20},
        {"deck_type": "other_water", "display_name": "その他水系", "probability": 0.12},
        {"deck_type": "other", "display_name": "other", "probability": 0.10}
    ],
    "tags": {
        "starmie_engine": 0.78,
        "froslass_engine": 0.66,
        "water_deck": 0.94
    }
}
```

実戦判断では、「デッキ名が何か」よりも
「どのエンジン・色・脅威カードを警戒すべきか」の方が重要な場合がある。
そのため、曖昧な構築ほど `tags` を別軸で持つ。

## 11. フォールバック方針

最終的な呼び出し側は、以下の順で安全に使う。

1. `learned_predictor` が使える場合は学習済み分布を使う。
2. 学習モデルが未配置・例外・低信頼の場合は `deck_distribution.from_rough_result()` を使う。
3. `rough_predictor` でも候補がない場合は `other: 1.0` を返す。

```python
try:
    distribution = learned_predictor.predict_distribution(state, knowledge)
except Exception:
    rough_result = rough_predictor.predict(state, knowledge)
    distribution = deck_distribution.from_rough_result(rough_result)
```

提出時は例外で agent 全体を落とさないことを優先する。

## 12. 実装ステップ

### Phase 1: ルールベース分布

- `deck_distribution.py` を追加する。
- `rough_predictor.predict()` の返り値を受け取り、`deck_distribution` を返す。
- `other` を必ず含め、合計 1.0 を保証する。
- 既存の `rough_predictor` の返却互換性は壊さない。

### Phase 2: 評価用ログ

- 予測時点ごとの `rough_result` と `deck_distribution` を保存できるようにする。
- リプレイ・ローカル対戦から「ターンごとの推定変化」を確認する。
- 序盤、中盤、主軸カード公開後で分布が自然に動くかを見る。

### Phase 3: 教師あり学習データ

- `tools/opponent_modeling/build_training_dataset.py` を追加する。
- 既知デッキ同士の試合から、部分観測特徴量と正解ラベルを作る。
- 最初は少数 archetype で動作確認し、後から対象を増やす。

### Phase 4: 学習モデル推論

- `learned_predictor.py` を追加する。
- 軽量な学習済みモデルを読み込み、`deck_distribution` を返す。
- モデル未配置時は明示的に rough fallback へ流す。

### Phase 5: キャリブレーション

- 評価データで確率の過信・過小評価を確認する。
- 必要に応じて温度パラメータや補正テーブルを導入する。
- 「候補配分」と「キャリブレーション済み確率」を区別して記録する。

## 13. テスト観点

- `deck_distribution` の合計が常に `1.0` になる。
- `other` が負数にならない。
- 候補が空の場合は `other: 1.0` になる。
- `rough_predictor` の既存返却スキーマを壊さない。
- 同点・僅差候補で 1 候補に寄りすぎない。
- 主軸カードが見えた後は、該当デッキの分布が上がる。
- 曖昧デッキでは `deck_distribution` と `tags` の両方で表現できる。
- 学習モデルが読めない場合でも agent が落ちずに fallback する。

## 14. 今すぐやるなら

最初の実装は以下だけでよい。

1. `deck_distribution.py` を作る。
2. `from_rough_result(rough_result) -> dict` を実装する。
3. `rough_predictor` 本体は触らない。
4. テストで `other` と合計 1.0 を確認する。
5. README に「これは未キャリブレーションの候補配分」と明記する。

教師あり学習は、その後にリプレイやシミュレーションから
十分な教師データを作れる見通しが立ってから着手する。
