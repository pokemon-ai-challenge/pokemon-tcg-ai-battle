# ML 版 相手デッキ予測器 実装プラン

ブランチ: `feature/ml-opponent-deck-predictor`
作成日: 2026-07-19

## 目的

対戦中の任意の時点で、見えている相手の情報から**相手デッキのアーキタイプを確率分布として出力**する。

- 出力は全アーキタイプ + `other`(未知/定義外)の合計が **100% になる確率分布**
  - 例: `mega_lucario_ex: 80%, shirona_garchomp_ex: 12%, other: 5%, ...`
- 学習は**すべてローカル(提出前)**で行う。試合中は学習済み重み(JSON)を読み込んで
  純 Python の行列積 + softmax で推論するだけ。試合中の学習・MLライブラリ依存は一切なし。

## キーとなる事実(調査済み)

1. **正解ラベルの人手付けは不要。**
   Kaggle リプレイの step 0 のアクションに両プレイヤーの **60枚デッキリストがそのまま入っている**
   (デッキ選択フェーズの返り値)。つまり全リプレイに「相手のデッキは実際何だったか」の正解が記録済み。
2. `kaggle_replays/index/episodes_master.jsonl` に **719 エピソード・481 チーム**分のインデックスがあり、
   `kaggle_replays/replays/` にリプレイ本体(719件)がダウンロード済み。
3. `sample_submission/ptcg_ai/opponent_modeling/rough_predictor.json` に
   **20 アーキタイプのキーカード定義**(role: anchor / exclusive_core / signature 等)が既にある。
   → 60枚のフルデッキリストに対するアーキタイプ自動ラベリングに流用できる。
4. `opponent_knowledge.py`(`OpponentKnowledge`)が相手の公開カードを name 単位で蓄積する機能を実装済み。
   `get_prediction_features()["observed_cards"]` が `{カード名: 枚数}` を返す。
   → **学習時とランタイムで同一の特徴量抽出コードを共有**でき、特徴量のズレ(train/serve skew)を構造的に防げる。
5. リプレイ内の observation 辞書は `cg.api.to_observation_class(obs_dict)` で `Observation` に変換できる
   (main.py と同じ経路)。

## アーキテクチャ

```
【オフライン学習パイプライン】kaggle_replays/deck_predictor/
replays/*.json
  → extract_decks.py   … step0 アクションから両者の60枚デッキを抽出 → deck_db.jsonl
  → label_decks.py     … rough_predictor.json のキーカードでアーキタイプ自動ラベリング
                          → deck_labels.jsonl + 分布レポート(何%がどのアーキタイプか)
  → build_dataset.py   … 各意思決定時点の「見えている相手情報」を OpponentKnowledge で再現
                          → dataset.jsonl(特徴量 + 正解アーキタイプ)
  → train.py           … 多クラスロジスティック回帰を学習 → deck_predictor_weights_base.json
  → adjust_prior.py    … 事前分布シフト補正(intercept 補正)→ deck_predictor_weights.json
  → calibrate.py        … エビデンス数バケット別の温度スケーリングでキャリブレーション補正
                          (meta.calibration を追記。coef/intercept は変更しない)
  → evaluate.py        … ターン別正解率レポート(validation はエピソード単位分割)

【ランタイム(提出物)】sample_submission/ptcg_ai/opponent_modeling/
ml_predictor.py               … 重みJSONを読み、observed_cards + turn → softmax 確率分布(純Python)
deck_predictor_weights.json   … 学習成果物(main.py と同梱して提出)
```

## データフォーマット

### deck_db.jsonl(1行 = 1プレイヤー×1エピソード)

```json
{"episode_id": "86776342", "player_index": 0, "team_name": "...",
 "rank_at_fetch": 3, "deck_card_ids": [756, 756, ...60個],
 "deck_card_names": {"カード名": 枚数, ...}}
```

### deck_labels.jsonl

```json
{"episode_id": "86776342", "player_index": 0, "archetype": "mega_lucario_ex",
 "score": 29.5, "matched_cards": ["メガルカリオex", ...]}
```

ラベリング規則: 各アーキタイプについて rough_predictor.json の role 重みでスコアリングし、
最高スコアがしきい値以上ならそのアーキタイプ、未満なら `other`。
60枚全部見えている状態での判定なので、anchor / exclusive_core があればほぼ確定する。
分布レポートを出力し、`other` が多すぎる場合は人間が定義追加を検討する(ここだけが人の作業)。

### dataset.jsonl(1行 = 1意思決定時点)

```json
{"episode_id": "...", "player_index": 0, "turn": 5, "step": 42,
 "observed_cards": {"カード名": 枚数, ...},   ← OpponentKnowledge の出力そのまま
 "label": "mega_lucario_ex"}
```

サンプル数の目安: 719 エピソード × 2 視点 × 数十手 = 数万サンプル。

### deck_predictor_weights.json(学習成果物・提出物に同梱)

```json
{
  "meta": {"trained_at": "...", "n_episodes": 719, "n_samples": 12345,
           "feature_type": "card_name_counts+turn", "version": 1},
  "classes": ["mega_lucario_ex", "...", "other"],
  "feature_names": ["__turn__", "カード名A", "カード名B", "..."],
  "coef": [[...n_features個...], "... n_classes 行"],
  "intercept": ["...n_classes個..."]
}
```

- `feature_names` の順序が特徴ベクトルの定義。`__turn__` はターン数、以降は観測カード名(枚数が値)。
- **カード名は英語名**(例: `Alakazam`)。`OpponentKnowledge` がエンジンから受け取る名前が英語のため、
  学習・ランタイムとも英語名で統一される。一方、ラベリング(label_decks.py)は rough_predictor.json の
  日本語名と照合するため JP_Card_Data.csv を使う。混同しないこと。
- 語彙にないカード名は無視(学習データに現れなかったカード)。
- 推論: `z = coef @ x + intercept` → (キャリブレーションがあれば `z / T`) → softmax。
  数十KB程度、推論はミリ秒未満。

### meta.calibration(温度スケーリングによるキャリブレーション、任意フィールド)

観測エビデンス(見えているカードの種類数)が少ない序盤ほど、生ロジットの softmax は
過信(overconfidence)しやすい。`kaggle_replays/deck_predictor/calibrate.py`
(adjust_prior.py の後に実行するチューニング層)が、エビデンス数のバケットごとに
validation データ上で温度 `T` を最適化し、`meta.calibration` として重みJSONに追記する。
`coef` / `intercept` は変更せず、ランタイム(`ml_predictor.py`)が推論時に動的に
`z / T` を適用するだけ。フィールドが無い(古い)重みJSONを読んだ場合は常に `T=1.0`
(無補正)として扱われ、動作は導入前と完全に同じになる(後方互換)。

```json
"calibration": {
  "method": "evidence_count_bucketed_temperature_scaling",
  "fitted_at": "2026-...",
  "fitted_on": "validation split (output/model/split.json)",
  "buckets": [
    {"min_evidence": 0, "max_evidence": 0, "temperature": 3.8, "n_valid_samples": 1234,
     "logloss_before": 1.9, "logloss_after": 1.1, "top1_acc": 0.42},
    {"min_evidence": 1, "max_evidence": 1, "temperature": 2.1, "n_valid_samples": 987, ...},
    {"min_evidence": 2, "max_evidence": 3, "temperature": 1.4, ...},
    {"min_evidence": 4, "max_evidence": null, "temperature": 1.0, ...}
  ]
}
```

- `evidence_count(observed_cards)` = `feature_names` のうち `__turn__` を除くカード名で
  `observed_cards.get(name, 0) > 0` であるものの個数(ユニークなカード名の数。枚数の合計ではない)。
  この定義は学習(`calibrate.py`)とランタイム(`ml_predictor.evidence_count()`)で共有する。
- `buckets` は `min_evidence` 昇順で連続・網羅的。`max_evidence: null` は上限なし。
- 各サンプルの全クラスのロジットを同じ正の `T` で割るのは単調変換なので、
  **argmax(top-1予測クラス)はキャリブレーション前後で変わらない**。変わるのは確率の
  「強さ」(reliability)と log loss のみ。`evaluate.py` で top-1 正解率が変化した場合は
  実装バグ。
- **`T` は 1.0 を下回らない(慎重側に倒す設計判断)**。`calibrate.py` の温度最適化は
  `scipy.optimize.minimize_scalar(..., bounds=(1.0, 15.0), method="bounded")` で、探索範囲
  自体を `[1.0, 15.0]` に制限している(post-hoc なクランプではなく最適化の定義域そのもの)。
  T<1.0 はモデルの生ロジットより「シャープ化」する(確信度を強める)方向の補正であり、
  「エビデンスが薄いときに断定しすぎない」というキャリブレーション導入の目的に反するため、
  緩和(T>1)方向のみを許す。無制約最適化では `evidence_count=1` バケットが T*≈0.78
  (シャープ化)になり、判別力の低いカード1枚しか見えていない曖昧な場面(例:
  `{"Snorunt": 1}`)で誤答の確信度がむしろ悪化する実例が確認されたため、この制約を導入した。
  無制約最適解が 1.0 未満のバケットは境界の `T=1.0`(実質無補正)に張り付く。
- パイプライン順序: `train.py` → `adjust_prior.py` → `calibrate.py` → `evaluate.py`。
  `calibrate.py` は adjust_prior.py の出力(事前分布シフト補正済みの intercept)に対して
  ロジットを計算し、`meta.calibration` だけを追記する。

## モデル方針

- **多クラスロジスティック回帰(softmax回帰)+ L2 正則化**を第一候補とする。
  特徴量が「カードの有無/枚数」なので線形で十分強く、重みの解釈も容易。
- 学習は scikit-learn を使う(ローカルのみ)。scikit-learn が環境に入らない場合は
  numpy の勾配降下で自前実装してもよい(50行程度)。
- `other` は1つのクラスとして学習に含める。序盤で情報が少なければ自然に分布が平坦になり、
  「まだ分からない」が確率として表現される。
- 必要になったら後段で温度スケーリング等のキャリブレーションを検討(初版ではやらない)。

## 評価方針

- **train/valid 分割はエピソード単位**(同一対局のターン違いが両側に跨るリークを防ぐ)。
- 主要メトリクス: ターン別 top-1 正解率・log loss。
  「ターンNまでの観測でどこまで当たるか」の曲線をレポートする。
- 参考比較: 既存ルールベース rough_predictor と同一 validation での正解率比較(余力があれば)。

## データスケールアップ(フェーズ2・実装完了後)

> 詳細方針は [ml-predictor-phase2-scaling.md](ml-predictor-phase2-scaling.md) に分離した。以下は初期メモ。

- 初版は既存の 719 エピソードで学習・評価まで通す。
- その後 `fetch_top_episodes.py --top 100〜200 --max-episodes <大きめ>` を**定期実行して蓄積**
  (episodes_master.jsonl は追記式・重複排除済み)。リプレイは1件 3〜4MB なのでディスクと相談。
- メタ変化時は fetch → 再学習 → 重みJSON差し替え → 再提出のフロー。

## ランタイム統合(このブランチのスコープ外でもよい)

- `ml_predictor.py` は `predict(observed_cards: dict[str, int], turn: int) -> dict[str, float]` を提供。
- agent 本体への接続(推定結果を search_begin の相手デッキ予測に使う等)は別タスクとする。
  まずは予測器単体 + テストまで。

## 実装タスク分割

| タスク | 内容 | 置き場所 |
|---|---|---|
| A | オフラインパイプライン一式(extract / label / build / train / evaluate)+ 719件での実行 | `kaggle_replays/deck_predictor/` |
| B | ランタイム推論モジュール + ユニットテスト | `sample_submission/ptcg_ai/opponent_modeling/ml_predictor.py`, `sample_submission/tests/unit/` |

A と B は weights JSON スキーマ(本ドキュメント)を契約として並行実装可能。
