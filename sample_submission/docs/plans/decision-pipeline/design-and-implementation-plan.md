# 意思決定パイプライン統合 設計・実装プラン

## 背景と位置づけ

Kaggle Pokemon TCG AI 向けに、既に個別に完成している部品——確定リーサル探索
(`search/lethal_simple.py`)、模倣学習 Policy(`learning/policy_model.py`)、非公開情報推定
(`hidden_information/`)、相手デッキ推定(`opponent_modeling/`)、SDK シミュレータ
(`cg.api.search_*`)——を **1手ごとの意思決定パイプラインとして config-gated で統合**する。

個々のモデルの再学習・アーキテクチャ変更は行わない。**接続コードのみ**。

### 重要な前提(既存の知見)

- 既存の PIMC(`search/pimc.py`)は本パイプラインとほぼ同型(belief 決定化 + 自ターン先読み +
  N世界平均)だが、(1)候補を Policy top-k ではなくヒューリスティック優先順で絞る、
  (2)相手ターンをモデル化しない、(3)末端評価が ValueModel 固定、という3点で spec と異なる。
- `project_pimc_prod_validation`: **PIMC は本番(ml_policy)へ有意に転移しなかった**。本統合は
  「policy-guided PIMC + 相手モデル + 差し替え可能な手作り評価」であり、勝率が転移する保証は
  ない。狙いは *end-to-end の config 駆動パイプライン + ablation 基盤* を用意し、5構成を総当たり
  計測できるようにすること(それ自体が現状欠けている)。

## 統合アーキテクチャ

毎ターン、前段で決まれば後段をスキップ:

```
Step1 リーサル/危機チェック(厳密全探索, ML不使用)  ← 既存 lethal_simple を流用
Step2 Policy で候補を top-k に絞る                    ← 新規(pipeline.py)
Step3 belief サンプリングで N 世界を生成             ← 既存 search_adapter/hidden_information 流用
Step4 各世界×各候補をシミュレータで先読み評価       ← 新規(相手ターンは Policy を相手モデルとして適用)
       末端評価は LeafEvaluator(手作り既定, Value 差し替え可)  ← 新規(leaf_eval.py)
Step5 N世界平均が最大の手を選択(同点近傍は Policy 確率で決定)  ← 新規
時間管理: 残り時間 ÷ 推定残りターンで1手予算。超過時は Policy top1 へ即フォールバック
```

## 新規/変更ファイル

### 新規

1. `ptcg_ai/search/leaf_eval.py`
   - `LeafEvaluator` プロトコル: `evaluate(state, me: int) -> float`(0..1、me視点の勝ち見込み)。
   - `HandcraftedEvaluator`: サイド差・総打点・エネルギー加速・ベンチ展開数の線形結合。
     係数は `COEFFS` 定数1箇所に集約。`board_evaluation` の既存部品を読み取り専用で流用。
   - `ValueModelEvaluator`: 既存 `ValueModel.predict_win_prob_from_state` を me 視点へ整列して包む。
   - `build_evaluator(config) -> LeafEvaluator`: `config["leaf_eval"]["kind"]`("handcrafted"|"value")
     で選択。既定 "handcrafted"。

2. `ptcg_ai/search/pipeline.py`
   - チーム共通の探索インターフェース `search(state, legal_actions, context) -> list[int] | None`
     (lethal_simple/pimc と同一シグネチャ)。context キーも同じ(`observation`/`config`/
     `hidden_state_factory`)+ 追加で `policy_model`(候補絞り込み用)、`leaf_evaluator`。
   - Step2: `policy_model.score_options(obs, ...)` → top-k first-move 候補。top1 が
     `top1_shortcut_prob`(既定0.9、softmax後)以上に集中していれば即 top1 を返す(自明手の高速化)。
   - Step3: `hidden_state_factory()` を N 回呼んで N 決定化(pimc と同じ流用点)。
   - Step4: 各(世界, 候補)で `search_begin`→候補 first move を適用→自ターン終端まで展開→
     相手ターンを `opponent_depth` 手 Policy(=相手視点の score_options)で進める→
     `leaf_evaluator.evaluate(leaf_state, me)`。
   - Step5: 候補ごとに N世界平均、最大を選択。差が `tie_eps` 以下なら Policy スコア上位を採用。
   - 時間管理: `time_limit_ms`(pimc と同様の壁時計 deadline)。予算切れは探索済み最良/なければ None。
   - 危機フィルタ: `crisis_filter` 有効時、相手が次ターン勝ちうる盤面かを簡易判定し、回避できる
     候補のみ Step4 に渡す(初版はヒューリスティック; 厳密探索は将来拡張)。

### 変更(すべて config-gated、既定は挙動不変)

3. `ptcg_ai/ml_policy/ml_policy_agent.py`
   - `_SEARCH_MODULES` に `"pipeline": pipeline` を追加。
   - `_try_pipeline(obs, config)` を新設: `config["pipeline"]["enabled"]` が真のときだけ
     `pipeline.search` を呼ぶ。`_try_lethal` の後、Policy top1 baseline の前に挿入。
   - 既存の production config(`ml_lethal_attackplan_v0only`)は `pipeline` キーを持たないため
     `_try_pipeline` は常に None を返し、**本番挙動は完全に不変**。

### 新規 config(ablation 5構成)

`configs/` に追加。`abl_1` は rule_based エージェント自体なので config 不要(league 側で指定)。

- `abl_2_policy_only.json` — lethal/pipeline とも無効 → Policy top1 のみ。
- `abl_3_policy_eval1.json` — pipeline 有効, N=1, depth=1, belief 無し(dummy hidden state) → 1手読み。
- `abl_4_policy_searchN1.json` — pipeline 有効, N=1, 通常 depth, belief 無し(公開情報のみ)。
- `abl_5_full.json` — pipeline 有効, N=8, estimated belief, 相手モデル on。

### 新規 ablation ハーネス

4. `league/run_ablation.py`
   - `run_match.play_match` は変更しない。`run_league` の `build_agent`(config 注入)を流用し、
     5 競技者(rule_based + 上記4 config)を総当たりして勝率マトリクスを JSON/表で出力。

### テスト

5. `sample_submission/tests/unit/test_leaf_eval.py`, `test_pipeline.py`
   - leaf: 手作り評価が [0,1]、サイド有利で単調増加、me 視点整列。
   - pipeline: enabled=false で None(挙動不変)、top1 集中で即返し、contract 準拠、例外時 None。
   - `test_ml_policy_agent.py` に「pipeline キー無し config で挙動不変」の回帰1件。

## やらないこと

- 各モデルの再学習・アーキテクチャ変更。
- リーサル判定への ML 導入。
- デッキ固有ハードコード分岐(合法手・遷移は必ず SDK 経由)。
- フル MCTS 化(スコープ外)。まず end-to-end を動かす。
