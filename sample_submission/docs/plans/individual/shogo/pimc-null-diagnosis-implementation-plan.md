# 実装計画: PIMC本番null の真犯人特定（診断フェーズ）

作成: 2026-07-21
状態: 実装計画
前提: [pimc-production-validation-implementation-plan.md](./pimc-production-validation-implementation-plan.md) 完了。**PIMCも estimated hidden state も本番 `ml_policy` の勝率を動かさなかった（ミラーで両方 null）**（[summary](../../../results/2026-07-21_pimc_prod_validation_summary.md)、[[project_pimc_prod_validation]]）
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## この計画の目的

本番検証は「PIMC/hidden info は本番を強くしない」という答えを出したが、**なぜ効かないのか（真犯人）は未特定**。深掘りで候補が3つ出た:

- **(A) value網が粗い**: 探索の目的関数＝盤面評価が、カード正体を見ず集約特徴のみ（AUC 0.746）。物差しがぼやけている可能性。
- **(B) strategy fusion**: PIMCは根で山札の並びを1通り固定するため、探索が「引く前から引く札を知っている」透視状態でドロー系を過大評価。大量ドローのフーディンで特に悪い。
- **(C) PolicyModel 支配**: 模倣PolicyModelが既に強く、探索の出番が狭くて勝率に波及しない。

**どれが真犯人かで、次の大投資（value網改善／探索の作り込み／探索から撤退）が全く変わる。** この計画は、大投資の前に**安い診断で真犯人を証拠として特定する**ことだけを目的とする。

---

## スコープ

**やること**: (A)(B)(C) を切り分ける安い診断と、直交する山札切れ敗因の診断。すべて「計測・分析」であり、修正・新アルゴリズム実装は含めない。

**やらないこと（Non-goals）**:
- **実証済みインフラの feature/* 切り出し・integration へのPR・tsuoimorikaさん共有（＝足場固め） → 今回はやらない（ユーザー判断で保留）。** 次の実装セッションもこれを再提案しないこと。
- ISMCTS / chance node / expectimax の実装 → 診断結果が(B)を指した場合の**下流**。本計画では作らない。
- value網の再学習・特徴量追加 → 診断結果が(A)を指した場合の下流。本計画では作らない。
- `deck.csv` の変更 → 山札切れ診断が「デッキ構築問題」を指した場合の下流。本計画では変えない。
- `lethal_simple.py` / 本番の `AGENT_TYPE` の変更 → 触らない。

つまり本計画は**「どこに投資すべきかを証拠で決めるための計測」**に徹する。修正は次フェーズ。

---

## 事前確認済みの契約（Contracts）

### 1. value網の直接呼び出し（摂動probe用）

- `ptcg_ai/learning/value_model.ValueModel.predict_win_prob_from_features(features: list[float], turn) -> float`（既存、`value_model.py:144`）: **エンコード済み特徴ベクトルを直接渡せる**。摂動probeの中核。
- `ptcg_ai/learning/encoder.encode_state_from_state(state) -> list[float]`（既存、`value_model.py:128` が使用）: `State` から特徴ベクトルを得る。
- `ptcg_ai/learning/encoder.FEATURE_NAMES: list[str]`（既存、`encoder.py:125`）: 特徴ベクトルの各次元の名前。摂動したい特徴のインデックスをここから引く（例: `self_deck_count`, `self_prize_remaining`, `opp_prize_remaining`, `prize_diff`）。
- 未ロード時・`state=None` 時は `0.5` を返す設計なので例外ガード不要。

### 2. PIMC のサンプル数と統計（determinization スイープ用）

- config の `num_determinizations`（`pimc.py` DEFAULTS=4）と `time_limit_ms`（DEFAULTS=300）。予算は実測で使用率2〜4%（[cost step0](../../../results/2026-07-21_stage2_cost_step0_budget_reality_check.md)）なので15〜20倍の余地がある。
- `ptcg_ai.search.pimc.get_stats()` / `reset_stats()`（既存）: `searches` / `found` / `determinizations_run` / `determination_timeouts` を返す。**実際に完走したサンプル数**を確認できる。

### 3. head-to-head 基盤（スイープの対戦用）

- 本番検証 Step1 で追加済みの `ml_policy_agent.agent(obs, config=...)` 注入点を使い、`league/run_match.play_match(agent0, agent1, deck0, deck1, seed)` を直接呼ぶ使い捨てスクリプト。試合ごと `match_context.reset()`、先手/後手半々、Wilson 95% CI（本番検証と同じ方法論）。

### 4. 山札切れ診断（実ログ）

- `kaggle_replays/fetch_my_episodes.py`（README記載）: 直近提出のエピソードを取得。既存の敗因分析は n=28（[[project_deckout_loss_cause]]、山札切れ29%）だが**小標本**。追加取得で標本を増やす。
- 分析スクリプトの雛形は `kaggle_replays/_analyze_missed_lethal.py`（リーサル見逃し分析）が参考になる。

---

## ステップ

### Step 1: value網の摂動probe（真犯人(A)の判定＋山札切れ感度）【最優先】

- 多様な中盤〜終盤の実局面を数十件用意（自己対戦 or リプレイから `State` を収集）。各局面を `encode_state_from_state` で特徴ベクトル化。
- 特徴を1つずつ動かし、`predict_win_prob_from_features` の応答が**素直な向き・意味ある大きさ**で動くかを見る:
  - `self_deck_count` を高→0 に下げる → 勝率は**下がる**べき（山札切れリスク。**山札切れ問題の中核判定でもある**）
  - `self_prize_remaining` を減らす（自分がサイドを取る）→ **上がる**べき
  - `opp_prize_remaining` を減らす（相手が取る）→ **下がる**べき
  - `prize_diff` を有利方向へ → **上がる**べき
- **既知の注意**: 単一特徴の摂動は非現実的な盤面（deck_count だけ減って discard_count が整合しない等）を作る＝value網の学習分布外。純粋な単調性チェックとしては有効だが、補完として**実トラジェクトリでの相関**（実際に山札が減っていく試合で、予測勝率が山札切れ結末と相関するか）も見る。
- 完了条件: `results/2026-07-21_diag_valuenet_probe.md` に、特徴ごとの応答の向き・大きさ（局面集合での平均）と判定を記す。**特に「value網は山札枚数の減少を勝率低下として捉えているか」に明確なYes/Noを出す。** これが(A)の判定と山札切れ対処の方向（value網起因か否か）を同時に決める。

### Step 2: determinization スイープ（真犯人(B)の切り分け）

- `num_determinizations ∈ {4, 16, 64}`（予算余地を使い、必要なら `time_limit_ms` も引き上げて完走数を確保）で `ml_pimc` vs `ml_lethal` の head-to-head を再実行。各設定で勝率・95% CI・`pimc.get_stats()` の完走サンプル数を記録。
- 解釈: 勝率が**サンプル増で上がる** → ノイズが一因（安い改善余地あり）。**平坦** → ノイズは犯人でない → 残るは strategy fusion か目的関数(A)。
- **重要**: これは「サンプルノイズ」の検証であって strategy fusion は直さない（根固定のままなので）。この区別を結果に明記する。
- 完了条件: `results/2026-07-21_diag_determinization_sweep.md` に勝率推移と解釈を記録。

### Step 3（任意）: strategy fusion 過大評価probe（真犯人(B)の直接確認）

- ドロー/圧縮の選択肢がある決定点をサンプルし、(a) PIMCの探索が見積もったその手の価値 と (b) 透視せず正直にドローを多数再サンプルして前に進めた「正直な期待値」を比較。(a) が系統的に (b) より高ければ strategy fusion の兆候。
- やや手間なので任意。完了条件: 実施時のみ `results/2026-07-21_diag_strategy_fusion.md`。

### Step 4: 山札切れ敗因診断（直交レバー、並行）

- `fetch_my_episodes.py` で実ログを追加取得し標本を増やす（n=28 → できれば数倍）。山札切れ負けを特定し、各試合で山を削った決定を追って分類:
  - (a) 回避可能な特性/カード使用でエージェントが引きすぎた → **エージェント判断の問題**（value網感度 or ルールガード）
  - (b) デッキが元々ドロー過剰・回復札不足 → **デッキ構築の問題**（deck.csv、エージェントを変えても直らない）
  - (c) 強制・回避不能
- 完了条件: `results/2026-07-21_diag_deckout_classification.md` に、拡大標本での (a)/(b)/(c) 内訳と、「山札切れは主にエージェント判断かデッキ構築か」の判定を記す。

### Step 5: 統合と大投資のルーティング

- Step1-4 を1つの結論に集約し、**証拠に基づく次の大投資**を選ぶ。ルーティング:

| 診断結果 | 指す真犯人 | 次の大投資 |
|---|---|---|
| value網が deck_count/サイドに素直に反応しない | (A) | **value網の改善**（カード正体を含む特徴量・山札切れ感度・学習）。探索/盤面評価/山札切れの全部に効く最大レバレッジ |
| value網は健全 + サンプル増で勝率動かず | (B) | 探索の作り込み（chance node/ISMCTS or 発火域の絞り込み）。ただしミラー計測の限界を承知で |
| どちらも該当せず勝率不変 | (C) | 探索から離れる。PolicyModel改善 or 山札切れ/デッキ構築へ |
| 山札切れが主にデッキ構築問題 | 直交 | `deck.csv` 見直し（安い独立の勝ち候補） |

- 完了条件: `results/2026-07-21_diag_summary.md` に、選んだ大投資と根拠を記載。

---

## Definition of Done

- [ ] value網摂動probe完了。目的関数の信頼性と山札枚数感度に明確な判定
- [ ] determinization スイープ完了。ノイズ vs strategy fusion/目的関数 を切り分け
- [ ] 山札切れ敗因を拡大標本で分類し、エージェント判断 vs デッキ構築 を判定
- [ ] Step5 summary に、証拠に基づく次の大投資のルーティングを記載
- [ ] （明示的にやらないこと: 足場固めの切り出し・統合、および (A)(B)(C) いずれの大投資本体の実装も本計画では行わない）

この診断が終われば、「これ以上PIMCを勘で作り込むか」ではなく、**証拠が指す1点（おそらく value網 か 山札切れ）に投資する**判断ができる。大投資フェーズはその結果を見て別ファイルで計画する。
