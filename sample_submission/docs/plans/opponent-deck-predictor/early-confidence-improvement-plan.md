# 相手デッキ予測器: 序盤過信の改善プラン(生成的ベイズ化)

作成日: 2026-07-19
ブランチ: `feature/ml-opponent-deck-predictor`

## 背景 / 問題意識

対戦ビュアーで観戦していると、**カードが1〜数枚しか見えていない場面で予測器が特定アーキタイプを
断定的に表示する**ことがある。「そのカードは複数デッキで採用されているのに、なぜこの候補に決めた?」
という挙動で、根拠と確信度が釣り合っていない。

### 数字による裏付け(2026-07-19 時点の eval_report.md / calibration buckets)

- 全体 top-1 96.5% は終盤ターン・多数クラスに引っ張られた平均。
- evidence(見えているユニークカード種類数)別の top-1 正解率:
  **0枚: 33%、1枚: 80%、2〜3枚: 〜95%**。ユーザー体感と一致する。
- 学習データの事前分布は alakazam 32%、mega_lucario 14% と大きく偏る。evidence が少ない場面では
  ロジット中の intercept(≒メタ事前分布)が支配的になり、「カードで決めた」のではなく
  「メタで一番多いデッキと言っているだけ」の予測になる。
- 温度スケーリング(evidence バケット別 T)は**バケット内平均の過信しか直せない**。
  同じ「1枚見えた」でも、専用アンカーカード(ほぼ確定)と汎用カード(ほぼ prior)では
  必要な補正が正反対だが、単一 T は両方を同じだけ潰す。argmax も変えられない。

## 方針(優先度順)

### 1. 診断の解像度を上げる 【実装対象・フェーズA】

`kaggle_replays/deck_predictor/evaluate.py` を拡張:

- **evidence 数別バケット評価**(0 / 1 / 2-3 / 4-6 / 7-10 / 11+): サンプル数・top-1 正解率・
  log loss・平均 top-1 確信度・ECE(expected calibration error)。
  evidence 数は `MLDeckPredictor.evidence_count()` を再利用(fit/infer と同一定義)。
- **reliability テーブル**: top-1 確信度ビン(0.0-0.5, 0.5-0.6, …, 0.9-1.0)ごとの
  サンプル数・平均確信度・実際の正解率。「予測90%のとき本当に90%当たっているか」を可視化。
- **誤答ダンプ**(`--error-dump`): 誤答サンプルの
  (episode_id, turn, evidence_count, observed_cards, top-3 予測+確率, 正解)を JSONL 出力し、
  集計サマリ((予測, 正解) 混同ペア頻度、誤答時に見えていたカード頻度)を md で出力。
  → 「共通カード起因の過信」が誤答のどれだけを占めるか定量化する。

### 2. 生成的ベイズ(Naive Bayes)予測器 【実装対象・フェーズB】

`output/deck_db.jsonl`(フルデッキリスト)+ `output/deck_labels.jsonl` から
**P(カード採用 | アーキタイプ) を直接推定**し、ベイズ更新で posterior を出す:

```
log P(c | 観測) = log prior(c) + Σ_{観測カード n, 枚数 k} log P(deck が n を k 枚以上採用 | c)
```

- 汎用カード → 全クラスの尤度がほぼ同じ → posterior は prior に留まる(=正直に「未確定」)。
- 専用カード → 他クラスの尤度 ≈ 0 → 1枚で確信してよい。
- **なぜその予測かがカード別尤度で説明可能**(ビュアーに根拠表示できる)。

設計上の決定事項:

| 項目 | 決定 |
|---|---|
| カード名の対応付け | deck_db の `deck_card_ids` を `cg.api.all_card_data()` の `cardId → name` で EN 名化(OpponentKnowledge の observed_cards と同一経路 = train/serve skew なし) |
| 尤度 | 枚数対応: 観測枚数 k に対し P(≥k 枚 | c)。クラス c のラベル済みデッキ N_c 件から頻度推定 |
| 平滑化 | ラプラス平滑化 (count + α) / (N_c + 2α)、α=1.0(0確率を作らない) |
| prior | adjust_prior.py と同じ「直近N日 × 上位R位」ウィンドウのラベル分布(episode_window.py 再利用) |
| 未知カード名 | 語彙(ラベル済みデッキに現れた名前)に無い観測カードは無視(全クラス等尤度扱い) |
| turn 特徴 | 使わない(インターフェース互換のため引数は受けるが無視) |
| 重み出力 | `output/model/deck_predictor_nb.json`(classes / priors / likelihoods / meta) |
| ランタイム | `sample_submission/ptcg_ai/opponent_modeling/nb_predictor.py`。純Python・外部依存なし。`MLDeckPredictor` と同じ API(`predict(observed_cards, turn)` / `predict_top` / `is_ready`)でドロップイン可能に |

評価: `compare_nb.py` で LR(adjusted) vs NB を同一 validation(split.json)で比較。
指標は top-1 / log loss を evidence バケット別に。→ `output/nb_compare_report.md`。

### 3. 出力の意味論を「診断」から「候補分布」へ 【フェーズC・B の結果を見て】

- top-1 確率が閾値(例 0.6)未満のときは「未確定(候補: A 45% / B 30% / C 15%)」として
  ビュアー表示・エージェント利用の両方で扱う。`predict_top` は実装済みなので利用側の変更のみ。
- ベイズ版なら「観測カード別の尤度寄与」を根拠として表示できる。

### 4. データ側の補強 【フェーズD・並行】

- 少数クラス(ogerpon precision 0.39 / takeruraiko f1 0.51 / oliva 0.00)のリプレイ収集。
- LR を残す場合: エピソード内サンプルの重み付け(1エピソード合計重み1)で
  相関サンプルによる prior 汚染を緩和。クラスバランス学習 + prior 明示外付け。

### やらないこと / 制約

- キャリブレーションのさらなる高度化(vector scaling 等)は、ベイズ化で不要になる可能性が
  高いため後回し。やる場合も **T ≥ 1.0(慎重側、生の見積もりより強気にしない)** の制約を維持する。
- 重みJSONスキーマ(`deck_predictor_weights.json`)は LR ランタイムとの共有契約なので変えない。
  NB は別ファイル(`deck_predictor_nb.json`)にする。

## 成功基準

- フェーズA: 誤答のうち「共通カードのみ観測での過信」の割合が定量化されている。
- フェーズB: evidence 0〜3 バケットで NB の log loss が LR(adjusted)を下回る
  (top-1 は同等以上)。終盤バケットで大きく劣化しないこと。
- 最終的にどちらを採用するか(NB 単体 / evidence 数でブレンド / LR 継続)は
  compare レポートを見て判断する。

## 進捗ログ

### 2026-07-19: フェーズA 完了(診断強化)

`evaluate.py` に evidence バケット評価・reliability テーブル・誤答ダンプ(`--error-dump`)を追加。
判明したこと:

- 全体の誤答3,554件のうち**過半数(1,800件)が evidence 0枚**。混同ペア上位はすべて
  `observed_cards={}` での alakazam 誤予測(=prior をなぞっているだけ)。
- evidence 1枚: 正解率80%(全体)に対し平均確信度77〜91% → 「1枚で断定して見える」を数字で確認。
- **高確信誤答(確信度80%以上で誤答)が410件(誤答の11.5%)**。その際見えていたのは
  Ultra Ball / Lillie's Determination 等の汎用スタッフばかり → 共通カード起因の過信を裏付け。
- evidence 2枚以上はほぼ飽和(96〜100%)。改善の主戦場は evidence 0〜1。

### 2026-07-19: フェーズB 完了(NB 予測器)→ 採用判断は「ブレンド」

`train_nb.py` / `nb_predictor.py`(explain() 付き)/ `compare_nb.py` / ユニットテスト12件を実装。
固定 validation(8,693件)での LR vs NB:

| evidence | LR log loss | NB log loss | 勝敗 |
|---|---:|---:|---|
| 0 | 1.8713 | **1.3849** | NB |
| 1 | 0.1327 | 0.1480 | ほぼ同等 |
| 2-3 | 0.0484 | **0.0431** | NB |
| 4-6 | 0.0187 | 0.0170 | ほぼ同等 |
| 7-10 | 0.0094 | 0.1198 | LR |
| 11+ | 0.0003 | 0.3189 | LR |

- 序盤(evidence 0)は狙いどおり NB が優位。終盤は Naive Bayes の独立性仮定の破れ
  (60枚デッキの相関カード群の尤度を独立に掛け合わせる)で NB が大幅劣化。
- → 3択のうち「**evidence 数バケット別の log-space 幾何ブレンド**」を採用。
  `hybrid_predictor.py` + `fit_hybrid.py`(w を validation でグリッドサーチ、
  エピソード 50/50 分割で安定性チェック)を実装する。
- 課題(フェーズDへ): クラス別デッキ数が極小(toxtricity N=1, gekkouga_ex N=2, oliva_ex N=4,
  takeruraiko_ex N=14 等。N<30 は train_nb.py が警告)。"other" は雑多クラスで
  高 evidence 時のキャリブレーションを悪化させる。

### 2026-07-19: ハイブリッド実装・デプロイ完了

`hybrid_predictor.py`(evidence バケット別 log-space 幾何ブレンド)+ `fit_hybrid.py`(w グリッドサーチ +
エピソード 50/50 安定性チェック)を実装。フィット結果は evidence 0 の w=1.0 のみ安定、
evidence≥1 の各バケットは不安定警告 → **慎重側の判断で evidence≥4 は w=0(LR単体)に手動固定**
(重みJSON の meta.manual_adjustment に記録)。デプロイ設定: 0→w=1.0 / 1→w=0.3 / 2-3→w=0.55 / 4+→w=0。

最終比較(固定 validation 8,693件、`output/nb_compare_report.md`):

| evidence | LR log loss | hybrid log loss | top-1(両者) |
|---|---:|---:|---:|
| 0 | 1.8713 | **1.3849** | 60.33% |
| 1 | 0.1327 | **0.1286** | 97.01% |
| 2-3 | 0.0484 | **0.0382** | 100.00% |
| 4+ | (LRと完全一致) | 同左 | 99.8-100% |
| 全体 | 0.0462 | **0.0356** | 99.09% |

スモークテストで狙いどおりの挙動を確認:
- evidence 0: 確信度 58.5%(alakazam)≈ 実際の正解率 60.3% とほぼ一致(LR は 22% で過小)
- Ultra Ball(汎用)1枚: 20% / 16% / 13% と分散(=正直に「未確定」)
- Alakazam(専用)1枚: 99%(1枚で確信してよいケース)

デプロイ: `deck_predictor_nb.json` / `deck_predictor_hybrid.json` を
`sample_submission/ptcg_ai/opponent_modeling/` に配置。ビュアー(`live_match.py` / `export_replay.py`)を
`HybridDeckPredictor` に切替(API互換なので2行ずつの変更)。ユニットテスト 69 passed。

残タスク → 実装方針は `phase-c-d-implementation.md` に分離(C → D の順で実施):
- フェーズC: 未確定判定・根拠を**再利用可能な共通モジュール**(`prediction_summary.py`)として実装し、
  ビュアーはそれを表示するだけにする(将来エージェント/ML側から同じ基準を使うため)
- フェーズD: 少数クラスのリプレイ収集 → パイプライン再実行 → fit_hybrid 再フィット
  (安定性チェック通過バケットのみ w 更新)

## 関連ファイル

- ランタイム: `sample_submission/ptcg_ai/opponent_modeling/ml_predictor.py`(LR)/ `nb_predictor.py`(NB・新規)
- 学習: `kaggle_replays/deck_predictor/`(train.py, adjust_prior.py, calibrate.py, evaluate.py, train_nb.py・新規, compare_nb.py・新規)
- 既存プラン: `ml-predictor-plan.md`(フェーズ1)、`ml-predictor-phase2-scaling.md`(フェーズ2)
