# deck_predictor

相手デッキ予測器(ML版)のオフライン学習パイプライン。設計は
[`sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-plan.md`](../../sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-plan.md)(フェーズ1)と
[`sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md`](../../sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md)(フェーズ2、「ベースモデル + チューニング層」構成)
を参照。`kaggle_replays/replays/*.json`(ダウンロード済みリプレイ)から、
「対戦中の任意時点で見えている相手情報 → 相手デッキのアーキタイプ確率分布」を予測する
多クラスロジスティック回帰(softmax回帰)を学習し、ランタイム
(`sample_submission/ptcg_ai/opponent_modeling/ml_predictor.py`)が読む重みJSONを出力する。

学習・推論に使う特徴抽出コードは `sample_submission/ptcg_ai/opponent_modeling/opponent_knowledge.py`
の `OpponentKnowledge` をそのまま再利用している(train/serve skew を構造的に防ぐため)。

## ベースモデル + チューニング層

学習は2段構成になっている(詳細は phase2 doc の「ベースモデル + チューニング層」節):

- **ベース(`train.py`)**: 蓄積済みの全データで多クラスロジスティック回帰を学習する。たまに再学習すればよい。
  学習データのクラス事前分布を `output/model/deck_predictor_weights_base.json` の
  `meta.class_priors` に記録する。**このファイルはそのまま提出には使わない。**
- **チューニング(`adjust_prior.py`)**: ベースの intercept(事前分布)だけを、直近・上位帯の実データに
  合わせてシフト補正する(`b'_c = b_c + log(pi'_c / pi_c)`)。再学習不要・数秒で終わるので頻繁に実行する。
  出力(`output/model/deck_predictor_weights.json`)が提出/デプロイ対象。

## 実行順序

`kaggle_replays/replays/` にリプレイ(`episode-*-replay.json`)がダウンロード済みであること
(`kaggle_replays/fetch_top_episodes.py` 等を先に実行)。カレントディレクトリは
`kaggle_replays/deck_predictor/` を想定(各スクリプトのデフォルトパスはこのディレクトリ基準)。

```bash
cd kaggle_replays/deck_predictor

# 1. 各リプレイの両プレイヤーの60枚デッキリストを抽出する
#    (デッキ選択の返り値 = steps[1][player]["action"]。カード名は data/JP_Card_Data.csv で解決)
python extract_decks.py
#  -> output/deck_db.jsonl

# 2. rough_predictor.json のキーカード定義でアーキタイプを自動ラベリングする
python label_decks.py
#  -> output/deck_labels.jsonl, output/label_report.md (分布レポート)

# 3. 各意思決定時点の「見えている相手情報」を OpponentKnowledge で再現し、特徴量化する
#    (719リプレイ×2視点、進捗表示あり。1件のエラーで全体は止まらない)
python build_dataset.py
#  -> output/dataset.jsonl

# 4. 多クラスロジスティック回帰(softmax回帰)を学習する(ベース)
#    (train/valid はエピソード単位で80/20分割。同一エピソードの2視点は同じ側に入る)
python train.py
#  -> output/model/deck_predictor_weights_base.json (meta.class_priors 付き。まだデプロイしない)
#  -> output/model/split.json (evaluate.py / adjust_prior.py が同じ分割を再利用する)

# 5. 事前分布シフト補正をかけ、デプロイ用の重みを作る(チューニング層)
#    直近 N 日(既定14) x 相手ランク上位 R 位以内(既定200)のデッキラベル分布に intercept を合わせる。
#    --deploy を付けると sample_submission/ptcg_ai/opponent_modeling/deck_predictor_weights.json にもコピーする。
python adjust_prior.py --deploy
#  -> output/model/deck_predictor_weights.json
#  -> (--deploy時) sample_submission/ptcg_ai/opponent_modeling/deck_predictor_weights.json

# 6. evidence数(観測ユニークカード種類数)バケット別に温度スケーリングでキャリブレーションする
#    (T(温度)は1.0を下回らない制約付き。過信の抑制のみ行い、確信度を強気にする方向には補正しない)
#    --deploy を付けると sample_submission/ptcg_ai/opponent_modeling/deck_predictor_weights.json を上書きする。
python calibrate.py --deploy
#  -> output/model/deck_predictor_weights.json (calibration.buckets 追加)

# 7. validation セットで評価する(ランタイムと同じ MLDeckPredictor で推論)
#    既定で「固定validation」(直近--valid-recent-days日 x 相手上位--valid-top-rank位以内)に絞って評価する。
#    --all を付けると絞り込みなしの従来の全体評価もあわせて出す。base(補正前) vs adjusted(補正後) の比較も含む。
python evaluate.py --all
#  -> output/eval_report.md (ターン別/ランク帯別/期間別/evidenceバケット別 top-1 正解率・log loss・
#     ECE・reliabilityテーブル・クラス別 precision/recall・base vs adjusted)
#     --error-dump を付けると誤答サンプルを output/errors.jsonl + output/error_summary.md に出力する
```

### 序盤過信の改善(生成的ベイズ + LR×NBハイブリッド)

LR単体は evidence(観測ユニークカード種類数)が少ない場面で過信しやすい(evidence 0 で
正解率33%なのに確信度90%超、等)。これを補うため、`train.py`〜`calibrate.py`のLRパイプラインとは
別に、生成的ベイズ(NB)を学習してLRとブレンドする。方針の詳細・診断結果は
[`early-confidence-improvement-plan.md`](../../sample_submission/docs/plans/opponent-deck-predictor/early-confidence-improvement-plan.md)
を参照。ランタイムは
[`nb_predictor.py`](../../sample_submission/ptcg_ai/opponent_modeling/nb_predictor.py) /
[`hybrid_predictor.py`](../../sample_submission/ptcg_ai/opponent_modeling/hybrid_predictor.py)。

上記の手順1〜3(`extract_decks.py` → `label_decks.py` → `build_dataset.py`)と、LRパイプラインの
`output/model/deck_predictor_weights.json` / `output/model/split.json`(手順4・6)が先に必要。

```bash
cd kaggle_replays/deck_predictor

# 8. 生成的ベイズ(NB)を学習する。P(≥k枚 | アーキタイプ) をラプラス平滑化で頻度推定する
python train_nb.py
#  -> output/model/deck_predictor_nb.json

# 9. LR(手順6のdeck_predictor_weights.json) と NB(手順8) を比較する
python compare_nb.py
#  -> output/nb_compare_report.md (evidenceバケット別 top-1/log loss の LR vs NB 比較)

# 10. evidenceバケット別のブレンド重み w を validation でグリッドサーチしてフィットする
#     50/50分割で安定性チェックする。--baseline に現行デプロイ済みhybrid.jsonを渡すと、
#     安定して改善したバケットだけ新しいwを採用し、不安定なバケットは既存wを維持する(慎重側)。
#     --deploy を付けると sample_submission/ptcg_ai/opponent_modeling/deck_predictor_hybrid.json にコピーする。
python fit_hybrid.py --baseline ../../sample_submission/ptcg_ai/opponent_modeling/deck_predictor_hybrid.json --deploy
#  -> output/model/deck_predictor_hybrid.json

# 11. LR単体 vs ハイブリッドを比較する(compare_nb.py が hybrid.json の有無を自動検出して列を追加する)
python compare_nb.py
```

NBの重みJSON(`deck_predictor_nb.json`)を提出環境にも配置する場合は、`train_nb.py`の出力
(`output/model/deck_predictor_nb.json`)を`sample_submission/ptcg_ai/opponent_modeling/`へ
手動でコピーする(`train_nb.py`には`--deploy`は無い。LRの`weights.json`と違い頻度が低い想定のため)。

**注意**: `--recent-days` / `--valid-recent-days` の既定値(14日)は取得済みデータの日付範囲より狭くなりうる。
取得したリプレイが数日分しかない場合、ウィンドウが0件になることがある(`adjust_prior.py` は警告を出し、
`evaluate.py` の「固定 validation」も0件になりうる)。その場合は `--recent-days` / `--valid-recent-days` や
`--top-rank` / `--valid-top-rank` を緩めて実行する。

## 各スクリプトの役割

| スクリプト | 入力 | 出力 |
|---|---|---|
| `extract_decks.py` | `replays/*.json` + `index/episodes_master.jsonl` | `output/deck_db.jsonl` |
| `label_decks.py` | `output/deck_db.jsonl` + `rough_predictor.json` | `output/deck_labels.jsonl`, `output/label_report.md` |
| `build_dataset.py` | `replays/*.json` + `output/deck_labels.jsonl` | `output/dataset.jsonl` |
| `train.py` | `output/dataset.jsonl` + `output/deck_labels.jsonl` | `output/model/deck_predictor_weights_base.json`(class_priors付き), `output/model/split.json` |
| `adjust_prior.py` | `output/model/deck_predictor_weights_base.json` + `output/deck_labels.jsonl` + `index/episodes_master.jsonl` | `output/model/deck_predictor_weights.json`(デプロイ用)、`--deploy`時はランタイムへのコピー |
| `calibrate.py` | `output/model/deck_predictor_weights.json` + `output/dataset.jsonl` + `output/model/split.json` | `output/model/deck_predictor_weights.json`(evidenceバケット別温度スケーリング`calibration.buckets`追加、上書き)、`--deploy`時はランタイムへのコピー |
| `evaluate.py` | `output/dataset.jsonl` + `output/model/*` + `index/episodes_master.jsonl` | `output/eval_report.md`(`--error-dump`時は`output/errors.jsonl`/`output/error_summary.md`も) |
| `train_nb.py` | `output/deck_db.jsonl` + `output/deck_labels.jsonl` | `output/model/deck_predictor_nb.json` |
| `fit_hybrid.py` | `output/dataset.jsonl` + `output/model/deck_predictor_weights.json` + `output/model/deck_predictor_nb.json` + `output/model/split.json` | `output/model/deck_predictor_hybrid.json`(デプロイ用)、`--deploy`時はランタイムへのコピー |
| `compare_nb.py` | `output/dataset.jsonl` + `output/model/deck_predictor_weights.json` + `output/model/deck_predictor_nb.json` (+ 存在すれば `deck_predictor_hybrid.json`) | `output/nb_compare_report.md` |

`episode_window.py` は `adjust_prior.py` と `evaluate.py` が共有する、`episodes_master.jsonl` との
ジョイン(エピソード作成日時・相手ランク)とウィンドウ判定のユーティリティ(単独では実行しない)。

## ラベリングしきい値の調整

`label_decks.py` の `LABEL_MIN_SCORE`(既定 10.0)と `_GATE_ROLES` がラベリングの挙動を決める。

- `_GATE_ROLES`: そのアーキタイプの主軸級カード(anchor / exclusive_core / shared_anchor /
  signature / strong_evolution_line のいずれか)が実際にデッキへ1枚も入っていない限り、
  そのアーキタイプの候補にしない、というゲート。フルデッキ(60枚全部)が見えている前提では
  主軸カードの有無がほぼ決定的な証拠になる。このゲートを外すと、`core`/`flex`/`energy` などの
  汎用カード(ふしぎなアメ・シェイミ・ノコッチ等、複数デッキで共通に採用される)の積み上げだけで
  無関係なアーキタイプに誤判定してしまう(実装中に実際に踏んだ不具合)。
- `LABEL_MIN_SCORE`: ゲート通過後の最終しきい値。「anchor 1枚(role_weights["anchor"]=10点)で
  ほぼ確定」という感覚の初期値。`python label_decks.py --min-score <値>` で上書きできる。

`--min-score` を変えて再ラベリングしたら、`build_dataset.py` 以降を再実行すること
(ラベルが変わればデータセットも学習も作り直しになる)。

## 出力(`output/`)について

`output/` 配下は再生成可能な派生物のため、リポジトリには含めない(`.gitignore` 参照)。
