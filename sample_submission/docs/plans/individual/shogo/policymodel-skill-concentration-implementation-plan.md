# 実装計画: PolicyModel改善 — 上位パイロットへの技量集中で天井を61%へ

作成: 2026-07-21
状態: 実装計画（診断フェーズを経て初の「作る」計画）
前提: loss taxonomy でブローアウト負け69.2%が最大敗因と判明→検証で **piloting に約+16ptの伸びしろ**（自チーム44.6% vs リーダーボード≤20位のAlakazamパイロット**61.0%**、[[project_loss_taxonomy]] / [rank-stratified](../../../results/2026-07-21_blowout_valid_rank_stratified.md)）。
対象ブランチ: `experiment/pimc-hidden-info-integration`（または新規 `feature/policy-skill-concentration`）

---

## この計画の目的

現行 PolicyModel は Alakazam を回すプレイヤー**プール全体**を模倣している。順位重みは緩く（1-50位1.5倍〜1001位以下/不明1.0倍）、しかも学習データの順位構成は上位が薄い（下記Step0）。結果としてモデルが学ぶ棋力はプール平均（≈54%層）に引っ張られる。

**目的: 模倣を高技量（リーダーボード上位）パイロットに集中させて再学習し、PolicyModel の棋力の天井をプール平均54%から上位帯の≈61%へ引き上げる。** 期待効果は自チーム44.6%からの改善（現実的目標≈61%、+16pt）。これは実戦最大の敗因（ブローアウト69.2%＝一般的なプレイの質）に、証拠が指した唯一の直接的レバー。

---

## Step0の結果（既に確認済み・本計画の前提）

学習データ `training_data/policy_positions.jsonl.gz`（187,690意思決定点、2,380エピソード）の `rank_at_fetch` 構成:

| 順位帯 | 意思決定点 | 割合 |
|---|---|---|
| 1-20位 | 26,221 | 14.0% |
| 51-200位 | 257 | 0.1% |
| 201-1000位 | 63,601 | 33.9% |
| 1001位以下 | 61,572 | 32.8% |
| 不明 | 36,039 | 19.2% |

- **上位(≤200位)データは約26,478点（14%、335エピソード）あり、集中しても枯渇しない**（小型MLPには十分な規模）。
- 現状の緩い重みでは、上位20位の加重寄与は全体の約19%に留まり、**モデルは52-55%層＋不明層に支配されている**。これが「模倣の天井が54%層」になっている直接原因。
- 留保: ≤200位は335エピソード。特定少数プレイヤーに偏っていないか（単一スタイルへの過適合リスク）はStep1で確認する。

## 意思決定経路の事実（効果が本番に効く条件）

本番 `ml_policy` の判断は PolicyModel＋lethal。**PolicyModelの改善は本番の実際の律速に直接効く**（[[project_pimc_prod_validation]]で「探索bolt-onは律速でない、PolicyModelがPolicyの大半を担う」と確定済み）。ブローアウトが「終盤の詰め」でなく「中盤のプレイの質」の問題（接戦負け5%のみ）である点とも整合。

---

## スコープ

**やること**: `rank_at_fetch` による技量集中（フィルタ／リウェイト）で学習データを作り直し、複数構成で再学習し、(1) 上位パイロット決定との一致率 と (2) 現行 `ml_lethal` とのミラー head-to-head で評価し、候補 `policy_weights.json` を作る。

**やらないこと（Non-goals）**:
- `AGENT_TYPE` 切り替え・実際のKaggle提出 → 候補と証拠を出すところまで。提出判断はユーザー（共有提出物、[[feedback_respect_ownership_boundaries]]）
- PolicyModel の**アーキテクチャ**変更（層構成・特徴量追加）→ 本計画はデータ（技量集中）のレバーに限定。アーキは別軸
- 模倣を超える手法（self-play RL）・デッキ変更 → ≈61%はあくまで模倣＋現デッキの天井。それ以上は別の大投資
- 探索/PIMC・hidden info・山札切れ・足場固め → 対象外（parked/closed）
- 追加のリプレイ取得 → 学習データは既に `rank_at_fetch` を持つため不要

---

## 事前確認済みの契約（Contracts）

### 1. パイプライン

`training_data/policy_positions.jsonl.gz`（行=意思決定点、`observation`/`action`/`episode_id`/`rank_at_fetch`）→ `policy_net/build_features.py`（`weight_for_rank` で順位重み、md5でsplit）→ `features.npz` → `policy_net/train.py` → `policy_weights.json` → `sample_submission/ptcg_ai/learning/policy_model.py`（純Python推論）。

### 2. 変更してよい所／注意

- `build_features.py` docstring に順位重みは「**設計は固定（このファイル単体の都合で変更しないこと。step2-design.md §4）**」とある。→ 既定を黙って上書きせず、**rank フィルタ／リウェイトを引数（`--rank-max` / `--weight-scheme` 等）で追加**し、既定は現行のまま再現可能に保つ。実験は明示フラグで行う。
- 行順を変えない制約（`evaluate.py` が行番号で突き合わせる）は維持。フィルタする場合は features.npz と突き合わせ用の行対応を壊さない設計にする（フィルタ後のインデックス写像を保存する等）。

### 3. 評価

- オフライン: `policy_net/evaluate.py` の Top-1一致率。ただし本計画では**上位(≤200位)の held-out 決定に対する一致率**を主指標にする（プール平均でなく「上位の手をどれだけ再現できたか」を測る。md5 split は同一エピソード→同一splitなので上位エピソードもtrain/val/testに整合分割される）。
- オンライン: `ml_policy_agent.agent(obs, config=...)`（注入点）で候補重みを読ませ、現行 `ml_lethal` とミラー head-to-head（`league/run_match.play_match`、先手後手半々、Wilson CI）。**新policy vs 現policy の相対棋力**の決定的な局所指標。
- 最終確認: 実Kaggle提出（非ミラーの真の勝率）はユーザー判断。ミラーは相対比較には有効だが、対フィールドの絶対値ではない点を明記。

### 4. 基準値

上位≤20位61.0%（359試合）/ ≤200位61.4% / プール54% / 自チーム44.6%。現行 PolicyModel のオフラインTop-1一致率0.59（[[project_ml_value_network_step1]]系、step2）。

---

## ステップ

### Step 1: 技量集中構成の定義と上位データの健全性確認

- ≤200位データ335エピソードの**プレイヤー分布**を確認（単一プレイヤー偏重＝過適合リスクの有無）。偏重が強ければリウェイト側を主に、分散していればフィルタ側も可。
- スイープする構成を確定（例）:
  - **A. ハードフィルタ ≤200位**（約26k点、最も純度が高い「上位の模倣」）
  - **B. ハードフィルタ ≤1000位**（約90k点、55%層まで含め volume 確保）
  - **C. 強リウェイト**（全データ保持、≤20位を5〜10倍・不明を0.3倍等で上位を強調）
- 完了条件: `results/2026-07-21_skillconc_configs.md` に上位データのプレイヤー分布と、スイープ構成の定義。

### Step 2: 各構成でビルド＋学習

- `build_features.py` に rank フィルタ／リウェイトのフラグを追加（既定は現行維持）。各構成で `features.npz` → `train.py` → 候補 `policy_weights_<config>.json` を生成。
- 完了条件: 各構成の学習が完走し、train/val/test の損失・Top-1が記録される（volume を絞る A で過適合/未学習が起きていないかを val/test で確認）。

### Step 3: オフライン評価（上位一致率）

- 各候補と現行を、**≤200位 held-out 決定への Top-1一致率**で比較（プール平均一致率も併記）。「上位の手をより再現する」構成を特定。volume を絞りすぎた構成が val/test で崩れていないかも確認。
- 完了条件: `results/2026-07-21_skillconc_offline.md` に構成別の上位一致率・プール一致率、最良候補の選定。

### Step 4: オンライン評価（現行とのミラー head-to-head）

- 最良候補（1〜2個）を `ml_policy_agent.agent(config=...)` に読ませ、現行 `ml_lethal` と 100〜数百試合のミラー対戦。勝率・95%CI・エラーを記録。
- 完了条件: `results/2026-07-21_skillconc_headtohead.md` に、候補が現行を上回るか（CIが0.5を除外するか）。

### Step 5: 統合と提出候補の判断材料

- Step3/4を集約。候補が「上位一致率↑＋現行にミラーで勝ち越し」なら**提出候補**（`AGENT_TYPE`/提出はユーザー判断）。期待勝率上昇の目安（44.6%→61%天井のどこまで近づくか）を記す。
- どの構成も改善しない場合: 技量集中"データ"だけでは天井が上がらない（volume 上限 or モデル容量）→ 次はアーキ改善 or 追加の上位データ収集 or 受け入れ、を別途検討。
- 完了条件: `results/2026-07-21_skillconc_summary.md` に、提出候補の有無と根拠、次の分岐。

---

## Definition of Done

- [x] 上位≤200位データのプレイヤー分布を確認し、技量集中スイープ構成を定義 → [結果](../../../results/2026-07-21_skillconc_configs.md): 15人、上位2人で60.7%の強い偏重を確認。A(≤200)/B(≤1000)/C(全データ+濃縮リウェイト)を定義
- [x] rank フィルタ／リウェイトのフラグを追加（既定は現行維持・再現可能）し、≥2構成を学習 → `build_features.py`/`train.py`に追加(既定挙動不変、既存テスト・自己検証PASS)。A/B/C 3構成を学習
- [x] 上位(≤200位) held-out への Top-1一致率で候補を評価し、現行と比較 → [結果](../../../results/2026-07-21_skillconc_offline.md): 全構成で改善、Cが最良(+1.49pt)
- [x] 最良候補の現行 `ml_lethal` とのミラー head-to-head を記録 → [結果](../../../results/2026-07-21_skillconc_headtohead.md): C 58.0%(有意)、B 55.5%(有意差なし)
- [x] 提出候補の有無を証拠で判断（`AGENT_TYPE`/提出の変更は本計画では行わない＝ユーザー判断） → [結果](../../../results/2026-07-21_skillconc_summary.md): `policy_weights_configC.json` を提出候補として確定。昇格判断はユーザー
- [x] （明示的にやらないこと: アーキ変更・self-play・デッキ変更・提出・足場固め） → 遵守。本番 `policy_weights.json`・`AGENT_TYPE`は未変更

## 結果

計画完了。詳細は [結果まとめ](../../../results/2026-07-21_skillconc_summary.md) を参照。

技量集中(データのリウェイト)というレバーは機能したが、伸び幅はオフライン一致率で+1.5pt程度と計画の期待(プール平均54%層→上位帯61%相当のジャンプ)より小さかった。ただしミラー対戦では現行に対し58%対42%の有意な勝ち越しに変換されており、`policy_weights_configC.json` を提出候補として確定した。本番重み・`AGENT_TYPE`の切り替えはユーザー判断待ち。
