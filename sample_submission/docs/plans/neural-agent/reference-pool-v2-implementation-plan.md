# 実装計画書: Reference Pool v2 + Gate 2 Contract v2

作成日: 2026-07-26
対応設計: reference-pool-v2-design.md。**本書はレビュー承認後の実装手順。今回コードは書かない。**
配置原則: 実装は `kaggle_replays/measurement/`(v2 manifest/contract/runner 拡張)+ `kaggle_replays/opponent_training/`(strong 参照)の
**未追跡ローカル領域のみ**。**v1 artifact(reference_pool_v1.json / gate2_contract_v1.json / lucario・archaludon・marnie・crustle gate /
各 champion)は不変**。production/cg/shared/opponents/deck.csv/weights/configs 無変更。git add/commit/push なし。

---

## Step 1: Reference Pool v2 manifest builder + frozen manifest
- **file**: `measurement/reference_pool_v2.py`(新規)+ `measurement/reference_pool_v2.json`(生成物, frozen)。
- **responsibility**: v2 の 6 member を組む。各 member に `opponent_strength_class`(strong_champion/dedicated_rule/generic_policy)、
  strong は champion 重み path+hash(opponent_training/champions/ 由来)、generic は generic 重み、dragapult は dragapult_rule。
  recipe path+hash を frozen。meta share は v1 と同一(構成不変)。
- **input**: opponent_training champion metadata(lucario c4ca3599/archaludon 5da2fdd3/marnie f82175a9)、generic 735dd38a、
  archetype_decks recipes、dragapult_ex_deck.csv、v1 meta share。
- **output**: reference_pool_v2.json(6 member、hash 固定)。build/validate/allocate_games(v1 の reference_pool.py を踏襲)。
- **new/existing**: 新規(v1 の reference_pool.py を参照実装として複製・拡張。**v1 は変更しない**)。
- **tests**: `test_reference_pool_v2.py` — 6 member / strength_class 正しい / strong 重み hash 一致 / generic 重み一致 /
  recipe hash 一致 / dragapult_rule 解決 / validate が実 file 再 hash で drift 検出。
- **DoD**: build→validate 空、strong 3 + dedicated 1 + generic 2、hash 記録。
- **production impact**: なし。

## Step 2: opponent_strength_class + Strong subset を集計へ
- **file**: `measurement/field_aggregate.py`(拡張 or v2 用 `field_aggregate_v2.py`)。
- **responsibility**: per-archetype に `opponent_strength_class` を付与。**Strong subset**(strong_champion+dedicated_rule)/
  **Generic coverage**(generic_policy)/ Overall を別集計。meta-weighted Δ_field(Primary、v1 と同一式)に加え **Δ_strong**
  (strong subset の Candidate−Champion、opponent-uniform)を出す。
- **input**: field records(v2 runner)+ v2 manifest(strength_class)。
- **output**: aggregate に strong_subset / generic_coverage / delta_strong を追加。既存 per_stratum/per_archetype/meta_weight/equal_weight は維持。
- **new/existing**: 既存 field_aggregate に純追加(v1 の descriptive 出力は不変)。
- **tests**: strong/generic 分類集計 / Δ_strong 計算 / **不変性**(margin/勝ち筋を変えても winrate/CI/Δ 不変)/ v1 既存テスト green。
- **DoD**: Δ_field(Primary)と Δ_strong(secondary)が別に出る。全既存テスト green。
- **production impact**: なし。

## Step 3: Gate 2 Contract v2 calibration → freeze(**online 前**)【2026-07-26 完了(prebind)】
- **file**: `measurement/gate2_calibration_v2.py`(**作成済**)+ `measurement/gate2_contract_v2_prebind.json`(**frozen, hash 722c37a7**)。
  manifest bind 後に gate2_contract_v2.json へ最終 freeze。
- **完了内容**: v2 baseline 勝率(lucario0.478/archaludon0.550/marnie0.442/dragapult0.240/crustle0.664/rocket0.324)で MC 較正
  (30000reps, seed 20260726, report hash 24c0980d)。N sweep(250-500)/τ sweep(1-3pt)/G sweep(5-15pt)/null/positive/hidden
  catastrophe/strong-generic scenario/optional-extension type-I を評価。
- **確定値**: **N=400 固定**(SE 1.68pt、+5pt PASS 91%/+3pt 56%)、**τ=+2pt**(v1 維持)、**Guard G=−8pt**(v2 再較正=解像度 2.4·SE≈8pt、
  FWER 4.6%、−10pt 検出 77%)、Δ_strong=secondary diagnostic(Option A)、**adaptive extension 禁止**(optional stopping が false-PASS を 5.2→7.1% に膨張)、
  integrity err≤1%、PASS/FAIL/REVIEW は v1 構造。
- **tests(実装フェーズ)**: contract v2 分岐 unit(STRONG→PASS 等)/ calibration engine 一致 / Δ_strong diagnostic 出力。
- **DoD**: **達成(prebind)**。manifest bind は Step1 完了後。
- **production impact**: なし。**注**: online 結果を見てから閾値を変えない(pre-registration。変更は v3 別version)。

## Step 4: field_gate v2(Δ_strong secondary + class 別 report + Contract v2 判定)
- **file**: `measurement/field_gate_v2.py`(v1 field_gate.py を v2 manifest/contract 対応へ)。
- **responsibility**: Candidate/Champion を Reference Pool v2 で走らせ、Primary(meta-weighted Δ_field)+ Δ_strong secondary +
  Strong subset/Generic coverage report。判定は gate2_contract_v2(gate2_contract.py の generic ロジック再利用)。field_outcomes.jsonl
  真実源・resume・contract metadata(v2 hash)。flat single-pool(既存の効率化)を踏襲。
- **input**: v2 manifest、v2 contract、own_deck(deck.csv)、workers。
- **output**: field_report_v2.json(decision PASS/FAIL/REVIEW、Primary、Δ_strong、per-class、per-archetype guard)。
- **new/existing**: 既存 field_gate.py + field_eval flat pool + gate2_contract.py を再利用(strong opponent は policy-only=既存 build_opponent_agent
  が champion 重みを読むだけ。own-deck-safe 確認済)。
- **tests**: `test_field_gate_v2_integration.py`(mock run_stratum、schema/decision/Δ_strong/resume、cg 不要)。
- **DoD**: A/A で PASS 出ない(gate1 None)、Δ_strong 出力、integrity。
- **production impact**: なし。

## Step 5: v2 Current Champion A/A baseline(実装後の最初の正式 run。**本計画では未実行**)
- **file**: `measurement/step_champion_field_baseline_v2.py`(v1 の step_champion_field_baseline.py を v2 manifest/contract へ)。
- **responsibility**: Cand=Champ=abl_5_full を Reference Pool v2 で N 対戦(推奨 N=350-400、Step3 確定値)。null calibration +
  Current Champion vs Strong Field snapshot + throughput + Contract v2 実ゲーム確認。D0(v2 manifest/contract/weights/recipe/cg hash)。
- **input**: v2 manifest+contract(frozen)、deck.csv、workers=15(desktop)。
- **output**: results/champion_field_baseline_v2/(field_report_v2.json / field_outcomes.jsonl / field_manifest.json)。
- **budget/throughput**: 6arch×N×2。N=350→4200 games、N=400→4800 games。throughput ≈ 61 g/min(evaluated=abl_5_full heavy・
  opponent policy-only/rule 軽、v1 field baseline 実測)→ N=350 ~69min / N=400 ~79min。
- **new/existing**: 新規(v1 baseline entry を複製)。
- **tests**: 統合は smoke(小 N)で。integrity は field_integrity 再利用。
- **DoD**: **本計画では設計のみ**。実行はレビュー承認後の別フェーズ。
- **production impact**: なし。

## Step 6: v1/v2 coexistence + Champion registry 拡張
- **file**: `opponent_training/champions/` に strength metadata schema(§設計22)、v2 promotion 判定 metadata。
- **responsibility**: Challenger 昇格 = Gate1 PASS ∧ Gate2 v2 PASS。v1 は optional historical。champion registry に strength_source /
  promotion_evidence / external_strength を持てる形へ。
- **new/existing**: metadata schema のみ(manager 本体は作らない)。
- **DoD**: schema doc 化。league 変換余地を明記。
- **production impact**: なし。

---

## 実装順(推奨)
1. Step 1(v2 manifest)→ 2(strong subset 集計)→ 3(**Contract v2 calibration+freeze、online 前**)→ 4(field_gate v2)→ tests。
2. レビュー: N・τ・guard・メンバー確定。
3. Step 5(A/A v2 baseline 実行)は**別フェーズ・承認後**。

## production / shared / ownership 影響
- **変更なし**: production / cg / main.py / deck.csv / weights / configs / shared league / opponents / v1 全 artifact / 各 champion。
- **新規のみ**: measurement/ の v2 系ファイル(reference_pool_v2.* / gate2_calibration_v2.* / gate2_contract_v2.json / field_*_v2.py /
  step_*_v2.py)+ opponent_training 参照 = 全て未追跡・push 無し。
- **要担当者確認になり得る点**: なし(strong opponent は既存 own-deck-safe な policy-only。production 注入不要)。

## レビューで確定すべき事項(実装前)
設計 §23 Open decisions と同一(v2 メンバー / Crustle・Rocket generic / Primary+Δ_strong / Generic を Primary / N 再較正 /
τ・guard 流用可否 / v1-v2 役割 / eval-training 分離 / 今回やらないこと)。**全確定後に実装着手**。
