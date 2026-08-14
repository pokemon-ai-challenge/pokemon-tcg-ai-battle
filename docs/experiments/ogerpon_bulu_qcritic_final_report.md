# オーガポン(カプ・ブルル)Q-critic 最終レポート

## 結論

```text
The Bulu Q-critic is not approved for active use.
The current feature/model approach cannot identify positive
SINGLE_PRIZE_ROTATION states reliably.
```

本開発は **shadow-only の状態で終了**する。本番config(`sample_submission/configs/abl_5_full.json`)は
`ogerpon_q_critic.enabled=false` に固定し、推論・ログ記録・OptionState起動のいずれも本番では発生しない。
再開する場合は本レポート末尾の「再開する場合に必要な変更」を参照すること。

---

## 1. 実験目的

「カプ・ブルル中継戦略(SINGLE_PRIZE_ROTATION)」が、通常のテンポ重視プレイ(EX_TEMPO)より
戦略的に優れている局面を、盤面・オプション特徴量から学習したQ-critic(MLPアンサンブル)で
識別できるかを検証する。識別できるなら、strict override(`lcb_delta >= 0.02`)条件を満たした
局面でのみSINGLE_PRIZE_ROTATIONへ本番行動を切り替える設計(design.md §11.2)。

複数フェーズの診断([Diagnostic Phase 1-5](#7-model-abd-の結果)、Model B/D ablation)で
strict override効果が確認できなかったため、最終段階として
**label-fidelity study**(教師ラベル自体の信頼性を、決定化数を4→64へ増やして直接検証する)
を実施した。

## 2. 対象commit

| 対象 | commit |
|---|---|
| 本番重み(4,214-state, shadow_candidate)を生成したcommit | `fbb3c33` |
| 診断ablationコード追加 | `067b79c` |
| label-fidelity study分析コード | `f3e3f7f` |
| 本レポート・config無効化・確認テスト | (本コミット、下記「関連commit SHA」参照) |

## 3. Dataset hash

| データセット | 件数 | sha256 |
|---|---|---|
| 4,214-state訓練データ(`ogerpon_stage3_4200_dataset.jsonl`) | 4,214状態 / 33,712 records | `9a779feaa59205db73c60b13c92157cc719243c4aaefeb0af505f9664f13cf63` |
| label-fidelity Part1(`ogerpon_labelfidelity.jsonl`) | 164ゲーム | `f991067c12270677e1101f47eb831763387ae4e3646baa9c75a76a375fe14fc4` |
| label-fidelity Part2(`ogerpon_labelfidelity_part2.jsonl`) | 106ゲーム | `4f83a127589cbaeaf16948dc6a07aa66085ae306d8ae5ac34a76865c10584b22` |
| label-fidelity 統合(`ogerpon_labelfidelity_combined.jsonl`) | 270ゲーム / 478状態 / 61,184 records | `8d0dcd55c0f6f77f13baa31d3ed84e58b4787fe8c2a90a34c14dc0877a701995` |

生データ(jsonl)は既存方針により git 管理外(`kaggle_replays/rl/runs/` はローカルのみ)。
再現には下記「実行コマンド」節のコマンドをそのまま使用する。

## 4. Seed範囲

| データ | seed範囲 | ゲーム数 | 決定化数 |
|---|---|---|---|
| 4,214-state訓練データ(2,188-state分) | 20260814–20262413 | — | 4 |
| 4,214-state訓練データ(追加2,026-state分) | 20262414–20263913 | — | 4 |
| 941-state held-out dev set(v1/v2/v3統合) | 30000000 / 31000000 / 32000000 | 150+150+400 | 16 |
| label-fidelity Part1 | 40000000–40000269 | 270(うち164完了時点でユーザー指示により一時停止) | 64 |
| label-fidelity Part2(再開、13 workers) | 40001000–40001105 | 106 | 64 |

## 5. 4,214-state 学習データの構成

- `ogerpon_stage3_2000_dataset.jsonl`: 2,188状態(seed 20260814–20262413)
- `ogerpon_stage3_extra2.jsonl`: 2,026状態(seed 20262414–20263913、既存2,188件との重複無し)
- 上記2ファイルを連結した `ogerpon_stage3_4200_dataset.jsonl`(4,214状態、33,712 records)が学習に使用された最終データ
- 決定化数: 4(EX_TEMPO / SINGLE_PRIZE_ROTATIONの各ペアにつき4本ずつ、paired determinization)
- train/val/test分割: 5,906 / 1,246 / 1,276 examples
- データ品質: n_errors=9(error_rate 0.027%)、不正option名0、NaN/Inf 0、incomplete pairs 0

## 6. 475-state・64-determinization検証データの構成

label-fidelity study用に、4,214-state訓練データとは**完全に独立した新規seed**(40000000系列)
から新規収集。EX_TEMPO/SINGLE_PRIZE_ROTATIONの同一hidden-state仮説から、ペアごとに
**64本の決定化**(訓練データの4本の16倍)を実行し、N=4/8/16/32/64のnested prefixで
ラベルの収束性を直接測定した。

- 総ゲーム数: 270(Part1 164 + Part2 106)
- 収集状態数(生): 478 / 全64ペア完備の状態数: **475**(3状態は`max_rollout_steps`タイムアウトで一部決定化が欠落し解析対象外)
- 品質ゲート: engine error 0、illegal actions 0、incomplete pairs 0、hidden-state hash mismatch(pair内) 0、
  benign `max_rollout_steps` timeout 3件(61,184 recordsの0.005%)
- Model A(現行本番重み、`fbb3c33`)によるpredicted_delta層別: negative 189 / near_zero 180 / positive 106
  (うちstrict subset `lcb_delta>=0.02`: 31)

## 7. 総rollout数

- 4,214-state訓練データ: 33,712 records(n_pairs 16,856)
- label-fidelity検証データ: 61,184 records(Part1 41,600 / Part2 18,816+一部)、うちcomplete pairs 30,592
- 累計(本レポート対象実験のみ): 約 **94,896 rollout records**

## 8. N=4/8/16/32/64 の tie rate

| N | tie rate |
|---|---|
| 4 | 55.6% |
| 8 | 44.4% |
| 16 | 33.9% |
| 32 | 22.3% |
| 64 | 16.6% |

N=4(訓練データが実際に使用した決定化数)では過半数が完全な引き分け(win_rate差ゼロ)であり、
`pairwise_ranking_min_gap=0.02` によるペア除外の大半はここに由来する(小さいが実在する差の
除外ではなく、統計的な引き分けの除外)。

## 9. N=64との符号一致率・相関・MAE

| N | sign agreement vs N=64 | Pearson vs N=64 | Spearman vs N=64 | MAE vs N=64 |
|---|---|---|---|---|
| 4 | 68.2% | 0.509 | 0.387 | 0.1532 |
| 8 | 69.7% | 0.638 | 0.473 | 0.1091 |
| 16 | 77.1% | 0.785 | 0.685 | 0.0722 |
| 32 | 86.3% | 0.905 | 0.832 | 0.0426 |
| 64 | 100%* | 1.0* | 1.0* | 0* |

*N=64はN=64自身との比較のため恒等的に1.0/0になる(参考値)。

符号反転率(隣接N間): 4→8: 27.4%、8→16: 29.7%、16→32: 25.1%、32→64: 20.4%。
**どのNでも「安定」せず**、N=32→64でも約2割が反転し続ける。

## 10. confident positive/negative/ambiguous件数

N=64の64個の paired outcome(∈{-1,0,+1})から、match非依存の90% bootstrap CI(3,000リサンプル)で
平均paired outcomeを推定し分類(全475状態):

| 区分 | 件数 | 割合 |
|---|---|---|
| confident_positive(CI下限>0) | 59 | 12.4% |
| confident_negative(CI上限<0) | 75 | 15.8% |
| ambiguous(CIが0を含む) | 341 | 71.8% |

N=64という最高精度のラベルでも**7割超が依然ambiguous**である。

## 11. strict subsetのprecision

Model Aのstrict override条件(`lcb_delta>=0.02`)を満たす31状態のうち:

- confident_positiveと一致: **7/31 = 22.6%**(precision)
- confident_negative(符号が逆): 6/31 = 19.4%
- ambiguous: 18/31 = 58.1%

参考: strict条件を課さない「positive予測」bucket全体(n=106)でのprecisionは **14.15%**(15/106)。
真にconfident_positiveな59状態のModel A bucket割り当ては negative 24 / near_zero 20 / positive 15
であり、bucketの基準率(39.8%/37.9%/22.3%)とほぼ一致——**ランダムな割り当てと統計的に区別できない**。

## 12. strict subsetの平均actual delta

strict subset(n=31)の N=64 実測平均paired outcome delta = **0.0005**(統計的にゼロ)。
positive予測bucket全体(n=106)の平均実測delta = **−0.0071**(符号すら逆転)。
Model Aのpredicted_deltaとN=64実測deltaの全体相関: Pearson **0.24**、Spearman **0.13**。

## 13. Model A/B/D の結果

- **Model A**(既定、win_bce選択・pairwise ranking loss、4,214-state): active gate不合格
  (下記14節参照)。今回のlabel-fidelity studyでも strict subset precision 22.6%、平均実測delta≈0を確認。
- **Model B**(direct delta head + Huber loss、delta_loss_weight ∈ {0.15, 0.5}、win_bce/combined選択):
  delta_loss_weightを上げるほどモデルがtied/noisyな訓練ラベルへ収縮し、strict override候補が
  消滅(n=0、判定不能)するか、Model Aよりprecision/平均deltaが悪化。改善効果は確認できず。
- **Model D**(match-level bootstrap ensemble、3シードをmatch単位でリサンプリング):
  `delta_std` と絶対誤差の相関はやや改善(0.13→0.21)したが、`delta_std` と符号誤りの相関は
  ほぼゼロのまま(−0.017→−0.043)。strict subset(n=8)のprecisionはModel Aより悪化(25.0%)。
  改善効果は確認できず。
- Model C(direct delta head を主力とする設計、Phase5相当)は、Phase4でHypothesis Bを
  採用したため**未着手**。

いずれのablationも「訓練ラベルの表現方法・アンサンブル手法を変える」アプローチであり、
今回のlabel-fidelity studyは「訓練ラベルの精度(決定化数)を上げても、真の効果と
モデル出力の相関は弱いままである」ことを、独立した高精度検証データで直接示した点が新しい。

## 14. active gate不合格の理由

本番重み(`ogerpon_strategy_weights.json`, commit `fbb3c33`)の `meta.active_gate`:

```json
{
  "required_strict_override_states": 30,
  "observed_strict_override_states": 46,
  "sample_size_requirement_met": true,
  "passed": false,
  "reason": "strict_override_effect_not_confirmed",
  "detail": "n=46 held-out strict-override states (>=30 required). Sign agreement / precision = 45.7% (at/below chance). Mean actual paired win-rate delta = -0.0175 (need positive). 90% bootstrap CI = [-0.0581, 0.0217] (crosses zero)."
}
```

n=7の小サンプルでは85.7% precision・+0.152 mean deltaと好成績に見えたが、n=46まで拡大すると
再現しなかった——これが≥30サンプル要件を設けた理由そのものである。今回のlabel-fidelity study
(N=64、独立した475状態)は、この不合格判定を**別の切り口(教師ラベル精度そのものの検証)から
追認**する結果となった。

## 15. Hypothesis B を採用した根拠

design.mdで事前定義された2つの仮説:

- **Hypothesis A**(訓練ラベルのノイズが主因): より高精度なラベルで再学習すれば改善する
- **Hypothesis B**(戦略的効果がほぼ識別不能): 現在の特徴量・モデルでは識別できない

**採用: Hypothesis B**

判定の決め手は「Nを増やすとラベル推定のばらつきが縮む(自明な統計的事実)」ではなく、
「**Model Aの予測がN=64の高精度ラベルと相関するか**」である。相関しない:

- strict subset(モデルが最も自信を持つ主張)の実測平均delta ≈ 0(0.0005)、precision 22.6%
- positive予測bucket全体の実測平均delta が **負**(−0.0071、符号すら逆)
- 全体相関 Pearson 0.24 / Spearman 0.13 と弱い
- 真にconfident_positiveな状態へのbucket割り当てが基準率とほぼ一致(ランダムと区別不能)

一方で、Hypothesis Bを完全な「効果ゼロ」と断定しない理由: confident_positive + confident_negative
= 134/475(28.2%)が実在し、効果自体が完全に架空というわけではない。ただし**その効果がどの状態に
存在するかを、現在の特徴量・モデルクラスでは特定できていない**。

## 16. 今後再開する場合に必要な変更

再開するなら、以下のいずれか(または複数)が必要:

1. **特徴量の再設計**: 現在のoption/state特徴量がQ(s, option)の真の差を説明できていない
   可能性が高い。盤面のどの要素が真に効いているかをこの実験からは特定できていないため、
   まず「confident_positive vs confident_negative」の475状態を人手/統計的に比較し、
   識別に効きそうな新特徴量の仮説を立てる必要がある。
2. **N=64以上の決定化**: N=32→64でも符号反転率が約2割残るため、N=64自体が「収束済み」の
   保証にはならない。より高いNでの再検証が前提条件になりうる(ただし本studyは64を
   上限として設計されており、これを超える根拠は現時点で無い)。
3. **モデルクラスの変更**: MLPアンサンブル以外(例: 木構造の相互作用を捉えるモデル)を
   試す余地はあるが、Pearson 0.24程度の相関では、モデルクラスを変えるだけで
   `active_gate`(precision明確に>50%)を満たせる保証はない。
4. いずれの再開でも、`sample_submission/configs/ogerpon_q_shadow_research.json`
   (`enabled=true, shadow_only=true`)を使い、必ずshadow-onlyで再検証してから
   active化を検討すること。本番config(`abl_5_full.json`)を直接変更しないこと。

## 17. 実行コマンド

```bash
# label-fidelity Part1(270ゲーム目標、164ゲームでユーザー指示により一時停止)
python3 kaggle_replays/rl/collect_ogerpon_counterfactuals.py \
  --deck sample_submission/models/gen2_candidates/deck_ogerpon_teal_ex_01.csv \
  --weights sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json \
  --games 270 --determinizations 64 --workers 16 --seed 40000000 \
  --output kaggle_replays/rl/runs/ogerpon_labelfidelity.jsonl

# label-fidelity Part2(再開、13 workers = 前回の約80%)
python3 kaggle_replays/rl/collect_ogerpon_counterfactuals.py \
  --deck sample_submission/models/gen2_candidates/deck_ogerpon_teal_ex_01.csv \
  --weights sample_submission/ptcg_ai/learning/policy_weights_ogerpon_teal_ex_rl_mixogerpon.json \
  --games 106 --determinizations 64 --workers 13 --seed 40001000 \
  --output kaggle_replays/rl/runs/ogerpon_labelfidelity_part2.jsonl

# 統合
cat kaggle_replays/rl/runs/ogerpon_labelfidelity.jsonl \
    kaggle_replays/rl/runs/ogerpon_labelfidelity_part2.jsonl \
    > kaggle_replays/rl/runs/ogerpon_labelfidelity_combined.jsonl

# nested-prefix分析(Phase2/3の主要指標)
python3 kaggle_replays/rl/analyze_label_fidelity.py \
  --data kaggle_replays/rl/runs/ogerpon_labelfidelity_combined.jsonl \
  --model-a-weights sample_submission/ptcg_ai/learning/ogerpon_strategy_weights.json \
  --output kaggle_replays/rl/runs/labelfidelity_full_analysis.json
```

## 18. テスト結果

| スイート | pass | fail | skip |
|---|---|---|---|
| `sample_submission/tests`(config無効化テスト9件含む) | 527 | 0 | 13 |
| `kaggle_replays/rl/tests` | 80 | 0 | 0 |

新規追加した本番config無効化検証テスト(`sample_submission/tests/unit/test_ml_policy_agent.py`、
全てpass、上記527件に含まれる):

- `test_production_config_has_ogerpon_q_critic_disabled`
- `test_production_config_ogerpon_q_shadow_never_loads_model_or_detects_trigger`(推論回数0)
- `test_production_config_ogerpon_option_continue_never_starts_option_state`(OptionState開始0)
- `test_production_config_agent_output_matches_baseline_even_if_q_critic_would_override`(行動差0)
- `test_production_config_non_ogerpon_deck_unaffected`(非オーガポンデッキ影響0)
- `test_production_config_lethal_path_unaffected_by_q_critic`(lethal経路影響0)
- `test_production_config_works_without_weights_file`(weights未配置でも動作)
- `test_production_config_overhead_is_negligible`(動作時間の粗いスモークガード)
- `test_research_shadow_config_agent_output_matches_baseline`(研究用configでも行動差0)

既存のlethal優先テスト(`test_phase4_active_override_never_replaces_lethal_action`,
`test_ogerpon_q_shadow_is_lethal_suppresses_override_and_records_it`)、
persistent Option Controller単体テスト(`test_persistent_option_controller_full_rotation_lifecycle`,
`test_persistent_option_controller_abort_returns_to_normal_policy`,
`test_persistent_option_controller_lethal_interrupts_mid_rotation`,
`test_persistent_option_controller_resets_on_new_match`)は全てpass継続。

## 19. 関連commit SHA

| commit | 内容 |
|---|---|
| `b88ab08` | Phase1: 固定Option Controller追加 |
| `6264df6` | Phase2: Strategy Window Builder・反実仮想rollout収集器 |
| `5828b5d` | Phase3: encoder・純Python推論・学習パイプライン |
| `1613b6f` | lethal最優先修正・Phase3/4のshadow/active gate配線 |
| `1f1874c` | 永続Option Controller実戦配線 |
| `a06d54e` | shadow candidate 2,188-state版 |
| `fbb3c33` | shadow candidate 4,214-state版(現行本番重み) |
| `067b79c` | 診断ablation(Model B/D)コード追加 |
| `f3e3f7f` | label-fidelity study分析コード追加、Hypothesis B採用 |

---

*本レポート作成日: 2026-08-15*
