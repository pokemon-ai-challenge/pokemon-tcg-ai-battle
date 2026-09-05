# Negative Result 総括: 特徴量追加だけでは実戦勝率が安定して改善しなかった

作成日: 2026-07-23
関連計画: `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`
関連: `docs/plans/policy-feature-expansion/`(Tier1/Tier3 の設計・実装計画)

---

## 要旨(1 行)

Tier1(raw identity 特徴)・Tier3(consequence 特徴)を通して、
**「特徴量を足して offline 模倣 Top-1 を少し上げる → 実戦勝率も上がる」という関係が
安定して成立しなかった。** よって特徴量追加を一旦停止し、次はモデル容量 ablation
(`model-capacity-ablation-implementation-plan.md`)へ移る。

これは失敗の記録ではなく、**「特徴追加は当たりが不安定でコスト効率が悪い」という
重要な negative result** として、結果・コード・検証ハーネスごと保存する。

---

## 現行 production(比較の基準)

- モデル: PolicyModel(hidden=32 小型 MLP、`Linear(239,32)→ReLU→Linear(32,1)`、
  card embedding 8 次元)。
- 重み: `policy_weights.json`(= 構成C = `features_configC.npz` 由来、上位パイロット重み付け)。
- 意思決定: lethal search → attack_plan(`ml_lethal_attackplan_v0only`)。

---

## Tier1: raw identity 特徴(盤面 Pokemon / stadium / tool / energy / ability の identity)

「カード情報を増やせば強くなる」という単純仮説の検証。

| 実験 | offline Top-1 | ミラー勝率 | 判定 |
|---|---|---|---|
| Tier1abc(全部盛り) | 一部構成で微増 | **43.0%** | 不採用 |
| Tier1a 単独 | — | **44.3%** | 不採用 |

→ identity を増やしても実戦はむしろ負け越し。単純仮説は不支持。
詳細メモ: [[project_tier1abc_mirror_negative]]、`results/2026-07-22_policy_feature_expansion_tier1abc.md`。

---

## Tier3: consequence 特徴(「その option を実行すると何が起きるか」)

| 実験 | 特徴 | offline Top-1 | ミラー勝率 | 95% CI | 判定 |
|---|---|---|---|---|---|
| Exp B | opp_hp_loss / opp_energy_removed / opp_special_energy_removed | +0.11pt | 55.0% | [49.3, 60.5] | 境界 |
| Exp C | delta_best_effective_attack_damage / delta_can_ko | +0.21pt | 54.7% | [49.0, 60.2] | 境界 |
| Exp D | delta_attack_ready / delta_energy_shortfall | −0.27pt | 35.0% | [29.8, 40.6] | 悪化・不採用 |
| B+C 統合 | B + C | — | 40.7% | [35.3, 46.3] | 悪化・不採用 |
| delta_best 単独 | delta_best_effective_attack_damage | — | 47.0% | [41.4, 52.6] | フラット |

関連結果ファイル: `results/2026-07-23_tier3_experiment_b_results.md`、
`results/2026-07-23_consequence_dummy_hidden_state_precheck.md`。

---

## 解釈(なぜ negative result なのか)

- **offline↑ が実戦へ転移しない**: Exp B/C は offline Top-1 が +0.1〜0.2pt 改善したが、
  ミラー勝率の 95% CI 下限は 50% を割り、採用条件(CI 下限 > 50%)を満たさなかった。
- **不安定**: 特徴を少し変えるだけで実戦勝率が 55%(B)→ 47%(delta_best 単独)、
  良さそうな B と C を統合すると 40.7% と**むしろ悪化**。proxy 特徴 D は 35% と明確に悪化。
- 「良い特徴を足せば単調に良くなる」という前提が崩れており、特徴探索の当たり判定が
  offline 指標では信用できない。

---

## 次のアクション

1. **特徴量追加を一旦停止する。** これ以上新しい特徴を大量に考えない。
2. 「現在の特徴を hidden=32 の小型 MLP が使い切れていないのでは」という仮説を、
   **モデル容量 ablation(M32/M64/M128、hidden 以外を固定)**で検証する
   (`model-capacity-ablation-implementation-plan.md`)。
3. **容量を上げても改善しなければ、特徴追加には戻らない。**
   「Behavior Cloning 自体がボトルネック(`P(human action|state)` を最適化していて
   `P(win|state,action)` ではない)」という次の仮説へ進む(別フェーズで設計)。

---

## 保存する資産(削除・移動しない)

Tier3 実装は**既定 OFF・後方互換のまま保存**する。将来 search / ValueModel /
action evaluation / offline RL で再利用できるため(production PolicyModel には現時点で不採用、
`meta.consequence_fields` 空が既定)。

| 種別 | パス |
|---|---|
| Tier3 設計・実装計画 | `docs/plans/policy-feature-expansion/tier3-consequence-features-design-and-implementation-plan.md` |
| Tier1 設計・実装計画 | `docs/plans/policy-feature-expansion/stage1-tier1abc-implementation-plan.md` |
| consequence 特徴計算 | `ptcg_ai/board_evaluation/consequence.py` |
| encoder 配線 | `ptcg_ai/learning/encoder.py`(`encode_option_consequence_features`、`CONSEQUENCE_FEATURE_NAMES`) |
| build_features 配線 | `kaggle_replays/policy_net/build_features.py`(`--with-consequence-features`) |
| PolicyModel consequence 対応 | `ptcg_ai/learning/policy_model.py`(`_append_consequence_features`、既定は空で不使用) |
| train 配線 | `kaggle_replays/policy_net/train.py`(`--consequence-fields`) |
| 診断テスト | `tests/unit/test_consequence.py` / `test_encoder_consequence.py` / `test_policy_model_consequence.py` |
| head-to-head ハーネス | `league/_diag_expB_head_to_head.py` / `league/_diag_tier1abc_head_to_head.py` |
| Tier1 結果 | `results/2026-07-22_policy_feature_expansion_tier1abc.md` |
| Tier3 結果 | `results/2026-07-23_tier3_experiment_b_results.md` ほか |
