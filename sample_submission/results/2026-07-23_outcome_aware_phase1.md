# Phase 1(案C: Outcome / Advantage-aware Imitation)結果総括

作成日: 2026-07-23
計画: `docs/plans/beyond-bc/phase1-outcome-aware-implementation-plan.md`
設計: `docs/plans/beyond-bc/beyond-behavior-cloning-design.md`
ブランチ: `experiment/beyond-bc-phase1`
control: **M32 = 現行 production**(`policy_weights.json`、features_configC 由来、byte 一致確認済)

---

## 結論(1 行)

**学習時に勝敗/advantage で目的を勝率へ寄せても(C1・C2 とも)、実戦勝率は M32 を上回らなかった。**
C1(outcome-weighted/filtered BC)は明確に悪化、C2(advantage-weighted)は 900 試合で 50.7%(CI 50%跨ぎ)。
**production 据え置き。** 目的関数を学習時にいじる案C は不採用 → 設計書 §3 の **Phase 2(案A: 決定時
Policy shortlist + Value rollout)** へ進む材料とする。

---

## 検証設計(推論不変・weights のみ・hidden 以外固定)

- 入力: `features_configC_outcome.npz`(= 構成C 重み + `won` ラベル)。base 重みは production と同じ構成C。
- 変えたのは sample weight のみ。モデル構造・入力特徴・推論経路・探索は一切不変。
- **join 検証**(コードで確認): policy 決定点 187,690 の **99.94%(187,582)で勝敗が引ける**、
  `(episode_id, player_index) → label` の不整合 **0**(value_positions 由来)。
- **後方互換**: `--outcome-weighting none` + features_configC.npz → production と **全層 byte 一致**
  (train.py 追加が既定パスを壊していないことの証明)。
- **V(s) 数値健全性**: C2 の numpy value forward は ValueModel と **最大誤差 2.22e-16**(フルデータ)。

---

## C1: outcome-weighted / filtered BC(ValueModel 不要)

負け試合の decision 重みを γ 倍(filter は γ=0)。

| 候補 | 学習時 factor | offline test Top-1 | **Stage3 勝率(300)** | 95% CI | 判定 |
|---|---|---|---|---|---|
| M32(control) | — | 0.5806 | — | — | — |
| C1 γ=0.50 | 負×0.5 | 0.5796 | **43.0%**(129) | [37.5, 48.7] | 有意に悪化 |
| C1 γ=0.25 | 負×0.25 | 0.5793 | **45.0%**(135) | [39.5, 50.7] | 効果薄/負 |
| C1 filter(γ=0) | 勝ちのみ | 0.5792 | **42.7%**(128) | [37.2, 48.3] | 有意に悪化 |

→ **C1 は3候補すべて負け越し。** 負け試合の重みを下げると、負け試合中の多くの正しい手(模倣シグナル)
まで捨て、構成Cの上位模倣シグナルを薄めて逆効果。errors 0。

## C2: advantage-weighted BC(AWR、A = won − V(s)、weight×exp(A/β))

| 候補 | factor範囲(mean) | offline test Top-1 | **Stage3 勝率** | 95% CI | 判定 |
|---|---|---|---|---|---|
| M32(control) | — | 0.5806 | — | — | — |
| C2 β=0.5 | [0.14, 7.12](1.387) | 0.5781 | 300: **52.0%** / **900: 50.67%** | 900: **[47.4, 53.9]** | 増試合で五分に回帰・不採用 |
| C2 β=1.0 | [0.38, 2.67](1.086) | 0.5790 | 300: 46.3%(139) | [40.8, 52.0] | 負 |
| C2 β=2.0 | [0.62, 1.63](1.019) | 0.5834 | 300: 50.3%(151) | [44.7, 56.0] | 五分 |

C2 β0.5 の詳細(increase 判定):

| バッチ | 勝率 | 95% CI |
|---|---|---|
| 300(seed 50000) | 52.0%(156/300) | [46.4, 57.6] |
| +600(seed 60000) | 50.0%(300/600) | [46.0, 54.0] |
| **合算 900** | **50.67%(456/900)** | **[47.4, 53.9]** |

→ 300試合の 52.0% は**ノイズ**で、独立な 600試合はちょうど 50.0%、合算 900 で 50.67%。
**CI 下限 47.4 < 50% で採用条件を満たさない。**(300試合スクリーニングの偽陽性を増試合で回避できた例。)

---

## 採用判定

**production 据え置き(M32 = 現行 `policy_weights.json` 変更なし)。C1・C2 とも不採用。**
理由: いずれも実戦勝率で CI 下限 > 50% を満たさず(C1 は有意に悪化、C2 は五分)。

## 解釈(なぜ negative か)

- **勝敗は疎・遅延・ノイジー**な信号。1試合の勝敗は個々の手の良否とほぼ独立(負け試合にも良い手が多い)。
- 構成C(上位パイロット模倣)の重みが既に最良のシグナルで、それを勝敗で薄めると悪化(C1)、
  advantage で正規化しても中立(C2)。**学習時に outcome を混ぜるだけでは P(win) 方向へ有効に動かない。**
- これは「特徴追加(Tier1/3)・容量増(M64/128)・目的関数の学習時変更(案C)」がいずれも
  実戦に転移しなかった、という一連の負の結果に連なる。

## 次(設計書 §3 Phase 2 へ)

学習時の目的関数変更では動かなかった → **決定時に価値を使う案A(Policy shortlist + Value rollout)**
を次に検討する(attack_plan の rollout を top-k policy 候補へ一般化し ValueModel で結果盤面を評価)。
本フェーズでは実装しない。

---

## 可逆性・成果物

- production 推論・agent・既定 weights・config は**未変更**。ソース変更は build_features(`--with-outcome`)と
  train.py(`--outcome-weighting`)の追加のみ、**いずれも既定 OFF で挙動不変**(byte 一致で証明)。
- 不採用のため実験 weights/features/config は記録として残す(昇格も削除もしない)。

| 種別 | パス |
|---|---|
| 実装計画 | `docs/plans/beyond-bc/phase1-outcome-aware-implementation-plan.md` |
| features | `kaggle_replays/policy_net/features_configC_outcome.npz` |
| 実験 weights | `ptcg_ai/learning/policy_weights_c1_{g050,g025,g000}.json` / `policy_weights_c2_{b05,b10,b20}.json` |
| offline 指標 | `kaggle_replays/policy_net/outcome_aware/offline_{c1,c2}_*.json` |
| head-to-head | `league/results/2026-07-23_outcome_m32_vs_{c1,c2}_*_300.json` + `c2_b05_ext600.json` |
