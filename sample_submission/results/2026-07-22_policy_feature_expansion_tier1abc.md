# PolicyModel 特徴量拡張 Stage1(Tier1a/1b/1c)検証結果 — 不採用

日付: 2026-07-22
関連: [`docs/plans/policy-feature-expansion/`](../docs/plans/policy-feature-expansion/)
(方針書 `design-and-implementation-plan.md` / 実装計画 `stage1-tier1abc-implementation-plan.md`)
メモリ: `project_tier1abc_mirror_negative`

## 結論(TL;DR)

盤面ポケモン identity 埋め込み(Tier1a)・スタジアム identity(Tier1b)・道具/エネ/特性スカラー
(Tier1c)を追加した拡張 PolicyModel は、**統制群比較のミラー対戦で有意に負け、不採用**。
Tier1a 単独でも効かず、**素朴な per-slot カード埋め込みを小型 MLP(隠れ32・同一データ)へ
足す方式そのものが機能しない**と確定した。オフライン一致率の微増(+0.45pt)は実戦へ転移せず、
むしろ悪化した([`project_pimc_prod_validation`] と同型の「オフライン優位が本番非転移」事例)。

**現行 production への影響はゼロ**。既定重み `policy_weights.json`・既定 config・`features.npz` は
すべて無変更で、実装は後方互換(旧重み=Tier 空で現行とビット同一動作)。

## 実験設計(統制群比較 / clean ablation)

tier 特徴の純粋な効果だけを切り分けるため、**同一データ・同一既定レシピ(既定 weight scheme)**で
以下を学習し、tier 特徴の有無だけを変えた:

- **統制群(control)**: tier なし → `policy_weights_base_ctrl.json`
- **Tier1abc**: tier1a+1b+1c → `policy_weights_tier1abc.json`
- **Tier1a 単独**: tier1a のみ → `policy_weights_tier1a.json`

学習データ: Kaggle公開リプレイ全量(train 147,705 / val 17,359 / test 22,626)。
学習/実行 parity は全モデルで self-check PASS(pure_python_forward vs PyTorch、誤差 <1e-14)。

> 注: production 既定重み(`policy_weights.json`, 構成C=技量集中の重み付け)とは重み付けレシピが
> 異なるため直接比較しない。統制群も既定 weight scheme で学習し、tier 効果だけを分離した。

## オフライン結果(test split, top1 一致率)

| モデル | 入力次元(state/opt) | test top1 | option-only ベースライン |
|---|---|---|---|
| 統制群(tier なし) | 166 / 65 | 0.5923 | 0.5313 |
| Tier1a 単独 | 166 / 65 (+盤面12×8 埋め込み) | **0.5913**(−0.10pt) | 0.5314 |
| Tier1abc | 214 / 69 (+盤面12×8+スタジアム8) | **0.5968**(+0.45pt) | 0.5310 |

- option-only ベースラインが 3 者ほぼ同一(0.531)= レシピが正しく揃っている確認。
- Tier1abc はごく僅かに一致率が上がるが、**Tier1a 単独は統制群を下回る**(一致率すら上げない)。

## ミラー対戦結果(head-to-head, 各300試合・先後半々・Wilson 95%CI)

ハーネス: [`league/_diag_tier1abc_head_to_head.py`](../../league/_diag_tier1abc_head_to_head.py)
(両側 `policy_weights_path` 指定可、`config-base=ml_lethal_attackplan_v0only`、seed 40000〜40299、
0 エラー)。相手は両実験とも同一の統制群 `policy_weights_base_ctrl.json`。

| 候補 | 候補勝ち / 統制勝ち | 候補勝率 | 95%CI | 判定 |
|---|---|---|---|---|
| Tier1abc | 129 / 171 | **43.0%** | [37.5%, 48.7%] | **有意に負け**(CI上限<50%) |
| Tier1a 単独 | 133 / 167 | **44.3%** | [38.8%, 49.99%] | 負け(CI上限ほぼ50%、方向性は負け) |

出力 JSON: `league/results/_diag_tier1abc.json` / `league/results/_diag_tier1a.json`。

## 考察

- **Tier1a(盤面identity)が問題の核**。単体でもオフライン・ミラーとも効かず、1b/1c を足すと
  さらに悪化(43.0%)。identity 情報自体に価値がある可能性は残るが、現状の実装方式では逆効果。
- **推定原因**: 入力を 239→395 次元(Tier1abc)へ増やしたのにモデル(隠れ32・同一データ)は据え置き。
  盤面12スロット分のカード埋め込みは学習データに出ない/希少なカードが多く**未学習埋め込みのノイズ**が
  乗る。模倣 top1 には僅かに効いても、実戦では分布シフトで誤差が複利的に効き悪化した。

## 可逆性(本番影響ゼロの担保)

- 無変更: `ptcg_ai/learning/policy_weights.json`(現行最善)、`configs/ml_lethal_attackplan_v0only.json`
  (既定 config)、`kaggle_replays/policy_net/features.npz`。
- 追加(現行に不干渉): `policy_weights_{tier1abc,tier1a,base_ctrl}.json`、`features_tier1abc.npz` 等、
  `configs/ml_lethal_attackplan_tier1abc.json`。
- 後方互換: encoder/policy_model/build_features/train の tiers 対応はデフォルト無効。旧重み
  (`meta.feature_tiers` 未記載)は自己記述により Tier 空へ落ち、現行と完全に同一動作。

## 次アクション候補(未実行)

1. **設計見直しで再挑戦**: 埋め込み次元・pooling・モデル容量(隠れ32→拡大)・未学習カードの正則化を
   見直してから再測。素朴実装が不可と分かったため、ここを変えないと同じ結果になる公算大。
2. **この方向は保留**して別施策へ。実装・可逆性基盤・ミラーハーネスは温存済みで、いつでも再開可能。
