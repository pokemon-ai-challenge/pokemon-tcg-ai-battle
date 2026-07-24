# 意思決定パイプライン ablation 結果(2026-07-25)

`design-and-implementation-plan.md` の統合パイプラインについて、5構成 ablation を実施した記録。
実行は GPU デスクトップ([[project_gpu_desktop_tailscale]]、Windows/Python 3.11.9、`C:\dev\pokemon-tcg-ai-battle`)。
`league/run_ablation.py` を使用。**すべて同一デッキ(`deck.csv`)のミラー自己対戦**、先手/後手バイアスは
`_play_game` が交互入替で除去。

## 第1弾: 5構成総当たり(各20ゲーム、方向性把握)

| ↓が→に勝つ率 | rule_based | policy_only | policy_eval1 | policy_searchN1 | full |
|---|---|---|---|---|---|
| rule_based | — | 0.25 | 0.10 | 0.15 | 0.05 |
| policy_only | 0.75 | — | 0.65 | 0.45 | 0.45 |
| policy_eval1 | 0.90 | 0.35 | — | 0.30 | 0.30 |
| policy_searchN1 | 0.85 | 0.55 | 0.70 | — | 0.60 |
| full | 0.95 | 0.55 | 0.70 | 0.40 | — |

- 総合順位: policy_searchN1 0.675 > full 0.650 > policy_only 0.575 > policy_eval1 0.463 > rule_based 0.138。
- **policy_eval1(1手読み・ロールアウト無し)は policy_only より弱い**(0.35)。手作り評価を1ステップだけで
  使うと良い Policy 手を悪い評価で上書きしてしまう、という明確なネガティブ。ちゃんとロールアウトする
  searchN1 が効くのと対照的。
- 20ゲームでは CI が±0.2前後で widely 非有意。有意なのは「rule_based が全構成に負ける」のみ。

## 第2弾: 有望3構成の直接対決(各150ゲーム、Wilson 95%CI)

| ↓が→に勝つ率 | policy_only | policy_searchN1 | full |
|---|---|---|---|
| policy_only | — | 0.407 | 0.433 |
| policy_searchN1 | **0.593** | — | 0.393 |
| full | 0.567 | **0.607** | — |

| 対戦 | 勝率 | 95%CI | 判定 |
|---|---|---|---|
| policy_searchN1 → policy_only | **0.593** | [0.513, 0.669] | ✅ 有意 |
| full → policy_searchN1 | **0.607** | [0.527, 0.681] | ✅ 有意 |
| full → policy_only | 0.567 | [0.487, 0.643] | ✖ 非有意(0.5含む) |

総合平均: **full 0.587 > policy_searchN1 0.493 > policy_only 0.420**。

## 結論

1. **探索(パイプライン)は素の Policy を有意に上回る**(searchN1 → policy_only = 0.593、有意)。統合の
   先読みがこの設定では効いている。
2. **belief サンプリング(full: N=8・estimated 相手推定)が最強**(full → searchN1 = 0.607、有意)。
   N=8 決定化+相手デッキ推定が、N=1・公開情報のみの searchN1 を有意に上回る = **belief 層に価値あり**。
   20ゲーム版の「full は searchN1 を上回らない」を 150 ゲームで覆した。
3. full 対 policy_only 直接は非有意(0.567)。full>searchN1>policy_only は各々有意なのに端点だけ非有意
   という弱い非推移性で、n=150 のノイズ範囲。full の「素の Policy に対する」優位はまだ盤石ではない。

## 重要な注意(過大評価しない)

- **同一デッキのミラー自己対戦**であり相手も同じ Policy。「同型対戦で探索/belief が意思決定を改善するか」を
  測ったもので、**Kaggle 実フィールドでの勝率改善は未証明**。[[project_pimc_prod_validation]](PIMC 本番
  非転移)は文脈が別(本番 ml_policy vs ml_lethal)。
- ここの `policy_only` は lethal OFF で、現行本番 config(`ml_lethal_attackplan_v0only` = lethal+attackplan)
  より弱い基準線。昇格判断の本命比較は「**パイプライン config vs 現行本番 config**」。

## 次の一手

`full` を既定 OFF のまま、現行本番 `ml_lethal_attackplan_v0only` との**直接 A/B**(実相手、200試合〜)に
かけるのが本当の判定。そのため `run_league` を **A/B で別 config を注入できるよう拡張**してから実行する
(下記 §実行ログに追記予定)。

## 生成物

- ablation JSON(デスクトップ): `league/results/d_po_sn1.json` / `d_po_full.json` / `d_sn1_full.json`、
  第1弾は `ablation_run1.json`。
- 実行スクリプト: `league/run_ablation.py`(5構成総当たり)、`league/run_league.py`(A/B)。
