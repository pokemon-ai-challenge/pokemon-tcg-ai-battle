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

## 第3弾: 本命 A/B — 現行本番 vs full(200ゲーム、ミラー、エラー0)

`run_league` を A/B 別 config 対応に拡張(`--config-base-a`/`--config-base-b`)して実行:

```
python league/run_league.py --agent-a ml_policy --agent-b ml_policy \
  --config-base-a ml_lethal_attackplan_v0only --config-base-b abl_5_full \
  --games 200 --workers 6 --out league/results/ab_prod_vs_full.json
```

| | 勝率 | Wilson 95%CI |
|---|---|---|
| A = `ml_lethal_attackplan_v0only`(現行本番) | 0.490 | [0.422, 0.559] |
| B = `abl_5_full`(パイプライン full) | 0.510 | [0.441, 0.578] |

先手/後手内訳(A視点): 先手 0.570 / 後手 0.410。平均12.5ターン。

### 判定: 有意差なし → 昇格しない

**full は現行本番 config を上回らなかった**(CI が 0.5 を含む)。ablation では full が素の `policy_only` を
有意に上回り belief 層の価値も示せたが、**すでに lethal+attackplan を備えた本番 config が相手だと測定可能な
優位は消える**。[[project_pimc_prod_validation]](PIMC 本番非転移)と完全に整合。n=200 で CI 半幅±0.07 なので
7% を超える優位ならほぼ検出できたはず。**パイプラインは既定 OFF のまま維持**が妥当。

## 第4弾: 多様フィールド対戦(ミラーの限界を埋める controlled 比較)

`league/field_gauntlet.py`。alakazam(フーディン=現提出インカンベント、weights=構成C)を config だけ
full/v0only で振り、メタ7アーキ(各 deck+専用重み、config は v0only 固定)相手に対戦(100G/ペア)。
full と v0only で同じデッキ・同じ重み・同じ相手・同じ seed にそろえ、違いを意思決定パイプライン
有無だけに絞った。フィールド定義は `round_robin.ARCHS` を再利用。

| 相手 | full | v0only | Δ | share |
|---|---|---|---|---|
| mega_lucario_ex | 78.0 | 82.0 | −4.0 | 1257 |
| archaludon_ex | 79.0 | 78.0 | +1.0 | 1078 |
| crustle | 40.0 | 43.0 | −3.0 | 737 |
| dragapult_ex | 85.0 | 83.0 | +2.0 | 625 |
| marnie_grimmsnarl_ex | 56.0 | 56.0 | 0.0 | 591 |
| rocket_mewtwo_ex | 34.0 | 39.0 | −5.0 | 247 |
| shirona_garchomp_ex | 75.0 | 71.0 | +4.0 | 181 |

**対フィールド期待勝率(メタシェア加重): full 68.0% / v0only 69.2%(full −1.1pt)。**
全field集計 full 63.9% CI[60.2,67.3] vs v0only 64.6% CI[61.0,68.0] = **有意差なし**。full が有意に勝つ
相手は無し。ミラーの限界を埋めた実フィールド寄りの読みでも full は本番を上回らない。

## 第5弾: 実 Kaggle 提出(full)の収束スコアと相手アーキタイプ別 W-L

full を Kaggle 提出(ref 54956037)。**このコンペはレーティング型で publicScore は提出直後の
初期値から時間をかけて収束する**(投入直後 600.0 → 収束 **702.8**)。本番相当(構成C+リーサル+
attack_plan, ref 54883922)の 717.2 に肉薄(−14.4)。決定的な劣後ではないが上回りもせず、ローカル
gauntlet(full −1.1pt)とも整合。

実対戦46試合を相手アーキタイプ別に集計(`kaggle_replays/_archetype_winloss.py`。自エージェントの
観測カードを production `rough_predictor.predict()` に通し、全ステップで最も証拠の多い判定を採用。
勝敗は replay の rewards)。**総合 22-24(勝率 0.478)**:

| 相手 | W-L | 勝率 | 試合 |
|---|---|---|---|
| unknown(off-meta/その他) | 6-3 | .67 | 9 |
| mega_lucario_ex | 4-5 | .44 | 9 |
| **alakazam(フーディン・ミラー)** | **2-6** | **.25** | 8 |
| archaludon_ex | 3-3 | .50 | 6 |
| crustle | 3-2 | .60 | 5 |
| marnie_grimmsnarl_ex | 1-3 | .25 | 4 |
| shirona_garchomp_ex | 1-1 | .50 | 2 |
| mega_abomasnow_ex | 1-0 | 1.0 | 1 |
| mega_starmie_ex | 0-1 | .00 | 1 |
| dragapult_ex | 1-0 | 1.0 | 1 |

- **最大の弱点はフーディン・ミラー(alakazam)2-6(25%)**。ローカル gauntlet は同型を除外していた
  ため新情報。次いで marnie 1-3、mega_lucario 4-5 も負け越し。
- 注意: 46試合の小標本で各アーキ別は n=1〜9 とノイズ大(傾向の目安)。unknown 9件は主軸カード
  不一致=off-meta/その他デッキで分類不能(バグではない)。
- 生データ: `kaggle_replays/_archetype_winloss_54956037.json`。

## まとめ

- パイプライン統合の実装は健全に動作し(unit 277 pass、e2e エラー0)、探索・belief 層は**弱い基準線に対しては
  有意に効く**ことを確認できた(searchN1→policy_only 0.593、full→searchN1 0.607)。
- しかし**現行本番 config を上回る証拠は得られず**、採用は見送り。実装・config・ablation 基盤は残し、
  将来デッキやモデルが変わったときに再計測できる状態にしておく。
- 次に本気で勝率を上げるなら、パイプラインの微調整より **deck.csv/Policy 重み側**か、**実フィールド相手での
  評価**(ミラー自己対戦の限界)に投資するのが筋。

## 生成物

- ablation JSON(デスクトップ): `league/results/d_po_sn1.json` / `d_po_full.json` / `d_sn1_full.json`、
  第1弾は `ablation_run1.json`。
- 実行スクリプト: `league/run_ablation.py`(5構成総当たり)、`league/run_league.py`(A/B)。
