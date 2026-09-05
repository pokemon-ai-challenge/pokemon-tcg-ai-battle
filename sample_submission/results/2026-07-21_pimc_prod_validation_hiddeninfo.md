# PIMC本番検証 Step3(分離比較): hidden info 単独の効果(ml_lethal vs ml_lethal_estimated)

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [pimc-production-validation-implementation-plan.md](../plans/individual/shogo/pimc-production-validation-implementation-plan.md) Step3
- 前提: Step2 と同じ方法論(`ml_policy_agent.agent(obs, config=...)` の注入点、ミラー `deck.csv`、試合ごと `match_context.reset()`、Wilson 95% CI)。

## 問い

Step2 で `ml_pimc`(pimc + estimated)は現行提出 `ml_lethal`(lethal_simple + dummy)に有意差なしだった。本 Step は **探索(PIMC)を入れず、Stage1 で配線した estimated hidden state だけ**で本番が改善するかを問う。改善するなら、探索コストゼロのより安い提出候補になる。Stage1 では自己対戦しか測っておらず、この head-to-head は未実施だった。

- A = `ml_lethal`(lethal_simple + **dummy** hidden state、現行提出)
- B = `ml_lethal_estimated`(lethal_simple + **estimated** hidden state、探索方式は同一・hidden state だけが変数)

## 結果

| config | 役割 | 試合数 | 勝ち | 勝率 | 95% CI |
|---|---|---|---|---|---|
| `ml_lethal`(dummy、**現行提出**) | A | 100 | 55 | 0.550 | [0.452, 0.644] |
| `ml_lethal_estimated`(estimated) | B | 100 | 45 | 0.450 | [0.356, 0.548] |

先手/後手別の内訳(A=`ml_lethal` 視点):

| Aの立場 | 試合数 | Aの勝ち | 勝率 | 95% CI |
|---|---|---|---|---|
| player0(先手) | 50 | 34 | 0.680 | [0.542, 0.792] |
| player1(後手) | 50 | 21 | 0.420 | [0.294, 0.558] |

- エラー: **0/100件。**
- 平均ターン数: 13.87、平均ステップ数: 150.52。
- 総実行時間: 294.9秒(100試合、約2.9秒/試合。pimc を含まないため Step2 の 1/7 のコスト)。

## わかったこと

1. **estimated hidden state 単独でも、本番 `ml_policy` の勝率に有意な差は出なかった。** A(dummy)の 95% CI [0.452, 0.644] は 0.5 を含み、B(estimated)の CI [0.356, 0.548] も 0.5 を含む。むしろ点推定は dummy 側がわずかに高い(0.55)が、有意ではない。
2. **先手/後手別の内訳は強い先手有利を示す(先手 0.68 / 後手 0.42)が、これは両陣営で対称なので全体では相殺される**(A は 50/50 で先手・後手を務める)。config 間の差ではなく、このミラーマッチ自体の先手ゲーが強いことの現れ。lethal_simple 同士だと PolicyModel と確定リーサルの寄与が同一で、勝敗は主に手番運で決まっていることを示唆する。
3. **「より安い提出候補(hidden info だけ入れる)」は成立しなかった。** estimated 化は探索コストこそ小さいが、勝率を動かさないため提出理由にならない。

## 結論(Step3)

**estimated hidden state 単独では本番は改善しない**(有意差なし)。Step2(pimc+estimated も有意差なし)と合わせ、本番 `ml_policy` のミラーマッチでは PIMC も hidden info 推定も勝率に転移しないことが確認された。最終判断は [2026-07-21_pimc_prod_validation_summary.md](2026-07-21_pimc_prod_validation_summary.md) に集約する。
