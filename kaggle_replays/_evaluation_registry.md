# Evaluation registry（§61）

## E0: cg エンジン RNG 監査（実測）
- `battle_start(deck0, deck1)` に **seed 引数なし**。`cg/game.py` `cg/api.py` に random/seed 参照なし。
  シャッフルはネイティブ DLL 内部。
- **cross-process 決定性テスト**: 同一引数で 20 games を2回、別プロセスで実行
  → outcomes / steps とも不一致（signature 446bebb9 vs 414d6cf4）。
  = **DLL は非再現な entropy で seed される。CRN は Python seed でも fresh-process でも不可能。**
- したがって分散を下げる手段は「N を増やす」と「1ゲームを安くする」の2つだけ。

## E1: A/A validation（同一 policy = climb を独立 seed で反復）
policy-only FIELD 評価器（`_p1915_field.py` / `parallel_collect_field`, temperature 0.01）
```
n=2400 x 4 seeds:  0.6454 / 0.6283 / 0.6538 / 0.6254
mean 0.6382   sd 0.0136
binomial SE(n=2400,p=.64) = 0.0098  ->  分散膨張係数 1.39x（相手mix抽選のゆらぎ）
```
**注意: 各runが報告する Wilson CI(±0.019 @2400) は真のseed間ばらつきを約1.4倍過小評価する。**

## 評価器のコスト比較（実測）
| evaluator | 1ゲーム | 2400ゲーム | 備考 |
|---|--:|--:|---|
| production full-game A/B (abl_5_full) | ~22 s | ~1.0 h | lethal+PIMC込み。**±6〜7pt @n=420** |
| policy-only FIELD (screen) | ~0.05 s | ~123 s | 探索なし。**~400倍安い** |

## 必要 N の式
同一 seed0 なら task 列（相手・先後）が一致するので相手mixの分散は差分で相殺し、
差の SE ≈ sqrt(2p(1-p)/n)（binomial のみ）。p≈0.63 で
```
MDE(95%CI) = 1.96 * sqrt(2*0.233/n)
  n= 2,400/arm -> ±2.7pt      n= 9,600/arm -> ±1.4pt   (~8分/arm)
  n=38,400/arm -> ±0.7pt      (~32分/arm)
production A/B で ±2pt を出すには n≈4,600/arm = 約4時間
```

## 確定した4段パイプライン（§15）
```
SCREEN     policy-only FIELD, 同一 seed0, n=9,600/arm  (±1.4pt, 8分)
CONFIRM    別 seed で再現 + ladder-share 加重, n=9,600  (±1.4pt)
PRODUCTION abl_5_full full-game A/B, N は上式から算出（±2pt なら 4,600/arm）
KAGGLE
```

## 既知の限界
- SCREEN は探索なしなので production と完全には一致しない。あくまで negative を落とす篩。
- 逆に climb 自身がこの指標で選ばれて LB best になっている事実は、この指標の proxy 妥当性を
  弱いながら支持する（[[project_local_ab_noise_floor]] も参照）。
