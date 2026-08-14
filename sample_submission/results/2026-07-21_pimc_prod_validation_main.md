# PIMC本番検証 Step2(主比較): 現行提出 ml_lethal vs 本命 ml_pimc

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [pimc-production-validation-implementation-plan.md](../plans/individual/shogo/pimc-production-validation-implementation-plan.md) Step2
- 前提: `ml_policy_agent.agent(obs, config=...)` の config 注入点(本計画 Step1 で追加、後方互換)を使い、同一プロセス内で `ml_lethal` と `ml_pimc` を混線なく head-to-head。両陣営とも `sample_submission/deck.csv`(ミラー)。

## 問い

Stage2 の head-to-head は「PIMC は lethal 探索より強い」を rule_based フレームワークで示した([2026-07-21_stage2_head_to_head.md](2026-07-21_stage2_head_to_head.md)、`rule_pimc` 61% vs `rule_lethal_estimated` 39%)。**本 Step はそれを本番エージェント `ml_policy` の上で問い直す**: 現行提出 `ml_lethal`(lethal_simple + dummy hidden state)に対し、`ml_pimc`(pimc + estimated hidden state)は本番を強くするのか?

## 実験方法

- `league/run_match.play_match` を直接呼ぶ使い捨てスクリプト(スクラッチパッド、リポジトリには含めない)。
- `agent0`/`agent1` は `ml_policy_agent.agent(obs, config=X)` を束縛したクロージャ。`agent()` 内部が `match_context.update` → value_shadow → `_try_lethal(config=X)` → PolicyModel を一貫して同じ config で処理する(Step1 の注入点)。`_get_config()` のグローバルキャッシュは経由しないため、`ml_lethal` と `ml_pimc` が1プロセス内で衝突しない。
- 先手/後手バイアス排除のため 100 試合中 50 試合は `ml_lethal` が player_index=0(先手)、50 試合は player_index=1(後手)。
- 試合ごとに `match_context.reset()`。Wilson 95% CI。

## 結果

| config | 役割 | 試合数 | 勝ち | 勝率 | 95% CI |
|---|---|---|---|---|---|
| `ml_lethal`(lethal_simple + dummy、**現行提出**) | A | 100 | 52 | 0.520 | [0.423, 0.615] |
| `ml_pimc`(pimc, num_determinizations=4 + estimated) | B | 100 | 48 | 0.480 | [0.385, 0.577] |

先手/後手別の内訳(A=`ml_lethal` 視点):

| Aの立場 | 試合数 | Aの勝ち | 勝率 | 95% CI |
|---|---|---|---|---|
| player0(先手) | 50 | 27 | 0.540 | [0.404, 0.670] |
| player1(後手) | 50 | 25 | 0.500 | [0.366, 0.634] |

- エラー(異常終了・不正選択・`MAX_STEPS`超過): **0/100件。**
- 平均ターン数: 13.71、平均ステップ数: 154.8。
- 総実行時間: 1991.7秒(100試合、約19.9秒/試合)。1エージェントあたり予算(600秒/試合)に対して十分小さい。

## わかったこと

1. **`ml_pimc` は現行提出 `ml_lethal` に対して統計的に有意な差がない。** A(`ml_lethal`)の 95% CI [0.423, 0.615] は 0.5 を含み、B(`ml_pimc`)の CI [0.385, 0.577] も 0.5 を含む。**PIMC 化+estimated hidden state 化は、本番 `ml_policy` の上ではミラーマッチで勝率を動かさなかった。**
2. **Stage2 の研究結果(vs lethal で PIMC 61%)は本番に転移しなかった。** rule_based では PIMC が lethal を明確に上回ったが、`ml_policy` 上では両者ほぼ互角。これは「一次判定=提出すべきか」に対して**No(そのままでは提出理由にならない)**を意味する。
3. **推定原因(要検証)**: `ml_policy` は既に PolicyModel が意思決定の大半を担い、lethal/pimc が発火するのは終盤の詰め局面に限られる。その狭い領域で pimc が lethal に勝る分の利得が、試合全体の勝率にはほとんど波及しない。加えてミラーマッチでは相手モデリング(PIMC の本質的な強み)の利得が出にくい。この2点が「rule_based では効いたのに ml_policy では効かない」差の候補。

## 結論(Step2 の一次判定)

**`ml_pimc` を現行提出 `ml_lethal` の置き換えとして提出する根拠は、本計測では得られなかった**(有意差なし)。次に Step3(hidden info 単独の効果)を測り、より安い改善候補が残っているかを確認する。最終判断は Step5 summary に集約する。
