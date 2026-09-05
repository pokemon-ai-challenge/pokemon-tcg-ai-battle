# Stage2 Step4: rule_lethal_estimated vs rule_pimc の実際のhead-to-head評価

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage2-match-context-separation-implementation-plan.md](../plans/individual/shogo/stage2-match-context-separation-implementation-plan.md) Step4
- 前提: 本Stageで `hidden_information/match_context.py` の `_own_state`/`_opponent_state`/`_knowledge` を `player_index` 別のdictに分離した([[project_match_context_single_perspective_limitation]] で判明した制約の解消)。これにより、[2026-07-21_stage2_pimc_ab.md](2026-07-21_stage2_pimc_ab.md) で「できなかった」と記録した**異なるconfigの2エージェントの直接対戦**が初めて可能になった。

## 実験方法

- `league/run_match.play_match` を直接呼ぶ使い捨てスクリプト(スクラッチパッド上、cost step0と同じ方針でリポジトリには含めない)。
- `agent0`/`agent1` はそれぞれ `ptcg_ai.action_selection.selector.select_action(obs, full_deck, config=...)` を束縛したクロージャとして用意し、`rule_based_agent.agent` を経由しない(プロセス全体で共有される `selector._CONFIG_CACHE` を経由せず、`rule_lethal_estimated` と `rule_pimc` を1プロセス内で確実に別configとして扱うため)。各クロージャは `rule_based_agent.agent()` と同じく先頭で毎ターン `match_context.update(obs)` を呼ぶ。
- 先手/後手バイアスを排除するため、100試合のうち50試合は `rule_lethal_estimated` がplayer_index=0(先手)、残り50試合はplayer_index=1(後手)。デッキは両陣営とも `sample_submission/deck.csv` で共通(ミラーマッチ、AIの意思決定方式のみが変数)。
- 試合ごとに `match_context.reset()` してから `play_match` を呼ぶ(前の試合の推定状態を持ち越さない)。
- 勝率の信頼区間はWilson score interval(95%)。

## 結果

| config | 役割 | 試合数 | 勝ち | 勝率 | 95% CI |
|---|---|---|---|---|---|
| rule_lethal_estimated (`lethal_simple`, `hidden_state_source=estimated`) | A | 100 | 39 | 0.390 | [0.300, 0.488] |
| rule_pimc (`pimc`, `num_determinizations=4`, `hidden_state_source=estimated`) | B | 100 | 61 | 0.610 | [0.512, 0.700] |

先手/後手別の内訳(A=rule_lethal_estimated視点):

| Aの立場 | 試合数 | Aの勝ち | 勝率 | 95% CI |
|---|---|---|---|---|
| player0(先手) | 50 | 21 | 0.420 | [0.294, 0.558] |
| player1(後手) | 50 | 18 | 0.360 | [0.241, 0.499] |

- エラー(異常終了・不正選択・`MAX_STEPS`超過): **0/100件。**
- 平均ターン数: 15.2、平均ステップ数: 122.74。
- 総実行時間: 1390.2秒(100試合、約13.9秒/試合)。1エージェントあたりの予算(600秒/試合、[2026-07-21_stage2_cost_step0_budget_reality_check.md](2026-07-21_stage2_cost_step0_budget_reality_check.md))に対して十分小さく、コスト面の問題は今回も再確認されなかった。

## わかったこと

1. **`match_context` のプレイヤー分離は健全に機能した。** 100試合・エラー0件で完走し、異なるconfigの2エージェントが同一プロセス内で混線せずに対戦できることを実証した(本Stageの主目的)。
2. **`rule_pimc` は `rule_lethal_estimated` に対して統計的に有意に勝ち越している。** 全体勝率はA(rule_lethal_estimated) 39% vs B(rule_pimc) 61%で、Aの95% CI [0.300, 0.488] は0.5を含まない。先手/後手いずれの内訳で見てもAは40%前後にとどまっており、先手/後手バイアスで説明できる差ではない。
3. **[2026-07-21_stage2_pimc_ab.md](2026-07-21_stage2_pimc_ab.md) の自己対戦(A vs A)の勝率が0.5近辺だった結果と矛盾しない。** あちらは「壊れていないか」の確認に留まり「強さ」を測れていなかった(本ファイル冒頭の前提参照)。今回初めて「強さ」の問いに直接答えが出た。

## 結論(Stage2全体の問いへの回答)

**PIMC(`rule_pimc`)はlethal_simple(+value shadow、`rule_lethal_estimated`)に対して有利。** 100試合・95%信頼区間で明確に勝ち越しており、実行コストも予算に対して問題ないレベル(Step0で確認済み)。

## 次のステップ

- Definition of Doneの残項目: `selector.py`/`ml_policy_agent.py` への累積の追記内容(Stage1・Stage2・本Stage分)をtsuoimorikaさんに共有・合意する(人間側の対応が必要、未実施)。
- 本結果により、v1(policy prior統合・相手ターンをまたぐ探索)に進む価値があると判断できる。次の分岐点は、v1に進むか、他の優先課題(実戦の敗因である山札切れ対策など、[[project_deckout_loss_cause]])を先に対応するかをユーザーと相談して決める。
