# Stage2 Step4: PIMC 時間予算の実測

> **訂正(2026-07-21、[2026-07-21_stage2_cost_step0_budget_reality_check.md](2026-07-21_stage2_cost_step0_budget_reality_check.md)参照)**: 本ファイル末尾「わかったこと」5.の「Kaggleの持ち時間制約に対して危険域」という結論は不正確だった。ここで測った「1試合平均21〜28秒」は**自己対戦を1プロセスで駆動した際の対局全体の壁時計時間(両陣営の思考時間の合計)**であり、実際の予算(1エージェントあたり600秒、`remainingOverageTime`、対局を通じての累積)と比較すべき「1エージェント分の消費時間」ではない。1エージェント視点で測り直した結果、平均使用率2.22%・最大4.04%と十分低く、危険域ではなかった。以下の実測データ自体(`avg_time_ms`・`determinizations_run`等)は訂正不要、解釈のみ訂正する。

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage2-pimc-implementation-plan.md](../plans/individual/shogo/stage2-pimc-implementation-plan.md) Step4
- 実験方法: `ptcg_ai.rule_based.rule_based_agent.agent` を直接駆動する自己対戦（`league/run_match.py` と同じ低レベル API、`cg.game.battle_start/battle_select/battle_finish`）。`ptcg_ai.action_selection.selector._CONFIG_CACHE` を直接差し替えて `lethal_search.module="pimc"`、`hidden_state_source="estimated"`、`time_limit_ms=300` を固定し、`num_determinizations` だけを 1 / 2 / 4 / 8 と振った。各設定で自己対戦3試合を実行し、`ptcg_ai.search.pimc.get_stats()`（プロセス内累計）を記録した。
- 目的: 契約§4「`time_limit_ms` は探索全体（全determinization合計）の壁時計デッドラインとして扱う」設計が意図通り機能しているか（＝`num_determinizations` を増やしても1回の `search()` 呼び出しの実行時間が線形に伸びないか）を実測で確認し、Kaggleの持ち時間制約に対する安全性を判断する材料を得る。
- サンプルサイズについて: 1試合あたりの実行時間が非常に長い（後述）ため、各設定3試合（探索呼び出し数としては261〜421件/設定）に留めた。ハイパラの最終決定はしない（Non-goals参照）。

## 結果: `pimc.get_stats()`(各設定・自己対戦3試合累計)

| num_determinizations | searches | found | determinizations_run | determination_timeouts | node_limit_hits | avg_time_ms | max_time_ms |
|---|---|---|---|---|---|---|---|
| 1 | 421 | 143 (34.0%) | 421 | 257 (61.0%) | 0 | 199.1 | 303.5 |
| 2 | 300 | 70 (23.3%) | 385 | 220 (57.1%) | 0 | 230.6 | 307.7 |
| 4 | 331 | 149 (45.0%) | 792 | 191 (24.1%) | 0 | 199.2 | 306.8 |
| 8 | 261 | 76 (29.1%) | 774 | 198 (25.6%) | 0 | 243.0 | 305.2 |

## 試合単位の実行時間

| num_determinizations | 3試合の秒数 | 平均秒数/試合 | 最大秒数/試合 | 平均手数/試合 |
|---|---|---|---|---|
| 1 | 13.8 / 34.7 / 35.8 | 28.1 | 35.8 | 148.3 |
| 2 | 34.6 / 3.9 / 31.2 | 23.2 | 34.6 | 107.7 |
| 4 | 14.5 / 41.8 / 10.1 | 22.2 | 41.8 | 118.0 |
| 8 | 38.1 / 12.1 / 13.7 | 21.3 | 38.1 | 94.3 |

## わかったこと

1. **`avg_time_ms` は `num_determinizations` を増やしても伸びない(199〜243ms、`time_limit_ms=300`以下に収まる)。契約§4の「壁時計デッドラインを探索全体で共有する」設計は意図通り機能している。** `max_time_ms` はどの設定でも303〜308ms程度で、`time_limit_ms=300`をわずかに超える(締切チェックがループの境界でしか行われないための想定内のオーバーシュート。lethal_simpleの`time_limit_ms=100`設定でも同様の超過が過去に観測されている)。
2. **`num_determinizations` を増やしても、実際にフルで走るサンプル数は線形には増えない。** `determinizations_run / searches` は 1.0(d=1) → 1.28(d=2) → 2.39(d=4) → 2.97(d=8) で、設定値に対して頭打ちになる(d=8でも平均約3サンプルしか走らない)。共有デッドライン設計の直接的な帰結であり、想定通り。
3. **`found`(=`aggregate`が空でなく、何らかの行動を返せた)の割合が23〜45%と低い。** つまり過半数の`search()`呼び出しで、締切までに1件のdeterminizationも完走せず`aggregate`が空のまま`None`を返し、呼び出し側(`router.route()`)にフォールバックしている。契約§4に記載されていた「`to_search_begin_kwargs()`のコストが探索予算に乗る」懸念([stage1-wiring-implementation-plan.md](../plans/individual/shogo/stage1-wiring-implementation-plan.md)由来)が、determinizationの複数回化で増幅されている可能性が高い(1回のdeterminizationの開始コストだけで締切に達するケースが一定数ある)。
4. **`node_limit_hits` は0件。** ノード数上限(`max_nodes=10000`)には一度も達しておらず、実際のボトルネックは時間予算(value networkの葉評価コスト・`search_begin`のセットアップコスト)であってノード数上限ではない。
5. **1試合あたりの実行時間が非常に重い(平均21〜28秒、最大42秒)。** `lethal_simple`(`max_remaining_prizes`によるゲート、`time_limit_ms=100`)がサイド2枚以下でしか発火しないのに対し、PIMCは契約§4のとおりプライズ枚数によるゲートを持たないため、自分のターン中のほぼ全ての意思決定点(1試合あたり実測94〜148手のうち大部分)で最大300ms級の探索が起動する。これは Step0-3 の「壊れていないか」の確認としては問題ないが、**Kaggleの持ち時間制約に対しては現在の既定値(`time_limit_ms=300`・ゲート無し)のままでは危険域**である可能性が高い。ハイパラ調整自体はNon-goals(Step5以降)だが、この安全性リスクは実測事実として明記しておく。

## 次のステップ

- Step5(A/B比較)は、この実測に基づき試合数を大きく絞る(1試合が数十秒かかるため、`rule_lethal_estimated`の120試合規模には合わせない)。
- 本番投入を検討する場合は、`max_remaining_prizes`相当のゲート追加や`time_limit_ms`の引き下げ、または`hidden_state_factory`呼び出しコスト自体の削減を、Step5以降の別プランとして検討する必要がある(本ファイルは実測記録であり、対策の実装はスコープ外)。
