# Stage2 Step5: rule_lethal_estimated vs rule_pimc

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage2-pimc-implementation-plan.md](../plans/individual/shogo/stage2-pimc-implementation-plan.md) Step5
- 実験方法: Stage1 Step3 と同じ方法論(自己対戦、`league/run_league.py` の既知の限界を踏まえる)。ただし本比較は同一config同士の自己対戦(A vs A)であり、`rule_lethal_estimated` vs `rule_pimc` の直接対戦(head-to-head)ではない。
- **head-to-headにしなかった理由**: `hidden_information.match_context` はプロセス内シングルトンで、「自分の隠れ情報」を `state.yourIndex` に応じて同じオブジェクトへ書き込む設計([match_context.py](../../ptcg_ai/hidden_information/match_context.py))。1試合の中で player0側とplayer1側に**異なるconfig**(片方lethal_simple、片方pimc)を割り当てると、`_own_state`/`_opponent_state` が両プレイヤーの情報を交互に上書きし合い、`hidden_state_source="estimated"` の推定が両陣営とも壊れる。Stage1のhead-to-head(`league/run_league.py`のml_policy vs rule_based、500試合)は両configとも `hidden_state_source` 省略(dummy)だったためこの問題を踏んでいなかった。本Stepでは「推定を使う2つのconfigを同時に対戦させる」という新しい状況のため、Stage1と同じ「自己対戦のみ」の方法論を踏襲した(この限界の是正はスコープ外)。
- 従って本結果は**「強さの比較」ではなく「壊れていないか + 実行コストの比較」**として読む(自己対戦の勝率は先手/後手ノイズであり、五分から外れていても強さの差ではない)。

## サンプルサイズについて

`rule_pimc` は1試合あたり平均26.6秒(実測)と非常に重く、`rule_lethal_estimated` の120試合規模には合わせられなかった。`rule_lethal_estimated` は30試合(高速、40秒未満で完走)、`rule_pimc` は15試合(約6分40秒)に留めた。ハイパラ最終決定はしない(Non-goals、Step4の実測を参照)。

## 結果

| config | 試合数 | p0勝率(自己対戦) | 平均ターン数 | エラー | 合計実行時間 | 平均秒数/試合 | 最大秒数/試合 |
|---|---|---|---|---|---|---|---|
| rule_lethal_estimated | 30 | 0.500 (15/30) | 15.5 | 0 | 16.8s | 0.56s | 3.7s |
| rule_pimc (num_determinizations=4) | 15 | 0.467 (7/15) | 14.4 | 0 | 399.2s | 26.6s | 51.9s |

**両configとも30/15試合を通じてエラー0件。** `rule_pimc` は Step0-3 のユニットテスト・スモークテストに続き、実対戦(自己対戦)でもクラッシュ・不正選択なく完走することを確認できた。

## 探索モジュールの `get_stats()`

| config | searches | found | 備考 |
|---|---|---|---|
| rule_lethal_estimated (`lethal_simple`) | 182 | 22 (12.1%) | `found`=確定リーサルを検出した回数 |
| rule_pimc (`pimc`) | 1745 | 505 (28.9%) | `found`=締切までに1件以上のdeterminizationが完走し、行動を返せた回数(**勝ちを見つけた回数ではない**) |

**両モジュールの `found` は意味が異なるため直接比較できない**(`lethal_simple` の `found` = 確定勝ちの検出、`pimc` の `found` = 何らかの評価値付き行動を返せたかどうか)。`rule_lethal_estimated` は `max_remaining_prizes=2` のゲートがあるため探索回数(searches)自体が少ない一方、`rule_pimc` はゲートが無いため1試合あたりの探索起動回数が約10倍(182/30≈6.1回/試合 vs 1745/15≈116.3回/試合)。

## わかったこと

1. **健全性: 問題なし。** 45試合(30+15)を通じてエラー0件、不正選択0件。`rule_pimc` は Definition of Done の「壊れていないこと」の確認基準を満たす。
2. **自己対戦勝率はどちらも0.5近辺(0.500 / 0.467)で、既知のノイズ範囲内。** 「強くなったか」を主張する根拠にはならない(そもそも同一config同士の対戦のため、原理的に強さの差を測れない)。
3. **実行コストは`rule_pimc`が`rule_lethal_estimated`の約47倍(平均秒数/試合ベース)。** [2026-07-21_stage2_pimc_timing.md](2026-07-21_stage2_pimc_timing.md) で確認した「プライズ枚数によるゲートが無いため、ほぼ全ての自ターン意思決定点で探索が起動する」ことが原因。ただし**この「47倍」は対局全体(両陣営合計)の壁時計時間の比較であり、実際のKaggle予算(1エージェントあたり600秒)に対する使用率で見ると平均2.22%・最大4.04%(30サンプル)と十分低いことが後日判明した([2026-07-21_stage2_cost_step0_budget_reality_check.md](2026-07-21_stage2_cost_step0_budget_reality_check.md))。「リスクが高い」は誤りで、ゲート追加やパラメータ調整は本番投入のブロッカーではなく優先度の低い改善項目に格下げされた。
4. **本当の「強さ」の比較(head-to-head)は今回できていない。** 上述の `match_context` シングルトンの制約により、`hidden_state_source="estimated"` を使う2つの異なるconfigを同一試合内で対戦させる仕組みが現状無い。これは今回新たに判明した限界であり、v1以降でhead-to-head評価をやる場合は `match_context` をプレイヤーごとに分離する(またはプロセスを分けて対戦させる)設計変更が必要になる。

## 次のステップ

- Step0-5はすべて完了。Definition of Done の残項目(`selector.py`/`ml_policy_agent.py` への追記内容をtsuoimorikaさんに共有・合意)は人間側の対応が必要。
- v1以降の検討事項(このファイルはあくまで実測記録、対応はスコープ外):
  - `rule_pimc` の実行コスト対策(プライズ枚数ゲートの追加、`time_limit_ms`/`num_determinizations`のチューニング)
  - head-to-head評価を可能にする `match_context` のプレイヤー分離
  - policy prior統合、相手ターンをまたぐ探索への拡張([stage2-pimc-implementation-plan.md](../plans/individual/shogo/stage2-pimc-implementation-plan.md) 末尾参照)
