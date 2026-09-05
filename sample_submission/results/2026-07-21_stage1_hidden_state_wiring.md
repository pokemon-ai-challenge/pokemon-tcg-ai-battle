# Stage1 Step3/Step4: hidden_state_source 実推定 A/B と value network shadow mode

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage1-wiring-implementation-plan.md](../plans/individual/shogo/stage1-wiring-implementation-plan.md) Step3・Step4

## 方法論の注記(重要)

`league/run_league.py` / `run_match.play_match` は1プロセス内で複数試合をループする際、
`hidden_information.match_context` のシングルトン状態(`OwnHiddenState` / `OpponentHiddenState`)を
試合間でリセットしない(`obs.select is None` によるリセットは、ローカルの `battle_start(deck0, deck1)`
がデッキを直接引数で受け取るため実際には発火しない。Kaggle本番相当の「1プロセス1試合」なら
問題にならないが、ローカルの自己対戦ループでは前の試合の推定状態が次の試合に持ち越される)。
これは Stage1 で新たに作った問題ではなく、`match_context.update(obs)` 自体は既に
`rule_based_agent.agent()` から毎ターン呼ばれていた(が今回まで出力を誰も使っていなかった)ため
表面化していなかった既存のテスト基盤の限界。本計測(Step3)はこの限界を踏まえたうえで、
「壊れていないこと」の確認に限定して読む(plan本文の注記どおり)。Step4の健全性チェックでは、
試合ごとに `match_context.reset()` / `value_shadow_log.reset()` を明示的に呼んで独立性を確保した。

もう一点、`league/run_league.py` の `seed` はモジュール ``random`` のみを seed し、ネイティブ
エンジン(`cg.dll`)側のシャッフル/コイントスは別RNGのため制御できない(同一seedでも試合結果が
再現しない: `rule_based` 同一config・同一seed_startで2回実行し勝敗が完全に一致しないことを確認済み)。
そのためStep1完了条件に書いた「決定論的な行動列が変わらない」というのは、実際には既存の
ベースライン自体が個々の試合単位では非決定的であり厳密な意味では検証できない。個々の
行動列比較の代わりに、Step0/Step1後の同一configでの再実行結果が統計的に同じレンジに収まって
いること(下記 rule_based の searches/found の桁・timeouts比率など)で代替確認とした。

## Step3: dummy vs estimated (各120試合、自己対戦)

`league/run_league.run_league()` を直接呼び出し、`rule_based`/`ml_policy` それぞれについて
既定(dummy)configと`_estimated`configで120試合ずつ自己対戦。

| 構成 | 試合数 | 勝率(A視点) | 95% CI | 平均ターン | エラー | 実行時間 |
|---|---|---|---|---|---|---|
| rule_based (dummy, rule_lethal) | 120 | 0.625 (75/120) | [0.536, 0.706] | 15.43 | 0 | 56.0s |
| rule_based (estimated, rule_lethal_estimated) | 120 | 0.583 (70/120) | [0.494, 0.668] | 15.09 | 0 | 58.4s |
| ml_policy (dummy, ml_lethal) | 120 | 0.500 (60/120) | [0.412, 0.588] | 13.23 | 0 | 285.0s |
| ml_policy (estimated, ml_lethal_estimated) | 120 | 0.483 (58/120) | [0.396, 0.572] | 13.23 | 0 | 342.3s |

自己対戦(同一エージェント同士)なので勝率はいずれも五分近辺で、dummy/estimated間の差は
方法論注記の非決定性の範囲内(強さの向上を主張するものではない)。**4構成とも `errors.count == 0`**
で、実推定への切り替えで新たな例外・クラッシュは発生していない。

### lethal_simple.get_stats() (120試合累計)

| 構成 | searches | found | timeouts | node_limit_hits | verify_rejects | avg_time_ms | max_time_ms |
|---|---|---|---|---|---|---|---|
| rule_based dummy | 576 | 102 | 393 | 0 | 0 | 71.2 | 109.7 |
| rule_based estimated | 614 | 101 | 422 | 0 | 0 | 70.9 | 117.2 |
| ml_policy dummy | 2236 | 148 | 1787 | 0 | 4 | 81.6 | 115.2 |
| ml_policy estimated | 2619 | 155 | 2157 | 0 | 3 | 84.0 | 128.2 |

- 探索の発見件数(found)・タイムアウト率は dummy/estimated 間で同程度のオーダー。`node_limit_hits`
  は両方とも0で健全。`verify_rejects`(非公開情報を差し替えた再検証で棄却された行数)は
  estimated側でもdummy側と同程度(むしろわずかに少ない)で、実推定に切り替えたことで
  検証段階の棄却が急増するような不具合は見られない。
- `max_time_ms` が estimated側でやや高い(117〜128ms vs 100〜115ms)のは、
  `search_adapter.to_search_begin_kwargs()` 自体の計算コスト(超幾何サンプリング等)が
  探索の時間予算に上乗せされるため。`time_limit_ms: 100` の予算超過分は数十ms程度で、
  対局全体の実行時間に占める影響は小さい(rule_based: 56.0s→58.4s、+4%程度)。

## Step4: value network shadow mode 健全性チェック

`rule_lethal_estimated.json`(`value_shadow_logging: true` を含む)で `rule_based` の自己対戦を
50試合実行。`league/run_league.py` の試合間状態リークを避けるため、`run_match.play_match` を
直接ループで呼び、**試合ごとに** `match_context.reset()` / `value_shadow_log.reset()` を呼んでから
1試合を実行し、`value_shadow_log.get_log()` を試合終了直後に回収した。

| 項目 | 結果 |
|---|---|
| 試合数 | 50 |
| 例外・タイムアウト | 0 |
| 観測された `win_prob` の範囲 | [0.0383, 0.9702](常に [0, 1] 内) |
| 終盤方向の一致率 | 46/50 = 92.0%(各試合最後にログされた`win_prob`が0.5を跨ぐ方向と実際の勝者が一致した割合) |

- 50試合すべて例外・クラッシュなしで完走(Step4完了条件を満たす)。
- 値は常に `[0, 1]` に収まっている(較正のsigmoid変換により保証されている設計どおり)。
- 終盤(各試合の最後にログされたターン)の予測方向と実際の勝敗が92%一致しており、
  [[project_ml_value_network_step1]] のオフライン評価(test AUC 0.746)と整合する自然な傾向。
  厳密なキャリブレーション検証ではなく健全性チェックである点はplan記載どおり。

## 結論

- Step1/Step2で追加した `hidden_state_source: "estimated"` 分岐は、`rule_based`・`ml_policy`
  いずれも新規例外を発生させずに動作する。
- Step4で追加した value network shadow mode ログも、例外なく毎ターン記録でき、出力レンジ・
  終盤の勝敗方向との整合性ともに健全。
- Stage1 のスコープ(配線のみ、意思決定は変えない)を満たしたまま、Stage2(PIMC/探索本体、
  value networkを実際の候補比較に使う設計)へ進める状態になった。
