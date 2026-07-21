# Stage1 Step0: ベースライン記録

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage1-wiring-implementation-plan.md](../plans/individual/shogo/stage1-wiring-implementation-plan.md) Step0
- 実験方法: `league/run_league.py` の `run_league()` を直接呼び出し、各エージェントの自己対戦(同一エージェント同士、先手/後手を半々で入れ替え)を50試合実行。config はコード変更前の既定値(`rule_based` → `configs/rule_lethal.json`、`ml_policy` → `configs/ml_lethal.json`)。`ptcg_ai.search.lethal_simple.get_stats()` をリセットしてから実行し、探索の起動回数・検出件数・タイムアウト件数・平均/最大探索時間を記録した。
- 目的: Step1以降で `hidden_state_source` 分岐を追加した際、既存config(キー省略時)の挙動が完全に変わっていないことを確認するための基準値を作る。

## 結果

| エージェント | config | 試合数 | 勝率(A視点) | 95% CI | 平均ターン数 | エラー | 実行時間 |
|---|---|---|---|---|---|---|---|
| rule_based | rule_lethal | 50 | 0.480 (24/50) | [0.348, 0.615] | 13.62 | 0 | 23.7s |
| ml_policy | ml_lethal | 50 | 0.400 (20/50) | [0.276, 0.538] | 13.88 | 0 | 100.3s |

自己対戦(同一エージェント同士)のため勝率は五分近辺に収まっており(誤差の範囲)、これは「強さ」の指標ではなく「壊れていないこと」の確認用。

## lethal_simple.get_stats() (50試合累計)

| エージェント | searches | found | timeouts | node_limit_hits | verify_rejects | avg_time_ms | max_time_ms |
|---|---|---|---|---|---|---|---|
| rule_based | 261 | 36 | 179 | 0 | 0 | 70.8 | 100.7 |
| ml_policy | 806 | 70 | 623 | 0 | 0 | 79.3 | 101.1 |

- `max_remaining_prizes` が rule_lethal=2 / ml_lethal=3 のため、ml_policy 側の方が探索起動回数(searches)・検出件数(found)とも多い(想定通り)。
- どちらもタイムアウト率が高い(rule 68.6%、ml 77.3%)が、これは `time_limit_ms: 100` 到達時に fallback するだけの既知挙動(2026-07-17_lethal_search_trigger.md と同様の傾向)。
- `node_limit_hits` / `verify_rejects` は0件で、既知の健全性と一致。

## 次のステップ

Step1(`selector.py` への `hidden_state_source` 分岐追加)後、`configs/rule_lethal.json`(キー省略)で同条件を再実行し、この表と完全に一致すること(決定論的な行動列が変わらないこと)を確認する。
