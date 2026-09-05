# ISMCTS v1 — 実装・検証・Budget 結果

作成日: 2026-07-26
対応設計: ismcts-v1-design-and-implementation-plan.md。実装は `kaggle_replays/{search,challengers}/`(未追跡)。
production/cg/main.py/deck.csv/weights/configs/Reference Pool/Gate2/Strong Opponent 変更なし。

## 1. 実装(production 非改変・read-only 再利用)
- **`kaggle_replays/search/ismcts.py`** — ISMCTS 中核。value domain 集約(`leaf_to_value`=2p−1 / `terminal_value` win+1
  loss−1 / fixed root-perspective negamax `signed_for`)、`info_set_key`(観測可能のみ・相手手札除外=leakage-safe)、
  PUCT(`Q + c·P·√ΣN/(1+N)`)、Champion Policy prior(`score_options_from_state` softmax)、expansion(prior 最大の未展開)、
  rollout(**Champion `pipeline._rollout_and_eval` と同停止条件**=policy greedy + opponent_depth=1 で handcrafted leaf、
  terminal は ±1 優先)、negamax backup、most-visited root 選択、instrumentation(SearchStats)、watchdog(deadline)。
  **決定化 world プールを事前サンプル**(belief 推論を world_pool_size 回に償却。cg は search_step で state を消費し
  巻き戻せないため各 iteration で search_begin は必要)。
- **`kaggle_replays/challengers/ismcts_v1_agent.py`** — Champion(abl_5_full)経路を read-only 再利用し、single-select MAIN
  だけ ISMCTS へ差替。`ismcts.enabled=false` で **Champion 完全同一**(F6)。信念更新(match_context.update)を Champion と
  同契約で実行。fallback は lethal→pipeline→policy(Champion tail)。`default_agent`=harness 用 picklable 関数。
- **`kaggle_replays/challengers/configs/ismcts_v1.json`**(hash 59384591)= abl_5_full + `ismcts` ブロック。差分は Search 構造のみ。
  freeze: iterations 64 / world_pool_size 8 / c_puct 1.4 / opponent_depth 1 / max_rollout_steps 40 / max_depth 60 / **budget_ms 1350**(Kaggle 予算整合)。

## 2. Correctness tests(全 PASS)
- **unit(cg 不使用)9/9**: F3 value domain(leaf/terminal/negamax 符号)、F0 info_set_key が相手手札を含まない、
  F5 prior 決定性、F4 rollout terminal 優先、F6 structural(disabled→Champion 委譲 / lethal 先行 / 非MAIN→Champion fallback / MAIN→ISMCTS)。
- **cg(実ゲーム)6/6**: F1 determinization world フィールド、F2 ISMCTS が合法 action(is_valid_action、mapping_errors 0)、
  F2b enabled agent 合法、F6 disabled agent が Champion 合法 action、F7 module-global tree 無し(game 跨ぎ leak なし)。

## 3. Budget sweep(Phase G・固定 12 MAIN state・**信念現行のうちに計測**・勝率非依存)
branching: min 2 / median 5 / max 6 / mean 4.8。

| iters | latency p50 | p95 | max_depth med | mean_depth | nodes med | change rate(vs Policy top1) |
|---|---|---|---|---|---|---|
| 8 | **1049ms** | 1482ms | 2 | 1.33 | 3 | 4/12 |
| 16 | 2389ms | 3150ms | 2 | 1.75 | 4 | 3/12 |
| 32 | 4509ms | 6276ms | 3 | 2.21 | 8 | 3/12 |
| 64 | 7284ms | 15446ms | 4 | 2.69 | 10 | 3/12 |

**最重要 finding(compute-bound)**: per-iteration ~130ms(cg `search_begin` が支配。1 iteration=1 search_begin+descent+短 rollout。
cg は巻き戻せず iteration ごとに search_begin 必須)。**Kaggle 予算 ~1350ms/select では ~8 iteration しか回らず、tree は浅い
(max_depth ~2, ~3 nodes)**。深い探索(64 iter, depth 4)は 7.3s/decision = 予算 5倍超。ISMCTS が Champion Policy top1 から
action を変えるのは ~33%(8 iter)。→ **feasible 予算内では ISMCTS は Champion の浅 PIMC からほぼ深化できない**。

## 4. Small H2H screen(health + preliminary、60 games、budget_ms=1350、workers=15)
- **ismcts winrate = 34/60 = 0.5667、Wilson CI [0.441, 0.684]**(50% を含む)。errors 0 / timeout 0(**健全性 PASS**)。throughput 8.2 g/min。
- **LLR 軌跡が null へ回帰**: N=10→100% / N=20→75% / N=30→73%(llr 1.25 ピーク)/ N=40→67.5% / N=50→62% / N=60→56.7%(llr 0.50)。
  早期高勝率は小標本ノイズで、N 増で ~50-57% へ収束・SPRT 証拠は弱まる(PROMOTE 境界 A=2.89 へ向かっていない)。
- channel: prize_out 0.533 / deckout 0.283 / no_pokemon 0.183(errors 0)。
- **解釈**: budget sweep(compute-bound・浅 tree・action 変更 ~33%)と整合。**feasible 予算内では ISMCTS v1 は Champion とほぼ互角**(明確な改善の兆候なし)。

## 5. Formal Gate 1 / Decision(要ユーザー判断)
budget sweep(compute-bound: ~8 iter/浅 tree)+ screen(llr 1.25→0.50 で null 回帰、CI が 50% を含む)より、**真の効果は ~0 に近い**公算大。
正式 Gate 1(δ_min=0.05, n_max=3000)を回すと、真効果≈0 なら SPRT は決定せず n_max まで進み **TRUNCATED_NO_PROMOTE**(約 6 時間 @8.2 g/min)になる見込み。
- **選択肢A**: それでも正式 Gate 1 を完走(事前登録どおり、~6h、likely TRUNCATED)。
- **選択肢B(推奨)**: budget sweep + screen を「ISMCTS v1 は compute-bound で feasible 予算では Champion 深化に至らず」の十分な証拠とし、
  正式 6h を回さず **Phase O Option D(compute/branching bottleneck)へ**。次の独立 increment 候補:
  (1) search_begin を world 単位で償却(iteration 内 re-fork)+ leaf-at-node(rollout 省略)で per-iteration コストを大幅削減 →
      同予算で深い tree を可能にしてから Gate 1、または (2) progressive widening / policy top-k expansion で branching を絞る。
- **重要**: どちらでも δ_min/契約は結果後に変更しない。追加 variant は新 pre-registration(ismcts_v2)とする。

### 【修正】結論の訂正(2026-07-26、方針修正)
**前回「ISMCTS v1 は compute-bound で Champion 互角」は言い過ぎだった**。提出時計算量(~1350ms/select→~8 iter→depth~2)を早期に
主制約にしたのが研究目的とずれていた。**現時点で言えるのは**: (1) ISMCTS v1 は correct(F0-F7 PASS)。(2) 提出相当の低 compute
(~8 iter)では proper ISMCTS の深さを得られず、Champion を明確に上回る証拠は「まだ」ない。**→ ISMCTS そのものの強さは未判定**。
**現研究質問(修正後)= Search 時間を十分与え iteration/depth を増やせば ISMCTS v1 は Champion より強くなるか**(compute-unconstrained
strength scaling)。提出時の高速化・蒸留はアルゴリズム有効性を確認した後に行う。→ §7 へ。

### (旧・保留)効率化案 = ismcts_v2
per-iteration の cg `search_begin` コスト(~130ms、支配項)を削るのが核心。
1. **search_begin を world 単位で償却**: 現状は iteration ごとに search_begin。world プールの各 world で search_begin を1回行い、
   その root から iteration ごとに **re-fork(root.searchId から search_step で再降下)**して tree を成長 → search_begin 回数を
   iteration 数から world 数へ削減。cg が root からの再 step を許すことは Champion pipeline(`_evaluate_candidate` が root から
   候補ごとに search_step)で確認済み。
2. **leaf-at-node(rollout 省略の是非)**: 短 rollout(policy greedy)も policy forward × 数手でコスト。expand ノードで
   handcrafted leaf を即評価(rollout 省略)すれば大幅減。ただし**これは leaf 評価 timing の変更**=「Search 構造以外を変えない」
   原則との境界を要検討。v1 は Champion 同等 rollout を維持したので、rollout 省略は **ismcts_v2 の明示的な設計変数**として
   pre-registration に記す(v1 との差分を「search_begin 償却」だけに留めるか、leaf-at-node も入れるかは v2 設計で確定)。
3. 代替: progressive widening / policy top-k expansion で branching を絞り実効 iteration/深さを稼ぐ。
**ismcts_v2 は新 config/新 pre-registration。v1 の frozen(config 59384591・実装・test)は historical として保持**。実装はレビュー後。

## 7. Compute-Unconstrained Strength Scaling(方針修正後の本命)
提出時計算量を制約にせず「十分な探索で ISMCTS v1 は Champion より強いか」を測る。ISMCTS v1 は frozen 維持
(config 59384591・実装・F0-F7 不変)。budget 別 config=`configs/ismcts_hc{8,16,32,64,128}.json`(iterations のみ変更・時間 cap 無し、
`ISMCTS_CONFIG` env で worker へ渡す)。

### 7.1 depth/nodes scaling(offline、desktop、12 MAIN state、branching median 4)
| iters | latency p50 | max_depth med | mean_depth | nodes med | action change(vs Policy top1) |
|---|---|---|---|---|---|
| 8 | 279ms | 2 | 1.57 | 3 | 2/12 |
| 16 | 550ms | 3 | 2.07 | 6 | 3/12 |
| 32 | 1318ms | 4 | 2.66 | 10 | 1/12 |
| 64 | 2941ms | 4 | 3.51 | 18 | 3/12 |
| 128 | 5578ms | **7** | 4.43 | 28 | 3/12 |
→ **depth/nodes は compute とともに単調増加**(tree は実際に深化)。ただし **action change 率は ~20-25% で頭打ち**
(深くしても Policy top1 から変える割合は増えない=handcrafted-leaf 探索は多くの局面で Policy に同意)。

### 7.2 strength scaling H2H(ISMCTS-N vs abl_5_full、mirror 同 deck、手番交互、workers=15、errors 0)
| iters | games | **winrate** | Wilson95 CI | SPRT(δ_min=0.05) |
|---|---|---|---|---|
| 8 | 100 | **0.430** | [0.337, 0.528] | — |
| 16 | 100 | **0.620** | [0.522, 0.709] | — |
| 32 | 100 | **0.620** | [0.522, 0.709] | — |
| 64 | 100 | **0.670** | [0.573, 0.754] | **PROMOTE**(llr 2.91) |
| 128 | 58 | **0.776** | [0.653, 0.864] | **PROMOTE**(llr 2.92, 早期決定) |

→ **winrate は compute とともに単調上昇**(8→0.43 / 16→0.62 / 32→0.62 / 64→0.67 / 128→0.78)。**8 iter は 0.43 で Champion より弱い**
(浅すぎて Champion の PIMC(8世界×4候補=depth1 多点評価)に劣る)が、**16 iter で 0.62 へ crossover、64/128 で Champion を
formal SPRT で PROMOTE**(δ_min=0.05、hc128 は 58 games で早期決定)。128 でもまだ上昇中(飽和未確認)。

### 7.3 Conclusion = **Case A(ISMCTS は compute を与えると強くなる)**
- **ISMCTS v1 は algorithmically Current Champion より強い**(64/128 iter で formal PROMOTE、winrate 単調 scaling)。
- **前回の「compute-bound で Champion 互角」は誤り**だった。提出相当の低 compute(~8 iter)では 0.43 と**むしろ弱い**ため、
  time-capped screen が null/負に見えていた。**proper ISMCTS の強さは十分な探索で明確に発現する**。
- **ボトルネック = compute efficiency**(strength でなく、submission 予算で iterations が足りないこと)。depth/nodes は scaling
  するが per-iteration が cg search_begin で重い(§3/§7.1)。
- **High-Compute "Research Challenger"**: hc64(config fabfe2fc、PROMOTE 67%、~2.9s/decision)を formal 確認済み下限、
  hc128(config 106a9807、77.6%、~5.6s/decision)を上限として選定。**提出候補でなく algorithmic strength の実証**。
- **次に推奨する1ステップ(Case A)= ISMCTS v2 Efficiency**: per-iteration の search_begin を world 単位で償却(root からの
  re-fork)+ tree/simulator state 再利用 → 同 wall-time で iteration/depth を増やす。並行して **strong ISMCTS を teacher とした
  distillation**(Search action→Policy / Search outcome→Value)で提出時軽量 Search へ落とす道も有力(§J)。leaf-at-node/Value leaf/
  progressive widening は efficiency 後の独立 ablation。**いずれも新 pre-registration(ismcts_v2)。v1 frozen は historical 保持**。

## 6. Integrity
tracked 差分 0。Reference Pool v1/v2・Gate2 v1/v2・Alakazam Champion(abl_5_full)・Strong Opponent Champions・production/cg/
deck.csv/weights/configs 不変。新規は未追跡 `kaggle_replays/{search,challengers}/` + docs のみ。git add/commit/push なし。
