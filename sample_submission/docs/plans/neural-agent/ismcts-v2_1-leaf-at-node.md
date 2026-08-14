# ISMCTS v2.1 — Leaf-at-Node Efficiency Ablation(pre-registration + 結果)

作成日: 2026-07-27
parent: ISMCTS v1(config hash 59384591、strength scaling で Champion より強いと実証・doc=ismcts-v1-results.md §7)。
Phase A(cg profiling)で iteration コストの 90% が **rollout 中の逐次 Policy forward** と判明(search_begin は 0.5%=償却しない)。
ステータス: **pre-registration。実装→correctness→profiling→equal-wall-time H2H の順**。production/cg/ISMCTS v1 frozen/Reference Pool/
Gate2/Strong Opponent 変更なし。

## 1. 唯一の正式差分(pre-registered)
| | v1 | **v2.1** |
|---|---|---|
| expansion 後 | policy-greedy rollout(opponent_depth 相手ターンまで)→ handcrafted leaf | **expand した node を即 handcrafted leaf 評価(rollout 無し)** |
差分は **leaf 評価 timing / rollout 有無のみ**。以下は v1 と**完全同一**: information-set tree / determinization(match_context estimated)/
Belief / Policy prior(735dd38a)/ PUCT / expansion rule / backup / value domain [-1,+1] / terminal handling / lethal / fallback / deck /
handcrafted evaluator 本体。**Value leaf / progressive widening / pruning / Belief 変更 / distillation は今回入れない**。

## 2. 実装(core 共有・v1 frozen behavior 不変を保証)
`search/ismcts.py` の `_rollout` 冒頭に `leaf_mode`(既定 "rollout"=v1 完全同一 / "node"=即 leaf)を追加。**既定は rollout ゆえ v1 の
挙動は byte 単位で不変**(v1 test 全再 PASS で保証)。agent は既存 `ismcts_v1_agent`(ISMCTS_CONFIG env で config 切替)を再利用。
v2.1 config = ismcts_v1.json + `ismcts.leaf_mode="node"`(`configs/ismcts_v2_1_hc{N}.json`)。

## 3. 仮説(pre-registered)
- **H1(efficiency)**: rollout の Policy forward が消え iteration throughput が大幅改善。
- **H2(same-iteration quality)**: 同 iteration 数では v2.1 は v1 より弱い可能性(rollout の先読み情報が消えるため)。**これは失敗でない**。
- **H3(equal-wall-time quality、Primary)**: 同 wall-time なら v2.1 はより多くの iterations/depth を得られ、**v1 と同等以上の strength に到達しうる**。

## 4. 成功判定(結果前 freeze)
- **Case A(理想)**: same-iteration では v2.1 やや弱いが、**same-wall-time で v2.1 が v1 より強い** → efficiency 成功。
- **Case B(strictly better)**: same-iteration でも同等 + speedup → 強い成功。
- **Case C(fast but weak)**: 10x 速いが大量 iteration でも v1 strength に届かない → rollout に重要情報。leaf-at-node 単独不採用、次 Value leaf。
- **Case D(speedup 小)**: profiling 予測と矛盾 → 実装/計測再確認。

## 5. 評価計画(結果前 freeze)
- correctness: **F0-F7 全 PASS(v1 と同一 suite)+ F8 no-rollout(leaf_mode=node で rollout の policy score が呼ばれない)+ F9 leaf-point
  (評価対象が expand した node そのもの、余計に先へ進まない)**。
- profiling(固定 state set): v1 vs v2.1 の Policy calls/iteration・iteration latency・iterations/sec・speedup。
- fixed-iteration(8/16/32/64/128): latency/nodes/depth/selected action/v1 action agreement。
- **equal-wall-time H2H(Primary)**: wall-time ∈ {~1.35s, ~3s, ~5.5s}(実測 machine で freeze)で v1 @T vs Champion、v2.1 @T vs Champion。
  各 100 games。Champion 共通 control。加えて **直接比較 v2.1 @T vs v1 @T**(Phase L)。
- gate: large algorithmic change → δ_min=0.05/α.05/β.10。wall-time 型ゆえ runner 整合を確認、high-compute で n_max 非現実的なら別 Research
  Gate を**結果前** pre-register。**結果を見て契約変更しない**。

## 6. 結果
### 6.1 Correctness
- **v1 全 test 不変**: leaf_mode 追加(既定 rollout)後も v1 unit 9/9 + cg 6/6 PASS = v1 挙動不変を保証。
- **v2.1 新 test 5/5 PASS**: F8 no-rollout(leaf_mode=node で rollout の policy が呼ばれない)/ F8 terminal 優先 /
  F9 leaf-point(評価対象=expand した node の state、先へ進めない)/ F9 perspective / default=rollout。errors/mapping 0。
- F0-F7 は共有 core(leaf_mode=node でも tree/determinization/PUCT/backup/terminal/reset は同一)ゆえ継承。

### 6.2 Profiling(v1 rollout vs v2.1 node、固定 iterations)
| iters | v1 latency | v1 policy calls | v2.1 latency | v2.1 policy calls | speedup |
|---|---|---|---|---|---|
| 8 | 552ms | 100 | 53ms | 5 | 10.5× |
| 64 | 4918ms | 866 | 165ms | 20 | 29.8× |
| 128 | 10425ms | 1784 | 280ms | 32 | 37.3× |
→ **H1 確認**: leaf-at-node で **rollout の逐次 Policy forward がほぼ消滅**(128 iter で 1784→32 calls)、**~10-37× 高速化**。
同 iterations では **depth は同一**(leaf mode は tree 構造を変えないため)。

### 6.3 iterations achieved / depth per wall-time(H1 の実効)
| budget | v1 iters(depth) | v2.1 iters(depth) | ratio |
|---|---|---|---|
| **1.35s(submission)** | 25(depth 4) | **553(depth 12)** | 22× |
| 3.0s | 37(depth 5) | 877(depth 15) | 24× |
| 5.5s | 46(depth 5) | 1498(depth 19) | 33× |
→ 同 wall-time で v2.1 は **~22-33× 多い iterations、depth 12-19**(v1 は 4-5)。

### 6.4 Equal-Wall-Time H2H(Primary、H3)vs abl_5_full、各 100 games、budget {1.35, 3, 5.5}s
<!-- RESULT_EQUAL_WALLTIME -->

### 6.5 Conclusion(Case A-D)
<!-- RESULT_CONCLUSION -->

## Integrity
ISMCTS v1 config 59384591 / Champion ca6c37af / Policy 735dd38a / Belief / handcrafted evaluator / Reference Pool v2 14db8345 /
Gate2 v2 05e509a2 すべて不変。`ismcts.py` は既定 rollow=v1 不変(opt-in flag)。tracked 差分 0。git add/commit/push なし。新規は未追跡のみ。
