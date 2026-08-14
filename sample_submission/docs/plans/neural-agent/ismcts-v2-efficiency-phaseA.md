# ISMCTS v2 Efficiency — Phase A 調査結果(方針転換)

作成日: 2026-07-27
前提: ISMCTS v1 は strength scaling で Current Champion より強いと実証済(hc64/hc128 formal PROMOTE、doc=ismcts-v1-results.md §7)。
v2 の目的は **探索意味論を変えずに wall-time あたりの iterations/depth を増やす効率化**。第一候補 = **`cg.search_begin` 償却**。
ステータス: **Phase A(cg lifecycle 実挙動調査 + profiling)完了。結論: search_begin 償却は無意味 → 実装しない**。
production/cg/main.py/deck.csv/weights/configs/ISMCTS v1 frozen/Reference Pool/Gate2/Strong Opponent 変更なし。read-only 調査。

## Phase A-1: cg search lifecycle(実挙動)
`cg_api.search_begin/step/end/release` を実 obs+world で計測(challengers 局所調査、破棄済):
- **re-fork は可能**: `search_step(root.searchId, sel)` は **新しい searchId を fork**(root=0 → c1=1, c2=2 と distinct)、root は fork 後も
  再 step 可能(c3=3)、child をさらに深く step 可能(gc=4)。→ **1 world を search_begin 1回して root から iteration ごとに再 fork する
  v2 構造は技術的に実現可能**(Phase A2 STOP 条件=「root reuse 不可」には該当しない)。
- **しかし search_begin は極めて安い**: mean **0.4ms**、search_step(fork)0.26ms、比 1.6x。

## Phase A-2: iteration コスト分解(ISMCTS-32、profiling)
| component | 時間占有 | calls | per-call |
|---|---|---|---|
| **Policy scoring(`score_options_from_state`)** | **90.0%** | 590 | 3.50ms |
| search_step | 8.8% | 625 | 0.32ms |
| **search_begin** | **0.5%** | 32 | 0.33ms |
| leaf eval(handcrafted) | 0.0% | 32 | 0.02ms |

→ **`search_begin` は総コストの 0.5%**。iteration あたり ~18 回の **Policy forward(neural net、3.5ms/call)が 90% を占め、その大半は
policy-greedy rollout**(opponent_depth=1 の各 ply で `score_options_from_state`)+ expansion の `_priors`。

## 結論(Case C: search_begin は支配コストでなかった)
- **v2 の第一候補「search_begin 償却」は実装しない**。search_begin=0.5% ゆえ、iteration→world 単位に減らしても speedup は ~0.5%(無意味)。
  Phase A の「premise が成り立たなければ無理に実装しない」に従う。
- **真のボトルネック = Policy scoring(90%)**、特に **rollout の policy-greedy**(iteration あたり ~15 回の policy forward)。
- **探索意味論を変えない範囲での効率化余地は小さい**:
  - policy score cache: rollout の state はほぼ一意で hit 率低い(限定的)。
  - policy batching: rollout は逐次(前 action に依存)でバッチ化困難。
  - → search 品質を保ったまま policy forward を大幅に減らす手はほぼ無い。
- **最も効く efficiency lever = leaf-at-node(rollout 省略、expand ノードで handcrafted leaf を即評価)**: rollout の ~15 policy forward/iter を
  除去 → iteration を ~10x 高速化(leaf は 0.02ms)。**ただしこれは evaluation timing の変更 = 別 variant(ismcts_v2.1、前フェーズで deferred)**。
  profiling は「leaf-at-node こそが唯一の高効果 lever」であることを定量的に示した。

## 推奨(次の1ステップ、要判断)
1. **search_begin 償却 v2 は破棄**(0.5% 効果)。
2. **leaf-at-node(ismcts_v2.1)を次の正式 efficiency 変数にする**: rollout を省き expand ノードで handcrafted leaf 評価。
   これは「探索構造以外を変えない」原則の境界(evaluation timing 変更)だが、profiling 上 **唯一 iteration コストを桁で下げられる手**。
   strength への影響(rollout 無し handcrafted leaf が深い探索の leaf として妥当か)は独立 ablation で測る。**新 pre-registration**。
3. 代替(leaf-at-node を避ける場合)= **strong ISMCTS(hc64/hc128)を teacher とした distillation**(Search→Policy/Value)へ進み、
   提出時は軽量 Policy/Value で再現。Teacher の効率は leaf-at-node で上げると生成速度も改善。

## Integrity
ISMCTS v1 config `59384591` 不変 / Current Champion `ca6c37af` 不変 / Reference Pool v2 `14db8345` / Gate2 v2 `05e509a2` /
Policy `735dd38a` / Belief / handcrafted leaf すべて不変。production/cg/shared 無変更。tracked 差分 0。git add/commit/push なし。
本フェーズは **read-only 調査 + doc のみ**(調査 script は破棄、v2 実装は未着手=方針転換のためレビュー待ち)。
