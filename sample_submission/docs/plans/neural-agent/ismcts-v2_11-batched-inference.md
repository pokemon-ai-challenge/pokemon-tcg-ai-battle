# ISMCTS v2.11 — H4 Batched Inference / Deployment Optimization(結果)

作成日: 2026-07-28
parent: v2.4 H4(e7e74ac7、唯一の deployment efficiency 成功)。
Primary Research Question: **v2.4 H4 の Policy function/weights/action ranking/rollout horizon を一切変えず、rollout 内 Policy scoring の実装だけを
numpy batch 化して高速化すると、同じ 1350ms で iteration を増やし v2.4 H4 をさらに強化できるか**。唯一差分 = **rollout scoring 実装(意味不変)**。
experiment_start_HEAD `ccdc63c`。

## A/C. Baseline / Bottleneck
PolicyModel.score_options_from_state は option ごとに pure-Python 三重ループ matmul(NN が call の ~96%、v2.4 audit)。
options/call: mean 6.59 / median 7 / P90 10 / max 14。H4 old scoring: **0.340 ms/call**(desktop, 実 rollout states)。

## D. Batched Scorer（新規 research asset、production 不変)
`BatchedPolicyModel`(PolicyModel 継承): 同一 weights/encoder/standardization/embedding を使い、legal options を [N,239] 行列に束ねて
**numpy batched matmul**(`X @ W.T + b`, relu, 最終層生スコア)。依存は numpy(既存)のみ。**production の PolicyModel は変更しない**
(challenger 経路 `ismcts.rollout_policy_batched` opt-in、既定 False=v2.4/v1 byte 不変)。legal option 順序不変。

## E. Numerical / Behavioral Equivalence（PASS、最重要)
実 rollout states(204 scoring calls）:
- **score max abs error = 5.33e-15 / mean 9.04e-16**(freeze bar ≤ 1e-6)。
- **top1 agreement = 204/204 = 100%**。tie-sensitive(gap<1e-6): 0 件。
- feature 構築は encoder 共有ゆえ入力完全同一。→ **同一入力に同一行動、計算だけ速い**を実証。
- fixed-128-iter 探索: old vs batched action 10/12(2 件は `determinize()` 確率的 world プール由来=old-vs-old baseline と同水準、scorer 差ではない)。

## F/H. Performance（PASS）
| metric | v2.4 H4 old | v2.11 batched | change |
|---|---|---|---|
| ms/call | 0.340 | **0.100** | **−70.5%(3.39×)** |
| iterations@1350ms | 124.6 | **206.5** | **+66%** |

perf bar(≥20% reduction or ≥20% iterations増)を大きく超過。

## K/L. Formal Parent H2H（v2.11 batched vs v2.4 H4, 1350ms, errors 0）
**175/355 = 0.4930 [0.441, 0.545]、SPRT FUTILITY** = **v2.4 H4 と互角**(親より強くない)。

## M. Conclusion（Case B）
**Case B(speedup but online neutral)確定**。engineering は成功(scorer 3.39×高速・iterations +66%・意味完全同一)だが、
**1350ms での追加 iterations は strength を上げない**(v2.4 H4 は既に ~125 iters=v1 scaling の平坦域: 64→0.67/128→0.78 で頭打ち、+66%→206 の
marginal value が小さい)。semantic equivalence 検証済ゆえ Case C/E ではない。

**deployment 上の含意(前向き)**:
- batched scorer は **v2.4 H4 と完全に同一挙動で 3.4× 高速** = **提出時の deployment 最適化として単体で有用**(同 strength を低 latency・広 budget margin で実行=
  budget overrun リスク減、submission 安全性向上)。strength を上げないだけで、**同じ強さを安全に走らせる改善**として採用価値がある。
- **strength を 1350ms で上げるには「一様な iteration 増」では頭打ち** → 次は **Adaptive Search Budget**(難しい root へ compute 集中)。

**決定**:
- v2.11 batched scorer は **deployment 最適化として採用候補**(v2.4 H4 の submission に組込可、挙動不変)。ただし strength candidate ではない(親互角)。
- **次の 1 ステップ(pre-reg Phase M Case B)= Adaptive Search Budget**。today は実装しない(レビュー待ち)。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。**production の PolicyModel/policy_model.py 不変**(BatchedPolicyModel は
challenger の subclass)。H4 weights(rollout_student_h4.json)/ Original Policy 735dd38a 不変。runner preflight・errors 0。
ISMCTS v1 59384591 / v2.1-v2.10 / v2.4 H4 e7e74ac7 / Value 0657f7af / Belief / handcrafted leaf / Champion / **Reference Pool v2 14db8345 不使用** /
Gate2 v2 05e509a2 すべて不変。新規 dependency なし(numpy は既存)。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
tracked 差分は研究資産のみ(ismcts_v1_agent.py の batched opt-in、既定 False=不変)。
