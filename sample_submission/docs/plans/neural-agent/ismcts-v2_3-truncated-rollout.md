# ISMCTS v2.3 — Semantic Truncated Rollout Ablation(pre-registration + 結果)

作成日: 2026-07-27
parent: ISMCTS v1 full rollout(config 59384591、Champion 超え実証)。
Primary Research Question: **full Policy rollout のどこまでが本当に必要か** — rollout を semantic boundary(turn handoff)で途中打ち切りし、
その時点で既存 handcrafted evaluator を使うことで、v1 strength の大部分を維持しつつ Policy forward を大幅削減できるか。
ステータス: **pre-registration(online H2H 前 freeze)**。production/cg/ISMCTS v1・v2.1・v2.2 frozen/Reference Pool v2/Gate2 v2/Strong Opponent 変更なし。
experiment_start_HEAD: `ccdc63c`。

## A. Rollout Audit(Phase A、source=実コード + desktop 実測)
`ismcts._rollout` FULL 経路(`opponent_depth=1`, `max_rollout_steps=40`):
- 開始 = expand した node の action 後 state。`actor=s.yourIndex`、`prev_actor` 追跡。terminal 優先(cutoff/leaf より前に official ±1)。
- **停止**: `actor==ref かつ prev!=ref かつ opp_turns>=1`(= control が ref に戻る=`handoff_return`)、または select 無し/greedy 空/max_steps。
- **実測(desktop, 14 MAIN states)**: Policy forwards/iteration = NODE 0.30(prior/tree)/ FULL 17.26 → **rollout が Policy forward の 98%**
  (~17/iter)。actor 分布 own 56% / opp 43%。単発 v1 FULL rollout は **一貫して 2 turn handoffs で `handoff_return` 停止**
  (own ~5.5 + opp ~9 policy calls、mean 14–17 calls/rollout)。
- **含意(重要)**: **v1 FULL は「2nd handoff で評価」= H2 と等価**(ref-start の場合)。ゆえに genuinely 新しい短縮版は
  **H1(1st handoff で評価=自ターン完了・相手応答前、opp rollout calls を節約)**。H2 は FULL の明示 twin(equivalence 確認用)。

## B. Variants(pre-registered、freeze)
| id | 定義 | 実装 |
|---|---|---|
| **H0 = NODE** | expand した node を即 handcrafted leaf(rollout 無し)= **既存 v2.1** | `leaf_mode=node` |
| **H1 = ONE_HANDOFF** | rollout を **1st turn handoff** で打ち切り handcrafted leaf | `leaf_mode=rollout` + `rollout_cutoff=one_handoff` |
| **H2 = TWO_HANDOFF** | rollout を **2nd turn handoff** で打ち切り(ref-start で FULL と一致) | `leaf_mode=rollout` + `rollout_cutoff=two_handoff` |
| **HFULL = FULL** | 既存 ISMCTS v1 rollout そのもの(opp_turns 基準、byte 不変) | `leaf_mode=rollout`(cutoff 無し=既定 full) |
全 variant で **leaf = 既存 handcrafted evaluator のみ**(Value 禁止)、Policy 735dd38a / Belief / PUCT / backup / determinization / deck 不変。
実装 = `ismcts._rollout` に **core 共有 opt-in `rollout_cutoff`**(既定 full=None → 早期 return 無効 = v1 byte 不変)。agent 変更不要(config の
ismcts ブロックが _rollout へ流れる)。handoff = actor 交代。cutoff は **actor 交代直後の合法 decision 境界**(select 有・pending 途中選択なし)で評価、terminal 優先。

config git blobs: h1 t1350 `4a468d3a` / t3000 `6e06e8e9` / t5500 `31f13f90` / h2 t1350 `da8f5d6c` / t3000 `7a65f47b` / t5500 `dd56df05`。
H0 anchor=`ismcts_v2_1_t*.json`、HFULL anchor=`ismcts_v1_t*.json`(既存 frozen)。

## B1. 仮説(pre-registered)
- **H1(compute)**: rollout horizon 短縮で Policy calls/iteration ↓・iteration latency ↓・iterations/sec ↑。
- **H2(strength recovery)**: H0 より H1/H2 が強い(rollout の最初の数 semantic steps で leaf estimator 品質を大きく回復)。
- **H3(Primary)**: 少なくとも 1 つの truncated variant が equal-wall-time で **v1 に competitive または v1 以上**。

## C. 成功判定(結果前 freeze、task Phase N)
- **Case A(strong success)**: H2≈FULL strength、Policy calls 50%+ 削減、equal-wall-time で FULL 以上 → 成功。次=small rollout Policy distillation。
- **Case B(useful compromise)**: FULL 比わずかに低下も数倍高速、同 wall-time で v1 competitive → deployment 候補。次=small rollout policy/distillation。
- **Case C(only FULL strong)**: かなり長く rollout しないと strength 戻らない(H0/H1/H2 が FULL に大きく届かない)→ truncated 単独不採用。次=small fast rollout Policy。
- **Case D(H1/H2 異常に弱い、< H0)**: boundary/perspective/evaluator-state mismatch を疑い correctness 再確認。
- **Case E(non-monotonic)**: 特定 semantic horizon のみ有効 → leaf-state diagnostic で mechanism 分析。

## D. 評価計画(結果前 freeze)
- correctness: **F0-F7(共有 core 継承)+ F8 H0=v2.1 / F9 FULL=v1 / F10 H1 boundary / F11 H2 boundary / F12 no partial-selection / F13 perspective + terminal 優先**。
- profiling(E/F): 固定 state で H0/H1/H2/FULL の Policy calls/iteration(rollout own/opp 内訳)・latency・iterations/sec・speedup vs FULL。
- leaf-state diagnostic(F): 各 variant の leaf state 特性(prize 差・HP・turn player・handcrafted leaf 値)。
- fixed-iteration(G): 16/32/64 で latency/depth/nodes/selected + **FULL との action agreement**(offline agreement だけで採用しない=v2.2 の教訓)。
- **development equal-wall-time screen(H、Primary budget 1350ms)**: H1/H2 vs Champion abl_5_full、各 100 games、errors/timeout 記録。
  anchor H0=v2.1 0.550 / HFULL=v1 0.677(同 desktop・同 runner・同 control ゆえ流用、曖昧なら再計測)。
- Pareto(I): x=Policy calls/decision(or latency)、y=Champion winrate。**strength retention 閾値 = FULL − 3pt(≥0.647)**(結果前 freeze)。
  candidate = retention を満たす最小 compute の variant、無ければ Pareto 最適点。
- multi-budget(J)+ direct v1 H2H(K): 有望時のみ 3000/5500ms + parent v1 直接比較。
- **結果を見て boundary/threshold/contract を変更しない**。

## E. 結果
### E1. Correctness — **PASS**
- v1 unit 9/9 + cg 6/6 不変(full byte 不変)。v2.1 5/5・v2.2 6/6 不変。**v2.3 boundary 7/7**: F8 node(cutoff 無視)/ F9 full=v1
  (handoff_return)/ F10 one_handoff=1st handoff / F11 two_handoff=2nd handoff / F12 decision 境界のみ / F13 perspective(me=ref)/ terminal 優先。

### E2. Profiling / Fixed-iteration（Phase E/F/G、desktop、14 MAIN states）
| variant | policy/iter | latency@32 | speedup vs FULL | policy cut |
|---|---|---|---|---|
| H0 node | 0.26 | 25ms | 50.8× | 99% |
| **H1 one_handoff** | **7.48** | 399ms | **3.2×** | **66%** |
| H2 two_handoff | 18.19 | 1007ms | 1.3× | 16% |
| FULL | 21.70 | 1296ms | 1.0× | 0% |

- **H1 = policy 66% 削減・3.2× 高速**。**H2 ≈ FULL**(policy cut 16%)= 監査の H2≡FULL を裏付け(2-handoff horizon が v1 の実効 horizon)。
- **action agreement(of 14, vs FULL / vs policy top1)**: iters32 → H1 **14**/12・H2 11/11・H0 11/11; iters64 → H1 12・H2 12・H0 9。
  **H1 は iters32 で FULL と 14/14 一致**(truncated でも同 action)。ただし **offline agreement だけで採用しない**(v2.2 の教訓)。
- **leaf-state diagnostic(handcrafted leaf 値、iters64)**: H0 mean0.598/stdev**0.147**/[0.40,0.83] → H1 0.603/0.179/[0.11,0.89] →
  H2 0.609/0.220 → FULL 0.561/**0.228**/[0.04,0.93]。**rollout が長いほど leaf 値の分散拡大**(より決定的局面に到達、handcrafted が鋭い signal)。
  = H2 仮説(rollout の最初の数 step で estimator 品質回復)の機構的裏付け。fixed-iter で depth はほぼ同一(cutoff は tree 構造不変)。

### E3. Development Equal-Wall-Time H2H（Phase H、1350ms、vs abl_5_full）
実行: desktop, workers15, alternate_sides, mirror, δ_min=0.05/α.05/β.10。errors 0 / timeout 0。anchor H0/HFULL は同 desktop 既測を流用。

| variant | policy/iter | winrate [Wilson95] | vs Champion |
|---|---|---|---|
| H0 node(v2.1) | 0.26 | 0.550 | anchor(既測) |
| **H1 one_handoff** | 7.48 | **0.580** [0.482, 0.672] | SPRT None(llr +1.10) |
| **H2 two_handoff** | 18.19 | **0.630** [0.532, 0.718] | SPRT None(llr +2.11、CI下限>0.50) |
| HFULL(v1) | 21.70 | 0.677 | anchor(既測) |

- **strength は rollout horizon と単調**: 0.550 < 0.580 < 0.630 < 0.677。
- H1(66% policy 削減)= 0.580(H0比 +0.03 = H0→FULL gain 0.127 の **24% 回収**)。H2(16% 削減、≈FULL)= 0.630(gain の **63% 回収**)。
- H2 は Champion に勝ち越し(CI 下限 0.532)、FULL とは CI 重複で有意差なし。H1 は Champion 互角〜やや上(CI 0.482–0.672)。
- 判定: **H1(compute)✓ / H2(strength recovery)✓**(H0<H1<H2 単調)/ **H3(Primary: truncated が equal-wall-time で v1 competitive/以上)✗**
  (H2 が最接近も 0.630<0.677、かつ compute 節約 16% で効率変数として無意味)。

### E4. Pareto / Multi-Budget / Direct v1（Phase I/J/K）
**Pareto**(x=policy calls/iter, y=Champion winrate): H0(0.26,0.550)→H1(7.48,0.580)→H2(18.19,0.630)→FULL(21.70,0.677)。
frontier は **curve 全体** = 各 rollout 増分が strength を compute にほぼ比例して買う。「most of FULL strength を大幅低 compute で」という **knee は存在しない**。
- **strength retention(pre-reg 閾値 FULL−3pt=0.647)**: H2=0.630 / H1=0.580 とも **未達**。
- H2 は strength 最接近だが policy cut 16%(≈FULL)= 効率 candidate でない。H1 は 66% cut だが gain の 24% しか回収せず。
- ゆえに **「有望 candidate」不在 → Phase J(multi-budget 3000/5500)/ Phase K(direct v1)は「有望時のみ」の条件不成立で不実施**(v2.2 と同判断)。
- 統計注記: n=100 で Wilson CI 幅 ~±0.09、H1/H2/FULL は個別 pairwise では有意分離せず。ただし単調 trend は profiling(leaf 分散拡大)と
  v2.1/v2.2 の一貫パターン(rollout を削ると弱くなる)に整合し、構造的結論は頑健。

### E5. Conclusion（Case A-E）
**Case C 確定(only near-full rollout is strong)**。strength は rollout horizon と単調で、compute を節約する truncation(H1, 66% cut)は
strength gain の大半を失い(0.580)、strength を保つ H2(0.630)は compute をほぼ節約しない(16%≈FULL)。**「v1 strength の大部分を大幅低 compute で」
という truncation point は存在しない**。H3(Primary)✗。

**機構**: profiling の leaf-state diagnostic どおり、rollout が長いほど leaf 値の分散が拡大(H0 stdev 0.147 → FULL 0.228)= より決定的な局面に
到達して handcrafted が鋭い signal を出す。数手 rollout(H1)では中間的にしか回復しない。これは v2.1(no rollout=Case C)/ v2.2(value leaf=Case D)が
示した「**full rollout の先読みが strength に本質的**」と整合する。

**決定**:
- **truncated rollout 単独は不採用**。ISMCTS v1(full rollout, 59384591)を strength baseline として維持。
- **Phase J/K 不実施**(有望 candidate 不在)。boundary/threshold/contract は結果を見て変更せず。
- **次の 1 ステップ(pre-reg Phase N Case C)= full rollout horizon を維持したまま rollout Policy 自体を安くする「small fast rollout Policy」**。
  根拠: rollout の逐次 policy forward が compute の 98%。horizon を削る(v2.3)と strength を失うので、**horizon は保ち rollout 内の policy を
  軽量化**する。これは v2.2 Case D の「strong v1 teacher からの Policy distillation」と収束 = 証明済みに強い v1 の探索を、提出時 feasible な
  compute で得るのが一貫した筋。**today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`(実験中 commit なし・他者 commit なし)。
ISMCTS v1 59384591 / v2.1 / v2.2 / Champion abl_5_full / Policy 735dd38a / Value 0657f7af / Belief / handcrafted evaluator本体 /
Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。`ismcts._rollout` は `rollout_cutoff` 既定 full=v1 byte 不変(opt-in)。
production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
