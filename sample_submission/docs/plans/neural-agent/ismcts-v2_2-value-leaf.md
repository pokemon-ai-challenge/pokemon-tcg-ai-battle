# ISMCTS v2.2 — Value-Leaf Ablation(pre-registration + 結果)

作成日: 2026-07-27
parent: ISMCTS v2.1 leaf-at-node(rollout 省略・expand ノードで handcrafted leaf 即評価)。v2.1 は 10–37× 高速だが equal-wall-time で
Champion 互角どまり(Case C)= **handcrafted-at-node の leaf estimator の質不足**、rollout の policy-greedy 先読みが strength に本質的、と判明
(doc=ismcts-v2_1-leaf-at-node.md §6.5)。
Primary Research Question: **高価な Policy rollout(未来予測)を、既存 Value Network の 1 forward で近似できるか**。
ステータス: **pre-registration(online H2H 前に freeze)**。production/cg/ISMCTS v1・v2.1 frozen/Reference Pool v2/Gate2 v2/Strong Opponent 変更なし。
experiment_start_HEAD: `ccdc63c`(他者の out-of-band commit `chore(share)` 以後。自分の git 操作なし)。

## 1. 唯一の正式差分(pre-registered)
| | v2.1 | **v2.2** |
|---|---|---|
| expansion 後 leaf | handcrafted evaluator を即評価 | **既存 Value Network を即評価** |
差分は **leaf evaluator のみ**(handcrafted → value)。以下は v2.1 と**完全同一**: leaf_mode=node(rollout 無し)/ information-set tree /
determinization(match_context estimated)/ Belief / Policy prior(735dd38a)/ PUCT / expansion / backup / value domain [-1,+1] / terminal
handling / lethal / fallback(**handcrafted のまま**)/ deck / time-budget。**Policy 再学習 / Value 再学習 / Belief 変更 / progressive
widening / tree reuse / macro action は入れない**。Value 改善は次 variant。

## 2. 実装(core 共有・v1/v2.1 frozen behavior 不変)
`challengers/ismcts_v1_agent.py::_get_evaluator` に **ISMCTS 専用 leaf キー `ismcts.leaf_eval`** を追加。解決順は
`ismcts.leaf_eval` → `pipeline.leaf_eval`(Champion handcrafted)→ handcrafted 既定。**v1/v2.1 config は `ismcts.leaf_eval` を持たない
ため従来 handcrafted に解決 = byte 単位で不変**(v1 unit 9/9 + cg 6/6 + v2.1 5/5 全再 PASS で保証)。pipeline fallback は `pipeline.leaf_eval`
(handcrafted)を使い続けるため **fallback 挙動も不変**(ISMCTS leaf だけ value 化)。v2.2 config = v2.1 config + `ismcts.leaf_eval={"kind":"value"}`。
leaf 評価は既存 `ptcg_ai/search/leaf_eval.ValueModelEvaluator`(perspective 整列 + terminal override 実装済み)を read-only 再利用。

## 3. Value Audit(Phase A、source=実コード)
- **artifact**: `sample_submission/ptcg_ai/learning/value_weights.json`(git blob `0657f7af`, 269KB)。推論 = `value_model.ValueModel`
  (純Python MLP, 166 feature = `encoder.FEATURE_NAMES`, 3層 relu/relu/sigmoid, turn-band 温度較正)。
- **training origin**: Kaggle 上位リプレイ ~4,698 件、ラベル = 最終勝敗(top-level `rewards`)を全局面へ付与、episode 単位 split、rank 重み。
- **output semantics**: `state.yourIndex` 視点の **P(win) ∈ [0,1]**、turn-band 温度較正(**T≥1.0 制約 = 慎重側**、`design.md` §4.3)。
- **perspective(C3)**: `ValueModelEvaluator.evaluate(state, me)` が `state.yourIndex==me` 以外で `1-p` に整列 → tree の root_me 視点 [-1,+1]
  へ `leaf_to_value(p)=2p-1` で写像(ismcts `_rollout` node 分岐)。**terminal は Value を呼ばず official ±1/0 を優先**(C4)。
- **input compat(A2)**: search leaf State → `encode_state_from_state` 直結。**新特徴の追加なし**。
- **leakage(A3/F13)**: encoder は**公開情報のみ**(相手側は `handCount`/`deckCount`/prize・discard・bench の枚数と場のポケモンのみ、
  相手手札の中身・山札順・サイド中身は読まない)。ゆえに determinization の hidden truth は Value 入力に**入らない**(F13 は encoder が
  hidden を読まない性質から成立)。
- **既知の質/OOD リスク(結果前に明記)**: offline AUC 0.746(較正後)。**ターン帯で非一様**(1-2=0.530 ほぼコイントス / 6-10=0.778 /
  11+=0.839)。**自デッキ alakazam=0.715(主要中で低め)**。probe で **deck_count に盲目**(山札切れ非検知、ただし handcrafted も同様=
  v2.1 比で退行ではない)。→ 序盤 leaf と山札切れ局面では value が弱い可能性。
- **STOP 条件(A4)**: Case1 変換不能=否/Case2 ground-truth hidden 要求=否/Case3 artifact 不明=否/Case4 根本非互換=否。**全て非該当 → 実装可**。

## 4. 仮説(pre-registered)
- **H1(efficiency)**: value 1 forward は v1 の rollout(逐次 policy-greedy)より大幅に安い。
- **H2(estimator quality)**: value leaf は v2.1 handcrafted-at-node より rollout-backed 推定に近い(高品質)。
- **H3(equal-wall-time、Primary)**: 同 wall-time で **v2.2 ≥ v1 rollout**。

## 5. 成功判定(結果前 freeze)
- **Case A(理想)**: equal-wall-time で **v2.2 > v1** → 高価な rollout を 1 value forward で代替成功。
- **Case B(competitive)**: v2.2 ≈ v1 かつ大幅に軽い compute → teacher 生成/提出構成として有望。
- **Case C(better than v2.1 but below v1)**: v2.1 < v2.2 < v1 → value は handcrafted-at-node を改善したが rollout 代替に不足。次は Value 改善/蒸留。
- **Case D(v2.1 並み)**: v2.2 ≈ v2.1(≈Champion 互角)→ 既存 Value はこの leaf 用途に不十分。単純 value leaf 不採用。
- **Case E**: interface/leakage/OOD で成立せず。

## 6. 評価計画(結果前 freeze)
- correctness: **F0-F7(v1 と同一 suite、共有 core ゆえ継承)+ F8 no-rollout / F9 leaf-point / F10 perspective / F11 domain / F12 terminal
  override / F13 hidden leakage**。
- offline value sanity(E): 固定 MAIN state で A=handcrafted-node / B=value-node / C=rollout-backed(v1)を比較(|A-C|,|B-C|,corr)+ 実 ISMCTS
  leaf の value 予測ヒストグラム(saturation/OOD)。**この diagnostic で Value を選び直さない**(artifact freeze 済)。
- profiling(F): value ms/forward、policy/value forwards per iteration、iteration latency、v1 比 speedup、v2.1 比 cost。
- fixed-iteration(G): 8/16/32/64/128 で latency/depth/nodes/selected + v1・policy top1 一致率。
- **equal-wall-time H2H(Primary、I)**: budget ∈ {1350, 3000, 5500}ms、各 100 games、control=abl_5_full、mirror・alternate_sides、
  δ_min=0.05/α.05/β.10。**v1/v2.1 は既測(ismcts-v2_1-leaf-at-node.md §6.4)を同軸比較に流用**、今回 **v2.2 を追加**。
- direct v1 vs v2.2(K): 有望時のみ、本命 budget 1 条件。
- **結果を見て契約変更しない**。

## 7. Freeze（online H2H 前）
| item | value |
|---|---|
| challenger_id | ismcts_v2_2_value_leaf |
| parent | ismcts_v2_1(leaf_mode=node) |
| value model | `value_weights.json` git blob `0657f7af` |
| value semantics | P(state.yourIndex wins)∈[0,1]、turn-band 温度較正(T≥1.0) |
| leaf mode | node（rollout 無し）+ ismcts.leaf_eval.kind=value |
| Policy | 735dd38a（不変） |
| deck | production deck.csv（不変） |
| PUCT c_puct | 1.4 |
| world_pool_size | 8 |
| determinization | match_context estimated（信念） |
| wall-time budgets | 1350 / 3000 / 5500 ms（v1/v2.1 と同一軸） |
| config git blobs | t1350 `a867362e` / t3000 `5eee4f64` / t5500 `23d87231` / hc8 `f87ec1cf` / hc16 `9a731773` / hc32 `2d33b1b7` / hc64 `850f1cd4` / hc128 `5542cc49` |
| Gate | large algorithmic change（δ_min=0.05）。Champion 共通 control。 |

## 8. 結果
### 8.1 Correctness — **PASS**
- **v1 不変**: leaf 追加後も v1 unit 9/9 + cg 6/6 PASS。**v2.1 不変**: 5/5 PASS(`_get_evaluator` 変更は後方互換 opt-in)。
- **v2.2 新 test 6/6 PASS**: F8 no-rollout(value leaf で policy rollout 未呼び出し)/ F9 leaf-point(評価対象=expand した node の state)/
  F10 perspective(`yourIndex==ref` で p、他で 1-p、符号が root 視点で反転)/ F11 domain(p=1→+1, 0.5→0, 0→-1)/ F12 terminal override
  (terminal で value 未呼び出し・official ±1)/ F13 hidden leakage(相手手札・サイド中身を変えても encoder 出力不変、handCount 変化は反映)。

### 8.2 Offline value sanity（Phase E、A=handcrafted-node / B=value-node / C=rollout-backed=v1）
desktop, 固定 18 MAIN states(branching med=6)。

| | mean | mean\|·−C\| | corr(·,C) |
|---|---|---|---|
| A handcrafted-node | 0.542 | 0.115 | 0.538 |
| B **value-node** | 0.551 | **0.098** | **0.851** |
| C rollout-backed(=v1) | 0.464 | — | — |

→ **H2 強く成立**: value leaf は rollout-backed 推定を **corr 0.851** で追従(handcrafted-node は 0.538)。「高価な rollout 先読み値を
1 value forward で近似できる」兆候。
**E3 OOD(実 ISMCTS leaf の value 予測ヒストグラム、v2.2 hc128、n=2303)**: min 0.260 / max 0.829 / mean 0.538 / stdev 0.120、
分布 `[.3)=20% [.4)=12% [.5)=33% [.6)=22% [.7)=7% [.8)=1%`。**saturation なし・0.26–0.83 に健全に分散**(0.5/0.99 張り付き無し)。

### 8.3 Profiling（Phase F、desktop、iters=32）
| variant | latency | policy forwards | value forwards |
|---|---|---|---|
| v1 rollout | 785ms | 481 | 31 |
| v2.1 handcrafted-node | 22.7ms | 7.7 | 32 |
| **v2.2 value-node** | **38.4ms** | 7.4 | 32 |
→ **value ms/forward ≈ 0.49ms**(policy 1.63ms/forward の約 1/3)。**v2.2/v1 ≈ 20.5× 高速**(H1 成立)、v2.2/v2.1 cost ≈ 1.69×。
v1 の iteration コストは rollout の逐次 policy(481 forwards/32iter ≈ 15/iter)が支配、v2.2 は value 1 forward/iter で置換。

### 8.4 Fixed-iteration（Phase G、desktop、8/16/32/64/128）
| variant | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| v1 rollout | 206ms d2 | 419ms d2 | 814ms d3 | 1749ms d5 | 3819ms d6 |
| v2.1 handcrafted | 6ms d2 | 12ms d2 | 23ms d3 | 48ms d5 | 102ms d6 |
| **v2.2 value** | 12ms d2 | 21ms d2 | 40ms d3 | 79ms d5 | 158ms d6 |

**v2.2 selected-action agreement**(of 18): vs v1 = 11/15/16/**18**/15、vs policy top1 = 15/16/**18**/**18**/16(budget 8/16/32/64/128)。
→ 同 iterations では depth 同一(leaf mode は tree 構造不変)、v2.2 は iter を上げると **v1 rollout と同じ action に収束**(64 で 18/18)。

### 8.5 Equal-Wall-Time H2H（Primary、H3、vs abl_5_full、各 100 games、budget {1.35, 3, 5.5}s）
実行: desktop, workers15, alternate_sides, mirror, δ_min=0.05/α.05/β.10。errors 0 / timeout 0。**全 SPRT が FUTILITY 早期決定**。
v1/v2.1 は既測(ismcts-v2_1-leaf-at-node.md §6.4)を同軸流用。

| budget | v1 rollout | v2.1 handcrafted-node | **v2.2 value-node** [Wilson95] SPRT |
|---|---|---|---|
| **1350ms** | 0.677 | 0.550 | **0.366** [0.264,0.482] **FUTILITY**(N=71) |
| **3000ms** | 0.660 | 0.480 | **0.408** [0.316,0.507] **FUTILITY**(N=98) |
| **5500ms** | 0.620 | 0.490 | **0.410** [0.319,0.508] **FUTILITY**(N=100) |

**観察(決定的)**:
- **v2.2 value-leaf は 3 budget すべてで Champion に FUTILITY**(formal 早期敗退、CI 上限 <0.51)。
- **順位は v2.2 < v2.1 < v1**: value-at-node は **handcrafted-at-node よりさらに弱い** leaf estimator(online)。budget 増でも伸びない(0.37→0.41→0.41)。
- **判定**: H1 ✓ / **H2(offline)✓ だが H3(Primary: equal-wall-time で v2.2 ≥ v1)✗**、かつ **v2.2 < v2.1**。
- **offline→online 乖離**: Phase E corr(value, rollout-backed)=0.851 / Phase G action 一致 18/18 は **online strength に転移せず**。

### 8.6 Conclusion（Case A-E）
**Case D 確定（既存 Value はこの leaf 用途に不十分）— さらに v2.2 < v2.1 で「handcrafted より弱い」**。

**なぜ offline が強いのに online で負けるか(機構仮説)**: value 予測は **leaf 間コントラストが低い**(E3: stdev 0.12、多くが 0.5 近傍、
序盤 AUC 0.53・自デッキ alakazam 0.715)。search は sibling leaf の **discrimination** が本質で、低コントラストな value は equal-wall-time で
得た ~20× iterations を **noise 駆動**にし、robust な policy prior から誤って乖離させる。handcrafted の prize-anchored value は offline 相関は
低く見えても、search には **単調で誤誘導しない signal** を与える(prize を取る線が確実に高評価)。offline の corr/agreement(18 state・
equal-iteration 測定)は online strength(native high-iteration)に **転移しなかった**([[project_measurement_protocol]] /
[[project_pimc_prod_validation]] の再現: 研究の勝ちが本番に転移しない)。

**決定**:
- **単純 value leaf は不採用**。ISMCTS v1(rollout, config 59384591)を frozen strength baseline として維持。
- **Phase K(direct v1 vs v2.2)は実施せず**(「有望時のみ」の条件不成立=FUTILITY)。
- **既存 Value をこの experiment 内で再学習/再較正しない**(Phase L 遵守)。Value 改善は別 variant。
- **次の 1 ステップ(pre-registration Phase N Case D)= strong v1 rollout ISMCTS を直接 Teacher とした Policy distillation**。
  根拠: v1 は scaling で Champion 超えを実証(64/128 iter で formal PROMOTE、equal-wall-time 全 budget 勝ち越し)。efficiency 経由の
  leaf 置換(v2.1 handcrafted / v2.2 value)は **両方とも v1 強度を安価に回収できず**。ゆえに「証明済みに強い v1 の探索改善行動
  (visit 分布 / root action)を Policy へ蒸留し、提出時に rollout 無しで v1 的強度を得る」のが筋。**today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`(実験中 commit なし・他者 commit なし)。
ISMCTS v1 config 59384591 / v2.1 frozen / Champion abl_5_full / Policy 735dd38a / **existing Value artifact `0657f7af`(再学習なし)** /
Belief / handcrafted evaluator本体 / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。`ismcts.py` は既定 rollout=v1 不変、
`_get_evaluator` は後方互換 opt-in(v1/v2.1 byte 不変)。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
