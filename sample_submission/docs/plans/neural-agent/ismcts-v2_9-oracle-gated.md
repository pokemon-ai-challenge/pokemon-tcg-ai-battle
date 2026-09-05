# ISMCTS v2.9 — Oracle-Gated Hybrid Evaluation(学習なし診断 + 結果)

作成日: 2026-07-28
目的: search-to-root Policy distillation 4 連続 online neutral(v2.5-2.8)が **signal 失敗**なのか **learning/capacity 失敗**なのかを、
**学習を完全に排除して**分離する。各 MAIN root で Teacher Search(v2.4 H4 @128 fixed iters)を実行し、arm の gate 発火時に
**Search top1 を 100% 採用**、そうでなければ Original top1 を 100% 維持 = 「correction を完璧に転移できた場合の online upper bound」。
Policy 学習・変更なし。experiment_start_HEAD `ccdc63c`。

## A. Teacher / gate 定義(全て frozen、新規 threshold なし)
Teacher = **v2.4 H4 @128 iters**(rollout=H4 student, tree=Original 735dd38a, handcrafted leaf, Belief/PUCT frozen)。`SearchStats` から
Search top1=`selected_action`(most-visited)/ Original top1=`root_policy_top1`(prior argmax=Original Policy argmax)/ visits・Q=`root_children_full`。
- **ALL**: CHANGED(Search top1 ≠ Original top1)なら Search。
- **V**: CHANGED & top1 visit share≥0.5 & top1-top2 gap≥0.2(**v2.7 gate exact**)。
- **Q**: CHANGED & a_orig visited & ΔQ=Q(search)−Q(orig)≥0.398(**v2.8 A75 gate exact**)。
- gate off = Original Policy argmax(= abl_2_policy_only の選択と一致)。lethal/pipeline なし(純 Policy + gate)。

## B. Correctness（Phase K、mock 6/6 PASS）
K2 ALL=CHANGED iff / K3 V=v2.7 exact(境界含む)/ K4 Q=v2.8 exact / K4b Q は a_orig unvisited を除外 / CONTROL never。
shadow no-side-effect(K1/K10): ismcts.search は別 searchId 上で real game を advance しない(v1 F7/cg 済を継承)。

## C. Runner
candidate=oracle_agent(ORACLE_ARM env、MAIN root で H4@128 search→gate)、control=abl_2_policy_only(純 Original、search なし)。
search は side-effect 無し=winrate 等価ゆえ control の shadow-search は省略(compute 半減)。errors 0。delta_min=0.03、SPRT。

## D. 結果（Oracle upper-bound、vs Original Policy-only、errors 0）
| arm | games | winrate [Wilson95] | Δ vs 0.50 | SPRT |
|---|---|---|---|---|
| **ALL** | 87 | **0.7931** [0.697, 0.865] | **+29.3pt** | **PROMOTE** |
| **V**(v2.7) | 83 | **0.8072** [0.710, 0.878] | **+30.7pt** | **PROMOTE** |
| **Q**(v2.8) | 139 | **0.6906** [0.610, 0.761] | **+19.1pt** | **PROMOTE** |

**3 arm すべて巨大正効果で formal PROMOTE(SPRT 早期停止)**。gate 発火率(dataset 由来、Teacher 同一): ALL≈25.4% / V≈8.5% / Q≈5% の MAIN root。

## E. Teacher Signal Audit（Phase H、必答)
- ALL correction 自体に online value はあるか → **Yes、+29.3pt(巨大)**。
- Visit-confidence(V)correction に value はあるか → **Yes、+30.7pt(最大)**。
- Q-advantage(Q)correction に value はあるか → **Yes、+19.1pt**。
- ideal recovery=100% での gain は何 pt か → **+19〜31pt(upper bound は小さくない、大きい)**。
- **v2.5-2.8 neutral は learning failure か signal failure か → 明確に learning/capacity failure**(signal は極めて価値がある)。
- capacity adapter へ投資する根拠はあるか → **強く Yes**。

## F. Conclusion（Case E: V/Q ともに positive、実際は ALL も）
**Case E 確定(全 arm 明確に positive)**。search-to-root correction には **明確で大きな online value**(+19〜31pt)がある。
**v2.5-2.8 の 4 連続 neutral は signal 失敗ではなく、同サイズ H32 Policy が correction を interference なしに学習できなかった learning/capacity 失敗**だった。
→ **v2.8 の「search-to-root 停止」判断を撤回**。Oracle 診断が signal の価値を実証した。

**次の 1 ステップ(pre-reg Phase J Case C/D/E)= Frozen Original + Residual Correction Adapter**。Original Policy 735dd38a を**凍結**し、
**correction 専用の追加容量(residual adapter head)**を足して、UNCHANGED を壊さずに correction を学ぶ。teacher = Oracle の action rule。
gate は effect size と simple さから **V(v2.7、+30.7pt・8.5%・最も selective で学習容易)または ALL(+29.3pt・全 correction)** を候補。
compute 注記: Oracle は各 root で H4@128 search を使う診断であり**提出候補ではない**(deployment には adapter 蒸留が必要)。**today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。**学習・重み変更なし**(frozen models のみ使用)。runner: 絶対パス preflight・
control=abl_2_policy_only・fallback 0。ISMCTS v1 59384591 / v2.1-v2.8 / v2.4 H4 e7e74ac7 / Original Policy 735dd38a / H4 rollout student /
Value 0657f7af / Belief / handcrafted leaf / Champion / **Reference Pool v2 14db8345 不使用** / Gate2 v2 05e509a2 すべて不変。
production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。新規 research asset のみ(oracle_agent/oracle_h2h/test_oracle/config)。
