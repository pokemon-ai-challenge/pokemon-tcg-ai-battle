# ISMCTS v2.8 — Advantage-Gated Search-Correction Distillation(pre-registration + 結果)

作成日: 2026-07-27
parent: v2.7(HIGH_CONF visit-confidence gate、Case C)。唯一差分 = **gate を visit-confidence → Search Q advantage
`ΔQ = Q(a_search) − Q(a_orig)`**。dataset/split/H32/Original 初期化/Teacher(v2.4 H4 @128)は v2.5-2.7 と同一。experiment_start_HEAD `ccdc63c`。

## A. Root Q semantics 監査
`SearchStats.root_children_full=[(action,visits,Q,prior)]`。root node の `Q(a)=W[a]/N[a]`、W は `signed_for(v_ref,to_move=root_me,root_me)=v_ref`
ゆえ **root_me 視点・[-1,+1]**。a_search/a_orig とも root の子=**同一視点で比較可能**。visit=0 の action は Q=0(既定・不正)→ a_orig unvisited は除外。
実測: train CHANGED 1,274 中 **valid-Q 1,274 / unvisited 0**(Q 信頼できる)。

## B. ΔQ 分布(train CHANGED, valid-Q)
mean +0.296 / median +0.202 / std 0.303 / range [−0.071, +1.896]。**positive ΔQ 99.1%**(Search action の Q が Original より高い)。
percentile(positive): P25 +0.098 / P50 +0.205 / **P75 +0.398** / P90 +0.644 / P95 +0.947。
**visit-confidence との相関**: corr(ΔQ, gap)=**+0.548** / corr(ΔQ, share)=+0.388 / corr(ΔQ, entropy)=−0.106。→ ΔQ と visit-confidence は
中程度相関(別物ではないが同一でもない)。

## C. Advantage threshold（train-only freeze）
positive-ΔQ CHANGED の **P75 = +0.398(A75, primary)** / P90 = +0.644(A90)。**train split のみ**から算出し val/test へ適用(leakage 防止)。

## D. ADV_CHG counts / HC overlap
A75: ADV_CHG train/val/test = 316/64/51。overlap(train): ADV∩HC=192 / ADV_only=124 / HC_only=221(≈ 40% 重複)。A90: 127/26/22。

## E-F. Objective / Student
`L = mean_ALL[KL(π_original‖student)] + λ_adv·mean_ADV_CHG[KL(π_search‖student)]`(correction は ADV_CHG subset 内平均）。λ_adv∈{1,2}。
Student = H32、Original 735dd38a 初期化(容量は増やさない)。retention guard 0.90(pre-registered、v2.7 の 0.95 より緩め)。

## G-I. Offline 結果
訓練 trajectory(val): λ=1 CHG_rec 0.19–0.20 / retention **0.79–0.83**、λ=2 CHG_rec 0.22–0.24 / retention **0.75–0.79**。
= **corrections は学習できるが retention 0.90 guard を毎 epoch 割る**(v2.7 と同パターン)。best-checkpoint は Original に退避。
**advantage-gating は interference を減らさなかった**(ΔQ と visit-gap が相関するため、ADV gate ≈ HC gate)。

## J-L. Policy-only H2H（preflight: abs-path + is_ready + sha1、fallback 0、sanity 0.53≈0.50）
| candidate | vs Original winrate [Wilson95] | 判定 |
|---|---|---|
| best-checkpoint(guard=Original 退避) | =Original(未計測、構造上 ~0.50) | 中立 |
| **adv_l1_final(A75, ret 0.79, 高ΔQ 学習)** | 0.459 [0.386, 0.534] FUTILITY | **中立** |
| **adv_l2_final(A75, ret 0.76)** | 0.533 [0.477, 0.589] None | **中立** |

historical: v2.5 0.5167 / v2.6 0.530–0.540 / v2.7 ≈0.50 / **v2.8 ≈0.50**。

## M. Conclusion（Case C）+ Interpretation
**Case C(Offline advantage recovery / Online neutral)確定**。**高 ΔQ correction は学習できる**(CHG_rec 0.19–0.24、ΔQ 上位も学習)、
**だが online は中立**(0.46–0.53、有意な transfer なし)。**Phase I の答え**:
- High-advantage correction は Policy に学習できたか → **Yes**(retention を犠牲にすれば)。
- High-advantage recovery は high-confidence より有望か → **No**(ΔQ と gap が相関、挙動ほぼ同一)。
- UNCHANGED interference は改善したか → **No**(retention 0.76–0.83、v2.7 と同水準)。
- ΔQ が大きいほど recovery したか / offline advantage が online へ転移したか → **online neutral = 転移せず**。
→ **root Q difference(128-iter H4)自体が transferable deployment strength の良い教師ではない**。

## O. Stop Rule 発動
**search-to-root Policy distillation は v2.5 / v2.6 / v2.7 / v2.8 の 4 連続 online neutral**(dilution → anchor → visit-confidence → Q-advantage、
gating を尽くしても転移せず)。pre-registered Phase O に従い、**この研究線(loss/gating 微調整)を一旦停止する**。H4 統合不実施。
**次の大きな lever をレビュー(Phase O1 候補)**: (1) **Search-trained Value**(root Q/outcome を Value target に。Policy でなく Value へ蒸留)、
(2) **ISMCTS そのものの改善**、(3) **v2.4 H4 deployment optimization**(唯一の成功=提出候補の詰め)、(4) Self-play/RL 準備。
Adapter/capacity 分離は「advantage に online 価値の兆候があったのに interference で失った」場合のみ優先 → **今回 online 価値の兆候なし**ゆえ Adapter は非優先。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。runner: 絶対パス+preflight(is_ready・sha1・fallback 0)。
ISMCTS v1 59384591 / v2.1–v2.7 / v2.4 H4 e7e74ac7 / Original Policy 735dd38a / H4 rollout student / Value 0657f7af / Belief / handcrafted leaf /
Champion / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。Q は gating(教師選別)のみ使用、loss には未使用。追加 dataset/Teacher/iterations 変更なし。
production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
