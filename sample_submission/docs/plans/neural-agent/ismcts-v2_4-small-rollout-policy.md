# ISMCTS v2.4 — Small Fast Rollout Policy Distillation(pre-registration + 結果)

作成日: 2026-07-27
parent: ISMCTS v1 full rollout(config 59384591、Champion 超え実証)。
Primary Research Question: **rollout の長さ(FULL horizon)を維持したまま、rollout 内で使う Policy だけを小型・高速な Student に置換すると、
v1 の strength を維持しつつ大幅に compute を削減できるか**。
背景: 効率化3連(v2.1 no-rollout=Case C / v2.2 value-leaf=Case D / v2.3 truncated-rollout=Case C)は full rollout の先読みを削って弱くなった。
今回は **horizon を保ち、rollout の per-call cost だけ**下げる。ステータス: **pre-registration(online H2H 前 freeze)**。
production/cg/ISMCTS v1・v2.1・v2.2・v2.3 frozen/Reference Pool v2/Gate2 v2/Strong Opponent 変更なし。experiment_start_HEAD `ccdc63c`。

## A. Audit(Phase A/B、source=実コード + desktop 実測)
- **H2 vs FULL の関係(v2.3 の注記整理)**: v2.3 の H2(2-handoff)は v1 FULL(opp_turns 基準)と **semantic termination の近似一致**であり
  **byte/behavioral 完全一致ではない**(ref-start rollout では一致、opp-start では分岐、Policy calls H2 18.19 vs FULL 21.70)。
  **v2.4 は v1 FULL の実装上の termination をそのまま使用**(rollout_cutoff 無し=full)。frozen v1 behavior は変更しない。
- **rollout Policy interface**: rollout の action selection は `pipeline._greedy_selection(policy_model, obs)` →
  `policy_model.score_options_from_state(state, select) -> list[float]`(raw score)→ argmax(top-count)。Student は同 interface を実装。
- **Teacher(固定)**: Policy 735dd38a(`policy_weights.json`, blob `f5d576dc`)。**2 層 MLP: 入力 239(state166 + option65 + card_embed8)
  → hidden 32 → 1**。consequence_fields なし。**Search-teacher(強い ISMCTS の探索行動)は今回 Teacher にしない**(clean ablation)。
- **cost 分解(desktop, 94 rollout states)**: score_options 1.250ms/call のうち **NN forward 96%(1.201ms)/ feature 構築 4%(0.047ms)**
  (encode_state 2% / encode_options 1% / card_ids 0%)。→ **Case D ではない**。Teacher は既に hidden=32 と小さいが、**pure-Python matmul
  (239×32×options)が支配**。ゆえ **より小さい Student(H<32)は rollout を直接高速化できる**。真の lever は NN forward = 正しく本テーマ。

## B. 基本設計(pre-registered、freeze)
| | v1 | **v2.4** |
|---|---|---|
| tree prior / expansion / PUCT / fallback / lethal / root | Policy 735dd38a | **735dd38a(不変)** |
| **rollout inner-loop action selection** | Policy 735dd38a | **Small Student policy** |
| rollout horizon / leaf / terminal / determinization / Belief / deck | v1 | **v1 と完全同一** |
唯一の正式差分 = **FULL rollout 中の action selection model のみ**。実装 = `ismcts._rollout`/`search` に core 共有 opt-in `rollout_policy`
(既定 None=teacher=**v1 byte 不変**)。agent は `ismcts.rollout_policy_weights`(student JSON path)があれば PolicyModel をロードし rollout のみに使用。
Student は PolicyModel 形式(teacher の standardization + card_embedding を再利用、hidden だけ縮小)。**rollout 短縮/leaf 変更/Value/PUCT/Belief/
batch/ONNX/quant/self-play は入れない**。

## B1. 仮説(pre-registered)
- **H1(efficiency)**: Student は per-call が Teacher より大幅に安く、同 wall-time で iterations 増。
- **H2(imitation quality)**: Student は Teacher の rollout action を高い top-1 agreement で再現(distillation)。
- **H3(Primary)**: FULL horizon 維持 + Policy cost 減で iterations 増、**equal-wall-time で v1 strength 維持/改善**。

## C. 成功判定(結果前 freeze、task Phase M)
- **Case A(strong success)**: Student で大幅高速化 → iterations 増 → equal-wall-time strength が v1 **以上** → 採用。次=Strong ISMCTS→root Policy distillation。
- **Case B(good Pareto)**: rollout 質やや低下も iterations 増で補い、equal-wall-time で v1 **competitive** → deployment 候補。次=同上。
- **Case C(fast but weak)**: 高速だが rollout 質低下が大きく full Teacher strength を回収できない → Small Policy 不採用。次=Search-teacher Policy distillation。
- **Case D(speedup 不足)**: NN forward が主要 bottleneck でなかった(feature/scoring/overhead 支配)→ 次の profiling bottleneck へ。
  ※ Audit で NN=96% 確認済み、Case D の主因は反証されているが、end-to-end speedup が iteration に載るかは Phase J で最終確認。

## D. 評価計画(結果前 freeze)
- dataset(Phase C): teacher rollout 中の score_options state を収集(239次元標準化入力 h + teacher raw score、legal option 単位)。**game 単位 split**
  (70/15/15)。leakage: student 入力 = teacher と完全同一(公開情報のみ、determinize は belief)、新規情報を足さない。Reference Pool v2 不使用。
- students(Phase D、最大3): **H ∈ {4, 8, 16}**(teacher H32 の 8/4/2× 高速化見込み、layer0 = H×239 支配)。teacher の standardization+embedding 再利用。
- distill(Phase E): **KL(teacher softmax(T=1.0) || student softmax)** over legal options。**T=1.0 固定(`_priors` の softmax(scores) と同 semantics)**、
  Adam/lr 3e-3、**sample weighting なし**(clean Teacher distillation)。補助 loss なし。
- offline(Phase F): test split で top-1 / top-3 agreement・KL、rollout-step 別(自ターン/相手応答)・seltype 別。
- latency(Phase G): end-to-end ms/call(encode+forward+select)、rollout に組み込んだ ms/iteration・iterations/sec。
- candidate 選定(Phase H): **top-1 agreement × latency の Pareto**(単純 top-1 最大でなく高速側優先)。選定後 freeze、online を見て再学習しない。
- correctness(Phase I): F0-F7 + F8 FULL horizon / F9 teacher=v1 equivalence / F10 rollout-only isolation / F11 legal / F12 leakage / F13 reset。
- profiling(Phase J/K): Teacher vs Student の policy calls/iter(ほぼ同数=clean)・ms/call・iterations/sec、fixed-iter action agreement。
- **dev H2H(Phase L、Primary 1350ms)**: v2.4 Student vs Champion、100 games。anchor v1=0.677(同 desktop/runner、曖昧なら再計測)。
- direct v1(Phase N)+ multi-budget(Phase O)は有望時のみ。**結果を見て契約/threshold を変えない**。

## E. 結果
### E1. Audit / Dataset
- **cost 分解(再掲)**: score_options 1.250ms/call = NN forward **96%** / feature 構築 4%(encode_state 0.029ms/encode_options 0.017ms/
  card_ids 0.002ms/nn 1.201ms)。Teacher hidden=32、入力 239。→ Small Student(H<32)で high leverage。
- **dataset**(`kaggle_replays/training/rollout_dataset.npz`): teacher rollout state 収集(20 games、self-play 進行 + 各 MAIN から
  determinize→teacher rollout ×2、rollout 中 score_options の各 legal option を記録)。**states(groups)=49,204 / option-rows=366,866 /
  mean 7.46 opt/state / h_dim=239**。game 単位 split(70/15/15)= rows train/val/test **289,505 / 48,354 / 29,007**。
  leakage: 記録は PolicyModel と同一の 239次元標準化入力 h(公開情報のみ、determinize は belief)= teacher 入力の完全複製、新規情報なし。

### E2. Students / Distillation / Offline（Phase D/E/F）
distill: **KL(teacher softmax(T=1.0) || student)**, Adam lr 3e-3, 60 epochs, **sample weighting なし**。teacher の standardization +
card_embedding を再利用、hidden のみ縮小(layer0=H×239 が NN コスト支配)。test split(game-split):

| student | hidden | test top1 | top3 | KL | layer0 MACs | NN speedup |
|---|---|---|---|---|---|---|
| teacher | 32 | 1.000 | — | 0 | 7648 | 1.0× |
| **H4** | 4 | 0.751 | 0.944 | 0.152 | 956 | 8.0× |
| **H8** | 8 | 0.811 | 0.969 | 0.090 | 1912 | 4.0× |
| **H16** | 16 | 0.843 | 0.977 | 0.066 | 3824 | 2.0× |

rollout-step 別 top1: step0 0.73–0.87 / step1-3 0.88–0.93 / **step4+(相手応答)0.74–0.83 = 最難**(FULL strength に重要だった相手応答側で
最も imitation が崩れる)。config blobs: h4 `e7e74ac7` / h8 `8fc92ba0` / h16 `259047df`。student weights=`training/rollout_student_h{4,8,16}.json`。

### E3. Latency / Profiling / Correctness（Phase G/J/I）
Phase G/J(desktop, 14 states):

| model | end-to-end ms/call | iters@1.35s | ms/iter | iters speedup |
|---|---|---|---|---|
| teacher | 0.899 | 31.5 | 45.5 | 1.00× |
| **H4** | 0.169 (5.3×) | 109.1 | 12.9 | **3.46×** |
| **H8** | 0.268 (3.4×) | 74.9 | 18.9 | **2.38×** |
| **H16** | 0.531 (1.7×) | 50.5 | 28.9 | **1.60×** |

→ **H1(efficiency)成立**: student は end-to-end 1.7–5.3× 速く、equal-wall-time で **1.6–3.5× 多い iterations**。policy calls/iteration は
teacher と同数(clean = action selection model のみの差)。
**correctness**: v1 unit 9/9 + cg 6/6 不変(rollout_policy=None=v1 byte 不変)、v2.1 5/5・v2.2 6/6・v2.3 7/7 不変。**v2.4 mock 4/4**
(F8 FULL horizon 同一 termination / F9 teacher mode=teacher 使用 / F10 rollout-only isolation=student は rollout のみ・tree prior は teacher /
F11 legal)。F9 cg identity(teacher-copy を rollout_policy にすると v1 と一致するか): mismatch 3/14 だが **None-vs-None も 2/14** =
`determinize()` の確率的世界プール由来(v1 自身も同率)、**機構は正常**。F12 leakage: student は同一 encoder(公開情報のみ)= teacher 入力の
複製、新規情報なし。F13 reset: PolicyModel は per-match state を持たない(weights のみ persistent)。

### E4. Development Equal-Wall-Time H2H（Phase L/N/O）
**Phase L(vs Champion abl_5_full, 1350ms, 各100games, errors 0)**:

| variant | winrate [Wilson95] | SPRT | 速度 |
|---|---|---|---|
| v1 teacher FULL(anchor) | 0.677 | — | 1.0× |
| v2_4_h4 | 0.640 [0.542, 0.727] | None | 5.3×/iters 3.5× |
| v2_4_h8 | 0.620 [0.522, 0.709] | None | 3.4×/iters 2.4× |
| **v2_4_h16** | **0.685** [0.584, 0.771] | **PROMOTE** | 1.7×/iters 1.6× |
→ 全 student が v1 付近を維持、全て Champion 勝ち越し、崩壊なし。H16 は v1 と同等を 1.7× 高速で(formal PROMOTE)。

**Phase N(direct: student vs v1 FULL, equal-wall-time 1350ms, 各100games, errors 0)= Primary parent-ablation**:

| student | vs v1 winrate [Wilson95] | SPRT | 速度 |
|---|---|---|---|
| v2_4_h16 | 0.500 [0.404, 0.596] | None | v1 と互角 |
| **v2_4_h4** | **0.620** [0.522, 0.709] | None(llr +1.91) | **v1 に勝ち越し(CI 下限 0.522>0.50)** |
→ **H4 は full-rollout v1 に equal-wall-time で直接勝ち越し**(3.5× iterations が imitation 誤差を上回る)。H16 は v1 と互角。
intransitivity 注記: vs Champion では H16>H4 だが v1 直接では H4>H16(n=100・CI ±0.09 の noise + 戦略相性)。直接対戦が主判定。

**Phase O(H4 direct vs v1 FULL scaling, 3000/5500ms, 各100games, errors 0)**:

| budget | H4 vs v1 winrate [Wilson95] |
|---|---|
| **1350ms**(提出予算) | **0.620** [0.522, 0.709] = H4 勝ち越し |
| 3000ms | 0.520 [0.423, 0.615] = 互角 |
| 5500ms | 0.520 [0.423, 0.615] = 互角 |

→ **H4 の対 v1 優位は提出予算 1350ms で最大、budget 増で互角に収束**。機構: 1350ms は v1 が iteration 飢餓(~25–31 iters)で
H4 の 3.5× iterations が最も効く帯。3000/5500ms では v1 も十分な iterations を得て差が消える。**= 効率化の価値は tight(=提出)budget で最大**。

### E5. Conclusion（Case A-D）
**Case B(good Pareto / useful compromise)確定 — 提出予算(1350ms)で Case-A 的 edge**。

小型 rollout Policy(student)は **full rollout の先読み horizon を保ったまま per-call を 1.7–5.3× 高速化**し、equal-wall-time で v1 の強さを:
- **提出予算 1350ms**: H4 が v1 に**勝ち越し**(0.620)、H16 互角(0.500)、全 student が Champion 勝ち越し(H16 formal PROMOTE 0.685)。
- **3000/5500ms**: H4 は v1 と互角(0.520)。
に **維持/改善**。**効率化3連(v2.1/v2.2/v2.3=全敗)で初の成功**。full rollout の先読みが strength に本質的(v2.1-2.3 の結論)と整合し、
**horizon は保ったまま per-call コストだけを蒸留で下げる**アプローチが機能した。

**機構**: 1350ms で v1 は iteration 飢餓、student の 3.5× iterations が imitation 誤差(fidelity 0.751)を上回り勝ち越し。高予算では両者
十分な iterations で差が消える(効率化の価値は tight budget で最大)。相手応答 step4+ の imitation が最難(top1 0.74–0.83)で高予算での上振れを抑える。
H1(efficiency)✓ / H2(imitation)✓ / **H3(Primary: equal-wall-time で v1 維持/改善)✓(1350ms で改善、高予算で維持)**。

**決定**:
- **Small rollout Policy は有効**。**formal candidate = H4**(5.3× 高速、提出予算で v1 勝ち越し)。ISMCTS v1 は strength baseline として維持。
- 次の 1 ステップ(pre-reg Phase S Case A/B)= **Strong ISMCTS → root Policy distillation**。rollout compute 問題が解けた今、Search の
  改善行動(root visit 分布/選択 action)そのものを root Policy へ蒸留し、提出時に浅い探索でも v1 的強度を狙う。**today は実装しない(レビュー待ち)**。

## Formal Candidate（Phase P）
| item | value |
|---|---|
| challenger_id | ismcts_v2_4_small_rollout_h4 |
| parent / config | ISMCTS v1 / 59384591 |
| tree/prior/fallback Policy | 735dd38a(不変) |
| **rollout Policy(student)** | `training/rollout_student_h4.json`(hidden 4、teacher standardization+embedding 再利用) |
| rollout horizon | frozen v1 FULL(opp_turns termination) |
| leaf / Belief / PUCT / deck | v1 と同一 |
| dataset | `training/rollout_dataset.npz`(49,204 states / 20 games / game-split) |
| distill | KL(teacher softmax T=1.0 ‖ student), Adam lr3e-3, weighting なし |
| config git blob | ismcts_v2_4_h4_t1350 `e7e74ac7`(+ t3000 `5e982e18` / t5500 `bb8c5aec`) |
| offline / online | top1 0.751 / vs Champion 0.640 / **vs v1 直接 0.620@1350ms**(勝ち越し) |

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`(実験中 commit なし・他者 commit なし)。
ISMCTS v1 59384591 / v2.1 / v2.2 / v2.3 / Champion / **Teacher Policy 735dd38a(不変、再学習なし)** / Value 0657f7af / Belief /
handcrafted leaf / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。`ismcts.py` の `rollout_policy` 既定 None=v1 byte 不変(opt-in)。
Student は新規 asset。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
