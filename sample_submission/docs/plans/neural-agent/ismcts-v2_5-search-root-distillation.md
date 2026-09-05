# ISMCTS v2.5 — Search-to-Root Policy Distillation(pre-registration + 結果)

作成日: 2026-07-27
parent: v2.4 H4(small fast rollout policy、Case B 成功)。
Primary Research Question: **Strong ISMCTS が root で Original Policy を修正した判断(root visit 分布)を、同サイズの Root Policy weight へ蒸留できるか**。
- **Primary**: Search-Distilled Policy 単体が、**Search なしで** Original Policy 単体より online で強いか。
- **Secondary(Primary positive 時のみ)**: Distilled Policy を H4 ISMCTS の **root prior** に入れると 1350ms で H4 がさらに強くなるか。
ステータス: **pre-registration(online 前 freeze)**。production/cg/ISMCTS v1・v2.1-2.4/Reference Pool v2/Gate2 v2/Strong Opponent 変更なし。
experiment_start_HEAD `ccdc63c`。

## A. Teacher Search Freeze
Teacher = **ISMCTS v2.4 H4**: FULL rollout horizon、rollout policy=frozen H4(`training/rollout_student_h4.json`)、root/tree prior=Original
Policy **735dd38a**、handcrafted leaf、Belief=match_context、PUCT c_puct=1.4、world_pool=8。**dataset は wall-time でなく fixed iterations**
(再現性・visit 品質の均一化)。**N_teacher = 128 iterations/root(freeze)**。
root 統計(instrumentation `SearchStats.root_children_full`、探索挙動不変): 各 legal action の **visit N(a) / Q(a) / prior P(a)**、
selected action、total visits、Original Policy score。存在しない値は作らない。

## B. 蒸留 target(pre-registered)
- **Primary target**: `π_search(a|s) = N(a) / Σ_b N(b)`(root visit 分布)。
- hard target `argmax_a N(a)` は診断用に保存(Primary は soft)。**Q(a) は保存のみ(今回 loss 不使用)**。
- **forced(legal action 1個)は改善 signal なし → dataset から除外**(件数記録)。

## C. Pilot(H4 @128, mirror + archetype、read-only)
- 2 game mirror: usable roots 44 / forced 4、**CHANGED 27.3%**、top1 visit share 0.696 / gap 0.489 / entropy 0.721、high-conf 68%。
- 3 game(mirror+archaludon+dragapult): roots 77、**CHANGED 18.2%**、high-conf 74%、legal action median 6。
→ **128 iters で visit 分布は well-formed(Phase A2 確認)**、CHANGED 率 ~18-27% = 明確な改善 signal。

## D. Dataset source(freeze)
- **opponents**: mirror + archaludon_ex / dragapult_ex / mega_lucario_ex / marnie_grimmsnarl_ex / crustle / rocket_mewtwo_ex
  (`kaggle_replays/meta_analysis/archetype_decks/<arch>/01.csv`、開発資産。**Reference Pool v2 不使用**)。opponent は Original Policy greedy。
  alakazam(deck.csv、player 0=確認済)側の root のみ記録(mirror は両側)。「Mirror だけにしない」= coverage 対策。
- N_teacher=128、target ~4000 roots(180 games)、**game 単位 split(70/15/15)**、duplicate 監査。

## E. Student architecture(freeze)
- **Original Policy と完全同一**(input 239 = state166+option65+embed8、hidden **32**、option 表現・scoring 同一)。差分は **weights のみ**。
- **Original Policy 735dd38a で初期化**(ゼロ学習しない=人間模倣能力を保持しつつ Search 改善を上乗せ)。Root Policy 小型化は別フェーズ。

## F. Distillation objective(freeze)
- **KL(π_search || π_student)**, **τ=1.0**(visit 分布そのまま)。Adam lr 1e-3(Original init 微調整)、validation KL で epoch 決定。
- **今回入れない**: CHANGED/confidence/Q/advantage/outcome/win-loss weighting、Value loss、BC replay weighting。**pure Search visit distillation**。
- online H2H で hyperparameter を選ばない(結果後再学習禁止)。

## G. 仮説 / 成功判定(結果前 freeze)
- **H1**: Student は Search visit を模倣(overall agreement↑、CHANGED recovery↑)。
- **H2(Primary)**: Distilled Policy-only > Original Policy-only(online)。
- Case A(strong): Policy-only 改善 **かつ** Distilled-root H4 > Original-root H4 → Search knowledge transfer 成功。
- Case B: Policy-only 改善、H4 統合は neutral → 軽量 Policy deployment に価値。
- Case C: offline CHANGED recovery 良好も online 改善なし → offline-online transfer 失敗。次=CHANGED/confidence weighting。
- Case D: overall agreement 高いが CHANGED recovery 低い(=Original コピー)→ signal dilution。次=CHANGED weighting。
- Case E: Distilled が Original より弱化(catastrophic forgetting/noisy target)→ 原因分析。

## H. 評価計画(結果前 freeze)
- offline(Phase G): overall Search agreement / KL / **CHANGED recovery(最重要)** / Original-stuck / UNCHANGED retention / turn 帯別。
- **Policy-only H2H(Phase I、Primary)**: Distilled Policy-only vs Original Policy-only、**唯一差分=Policy weights**(lethal/fallback/deck 一致)。
  health 100 games(errors/illegal/timeout 0)→ 正式 H2H(δ_min=0.05/α.05/β.10)。
- root-only integration(Phase K/L、Primary positive 時のみ): H4 の **root prior だけ** Distilled に(internal tree/rollout/fallback は Original/H4 のまま)。
  isolation test(student call: root>0 / rollout=0 / internal tree=0 / fallback=0)→ 1350ms direct H2H vs H4 parent。
- **結果を見て target/contract を変えない**。Reference Pool v2 不使用。

## I. 結果
### I1. Teacher / Pilot / Dataset
- Teacher=v2.4 H4 @**128 iters**(freeze)。dataset `training/search_root_dataset.npz`: **usable roots 6,812 / forced 181 /
  option-rows 62,511**、opponents=mirror+archaludon_ex+dragapult_ex+mega_lucario_ex+marnie_grimmsnarl_ex+crustle+rocket_mewtwo_ex
  (各 ~26 games)、game-split rows tr/va/te=45,203/9,120/8,188。**CHANGED(Search top1≠Original top1)=25.4%(1,729)**、
  high-conf CHANGED=581(全体の 8.5%)、visit share mean 0.585 / gap 0.401 / entropy 1.202。

### I2. Distillation / Offline（CHANGED recovery）
student=Original 735dd38a と同一 H32、Original 初期化、**KL(π_search‖student) τ=1.0、weighting なし**、best-val-KL checkpoint(Phase F3)。

| 指標 | Original init | **Distilled** |
|---|---|---|
| overall Search agreement | 0.761 | **0.705**(低下) |
| **CHANGED recovery** | 0.000(定義上) | **0.150** |
| Original-stuck(CHANGED で Original のまま) | 1.000 | **0.723** |
| UNCHANGED retention | 1.000 | **0.879**(低下) |

turn 帯別 CHANGED recovery: early 0.138 / mid 0.279 / late 0.105。
→ **弱い transfer**: CHANGED の **15% しか回収せず 72% は Original のまま**、かつ UNCHANGED を 12% 破壊、overall は Original より **低下**。
high-conf CHANGED が全体の 8.5% しかなく、**pure visit-KL が改善 signal を UNCHANGED 75% + 低信頼 CHANGED の noise に希釈**された。

### I3. Policy-only H2H（Primary transfer、Distilled vs Original, no search, 唯一差分=weights, 300 games, errors 0）
**distilled vs original = 0.5167 [0.4603, 0.5726]、SPRT None(CI が 0.50 を含む)= 有意な転移なし(中立)**。

### I4. Root-prior Integration（positive 時のみ）
**不実施**。Phase I(Policy-only)が中立=positive transfer 不成立ゆえ、pre-reg 条件(Primary positive 時のみ H4 統合)に該当せず。

### I5. Conclusion（Case A-E）
**Case D(Signal Dilution)確定**。offline **CHANGED recovery 低(0.15)/ Original-stuck 高(0.72)**、online Policy-only **中立(0.517)**。
= **Search が Original を改善した判断(CHANGED 25%、うち high-conf は 8.5%)が、UNCHANGED 75% + 低信頼 CHANGED の noise に希釈され、
pure Search-visit KL 蒸留では weight へ十分移らなかった**。H1(imitation)は部分的(CHANGED 15% 回収)、**H2(Primary transfer)✗**。

**機構**: 蒸留 target が全 root の visit 分布(entropy 1.202、多くが spread/低信頼)ゆえ、KL は policy を「Original を少し崩して spread に寄せる」方向へ
動かし、**改善の多数を占める minority signal を拾えず overall agreement も低下**。これは v2.2 とは逆向きの乖離(offline も弱く online も中立)。

**決定**:
- 単純 Search-visit distillation は不採用。Original Policy 735dd38a を維持。**H4 統合は不実施**(条件不成立)。target/loss はこの実験内で変えない(Phase F 遵守)。
- **次の 1 ステップ(pre-reg Phase O Case D)= CHANGED / high-confidence Search-target weighting**。改善 signal(CHANGED かつ high-conf な root)を
  重み付け(または dataset を CHANGED 中心に再構成)して蒸留し、希釈を解消できるかを独立 ablation で検証。**today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`(実験中 commit なし・他者 commit なし)。
ISMCTS v1 59384591 / v2.1 / v2.2 / v2.3 / **v2.4 H4 e7e74ac7** / Original Policy 735dd38a / H4 rollout student / Value 0657f7af / Belief /
handcrafted leaf / Champion / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。`ismcts.py` は `root_children_full` instrumentation
追加(探索挙動不変、v1 9/9+6/6 再 PASS)。Distilled Policy は新規 asset。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
