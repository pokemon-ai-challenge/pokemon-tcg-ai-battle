# ISMCTS v2.6 — Anchored Search-Correction Distillation(pre-registration + 結果)

作成日: 2026-07-27
parent: v2.5(Search-to-Root distillation、Case D=signal dilution: 全 state を Search visit へ KL → UNCHANGED 破壊 + CHANGED 回収 15%)。
Primary Research Question: **Original Policy を明示的にアンカーしつつ、Search が Original を修正した CHANGED state だけに correction loss を追加すると、
改善 signal を同サイズ Policy weight へ転移できるか**。dataset/architecture(H32)/Original 初期化/Teacher/encoder/root target は v2.5 と同一、
**変更は training objective / weighting のみ**。experiment_start_HEAD `ccdc63c`。

## A. Dataset Audit(v2.5 dataset 再監査、read-only)
`training/search_root_dataset.npz`(v2.5 と同一): roots 6,812 / **CHANGED 25.4%(1,729)** / UNCHANGED 5,083 / **high-conf CHANGED 8.5%**。
visit entropy 1.202 / top1 share 0.585 / gap 0.401。CHANGED の visit share 0.457 / gap 0.232(= Search の「修正」は概して低〜中信頼)。

## B. Objective(pre-registered、唯一差分)
- Student = Original 735dd38a と同一 H32、Original 初期化。
- **L = L_anchor + λ·I_CHANGED·L_search**:
  - `L_anchor = KL(π_original ‖ π_student)`(**全 state**、UNCHANGED destruction 防止)。
  - `L_search = KL(π_search ‖ π_student)`(**CHANGED state のみ**)。
- **λ_changed ∈ {1, 2, 4}**(freeze)。Adam lr 1e-3、same splits/init/epochs。dataset oversample しない(loss weighting のみ)。
- candidate 選定基準(freeze): **UNCHANGED retention ≥ 0.95(guard)AND CHANGED recovery > 0.150(v2.5)**、満たす中で CHANGED recovery 最大。

## C. λ sweep 結果（Phase C-G、test split）
| λ | overall | CHANGED_rec | orig_stuck | UNCHANGED_ret | hi_rec | med_rec | lo_rec |
|---|---|---|---|---|---|---|---|
| 1 | 0.749 | 0.085 | 0.840 | 0.957 | 0.049 | 0.086 | 0.140 |
| 2 | 0.744 | 0.099 | 0.808 | 0.946 | 0.073 | 0.086 | 0.160 |
| 4 | 0.761 | 0.000(Original へ退避) | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| **v2.5(参考)** | 0.705 | **0.150** | 0.723 | 0.879 | — | — | — |

- **anchor が Original を強く保持 → CHANGED recovery は v2.5(0.150)より低下(0.085–0.099)**。λ=4 は correction を強めると UNCHANGED を
  破壊し retention guard を毎 epoch 割る → Original に退避(CHANGED_rec 0)。
- confidence 別 recovery は **low > high**(λ=2: low 0.160 / high 0.073)= **高信頼の改善を拾えず低信頼を反転**(逆効果)。
- **pre-reg bar(retention≥0.95 & CHANGED_rec>0.150)を満たす候補なし** = offline で Case D。online 確認用に λ=2(最大 correction)を選定。

## D. Policy-only H2H（Phase K、Primary、no search、唯一差分=weights、vs Original）
**測定バグの訂正**: 初回 run は `--weights` の**相対パスが spawn worker(cwd=sample_submission)で解決できず**、候補 policy がロード失敗
→ PolicyModel の index-0 fallback で play → 偽の regression(0.16–0.18)。**絶対パス化で修正**。sanity(Original vs Original)= **0.530 [0.433,0.625]
≈ 0.50** で harness 健全性を確認。

| candidate | vs Original winrate [Wilson95] | SPRT | 判定 |
|---|---|---|---|
| Original vs Original(sanity) | 0.530 [0.433, 0.625] | — | null≈0.50 |
| v2.5 distilled(参考) | 0.5167 [0.460, 0.573] | None | 中立 |
| **v2.6 anchored λ=1** | 0.530 [0.474, 0.586] | None | **中立** |
| **v2.6 anchored λ=2** | 0.540 [0.484, 0.596] | None | **中立** |

→ **v2.6 anchored は online 中立**(CI が 0.50 を含む、v2.5 と同水準)。有意な transfer なし。

## E. Conclusion（Case A-E）
**Case D(Signal still weak)確定**。anchor で UNCHANGED retention は改善(0.879→0.95)したが、**CHANGED recovery は v2.5 より低下(0.150→
0.085–0.099)**、online は中立(0.53–0.54、有意でない)。= **anchor loss(全 state KL to Original)が CHANGED correction(25%・低信頼・noisy
128-iter visit)を圧倒し、改善 signal を weight へ十分移せない**。λ を上げると shared H32 weight の generalization で UNCHANGED も壊れ retention
guard を割る。**simple CHANGED weighting では不足**。H1(imitation)部分的 / **H2(Primary transfer)✗**。

**決定**:
- anchored correction 単独は不採用。Original Policy 735dd38a を維持。**online 中立ゆえ Phase M(H4 root-prior 統合)は不実施**(Case A 条件不成立)。
- **次の 1 ステップ(pre-reg Phase K Case D)= High-Confidence CHANGED targeting**(v2.7)。改善 signal の中でも **high-confidence CHANGED**
  (top1 visit share/gap 大)だけに correction を絞る。dataset に `top1share/gap/entropy` と `Q(a)` 保存済み(Q は confidence 後の独立 ablation 素材)。
  **today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`(実験中 commit なし・他者 commit なし)。
ISMCTS v1 59384591 / v2.1 / v2.2 / v2.3 / v2.4 H4 e7e74ac7 / v2.5 historical / Original Policy 735dd38a / H4 rollout student / Value 0657f7af /
Belief / handcrafted leaf / Champion / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。訓練は v2.5 dataset(不変)を loss/weighting のみ変更。
`policy_only_h2h.py` に絶対パス化 fix(研究 harness)。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
