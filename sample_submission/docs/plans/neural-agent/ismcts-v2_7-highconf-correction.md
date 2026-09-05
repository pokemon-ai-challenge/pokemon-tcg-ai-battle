# ISMCTS v2.7 — High-Confidence Search-Correction Distillation(pre-registration + 結果)

作成日: 2026-07-27
parent: v2.6(anchor + ALL-CHANGED correction、CHANGED recovery 改善せず Case D)。
Primary Research Question: **Search が Original を修正し、かつ十分な confidence を持つ root だけを correction 対象にすると、有用な改善 signal を
同サイズ Policy weight へ転移できるか**。唯一差分 = correction 対象 **ALL CHANGED → HIGH_CONF CHANGED のみ**、かつ correction loss を
**HIGH_CONF subset 内で正規化**(v2.6 の mean-over-all 再希釈を禁止、Phase C)。dataset/split/H32/Original 初期化/Teacher は v2.5/v2.6 と同一。
experiment_start_HEAD `ccdc63c`。

## A. Confidence 定義(freeze)
**HIGH_CONF = CHANGED かつ top1 visit share ≥ 0.5 かつ top1-top2 gap ≥ 0.2**(pilot/v2.6 と同一、全 root の 8.5%)。**Q は今回不使用**(保存のみ)。

## B. HIGH_CONF audit(split 別)
| split | roots | CHANGED | HIGH_CONF_CHANGED |
|---|---|---|---|
| train | 4,948 | 1,274 | 413 |
| val | 971 | 242 | 86 |
| test | 893 | 213 | 82 |

## C. Loss(pre-registered、唯一差分)
`L = mean_over_ALL[KL(π_original‖student)] + λ_hc · mean_over_HIGH_CONF[KL(π_search‖student)]`。**correction は HIGH_CONF subset 内平均**
(mean-over-all の再希釈を禁止)。λ_hc ∈ {1(primary), 2(diagnostic)}。checkpoint: UNCHANGED retention≥0.95 の下で HIGH_CONF recovery 最大。

## D. 結果（Phase C-G, test / 訓練 trajectory）
**HIGH_CONF recovery は大幅改善したが UNCHANGED retention を破壊**（訓練 trajectory、val）:

| λ_hc | HIGH_CONF recovery | UNCHANGED retention |
|---|---|---|
| 1 (ep10–40) | 0.15–0.20 | **0.81–0.84** |
| 2 (ep10–40) | 0.19–0.22 | **0.76–0.80** |
| v2.6 λ2(参考) | 0.073 | 0.946 |

- **HIGH_CONF recovery は v2.6 の 0.073 → 0.17–0.22(~2.5–3×)に改善** = HC 正規化 correction は「Search が明確に修正した root」を実際に学べる。
- **しかし UNCHANGED retention が 0.95 guard を毎 epoch 割る(0.76–0.84)** = shared H32 で HC を学ぶと UNCHANGED に interference。
- retention guard(0.95、freeze)が全 epoch を弾き **best-checkpoint = Original init(ep0)に退避**。pre-reg 候補(retention≥0.95)は事実上 Original。

## E. Policy-only H2H（Phase J、no search、abs-path preflight + hash 検証、vs Original）
preflight: candidate sha1 ≠ control(policy_weights.json 8a83b7de)、is_ready 検証、index-0 fallback 0。sanity(Original vs Original)=0.53≈0.50。

| candidate | vs Original winrate [Wilson95] | 判定 |
|---|---|---|
| best-checkpoint(retention≥0.95 = Original 退避) | 0.505 [0.436, 0.574] | Original そのもの(中立) |
| final-epoch λ=1(retention 0.81, HC_rec 0.19) | pooled 155/309 = **0.502**(2 run: 0.422/0.545, 高分散) | **中立** |

→ **guard 版(=Original)も retention 破壊版も Original を超えない**。

## F. Conclusion（Case A-E）
**Case C(Retention tradeoff / capacity interference)確定**。**HIGH_CONF recovery は明確改善(0.073→0.17–0.22)= 改善 signal 自体は学習可能**、
だが **同サイズ H32 では UNCHANGED retention を破壊せずに学べない(0.95→0.76–0.84)**= shared weight の capacity/interference。retention を壊して
学んだ final モデルも **online は中立(0.502)**= その修正は online strength に転移しない(UNCHANGED 破壊の害と相殺)。H2(Primary transfer)✗。
online 中立ゆえ **Phase L(H4 統合)は不実施**。

**search-to-root 蒸留 3 連(v2.5 dilution / v2.6 anchor / v2.7 HC-focus)は全て online neutral** = **Search の root 修正(128-iter H4)を同サイズ Policy へ
蒸留しても online 強度に転移しない**、と確定。

**決定**:
- 同サイズ HC-correction は不採用。Original Policy 735dd38a を維持。**H4 統合不実施**。
- **次の 1 ステップ(pre-reg Phase J Case C)= Correction 専用 adapter / capacity 分離**。Original を凍結し、correction を **別容量(residual adapter 等)**で
  持たせて interference を回避できるか。ただし search-to-root 3 連の online neutral を踏まえ、**代替として「Search 修正の価値を Q(a)/advantage で
  重み付け(真に strength を上げた修正だけを転移)」も有力**(dataset に Q 保存済)。**today は実装しない(レビュー待ち)**。

## Integrity / Runner Safety
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。**v2.6 path bug 再発防止**: `policy_only_h2h.py` に絶対パス化 + preflight
(is_ready 検証 + candidate/control sha1 表示、index-0 silent fallback 禁止)を実装、fallback 0 を確認。
ISMCTS v1 59384591 / v2.1–v2.6 / v2.4 H4 e7e74ac7 / Original Policy 735dd38a / H4 rollout student / Value 0657f7af / Belief / handcrafted leaf /
Champion / Reference Pool v2 14db8345 / Gate2 v2 05e509a2 すべて不変。Q 不使用。production/cg/main.py/deck.csv/weights 無変更。git add/commit/push なし。
