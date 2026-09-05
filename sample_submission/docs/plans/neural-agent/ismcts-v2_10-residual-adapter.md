# ISMCTS v2.10 — Frozen-Original Residual Correction Adapter(pre-registration + 結果)

作成日: 2026-07-28
parent: v2.9(Oracle 診断: Search correction を完璧適用すると +30.7pt=signal 価値大、失敗は capacity/interference)。
Primary Research Question: **Original Policy 735dd38a を完全凍結し、correction 専用の Residual Adapter を別容量として足すと、
UNCHANGED を維持したまま Oracle V correction を学習し Policy-only online strength を改善できるか**。dataset は v2.5-2.8 と同一。
experiment_start_HEAD `ccdc63c`。

## A. Frozen-Original correctness（PASS）
- **zero-adapter == Original**: True(residual 出力層 zero-init、step0 で final==Original)。
- **gradient isolation**: Original params requires_grad=False(gradient 0)。
- **Original sha 不変**: before==after `8a83b7de9aec`(policy_weights.json 未書換)。
- 構造: `final_score = orig_score(frozen) + residual_score(trained)`。export は **単一 block-diagonal MLP(hidden 32+H)**として PolicyModel JSON 化。

## B-C. Teacher / Dataset
teacher=Oracle V gate(v2.7/v2.9 exact: CHANGED & share≥0.5 & gap≥0.2)、target=Search top1。dataset=`search_root_dataset.npz`(同一)。
**V-positive train/val/test = 413/86/82、V-OFF test = 811**。

## D-E. Adapter / Objective
Adapter=小型 MLP(hidden ∈ {4,8,16})、入力 239(同一)、zero-init。
`L = mean_VOFF[KL(π_original‖π_final)] + λ·mean_Vpos[KL(π_search‖π_final)]`(subset 内正規化、V-pos に anchor 掛けない)。λ=1。
retention guard VOFF_ret≥0.97、V recovery bar≥0.40(pre-registered)。

## F-I. Offline 結果（test）
| A | V_rec | VOFF_ret | UNCHG_ret | res_VON | res_VOFF | ep*(guard) |
|---|---|---|---|---|---|---|
| 4 | 0.024 | 0.990 | 0.994 | 0.120 | 0.129 | 2 |
| 8 | 0.012 | 0.991 | 0.997 | 0.027 | 0.029 | 1 |
| 16 | 0.024 | 0.986 | 0.993 | 0.071 | 0.072 | 1 |

訓練 trajectory(val, guard 無視): V_rec を 0.15–0.19 に上げると **VOFF_ret が 0.72–0.82 に崩壊**(guard で弾かれ best は residual≈0=≈Original)。
**決定的観察: res_VON ≈ res_VOFF**(例 A4: 0.120 vs 0.129)= **adapter が V-positive を選択的に検出できていない**。
理由(根本): **V-positive か否かは Search の visit 分布で定義され、adapter の入力(盤面 h)には無い**。ゆえに adapter は「どこを直すか」を入力から
判別できず、V-positive の residual を大きくすると V-OFF にも漏れる。**pre-reg bar(VOFF≥0.97 & V_rec≥0.40)を満たす候補なし**。

## L-M. Policy-only H2H（preflight: fallback 0, sanity 済）
| candidate | vs Original winrate [Wilson95] | 判定 |
|---|---|---|
| A4 selected(guard-passing, residual≈0) | 0.5000 [0.431, 0.569] | 中立(=Original) |
| A16 final(V_rec 0.19, VOFF 0.72) | 0.5133 [0.457, 0.569] | 中立 |

## J/O. Conclusion（Case D）+ Oracle recovery
| model | V recovery | UNCHANGED retention | online |
|---|---|---|---|
| v2.7 shared H32 | 0.17–0.22 | 0.76–0.84 | ≈0.50 |
| **v2.10 Adapter** | 0.02(guard)/0.19(final) | 0.99 / 0.72 | **≈0.50** |
| Oracle V | 1.00 | 1.00 | **0.807** |

**Case D(Recovery vs retention tradeoff remains)確定**。frozen-original + 別容量 adapter でも **recovery↑ → retention↓** の generalization
interference が残り、online 中立(0.50–0.51、Oracle upper-bound +30.7pt の回収率 ≈0%)。
**根本原因(v2.10 で明確化)**: **correction の trigger(V-positive-ness)は Search 統計であり盤面入力に無い**ため、search-less な Policy/adapter は
「どこを直すか」を判別できず選択的に correction できない(res_VON≈res_VOFF が証拠)。= **Oracle の価値は Search の lookahead で「いつ・どう直すか」を
知ることに由来し、search-less Policy へは構造的に転移しづらい**。

**決定**:
- residual adapter 単独は不採用。Original 維持。**H4 統合不実施**(online 中立)。
- pre-reg Phase O Case D の次候補 = explicit gate head / mixture-of-experts。**ただし gate も入力=盤面なら同じ限界**(V-positive-ness を盤面から
  予測できない)ため楽観視しない。
- **より確度の高い推奨(私見)= v2.4 H4 deployment 最適化**。Oracle が示した「search corrections の価値」は **search を走らせること**で実現され、
  **v2.4 H4 は提出予算 1350ms で v1 に勝ち越した唯一の online 成功=deployable な search**。distill でなく H4 の詰め(tarball 検証・Gate2 判定)が
  期待値最大。**today は実装しない(レビュー待ち)**。

## Integrity
experiment_start_HEAD `ccdc63c` / experiment_end_HEAD `ccdc63c`。**Original Policy 735dd38a 完全凍結**(sha 不変・gradient 0・zero-adapter 等価)。
runner: 絶対パス preflight・fallback 0。ISMCTS v1 59384591 / v2.1-v2.9 / v2.4 H4 e7e74ac7 / Original Policy 735dd38a / H4 rollout student / Value /
Belief / handcrafted leaf / Champion / **Reference Pool v2 14db8345 不使用** / Gate2 v2 05e509a2 すべて不変。production/cg/main.py/deck.csv/weights
無変更。git add/commit/push なし。新規 research asset のみ(train_residual_adapter.py + adapter JSON)。
