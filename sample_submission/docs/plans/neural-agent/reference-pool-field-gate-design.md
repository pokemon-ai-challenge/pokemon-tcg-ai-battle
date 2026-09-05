# 設計書: Reference Pool v1 + Field Evaluation Gate

作成日: 2026-07-26
種別: **設計書**(方向性・選定根拠。関数シグネチャ/完了条件は別 implementation-plan。[[feedback_design_vs_implementation_plan]])
前提: 測定基盤フェーズ完了([[project_measurement_protocol]]: SPRT 3分類/Wilson CI/勝ち筋分類/resume/再現メタ/
process 並列 desktop ~5x/較正済)。実案件 Track A で `abl_5_full` を Current Champion 候補として確認
(PROMOTE @N=217, winrate 0.627, CI[0.561,0.688])。Opponent 資産棚卸し済([[project_worthy_opponent_inventory]])。
ステータス: **設計のみ。実装未着手。production/cg/main.py/deck.csv/weights/configs/shared league 変更なし。read-only 調査に基づく。**

---

## 0. Background / Goal

Track A で `abl_5_full` は同一デッキ head-to-head で旧 production に決定的勝ち越し。だが**これはミラー
自己対戦であり、field への外的妥当性は未確認**。次に解くのは「Current Champion との直接対決で勝つ」でなく
**「環境全体で本当に強くなったか」を何をもって判定するか**。本書は Reference Pool v1 / Field Evaluation Gate /
Champion-Challenger 接続を正式設計する(実装は別フェーズ、レビュー後)。

---

## 1. 検証済み Opponent 資産(2026-07-26 read-only 実データ照合)

前回報告のコピーではなく repo 実データで再確認済(推測なし)。

- **強い専用 AI**: `dragapult_rule`(`opponents/dragapult_rule_agent.py` 33KB ＋ `opponents/dragapult_ex_deck.csv` 60枚)。
  唯一の強い専用相手(kiyotah 原典、模倣を大きく上回る [[project_dragapult_imitation_vs_rulebased]])。deck が固有(自前 belief)。
- **難デッキ matchup**(相手 AI は弱い ml_policy、識別力はデッキ相性由来): `_matrix.json` の Champion(alakazam)行 —
  **crustle 0.387 / rocket_mewtwo 0.443**(負け越し)、marnie 0.583(競る)、shirona .69/dragapult .79/lucario .80/archaludon .82(大勝)。
- **archetype pool**: `kaggle_replays/meta_analysis/archetype_decks/` **21アーキ**(大半 5 recipe。gekkouga 1・oliva/toxtricity 4)。
- **field share**(`meta_report.json`, field_pct): alakazam 31.9 / **mega_lucario 17.4 / archaludon 15.3** / crustle 7.1 /
  dragapult 6.3 / marnie 5.5 / mega_starmie 5.2 / rocket_mewtwo 1.8 / … 上位7で ~80%、以降は <1.5%(ほぼ不在)。
- **戦略軸(実カード `card_adoption` 根拠)**: crustle=control/壁(Mist Energy 保護) / rocket_mewtwo=特殊エネ(Team Rocket's Energy) /
  marnie=進化 setup+妨害(Marnie trainer-pokemon, Spikemuth Gym) / dragapult=spread テンポ(Stage2, Boss's Orders) /
  mega_lucario=闘アグロ / archaludon=鋼ビート高HP / mega_starmie=水高速。
- **弱すぎる相手**: 汎用 `rule_based`(ml_policy が各アーキ 0.75-0.94 勝ち、crustle 0.48 のみ競る)= 識別力不足。
- **calibration 専用**: `first_choice`(通常 field opponent には含めない)。
- **固定可能性**: config/weights/deck は全て hash 固定可(Track A の A-0 で実証済)。agent は id/version で固定。

**含意(重要)**: 強い専用 AI は `dragapult_rule` 1体のみ。他の難マッチは**弱 AI × 難デッキ**。→ worthy-opponent
ギャップは実在。Field eval は当面「デッキ相性の robustness」を測るもので「強 AI 相手の強さ」ではない(§Risks)。

---

## 2. Reference Pool v1(推奨 6 メンバー)

21アーキ全部を最初の正式 Gate にしない。**識別力 × 戦略多様性 × 実在資産**で 6 に絞る。

| メンバー | 層 | Champion勝率 | field% | 戦略軸 | 相手AI | 採用理由 |
|---|---|---|---|---|---|---|
| **crustle** | Core Worthy | 0.387 | 7.1 | control/壁 | 弱ml_policy(デッキ相性) | Champion 最大の負け越し=識別力最大 |
| **rocket_mewtwo_ex** | Core Worthy | 0.443 | 1.8 | 特殊エネ | 弱ml_policy | 2番目の負け越し・独自エネ系 |
| **marnie_grimmsnarl_ex** | Core Worthy | 0.583 | 5.5 | 進化+妨害 | 弱ml_policy | 競る帯・妨害軸 |
| **dragapult_rule** | Core Worthy | (未計測) | (6.3) | spread | **強rule-based** | 唯一の強AI相手 |
| **mega_lucario_ex** | Meta Coverage | 0.797 | 17.4 | 闘アグロ | 弱ml_policy | 最大 field share の非alakazam=非退行確認 |
| **archaludon_ex** | Meta Coverage | 0.823 | 15.3 | 鋼ビート | 弱ml_policy | 2番目 share=非退行確認 |

戦略軸 6 種(control/特殊エネ/妨害/spread/闘/鋼)で重複なし。alakazam は Champion 自身なので pool から除外
(ミラーは Gate 1 で既に扱う)。**除外**: 汎用 rule_based(弱・識別力不足)、first_choice(calibration 専用)、
field <1.5% の裾(ogerpon/kamitsuorochi/yadoking 等、ほぼ不在で robustness 上の価値低)。

### 2層構造(推奨: 分ける)
- **Core Worthy Pool**(crustle/rocket_mewtwo/marnie/dragapult_rule): Candidate の差を露出する層。Champion が
  苦戦 or 強 AI。**Δ_field の主対象**。
- **Meta Coverage Pool**(mega_lucario/archaludon): 強さでなく「主要メタで壊れていない」確認層。high-share・
  Champion が現状勝つ。**regression guard の主対象**。
- 分離理由: 目的が違う(差の測定 vs 非退行)。単一 pool に統合すると、meta share で lucario/archaludon が
  支配し Core の難マッチが埋没する(下記 §4 の数値がその証拠)。**v1 は分けて集計、最終判定で統合**。

---

## 3. Gate 1 — Champion head-to-head screening(既存 SPRT)

```
Challenger  vs  Current Champion(abl_5_full)
```
- 既存 measurement SPRT(3分類)を使用。契約: α=0.05, β=0.10, **small change δ_min=0.03 / large algorithmic change δ_min=0.05**。
- 目的: **明確に弱い Challenger を Field 評価前に安く落とす**(FUTILITY で早期棄却)。
- **Gate 1 通過だけでは Champion 昇格しない**(ミラー優位 ≠ field 転移)。Track A で `abl_5_full` は Gate 1 相当を
  通過済(PROMOTE)だが、それは v0only 比の話で、昇格には Gate 2 が要る。
- 実装: 既存 `driver.run_sprt_ab` そのまま(harness 無改変)。

---

## 4. Gate 2 — Field Validation(stratified、SPRT を混ぜない)

### 4.1 なぜ SPRT を混ぜないか(Step 5)
複数 Opponent を混ぜた勝敗系列は**単一固定 Opponent との iid Bernoulli ではない**(相手ごとに勝率が違う混合分布)。
これを現行 head-to-head SPRT にそのまま投入しない。**Gate 2 は stratified field evaluation** として設計。
既存 SPRT は Gate 1 専用でよい。

### 4.2 構造
Gate 1 通過 Candidate のみ、Reference Pool の**各 opponent を独立 stratum**として固定 N 対戦。各 stratum で保存:
`games / wins / losses / winrate / Wilson CI / errors / prize_out / no_pokemon / deckout / terminal_flags / recipe_id / agent_id`。
Candidate と **Current Champion の両方**を同一 Reference Pool に対して走らせる(§5)。

### 4.3 Field aggregate metric(Step 6・実データで比較)
Champion の対 pool 勝率を 2 weighting で実測(`_matrix.json` × field share):

| weighting | Champion 対pool勝率 | 性質 |
|---|---|---|
| A. Equal-archetype | **0.607** | 難マッチが平等に効く=robustness、人気デッキ非支配 |
| B. Meta-frequency(share_weight) | **0.705** | Kaggle 期待勝率に近いが、lucario17%+archaludon15% が支配し **crustle0.39/rocket0.44(低share)が埋没** |

**推奨 = C. Hybrid(Meta-weighted primary + per-archetype robustness guard)**:
- **Primary metric = meta-weighted field winrate**(外的妥当性=leaderboard が報いるもの)。
- **Guard = per-archetype catastrophic regression の不在**(meta-weight が埋没させる低share難マッチを別途保証)。
- **比率は決めない**: Hybrid は「加重ブレンド」ではなく **2 段(Primary メトリク＋Guard 合否)**。恣意的比率を持たない。
- 根拠: 上表の 0.607 vs 0.705 差が「meta-weight だけだと難マッチの劣化を隠す」ことの実証。だから guard が必須。

### 4.4 Candidate-vs-Champion field delta(Step 7)
Candidate の field 勝率単独でなく、**同一 Reference Pool** に対する
```
Δ_field = (Candidate の field performance) − (Champion の field performance)
```
を主判断にする。これで「Opponent 自体が弱い/強い」の影響を分離(弱い相手なら両者とも勝つので Δ に出にくい)。
Primary = meta-weighted Δ_field、各 stratum の Δ を guard で見る。

---

## 5. Catastrophic Regression Guard(Step 8・閾値は実データ根拠)

平均だけで昇格させない。「overall +5pt but crustle −15pt」を防ぐ。対象 = **Core Worthy + high-share Meta Coverage** の各 stratum。

**必要 N と検出可能 regression(Wilson 半幅@p=.5、Δ の SE=√(0.5/N)、実測)**:

| N/arch | Wilson半幅 | SE(Δ) | 2SE で検出可能な regression |
|---|---|---|---|
| 100 | ±9.8pt | 7.1pt | −14pt 級 |
| 200 | ±6.9pt | 5.0pt | **−10pt 級** |
| 300 | ±5.7pt | 4.1pt | −8pt 級 |
| 400 | ±4.9pt | 3.5pt | −7pt 級 |

**推奨(暫定・レビュー対象)**: N=200-300/arch で運用し、guard は「per-archetype Δ_field ≤ **−10pt** かつ CI 分離
(Candidate と Champion の Wilson CI が重ならない)」を catastrophic とみなす。−10pt は N=200 の 2SE、N=300 なら
より確実。**恣意的閾値でなく「その N で統計的に検出できる最小級」に合わせている**。閾値・N はレビューで確定。

---

## 6. Recipe allocation(Step 9・推奨: Fixed balanced + frozen manifest)

各アーキ ~5 recipe。1 recipe 固定への過学習を防ぐ。
- 候補: (a) Fixed balanced(各 recipe に N/5)、(b) Share-weighted recipe(per-recipe 出現率が要る)、(c) Random+frozen manifest。
- **推奨 = (a)+(c): 各 recipe 均等配分 ＋ 評価開始前に recipe sequence を manifest 固定**(再現性)。
  理由: per-recipe の field 出現率は現データ(`meta_report` は archetype 粒度)から**クリーンに取れない**ため share-weighted は保留。
  均等配分は単一 recipe 過学習を避け robustness を担保。frozen manifest で Candidate/Champion が同一 recipe 系列を見る。

---

## 7. Game budget(Step 10・desktop throughput 実測から)

固定 N を先に決めず、**検出したい Δ × matchup variance × throughput** から算出。Track A 実測 **34.7 games/min(workers=15, desktop)**:

| 構成 | 総試合(Cand+Champ 各) | desktop 時間 |
|---|---|---|
| 5arch × 200 | 2,000 | ~58 min |
| **6arch × 250** | **3,000** | **~87 min** |
| 6arch × 300 | 3,600 | ~104 min |
| 7arch × 300 | 4,200 | ~121 min |

**推奨**: v1 = 6arch × **250 games** × 2(Cand+Champ) ≈ 3,000 試合 ≈ **~90 分**。250 は N=200(−10pt検出)と
N=300(−8pt)の中間で、guard の実用感度を確保。borderline stratum のみ後から 300+ に延長(resume 活用)。
(注: field 対戦は Candidate=abl_5_full が重い側、opponent は弱 ml_policy/dragapult_rule で軽め。Track A と同程度〜やや速い想定。実測で校正。)

---

## 8. Reproducibility

各 Field eval run で最低限固定・保存: candidate/champion の **config hash・weights hash・deck hash・agent id**、
各 opponent の **agent id・deck(recipe) hash**、**recipe manifest(frozen sequence)**、workers、source ref、timestamp、
per-stratum outcomes.jsonl(truth source, resume 可)。measurement の再現メタ機構をそのまま踏襲。

---

## 9. Champion / Challenger 接続(Step 11・metadata のみ、manager 未実装)

Gate 2 合格 Challenger のみ: `Current Champion → Past Champion Pool`、`Challenger → New Current Champion`。
**今回は Champion manager 実装しない。metadata schema のみ設計**:
```
champion_id / timestamp / source_ref / config_hash / weights_hash / deck_hash /
gate1_report(path+decision+N) / gate2_report(path+Δ_field+guard) / predecessor(champion_id)
```

---

## 10. Self-Play League への接続(Step 12)

Reference Pool v1 は将来 `Current Champion + Past Champions + Fixed Meta Opponents + Dedicated Strong Agents +
Different Archetypes + Different Recipes` の Self-Play League へ育てる。**v1 が Champion ミラーだけに閉じない構造**に
する(opponent を registry/manifest で差し替え可能に、past champion を後で pool に足せる形)。self-play 実装は今回やらない。

---

## 11. Risks / Open decisions

**Risks**:
- **worthy-opponent ギャップ**: 強 AI は dragapult_rule 1体のみ。他 stratum は弱 AI×難デッキ=「デッキ相性 robustness」
  の測定で「強 AI 耐性」ではない。field eval の結論はこの限定付き。
- **deck.csv belief 制約(重大)**: ml_policy は内部で `read_deck_csv()`(cwd deck.csv)を「自分のデッキ」として belief/
  hidden-state に使う([[project_measurement_protocol]] 設計 §6 で既知)。field opponent を **ml_policy で archetype deck を
  持たせると、belief が deck.csv(alakazam)のまま**になり不正確。dragapult_rule は自前デッキを正しく読む(README)。
  → **field opponent を正しく走らせる方法が Open**(下記)。
- 弱い相手だと Candidate/Champion 双方が勝ち Δ が出にくい(Meta Coverage 層の識別力低)。
- meta shift でメンバー代表性が変わりうる(v1 は frozen、定期見直し前提)。

**Open decisions(実装前にユーザー確定)**:
1. Reference Pool v1 正式メンバー(上記 6 で良いか)。→ **確定(6 member、§13 の contract で frozen)**。
2. Core Worthy / Meta Coverage を分けるか(推奨: 分ける)。→ **確定: 分ける(集計は別、Guard は v1 一律。§13)**。
3. Field weighting(推奨: Hybrid=meta-weighted primary + guard)。→ **確定(§13 Primary=meta-weighted Δ_field + Guard)**。
4. Recipe allocation(推奨: fixed balanced + frozen manifest)。→ **確定(§13 budget)**。
5. Gate 2 採用条件(Primary Δ_field 閾値 + guard 条件)。→ **【解決】§13 で正式固定(power 分析 + MC 較正根拠)**。
6. Regression guard 閾値(推奨: −10pt & CI 分離 @N=200-300)。→ **【解決】−10pt & Holm 片側95%CI @N=250(§13)**。
7. 試合予算(推奨: 6arch×250×2 ≈ ~90分)。→ **確定: N=250(§13、power 根拠)**。
8. Champion 昇格条件(Gate1 PASS ∧ Gate2 Primary>閾値 ∧ guard 非発火)。→ **【解決】PASS/FAIL/REVIEW を §13 で正式定義**。
9. **field opponent の deck belief 制約の解法**。→ **【解決済(前フェーズ)**: (a) policy-only + dragapult_rule に限定(own-deck safe)+
   per-game `match_context.reset()`。own-deck-injection-investigation.md 参照**。
10. 今回やらないこと(field runner/pool/champion manager/self-play/新 agent/新 deck の実装)。

---

## 13. Gate 2 Decision Contract v1(2026-07-26 事前固定)

**正式 3000-game 評価の前に**判定契約を pre-registration(結果に合わせた後付けチューニング禁止)。数値は
`gate2_calibration.py`(power 分析 + Monte Carlo 20000 reps)の根拠に基づく。frozen 版 = `measurement/gate2_contract_v1.json`
(hash 固定、manifest hash 96868ca9ee34ea4f 紐付け)。`_matrix.json` は真値でなく plausible baseline(variance/power 入力)
としてのみ使用(Step 17)。A/A smoke の delta は設計に不使用(Step 16)。

### 13.1 Primary(Field 全体の改善)
- estimand = **meta-weighted Δ_field** = Σ_i w_i(pC_i − pH_i)。w_i = field share を pool 内で正規化(Σw=1)。
  weights: lucario 0.327 / archaludon 0.286 / crustle 0.133 / dragapult 0.119 / marnie 0.103 / rocket 0.033。
  **Meta Coverage が Σw 0.613 を占め、effective #arch=4.31**(Primary の検出力はこの2アーキに集中 → Core は Guard で保護)。
- CI = **analytic stratified Wald-normal**(Option A): Var=Σw_i²[pC(1−pC)/N + pH(1−pH)/N](Cand/Champ 別 game=独立)。
  parametric bootstrap で cross-validate(N=250 で正規近似は良好、MC で false-STRONG=α=5.0% を追認)。
- **practical threshold τ = +2.0pt**、one-sided α=0.05。
- 3値: **STRONG**(L>0 かつ point≥+2pt)/ **REGRESSION**(U<0 かつ point≤−2pt、対称)/ **INCONCLUSIVE**(他)。
- 検出力(N=250, plausible / p50): +5pt→STRONG 90% / 75%、+3pt→52% / 41%、+2pt→31% / 24%。
  → N=250 が確実に PASS 化できるのは **+5pt 級の field 改善**。小改善(+2/+3pt)は主に REVIEW(過小検出は既知・許容)。

### 13.2 Guard(重要 matchup の非退行)
- metric = per-archetype Δ_i = pC_i − pH_i、CI = Wald diff-of-proportions。
- **G = −10pt**、fire = (point Δ_i ≤ −10pt) かつ (**Holm** 補正片側95% CI 上端 < 0)= 壊滅量 かつ 統計的根拠(Guard A)。
- 多重比較 = **Holm**(6 family、FWER 制御)。MC FWER: none 7.7% / Bonferroni 5.2% / **Holm 5.2%**(Holm は Bonferroni を一様に上回る)。
- 検出力(N=250): −15pt→85% / −20pt→99% / −10pt→47% / −8pt→31%。catastrophic は確実、中程度は保守的安全側。
- **G=−10pt は N=250 の統計解像度(2.4·SE≈10.7pt)に整合**(design §5 を MC 追認)。層は v1 一律(単純優先)。

### 13.3 PASS / FAIL / REVIEW
- **PASS** = Gate1=PROMOTE ∧ Primary=STRONG ∧ Guard 非発火 ∧ integrity 正常(error_rate≤1%)。
- **FAIL** = Guard 発火 ∨ Primary=REGRESSION ∨ Gate1=FUTILITY(= Field で明確に悪い/matchup 破壊)。
- **REVIEW** = 他すべて(INCONCLUSIVE、Gate1=TRUNCATED、guard borderline、integrity 異常、N 使い切り未確定)。
  **N 消化で無理に二値化しない**。『昇格しない』≠『弱い』(劣ると断定は REGRESSION/Guard のみ)。

### 13.4 契約の operating characteristics(N=250, plausible, MC)
| true Δ_field | PASS | FAIL | REVIEW |
|---|---|---|---|
| 0pt | 5.0% | 7.7% | 87.2% |
| +2pt | 30.3% | 1.2% | 68.6% |
| +5pt | 89.7% | 0.1% | 10.2% |
| hidden −15pt(平均は+1.5pt) | 6.1% | **84.0%** | 10.0% |
| noisy 混在 | 21.8% | 21.9% | 56.3% |

false-PASS=5.0%(=α)。null での false-FAIL 7.7% は Primary+Guard 安全2重の和(Gate2 到達 Candidate は Gate1 通過済=真の null は稀。+2pt では FAIL 1.2%)。**hidden catastrophic を 84% で FAIL 化 = Guard の中核機能を実証**。

### 13.5 budget / 予算
6 arch × **250** × {Cand, Champ} ≈ **3000 試合**。N=250 は Guard 解像度に整合し +5pt Primary を ~90% 化。
N=400 は +3pt PASS を 52→70% に上げるが計算 60% 増で限界効用小。borderline は resume で 300+ に延長。

---

## 14. Current Champion Field Baseline v1(2026-07-26 実測)

Contract v1 を Field 実装へ反映し、**Current Champion(abl_5_full)の full-scale A/A baseline** を実行した
(desktop workers=15、flat single-pool)。**強さ比較でなく** null calibration + field snapshot + throughput 測定。
run=`measurement/results/champion_field_baseline_v1/`(未追跡)、report hash 8c01d378。

### 14.1 run / integrity
- 3000 試合(candidate 1500 + champion 1500)、**errors 0 / error_rate 0.0**、elapsed 2927s、**throughput 61.5 games/min**。
- integrity: records 3000/3000・dup 0・missing 0・**aggregation reproducible OK**(desktop 実行時 + laptop で raw JSONL 再集計 = 二重確認 PASS)。contract/manifest hash 一致。

### 14.2 A/A null calibration(Primary が正しく較正されている実証)
- **Δ_field = −0.0020(−0.20pt)、SE = 0.0153(1.53pt)、CI[−0.0271, 0.0232]、state = INCONCLUSIVE**。Guard fired = []。DECISION = REVIEW。
- 同一 AI 両群 → Δ≈0・INCONCLUSIVE・guard 非発火・REVIEW = **教科書的 null**。
- **実測 SE 1.53pt は contract の analytic 予測(§13: plausible 1.80pt / p50 2.15pt)とよく一致**(実際の Champion 勝率が
  0.24–0.94 と極端で分散が予測より小 → SE も小)。**analytic stratified CI 式が実データで妥当**であることを追認。

### 14.3 Current Champion field snapshot(champion side、N=250/arch)
| archetype | winrate | Wilson95 CI | 主な決着 |
|---|---|---|---|
| archaludon_ex | **0.940** | [0.903, 0.963] | no_pokemon 中心(圧勝) |
| marnie_grimmsnarl_ex | 0.840 | [0.789, 0.880] | prize_out 中心 |
| mega_lucario_ex | 0.820 | [0.768, 0.863] | prize_out 中心 |
| crustle | 0.664 | [0.603, 0.720] | deckout/prize_out 拮抗 |
| rocket_mewtwo_ex | **0.324** | [0.269, 0.384] | **deckout 敗(41–43/50)** |
| dragapult_ex | **0.240** | [0.191, 0.297] | prize_out 敗(強 rule に競り負け) |

**含意(重要)**:
- **負け越しは dragapult(0.24)と rocket_mewtwo(0.32)の2つ**。dragapult は唯一の強 AI(dragapult_rule)に prize_out で
  レース負け。rocket_mewtwo は **deckout 敗が支配的**([[project_deckout_loss_cause]] の構造コストが特定 matchup で顕在化)。
- **crustle は 0.664 で問題なし**。`_matrix.json` の 0.387 とは大きく異なる = **Step 17 の警告どおり `_matrix.json` は現 baseline でない**
  (別 agent/config の値)。**正式 baseline は本 run が source of truth**。
- meta-weight(lucario+archaludon が Σw 0.61)では強い2アーキが支配し、負けている rocket/dragapult は低 share。
  → Challenger 改善は **Guard が守る dragapult/rocket を壊さずに**、meta-weight 上位で伸ばせるかが焦点。

### 14.4 throughput 実測 → N 別見積り
61.5 games/min(workers=15, flat pool)。N=250 ≈ 49min / N=400 ≈ 78min / N=500 ≈ 98min(Cand+Champ 各 N×6arch×2)。
per-stratum 旧経路は 4.8 g/min(小 stratum で model 再ロード)→ flat 化で ~13x。**この baseline は将来 Challenger 評価の
比較用**だが、正式 Gate2 は原則 Candidate/Champion を同一 run で評価(古い baseline を固定 control に流用しない、§Step D7)。

---

## 12. 推奨実装順(参考・実装は別 plan、レビュー後)
1. Field runner(Candidate/Champion vs 単一 opponent stratum、deck0≠deck1、勝ち筋/CI/outcomes.jsonl)= measurement 拡張(局所・未追跡)。
2. Reference Pool manifest(opponent×recipe×agent id + hash を frozen JSON 化)。
3. Gate 2 集計(per-stratum stats → meta-weighted Δ_field + guard 判定)。
4. Gate1+Gate2 の Champion 判定レポート(metadata schema)。
5. deck belief 制約の解法(Open #9)を先に確定してから 1 に着手。
