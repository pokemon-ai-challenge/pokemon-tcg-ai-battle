# 設計書: Reference Pool v2 + Gate 2 Contract v2

作成日: 2026-07-26
種別: **設計書**(方向性・選定根拠。関数シグネチャ/完了条件は別 implementation-plan)。
前提: Reference Pool v1(hash 96868ca9)/ Gate2 Contract v1(hash ea496e9a)完了・freeze。Strong Opponent Initiative で
Lucario/Archaludon/Marnie Champion v1 昇格、Crustle は REVIEW(境界)。[[project_multi_archetype_imitation]] [[project_worthy_opponent_inventory]]。
ステータス: **設計のみ。実装・online・calibration 実行は未着手。v1/Contract v1/Champion/production 変更なし。read-only 調査に基づく。**

---

## 0. Background / v1 limitations

Reference Pool v1 は Candidate/Champion の**固定相対比較器**として正常動作(3000-game A/A baseline: Δ_field −0.20pt,
INCONCLUSIVE, integrity PASS)。だが opponent は **dragapult_rule 以外すべて generic policy-only(弱 AI)**。Strong Opponent
Initiative で、専用 BC opponent へ置換すると Alakazam Champion(abl_5_full)の勝率が大きく下がることが判明:

| archetype | generic 相手(v1 baseline) | strong 相手 | opponent-strength correction | strong mirror agg | offline BC Δ |
|---|---|---|---|---|---|
| Lucario | 0.820 | **0.478** | −34.2pt | 0.8176 | +0.264 |
| Archaludon | 0.940 | **0.550** | −39.0pt | 0.7754 | +0.259 |
| Marnie | 0.840 | **0.442** | −39.8pt | 0.8855 | +0.114 |
| Crustle | 0.664 | (作れず) | N/A | 0.5422(REVIEW) | +0.147 |

**含意**: v1 の generic opponent は opponent 絶対強度を **~35-40pt 過小評価**していた。v1 は historical benchmark として有効だが、
「複数の十分強い opponent に対して Candidate が改善したか」は測れない。**v2 は Strong Opponent を正式に組み込む**。

**Crustle 境界(重要)**: imitation-based Strong Opponent 生成は archetype 依存で失敗しうる。Crustle(control/壁)は BC が
online で generic を明確に上回れず(2/5 recipe, agg 0.5422, REVIEW)。**強引に Strong 扱いしない**。将来は hand-crafted/search/
value-guided/RL が候補(今回作らない)。

---

## 1. v1 は永久保持(書き換えない)

**Reference Pool v1 = Historical / Compatibility Benchmark**。用途: 過去 Candidate 比較 / generic 環境での回帰 / 長期トレンド /
fixed benchmark。v2 を作っても v1 manifest / baseline / Contract v1 を**上書きしない**(別 version・別 hash)。

---

## 2. Reference Pool v2 メンバー(推奨)

archetype 構成は v1 と同一(6)。**Strong Opponent が存在する archetype だけ opponent policy を upgrade**:

| archetype | v1 opponent | **v2 opponent** | opponent_strength_class | meta share | 採用理由 |
|---|---|---|---|---|---|
| mega_lucario_ex | generic | **Lucario Champion v1** | strong_champion | 17.45 | 最大 share・強化実証(−34.2pt) |
| archaludon_ex | generic | **Archaludon Champion v1** | strong_champion | 15.28 | 2番目 share・強化実証(−39.0pt) |
| marnie_grimmsnarl_ex | generic | **Marnie Champion v1** | strong_champion | 5.51 | elite23%・最大 shift(−39.8pt) |
| dragapult_ex | dragapult_rule | **dragapult_rule(不変)** | dedicated_rule | 6.35 | v1 で既に強・変更不要 |
| crustle | generic | **generic policy-only(維持)** | generic_policy | 7.08 | BC=REVIEW。coverage 維持で generic 残す(§3 Option A) |
| rocket_mewtwo_ex | generic | **generic policy-only(維持)** | generic_policy | 1.76 | strong 未作成(§4)。coverage として残す |

→ **v2 = Strong 4体(BC champion 3 + dragapult_rule)+ generic 2(crustle/rocket)**。
**注**: dragapult は v1 で既に dragapult_rule=v1→v2 で不変。upgrade は lucario/archaludon/marnie の 3 archetype のみ。

各 member に **`opponent_strength_class`(strong_champion / dedicated_rule / generic_policy)** を明示し、report で Strong subset /
Generic coverage / Overall を別集計可能にする(§6)。

---

## 3. Crustle の扱い(推奨: Option A)

- **Option A(推奨)**: generic Crustle を維持。archetype coverage を保ちつつ REVIEW の BC を Strong に入れない。
- Option B: generic + Crustle BC を diagnostic-only で両保持(複雑・v2 Primary には generic のみ)。
- Option C: Crustle を Primary から外す(coverage 低下=非推奨)。
→ **Option A**。Crustle BC(REVIEW)は Primary/Strong に採用せず、境界結果として doc 記録のみ。

## 4. Rocket の扱い(推奨: generic 維持)

data 最小(2003 test)・offline Δ 最小(+0.049)・Alakazam は既に generic Rocket に負け越し(0.324)。**strong champion を作らない**。
ただし generic Rocket は「Alakazam が既に苦戦する opponent」として情報量があり、coverage member として残す。

---

## 5. v2 の目的

Candidate が「弱い generic bot を倒せるか」でなく **「複数の十分強い opponent に対して改善したか」** を測る。
v1: deck diversity ◎ / opponent strength △。**v2: deck diversity ◎ / opponent strength ○〜◎**(strong 4/6, meta weight で 83.5%)。

---

## 6. Strong / Generic を report で区別

member 属性 `opponent_strength_class` により以下を別々に出す:
- **Strong subset performance**(lucario/archaludon/marnie/dragapult = 4体)
- **Generic coverage performance**(crustle/rocket = 2体)
- **Overall field performance**(6体、meta-weighted)

---

## 7. Primary Metric(推奨: meta-weighted Δ_field 維持 + Strong subset を secondary)

- **Primary = meta-weighted Δ_field = Σ w_i(pC_i − pH_i)**(v1 と同一、external validity=leaderboard)。weights は v1 の
  field share 正規化を**維持**(archetype 構成不変)。
- **重要な自然な性質**: v2 では strong 4体が meta weight の **83.5%**(lucario0.327+archaludon0.286+dragapult0.119+marnie0.103)を
  占める。→ **Primary meta-weighted Δ_field は v2 で自動的に strong opponent 主導**になる(generic crustle0.133+rocket0.033=16.5% のみ)。
- **Strong subset Δ_strong = strong 4体の(Candidate−Champion)を recipe/opponent-uniform で** 必ず secondary diagnostic として report
  (§9 Option B)。**Primary へ昇格させない**(単純さ優先。meta-weight が既に strong を主導するため冗長)。
- Generic members: **Option 2**=6体すべて Primary(meta-weighted)に入れる + Strong subset を必ず別 report。archetype coverage 維持。

## 8. Meta weighting の再評価

weights 自体は v1 と同一(構成不変)。effective #arch = 1/Σw² = **4.31**(不変)。だが **baseline winrate が激変**:
v2 Champion 各 opponent 勝率 = lucario0.478 / archaludon0.550 / marnie0.442 / dragapult0.240 / crustle0.664 / rocket0.324。
**meta-weighted Champion field winrate ≈ 0.487**(v1 は 0.751)。→ near-even field=より discriminating だが、勝率が 0.5 近傍=**分散増大**(§12)。

## 9. Strong subset metric(推奨: Option B)

- A: meta-weighted Δ_field のみ(strong を暗黙に含む)。
- **B(推奨)**: meta-weighted Δ_field(Primary)+ **Δ_strong(strong 4体の Candidate−Champion)を secondary metric**。
- C: Strong subset を Primary 昇格(過剰)。
→ **B**。Primary は変えず、strong subset を必須 secondary + per-opponent Δ を guard(§下記)で監視。

## 10. Generic による歪みの扱い(推奨: Option 2)

v2 も完全 Strong Field でない(crustle/rocket generic)。**Option 2**=6体すべて Primary + Strong subset を必ず別 report。
Option 3(strong4=Primary, generic2=coverage guard)は将来 v3 で検討可だが、v1 との連続性・単純さのため v2 は Option 2。

---

## 11. Regression Guard(v1 思想維持)

per-archetype Δ_i = pC_i − pH_i、**Candidate−Champion**(絶対 floor でない、Step 14)。Holm 補正・閾値 −10pt を全 6 member に一律。
- **Strong opponent で意味が変わる点**: 例 Alakazam vs Marnie Strong = 0.442。Candidate が 0.442→0.55 は価値大、0.442→0.34 は重大退行。
  絶対値が低くても **Candidate−Champion の差** で測るので v1 と同一枠組みで機能する。guard は「strong 相手を今より壊さない」を保証。

---

## 12. Gate 2 Contract v2 の Power 再設計(v1 を無条件コピーしない)

**baseline が 0.5 近傍へ寄り分散が増える**。v2 A/A(Cand=Champ)の analytic SE(Δ_field)= √(Σ w_i²·2p_i(1−p_i)/N):

| N/arch | v2 SE(Δ_field) | 参考: v1 実測 SE |
|---|---|---|
| 250 | **2.12pt** | 1.53pt(v1、勝率が極端でSE小) |
| 350 | 1.79pt | — |
| 400 | 1.68pt | — |
| 480 | ~1.53pt(v1 相当) | — |

→ **v2 は v1 と同じ N=250 だと SE が ~39% 大きい**(strong 相手で勝率が 0.5 近傍)。v1 の精度に合わせるには N≈480。
**推奨**: 実装フェーズで v2 baseline(上記勝率)を入力に synthetic power analysis(gate2_calibration.py 相当)を回し、
**N=350〜400 を第一候補**として PROMOTE 力(+3/+5pt)と guard 検出力を確認してから freeze。**数値契約は結果前に再較正・freeze**。

## 13. Contract v2 候補(統計方式は v1 ベース、数値は再較正)

再評価対象: Primary practical threshold(τ)/ α / N per archetype / Primary CI method / Guard threshold / Holm / PASS-FAIL-REVIEW。
- **流用可(方式)**: analytic stratified Wald CI、Holm guard、PASS/FAIL/REVIEW の 3 分類ロジック、recipe fixed-balanced allocation。
- **再較正必須(数値)**: N(250→350-400 候補)、τ(v1=+2pt を維持可か power で確認)、guard −10pt(strong 相手の SE 増で検出力再確認)。
- freeze は v2 calibration report + v2 manifest hash 紐付けで、online 結果前に固定。

## 14. Guard の意味(strong で不変の思想)

上記 §11。strong opponent は絶対勝率が低い(0.24-0.55)ため、absolute floor guard は不適。**Candidate−Champion の per-arch Δ を guard**
とする v1 思想をそのまま維持。閾値 −10pt は power 再確認の上で流用候補。

---

## 15. v2 baseline 計画(実装後の最初の正式 run。今回実行しない)

**A/A v2 baseline**: Candidate = Champion = abl_5_full を Reference Pool v2 に対し N 対戦。目的: v2 null calibration(Δ_field≈0 期待)/
Current Champion vs Strong Field snapshot / throughput / Contract v2 実ゲーム確認。budget = 6arch×N×2。

## 16. v2 Current Champion baseline で欲しいもの

Alakazam vs {Lucario Champion / Archaludon Champion / Marnie Champion / dragapult_rule / generic Crustle / generic Rocket}。
**既存個別 diagnostic(200/recipe, field_eval flat pool)を v2 baseline に流用しない**理由: v2 は field_gate manifest runner + Contract v2
の N/allocation で走る=runner/N/manifest が異なる。**流用条件**(同一 agent/config/weights/recipe/N/runner/contract)を満たさないため、
**v2 baseline で全 opponent を統一 runner・統一 N で再測**する。個別 diagnostic は方向性の事前情報としてのみ使用。

---

## 17. v1 / v2 の役割分担(推奨)

| | 役割 | 使用タイミング |
|---|---|---|
| Gate 1 | Champion H2H SPRT(δ_min 0.03/0.05) | 全 Challenger(安価な早期棄却) |
| **Gate 2 v2** | **Strong Field**(promotion 主判定) | 全 Challenger(Gate1 通過後) |
| Reference Pool v1 | Historical / regression diagnostic | **任意・定期**(毎回でなく回帰確認時) |

**Challenger 昇格 = Gate1 PASS ∧ Gate2 v2 PASS**。**v1+v2 を毎回両方正式 Gate にしない**(試合数倍増=費用対効果悪)。v1 は
periodic な historical 回帰チェックに留める。

## 18. Training League への接続(今回 self-play 実装しない)

v2 manifest を将来 league members へ変換可能に設計。league 候補 = Current Alakazam Champion + Strong(Lucario/Archaludon/Marnie/
dragapult_rule)+ Coverage(Crustle/Rocket generic)+ Past Alakazam Champions。member schema は §22 の strength metadata を流用。

## 19. Evaluation Pool と Training Pool を分けるか(推奨: 分ける方向)

**評価 Pool = 学習 Pool を完全同一にすると evaluation opponent への過学習リスク**。
- A: 同一(単純だが過学習)。
- **B/C(推奨方向)**: 一部共有 + 一部 held-out opponents、または recipe split で training/evaluation 分離。
→ v2 設計では**方針だけ**「held-out opponent/recipe を Gate2 専用に確保」を決め、self-play 実装時に具体化。今回実装しない。

## 20. Recipe leakage(現状 5 recipe の制約)

Strong Champion は replay の pilot **行動**で学習(recipe=deck list そのものでは学習していない)が、Gate2 と将来 self-play が同じ 5 recipe を
使うと recipe-specific overfitting の懸念。**現 deck 資産 = archetype_decks/<arch>/ に ~5 recipe/arch のみ**(新規 recipe を捏造しない)。
分離案: 5 recipe を **train3 / held-out-eval2** に分割(eval robustness は落ちる)。→ 実装フェーズで recipe 在庫を確認し held-out 余地を調査。

## 21. Crustle 境界を v2 doc へ明記

**imitation-based Strong Opponent 生成は archetype 依存で失敗する**(control/壁 Crustle=REVIEW)。v2 では Crustle を generic のまま残し、
将来の Crustle strong opponent は hand-crafted dedicated agent / search / value-guided policy / RL を候補とする(今回作らない)。

## 22. Strong Opponent 品質 metadata(schema のみ)

将来 opponent 増加時の品質記述(単一 scalar rating はまだ作らない):
```
strength_source: bc_promoted | dedicated_rule | generic
promotion_evidence: {mirror_aggregate, recipe_pass_count, gate_hash}
external_strength: {alakazam_diagnostic_winrate}
```

---

## 23. Risks / Open decisions

**Risks**:
- v2 は strong 相手で勝率 0.5 近傍=**分散増**→ 同 N で power 低下(N 再設計必須)。
- crustle/rocket generic のまま=v2 も完全 Strong Field でない(coverage と strength のトレードオフ)。
- 5 recipe しかなく training/eval 分離余地が限定的。
- Strong champion は elite0%(lucario/archaludon)の中位模倣=絶対 top-human でない。v2 は「generic より遥かに強い」opponent field。

**Open decisions(実装前にユーザー確定)**:
1. v2 メンバー(上記 6、upgrade 3 で良いか)。
2. Crustle=generic 維持(Option A)で良いか。
3. Rocket=generic 維持で良いか。
4. Primary=meta-weighted Δ_field 維持 + Strong subset Δ_strong を secondary。→ **【解決】Option A(Δ_strong=diagnostic のみ、gate 化しない。MC で二重評価不要)**。
5. Generic を Primary に入れる(Option 2)で良いか。
6. Contract v2 の N。→ **【解決】N=400(fixed)。2026-07-26 MC 較正(gate2_calibration_v2.py)**。
7. τ / guard 閾値の再較正。→ **【解決】τ=+2pt(v1 維持)/ Guard G=−8pt(v2 再較正、Holm、FWER 4.6%)。詳細は下記 §Contract v2 + gate2_contract_v2_prebind.json**。
8. v1/v2 役割分担(Gate2 v2=promotion, v1=optional historical)で良いか。
9. Evaluation/Training 分離方針(held-out 確保)で良いか。
10. 今回やらないこと(v2/Contract v2 実装・online・league・self-play)。

---

## 25. Gate 2 Contract v2(2026-07-26 較正・prebind freeze)

online 前 pre-registration。engine=`measurement/gate2_calibration_v2.py`(cg-free MC 30000reps, seed 20260726, report hash
24c0980d)、frozen 版=`measurement/gate2_contract_v2_prebind.json`(hash **722c37a7**、manifest 未 bind=Option A)。
v2 plausible baseline(真値でない): Champion 対 opponent 勝率 lucario0.478/archaludon0.550/marnie0.442/dragapult0.240/
crustle0.664/rocket0.324。meta-weighted Champion field=0.486、Strong Σw=0.835。

### 25.1 Primary(v1 方式維持・数値 τ=+2pt)
- meta-weighted Δ_field、analytic stratified Wald 片側95% CI、α=0.05、**τ=+2pt**。STRONG/REGRESSION/INCONCLUSIVE。
- SE(Δ_field): N250=2.12 / N350=1.79 / **N400=1.68** / N500=1.50pt(v1 実測1.53pt=strong 相手で分散増)。
- power(N400): false-STRONG=α=5.0%、+2pt→33%、+3pt→56%、+5pt→91%。

### 25.2 Guard(v2 再較正 G=−8pt)
- per-arch Δ_i=Candidate−Champion、Holm、**G=−8pt**、fire=(point≤−8pt ∧ Holm 片側95%上端<0)。
- **再較正根拠**: strong opponent baseline が 0.5 から離れ per-arch SE 小(N400 strong mean≈3.3pt)→ 解像度 2.4·SE≈8pt。v1 の −10pt
  (p=.5/N=250 用)を流用せず v2 解像度へ。**FWER 4.6%**(v1 が −10pt で狙った ~5% と同水準)、reg −10pt→77%/−12pt→93%/−15pt→99.7%。

### 25.3 N / extension / budget
- **N=400 固定**(4800 games、~79min 見込み)。**adaptive extension 禁止**(optional stopping N400→500 は null false-PASS を 5.2→7.1% に
  増やす=type-I 膨張)。borderline REVIEW は別 pre-registered run。
### 25.4 決定・operating characteristics(N400/τ2/G8, MC)
- PASS=Gate1 PROMOTE∧Primary STRONG∧Guard 非発火∧integrity(err≤1%)/ FAIL=Gate1 FUTILITY∨REGRESSION∨Guard∨integrity failure / REVIEW=他。
- null: PASS 5.1% / FAIL 8.4% / REVIEW 86.5%。+2pt: PASS 33% FAIL 1.0%。+5pt: PASS 91%。**hidden catastrophe(marnie−15)→FAIL 97.9%、dragapult−20→FAIL 100%**。
  null false-FAIL 8.4% は Primary-regression+Guard(FWER4.6%)安全2重の和(Gate2 到達 candidate は Gate1 通過済=真 null 稀)。
- **manifest bind**: 実装後 member/class/weight(expected を contract に保存)一致で hash bind→gate2_contract_v2.json 最終 freeze。不一致は bind 拒否。

---

## 24. 今回やらないこと
Reference Pool v2 実装 / manifest v2 生成 / Gate2 Contract v2 freeze / online A/A baseline / Challenger 評価 / Rocket Champion /
Crustle 再学習 / Training League / Self-play / RL / ISMCTS / Transformer / production 変更。**設計 doc + implementation-plan doc のみ**。
