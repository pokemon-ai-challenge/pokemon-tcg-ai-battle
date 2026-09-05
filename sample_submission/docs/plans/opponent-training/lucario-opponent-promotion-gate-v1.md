# Lucario Opponent Promotion Gate v1(設計・結果前 freeze 提案)

作成日: 2026-07-26
種別: **昇格契約(pre-registration)**。frozen JSON = `kaggle_replays/opponent_training/lucario_opponent_gate_v1.json`。
位置づけ: Strong Opponent Initiative Phase 1(Mega Lucario)。Reference Pool の opponent を強くする第1号。
**Reference Pool v1(96868ca9)/ Gate2 Contract v1(ea496e9a)は変更しない**。Alakazam Champion(abl_5_full)も変更しない。

---

## 0. Phase A(read-only)要約

- **Lucario BC は既に学習済**([[project_multi_archetype_imitation]] のパイプライン成果物 `policy_weights_mega_lucario_ex.json`、
  production learning dir、自己検証 PASS)。今回はそれを read-only で検証。
- **Data**: 4975 replays → 1257 Lucario episode-players → **77,956 決定点**。split=**episode 単位**(md5、game 跨ぎ leakage なし)、
  train 62744 / val 8472 / test 6740。**rank は低位偏重**(test: ≤200 が 0 件・1000+ が 72%)。
- **Offline C1(同一 Lucario held-out test)**: generic **0.4393** vs Lucario BC **0.7037**(**Δ+0.264**)。
  MAIN(select_type0): generic 0.32→BC 0.67(+0.35)。= generic は Lucario をほぼ操縦できず、BC は明確に模倣。
- **own-deck**: 両者 `consequence_fields` 空 = **policy-only, own-deck-safe**(Open #9)。**explicit own_deck injection 不要**。
- **限界**: elite(rank≤20)データ 0% → BC は**中位 pilot 模倣=有能だが top-human 級ではない**。round-robin の Lucario
  field 勝率 38.7% も低い。「strong opponent」は **generic 比で明確に強い**が絶対強度は中程度。
- 先行 online 証拠: 案A ablation で lucario 専用 79.7% vs alakazam 重み(同デッキ)= mirror 結果はほぼ既知。

---

## 1. Gate 設計

### 1.1 Mirror(何を測るか)
「Lucario BC は generic policy-only より優れた **Lucario 操縦者**か」を、**同一 recipe を両者が操縦する mirror**(手番交互)で測る。

- Candidate = `abl_2_policy_only` + `policy_weights_mega_lucario_ex.json`(hash c4ca3599)
- Control  = `abl_2_policy_only` + generic `policy_weights.json`(hash 735dd38a)
- 両者 own-deck-safe なので measurement harness(`driver.run_sprt_ab`)をそのまま再利用。`run_type=lucario_opponent_promotion`。

### 1.2 統計処理(Gate2 の教訓を踏襲)
- **混合 5-recipe 系列に単一 SPRT を使わない**(recipe ごとに勝率 p_r が異なり iid でない)。
- **recipe 単位は同一デッキ mirror=iid Bernoulli** なので **per-recipe SPRT** は valid。
  δ_min=0.03 / α=0.05 / β=0.10 / n_max=600/recipe。offline 予測 ~80% のため PROMOTE は高速の見込み。
- **aggregate** = recipe-uniform mean の Candidate 勝率 + stratified Wald 片側 95% 下限。推定量は**二値勝率のみ**。

### 1.3 判定(結果前 freeze)
- **PROMOTE(= Lucario Champion v1)**:
  1. offline: BC held-out > generic → **✅ 既達(0.704 vs 0.439)**
  2. online: **5 recipe 中 ≥4 が SPRT=PROMOTE** ∧ **aggregate 勝率片側95%下限 > 0.53**(mirror tie=0.50 に +3pt=Gate1 δ_min 整合)
  3. integrity: error≤1% / dup0 / missing0
  4. **collapse なし**
- **collapse guard**(recipe 単位): Candidate 勝率 < 0.45 ∧ Wilson95 上端 < 0.50 → BC がその recipe で generic より明確に弱い → PROMOTE 阻止。
- **FAIL**: 過半 recipe が FUTILITY ∨ collapse 発生。
- **REVIEW**: それ以外(N 使い切りで無理に二値化しない。「昇格せず」≠「弱い」)。

### 1.4 Strong-opponent diagnostic(別保存・Reference Pool 非変更)
`abl_5_full`(Alakazam Champion, deck.csv)vs Lucario BC を各 recipe で fixed-N(200/recipe=1000 games)。
Alakazam の対 strong-Lucario 勝率(recipe 別 + aggregate)を出し、**frozen baseline 0.820 と比較(診断のみ)**。
低下は「opponent が強くなった」証拠であり Alakazam の弱化ではない。**Reference Pool v1 の 0.820 は不変**。`run_type=strong_opponent_diagnostic`。

---

## 2. Lucario Champion v1 registry(最小 metadata)

自動 manager は作らない。schema のみ(frozen JSON §champion_registry_schema):
`archetype / version / model{weights,hash,config,config_hash,training_dataset_hash} / evaluation{offline,mirror,recipes,diagnostic} / predecessor / status`。

**Lucario Champion v1 最低条件**: ① offline held-out で generic 超え(✅)② online mirror で generic を明確に上回る(§1.3)③ errors/integrity OK ④ recipe 横断で崩壊なし。

---

## 3. 再利用性(Phase F の土台)

本 gate は **archetype 引数化で archaludon/rocket_mewtwo/marnie/crustle へ横展開**する雛形。
- dataset extraction = `run_archetype_pipeline.py --archetype <arch>`(既存)
- offline = `policy_net/evaluate.py`(既存、arch features に向ける小改修のみ)
- online = measurement harness(既存)
- 各 Champion を将来 `archetype/version/deck recipes/policy weights/value weights/algorithm/training origin/predecessor/evaluation`
  で登録できる schema を用意(League 本体は今回作らない)。

---

## 4. freeze 対象・変更禁止

- **freeze(結果前)**: 本 gate の閾値(δ_min/α/β/margin 0.53/collapse 0.45)・recipe 集合・weights/config hash。
- **変更禁止**: Reference Pool v1(96868ca9)・Gate2 Contract v1(ea496e9a)・Alakazam Champion(abl_5_full)・production・cg/・deck.csv・shared league・opponents/。
- **今回やらない**: 全arch同時・Self-play/RL/League本体・explicit own_deck injection・production 大規模変更。
- git add/commit/push なし。実装(orchestrator)は承認後に新規 evaluation 領域(`kaggle_replays/opponent_training/`)で行う。
