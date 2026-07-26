# 実装計画書: Reference Pool v1 + Field Evaluation Gate

作成日: 2026-07-26
種別: **実装計画書**(設計書 [reference-pool-field-gate-design.md](./reference-pool-field-gate-design.md) を、対象ファイル・
責務・I/O・完了条件・production 影響・ownership まで落とす。**コードは書かない**。[[feedback_design_vs_implementation_plan]])
ステータス: **計画のみ。実装未着手。レビュー前に本実装へ進まない。** production/cg/main.py/deck.csv/weights/configs/
shared league(`league/`)/他者担当コード 変更なし。全成果物は `kaggle_replays/measurement/`(ローカル・git 未追跡・push 無し)。

---

## 0. 全体方針・制約

- **配置**: 新規実装は全て `kaggle_replays/measurement/`(未追跡、[[project_measurement_protocol]] の運用に準拠)。
- **既存 harness 資産の再利用**: `runner.play_game(agent0, agent1, deck0, deck1)` は既に **deck0≠deck1 対応**、`terminal.classify_terminal`、
  `agents.build/ensure_production_cwd`、`sprt.wilson_interval`、`driver.AgentSpec`、process 並列(`_par_*`)。
- **shared code**: `opponents/`(dragapult_rule + deck)・`league/`・`results/round_robin/_matrix.json`・
  `meta_analysis/` は **read-only 参照のみ**(登録済 agent/deck/hash を読む)。改変が要るなら「要担当者確認」で停止。
- **前提 blocker**: 設計 §11 Open #9(field opponent の deck.csv belief 制約)を**先に方針確定**してから Step 1 に着手。

---

## 1. Step: Reference Pool manifest(frozen)

- **対象**: `measurement/reference_pool_v1.json`(新規・データ)、`measurement/reference_pool.py`(新規・ローダ)。
- **責務**: v1 の opponent 群を frozen manifest 化。各 opponent stratum = {stratum_id, agent(id/kind: ml_policy/dragapult_rule),
  agent_config(あれば), opponent_deck recipe paths(archetype_decks/<arch>/NN.csv), 各 deck の hash, layer(core/coverage),
  field_share, strategy_axis}。recipe manifest(frozen sequence)もここで固定。
- **I/O**: 入力=設計 §2 の 6 メンバー + 各 archetype の recipe csv 群。出力=検証済 manifest dict(存在・hash・60枚を assert)。
- **完了条件**: 全 deck/agent が実在・hash 一致・60枚。`reference_pool.py` が manifest を読み各 deck を int list 化。frozen 化(再実行で同一)。
- **production 影響**: なし。**ownership**: 自作。opponents/archetype_decks は read-only 参照。

## 2. Step: Field runner(単一 stratum、deck0≠deck1)

- **対象**: `measurement/field_eval.py`(新規)。**`runner.play_game` を再利用**(deck0≠deck1 対応済)。
- **責務**: `run_field_stratum(cand: AgentSpec, opp_spec, cand_deck, opp_deck, n_games, workers, run_dir, resume) -> dict`。
  Candidate(deck.csv)を P0/P1 交互で opponent(opp_deck)と N 対戦。`_record_from_result` + `classify_terminal` で
  per-game レコード(勝敗・勝ち筋・terminal_flags・margin・recipe_id・agent_id)を outcomes.jsonl(truth source, resume 可)へ。
  process 並列は `driver._par_*` パターンを mirror(worker-init で chdir + 両 agent 構築、cg プロセス singleton のため process 並列必須)。
- **I/O**: 入力=cand/opp spec + 2 デッキ + N。出力=stratum_report{games/wins/losses/winrate/wilson_ci(二値のみ)/errors/
  prize_out/no_pokemon/deckout/terminal_flags 分布/recipe 別内訳}。
- **完了条件**: dragapult_rule stratum(自前デッキ)で errors0・winner 整合・outcomes 整合(len==N)。並列⇔逐次同値(既存 parallel_check 同型)。
  **前提**: Open #9 解決後(ml_policy opponent の belief)。
- **production 影響**: なし(harness 拡張、未追跡)。**ownership**: 自作。**注**: `run_sprt_ab` は mirror 専用なので改変せず、
  field 用は新規 driver にする(混同回避)。

## 3. Step: Open #9 — field opponent の deck belief 解法(Step 1 の前提)

- **対象**: 調査 + `field_eval.py` 内の opponent 構築方式(production ml_policy は**改変しない**)。
- **責務**: ml_policy が `read_deck_csv()`(cwd deck.csv)を自デッキ belief に使う制約への対処を1つ選ぶ:
  - (a) **opponent を自前デッキを正しく読む agent に限定**(dragapult_rule 型)。他アーキは資産不足(E)→ pool 縮小 or 新規相手作成(今回やらない)。
  - (b) **harness 側で opponent の belief を対応デッキに向ける**(例: opponent を別 process で cwd=対応デッキの dir にする、
    または ml_policy の deck 取得を明示注入できる口があるか調査)。production 改変が要るなら**要担当者確認で停止**。
  - (c) **belief 不整合を許容**(policy-only 系 opponent は belief 依存が小さい仮説を実測で確認)。
- **I/O**: 各案の belief 正確性・実装コスト・production 触るか の比較。
- **完了条件**: 方針決定(ユーザー確定)。**production 影響**: (b)で ml_policy 改変が必要なら発生 → その場合は停止・報告。**ownership**: ml_policy は他者/production。

## 4. Step: Gate 2 集計(stratified、Δ_field + guard)

- **対象**: `measurement/gate2.py`(新規)。`sprt.wilson_interval` 再利用。**単一 SPRT を混合系列に使わない**(設計 §4.1)。
- **責務**:
  - Candidate と Champion を同一 Reference Pool で `field_eval` 実行(各 stratum の stratum_report を収集)。
  - **Primary = meta-weighted Δ_field**: Σ(field_share × (cand_wr − champ_wr)) / Σ(field_share)。equal-weight も併記(診断)。
  - **Guard = per-archetype catastrophic regression**: 各 core/coverage stratum で `champ_wr − cand_wr ≥ 0.10` かつ
    Candidate/Champion の Wilson CI が非重複 → guard 発火(設計 §5、閾値/CI はレビュー確定)。
  - 出力=field_report{per_stratum(cand/champ の全項目), meta_weighted_delta, equal_weighted_delta, guard{発火 stratum}, 昇格判定}。
- **I/O**: 入力=field_eval の stratum_report 群 + field_share。出力=field_report JSON(再現メタ付き)。
- **完了条件**: 既知データで sanity(Champion vs 自 pool の meta/equal weighting が §4.3 の 0.705/0.607 を再現)。guard が
  合成ケース(overall+, 1 stratum −15pt)で正しく発火。**production 影響**: なし。**ownership**: 自作。

## 5. Step: Champion 判定 + metadata(manager は未実装)

- **対象**: `measurement/champion_record.py`(新規、schema + シリアライズのみ)。**Champion manager 本体は今回作らない**。
- **責務**: Gate1(既存 SPRT report)+ Gate2(field_report)を束ね、昇格判定 = `Gate1 PASS(PROMOTE) ∧ Gate2 Primary Δ_field>閾値
  ∧ guard 非発火`。合格時の champion_record metadata を出力(設計 §9 schema: champion_id/timestamp/source_ref/config・weights・
  deck hash/gate1_report/gate2_report/predecessor)。
- **I/O**: 入力=gate1 report + gate2 field_report。出力=champion_record JSON(判定 + metadata)。実際の Champion 差し替え・
  Past Champion Pool 移動は**しない**(schema と判定のみ)。
- **完了条件**: 合格/不合格ケースで正しい判定 + metadata 生成。**production 影響**: なし。**ownership**: 自作。

## 6. Step: 予算・実行(desktop 並列)

- **対象**: `measurement/run_field_gate.py`(新規・オーケストレータ、**__main__ ガード必須**=Windows spawn)。
- **責務**: Reference Pool v1 × N(推奨 250)× {Candidate, Champion} を desktop 並列(workers=15)で実行、resume 可。
  設計 §7 の予算(6arch×250×2 ≈ 3000 試合 ≈ ~90分)を desktop で。frozen manifest + 再現メタ保存。
- **完了条件**: 小 N スモーク(各 stratum 4 試合)で通し動作 + A-0 型 hash ガード。desktop 現行化(前回同期済、実行前に再確認)。
- **production 影響**: なし。**ownership**: 自作。**注**: `python -c`/heredoc から並列起動しない(BrokenProcessPool)。

---

## 7. 実装順(推奨)

1. **Open #9 の方針確定**(Step 3、前提 blocker)。
2. Reference Pool manifest(Step 1)。
3. Field runner(Step 2)+ dragapult_rule stratum で検証。
4. Gate 2 集計(Step 4)。
5. Champion 判定 metadata(Step 5)。
6. オーケストレータ + desktop 実行(Step 6)。

## 8. production / shared / ownership 影響 まとめ

- **変更なし**: production(main.py/ml_policy/weights/configs/deck.csv)、cg/、shared league(`league/`)、opponents/、archetype_decks。
- **新規のみ**: `measurement/` 配下(reference_pool_v1.json / reference_pool.py / field_eval.py / gate2.py / champion_record.py /
  run_field_gate.py)= 全て未追跡・push 無し。
- **要担当者確認になり得る点**: Open #9(b) で ml_policy(production)の deck 注入口が必要になった場合 → 実装せず停止・報告。

---

## 9. レビューで確定すべき事項(実装前)

設計 §11 Open decisions と同一(Reference Pool v1 メンバー / 2層分離 / field weighting / recipe allocation / Gate2 採用条件 /
regression guard 閾値 / 試合予算 / Champion 昇格条件 / **Open #9 の解法** / 今回やらないこと)。**全て確定後に実装着手**。

---

## 10. MVP 実装完了(2026-07-26)

Reference Pool v1 + Field Gate の **MVP** を `kaggle_replays/measurement/`(全て git 未追跡)に実装し、
小規模 smoke で通し動作を確認した。**full 3000 試合・Champion 昇格判定は未実施**(閾値 OPEN のため REVIEW 止まり)。

**実装ファイル**(§8 の想定名から実際の命名へ):
- `reference_pool.py` + `reference_pool_v1.json` — frozen manifest(6 member)。build/validate/allocate_games。
- `field_eval.py` — Field runner(`run_stratum`)。evaluated(Candidate/Champion=abl_5_full, deck.csv)×
  1 opponent×1 recipe を N 対戦。deck0≠deck1・手番 game_index%2 交互・`runner.play_game` 経由で毎 game reset。
  process 並列は `driver._par` を mirror。opponent は policy-only ml_policy(abl_2) or dragapult_rule のみ(Open #9 Safe)。
- `field_aggregate.py`(= 計画の gate2.py)— per-stratum / per-archetype / equal-weight / meta-weight /
  Δ_field / regression guard(閾値は manifest から。champ−cand≥閾値 かつ Wilson95 CI 非重複で発火)。
  **推定量は evaluated_won 二値のみ**(margin/勝ち筋は diagnostic で winrate/CI/Δ に非混入=不変性テスト有)。
- `field_gate.py`(= 計画の champion_record.py + run_field_gate.py 相当)— オーケストレータ。
  Candidate/Champion を同一 Reference Pool で走らせ `field_outcomes.jsonl`(真実源)/`field_report.json`/
  `field_manifest.json` を保存。resume は frozen allocation 再割当せず stratum 単位で done 数 skip。
  Champion decision metadata(gate1=外部 SPRT 参照 slot / gate2=primary_metric・candidate/champion value・
  delta_field・guard_pass・per_archetype)。**promotion_recommendation は常に REVIEW**(Gate2 primary 閾値 OPEN, §11 #5)。
- `step_field_smoke.py` — smoke オーケストレータ(__main__ ガード)。既定 Candidate=Champion=abl_5_full の A/A 相当。

**テスト**(新規 18・既存 23 = 41 green):
- `test_reference_pool.py`(7): build+validate OK / member 欠落・deck hash 不一致・deck 不在・recipe 重複・own_deck hash 不一致 reject / allocate 保存。
- `test_field_eval.py`(5): 手番交互 + stratum メタ / start_index(resume)オフセット / deck0≠deck1 で error 0 / **per-game reset が field 経路でも毎 game 呼ばれる** / evaluated 例外 → error 記録・evaluated_won=0。
- `test_field_aggregate.py`(6): per-stratum 集計 / equal vs meta 区別 / per-arch Δ / guard 発火(壊滅 stratum)/ guard pass(小 N 非発火)/ **不変性(margin・勝ち筋を変えても winrate/CI/Δ 不変)**。

**smoke 結果**: 6 member 全走行・全 recipe 参照・reset 正常・errors 0・result schema/aggregation 正常を確認(強弱結論は出さない)。

**未着手(次フェーズ・要レビュー)**: full 3000 試合の desktop 実行 / Gate2 primary 閾値の確定(現状 OPEN=REVIEW 止まり)/
regression guard 閾値の provisional(-10pt)確定 / Gate1(現 Champion との SPRT)との統合判定。

---

## 11. Gate 2 Decision Contract v1 → 実装 gap / diff plan(2026-07-26、未実装)

契約は `measurement/gate2_contract_v1.json`(design §13)で **事前固定済**。較正エンジン=`gate2_calibration.py`
(cg 非使用の合成 MC、power/FWER 根拠)。**契約決定に不要な実装変更は今回しない**(Step 20)。現行 MVP との差分は下記の
**局所変更のみ**(全て `measurement/` 未追跡領域で完結、production/cg/shared 非改変):

**field_aggregate.py の diff(3点)**:
1. **Primary CI 追加**: `meta_weight` に `delta_field` の analytic stratified Wald SE と one-sided 95% CI [L,U] を追加
   (`Var=Σw_i²[pC(1−pC)/N + pH(1−pH)/N]`)。`primary_state ∈ {STRONG, REGRESSION, INCONCLUSIVE}` を τ=+2pt/α=.05 で付与。
   ※ `gate2_calibration.primary_decision` をそのまま移植可(検証済ロジック)。
2. **Guard 差し替え**: 現行の `ci_separated_candidate_below`(Wilson 非重複)を **Δ_i=pC−pH の Wald diff CI + Holm 補正**へ変更。
   fire=(point≤−10pt ∧ Holm 片側95%上端<0)。`gate2_calibration.guard_decision(method="holm")` を移植。
   ※ 現行の Wilson 非重複はより保守的だが多重比較未補正=契約と不一致。要差し替え。
3. per-archetype に `delta_ci`(Δ_i の片側CI)と `holm_reject` フラグを追加(監査可能性)。

**field_gate.py の diff(1点)**:
4. **PASS/FAIL/REVIEW 実装**: 現行 `promotion_recommendation` 固定 REVIEW を、契約の decision 関数へ置換
   (Gate1 結果=外部 slot を引数化、Primary state + Guard + integrity(error_rate≤1%)で分岐)。
   `gate2_calibration.contract_decision` が参照実装。`gate2_contract_v1.json` を読み込み閾値を hard-code しない。

**テスト追加(次フェーズ)**: contract 分岐の unit(STRONG→PASS / REGRESSION→FAIL / guard→FAIL / INCONCLUSIVE→REVIEW /
integrity→REVIEW)、Holm 補正の FWER 回帰、analytic CI と bootstrap の一致。**いずれも合成 record で cg 不要**。

**規模**: 純追加 + guard ロジック1本差し替え。大改修なし。契約 freeze 後に別フェーズで実施(レビュー後)。
