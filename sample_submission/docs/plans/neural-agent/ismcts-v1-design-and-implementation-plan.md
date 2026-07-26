# ISMCTS v1 — 設計 + 実装計画(First Major Challenger)

作成日: 2026-07-26
種別: 設計 + 実装計画。前提: 評価基盤 freeze(Reference Pool v2 14db8345 / Gate2 Contract v2 05e509a2、A/A baseline PASS)。
研究質問: **Current Champion(abl_5_full)の Policy/Value/Belief/feature/weights をできる限り固定したまま、意思決定時の
Search を ISMCTS へ拡張すると Champion より強くなるか**。ステータス: **Phase A(実コード把握)完了。実装未着手。
production/cg/main.py/deck.csv/weights/configs/Reference Pool/Gate2/Strong Opponent 変更なし。**

---

## Phase A: Current Champion(abl_5_full)の意思決定経路(実コード=source of truth)

`configs/abl_5_full.json` + `ptcg_ai/ml_policy/ml_policy_agent.py` + `ptcg_ai/search/{pipeline,leaf_eval,lethal_simple}.py`
+ `hidden_information/{match_context,search_adapter}.py` を実読して確定(記憶・doc からの推測でない)。

### A0. 決定経路(図)
```
agent(obs) → _select_action(obs):
  1. _try_lethal(lethal_simple)       確定リーサル探索(time 100ms/depth20/nodes1e4)。found→return
  2. _try_pipeline(pipeline.search)   ← 中核 Search(PIMC 風)。SelectType.MAIN ∧ maxCount==1 のみ
       Step2 Policy が option を score → top_k=4 first-move 候補。top1 集中≥0.9 なら top1 即決(shortcut)
       Step3 hidden_state_factory を num_determinizations=8 回 → 8 決定化世界
             (hidden_state_source="estimated" → match_context 信念: own_state+opp_state を search_adapter で)
       Step4 各(世界,候補): search_begin → 候補 first move → 自ターン残りを Policy 貪欲展開 →
             相手ターンを opponent_depth=1 回 Policy 貪欲(相手モデル)→ leaf_eval.evaluate(leaf, me) [handcrafted]
       Step5 候補ごと 8 世界平均 → 最大採用。差≤tie_eps(0.02)なら Policy 上位。found→return
  3. fallback(pipeline None=非MAIN/予算切れ/失敗): Policy argmax(model.select_option)
     ※ abl_5_full config に attack_hybrid / attack_plan キー無し → 両者スキップ(NOT USED)
```

### A1. component USED / NOT USED(abl_5_full 実 config)
| component | 状態 | 使用箇所 |
|---|---|---|
| Policy Model(policy_weights.json 735dd38a) | **USED** | pipeline top_k 候補生成 / 自ターン貪欲展開 / 相手モデル / fallback argmax |
| Value Network | **NOT USED** | leaf_eval="handcrafted"(value でない)。value_shadow_logging は shadow log のみ=意思決定不使用 |
| Belief / hidden information(match_context estimated) | **USED** | pipeline 決定化(to_search_begin_kwargs)、lethal の estimated |
| opponent deck predictor / own deck+prize estimator | **USED** | match_context 信念の一部(決定化入力) |
| simulator(cg search_begin/step) | **USED** | pipeline / lethal |
| legal action enumeration | **USED** | obs.select.option |
| attack_plan | **NOT USED** | config にキー無し |
| lethal_simple | **USED** | _try_lethal |
| 既存 Search(pipeline=PIMC) | **USED** | 中核。**= 浅い決定化探索(top_k4×8世界×相手1手×rollout40×handcrafted leaf)** |
| rule veto(attack_hybrid) | **NOT USED** | config にキー無し |
| fallback | **USED** | Policy argmax |

### A2. 最重要発見
**abl_5_full は既に「決定化 Search 」agent**(pipeline=PIMC-lite)。ISMCTS v1 は search-less agent に search を足すのでなく、
**既存の浅い depth-1 決定化探索を、Policy prior・信念決定化・handcrafted leaf・lethal をそのまま再利用したまま、
proper なツリー探索へ深化する**もの。→ "search-only difference" が明確に定義できる。

### A3. ISMCTS v1 で **固定(freeze)する**もの(= Champion と同一)
Policy weights(735dd38a)/ feature encoder / **leaf eval = handcrafted(value 未使用のまま)** / Belief=match_context estimated /
determinization=`search_adapter.to_search_begin_kwargs` / lethal_simple 前段 / terminal 判定(cg)/ deck(deck.csv)/ fallback。
→ **Value/Belief は今回の実験変数にしない**(§Phase H の S1/S2 は別 ablation)。

---

## Phase B: Challenger Contract(実験差分の事前定義)
- **Champion**: abl_5_full = lethal → pipeline(浅 PIMC)→ policy fallback。
- **Candidate(ismcts_v1)**: lethal → **ISMCTS(深ツリー、同 Policy prior/信念決定化/handcrafted leaf)** → policy fallback。
- 差分は **Search 構造のみ**(pipeline の depth-1-per-candidate rollout → 反復ツリー探索)。deck/weights/encoder/Value/Belief/lethal は同一。
- **実装形態**: production(ml_policy_agent.py)は変更しない。**別モジュール**(`kaggle_replays/search/ismcts.py` +
  `kaggle_replays/challengers/ismcts_v1_agent.py`)が production の PolicyModel / leaf_eval / search_adapter / match_context /
  lethal_simple を **read-only import** して再利用し ISMCTS を足す。config `ismcts_v1` = abl_5_full + `ismcts` ブロック。
- **metadata**: challenger_id=ismcts_v1 / parent=abl_5_full / search_algorithm=ISMCTS / policy_hash 735dd38a /
  value_hash=N/A(未使用)/ config_hash / deck_hash 45c681cc / search_config_hash / source_ref。

---

## Phase C: ISMCTS 設計

### C0/C1 leakage-safe determinization(再利用)
Opponent の真の hidden state を Search へ渡さない。決定化は **`search_adapter.to_search_begin_kwargs(match_context 信念)`**
(pipeline と同一の口)を毎 iteration/世界で呼び、observed cards / known deck / remaining counts / legal constraints に整合。
**Belief 自体は実験変数にしない**(Champion が使う信念をそのまま)。→ F0 leakage test で ground-truth 非参照を保証。

### C2 information-set node
node key = **観測可能情報のみ**(observable state signature + decision context + legal action signature + current player +
turn/phase)。**hidden ground truth を key に混ぜない**。同一情報集合の複数決定化 state で統計を共有(ISMCTS の要)。
実装: 各 iteration で 1 決定化 world を sample、root は情報集合、tree は情報集合 node、edge=legal action。cg の
searchId は world ごとの完全 state。visit/Q は情報集合 node（=観測 key）に集約。

### C3 selection(PUCT、Policy prior 再利用)
`score(a) = Q(a) + c_puct · P(a) · √(ΣN) / (1+N(a))`。P(a)=Champion Policy の option 確率(softmax(score_options))。
Q は zero-sum backup（自視点 [-1,1] or leaf の [0,1] を [-1,1] へ写像）。c_puct は config。

### C4 policy prior(Champion Policy そのまま)
`PolicyModel.score_options(obs, factory, deadline)` → legal action の prior。**新規学習なし**。Policy action と simulator
legal action の mapping を厳密確認（不一致時 silent fallback しない=明示 error/log）。

### C5 expansion
未探索 legal action を展開。v1 は複雑な pruning を足さない。branching が大きい局面はまず**測定**（legal count /
policy top-k coverage / branching 分布）。測定前に top-k を恣意的に絞らない（config `expansion_top_k` で opt-in、既定=全 legal）。

### C6/C7 rollout / leaf(再利用)
rollout policy = Champion Policy 貪欲（`pipeline._greedy_selection` 相当を read-only 再利用、完全 random にしない）。
leaf = **`leaf_eval.build_evaluator({"kind":"handcrafted"}).evaluate(state, me)`**（Champion と同一 handcrafted）。
**Value leaf は v1 で足さない**（S1 別 ablation）。

### C8 terminal
cg の正式 terminal（prize_out/no_pokemon/deckout/other）を使用。サイド差だけで terminal 価値を置換しない。

### C9 backup / Value domain(Phase 0 で確定)
**Value domain を tree 全体で統一**(一箇所へ集約、独自変換を複数箇所に置かない):
- handcrafted `evaluate()→p∈[0,1]` を **`leaf_to_value(p)=2p−1 ∈[-1,+1]`** で正規化(そのまま符号反転しない)。
- terminal: **win=+1 / loss=−1 / draw=0**(存在時)。同 domain。
- 実装は **fixed root-perspective negamax**: leaf/terminal は常に root 決定者(root_me)視点で v∈[-1,+1] を出す。
  各 tree node は `signed = v if node.to_move==root_me else −v`(= perspective 反転)を backup、Q は node.to_move 視点=selection は最大化。
- **F3 必須条件**: `leaf_to_value(1.0)=+1 / (0.5)=0 / (0.0)=−1`、`flip(+0.6)=−0.6`、`flip(flip(+0.6))=+0.6`、terminal win=+1/loss=−1。

---

## Phase D: instrumentation
1 decision ごと（diagnostic log、通常 run は量制御）: decision_id/turn/phase / legal_actions / root_policy_top1 /
search_selected_action / policy_changed_by_search / iterations / nodes / expanded_nodes / max_depth / mean_depth /
determinizations / unique_information_sets / elapsed_ms / budget_ms / root(visits,Q,prior) / fallback_used / timeout / invalid_action。

## Phase E: compute budget（config 化・hard-code 禁止）
`ismcts.budget_ms`（候補 100/250/500/1000）or `ismcts.iterations`（32/64/128/256）。Kaggle 予算（pipeline time_budget
total_ms 540000 / assumed 400 selects ≈ 1350ms/select 相当）に整合。**budget sweep**（同一局面集合で iter/nodes/depth/
action stability/elapsed を比較）→ 安定に必要な budget を勝率非依存で決める。**turn watchdog**: 1 turn 総 Search 時間監視、
timeout→Champion fallback（illegal/END 暴走禁止）。

## Phase F: correctness tests（cg 使用・最小局面）
F0 leakage（ground-truth opp hidden 非参照）/ F1 determinization consistency（観測と矛盾なし）/ F2 legal action 100%
（is_valid_action）/ F3 backup 符号（手番反転）/ F4 terminal 価値到達 / F5 seed 固定再現 / **F6 no-search equivalence
（ismcts.enabled=false → Candidate==Champion、行動一致）** / F7 per-game reset（match_context + tree/determinization cache
を game 跨ぎで残さない、model weights は immutable cache 再利用可）。

## Phase G: tactical diagnostics（既存 repo の再現 case 優先、捏造しない）
0ダメージ攻撃回避 / 特殊エネ除去後攻撃 / lethal / 次ターン KO 回避 / deckout 近傍 / retreat-switch / resource conservation。
Champion action vs ISMCTS action + visits/Q + simulated consequence を記録（correctness 診断、昇格判定に使わない）。

## Phase H: search ablation（1変数ずつ）
- **S0 = Search-only**（Champion + ISMCTS、Policy/Value/Belief/leaf=Champion と同一）← **第一の正式 Challenger**。
- S1 = Value leaf 改善（S0 が成立し独立評価可能な場合のみ、別 variant）。**abl_5_full は Value 未使用なので S1 は将来の別 ablation**。
- S2 = Belief 改善 = 今回やらない。

## Phase I/J: 開発評価 + 小 H2H screen
**Reference Pool v2 を開発 tuning に使わない**（held-out promotion field）。開発は Champion mirror H2H / curated diagnostics /
search stability / compute benchmark。正式 Gate1 前に S0 vs Champion 50-100 games で crash/illegal/timeout/catastrophic を screen
（小標本で「強い」と断定しない）。正式 Gate1 candidate は **1つに絞る**（選定基準=correctness/runtime/stability、事前定義）。

## Phase K/L: Gate 1 pre-registration + formal Gate 1
- Gate1 = Candidate(ismcts_v1) vs Champion(abl_5_full)の mirror H2H SPRT（同 deck、手番交互）。既存 `measurement/driver.run_sprt_ab`。
- **ISMCTS = large algorithmic change → δ_min=0.05**（reference-pool-field-gate-design.md §3 の規約）。α=0.05, β=0.10, n_max=3000。
- freeze（結果前）: challenger source / search config(budget,iterations,c_puct,prior params,rollout depth) / weights hashes /
  deck hash / seed policy。**結果後に δ_min 等を変えない**。
- Candidate は AgentSpec(agent_fn=ismcts_v1_agent) vs AgentSpec("abl_5_full")。desktop workers=15。
- 記録: games/wins/winrate/Wilson CI/LLR/SPRT state/P0/P1/errors/side split/terminal breakdown/runtime。

## Phase M/N: search behavior + compute analysis
% decisions changed from Policy top1 / changed の win-loss 傾向 / mean iter/depth/nodes / timeout/fallback/invalid 率 / phase 別。
runtime: games/min, decision latency p50/p95/max, iter-nodes/sec, Champion 比 slowdown, worst turn, submission feasibility。

## Phase O: 次判断（Gate1 後）
A PROMOTE→Gate2 v2 / B Search 有望だが leaf 弱→Value ablation / C hidden info ボトルネック→Belief / D branching→
progressive widening/macro-action / E Search 不適→別方向。

---

## 実装増分（推奨・レビュー後）
1. `search/ismcts.py`（tree/selection/expansion/rollout/backup、leakage-safe determinization、instrumentation）。
2. `challengers/ismcts_v1_agent.py`（Champion 経路 read-only 再利用 + ISMCTS 差込、ismcts.enabled=false で Champion 同一）。
3. config `ismcts_v1`（abl_5_full + ismcts ブロック）。
4. correctness tests F0-F7 + tactical diagnostics。
5. budget sweep + small H2H screen → candidate 1 本確定 → Gate1 freeze → desktop formal Gate1。

## 禁止事項（再掲）
Gate2 v2 で tuning / ground-truth hidden leakage / silent illegal（必ず Champion fallback）/ 結果後 Gate1 contract 変更 /
複数 variant 同時正式 Gate。production/cg/frozen 評価資産変更なし・git add/commit/push なし。
