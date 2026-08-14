# Beyond Behavior Cloning: 「人間と同じ行動」から「勝つ行動」への設計書

作成日: 2026-07-23
種別: **設計書(方向性の評価と推奨。関数シグネチャ・完了条件付きの実装計画は別ファイルに分ける
—— [[feedback_design_vs_implementation_plan]])**
前提: `docs/plans/policy-capacity/model-capacity-ablation-implementation-plan.md`(容量ablation)、
`results/2026-07-23_feature_addition_negative_result.md`(特徴追加の negative result)
ステータス: **設計のみ。production 未変更。実装は本書で方向を決めてから別プランに落とす。**

---

## 0. なぜこのフェーズか(問題定義)

現行 PolicyModel が最適化しているのは

```
P(human action | state)   ← 上位パイロットの選択を argmax 模倣(listwise softmax CE)
```

であって

```
P(win | state, action)    ← その行動が勝ちに繋がるか
```

**ではない**。この乖離が、これまで繰り返した

```
offline 模倣 Top-1 が微増 → 実戦勝率はフラット/悪化
```

の原因である可能性が高い。実証済みの negative result:

| 施策 | offline | 実戦(ミラー) | 結論 |
|---|---|---|---|
| Tier1 identity 特徴 | 一部微増 | 43–44% | 転移せず |
| Tier3 consequence 特徴(B/C/D/B+C) | +0.1〜0.2pt or 悪化 | 35–55%(不安定) | 転移せず |
| 容量増 M64/M128 | +0.5pt 単調・過学習なし | 49.0% / 49.3%(CI 50%跨ぎ) | 転移せず |

→ **特徴追加でも容量増でも「模倣精度の微増」は実戦に転移しない。** 入力・容量ではなく
**目的関数そのもの**(何を最適化しているか)を変える段階に来た、というのが本フェーズの仮説。

**本書のゴール**: 目的を「模倣一致率」から「実戦勝率」へ寄せる方法を、既存資産で実現できる形に
評価し、**最小コスト・最大可逆性・最小転移リスク**の順で着手する順序を決めること。

---

## 1. 確認済みの再利用可能資産(コードで確認・推測しない)

| 資産 | 実体 | 状態・実測 |
|---|---|---|
| **ValueModel**(state→勝率) | `ptcg_ai/learning/value_model.py` + `value_weights.json` | 較正済 **test AUC 0.746** / logloss 0.587(baseline prize_diff 0.682, matchup 0.566 を上回る)。pure-Python、ターン帯温度較正。state 166次元は PolicyModel と共有。**そこそこ精度**の評価器。 |
| **探索ロールアウト** | `cg.api.search_begin` / `search_step` | 「obs + 行動 → 結果 State」を取得可能。`search/attack_plan.py`・`search/pimc.py` が実運用中。**ただし attack_plan の評価は手製ヒューリスティック(damage/KO/prize)で ValueModel は未接続。** |
| **PolicyModel**(state→P(human action)) | `ptcg_ai/learning/policy_model.py` + `policy_weights.json` | 現行 production。層を動的読込するため容量非依存。option 集合に prior を与える器として流用可。 |
| **outcome ラベル**(勝敗) | `kaggle_replays/value_net/` の value_positions(win=1/loss=0、episode_id・player_index・step_index 付き) | value_net の教師。policy の decision point への join は build_features の小拡張が必要(下記 §3-C)。 |
| **hidden info 決定化** | `hidden_information/` + `search_state_stub` | search_begin に必要な相手非公開情報の推定/ダミー。attack_plan/pimc/lethal が既に factory 経由で使用。 |
| **Tier3 consequence 特徴** | `board_evaluation/consequence.py`(既定OFF) | option→consequence。将来 action evaluation の素性として再利用可。 |
| **3段階評価ゲート** | `league/_diag_*head_to_head.py` + Stage1/2/3 手順 | 容量ablfor確立。**offline だけで採らない / CI下限>50% で採用**を踏襲。 |

**含意:** 「Policy で候補を絞る」「行動を1手進めて結果盤面を作る」「その盤面を勝率評価する」
「勝敗ラベルで学習を重み付ける」——**必要部品はすべて既にリポジトリにある**。新規に足りないのは
「それらの接続」と「目的関数の変更」だけ。

---

## 2. 候補の評価(A / B / C / D)

評価軸: **必要資産(有/無)・実装コスト・推論レイテンシ・可逆性・転移リスク・既知の弱点**。

### A. Policy shortlist → Value rollout(決定時に価値で選ぶ)

やること: 各意思決定点で PolicyModel が上位 k 個の候補に絞り、各候補を `search_step` で1手
(必要なら数手)進めた**結果盤面を ValueModel で勝率評価**し、最大の候補を選ぶ。

- 必要資産: **すべて有**(PolicyModel の shortlist、search rollout、ValueModel)。attack_plan の
  「行動→結果盤面→評価」を **attack 限定+手製ヒューリスティックから全 option+ValueModel へ一般化**
  したものに等しい。
- 実装コスト: 中(rollout と ValueModel を ml_policy/search 内で接続。**学習不要**)。
- 推論レイテンシ: **増える**。候補 k 件 × rollout(search_step)× ValueModel forward。attack_plan の
  予算実績(各100ms、Kaggle 600s/agent)内に収まる設計は可能だが、k と深さで制御が要る。
- 可逆性: **高**(config フラグで opt-in、既定 OFF。weights も探索ロジックも production 据え置き)。
- 転移リスク: 中。**ValueModel AUC 0.746 のノイズがそのまま選択の質に乗る**([[feedback_conservative_confidence]]:
  価値は慎重側に。誤った自信で Policy の良い手を上書きしない設計が要る)。
- 既知の弱点: 決定化(hidden info)1本での評価は分散大 → pimc 的に複数決定化の平均を取ると更にコスト増。
  「Policy が十分よい手を、ノイジーな Value で上書きして悪化」する典型失敗に注意。

### B. Policy prior + search(Policy を最終決定器でなく探索の prior に)

やること: PolicyModel を prior、ValueModel を leaf 評価にした探索(PUCT/MCTS 系)で手を選ぶ。

- 必要資産: PolicyModel/ValueModel は有だが、**prior 誘導つき木探索の枠組みは新規**(pimc は
  決定化サンプリングであって policy-prior guided ではない)。
- 実装コスト: **高**(探索器の新規実装 + 予算管理)。
- 推論レイテンシ: 高(探索ノード数に比例)。
- 可逆性: 高(opt-in)。転移リスク: 中〜高(A と同じ Value ノイズ + 探索の分散)。
- 位置づけ: **A の上位互換だが投資が大きい。A が有望と分かってから。**

### C. Advantage / outcome-aware learning(学習時に目的を勝率へ寄せる)

やること: 上位パイロットの行動模倣に、**最終勝敗 / ValueModel の advantage** を教師信号として
組み込み、`P(win)` に相関する行動へ重みを寄せた PolicyModel を学習する。具体案(A/B する下位案):

- **C1 outcome-weighted / filtered BC**: 各 decision point の listwise CE を「その試合を勝ったか」で
  重み付け(または勝ち試合のみ学習)。既存の sample weight(構成C)に **勝敗係数**を掛けるだけ。
- **C2 advantage-weighted regression(AWR系)**: 各 decision の重みを ValueModel 由来の advantage
  `A = V(s') - V(s)`(または最終 outcome とのTD)から `exp(A/β)` で決める。良い手ほど強く模倣。

- 必要資産: **ほぼ有**。outcome ラベルは value_positions にあり、policy decision point への join は
  build_features に **episode_id/player_index を出させて勝敗を突き合わせる小拡張**(§3-C)。
  ValueModel は既存。**学習コード(train.py)の loss 重み変更**が主。
- 実装コスト: **低〜中**(訓練時のみ。推論・探索・レイテンシは一切増えない)。
- 推論レイテンシ: **増えない**(現行 PolicyModel 推論のまま。weights が変わるだけ)。
- 可逆性: **最高**(weights ファイルの差し替えのみ。容量ablと同じ運用で production 据え置き)。
- 転移リスク: 中。**目的関数を直接勝率側へ動かす**ので、これまでの「入力/容量だけ触る」施策より
  筋が良い可能性がある一方、勝敗ラベルはノイズ源(1試合の勝敗は個々の手の良否と一致しない)。
  filtered/weighted の強度 β の取り方が肝。
- 既知の弱点: 勝敗は疎で遅延した信号。advantage を ValueModel から作ると Value ノイズが混入。

### D. Self-play / offline RL(模倣を初期方策に、勝率目的で改善)

やること: 模倣 Policy を初期化に、ログ(または self-play)で勝率目的の RL(offline: CQL/IQL 等)。

- 必要資産: 環境ロールアウト(cg エンジン)は有。**offline RL の学習器は新規・大規模**。
- 実装コスト: **最高**。可逆性: 高(別weights)。転移リスク: 高(分布ずれ・過大評価)。
- 位置づけ: **最も重く最もリスキー。A/C で価値/探索方向の有望性が確認できてから。今は着手しない。**

---

## 3. 推奨シーケンス

> **原則: 学習時変更(推論コスト0・可逆性最高)から。決定時変更(レイテンシ増・Value依存)は次。
> 探索器新規・RL は最後。** これまでの「安いものから、offlineで採らず実戦で決める」を踏襲。

### Phase 1(最初に着手): C —— outcome/advantage-aware imitation

理由:
- **推論を一切変えない**(現行 PolicyModel 推論・探索・レイテンシ・提出サイズ不変、weights のみ)。
- **可逆性が容量ablと同じ最高水準**(weights 差し替えのみ、production 据え置き)。
- **目的関数を初めて `P(win)` 側へ動かす**——本フェーズ仮説の core を最小コストで検証できる。
- 必要な新規実装は「build_features に勝敗 join(§3-C 小拡張)」+「train.py の loss 重み」だけ。

進め方(下位案を独立に比較、hidden 以外を固定した容量ablと同じ clean 精神):
1. **C1 先行**(最小): sample weight に勝敗係数を掛ける/勝ち試合フィルタ。ValueModel 不要で最速。
2. **C2**(advantage): ValueModel の `A=V(s')-V(s)` で `exp(A/β)` 重み。β を数点スイープ。
- control は **M32(=production、features_configC 由来。容量ablで byte 一致を確認済み)**。

### Phase 2: A —— Policy shortlist + Value rollout(決定時)

C が有望 or 頭打ちなら、推論時の価値利用へ。attack_plan の rollout を top-k policy 候補へ一般化し
ValueModel で評価。**既定 OFF の config opt-in**。レイテンシ実測を必ず取り、Kaggle 予算比を出す。
[[feedback_conservative_confidence]] に従い、Value が僅差のときは Policy の第1候補を尊重する
(誤上書き防止)ガードを入れる。

### Deferred: B(prior+search)・D(offline RL)

A で「価値で選ぶと勝てる」兆候、または C で「勝率目的の学習が効く」兆候が出てから投資する。
現時点では着手しない。

---

## 4. 評価方法(全案共通、これまでのゲートを踏襲)

**offline 改善だけで採用しない。** 容量abl・Tier3 と同じ3段階 + 採用条件:

- **Stage 1 offline**: 破綻/過学習の足切りのみ(C は模倣一致率が下がっても構わない——勝率が目的)。
- **Stage 2 診断**: 既知局面(ATTACK一致率・山札切れ関連)で意思決定が壊れていないか。
- **Stage 3 head-to-head**: **M32 control vs 候補**、`ml_lethal_attackplan_v0only`、先後半々、
  300試合スクリーニング → 段階判定(45%未満打切/55%以上増試合)。
- **採用条件: 95% CI 下限 > 50%。** 満たさなければ production 据え置き。
- ハーネスは `league/_diag_tier1abc_head_to_head.py`(`--baseline/--candidate-weights` 汎用)を流用。
- **対フィールド検証**(Kaggle 提出)は必要に応じ。ミラー≠フィールドの分散に注意
  (M128 提出でも 27試合 55.6% CI[37,72] と判別不能だった)。

### C 専用の注意(勝敗ラベル join、§3-C)

- policy `build_features.py` に **episode_id / player_index / step_index を出力**させ、
  value_positions(または replay の `rewards`)から **その decision point の player の最終勝敗**を突き合わせる。
- split(train/val/test)は現行の episode 単位分割を維持(**同一試合が train と test に跨がない**こと)。
- 勝敗係数/advantage は [[feedback_conservative_confidence]] に従い、**過度に強い重みで少数の勝ち試合へ
  過適合しない**よう強度をスイープして最も穏当な設定から評価する。

---

## 5. 非目的・境界

- **production 据え置き**: `policy_weights.json` / `value_weights.json` / 既定 config / 推論経路は、
  各案が採用条件を満たすまで変更しない。実験は別 weights/別 config/別ブランチ。
- **担当領域の尊重**([[feedback_respect_ownership_boundaries]]): A/B の rollout・評価は
  `ml_policy/` `search/` `learning/` 内で完結させる。`rule_based/` `action_selection/` の
  ゲーム判断ロジックには手を入れない(必要になれば独断せず AskUserQuestion)。
- **特徴追加・容量増には戻らない**(本フェーズの前提)。C は入力特徴を増やさず目的関数だけを変える。
- **実装計画は別ファイル**([[feedback_design_vs_implementation_plan]]): 本書で Phase 1(C)を選んだら、
  関数シグネチャ・build_features 拡張の具体・loss 定義・完了条件を書いた実装計画を別途作る。

---

## 6. 次のアクション(決定ポイント)

推奨は **Phase 1 = C(まず C1 outcome-weighted/filtered BC、次に C2 advantage-weighted)**。
理由は §3 の通り「推論不変・可逆性最高・目的関数を初めて勝率側へ動かす・必要部品が既存」。

ユーザー判断を仰ぐ点:
1. Phase 1 を **C で確定**してよいか(推奨)。それとも **A(決定時 Value rollout)を先に**試すか。
2. C なら **C1 先行**(ValueModel 不要・最速)で始めてよいか。
3. 確定後、本書を基に **C の実装計画書**(build_features 勝敗 join / train.py loss / 評価手順 /
   完了条件)を作成する。
