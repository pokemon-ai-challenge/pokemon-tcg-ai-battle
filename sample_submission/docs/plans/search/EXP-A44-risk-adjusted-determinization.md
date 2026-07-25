# EXP-A44 リスク調整Determinization（CVaR）要件定義

> `test_plan/ptcg_ai_methods_roadmap.md` §3 A4 / バックログ P0「EXP-A44 CVaR」の要件定義。
> 実装前の設計文書であり、コードはまだ書かない。フォーマットは `test_plan/ptcg_ai_experiment_template.md` に準拠。

---

## 0. 実験メタデータ

| 項目 | 値 |
| --- | --- |
| Experiment ID | `EXP-A44`（前段の `EXP-A40`〜`A43` を同一実装のvariantとして内包） |
| 実験名 | リスク調整Determinizationによる候補手の頑健評価 |
| ステータス | `Planned` |
| 優先度 | `P0` |
| 研究トラック | `Belief` / `Search` |
| Branch | `feature/rule-based-fix`（起点） |
| 対応仮説 | H4（不完全情報とリスク）。副次的に H5（Value）の実戦価値検証 |
| 関連実験 | EXP-A32（階層型Best-first, 別ブランチで実施済）/ EXP-B10-B11（アーキタイプ推定, 実装済）/ EXP-402（Single-task Value, 学習済だが未配線） |
| 設定ファイル | `configs/rule_risk.json`（新規） |
| Feature Flag | `risk_determinization.enabled`（既定 `false`） |

---

## 1. Research Question

> **相手の隠し情報を1つのダミー/一点推定で固定するのではなく、複数の決定化サンプルにわたって候補手を評価し、平均ではなくCVaR（下側裾）で集約すると、優勢局面の事故負けを減らして勝率を改善できるか？**

---

## 2. Motivation

### 2.1 ポケカ固有の問題

- 相手の手札・山札・サイドは非公開で、同じ盤面でも「相手が特定カードを持っているか」で最善手が反転する（例: ベンチ狙撃、特殊エネ破壊、非ex突破役の有無）。
- 期待値だけで手を選ぶと、「99%勝てるが1%で即負け」の手と「95%で確実に有利」の手が同順位になる。サイドレースは取り返しがつかないため、優勢局面ではこの1%が支配的。
- 1試合10分・1判断3〜5秒という時間制約があり、全隠し情報を展開する探索は不可能。少数サンプルでの近似が必須。

### 2.2 現行方式の弱点（このブランチの実測状態）

- `action_selection/selector.py` は隠し情報として `hidden_information/search_state_stub.build_dummy_search_state()`（**ダミー**: 相手の山札・手札は基本エネで埋める）しか使っていない。
- したがって「相手が何を持っているか」は現在の意思決定に**一切反映されていない**（= EXP-A40 Beliefなし の状態）。
- 一方で以下は実装済みかつ**未配線**:
  - `hidden_information/own_hidden_state.py` … 自分の山札/サイドの決定化サンプラ（`sample()`）
  - `hidden_information/opponent_hidden_state.py` … アーキタイプ事後×多変量超幾何による相手の山札/手札/サイドのサンプラ（`sample()`、`archetype_card_pool.json` 同梱済）
  - `hidden_information/search_adapter.to_search_begin_kwargs()` … 上記2つを `search_begin()` 引数へ変換
  - `learning/encoder.encode_state_from_state()` + `learning/value_model.ValueModel.predict_win_prob_from_state()` … 166次元→勝率（`value_weights.json` 同梱、外部ライブラリ不要）
  - `search/lethal_simple.py` … `cg.api.search_begin()`/`search_step()` の使用実績とタイムアウト/検証パターン
- **未実装なのは「複数決定化上で候補手を評価してリスク集約する層」だけ**であり、A4は新規部品が最も少なく、既存資産の投資回収効果が最も大きい。

### 2.3 A1/A2/A3 ではなくA4を選んだ理由

| 候補 | 判断 |
| --- | --- |
| A1 Opening Book | `kaggle_replays/replays` が空。上位ログ再取得が前提でリードタイム大 |
| A2 Pairwise Ranking | 同上（`kaggle_replays/training_data` も空）。学習データ調達が律速 |
| A3 階層型Best-first | 別ブランチ `experiment/fukuda` で実装・A/B済（57.5%）。ここでの再実装は重複 |
| **A4 リスク調整Determinization** | **必要部品が全て同梱済で未配線。推論のみ（学習不要）。Feature Flagで安全に切れる** |

---

## 3. Hypothesis

### 3.1 主仮説

> 候補手を N 個の決定化サンプル上で評価し、下側分位（CVaR）で集約すると、平均集約や一点推定より「優勢局面からの勝ち切り率」が上がり、総合勝率も退行しない。

### 3.2 反証可能な予測

- P1: 決定化を導入（A42 Mean）した時点で、Beliefなし（A40）に対して勝率が **50%を有意に下回らない**（機能確認）。
- P2: CVaR集約（A44）は Mean集約（A42）に対し、**サイド差+2以上の優勢局面からの勝ち切り率を+3pt以上**改善する。
- P3: A44 の総合勝率は現行championに対し **50%を下回らない**（できれば+2pt以上）。
- P4: P95思考時間が **1判断あたり1,500ms以内**、ゲーム全体の思考時間が10分制限の50%以内。
- P5: 不正Action 0件・例外0件・ハング0件。

### 3.3 帰無仮説

> リスク集約方式（Mean / Mean−βStd / CVaR）の違いは、実戦上意味のある勝率差を生まない。

### 3.4 適用範囲

- 対象フェーズ: **中盤〜終盤**（序盤は情報が無く事後分布がほぼ事前分布のため効果が薄い。ターン閾値で制御）
- 対象select: **MAIN のみ**（`SelectType.MAIN`）。他のselect種別は従来どおりrouterへ
- 適用しない局面: 確定リーサル成立時（`lethal_simple` が先に返す）/ 候補手が1つ / 残り時間逼迫時 / `OpponentHiddenState.is_ready == False`

---

## 4. Compared Methods

### 4.1 Control（ベースライン）

- **名称**: `rule_lethal`（現行champion）
- **概要**: `lethal_simple` → `router.route()`（ルールベース）。隠し情報はダミー
- **固定する設定**: デッキ、`weights.py` のルールスコア、`lethal_search` 設定一式

### 4.2 Treatment（新手法）

- **名称**: `rule_risk`
- **概要**: MAIN選択時のみ、ルールベースの上位K候補を N決定化上で評価し、リスク集約スコアで並べ替える層を追加
- **差分**: 探索ステージ列に `risk_determinization` を1段追加するのみ。ルールスコア・ハンドラ群は不変

### 4.3 Variants（roadmapのEXP対応）

| Variant | 集約方式 | roadmap ID |
| --- | --- | --- |
| V0 | Beliefなし（＝Control） | EXP-A40 |
| V1 | 最頻アーキタイプへの一点推定（N=1、決定化を最有力リストで固定） | EXP-A41 |
| V2 | N決定化のMean | EXP-A42 |
| V3 | Mean − β·Std（β = 0.5 / 1.0） | EXP-A43 |
| V4 | CVaR_α（α = 0.2 / 0.3） | **EXP-A44（本命）** |
| V5 | 局面適応リスク（優勢→CVaR / 互角→Mean+CVaR混合 / 劣勢→上位分位） | EXP-A45 |

### 4.4 変更してはいけない変数

デッキ（`deck.csv`）/ 対戦相手 / seed集合 / 制限時間 / `lethal_search` 設定 / `weights.py` / `priorities/*.py` のスコア / `value_weights.json` / `archetype_card_pool.json`。

---

## 5. 実装要件

### 5.1 新規モジュール `ptcg_ai/search/risk_determinization.py`

チーム共通の探索インターフェース（`lethal_simple.search` と同一シグネチャ）を実装する。

```python
def search(state, legal_actions, context) -> list[int] | None
```

`context` キー（`selector.py` が組み立てる）:

| キー | 必須 | 内容 |
| --- | --- | --- |
| `observation` | 必須 | エージェントに渡された `Observation` |
| `config` | 任意 | config の `risk_determinization` セクション（欠損キーは `DEFAULTS`） |
| `hidden_state_factory` | 必須 | 引数なしで `search_begin()` 用 dict を返す callable。**呼ぶたびに別サンプル**（決定化） |
| `candidate_provider` | 必須 | 引数なしで候補手 `list[tuple[list[int], float]]`（action, rule_score）を返す callable |
| `deadline_ms` | 任意 | この探索が使ってよい壁時計予算 |

戻り値: 採用する option index list、または `None`（＝呼び出し側は従来どおり router にフォールバック）。

### 5.2 処理フロー

```text
1. ゲート判定（5.3）を通らなければ即 None
2. candidate_provider() で候補 K 件を取得（ルールスコア上位、K = top_k）
3. for n in 1..N（determinizations）:
       kwargs = hidden_state_factory()          # 相手の隠し情報を1通りに決定化
       for each candidate action a_k:
           search_begin(state, **kwargs)
           search_step(a_k)                     # 候補手を1手適用
           v[k][n] = ValueModel.predict_win_prob_from_state(結果state)
       （予算超過を毎ループ先頭でチェック。超過したら以降を打ち切り、
         集計済みサンプルだけで4へ。1サンプルも取れていなければ None）
4. 各候補を集約: score[k] = aggregate(v[k][*], mode)
5. 最終スコア = (1 - w) * normalized_rule_score[k] + w * score[k]
   （w = risk_weight、既定 0.25）
6. argmax の候補を返す。ただしトップがルールベース第1候補と同じなら None を返してもよい
   （呼び出し側の挙動は同一。ログ上は「探索が上書きした/しなかった」を記録する）
```

**重要（過去の教訓の反映）**: 混合比 `risk_weight` は**必ず従（0.25程度）から始める**。別ブランチの2ターン読み導入時、混合比0.5はA/B 46.5%で失敗し0.25で54%に転じた実績があるため、初版で0.5以上にしない。

### 5.3 ゲート条件（すべて満たすときのみ発火）

- `config.enabled == True`
- `obs.select.type == SelectType.MAIN`
- `obs.current is not None`
- 候補数 >= 2
- ターン番号 >= `min_turn`（既定 3）
- `OpponentHiddenState.is_ready == True` かつ `ValueModel.is_ready == True`
- 残りゲーム時間が `min_remaining_game_ms`（既定 120,000ms）以上
- 直前に `lethal_simple` が None を返している（selectorのステージ順で自然に担保）

### 5.4 リスク集約関数

`aggregate(values: list[float], mode: str, params) -> float`

| mode | 定義 |
| --- | --- |
| `mean` | 算術平均 |
| `mean_std` | `mean - beta * pstdev` |
| `cvar` | 昇順ソートし下位 `ceil(alpha * N)` 件（最低1件）の平均 |
| `adaptive` | サイド差で分岐: 優勢(自分の残サイド < 相手 -1) → `cvar` / 互角 → `0.5*mean + 0.5*cvar` / 劣勢 → 上位 `alpha` 分位の平均 |

純Python（`statistics` / `math` のみ）で実装し、単体テストで各modeの数値を固定値検証する。

### 5.5 候補手の生成

- `rule_based/main_turn_parts/proposals.collect_proposals(obs)` を利用する（既存API、`list[ActionProposal]`）。
- `proposals._total_score` 相当の合計スコア順に上位 `top_k`（既定4）を取る。
- **setup-first の不変条件を壊さないこと**: `SETUP_BEFORE_ATTACK` により「自己枯渇する下準備が残る間は attack より先」というルールが `decide()` にある。候補集合を作る際も同じフィルタを通した後の集合から上位Kを取る（＝リスク層は「setup-firstが許した候補の中での並べ替え」に限定する）。この不変条件は回帰テストで固定する。
- 候補が `decide()` の結果1件しか無い場合はゲート不成立として `None`。

### 5.6 決定化サンプラの配線

- `selector.py` に `OwnHiddenState` / `OpponentHiddenState` のインスタンスを**試合単位で1つ**保持する（`opponent_tracker` と同じライフサイクル）。
- 毎 select で `own_state.update(state, select)` / `opponent_state.update(...)` を呼び、`hidden_state_factory = lambda: search_adapter.to_search_begin_kwargs(own_state, opponent_state, obs, rng)` を渡す。
- `rng` は seed を config から受け取り、実験の再現性を確保する。
- **既存の `search_state_stub` は削除しない**。`OpponentHiddenState.is_ready == False` 時と `lethal_simple` 用のフォールバックとして残す。
- 例外は全て握り潰して `None`（＝ルールベースへ）。合法性は `selector.is_valid_action()` で最終担保。

### 5.7 設定 `configs/rule_risk.json`

```json
{
  "name": "rule_risk",
  "lethal_search": { "...": "rule_lethal.json と同一（変更禁止）" },
  "risk_determinization": {
    "enabled": true,
    "module": "risk_determinization",
    "mode": "cvar",
    "alpha": 0.3,
    "beta": 1.0,
    "determinizations": 6,
    "top_k": 4,
    "risk_weight": 0.25,
    "min_turn": 3,
    "time_limit_ms": 1200,
    "min_remaining_game_ms": 120000,
    "max_evaluations": 32,
    "rng_seed": null
  }
}
```

環境変数によるロールバック: `PTCG_DISABLE_RISK_DET=1` で強制無効化（過去の `PTCG_DISABLE_TURN_SIM` / `PTCG_SETUP_BEFORE_ATTACK` と同じ流儀）。

### 5.8 selector への組み込み

`_SEARCH_MODULES` に `risk_determinization` を登録し、ステージ列を
`lethal_simple → risk_determinization → router.route → fallback.safe_choice`
に一般化する。各ステージは「返せなければ次へ」の契約を守る。

### 5.9 計測・ログ

`get_stats()` / `reset_stats()` を `lethal_simple` と同じ形で提供し、少なくとも以下を集計する。

- 発火回数 / ゲート不成立の内訳（理由別カウント）
- ルール第1候補を**上書きした**回数と、その時のスコア差
- 決定化サンプル数の実績（予算打ち切りで減った回数）
- 1判断あたりの経過時間（P50/P95/P99）
- 評価回数（K×N）とタイムアウト率

---

## 6. Metrics

### 6.1 Primary

- **主要指標**: 対Control勝率（ペアA/B、先後交互）
- **改善と判定する最小差**: 400戦で片側 z >= 1.65（p < 0.05）

### 6.2 Strength

勝率／信頼区間／先攻後攻別勝率／**優勢局面（サイド差+2以上）からの勝ち切り率**（P2の主指標）／劣勢からの逆転率

### 6.3 Stability

不正Action率0／例外率0／P50・P95・P99思考時間／ゲーム総思考時間／fallback発動率／ハング0

### 6.4 Model-specific

決定化サンプル数の実績分布／リスク集約スコアと最終勝敗の相関／ルール上書き率と、上書きした局面の勝率

---

## 7. 評価計画と既知の測定限界

### 7.1 ハーネス

このブランチの `tests/local_sim/test_local_game_advanced.py` は `--opponent {self,random}` のみで、メタデッキ対面ベンチが無い。よって以下の順で測る。

- **Stage 1（機能確認・20〜50戦）**: `--opponent random` と self。不正Action/例外/ハング/時間の確認のみ。勝率は見ない。
- **Stage 2（小規模選別・200戦）**: **ペアA/B（同一エージェント内で `yourIndex` ごとにフラグを切り替え、先後交互）**。この repo で確立済みの方式（setup-first・retreatマージンで使用）。V2/V3/V4 を順に Control と比較。
- **Stage 3（昇格試験・400戦）**: 勝ち残ったvariantのみ。V5（適応型）はここで初めて追加。

### 7.2 【最重要】測定限界の事前宣言

**ミラー自己対戦では本手法の価値は構造的に過小評価される。** 相手が自分と同じデッキなら隠し情報の事後分布はほぼ正解に張り付き、「相手のリストが分からないことによる事故」自体が起きない。過去に keep-draw 方針と retreat トリガー再設計の2件が、同じ理由でミラー戦では有意差ゼロ（50.0%）に潰れている。

したがって本実験は以下を必須とする。

- 総合勝率が50%近傍でも、**P2（優勢局面からの勝ち切り率）が改善していれば Partial Adoption を検討する**。総合勝率だけで棄却しない。
- 可能なら `experiment/pimc-hidden-info-integration` 系にあるメタデッキ対面ベンチ（`--opponent-deck <archetype>`）へ移植して再測定する。これを Stage 3 の前提条件とするか、限界を明記して Research Only 判定にするかは、Stage 2 の結果を見て決める。
- 最低限の代替策として、**`random_agent` 相手＋相手デッキを別アーキタイプに差し替えた非対称対面**を1つ用意することを Phase 0 のタスクに含める。

---

## 8. Preconditions / Safety Checks

- [ ] Controlを凍結（`configs/rule_lethal.json` と `weights.py` に変更を入れない）
- [ ] 変更点が「探索ステージ1段の追加」に分離されている
- [ ] 全Actionが `selector.is_valid_action()` を通る
- [ ] 例外時に必ず `router.route()` → `fallback.safe_choice()` へ落ちる
- [ ] `time_limit_ms` の超過チェックが評価ループの毎周先頭にある
- [ ] seed / Episode ID / config Hash を保存
- [ ] Feature Flag と環境変数の両方で無効化できる
- [ ] `pytest` 全件パス（現状89件を下回らない）

---

## 9. 実装フェーズ

| Phase | 内容 | 完了条件 |
| --- | --- | --- |
| **P0 計測基盤** | ペアA/Bスクリプト（フラグ頭割り・先後交互・優勢局面からの勝ち切り率の集計）＋非対称対面の用意 | Controlを自分自身と回して50%±ノイズ、集計項目が全部出る |
| **P1 集約関数** | `aggregate()` の4mode + 単体テスト | 固定値テストが通る |
| **P2 決定化配線** | selectorに `OwnHiddenState`/`OpponentHiddenState` を持たせ、`hidden_state_factory` を実サンプラへ。**この時点では意思決定を変えない**（stubとの差し替えのみ、`lethal_simple` に供給） | 20戦で例外0・lethal成功率が退行しない |
| **P3 リスク層** | `risk_determinization.py` 本体（V2 Mean）＋ selector 組込み＋ stats | Stage 1 通過（不正0・P95内） |
| **P4 variant比較** | V1/V3/V4 を200戦ずつ | 最良variantの決定 |
| **P5 昇格試験** | 最良variant + V5 を400戦 | §12 の採否判断 |

各Phaseは独立コミット。P2で退行が出たらP3へ進まない。

---

## 10. リスクと対策

| リスク | 対策 |
| --- | --- |
| `search_begin` の再実行コストが高く時間予算を超える | `max_evaluations` で K×N を上限管理。予算超過時は集計済みサンプルで打ち切り、0件なら `None` |
| `ValueModel` がこのデッキ・この局面分布で校正されておらず、評価が雑音 | P3の前に「勝敗ラベルとの相関」をオフラインで確認。相関が無ければ `risk_weight` を下げるかValue側の再学習を別実験に切る |
| 相手アーキタイプ事後が外れると決定化が全部間違う | それ自体がCVaRで守られる設計（外れサンプルが下側裾に入る）。`is_ready == False` 時は発火しない |
| リスク層がsetup-firstやドロー特性ガードの既存改善を打ち消す | 候補集合を `collect_proposals` + setup-firstフィルタ後に限定（§5.5）。回帰テストで固定 |
| ミラー戦で差が出ず判断不能 | §7.2 の事前宣言どおり、優勢局面からの勝ち切り率を副次主指標に据える |

---

## 11. 回帰テスト計画

新規に追加する単体テスト（`tests/unit/`）:

- `test_risk_aggregation.py` … `aggregate()` 4modeの数値検証、N=1・全値同一・α境界のエッジ
- `test_risk_determinization_gate.py` … ゲート条件の各項目でON/OFFが期待どおり
- `test_risk_determinization_fallback.py` … `hidden_state_factory` が例外/None、Value未ロード、予算0 のとき `None` を返す
- `test_risk_candidates_respect_setup_first.py` … 下準備が残る局面で attack が候補集合に混じらない（既存不変条件の固定）

既存テストは1件も変更しない。変更が必要になった時点で「Controlを壊している」と判断する。

---

## 12. 採否判断（事前に決めておく）

- **Adopt**: 400戦で対Control 有意勝ち越し（p<0.05）かつ P95時間・不正0を満たす
- **Partial Adoption**: 総合は五分でも、優勢局面からの勝ち切り率が+3pt以上 → ゲートを「優勢局面のみ」に絞って採用
- **Re-test**: メタデッキ対面ベンチのあるブランチでの再測定を条件に保留
- **Reject**: 総合で負け越し、または時間基準を満たせない
- いずれの場合も、Feature Flag は残したまま既定 `false` に戻せる状態を維持する

---

## 13. 次の実験候補（本実験の結果に依存）

- 採用時: EXP-A45（局面適応リスク）→ EXP-B12（軽量粒子フィルタ、決定化の質を上げる）→ EXP-B13（粒子フィルタ + CVaR探索）
- 棄却時: 原因が Value 側なら EXP-B23（Multi-task Value）、Belief 側なら EXP-B11 の精度改善を先行
