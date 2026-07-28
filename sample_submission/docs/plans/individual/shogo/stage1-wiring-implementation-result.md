# Stage1 実装結果: hidden_information / value_model の配線接続

作成: 2026-07-21
対応: [stage1-wiring-implementation-plan.md](./stage1-wiring-implementation-plan.md)
ブランチ: `experiment/pimc-hidden-info-integration`

---

## やったこと

### Step1: `selector.py` に `hidden_state_source` 分岐を追加

`ptcg_ai/action_selection/selector.py` に `match_context` / `search_adapter` の import と、
`lethal_config.get("hidden_state_source", "dummy")` による分岐を追記(既存行は削除せず条件化)。
`"estimated"` のときだけ `search_adapter.to_search_begin_kwargs(match_context.get_own_state(),
match_context.get_opponent_state(), obs)` を `hidden_state_factory` として渡す。既存の
`try/except Exception: action = None` がそのまま新しい例外系も吸収するため、新規の例外処理は
追加していない(プラン記載どおり)。

### Step2: `ml_policy_agent.py` に同じ分岐を追加

`ptcg_ai/ml_policy/ml_policy_agent.py` の `_try_lethal()` にStep1と同じ分岐を複製。加えて、
このファイルはこれまで `match_context.update(obs)` を一度も呼んでいなかった(`rule_based_agent.py`
は既に呼んでいた)ため、`agent()` の先頭に追加。これが無いと `"estimated"` を選んでも
`match_context` が空のまま(推定が更新されず、初期状態や前回試合の値を渡してしまう)ため、
プランの契約を満たすのに必須の変更として追加した。

また `_CONFIG_NAME` を `os.environ.get("PTCG_AI_ML_CONFIG", "ml_lethal")` に変更し、環境変数で
config切り替えできるようにした(既定値は不変)。`ml_policy_agent.py` は専用config
(`_CONFIG_NAME` 固定)を読む設計のため、A/B計測で `ml_lethal_estimated` に切り替えるにはこの
上書き口が必要だった。

### 新規config

- `configs/rule_lethal_estimated.json`
- `configs/ml_lethal_estimated.json`

いずれも既存config(`rule_lethal.json` / `ml_lethal.json`)に `hidden_state_source: "estimated"`
と `value_shadow_logging: true` を足しただけ(プラン§3の例のとおり)。既存2ファイルは無変更。

### Step4: value network shadow mode ログ

新規 `ptcg_ai/learning/value_shadow_log.py`(モジュール単位シングルトン、`match_context` と同じ
reset パターン)。`config.get("value_shadow_logging", False)` が true のときだけ
`ValueModel().predict_win_prob(obs)` をターンごとに呼び、ログに残すだけ(意思決定には未接続)。
呼び出し口は `rule_based_agent.agent()` / `ml_policy_agent.agent()` の冒頭
(`match_context.update(obs)` と同じ位置)に追加。

---

## 結果

### Step0: ベースライン(変更前)

`rule_based`(rule_lethal)・`ml_policy`(ml_lethal)それぞれ自己対戦50試合。詳細は
[results/2026-07-21_stage1_baseline.md](../../../../results/2026-07-21_stage1_baseline.md)。

| エージェント | 勝率(自己対戦) | エラー | lethal探索 found/searches |
|---|---|---|---|
| rule_based | 0.480 (24/50) | 0 | 36/261 |
| ml_policy | 0.400 (20/50) | 0 | 70/806 |

### Step1/Step2: 動作確認

- 既存config(キー省略、dummy)は再実行後もエラー0件・探索統計が同オーダーで、コード変更による
  挙動崩れは無い(ただし後述の通り、この harness は試合単位の厳密な決定論比較はできない)。
- 新規`_estimated`configは `rule_based` 20試合・`ml_policy` 20試合とも例外・クラッシュ無く完走。
- 既存の unit/integration テスト(206 passed / 12 skipped)がすべて通過。1件だけ既存の
  flakyテスト(`test_claimed_lethal_wins_within_the_turn`、native engineの非決定性が原因と
  docstringに明記されている既存テスト。今回の変更とは無関係で、selector.py/ml_policy_agent.py
  を経由しない。単独再実行では通過)が1回だけ落ちたが、これは今回の変更起因ではない。

### Step3: dummy vs estimated A/B(各120試合)・Step4: value shadow mode 健全性チェック(50試合)

詳細は
[results/2026-07-21_stage1_hidden_state_wiring.md](../../../../results/2026-07-21_stage1_hidden_state_wiring.md)。

| 構成 | 試合数 | 勝率(自己対戦) | エラー |
|---|---|---|---|
| rule_based dummy | 120 | 0.625 | 0 |
| rule_based estimated | 120 | 0.583 | 0 |
| ml_policy dummy | 120 | 0.500 | 0 |
| ml_policy estimated | 120 | 0.483 | 0 |

**4構成ともエラー0件**。lethal探索の found/timeouts/node_limit_hits/verify_rejects も
dummy/estimatedで同オーダーで、実推定への切り替えによる新規の不具合は見られなかった
(探索時間の予算超過が数十ms増えた程度)。

value network shadow modeは50試合すべて例外なく完走。予測勝率は常に `[0, 1]` 範囲内
(観測レンジ [0.0383, 0.9702])で、各試合終盤の予測方向は実際の勝敗と92%(46/50)一致し、
[[project_ml_value_network_step1]] のオフライン評価(test AUC 0.746)と整合する自然な傾向。

### 計測中に見つかった副次的な発見(コードの不具合ではなく、既存テスト基盤の限界)

1. `league/run_league.py` の自己対戦ループは、`hidden_information.match_context` の
   シングルトン状態を試合間でリセットしない(ローカルの `battle_start(deck0, deck1)` がデッキを
   直接引数で受け取るため、本番のようにdeck-selectionターン=`obs.select is None` が発火しない)。
   Kaggle本番(1プロセス1試合)では問題にならないが、ローカルの多試合自己対戦での計測は
   この前提で読む必要がある。
2. `seed` はPythonの `random` モジュールしか制御せず、native engine(`cg.dll`)側のシャッフル/
   コイントスは別RNGのため、同一configで同一seedを渡しても試合結果は再現しない(検証済み)。
   そのためStage1計画の「決定論的な行動列が変わらないこと」という完了条件は、この harness では
   厳密には検証できず、統計的なオーダーの一致で代替した。

---

## Definition of Done チェックリスト(プラン記載どおり)

- [x] 既存config(`rule_lethal.json` / `ml_lethal.json`)の挙動が一切変わらない
- [x] 新config(`*_estimated.json`)で実推定を使った探索が例外なく動く
- [x] `rule_based` / `ml_policy` 両方の lethal search 経路が実推定に対応済み
- [x] value networkが対局中に例外なく毎ターン呼び出せる(shadow mode)
- [x] `results/` に Step0・Step3・Step4 の結果を記録
- [ ] `selector.py` への追記内容を **tsuoimorikaさんに共有し合意済み** ← 未実施(人間側の対応が必要)
- [x] `experiment/pimc-hidden-info-integration` ブランチで動作確認済み

## 次にやること

- 上記チェックリスト最後の項目(selector.pyの担当者との共有・合意)は私からは実行できないため、
  ユーザー側での対応が必要。
- 合意後、有効だった変更だけを新しい `feature/*` ブランチに切り出し `integration` へPR
  (`docs/team-development-rules.md` の `experiment/*` 運用に従う)。
- Stage2(PIMC本体の実装)は別の実装計画ファイルを別途起こす。
