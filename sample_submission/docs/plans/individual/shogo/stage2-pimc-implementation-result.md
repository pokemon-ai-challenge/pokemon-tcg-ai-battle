# Stage2 実装結果: PIMC(value-scored own-turn search)

作成: 2026-07-21
対応: [stage2-pimc-implementation-plan.md](./stage2-pimc-implementation-plan.md)
ブランチ: `experiment/pimc-hidden-info-integration`

---

## やったこと

### 新規ファイル `ptcg_ai/search/pimc.py`

`lethal_simple.py` は一切変更していない(新規ファイルのみ、Definition of Doneどおり)。

- `search(state, legal_actions, context) -> list[int] | None` を共通インターフェースとして実装。`context` の受け取り方(`observation`/`config`/`hidden_state_factory`)は `lethal_simple` と共通契約のまま。
- `_candidate_selections` / `_is_my_turn` / `_is_legal_selection` / `_begin` / `_state_key` / `_MAIN_OPTION_PRIORITY` は `lethal_simple.py` からコピーして独立させた(計画の契約どおり、`ml_policy_agent.py` が `selector.py` から `_is_valid_action` を複製している既存の慣習に合わせた)。
- `_explore` / `_dfs_score`: `lethal_simple._dfs` と同型の再帰だが、
  - 勝ち(`state.result == me`)を見つけたら即座にスコア`1.0`として記録し、その場でそのdeterminizationの探索全体を打ち切る(木の他の分岐は評価しない)。
  - 深さ上限・ノード上限・ターン終了(`yourIndex != me`)・相手勝ち(`0.0`)に達したら、そこを葉として `ValueModel().predict_win_prob_from_state(leaf_state)` をスコアにする。
  - 各「最初の選択」ごとに配下の最良スコアを記録する(単純なmaximizing、相手の応手は考慮しない)。
  - 同一状態への再訪問は `visited` にメモ化(`(depth_left, best_score, found_win)` のタプル)して再探索を避ける。
- `_run_determinizations` / `search`: `num_determinizations` 回 `hidden_state_factory()` を呼び直し、各回の `_explore` 結果を「最初の選択」キーごとにリストへ蓄積、最後に**キーごとに存在する分だけ**平均を取って最良のキーを返す(プラン契約どおり「選択のキーが揃っているものだけ平均を取る」= 早期打ち切りで一部のdeterminizationにしか現れないキーも、そのキーが存在する分だけで平均する)。determinizationループ自体は早期終了しない(1つのdeterminizationで勝ちが見つかっても、他のdeterminizationでその手が勝たない可能性を平均で反映させるため、契約§5の擬似コードどおり)。
- 計測用の `get_stats()`/`reset_stats()` を追加(`lethal_simple` と同じ形だが、`searches`/`found` に加えて `determinizations_run`/`determination_timeouts`/`determination_node_limit_hits` を持つ。Step4の実測に必須だったため計画外で追加)。
- **プライズ枚数によるゲートは意図的に実装していない**(契約§4のconfig例に `max_remaining_prizes` が無いことに対応。Step4/5で「これが実行コストの主因」と判明した、詳細は下記)。

### `selector.py` / `ml_policy_agent.py` への追記

計画どおり、それぞれ1行ずつの最小追記のみ:

```python
from ptcg_ai.search import lethal_simple, pimc

_SEARCH_MODULES = {
    "lethal_simple": lethal_simple,
    "pimc": pimc,
}
```

### 新規config `configs/rule_pimc.json`

計画の契約§4に記載の例をそのまま使用(`module: "pimc"`、`hidden_state_source: "estimated"`、`time_limit_ms: 300`、`num_determinizations: 4`)。既存configは無変更。

### 新規テスト `tests/unit/test_pimc.py`

`tests/unit/test_lethal_simple.py` と同じFakeEngineパターンに加え、`pimc._get_model` を差し替える `FakeValueModel` を導入。19件、計画のStep1-3それぞれの完了条件に対応させた:

- Step1相当: 確定勝ちの検出・複数手にまたがる勝ち筋・END未探索・相手勝ち枝が選ばれないこと
- Step2相当: value networkスコアの高い行動が選ばれること(逆に低い方が選ばれないことも)
- Step3相当: determinization間の平均集計が正しいこと、**早期打ち切りで一部のdeterminizationにしかキーが無い場合でも部分平均が機能すること**(打ち切られたdeterminizationでしか勝ちが出ないケースと、両方で評価されるが値が割れるケースの2パターン)、`hidden_state_factory` が `None` を返すサンプルをスキップすること
- ゲート/config: `enabled=False`・プライズ枚数ゲートが**無い**こと(lethal_simpleとの意図的な違いを明示するテスト)・自分のターンでない場合のスキップ・`hidden_state_factory` 未指定時のスキップ
- 統計・fallback: `get_stats()`、`_SEARCH_MODULES` への登録確認、`selector.select_action` からのフォールバック

`python -m pytest tests/unit -q`: 223 passed, 6 skipped(既存テストと合わせて全green)。`tests/integration -q`: 2 passed, 6 skipped(既存のskip理由と同じ、新規のskipは無い)。

---

## 実測結果(Step4/Step5)

### Step4: 時間予算の実測

詳細: [results/2026-07-21_stage2_pimc_timing.md](../../../../results/2026-07-21_stage2_pimc_timing.md)

`num_determinizations` を1/2/4/8と振って自己対戦3試合ずつ(探索呼び出し数261〜421件/設定)。

- `avg_time_ms`(199〜243ms)は `num_determinizations` を増やしても伸びない。契約§4「壁時計デッドラインを探索全体で共有する」設計は意図通り機能している。
- ただし `determinizations_run/searches` の比は設定値に対して頭打ち(d=8でも平均約3サンプルしか完走しない)。
- `found`(=何らかの行動を返せた率)が23〜45%と低く、半分以上の`search()`呼び出しで1件もdeterminizationが完走せず `router.route()` にフォールバックしている。
- **プライズ枚数によるゲートが無いため、ほぼ全ての自ターン意思決定点(1試合94〜148手の大部分)で最大300ms級の探索が起動し、1試合の実行時間が平均21〜28秒・最大42秒に達する。** これはKaggleの持ち時間制約に対して現状のまま使うにはリスクが高い、という実測に基づく警告として記録した(対策自体はNon-goals、v1以降)。

### Step5: A/B比較(`rule_lethal_estimated` vs `rule_pimc`)

詳細: [results/2026-07-21_stage2_pimc_ab.md](../../../../results/2026-07-21_stage2_pimc_ab.md)

- 自己対戦45試合(lethal_simple側30 + pimc側15、Step4の実測を踏まえ`rule_pimc`は試合数を大きく絞った)を通じて**エラー0件**。健全性の確認基準は満たした。
- 実行コストは`rule_pimc`が`rule_lethal_estimated`の約47倍(平均秒数/試合ベース: 26.6s vs 0.56s)。
- **本当のhead-to-head対戦による強さの比較はできなかった**(新たに判明した限界): `hidden_information.match_context` はプロセス内シングルトンで「自分視点」の状態を1つしか保持できず、1試合の中で異なるconfig(片方推定・片方PIMC)を両プレイヤーに割り当てると `_own_state`/`_opponent_state` が両陣営分を混同して壊れる。Stage1のhead-to-head(ml_policy vs rule_based、500試合)は両方とも `hidden_state_source` 省略(dummy)だったためこの問題を踏んでいなかった。今回`hidden_state_source="estimated"` を使う2つのconfigを対戦させようとして初めて顕在化した。そのためStage1 Step3と同じ「自己対戦のみ」の方法論に留め、勝率比較は「強さ」ではなく「壊れていないか」の確認とした。

---

## 計画からの逸脱・計画外で判明した事項

1. **`get_stats()`/`reset_stats()` の追加**: 計画には明記されていなかったが、Step4の時間予算実測に必須だったため`lethal_simple`と同型で追加した。
2. **プライズ枚数ゲート無しによる実行コストの重さ**: 契約§4のconfig例に既に反映されていた設計(ゲート無し)だが、実際にどれだけ重いか(1試合あたり平均21〜28秒)は実測するまで分からなかった。Step4のドキュメントに記録済み。
3. **head-to-head評価ができないという`match_context`の制約**: Stage2で新たに判明した限界。v1計画時にはこの制約への対応(プレイヤーごとの状態分離、またはプロセス分離での対戦)を検討事項に含める必要がある。

---

## 追記(2026-07-21): 実行コストに関する結論の訂正

上記Step4/5で「Kaggleの持ち時間制約に対してリスクが高い」としていた結論は不正確だった。
[stage2-cost-control-implementation-plan.md](./stage2-cost-control-implementation-plan.md) Step0
(`results/2026-07-21_stage2_cost_step0_budget_reality_check.md`)で、実際のKaggle予算(一次情報:
`kaggle_replays/replays/episode-85169987-replay.json` の `specification.observation`、
`remainingOverageTime.default=600`秒/エージェント/対局、`shared: false`)に対して**1エージェント視点**
で測り直した結果、平均使用率2.22%・最大4.04%(30サンプル)と十分低いことが判明した。Step4/5で
測っていた「1試合平均21〜28秒」は自己対戦を1プロセスで駆動した際の**対局全体(両陣営合計)の
壁時計時間**であり、実際に予算と比較すべき「1エージェント分の消費時間」ではなかった(この2つを
混同していた)。

結論として、`rule_pimc`の実行コスト対策(プライズゲート追加等)は本番投入のブロッカーではなく、
優先度の低い改善項目に格下げされた。詳細・留保点は上記Step0の結果ファイルを参照。

## Definition of Done チェックリスト(プラン記載どおり)

- [x] `pimc.py`が新規ファイルのみで完結し、`lethal_simple.py`は無変更
- [x] `selector.py`・`ml_policy_agent.py`への追記は`_SEARCH_MODULES`辞書への1行ずつのみ
- [x] Step1のunit testで既存の確定勝ち判定と同等の結果になることを確認済み
- [x] Step2のunit testでvalue評価による優劣判定が機能することを確認済み
- [x] Step3-4でdeterminization集約と時間予算が実測・記録済み
- [x] Step5でrule_lethal_estimatedとのA/B結果が`results/`に記録済み
- [ ] `selector.py`/`ml_policy_agent.py`への追記内容をtsuoimorikaさんに共有・合意済み ← 未実施(人間側の対応が必要、Stage1と同じ)

## 次にやること

- 上記チェックリスト最後の項目(担当者との共有・合意)はユーザー側での対応が必要。
- v1以降の分岐点(Step5結果を踏まえて別ファイルで計画する):
  - `rule_pimc` の実行コスト対策(プライズ枚数ゲート追加、`time_limit_ms`/`num_determinizations`のチューニング) — 上記追記のとおり本番投入のブロッカーではなく優先度の低い改善項目
  - head-to-head評価を可能にする `match_context` のプレイヤー分離
  - policy prior統合、相手ターンをまたぐ探索への拡張(`stage2-pimc-implementation-plan.md` 末尾参照)
