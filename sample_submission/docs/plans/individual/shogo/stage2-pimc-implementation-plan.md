# Stage2 実装計画: PIMC（value-scored own-turn search）

作成: 2026-07-21
状態: 実装計画（[ai-architecture-strategy.md](./ai-architecture-strategy.md) Stage2を実装セッションに渡せる粒度まで分解したもの）
前提: [stage1-wiring-implementation-plan.md](./stage1-wiring-implementation-plan.md) 完了済み（`hidden_state_source: "estimated"` 経路と value network shadow mode が動作確認済み）
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## スコープの決定（実装前に固定）

**今回作るPIMC v0は「自分の現在ターン内の行動系列を、value networkでスコアリングして最良のものを選ぶ探索」に限定する。相手ターンをまたぐ探索（相手の行動モデルが必要）はやらない。**

理由: `lethal_simple.py`の既存DFSは既に「自分のターン内のみ」に厳密にスコープを絞っている（`child_state.yourIndex != me: continue  # turn ended or control moved to the opponent`、`lethal_simple.py:298-299`）。この境界を守る限り、相手の行動を予測する必要が一切ない**単一エージェントの探索**で済み、既存インフラ（`_candidate_selections`のマクロアクション化、`search_begin`/`search_step`/`search_release`の呼び方、budget管理、`hidden_state_factory`経由の隠れ情報受け渡し）をほぼそのまま再利用できる。相手ターンをまたぐ探索は、相手の行動モデル（`ml_policy`を相手番に適用する等）が別途必要になり複雑度が跳ね上がるため、v1以降に切り出す。

**やること**: 「確実な勝ちがあれば勝つ（lethal_simpleと同じ）。無ければ、ターン内で可能な行動系列のうち、終了時の局面をvalue networkで評価して最も評価の高いものを選ぶ」という探索を、複数の隠れ情報サンプル（determinization）に対して行い、結果を集約する。

**やらないこと（Non-goals）**:
- 相手ターンをまたぐ探索・相手の行動モデル → v1以降
- `ml_policy`を候補手の事前確率（policy prior）として使う枝刈り → v1（本計画はvalue-onlyのPIMCをまず動かす。policy priorを同時に入れると「効果がどちらの寄与か」切り分けられなくなるため分離する）
- 本物のISMCTS（情報集合木の共有） → 別トラック（[ai-architecture-strategy.md](./ai-architecture-strategy.md)の却下理由に既述）
- `_candidate_selections`のマクロアクション化ロジック自体の改善 → 既存のものをそのまま使う
- `num_determinizations`・`max_depth`等のハイパラチューニング → v0は固定値で動くことの確認が目的。チューニングはStep5以降
- `lethal_simple.py`の変更 → 一切触らない（新規ファイルのみ）

---

## 事前確認済みの契約（Contracts）

実装前にコードを読んで確定させた。

### 1. 探索APIの使い方（`lethal_simple.py`から読み取った実際の呼び出しパターン）

```python
node = cg_api.search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active)
# node.searchId: int, node.observation: Observation (node.observation.current が State, .select が SelectData)
child = cg_api.search_step(node.searchId, selection)  # selection: list[int]。不正なら ValueError
cg_api.search_release(node.searchId)  # 1ノード解放。try/exceptで包む(lethal_simple.pyの慣習に合わせる)
cg_api.search_end()  # search()全体の最後に1回、finallyで呼ぶ
```

- ターン終了判定: `child.observation.current.yourIndex != me` になったら自分のターンが終わっている（探索対象外、葉として評価する）。
- 勝敗判定: `state.result == me`(勝ち) / `state.result != -1 and != me`(負け) / `state.result == -1`(継続中)。
- 候補手生成: `lethal_simple._candidate_selections(select, config)` と同じロジックを流用する（**コピーして独立させる**。`ml_policy_agent.py`が`selector.py`から`_is_valid_action`を意図的に複製している既存の慣習に合わせ、`search/lethal_simple.py`への依存はゼロに保つ）。

### 2. value network の葉評価

`ptcg_ai/learning/value_model.ValueModel.predict_win_prob_from_state(state) -> float`（**既存・あなた自身の実装**）。`State`を直接渡せる（`Observation`でラップ不要）ため、DFS中に`child.observation.current`をそのまま渡せる。未ロード時/`None`時は`0.5`を返し例外を出さない設計なので、呼び出し側での追加のtry/exceptは不要（Stage1の`hidden_state_factory`と同じく、既存の安全設計に乗るだけ）。

### 3. hidden-state factory

Stage1で確立した契約をそのまま使う。`context["hidden_state_factory"]: Callable[[], dict | None]`。PIMCは**この関数を`num_determinizations`回呼ぶ**（呼ぶたびに独立したサンプルが返る設計は`search_adapter.to_search_begin_kwargs`が内部で`own_state.sample(rng)`/`opponent_state.sample(rng)`を毎回引くことで既に保証されている、追加実装不要）。

### 4. モジュール登録とconfig（Stage1と同じ差し込み口を再利用）

`selector.py`・`ml_policy_agent.py`の`_SEARCH_MODULES`辞書に`"pimc": pimc`を追加するだけで良い（Stage1で既に確認済みの、両ファイルへの最小限の追記パターンを再利用）。**新しいconfigキー体系は作らず、既存の`lethal_search`セクションをそのまま使う**（`"module": "pimc"`にするだけで、`enabled`/`hidden_state_source`/`time_limit_ms`等は共通のまま）。PIMC固有のパラメータだけ`lethal_search`セクション内に追加する:

```json
{
  "name": "rule_pimc",
  "lethal_search": {
    "enabled": true,
    "module": "pimc",
    "hidden_state_source": "estimated",
    "time_limit_ms": 300,
    "max_depth": 20,
    "max_nodes": 10000,
    "max_combinations_per_select": 128,
    "num_determinizations": 4
  }
}
```

`time_limit_ms`は**探索全体（全determinization合計）の壁時計デッドライン**として扱う（determinizationごとに使い切りの予算を与えると`num_determinizations`倍で予算オーバーするため）。`num_determinizations`未指定時のデフォルトは4（Step4で妥当性を計測してから調整する、Non-goals参照）。

「`lethal_search`」というキー名は本来「確実なリーサル探索」を指すが、PIMCは確実勝ちが無くても最良の非勝利ラインを返す点で意味がやや広がる。**Stage2ではキー名は変更しない**（変更するとStage1で追加した`hidden_state_source`分岐にも波及し、tsuoimorikaさんとの合意が必要な箇所が増える）。将来的なリネームはfeature昇格時の検討事項として`ai-architecture-strategy.md`側に記録するに留める。

### 5. アルゴリズム（1回のdeterminizationあたり）

```
search(state, legal_actions, context):
    for i in range(num_determinizations):
        hidden_state = hidden_state_factory()
        root = search_begin(obs, **hidden_state)
        best_per_first_move = _explore(root, me, config, deadline)
        # best_per_first_move: dict[tuple[int, ...], float]  (最初の選択 -> このサンプルでの評価値)
        集計用の辞書に加算(選択のキーが揃っているものだけ平均を取る)
        search_end()
    最も平均評価値が高い最初の選択を返す(1件も見つからなければ None → 呼び出し側は router.route() にフォールバック)
```

`_explore`はlethal_simpleの`_dfs`と同型の再帰で、以下の点だけ異なる:
- 勝ち(`state.result == me`)を見つけたら即座にそのラインをスコア`1.0`として記録して打ち切る(lethal_simpleと同じ「確実勝ちなら即採用」の性質を保つ)。
- 深さ上限・ノード上限・ターン終了(`yourIndex != me`)に達したら、そこを葉として`predict_win_prob_from_state(leaf_state)`をスコアにする。
- 各「最初の選択」ごとに、その配下で見つかった最良スコアを記録する(単純なmaximizing、対戦相手の応手は考慮しない。Non-goals参照)。

---

## ステップ

### Step 0: 新規ファイルのスタブ登録（コード変更は最小、動作は無変化）

- 新規ファイル `ptcg_ai/search/pimc.py`。`search(state, legal_actions, context) -> list[int] | None` を実装するが、中身は常に`None`を返すだけのスタブ。
- `selector.py`・`ml_policy_agent.py`それぞれに `from ptcg_ai.search import pimc` と `_SEARCH_MODULES["pimc"] = pimc` を追記（Stage1と同型の最小追記）。
- 新規config `configs/rule_pimc.json`（スタブなので実質`router.route()`に全部委譲される）。
- 完了条件: 既存config・既存テストに影響なし。新configで`test_local_game_advanced.py --games 10`が例外なく完走（実質rule_baseと同じ挙動になるはず）。

### Step 1: 単一determinization・勝敗のみ判定（value評価なし）でlethal_simple相当を再実装

- `pimc.py`に`_candidate_selections`・`_begin`等を複製し、勝敗探索のみ動くDFSを実装（value networkはまだ呼ばない）。
- 目的: 探索API呼び出し（search_step/search_release/search_end）の基本部分が正しく動くことを、value評価という新しい変数を混ぜる前に単体で確認する。
- 完了条件: `tests/integration/test_lethal_search.py`と同型のfixtureを使い、`pimc.py`が`lethal_simple.py`と同じ局面で同じ「確実勝ちの有無」判定になることをunit testで確認する（既存の確実勝ちシナリオに対する回帰チェック）。

### Step 2: value networkによる葉評価を追加（determinizationは1のまま）

- 深さ上限・ノード上限・ターン終了時に`ValueModel().predict_win_prob_from_state(leaf_state)`を呼び、各「最初の選択」ごとの最良スコアを追跡する。
- 完了条件: 意図的に「勝ちはないが明らかに有利な局面」と「明らかに不利な局面」を作るfixtureを2つ用意し、有利な方の行動が高スコアで選ばれることをunit testで確認する。

### Step 3: determinizationループと集約を追加

- `num_determinizations`回ループし、`hidden_state_factory()`を毎回呼び直す。「最初の選択」をキーにスコアを平均し、最良のものを返す。
- 完了条件: `hidden_state_factory`をモックして毎回異なる隠れ状態を返すテストで、集約（平均）が正しく計算されることを確認。`test_local_game_advanced.py --games 20`（`num_determinizations=4`）が例外・タイムアウトなく完走。

### Step 4: 時間予算の実測とNon-goalsの検証

- Stage1で判明した「`to_search_begin_kwargs()`のコストが探索予算に乗る」問題は、determinizationが複数回に増える分、影響が拡大する。壁時計デッドラインを探索全体で共有する設計（契約§4）が機能しているか、実際の平均/最大実行時間を計測する。
- 完了条件: `results/2026-07-21_stage2_pimc_timing.md`のような形で、`num_determinizations`を1/2/4/8と振った時の平均/最大実行時間・timeoutカウント・node_limit_hitカウントを記録。Kaggleの持ち時間制約に対して安全な値をここで初めて決める（決め打ちしない）。

### Step 5: A/B比較（`rule_lethal_estimated` vs `rule_pimc`）

- Stage1のStep3と同じ方法論（`league/run_league.py`の既知の限界を踏まえる、統計的レンジでの比較に留める）で、勝率・平均ターン数・エラー件数を比較する。
- 完了条件: `results/2026-07-21_stage2_pimc_ab.md`に記録。ここで初めて「本当に強くなったか」の議論ができる（Step0-4はすべて「壊れていないか」の確認）。

---

## Definition of Done（Stage2 v0）

- [ ] `pimc.py`が新規ファイルのみで完結し、`lethal_simple.py`は無変更
- [ ] `selector.py`・`ml_policy_agent.py`への追記は`_SEARCH_MODULES`辞書への1行ずつのみ
- [ ] Step1のunit testで既存の確実勝ち判定と同等の結果になることを確認済み
- [ ] Step2のunit testで value 評価による優劣判定が機能することを確認済み
- [ ] Step3-4でdeterminization集約と時間予算が実測・記録済み
- [ ] Step5でrule_lethal_estimatedとのA/B結果が`results/`に記録済み
- [ ] `selector.py`/`ml_policy_agent.py`への追記内容をtsuoimorikaさんに共有・合意済み（Stage1と同じ相談ポイント）

Stage2 v0完了後、次の分岐点（v1: policy prior統合、または相手ターンをまたぐ探索への拡張）は、Step5の結果を見てから別ファイルで計画する。
