# Stage2 実装計画: PIMC 実行コスト対策

作成: 2026-07-21
状態: 実装計画（[stage2-pimc-implementation-plan.md](./stage2-pimc-implementation-plan.md) 完了後の優先課題への対応）
前提: Stage2 v0（[stage2-pimc-implementation-result.md](./stage2-pimc-implementation-result.md)）で「1試合平均21〜28秒・最大42秒、プライズ枚数ゲート無し」という実測結果が出ている
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## スコープ

**やること**: PIMCの実行コストを、実際のKaggle予算に対して安全な水準まで下げる。具体的には (1) 無駄な探索起動を減らす、(2) 必要なら`lethal_simple`と同様のプライズ枚数ゲートを追加する、(3) `time_limit_ms`/`num_determinizations`を実測ベースで調整する。

**やらないこと（Non-goals）**:
- policy prior統合・相手ターンをまたぐ探索 → 別の実装計画（Step0で判明する「本当に急ぐべきか」次第で優先度を見直す）
- `match_context`のプレイヤー分離（head-to-head評価） → 別の実装計画
- `lethal_simple.py`の変更 → 一切触らない
- 探索アルゴリズム自体（DFSの形・value評価の使い方）の変更 → v0のロジックは変えず、起動条件とパラメータだけを調整する

---

## Step 0（最優先・コード変更なし）: 実際のKaggle予算に対する正確な消費量を確定する

**背景**: Stage2の`results/2026-07-21_stage2_pimc_timing.md`は「1試合平均21〜28秒」を根拠に「Kaggleの持ち時間制約に対して危険域」と結論したが、この数値の性質を再検証する必要がある。

実際のKaggle競技仕様を確認した結果（`kaggle_replays/replays/episode-85169987-replay.json`の`specification`フィールド、実際の対戦データに埋め込まれた本物の仕様。ローカルのドキュメントではなく一次情報）:

```json
"actTimeout": {"default": 0, "description": "Maximum runtime (seconds) to obtain an action from an agent"},
"remainingOverageTime": {"default": 600, "description": "Total remaining banked time (seconds)... agent is disqualified when this drops below 0", "shared": false}
```

つまり実際の予算は **「1エージェントあたり合計600秒（10分）、試合を通じての累積」**（`actTimeout=0`なので毎手の思考時間がそのままこの600秒の残高から差し引かれる、`shared: false`なので両陣営は別々の残高を持つ）。

一方、Stage2で測定した「1試合平均21〜28秒」は、自己対戦（両陣営とも`rule_pimc`）をシーケンシャルな1プロセスで駆動した**試合全体の壁時計時間**であり、**1エージェント分の消費時間ではない**（両陣営の思考時間の合計）。この2つを混同したまま「危険域」と結論するのは正確ではない可能性がある。

### 作業内容

- Stage2のタイミング計測スクリプトを改修し、`player0`（1エージェント分）が実際に`pimc.search()`の中で消費した時間だけを分離して集計する（`pimc.get_stats()`の`total_time_ms`は既にプロセス全体累積なので、自己対戦で両陣営が同じプロセス内の同じ`pimc`モジュールを共有している場合はplayer0/player1の内訳が取れない。これを分離するには、両陣営を別プロセスで対戦させる、または`pimc.py`にプレイヤー別の統計を持たせる一時的な計測用フックが必要）。
- 得られた「1エージェントが1試合で実際に消費する思考時間」を600秒と比較し、使用率（%）を算出する。

### 完了条件

- `results/2026-07-21_stage2_cost_step0_budget_reality_check.md` に、(a) 実際のKaggle予算(600秒/エージェント/試合、出典付き)、(b) 1エージェントあたりの実測消費時間、(c) 予算に対する使用率(%)を明記する。
- **この結果次第で以降のステップの優先度・目標値が変わる**。使用率が十分低ければ（例: 10%未満）、Step1-3は「安全マージンを取るためのベストプラクティス」程度の優先度に格下げしてよい。使用率が高い、または変動が大きい（一部の試合で予算の大半を使う）場合は、Step1-3を本番投入のブロッカーとして扱う。

---

## Step 1: 無駄な探索起動を減らす（`SelectType.MAIN`以外をスキップ）

- 現状、`pimc.search()`は`SelectType`を見ずに、自分のターンであれば常にフルの`hidden_state_factory()`呼び出し＋DFSを試みる。しかし`_candidate_selections`内の`OptionType`優先順位付けは`SelectType.MAIN`前提の設計であり、YES_NOやカード選択などの非MAIN選択でも同じコストの探索が起動している（Stage2実測の`searches: 1745件/15試合 ≈ 116件/試合`という高い起動回数の一因と見られる）。
- `search()`の冒頭、`hidden_state_factory()`を呼ぶ前に`obs.select.type == SelectType.MAIN`でない場合は即座に`None`を返すガードを追加する（`hidden_state_factory`のセットアップコスト自体を払う前にスキップする点が重要）。
- 完了条件: 同じ自己対戦シナリオで`pimc.get_stats()["searches"]`が明確に減ることを確認（何件から何件に減ったかを記録）。非MAIN選択が従来通り`router.route()`/`PolicyModel`にフォールバックし、legal actionを返すことを既存テストで確認（回帰なし）。

## Step 2: プライズ枚数ゲート（オプション、既定は無効＝現状互換）

- `lethal_simple.py`の`max_remaining_prizes`と同じ形のconfigキーを`pimc.py`にも追加する。ただし**既定値は「ゲート無し」のまま**にする（Step0の結果次第でデフォルトを変えるかを判断するため、ここでは選べるようにするだけ）。
- 実装: `config.get("max_remaining_prizes")`が`None`でなければ`len(state.players[me].prize) > max_remaining_prizes`のとき`search()`は即座に`None`を返す（`lethal_simple.py:141`と同型のガード、コピーして独立させる）。
- 完了条件: `max_remaining_prizes`を指定したconfigとしないconfigの両方でunit testを追加し、ゲートの有無で挙動が変わることを確認。既定（未指定）では現行v0と完全に同じ挙動（回帰なし）。

## Step 3: `time_limit_ms`/`num_determinizations`の再調整

- Step0-2の結果を踏まえ、実測ベースで妥当な値を決める（決め打ちしない）。候補として`time_limit_ms`を下げる、`num_determinizations`を下げる、あるいはStep0の結果次第では現状維持もありうる。
- 完了条件: `results/2026-07-21_stage2_cost_step3_retuning.md`に、調整後のパラメータでの再計測（1エージェントあたりの消費時間、予算に対する使用率）を記録。

## Step 4: 再検証（Step1-3適用後のA/Bスモークテスト）

- Stage2 Step5と同じ方法論（自己対戦のみ、`match_context`の限界を踏まえる）で、Step1-3適用後の`rule_pimc`が引き続きエラー0件で動くことを確認する。
- 完了条件: `results/2026-07-21_stage2_cost_step4_recheck.md`に記録。1エージェントあたりの予算使用率が明確な数値として残っていること。

---

## Definition of Done

- [ ] Step0で実際のKaggle予算（600秒/エージェント/試合）に対する正確な使用率が判明している
- [ ] `SelectType.MAIN`以外への無駄な探索起動が無くなっている（探索回数の削減を数値で確認済み）
- [ ] プライズ枚数ゲートがオプションとして実装され、既定動作は後方互換
- [ ] Step0の結果に基づいた妥当な`time_limit_ms`/`num_determinizations`が再計測込みで記録されている
- [ ] Step1-3適用後もエラー0件を確認済み
- [ ] `selector.py`/`ml_policy_agent.py`への追記内容をtsuoimorikaさんに共有・合意済み（Stage1/2から持ち越しの項目、まだ未実施なら本Stageの分もまとめて共有する）

Step0の結果が「実は十分安全だった」場合、Step1-3は優先度を下げて良い（Definition of Doneの該当項目を「対応不要と判断」に変えて良い）。その場合は次の優先課題（`match_context`のプレイヤー分離、またはpolicy prior統合）に進む判断を、本ファイルの結論を踏まえてユーザーと相談する。
