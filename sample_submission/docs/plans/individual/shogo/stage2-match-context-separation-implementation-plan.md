# Stage2 実装計画: match_context のプレイヤー分離（head-to-head評価の有効化）

作成: 2026-07-21
状態: 実装計画
前提: [stage2-cost-control-implementation-plan.md](./stage2-cost-control-implementation-plan.md) Step0完了（実行コストは問題無しと判明、これ以上のコスト対策は優先度低に格下げ）
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## スコープ

**やること**: `hidden_information/match_context.py` のモジュールグローバル状態（`_own_state`/`_opponent_state`/`_knowledge`）を、`player_index`（0/1）ごとに独立させる。これにより、1プロセス内で異なるconfigの2エージェント（例: `rule_lethal_estimated` と `rule_pimc`）を対戦させても、互いの隠れ情報推定が混線しなくなる。

**背景**: [[project_match_context_single_perspective_limitation]]で判明した通り、現状は`_own_state`等が単一のグローバルオブジェクトで、`update(obs)`が`obs.current.yourIndex`に応じて同じオブジェクトを上書きする設計。**本番のKaggle実行(1プロセス1エージェント)ではこれで正しい**(そのプロセスには常に1人分の視点しか存在しないため)。問題が起きるのは、ローカルの自己対戦harness(`league/run_match.play_match`)が1プロセス内で両陣営の`agent(obs)`を交互に呼ぶ場合のみ。

**やらないこと（Non-goals）**:
- `league/run_league.py`のCLI(`AGENT_REGISTRY`/`--agent-a`/`--agent-b`)に「片側だけ別config」を選べるオプションを追加すること → 別タスク。今回は`run_match.play_match`を直接呼ぶカスタムスクリプト(Stage1/Stage2の計測スクリプトと同型)で検証するに留める。
- Kaggle本番の実行トポロジー自体の変更 → 不要（本番はそもそも問題が起きない）。
- `obs.select is None`（デッキ選択）時の`reset()`のプレイヤー別化 → **ローカルharnessは`battle_start(deck0, deck1)`でデッキを直接引数として渡すため、この経路(`obs.select is None`)自体を通らないことをStage1で確認済み**（`results/2026-07-21_stage1_hidden_state_wiring.md`参照）。本番は1プロセス1エージェントなので、この経路で全体を`reset()`しても他プレイヤーを巻き込まない。よって`reset()`の全体クリア(引数無し)という既存の意味は変更しない。
- policy prior統合・探索アルゴリズム自体の変更・コスト対策Step1-3(優先度低のまま) → 別スコープ

---

## 事前確認済みの契約（Contracts）

### 1. 現状の呼び出し箇所（全て`grep`で洗い出し済み）

- `rule_based_agent.py:36`: `match_context.update(obs)`
- `ml_policy_agent.py:67`: `match_context.update(obs)`
- `selector.py:76`: `match_context.get_own_state()` / `match_context.get_opponent_state()`
- `ml_policy_agent.py:144`: 同上
- `tests/unit/test_match_context.py`: 約14箇所（`update`/`get_own_state`/`get_opponent_state`/`reset`）
- `tests/local_sim/test_local_game_hidden_info.py`: `get_own_state()`呼び出し1箇所

これ以外に本番コードからの呼び出しは無い（`grep`で確認済み、契約の範囲はこれで閉じている）。

### 2. `Observation`/`State`のフィールド制約

`State.yourIndex`は`State`のフィールドであり、`obs.current`が`None`（デッキ選択ターン）の間は取得できない（`cg/api.py:439-443`、`Observation.current: State | None`）。**`update()`の`obs.select is None`分岐では`yourIndex`を得る手段が無い**ため、この分岐のプレイヤー別対応はスコープ外とする（Non-goals参照、実害が無いことも確認済み）。通常ターン（`obs.current`が非None）では`state.yourIndex`が常に取得できるため、こちらは問題なくキーにできる。

### 3. 新しいシグネチャ

```python
# 変更前 → 変更後
def update(obs: Observation) -> None: ...  # シグネチャ不変(内部でstate.yourIndexをキーに使うだけ)
def get_own_state(player_index: int) -> OwnHiddenState: ...       # player_index を必須化(新規引数)
def get_opponent_state(player_index: int) -> OpponentHiddenState: ...  # 同上
def reset() -> None: ...  # シグネチャ不変(全player_indexぶんまとめてクリア、既存の意味を維持)
```

`get_own_state`/`get_opponent_state`にデフォルト値を与えない（例: `player_index: int = 0`のような暗黙のデフォルトは、今回直そうとしている「暗黙の単一視点」バグを形を変えて再導入するだけなので避ける）。呼び出し側は全箇所で`obs`が既にスコープ内にあるため、`obs.current.yourIndex`を明示的に渡せる。

### 4. `league/run_match.play_match`は変更不要

`play_match(agent0, agent1, deck0, deck1, seed=None)`（`league/run_match.py`）は、既に player_index=0/1 それぞれに**独立したコールバック**を受け取れる設計になっている（確認済み、新規実装不要）。今回の変更が完了すれば、`agent0`/`agent1`に別々のconfigを束縛したクロージャを渡すだけで、正しいhead-to-head対戦が成立する。

---

## ステップ

### Step 1: `match_context.py` のコア変更

- `_own_state: OwnHiddenState | None` → `_own_states: dict[int, OwnHiddenState]`（`_opponent_states`/`_knowledge`も同様にdict化）。
- `update(obs)`の通常ターン分岐で`me = state.yourIndex`をキーに、該当エントリが無ければ遅延生成する（既存の「未初期化でも例外を出さない」フォールバック挙動を、dictの`.setdefault`相当で維持する）。
- `get_own_state(player_index)`/`get_opponent_state(player_index)`は該当エントリを返す（無ければ`get_own_state`の既存フォールバック=空デッキから作り直す、`get_opponent_state`の既存フォールバック=空のインスタンス、をそれぞれ`player_index`単位で行う）。
- `reset()`は両方のdictを丸ごとクリアする（既存の全体リセット動作を維持）。
- 完了条件: `tests/unit/test_match_context.py`を新シグネチャに書き換えた上で全green。加えて新規テストケースを追加: 「`update()`をplayer0視点→player1視点→player0視点…と交互に呼んでも、`get_own_state(0)`と`get_own_state(1)`が互いに独立した値を返し続ける」ことを検証する（今回のバグそのものの再発防止テスト）。

### Step 2: 呼び出し側（`selector.py`/`ml_policy_agent.py`）の更新

- `selector.py:76`: `match_context.get_own_state()` → `match_context.get_own_state(obs.current.yourIndex)`（`get_opponent_state`も同様）。
- `ml_policy_agent.py:144`: 同様の変更。
- 完了条件: 既存の単体・統合テストが引き続きgreen。`test_local_game_advanced.py --games 20`で回帰なし（Stage1/Stage2と同じ確認方法）。

### Step 3: `tests/local_sim/test_local_game_hidden_info.py` の呼び出し更新

- `get_own_state()` → `get_own_state(player_index)`（該当箇所の文脈から正しい`player_index`を渡す）。
- 完了条件: 該当テストファイルがgreen。

### Step 4: 実際のhead-to-head検証（`rule_lethal_estimated` vs `rule_pimc`）

- Stage1/Stage2の計測スクリプトと同型のカスタムスクリプトを書き、`league/run_match.play_match`を直接呼ぶ。`agent0`/`agent1`は、それぞれ異なるconfig(`rule_lethal_estimated`/`rule_pimc`)を束縛したクロージャとして用意する(`ptcg_ai.action_selection.selector.select_action(obs, full_deck, config=...)`が既に`config`引数を受け付けるため、`rule_based_agent.agent`を直接使わずこのクロージャから`selector.select_action`を呼べば、プロセス全体で共有される`_CONFIG_CACHE`を経由せずに済む)。
- 先手/後手を入れ替えながら十分な試合数(コスト対策Step0の実測: `rule_pimc`側は1エージェントあたり平均13秒/試合、45〜100試合程度なら現実的な時間で回せる見込み)を対戦させ、勝率・95%信頼区間・エラー件数を記録する。
- 完了条件: `results/2026-07-21_stage2_head_to_head.md`に、PIMCがlethal_simple(+value shadow)に対して有利/互角/不利のどれかを、統計的な言葉で記録する。**ここで初めてStage2全体の問いである「PIMCは本当に強いのか」に答えが出る。**

---

## Definition of Done

- [ ] `match_context.py`が`player_index`別に状態を保持するよう変更済み
- [ ] 新規の「player0/player1独立性」回帰テストを含め、既存テストが全てgreen
- [ ] `selector.py`/`ml_policy_agent.py`の呼び出し側が新シグネチャに対応済み
- [ ] `rule_lethal_estimated` vs `rule_pimc`の実際のhead-to-head結果が`results/`に記録済み
- [ ] `selector.py`/`ml_policy_agent.py`への累積の追記内容(Stage1・Stage2・本Stage分)をtsuoimorikaさんに共有・合意済み(まだなら本Stage完了のタイミングでまとめて共有するのを推奨)

Step4の結果が出て初めて、v1(policy prior統合・相手ターンをまたぐ探索)に進む価値があるか、あるいはPIMCの方向性自体を見直すべきかが判断できる。次の分岐点はStep4の結果を見てから決める。
