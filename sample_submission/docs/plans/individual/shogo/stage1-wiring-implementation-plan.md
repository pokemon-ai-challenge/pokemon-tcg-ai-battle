# Stage1 実装計画: hidden_information / value_model の配線接続

作成: 2026-07-21
状態: 実装計画（[ai-architecture-strategy.md](./ai-architecture-strategy.md) のStage1を、実装セッションにそのまま渡せる粒度まで分解したもの）
対象ブランチ: `experiment/pimc-hidden-info-integration`（`feature/ml-imitation-policy` 上にスタック）
このファイルの役割: **design(なぜ)ではなく、何を・どの順で・どう検証するか**を固定する。実装セッション開始時はこのファイルをそのまま渡す。

---

## スコープ

**やること**: 既に実装・学習済みの `hidden_information`（自分/相手の隠れ情報推定）と `value_model`（value network）を、実際の対戦ループに「配線」する。新しいアルゴリズムは書かない。

**やらないこと（Non-goals）**:
- PIMC/探索アルゴリズム本体の実装 → Stage2
- value networkの予測を使って意思決定を変える（ATTACK選択の上書き等） → Stage2以降。Stage1では**ログに出すだけ**（shadow mode）に留める。理由: 決定を変える設計は「候補手の結果をどう作るか」まで踏み込む必要があり、配線だけでは終わらないため。旧版の`ai-architecture-strategy.md`が「Stage1でATTACK補強に使う」と書いていたのは実質Stage2の内容だったので、本計画で訂正する。
- `hidden_information` 推定精度自体の改善
- `ptcg_ai/search/lethal_simple.py` の内部ロジック変更（一切触らない）

---

## 事前確認済みの契約（Contracts）

実装前にこれを確定させておく。ここがブレると実装セッションごとに設計が変わってしまう。

### 1. hidden-state factory の形

`action_selection/selector.py` と `ml_policy/ml_policy_agent.py` は、どちらも既に `context["hidden_state_factory"]: Callable[[], dict | None]` という差し込み口を持っている（`lethal_simple.py` はこれを外から受け取るだけで中身を知らない）。返す `dict` の形は固定:

```python
{
    "your_deck": list[int],
    "your_prize": list[int],
    "opponent_deck": list[int],
    "opponent_prize": list[int],
    "opponent_hand": list[int],
    "opponent_active": list[int],
}
```

現状: `hidden_information/search_state_stub.build_dummy_search_state(obs, full_deck, rng=None) -> dict | None`（ダミー、tsuoimorikaさん作）。

差し替え先: `hidden_information/search_adapter.to_search_begin_kwargs(own_state, opponent_state, obs, rng=None) -> dict`（実推定、**あなた自身が既に実装済み**、同じキー形状）。

`own_state` / `opponent_state` は `hidden_information/match_context.get_own_state()` / `get_opponent_state()`（**これもあなた自身の実装**）から取得する。`match_context.update(obs)` は既に `rule_based_agent.agent()` の先頭で毎ターン呼ばれているため、追加の更新処理は不要。

### 2. value network の呼び出し

`learning/value_model.ValueModel.predict_win_prob(obs: Observation) -> float`（**あなた自身の実装**、既に学習済み `value_weights.json` をロードするだけで動く）。Stage1では戻り値を**ログに出すだけ**で意思決定には使わない。

### 3. config schema の追加

既存の `lethal_search` セクションに1キー追加する。省略時は現状と完全に同じ挙動（後方互換）。

```json
{
  "name": "rule_lethal_estimated",
  "lethal_search": {
    "enabled": true,
    "module": "lethal_simple",
    "hidden_state_source": "estimated",
    "max_remaining_prizes": 2,
    "time_limit_ms": 100,
    "max_depth": 20,
    "max_nodes": 10000,
    "max_combinations_per_select": 128,
    "verify_shuffles": 1
  },
  "value_shadow_logging": true
}
```

`hidden_state_source` は `"dummy"`（デフォルト）| `"estimated"`。`value_shadow_logging` は `true` | `false`（デフォルト `false`）。

---

## ステップ（各ステップに完了条件を必須で付ける）

### Step 0: ベースライン記録（コード変更なし）

- 現行 `rule_lethal.json` / `ml_lethal.json` で `test_local_game_advanced.py --games 50 --opponent self` を実行し、勝率・平均行動時間・エラー回数を記録する。
- 完了条件: `results/2026-07-21_stage1_baseline.md` に記録済み（チームの実験結果テンプレートに従う）。
- 目的: Step1以降で「既存configの挙動が変わっていないか」を数値で確認できる基準を作る。

### Step 1: `selector.py` に hidden_state_source 分岐を追加

- ファイル: `ptcg_ai/action_selection/selector.py`（既存・tsuoimorikaさん担当・**追記のみ**）
- 追加するimport: `from ptcg_ai.hidden_information import match_context, search_adapter`
- `select_action()` 内、現在の

  ```python
  "hidden_state_factory": lambda: build_dummy_search_state(obs, full_deck),
  ```

  を、次のように分岐させる（追記のみ、既存行は削除せず条件化）:

  ```python
  hidden_state_source = lethal_config.get("hidden_state_source", "dummy")
  if hidden_state_source == "estimated":
      factory = lambda: search_adapter.to_search_begin_kwargs(
          match_context.get_own_state(), match_context.get_opponent_state(), obs
      )
  else:
      factory = lambda: build_dummy_search_state(obs, full_deck)
  context = {..., "hidden_state_factory": factory}
  ```

- **例外安全性の確認（新規コード追加は不要）**: `select_action()` は既に `try: action = module.search(...) except Exception: action = None` で `module.search()` 呼び出し全体を囲んでいる。`hidden_state_factory` 呼び出しは `lethal_simple.search()` の内部で行われるため、factory側で例外が出ても既存の外側try/exceptで捕捉され、自動的に `router.route()` へフォールバックする。つまりStage1のこの変更は**新しい例外処理を書く必要がない**（既存の安全網に乗るだけ）。ここは実装時に既存コードを読んで確認すること（推測で進めない）。
- 完了条件: `configs/rule_lethal.json`（`hidden_state_source`キーなし）で動かした結果がStep0のベースラインと完全に一致する（決定論的な行動列が変わらない）。新規 `configs/rule_lethal_estimated.json` を作り、`test_local_game_advanced.py --games 20` が例外・タイムアウトなく完走する。

### Step 2: `ml_policy_agent.py` に同じ分岐を追加

- ファイル: `ptcg_ai/ml_policy/ml_policy_agent.py`（既存・**あなた自身の担当**・追記のみ）
- `_try_lethal()` 内の同等箇所に、Step1と同じ分岐を追加する（このファイルは意図的に `action_selection/` から独立複製されているため、Step1のロジックをコピーする形になる）。
- 新規config: `configs/ml_lethal_estimated.json`。
- 完了条件: 既存 `ml_lethal.json` の挙動が不変。新configが `test_local_game_advanced.py --games 20` で完走。

### Step 3: A/Bで実推定 vs ダミーの差を計測

- `rule_lethal` vs `rule_lethal_estimated`、`ml_lethal` vs `ml_lethal_estimated` をそれぞれ100試合以上対戦させ、勝率・確定リーサル検出件数・平均行動時間を比較する（`league/run_league.py` または `test_local_game_advanced.py`）。
- 完了条件: `results/2026-07-21_stage1_hidden_state_wiring.md` に結果を記録。100試合では有意差の検出力は限定的（過去の実験メモ通り、数%差の検出には500試合以上必要）なので、ここでの目的は「壊れていないこと」の確認であり、「強くなったこと」の証明ではない、と明記する。

### Step 4: value network を shadow mode で配線（意思決定には未接続）

- 呼び出し口を1箇所決める（`core/agent.py` の入口、もしくは `match_context.update()` と同じタイミング）。`config.get("value_shadow_logging", False)` が `true` のときだけ `ValueModel().predict_win_prob(obs)` を呼び、ターンごとの値を対局終了時にログとして残す（`results/` 配下 or 簡易JSON）。
- 完了条件: 50試合、例外・タイムアウトなしで完走。ログされた値が `[0, 1]` の範囲に収まり、試合終盤で実際の勝敗方向に寄っていく傾向がある（厳密な検証ではなく健全性チェック）。

---

## Definition of Done（Stage1全体）

- [ ] 既存config（`rule_lethal.json` / `ml_lethal.json`）の挙動が一切変わらない（`hidden_state_source` 省略時 = 従来のダミー動作と一致）
- [ ] 新config（`*_estimated.json`）で実推定を使った探索が例外なく動く
- [ ] `rule_based` / `ml_policy` 両方の lethal search 経路が実推定に対応済み
- [ ] value networkが対局中に例外なく毎ターン呼び出せる（shadow mode）
- [ ] `results/` に Step0・Step3・Step4 の結果を記録
- [ ] `selector.py` への追記内容を tsuoimorikaさんに共有し合意済み
- [ ] `experiment/pimc-hidden-info-integration` ブランチで動作確認済み

Stage1が完了したら、有効だった変更だけを新しい `feature/*` ブランチに切り出し `integration` へPRする（`docs/team-development-rules.md` の `experiment/*` 運用に従う）。Stage2（PIMC本体）は別の実装計画ファイルを別途起こす。
