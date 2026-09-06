# Step 0 Capability Report — リーサル探索 Phase 1/2/3

- 対象: `ptcg_lethal_search_phase1_2_3_claude_design.md`（以下「原設計」）§16 Step 0
- 対象ブランチ: `integration`（HEAD `79eca91` + 作業ツリーの未コミット変更）
- 状態: **調査のみ完了。実装コードは一切変更していない。**
- 姉妹文書: [`design.md`](./design.md) / [`../lethal-macro-search/design.md`](../lethal-macro-search/design.md)

本書は「何を測ったか」「どう測ったか」「何が言えて、何が言えないか」を分離して記録する。
**測定は全て単一環境・単一デッキ（フーディン）・限定サンプル**であり、
一般的なエンジン仕様の証明ではない。設計はこの限界を前提に「推論せず検証する」方針を取る（§5.3, §9）。

---

## 1. 実ファイル／実クラス／実関数との対応表

### 1.1 原設計の抽象名 → 実体

| 原設計の抽象名 | 実体（ファイル:行） | 備考 |
|---|---|---|
| 行動決定エントリポイント | `main.py:18` `agent()` → `ptcg_ai/core/agent.py` → `ptcg_ai/action_selection/selector.py:47` `select_action()` | 返り値は `list[int]`（option インデックス） |
| 探索モジュール契約 | `selector.py:22` `_SEARCH_MODULES` / `search(state, legal_actions, context)` | 現在の実装は `lethal_simple` のみ |
| 既存 Phase 1・2 実装 | `ptcg_ai/search/lethal_simple.py:127` `search()` | 反復深化 DFS。439 行 |
| 合法手 | `obs.select.option`（エンジン生成） | **単一 ID ではなく複数選択の `list[int]`**（`minCount`〜`maxCount`、重複不可） |
| `Observation` | `cg/api.py:439` | `select` / `logs` / `current` / `search_begin_input` |
| 探索用完全状態 | `cg/api.py:448` `SearchState(observation, searchId)` | **状態複製 API は無い** |
| 状態 | `cg/api.py:367` `State` | `turn` / `yourIndex` / `result` / `players[2]` / 各種フラグ |
| 勝利判定 | `State.result == State.yourIndex`（`-1` は継続） | `lethal_simple.py:313` が同じ判定 |
| ターン終了判定 | ステップ後 `State.yourIndex != me` | `lethal_simple.py:316` |
| 探索 API | `cg/api.py:517` `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False)` / `:597` `search_step(search_id, select)` / `:629` `search_end()` / `:633` `search_release(search_id)` | `search_begin` は `obs.search_begin_input` 必須 |
| 非公開情報の構築 | `ptcg_ai/hidden_information/search_state_stub.py:91` `build_dummy_search_state()` | 自分側は厳密（デッキリスト − 可視カード）、相手側はダミー |
| critic | `ptcg_ai/learning/value_model.py:105` `ValueModel.predict_win_prob(obs)` | 重み `value_weights.json` 同梱 |
| critic 特徴量 | `ptcg_ai/learning/encoder.py:285` | `state.yourIndex` を基準に自分/相手を並べ替え |
| critic ラベル定義 | `kaggle_replays/extract_value_dataset.py:105` | 「その観測の持ち主が勝ったか」= **現在プレイヤー視点** |
| 通常方策 | `ptcg_ai/action_selection/router.py` `route(obs)` | ルールベース |
| 相手デッキ予測 | `ptcg_ai/opponent_modeling/tracker.py:29` `update(obs)` | `update()` の呼び出しは `selector.py:70` のみ |
| 合法性チェック | `selector.py:36` `is_valid_action()` | 提出契約と同じ検査 |
| 設定 | `configs/rule_lethal.json` / `ptcg_ai/core/config.py:39` `load_config()` | 既定名 `rule_lethal` |

### 1.2 API シグネチャ上の重要事項（推測ではなくコードから）

- `search_begin()` は `obs.select.deck != None` のとき **`your_deck` を破棄して実物のデッキを使う**
  （`cg/api.py:553-554`）。← B1 の根拠。§4.5 で実測。
- `search_begin(manual_coin=True)` を渡すとコインの表裏を選べる（`cg/api.py:536`）。
- `SelectContext.COIN_HEAD = 46`（`cg/api.py:120` 付近）。`YES` を選ぶと表。
- `LogType.DRAW` は `cardId` / `serial` を持つ（自分のドローは中身が見える）。
  `LogType.SHUFFLE` / `LogType.COIN(head)` も個別のログ種別。
- `Log.type` などの enum フィールドは `to_dataclass` で **int のまま**返ることがある
  （`IntEnum` 比較は成立するが `.name` は使えない）。実装時の落とし穴。

---

## 2. 測定条件

| 項目 | 値 |
|---|---|
| OS / Python | Windows 11 (10.0.26200) / CPython **3.14.6** |
| エンジン | `sample_submission/cg/cg.dll`（リポジトリ同梱） |
| デッキ | `sample_submission/deck.csv`（フーディン。両プレイヤー同一） |
| 局面コーパス A | ランダム対戦 2 試合から抽出した**自分の MAIN 選択 60 件**（`turn >= 3`）→ `positions.jsonl` |
| 局面コーパス B | ランダム対戦 4 試合から抽出した **`select.deck != None` の局面 40 件** → `deck_positions.jsonl` |
| 隠れ情報 | `build_dummy_search_state(obs, deck, rng=Random(seed))`（自分側は厳密、相手側はダミー） |
| 実行形態 | 対戦本体とは**別プロセス**でリプレイ（`search_begin_input` のみ使用）。§4.5 と §4.7 のみ実対戦内で実行 |
| 計測 | `time.perf_counter()`。critic はウォームアップ後 |

**コーパスの性質に関する注意**: 局面はランダムプレイから採取したので、
**リーサル局面の分布を代表していない**。本書の数値は「エンジンの機構」に関するものであり、
「リーサル発見率」に関する主張には使えない（発見率の基準値は姉妹文書の 78 局面ベンチ）。

---

## 3. 再現手順

プローブは全てスクラッチパッドに置いてある（本実装側へは移していない）。

```
C:\Users\USER\AppData\Local\Temp\claude\C--Users-USER-lab----PTCG-AI-short-search-pokemon-tcg-ai-battle\883291fb-44d0-471c-9115-2930d50a8bd3\scratchpad\
```

実行は必ず `sample_submission/` をカレントディレクトリにする（`deck.csv` / `configs/` の解決のため）。

```bash
cd sample_submission && python <scratchpad>/collect_positions.py
```

| プローブ | 目的 | 出力の要点 |
|---|---|---|
| `collect_positions.py` | コーパス A 生成 | `positions.jsonl` |
| `probe_capabilities.py` | 山札順・再現性・step コスト・manual_coin 受理 | §4.1 / §4.6 |
| `probe_a_deck_order.py` | 山札順のテストケース A1〜A4 | §4.1 |
| `probe_shuffle.py` | シャッフル後の非決定性 | §4.3 |
| `probe_uniform2.py` | シャッフル後分布の χ² 検定 | §4.3 |
| `probe_b_coin.py` / `b2` / `b3` / `b5` | コインの可制御性 | §4.2 |
| `probe_c_nondet.py` | 非決定性が SHUFFLE 以外で起きるか | §4.4 |
| `probe_e_leak_live.py` | 実対戦での漏洩テスト + コーパス B 生成 | §4.5 |
| `probe_e2_b1_trace.py` | **B1** デッキサーチの追跡 | §4.5 |
| `probe_f_qbase.py` | 通常方策ロールアウトの安全性 | §4.7 |
| `probe_g_latency.py` | 現行エージェントの実測レイテンシ | §4.8 |
| `probe_dist_and_critic.py` / `probe_turnend.py` | critic の可用性・視点・ターン終了時の見え方 | §4.6 |

---

## 4. 実測結果

### 4.1 `your_deck` の順序がドロー順を決めるか（原設計 §5.2 / ユーザ指摘 1）

**山札の一番上 = `your_deck` の末尾**。ドローは末尾から逆順に消費される。

`probe_a_deck_order.py`（コーパス A 60 件。うち深さ1でドローが起きる 18 件が対象。
残り 42 件は深さ1にドロー選択肢が無く対象外）:

| ID | テストケース | 結果 |
|---|---|---|
| A1 | 任意のカード X を末尾に置くと、最初に引かれるのは X（1 局面につき 3 種類の X） | **54 / 54 PASS** |
| A2 | 末尾 k 枚を固定したまま先頭側だけを 3 通りにシャッフルしても、引く k 枚は不変 | **54 / 54 PASS** |
| A3 | k 枚ドローの結果が `reversed(your_deck[-k:])` と完全一致 | **18 / 18 PASS** |
| A4 | 引いたカードは供給プールの部分 multiset である | **18 / 18 PASS** |

補助測定（`probe_capabilities.py`）:

| 検証 | 結果 |
|---|---|
| 同一 `your_deck` で 2 回実行 → 同一結果 | 18 / 18 |
| `your_deck` を逆順にすると引くカードが変わる | 18 / 18 |

実例: 供給末尾 `[…, 5, 1086, 1264, 1081]` → 引いた順 `[1081, 1264, 1086]`。

> **言えること**: シャッフルが介在しない限り、我々が供給した順序が draw を完全に決める。
> よって「特定の outcome を実現する順列」を作って `search_begin` し直せば任意 outcome を強制できる。
> **言えないこと**: これはこのデッキ・この 18 局面・深さ1のドローに関する観測である。
> 深い位置のドロー、他デッキ、特殊なドロー効果（「相手が引く」「デッキの下から」等）は未検証（§6）。

### 4.2 コインの可制御性（クラス M / ユーザ指摘 2）

- `manual_coin=True` で `SelectContext.COIN_HEAD`（YES/NO, `minCount=maxCount=1`）が挿入される。
- **自分のターン中のコインは `state.yourIndex == me` で提示される**
  （`probe_b5`: Dunsparce の Dig(attack 75) を強制した 3 局面すべてで
  `yourIndex=0, result=-1, turn=5/7/5`。YES→`COIN:H`、NO→`COIN:T`）。
- 60 局面 × 20 ロールアウト（両ターンにまたがる）での集計:

| 設定 | 結果 |
|---|---|
| `manual_coin=False` | COIN ログ 348（表 180 / 裏 168） |
| `manual_coin=True` + 常に YES | COIN ログ 345 が**全て表** |
| `manual_coin=True` + 常に NO | COIN ログ 368 が**全て裏** |

**当初「193 select に対し 358 coin ログ」で 1:1 でないと見えたのは誤読だった**。
`Observation.logs` は「その観測を受け取るプレイヤーの前回選択以降」の履歴なので、
相手番の選択まで進めると**同じコインが両プレイヤーのログに二重計上**される。
自分のターン内に限ったトレース（`probe_b5`）では **1 コイン = 1 select**。

> **言えること**: このデッキで実際に出るコイン（Dunsparce「Dig」1 種）は、
> 自分のターン中に 1 フリップ 1 select として提示され、表・裏を強制できる。
> **言えないこと**: 「2 枚以上のコインを同時に投げる効果」はこのデッキに存在せず**未検証**。
> 1 回の select が複数フリップをまとめて決める実装だと、混合 outcome（表・裏の組合せ）が
> 到達不能になり、多コイン効果の完全列挙は破綻する。→ §6 / §8 の未確認事項。

### 4.3 シャッフル後のドロー（クラス S）

`probe_shuffle.py`: 経路上に `LogType.SHUFFLE` があると、**同一入力でも結果が変わる**（3 / 3 で不一致）。
= エンジン内部 RNG。`search_begin` の入力では再現できない。

`probe_uniform2.py`: 1 局面・同一経路を 800 回再生し、シャッフル直後に引く 1 枚の分布を、
ログから再構成した山札 multiset（44 枚 / 22 種）と比較:

```
χ² = 19.2, dof = 21  →  χ²/dof = 0.91（p ≈ 0.57）
```

> **言えること**: この 1 局面・800 サンプルでは、既知 multiset の一様抽出と矛盾する証拠は無い。
> **言えないこと**: これはエンジンの確率モデルの証明では**ない**。
> 局面数 1、デッキ 1、事象 1 種類（シャッフル直後の 1 枚）に過ぎず、
> 一様性を仮定した確率計算をここから正当化してはならない。設計では
> 「クラス S は**エンジンにサンプリングさせる**（我々が分布を仮定しない）」方針を取る（§8）。

### 4.4 `SHUFFLE` ログだけで S と判定できるか（ユーザ指摘 2）

`probe_c_nondet.py`: ランダム経路の**全プレフィックス**を 2 回ずつ再生して一致を見る
（コーパス A の先頭 30 局面、深さ ≤ 8、457 プレフィックス × 2 モード）。

| 分類 | `manual_coin=False` | `manual_coin=True` |
|---|---|---|
| 決定的 / SHUFFLE 無し | 231 | 231 |
| 決定的 / SHUFFLE 有り | 168 | 169 |
| **非決定的** / SHUFFLE 有り | 57 | 57 |
| 非決定的 / SHUFFLE + COIN | 1 | 0 |
| **非決定的 / SHUFFLE も COIN も無し** | **0** | **0** |

> **言えること**: 観測された非決定性は全て SHUFFLE（と `manual_coin=False` 時のコイン）を含む経路で起きた。
> また SHUFFLE があっても決定的なまま（168 件）のことも多いので、
> SHUFFLE ログは**保守的な（過剰検出の）マーカー**として使える。
> **言えないこと**: 「SHUFFLE 以外に乱数源が無い」ことは証明されていない。
> 457 プレフィックス・1 デッキ・深さ 8 の観測に過ぎず、
> 「相手の手札からランダムに 1 枚」「デッキの上を公開」等の未使用効果は検証範囲外。
> → 設計では **推論せず検証する**: Phase 2 でクラス C を主張する前に、
> その chance ノードを 2 回再生して同一結果であることを必ず確認する（§8 の規則 R3）。

### 4.5 B1: `select.deck != None` 局面での実デッキ使用（最優先項目）

#### (a) 実対戦での end-to-end 漏洩テスト（`probe_e_leak_live.py`）

実対戦 4 試合の各自分手番で、「これから実際に打つ行動」を**先に探索側で試し**、
探索が予測したドローと、その後の**実対戦での実際のドロー**を突き合わせた。

| 指標 | 結果 |
|---|---|
| 自分のドローが発生した意思決定（`select.deck == None`） | 19 件 |
| 探索の予測が実際と **cardId 一致** | **0 / 19** |
| 探索の予測が実際と **serial 一致** | **0 / 19** |
| 別 seed の捏造シナリオが実際と一致（チャンスレベル） | 0 / 19 |

→ 通常局面では、探索は実際の山札順を一切知らない。

#### (b) デッキ公開局面の追跡（`probe_e2_b1_trace.py`、コーパス B）

`select.deck != None` の 14 局面について、ユーザ指定の 4 状態を追跡した。

| 追跡点 | 結果 |
|---|---|
| (1) サーチ前: `your_deck` は本当に無視されるか | **14 / 14 無視**（意図的に誤ったデッキを渡しても、公開される listing は実物と同一） |
| (2) 特定カード取得直後 | `MOVE_CARD` が記録される |
| (3) SHUFFLE 発生 | 取得と同じ step で 9 / 14。残り 5 件は選択が続く効果（複数枚選択・ベンチ配置）で、後続 step で発生 |
| (4) SHUFFLE 後のドロー | **観測された後続ドロー 430 サンプルは全て SHUFFLE 後**（SHUFFLE 前のドローは 0 件）。結果は**非決定的**（追跡した局面では 200 サンプル中 199 通り）。公開されていた listing の先頭を引く回数は 16 回で、一様期待値 14.3 と同程度 |

> **言えること**: デッキが公開された状態から探索を開始しても、
> 実際の山札順を使って後続のドローを当てることはできなかった。
> 公開 → 取得 → **シャッフル** → 以後はクラス S、という順序が 14 局面すべてで成立した。
> **言えないこと**: 「デッキサーチ後には必ずシャッフルが入る」ことは**ゲームルール上の期待であって、
> エンジン実装の保証ではない**。430 サンプル・1 デッキの観測にすぎない。
> → 設計では**実行時ガード**を必須にする（§8 の規則 R4）:
> 「デッキが公開された経路で、SHUFFLE を挟まずにドローが発生したら、その枝は
> `UNSUPPORTED_EFFECT` として棄却する」。検証ではなく仮定に頼らない。

### 4.6 critic（`ValueModel`）

| 項目 | 実測／根拠 |
|---|---|
| 可用性 | `is_ready == True`（`value_weights.json` 同梱） |
| レイテンシ | **1.30 ms/call**（ウォーム、200 回平均）。初回のみ約 31 ms（重みロード） |
| 探索中 Observation の受理 | 可（`predict_win_prob(node.observation)`） |
| 視点 | **現在プレイヤー視点**。特徴量は `encoder.py:285` で `state.yourIndex` を基準に並べ替え、学習ラベルは `extract_value_dataset.py:105` で「その観測の持ち主の勝敗」 |
| ターン終了後の見え方 | END 後は `yourIndex` が相手に移り、`players[me].hand` が `None`、相手手札は**我々が捏造したダミー**が見える |
| 参考値 | 同一局面で `V(END前, 自視点)=0.564` に対し `1 - V(END後)=0.665`（**同じ局面ではないので一致する必要は無いが、規約を混ぜると 0.1 規模の差になる**） |

> **言えないこと**: Phase 3 が生成する「リーサル失敗直後の局面」での校正は**未測定**。
> AUC 等はオフラインの実戦局面に対する値であり、探索が作る分布での妥当性は不明。

### 4.7 通常方策のロールアウト（`Q_base` の前提 / ユーザ指摘 3）

`probe_f_qbase.py`: コーパス A 60 局面 × 捏造シナリオ 5 通り = **300 ロールアウト**
（`router.route()` で自分のターンが終わるまで進め、最後に critic を 1 回呼ぶ）。

| 指標 | 結果 |
|---|---|
| `router.route()` が `None` を返した | 0 |
| 契約違反の行動（個数・重複・範囲） | 0 |
| エンジンが拒否（`ValueError`） | 0 |
| 例外 | 0 |
| 60 step 上限に到達 | 0 |
| 終了理由 | ターン終了 290 / 決着 10 |
| ステップ数 | 平均 9.8 / p95 19 / 最大 27 |
| 所要時間（`search_begin` + ロールアウト + critic 1 回） | 平均 **6.5 ms** / p95 11.2 ms / 最大 48.9 ms |
| **相手モデル tracker の状態変化** | **無し**（ロールアウト前後で内部状態のフィンガープリントが一致） |

コード上の裏付け:
- `tracker.update()` の呼び出し箇所は `selector.py:70` **のみ**。
  `router` 配下は `current_matchup_plan()`（読み取り）しか呼ばない。
- `router.route(obs)` は `Observation` 以外の入力を取らない。探索中 Observation では
  相手手札は `None`（実戦と同じ形）で、我々の捏造した相手デッキ内容は見えない。

> **言えること**: 300 ロールアウトの範囲では、通常方策を探索内で回しても
> 違法手・例外・グローバル状態の汚染は発生しなかった。
> **言えないこと**: これは「一般に安全」の証明ではない。300 サンプル・1 デッキ・
> ランダムプレイ由来の局面分布での観測にすぎない。
> → 設計では `Q_base` ロールアウトに**必ず**「合法性チェック・step 上限・例外捕捉・
> 失敗時は `Q_base` を `UNAVAILABLE` にしてゲートを通さない」を入れる（§8 の規則 R6）。

### 4.8 コスト・レイテンシ

| 項目 | 実測 |
|---|---|
| `search_step` | **0.316 ms**（n=732, p95 0.525 ms） |
| `search_begin` | **0.436 ms**（n=120） |
| `ValueModel.predict_win_prob` | **1.30 ms** |
| 通常方策ロールアウト（begin + 9.8 step + critic） | **6.5 ms**（p95 11.2） |
| outcome 1 個の適用コスト（深さ d） | `≈ 0.44 + 0.32 d` ms |

**現行エージェントの実測（`probe_g_latency.py`、`configs/rule_lethal.json`）**

| 条件 | 結果 |
|---|---|
| 自分 vs ランダム 4 試合 | 自分の意思決定 72〜88 回/試合、**エージェント総時間 1.80〜2.28 秒/試合** |
| 1 意思決定あたり | 平均 25.8 ms / p50 25.0 / p95 30.1 / **p99 38.1** / 最大 120.9 |
| リーサルゲート発火 | 0 回（この 4 試合では残りサイド ≤2 に到達せず） |
| 自分 vs 自分 3 試合（同一プロセス、両者ルールベース） | 両者合計 156〜186 意思決定、総時間 4.19〜5.37 秒/試合 |
| 同上・ゲート発火 | 1 試合目のみ 16 回（発見 2 / タイムアウト 13 / 平均 82.5 ms）、他 2 試合は 0 回 |

> **言えること**: この環境では、現行エージェントは 1 試合あたり約 2〜3 秒（片側）しか消費しておらず、
> 公式の 10 分（600 秒）に対して 0.5% 未満である。
> **言えないこと**: Kaggle 実行環境の CPU 性能・1 手あたりの上限・プロセスの永続性は**未確認**（§7 B2）。
> ローカルの余裕をそのまま提出時の予算に読み替えてはいけない。

---

## 5. 確認できたこと（設計に使える確定事実）

| # | 事実 | 根拠 |
|---|---|---|
| C1 | 山札の一番上は `your_deck` の末尾。シャッフルが無い限りドローは供給順で完全に決まる | §4.1（A1〜A4 全 PASS） |
| C2 | 任意 outcome の適用は「順列を作って `search_begin` し直す」で可能。コストは `0.44 + 0.32d` ms | §4.1 / §4.8 |
| C3 | `manual_coin=True` で、自分のターン中のコインを `yourIndex == me` の select として表裏指定できる（単発コインについて） | §4.2 |
| C4 | シャッフル後のドローはエンジン内部 RNG で、入力からは再現できない | §4.3 |
| C5 | 観測された非決定性は全て SHUFFLE / COIN を含む経路で発生（457 プレフィックス） | §4.4 |
| C6 | 実対戦の漏洩テストで、探索の予測ドローは実際と 0/19 一致（漏洩の証拠なし） | §4.5(a) |
| C7 | デッキ公開局面では `your_deck` は無視される（14/14）が、後続ドローは全て SHUFFLE 後で非決定的（430 サンプル） | §4.5(b) |
| C8 | critic は利用可能・現在プレイヤー視点・1.3 ms/call | §4.6 |
| C9 | 通常方策は探索内で安全に回せた（300 / 300 成功）。グローバル状態を汚さない | §4.7 |
| C10 | 現行エージェントの消費は 2〜3 秒/試合、p99 38 ms/手（ローカル環境） | §4.8 |

---

## 6. 未確認事項（設計で仮定してはいけないこと）

| # | 未確認 | なぜ重要か | 扱い |
|---|---|---|---|
| U1 | 多コイン効果（「コインを2回投げる」）で 1 フリップ 1 select になるか | 混合 outcome が到達不能なら多コイン効果の完全列挙が破綻する | 現デッキに存在しない。検出したら Phase 2 は `UNSUPPORTED_EFFECT` |
| U2 | SHUFFLE 以外の乱数源の有無（相手手札からランダム、デッキ上公開 等） | クラス分類の健全性 | **推論しない**。実行時の再生一致検査で判定（規則 R3） |
| U3 | シャッフルの分布が厳密に一様か | クラス S の確率計算 | 我々は分布を仮定せず、エンジンにサンプリングさせる（規則 R5） |
| U4 | サイド（`your_prize`）の取得が供給順で決まるか | Phase 3 の失敗枝で手札が変わる | 300 ロールアウトでサイド取得 0 件 → 未測定。Step 1 で計測 |
| U5 | 自ターン中に相手が選ぶ効果の実例・頻度 | Phase 2 の AND ノード実装要否 | Step 1 で `yourIndex != me` かつターン継続の発生をカウント |
| U6 | 深い位置のドロー・特殊ドローでも C1 が成り立つか | 完全列挙の適用範囲 | Step 1 で深さ別に A1〜A3 を再測定 |
| U7 | 他デッキ（相手アーキタイプ、将来の自デッキ変更）での挙動 | 一般性 | Step 1 で最低 2 デッキに拡張 |
| U8 | Kaggle 実行環境の 1 手上限・CPU 性能・プロセス永続性 | 予算設計の全前提 | §7 B2。提出前は保守的値を使う |
| U9 | critic の失敗葉での校正 | Phase 3 ゲートの妥当性 | §7 B3。校正するまで Phase 3 は既定 OFF |

---

## 7. ブロッカー（B1〜B7 の現状）

| ID | 内容 | 状態 | 根拠 / 次のアクション |
|---|---|---|---|
| **B1** | `select.deck != None` で `search_begin` が実デッキを使う（`cg/api.py:553`）＝ 実順序の漏洩懸念 | **緩和済み（条件付き）** | §4.5: `your_deck` 無視 14/14 を確認。ただし後続ドロー 430 サンプルは全て SHUFFLE 後・非決定的で、実順序は利用できなかった。**保証ではない**ため、実行時ガード（規則 R4）を設計に組み込み、Step 1 で「SHUFFLE を挟まないドロー」の発生 0 件を回帰テスト化する |
| **B2** | Kaggle の 1 手あたり時間上限・CPU 性能・プロセス永続性が不明 | **未解決** | 提出するまで確認できない。予算は「まず計測 → 段階的に引き上げ」に変更（§8 の予算方針）。既定値を先に決めない |
| **B3** | critic が Phase 3 失敗葉で校正されているか不明 | **未解決** | §4.6。Phase 3 は既定 OFF のまま。校正データセット作成は Step 5 以降 |
| **B4** | 自ターン中の相手選択効果の実例・頻度が未計測 | **未解決（計測待ち）** | U5。Step 1 で self-play にカウンタを入れる |
| **B5** | outcome 粗視化の健全性はデッキ依存 | **未解決（設計で回避）** | 既定は恒等写像（厳密列挙）。粗視化は opt-in + oracle テスト必須 |
| **B6** | `attack_features.resolve_damage()` がダメカン配置に弱点/抵抗を適用 | **未解決（影響限定）** | 姉妹文書 §1.4 の指摘。Goal は探索**順序**にのみ使い、枝刈りに使わないので確定性には影響しない |
| **B7** | Transformer value head の推論経路が未整備 | **未解決（対象外）** | `ValueEvaluator` 契約で差し替え可能にするのみ。今回は使わない |

**新規に判明したブロッカー候補**

| ID | 内容 | 状態 |
|---|---|---|
| B8 | 多コイン効果での select 粒度（U1） | 未検証。検出時は `UNSUPPORTED_EFFECT` |
| B9 | サイド取得の可制御性（U4） | 未検証。Phase 3 の失敗枝の精度に影響（勝敗判定自体には影響しない） |

---

## 8. 設計書への影響（`design.md` の改訂点）

ユーザ指摘 1〜5 を反映し、以下を `design.md` に反映した（詳細は同書）。

### 8.1 カンニング（determinization）の防止 — 具体化

| 規則 | 内容 |
|---|---|
| **R1** | 決定ノードのキー `InfoKey` は「公開情報の指紋 + 自分の手札 multiset + 自分の山札 **multiset と枚数** + 解決中の効果と選択段階」。**シナリオ（山札順列）・実順序・相手の非公開手札は含めない** |
| **R2** | `InfoKey` は 1 箇所（`infokey.py`）でのみ生成し、探索・transposition・特徴量・critic 入力はすべてそれを経由する。山札順列は `outcome.py` の内部にとどめ、他モジュールへ渡さない（型で分離: `Scenario` は `OutcomeApplier` の外に出さない） |
| **R3** | クラス C を主張する前に、その chance ステップを **2 回再生して同一結果**であることを確認する（SHUFFLE ログの有無だけで判定しない）。不一致ならクラス S |
| **R4** | デッキが公開された経路（`select.deck != None` を通過）で、SHUFFLE を挟まずにドローが起きたら `UNSUPPORTED_EFFECT` で棄却する |
| **R5** | クラス S は**我々が分布を仮定せず**、同一プレフィックスの再生でエンジンにサンプリングさせる。`EstimateKind.SAMPLED` を必ず付ける |
| **R6** | `Q_base` / `Q_try` のロールアウトは、合法性チェック・step 上限・例外捕捉付き。1 つでも失敗したらその候補の価値を `UNAVAILABLE` にし、採用ゲートを通さない |
| **R7** | critic 入力は葉の Observation のみ。`yourIndex != me` の葉では `1 - p` を使い、`Q_try` と `Q_base` で**同一規約**を適用する |

### 8.2 完全列挙できるケースと、できないケースの分離

| 区分 | 条件（すべて満たすこと） | Phase 2 | Phase 3 |
|---|---|---|---|
| **列挙可能** | ① 経路上で SHUFFLE 未発生 ② デッキ未公開で開始 ③ 再生一致検査（R3）に合格 ④ outcome が自分の山札 multiset から決まる ⑤ コインは単発で 1 select 1 フリップ | 全 outcome 列挙で `PROVEN_WIN` を主張可 | 厳密重み（`Fraction`）で `EXACT` |
| **列挙不可** | 上のいずれかを満たさない（SHUFFLE 後 / デッキ公開開始 / 再生不一致 / 多コイン / 相手選択） | `UNKNOWN`（`SHUFFLE_ENCOUNTERED` 等の理由付き） | サンプリング or 上下界。`SAMPLED` / `BOUNDED` |

### 8.3 予算方針の変更（ユーザ指摘 4）

- 「理論上 1 手 4〜5 秒」は**上限の余地**であり、推奨値ではない。**700 ms という既定値は撤回**。
- 手順: ①現行値（Phase 1・2 で 100 ms）で新実装を動かし、実測 → ②`p95/p99` と 1 試合総時間を見て段階的に引き上げ → ③A/B で効果が確認できた値のみ採用。
- **Phase 3 の時間予算は Phase 1・2 と独立した設定**にし、Phase 3 が Phase 1・2 の予算を奪えない構造にする
  （`phase12_ms` と `phase3_ms` を別キーにし、Phase 3 は「Phase 1・2 が終わった後の残り」からのみ取る）。
- 測定項目を設計に明記: 1 試合の総エージェント時間、1 手平均、p50/p95/p99、ゲート発火回数、
  ゲート 1 回あたりの時間、Phase 別内訳。ベースラインは §4.8 の実測値。

### 8.4 「隠れた山札順だけを変えたら root action が一致する」テストの具体化

```
test_root_action_invariant_to_hidden_deck_order:
  与えられた実局面 obs に対し
    for seed in 0..N-1:
        hs = build_dummy_search_state(obs, deck, rng=Random(seed))
        # hs["your_deck"] は同一 multiset の異なる順列（前提として assert する）
        r[seed] = run_search(obs, hs)      # Phase 1/2/3 それぞれ
  assert すべての seed で
      r[seed].first_action        が一致
      r[seed].proof               が一致
      r[seed].p_lethal (下限・上限) が一致（クラス S を含まない局面では厳密一致）
  ただしクラス S を含む局面では、p_lethal はサンプリング誤差の範囲内での一致に緩める
  （その場合も first_action と proof は厳密一致を要求する）
```

- 対象局面: 姉妹文書の 78 局面コーパス（`select.effect` 修正後は全 78 件が探索可能）。
- **この不変性テストは Phase 1 の実装より先に書く**（原設計 §18.2 の趣旨）。
- 併せて `InfoKey` の単体テスト（山札順を変えてもキーが不変・手札を変えるとキーが変わる）を置く。

---

## 9. Step 1 で検証すべき項目（実装着手前のチェックリスト）

| # | 項目 | 合格条件 | 対応する未確認/ブロッカー |
|---|---|---|---|
| S1-1 | プローブをリポジトリへ移設（`tools/lethal_bench/`）し、本書の全数値を再現 | 数値が本書と一致 | — |
| S1-2 | 姉妹文書の 78 局面ベンチを再現（`lethal_simple` 発見 8/58・timeout 76%・平均 81 ms） | ベースライン固定 | — |
| S1-3 | `search_state_stub` の `select.effect` 修正の回帰テスト | 78 局面すべてで hidden state が構築できる | — |
| S1-4 | **B1 ガード**: 「デッキ公開経路で SHUFFLE を挟まないドロー」の発生数 | self-play 全局面で 0 件。1 件でもあれば設計変更 | B1 |
| S1-5 | 深さ別・複数デッキでの A1〜A3 再測定 | 深さ 1〜6、2 デッキ以上で全 PASS | U6, U7 |
| S1-6 | 多コイン効果の select 粒度 | 該当カードを含むデッキで 1 フリップ 1 select を確認、または `UNSUPPORTED` 判定 | U1, B8 |
| S1-7 | サイド取得の可制御性 | `your_prize` の順序変更で取得カードが変わるか | U4, B9 |
| S1-8 | 自ターン中の相手選択効果の発生カウント | self-play で件数を得る（0 なら実装を後回しにできる） | U5, B4 |
| S1-9 | 非決定性の再検査を深さ 12 以上・複数デッキへ拡張 | 「SHUFFLE/COIN 無しの非決定性」が 0 件 | U2 |
| S1-10 | `router.route()` ロールアウトを 78 局面 × 20 シナリオへ拡大 | 違法手・例外 0。1 件でも出たら R6 のフォールバック経路をテストで固定 | §4.7 |
| S1-11 | critic の視点規約テスト（`yourIndex != me` の葉で `1-p`） | `Q_try` と `Q_base` が同一規約を通ることをテストで固定 | B3 の前段 |
| S1-12 | レイテンシ計測ハーネス（1 試合総時間・p50/p95/p99・ゲート発火・Phase 別内訳） | 本書 §4.8 の形式で継続測定できる | B2 |

---

## 10. 本書の限界

- 全測定は **1 デッキ（フーディン）・1 環境・ランダムプレイ由来の局面**に基づく。
- 「N 件で反例が出なかった」は「起こり得ない」ではない。
  設計は反例が出た場合に**安全側へ倒れる**（`UNKNOWN` / `UNSUPPORTED_EFFECT` / 通常方策へ fallback）
  構造にしてあり、確率的な観測の上に確定性を主張しない。
- 実装コードは本 Step で一切変更していない。プローブはスクラッチパッドにあり、
  リポジトリへの移設は Step 1 の作業とする。

---

## 11. RNG Control Capability（Step 1-16 で追加・正式な制約）

エンジンの乱数制御能力。**設計で仮定してはいけない事項の筆頭**として扱う。
測定は `scratchpad/rng_interference.py` / `scratchpad/rng_probe.py`、
固定テストは `tests/integration/lethal/test_rng_noninterference.py`。

### 11.1 確認済み（測定に基づく事実）

| 能力 | 状態 | 根拠 |
|---|---|---|
| seed injection | **unavailable** | `cg.dll` の export は 13 個のみ。`BattleStart` の引数はデッキ 120 枚だけ |
| RNG state read | **unavailable** | 該当 export 無し |
| RNG state write | **unavailable** | 該当 export 無し |
| battle state clone | **unavailable** | 該当 export 無し |
| reproducible `BattleStart` | **unavailable** | 同一デッキ・同一固定方策で 4 対局 → 4 通り。別プロセスの「最初の 1 戦」も 4 回とも別 |
| `SearchBegin`/`SearchStep` の対象 | `Select` と**同じ `battle_ptr`** | `cg/sim.py` の argtypes、`cg/api.py` の実装 |
| live state の直接変更 | **観測されず 0/434** | 探索直前・直後の `GetBattleData` 比較 |
| 非 shuffle 探索の決定性 | **決定的 179/179** | 同一 live state・同一入力で 3 回反復 |
| shuffle を含む探索の再現性 | **検証できない** | 同条件で 83/255 が不一致。入力で決まらない乱数源がある |

`cg.dll` の全 export（PE export table を直接解析）:

```
GameInitialize BattleStart BattleFinish GetBattleData Select AgentStart
VisualizeData AllCard AllAttack SearchBegin SearchStep SearchEnd SearchRelease
```

### 11.2 未確認（**API が無いため検証不能**）

以下は「調べていない」のではなく、**調べる手段が存在しない**。

- 探索が native RNG stream を消費するか
- 探索が本番の random stream へ影響するか

RNG 状態を読む API も seed を固定する API も無いため、軌跡比較でも分布比較でも判別できない。

### 11.3 production safety implication

必須条件

```
Exploration RNG Non-Interference
  = search の有無だけで本番 random outcome が変化しない
```

は**検証不能**。したがって:

```
RNG_NON_INTERFERENCE_VERIFIED = False
Phase 1 production = NOT SAFE TO ENABLE
```

**重要な区別**（誤読を防ぐため明記する）:

| 分類 | 定義 | 本件 |
|---|---|---|
| Confirmed unsafe | 実際に不正行動・状態破壊が確認された | **該当しない**（illegal 0 / 状態破壊 0/434 / false PROVEN_WIN 0） |
| **Not verified** | 安全を証明する SDK/API が無く検証できない | **これに該当** |

したがって「RNG interference が発生することが分かった」とは書かない。正しくは
**「RNG non-interference を検証するための seed/state control が存在せず、本番使用条件を満たせない」**。

### 11.4 再評価の条件

`RNG_NON_INTERFERENCE_VERIFIED` は永久に `False` と決め打ちしない。
以下のいずれかが起きたら再評価する。`test_engine_has_no_seed_or_clone_entry_point` が
export の増減を検出して落ちるので、変化には自動的に気づける。

- SDK が更新され export が増える
- state snapshot API が追加される
- deterministic battle mode / seed 指定が追加される
- 別プロセスでの完全複製可能な simulation instance が提供される

### 11.5 評価手法への制約（恒久）

seed control / state clone が無い限り、**paired self-play を評価方法として使わない**。
使用可能なのは以下だが、いずれも「対応する random trajectory が同一」とは主張できない。

- 独立試合の統計比較（非対応 2 標本）
- 大規模 sample
- 事前に固定した評価データ（保存盤面）
- opponent / deck を固定し config のみを変える

---

## 12. B8（コイン）の capability 記録

```
fixed-N coin:
    engine representation confirmed
    implementation possible
    current deck benefit unmeasured
    implementation deferred

unbounded coin:
    unsupported / UNKNOWN
```

現在の本番方策には接続しない。

---

## 13. B2（時間制約）の現状

| 項目 | 状態 |
|---|---|
| 1 試合 10 分 | **確認済み** |
| 1 手あたりの上限 | **未確認**（外部仕様の確認待ち） |
| process persistence | **未確認**（同上） |

リポジトリ内部の推測で決めない。なお Phase 1 は既に RNG 非干渉を満たせず NOT SAFE なので、
**B2 の確認をもって本番 ON へ進む理由にはならない**。

---

## 14. Outcome Branching Limitation（Step 1-32 で確定・engine API の制約）

```
A chance outcome can only be materialized by:
    SearchBegin(your_deck=<specific order>) + prefix replay

No:
    state clone
    snapshot / restore
    outcome injection
    mid-search deck replacement
    shared-prefix branching
    concurrent sessions (agent_ptr is shared)
```

**これは現 solver の未実装ではなく engine API limitation である。**

根拠（実測）:

| 確認項目 | 結果 |
|---|---|
| `search_step(search_id, select)` の引数 | 選択肢インデックスのみ。**引くカードを指定できない** |
| outcome を制御する唯一の入口 | `search_begin` の `your_deck` の並び |
| 同一 parent からの複数 child | **可能**（3 回連続成功、parent 生存）。ただし引くカードは指定不可 |
| 複数 session の同時保持 | **不可**（`SessionNestingError: agent_ptr is shared`） |
| state clone / snapshot / RNG 制御 | export 13 個に該当なし |

コスト分解（中央値）:

| Component | ms |
|---|---|
| `SearchBegin` + セッション初期化/解放 | **0.243** |
| `search_step` 1 回 | 0.093 |

`prefix 長 k ≈ 0.243 + k × 0.093 ms`。k=3 で **`SearchBegin` が 46%** を占める。
prefix を共有できれば 4〜5 倍の改善になるが、上記のとおり不可能。

### 14.1 将来 SDK へ要望する場合の優先順位

| 優先 | 要望 | これがあると何が改善するか |
|---|---|---|
| **1** | state clone / snapshot & restore | outcome ごとの `SearchBegin` + prefix replay が不要になり、materialization が 4〜5 倍高速化。`max_outcomes` を上げても総 WIN が減らなくなる |
| **2** | mid-search deterministic outcome injection | 「次に引くカード」を `search_step` で指定できれば、共通 prefix から直接 outcome へ分岐できる |
| **3** | shared-prefix branching | 1 と 2 の一部でも代替可能。共通 prefix を 1 回だけ実行できれば足りる |
| **4** | RNG seed / state control | `RNG_NON_INTERFERENCE_VERIFIED` を検証可能にし、paired A/B が成立する。Phase 1/2 の本番評価の前提条件 |
