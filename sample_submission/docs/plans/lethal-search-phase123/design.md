# リーサル探索 Phase 1 / 2 / 3 設計書

- 対象ブランチ: `integration`（HEAD `79eca91`）
- 元仕様: `ptcg_lethal_search_phase1_2_3_claude_design.md`（以下「原設計」）
- 本書の状態: **Step 0 完了（第2版・追加検証反映済み）。実装は未着手。**
- **測定の詳細・限界・再現手順は [`step0-capability-report.md`](./step0-capability-report.md) を参照**。
  本書 §1 はその要約であり、数値の一次情報は capability report 側にある。
- 姉妹文書: [`../lethal-macro-search/design.md`](../lethal-macro-search/design.md)
  （MacroAction 化の設計。本書はその上に確率レイヤを載せる）

原設計 §18.1「最初に Step 0 だけを実施し、調査結果とブロッカーを提示すること」に従い、
§1 に**実測で確定した事実**を、§2 以降にその事実に基づく設計を書く。
推測で埋めた箇所は明示的に「未確定」と書き、§11 のブロッカー一覧に載せる。

**第2版での主な変更**（追加検証の結果）:
1. コインの可制御性を「1 フリップ 1 select（自ターン・単発コインに限る）」へ厳密化。多コインは未検証。
2. クラス分類を**ログからの推論ではなく実行時の再生一致検査で確定**する方式へ変更（規則 R3）。
3. B1 に対する実行時ガード（規則 R4）を追加。
4. 予算の既定値（700 ms）を**撤回**し、「まず計測 → 段階的に引き上げ」方針へ。Phase 3 の予算は独立。
5. `Q_base` ロールアウトの失敗時フォールバック（規則 R6）を必須化。

---

## 1. Step 0: capability report（すべて実測）

計測環境: Windows 11 / `cg.dll` / Python 3.14 / デッキは `deck.csv`（フーディン）。
局面コーパスはランダム対戦2試合から自分の MAIN 選択 60 件を保存したもの
（プローブは `search_begin_input` だけを使うので、対戦本体とは別プロセスで実行した）。
プローブ実体はスクラッチパッドに置いてある（§10 Step 1 でリポジトリへ移す）。

### 1.1 抽象名 → 実体の対応表

| 原設計の抽象名 | 実体 |
|---|---|
| 行動決定エントリポイント | `ptcg_ai/action_selection/selector.py:47` `select_action()` |
| 探索モジュール契約 | `search(state, legal_actions, context) -> list[int] \| None`（`selector._SEARCH_MODULES`） |
| 既存 Phase 1・2 実装 | `ptcg_ai/search/lethal_simple.py`（反復深化 DFS） |
| `Observation` | `cg/api.py:439`。`select` / `logs` / `current` / `search_begin_input` |
| 探索用完全状態 | `cg/api.py:448` `SearchState(observation, searchId)`。**自前の State 表現は存在しない** |
| 合法手 | `obs.select.option`（エンジン生成）。返すのは**インデックスの list[int]**（単一 ID ではない） |
| 勝利判定 | `State.result == State.yourIndex`。`-1` は継続 |
| ターン終了判定 | ステップ後に `state.yourIndex != me`（END 後は相手番の選択が来る） |
| 非公開情報の供給 | `search_begin(obs, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin)` |
| 非公開情報の構築 | `hidden_information/search_state_stub.build_dummy_search_state()`（自分側は厳密、相手側はダミー） |
| critic | `ptcg_ai/learning/value_model.ValueModel.predict_win_prob(obs)` |
| 通常方策 | `ptcg_ai/action_selection/router.route(obs)`（ルールベース） |
| 資源解放 | `search_release(searchId)` / `search_end()`（メモリ再利用） |

**状態複製は不可能**。`SearchState` を複製する API は無い。ある局面へ戻る唯一の方法は
`search_begin()` からの**プレフィックス再生**である。これがコスト設計の前提になる（§1.4）。

### 1.2 ★最重要: ランダム性は 3 クラスに分かれ、うち 2 つは完全に可制御

原設計 §5.3 は「完全列挙 > 上下界 > サンプリング」の優先順位を置くが、
どれが可能かは実測しないと決まらない。測った結果、**乱数は 3 クラスに分離できる**。

#### クラス C（可制御・厳密）: シャッフル前のドロー

`search_begin()` に渡した `your_deck` の**末尾が山札の一番上**で、
ドローはその逆順に消費される。テストケース化した検証（コーパス A 60 件中、
深さ1でドローが起きる 18 件が対象）:

| ID | テストケース | 結果 |
|---|---|---|
| A1 | 任意のカード X を末尾に置くと最初に引かれるのは X（1 局面 3 種類） | **54 / 54** |
| A2 | 末尾 k 枚を固定して先頭側だけシャッフルしても引く k 枚は不変 | **54 / 54** |
| A3 | `drawn == reversed(your_deck[-k:])` | **18 / 18** |
| A4 | 引いたカードは供給プールの部分 multiset | **18 / 18** |
| 補助 | 同一 `your_deck` の再実行で同一結果 / 逆順にすると変わる | 18 / 18・18 / 18 |

→ **任意 outcome の適用が可能**。「上から k 枚がこの multiset になる順列」を作って
`search_begin` し直せば、その outcome を強制できる。原設計 §5.3 の「完全列挙」は
このクラスについて**実現可能**。

適用範囲の限界: 深さ 1 のドロー・1 デッキでの観測であり、深い位置のドローや
他デッキは未検証（capability report §6 U6/U7）。よって**クラス C は「そう見える」ではなく
規則 R3 の再生一致検査に合格して初めて主張する**（§2.5）。

#### クラス M（可制御・条件付きで厳密）: コイン

`search_begin(..., manual_coin=True)` を渡すと、コイン判定の直前に
`SelectContext.COIN_HEAD`（YES/NO, `minCount=maxCount=1`）の選択が挿入される。

| 検証 | 結果 |
|---|---|
| 自分のターン中のコイン select の視点 | **`state.yourIndex == me`**（3 局面で確認。`result=-1`、ターン継続中） |
| YES を選ぶ / NO を選ぶ | `COIN:H` / `COIN:T` が必ず出る（345 / 345、368 / 368） |
| `manual_coin=False` | 表 180 / 裏 168（≒ 0.5） |
| 1 select が決めるフリップ数 | 自ターン内トレースでは **1 select = 1 フリップ** |

> 初回計測で「193 select に対し 358 コインログ」と見えたのは誤読だった。
> `Observation.logs` は受け取るプレイヤー基準の履歴なので、相手番まで進めると
> **同じコインが両者のログに二重計上**される。

→ 単発コインは表・裏の両枝を 0.5 ずつで**厳密に列挙でき**、Phase 2 の AND ノードとして扱える。
**未検証**: 「コインを2回投げる」等の多コイン効果は現デッキに存在しない。
1 select が複数フリップをまとめて決める実装なら混合 outcome が到達不能になり列挙が破綻するため、
**多コイン効果を検出したら Phase 2 は `UNSUPPORTED_EFFECT` で棄却する**（§3.2）。

#### クラス S（不可制御・サンプルのみ）: シャッフル後のドロー

経路上で `LogType.SHUFFLE` が起きると、以降のドローは**エンジン内部 RNG** になり、
同じ入力でも結果が変わる（同一経路の再実行で 3 / 3 不一致）。

なぜ Phase 2 の証明に使えないか（3 点とも独立に致命的）:

1. **任意 outcome を適用できない**。AND ノードの各枝へ到達する手段が無いので、
   「全 outcome で勝つ」を確認する操作そのものが定義できない。
2. **網羅を保証できない**。サンプリングで観測されなかった outcome が存在しないことを示せない。
3. **分布が既知でない**。800 サンプルの χ² 検定（χ²=19.2, dof=21, χ²/dof=0.91）は
   「一様と矛盾しない」だけで、**一様であることの証明ではない**（1 局面・1 デッキ・1 事象）。

→ 設計では**我々が分布を仮定しない**。クラス S の推定は
「同一プレフィックスを再生してエンジンにサンプリングさせる」方式に限定し、
`EstimateKind.SAMPLED` と信頼区間を必ず付ける（規則 R5）。
エンジンの一様性を前提にした解析的確率計算は**行わない**。

#### 3クラスの帰結

```
Phase 1 : ランダム事象が経路上に一切無い        → 確定
Phase 2 : ランダム事象がクラス C / M のみ       → 全 outcome 列挙で確定（証明可能）
Phase 3 : クラス S を含む／列挙打ち切り          → 区間 or SAMPLED（証明不可）
```

**クラス判定はログからの推論では確定させない。** `SHUFFLE` / `COIN` ログは
「クラス S かもしれない」を示す保守的マーカーとして使い、
クラス C を主張するときは必ず再生一致検査を通す（規則 R3、根拠は §1.2 補足）。

補足（`SHUFFLE` ログの十分性の実測）: ランダム経路の全プレフィックス 457 本を
2 回ずつ再生して比較したところ、**非決定的だったものは全て SHUFFLE（または COIN）を含む経路**で、
「SHUFFLE も COIN も無いのに非決定的」は 0 件だった。ただし SHUFFLE を含んでも
決定的なままの経路が 168 件あり、マーカーは**過剰検出**側に外れる。
また「SHUFFLE 以外に乱数源が無い」ことは証明されていない（capability report §6 U2）。

### 1.3 相手番・相手選択

- END を選ぶと `yourIndex` が相手へ移り、探索はそのまま相手番の選択を出し続ける。
  相手番の選択も `search_step` で我々が指定できる（＝相手のモデル化は我々の責任）。
- 我々のターン中でも相手が選ぶ効果があれば `yourIndex` が一時的に相手へ移る想定。
  検出は `state.yourIndex != me` かつターン継続で可能。Phase 2 では **AND**、
  Phase 3 では **min（OPPONENT_CHOICE）** として扱う（原設計 §5.3 と一致）。
- END 直後の Observation は**相手視点**になり、`players[me].hand` は `None`、
  相手の手札は我々が捏造したダミー（基本エネ）が見える。
  critic をここで呼ぶと相手視点の値になるため `1 - p` が必要で、
  かつダミー手札という分布外入力を食わせることになる（§7.2 で扱う）。

### 1.4 コスト実測

| 操作 | 実測 |
|---|---|
| `search_step` | **0.316 ms/回**（n=732, p95 0.525 ms） |
| `search_begin` | **0.436 ms/回**（n=120） |
| `ValueModel.predict_win_prob` | **1.30 ms/回**（初回のみ重みロードで約 31 ms） |
| 通常方策 1手 `router.route()` + step | 約 0.5 ms（下記ロールアウト実測から） |
| 通常方策でターン終了までロールアウト（begin + 平均 9.8 手 + critic 1 回） | **6.5 ms**（300 本、p95 11.2 ms、最大 48.9 ms、失敗 0） |

**現行エージェントの実測**（`configs/rule_lethal.json`、自分 vs ランダム 4 試合）:

| 指標 | 実測 |
|---|---|
| 自分の意思決定 | 72〜88 回/試合 |
| エージェント総時間 | **1.80〜2.28 秒/試合**（10 分の持ち時間に対し 0.4% 未満） |
| 1 意思決定 | 平均 25.8 ms / p50 25.0 / p95 30.1 / **p99 38.1** / 最大 120.9 |
| リーサルゲート発火 | この 4 試合では 0 回。自分 vs 自分 3 試合では 1 試合のみ 16 回（発見 2 / タイムアウト 13 / 平均 82.5 ms） |

この値が予算判断の**ベースライン**であり、引き上げ後は同じ指標で再計測する（§6）。

姉妹文書の測定（78 局面）と整合: エンジン step が探索時間の約 90%。
**100 ms の予算 ≒ エンジン約 300〜380 回**という換算はそのまま使える。

**outcome 適用コスト**: 深さ `d` のノードで1 outcome を適用するには
`search_begin` + プレフィックス `d` 手の再生が要る。

```
cost(outcome @ depth d) ≈ 0.44 + 0.32 d  [ms]
例: d=6 なら 2.4 ms/outcome → 20 outcome の chance ノード1個で約 48 ms
```

### 1.5 critic（`ValueModel`）

- `value_weights.json` は同梱済み、`is_ready == True`。純 Python・166 特徴・
  ターン帯温度較正あり。出力は **`State.yourIndex` 視点の最終勝率**（`encoder` が視点正規化）。
- 探索中の Observation をそのまま食わせられる（`predict_win_prob(node.observation)` 動作確認済み）。
  探索状態でも相手手札は `None` として見えるので、入力の形は実戦と同じ。
- 学習元は Kaggle 上位リプレイの実戦局面。**Phase 3 が作る「リーサル失敗直後の局面」での
  校正は未測定**（原設計 §10.3）。→ Phase 3 は既定 OFF（§11 B3）。
- 代替 critic 候補: Transformer policy の value head（オフライン AUC 0.759 > 専用値ネット 0.7446、
  `docs/plans/transformer-policy/`）。ただし `policy_weights.npz` は未コミットで、
  推論経路も未整備。本設計では**使わない**（将来差し替え可能な `ValueEvaluator` 契約にする）。

### 1.6 通常方策と `Q_base`

原設計 §10.2 は `Q_base` の厳密計算を難しい前提で書いているが、**このリポジトリでは可能**。

`router.route()` は `Observation` だけを入力に取り、探索中の Observation でも動く。
60 局面 × 捏造シナリオ 5 通り = **300 ロールアウト**での実測:

| 指標 | 結果 |
|---|---|
| `None` 返却 / 契約違反の行動 / 例外 / エンジン拒否 / step 上限到達 | すべて **0** |
| 終了理由 | ターン終了 290 / 決着 10 |
| ステップ数 | 平均 9.8 / p95 19 / 最大 27 |
| 所要時間 | 平均 6.5 ms / p95 11.2 ms / 最大 48.9 ms |
| 相手モデル tracker の内部状態 | ロールアウト前後で**変化なし** |

情報境界の確認:
- `tracker.update()` の呼び出しは `selector.py:70` **のみ**。`router` 配下は
  `current_matchup_plan()`（読み取り）しか呼ばないので、ロールアウトが
  実戦の相手予測を汚染することはない。
- ロールアウト中の Observation で相手手札は `None`（実戦と同じ形）。
  我々が捏造した相手デッキ内容が通常方策に見えることはない。

```
Q_base(s) = E_outcome[ V( 通常方策でターン終了まで進めた葉 ) ]
```

`Q_try` の失敗葉と**同じ葉の定義・同じ視点変換・同じ outcome 重み**で評価するため、
critic の系統誤差が比較でかなり相殺される。原設計 §10.2 の「難しければ `V_policy(s)` で近似」
より一段良い。

> **300 本の成功をもって「一般に安全」とはしない。** 1 デッキ・ランダムプレイ由来の局面での観測である。
> 実装では規則 R6（合法性チェック・step 上限・例外捕捉・失敗時は `UNAVAILABLE` にしてゲートを通さない）
> を必須とし、失敗経路自体をテストで固定する（§9.4）。

### 1.7 試合時計

- 公式: 第1ラウンドは**1プレイヤー 10 分**、使い切ると敗北（原設計 §19）。
- 自分の意思決定回数はランダム対戦で **106 / 136 / 120 回/試合**。
  → 単純割りで **1 意思決定あたり約 4.4〜5.7 秒**が理論上の上限。
- 現行のリーサルゲート通過は姉妹文書の実測で **78 件 / 14 試合 ≒ 5.6 回/試合**。
  現行 100 ms 設定なら 1 試合あたり **0.56 秒**しか使っていない。
- **API から残り持ち時間は取得できない**。自前で `time.perf_counter()` を積算するしかない
  （プロセスは試合中永続する前提。§11 B2）。

→ 予算は「100 ms が上限」ではなく**大幅に余っている**。Phase 2/3 の列挙コストは
ここから捻出できる（§6）。

### 1.8 既存 Phase 1・2 実装との関係

`lethal_simple.py` の `verify_shuffles`（別の隠れ情報で再生して勝てるか確認）は、
§1.2 の分類で言えば「クラス C の outcome を **1 サンプルだけ**引き直す」検査である。
- 長所: 供給した山札順に依存する偽リーサルの大半を落とせる。
- 限界: **証明ではない**。1 サンプルで通った線は依然として運依存でありうるし、
  逆に「全 outcome で勝てる線（真の Phase 2 リーサル）」も、別シャッフルで
  手順が違法化すれば棄却されうる（偽陰性）。

新実装はこれを**クラス C の全 outcome 列挙**に置き換える。
`lethal_simple.py` は**一切変更しない**（config の `module` で切替）。

---

## 2. 設計の中核: 情報集合ベースの AND-OR / Expectimax

### 2.1 シナリオと outcome クラス

`your_deck` の順列 1 本を **シナリオ** と呼ぶ。シナリオを固定すると探索は完全に決定的になる
（クラス S を踏まない限り）。ここで最も危険な誤りが、原設計 §2.1 が禁じている

> **シナリオごとに独立に最善手を選ぶ（＝determinization / カンニング）**

である。「上から Rare Candy が来るシナリオでは進化を選び、来ないシナリオでは別の手を選ぶ」
を無条件に許すと、引く前から結果を知っている探索になる。

正しい構造は原設計 §8 の通り:

```
∃a₀ ∀o₁ ∃a₁(o₁) ∀o₂ ⋯ : Win
```

実装上は「**決定ノードのキーを情報集合にする**」ことで担保する。

### 2.2 決定の情報集合キー

決定ノードで参照してよいのは、その時点でプレイヤーが知り得る情報だけ:

```python
InfoKey = (
    公開情報の指紋,            # 両者の場・HP・状態異常・トラッシュ・サイド枚数・各種フラグ
    自分の手札 multiset,       # 引いた結果はここに現れる（＝合法な情報）
    自分の山札 multiset と枚数, # 順序は含めない  ★
    解決中の効果と選択段階,
)
```

**`your_deck` の順序そのものは絶対にキーに入れない**。同一 `InfoKey` の下では
同一の行動を選ぶことを構造的に強制する（transposition table のキーがこれになる）。

実装メモ: `lethal_simple._state_key()` は `repr(State)|repr(SelectData)` で、
`State` は山札順を含まない（`deckCount` のみ）ので**そのままでも順序は漏れない**。
ただし手札の順序や serial は含むため、正規化（id の multiset 化）を入れて合流率を上げる。

### 2.2.1 山札順序を閉じ込めるための規則

「キーに入れない」だけでは足りない。順列は**探索の他のどこにも出てはいけない**。

| 規則 | 内容 |
|---|---|
| **R1** | `InfoKey` は上記 4 要素のみ。シナリオ（山札順列）・実順序・相手の非公開手札は含めない |
| **R2** | `InfoKey` の生成は `infokey.py` 1 箇所に集約する。探索・transposition・Goal 分析・マクロ生成・critic 入力はすべてそれを経由し、**順列型 `Scenario` は `outcome.py` の `OutcomeApplier` の外へ出さない**（型で分離し、他モジュールのシグネチャに `Scenario` を出現させない） |
| **R2a** | critic への入力は葉の `Observation` のみ。特徴量に山札順由来の値（`your_deck` の要素・並び）を渡さない。`encoder` が使うのは `State` であり、`State` に順序情報は含まれない（`deckCount` のみ）ことをテストで固定する |
| **R2b** | 診断ログにも順列を書かない（原設計 §14）。記録するのは outcome の**等価類ラベルと確率**まで |

原設計 §2.1 の受入テスト「隠れた山札順だけを変えても、ドロー前の探索結果が同一」は、
これらに対する直接のテストになる（§9.1 に具体化）。

### 2.2.2 「シナリオごとに最善手」を防ぐ構造（determinization 対策）

危険なのは、outcome を適用するために `search_begin` をやり直すという**実装手段**が、
「シナリオを固定した決定的な木」を自然に作ってしまうことにある。以下で構造的に禁じる。

1. **探索木のノードは `InfoKey` で同一視する。** 異なるシナリオから到達した同一 `InfoKey` は
   同じノードであり、そこで選ぶ行動は 1 つに決まる（transposition table が強制する）。
2. **候補生成はシナリオを見ない。** `macro_generator` の入力は `InfoKey` 由来の情報だけで、
   `Scenario` を受け取らない（R2 の型分離）。
3. **outcome 適用は「行動を決めた後」にのみ行う。** 決定ノードで行動 `a` を選ぶ →
   `a` が新情報境界に達する → そこで初めて outcome を列挙し、各 outcome へ
   `search_begin` + プレフィックス再生で入る。行動選択が outcome を見て行われる経路を作らない。
4. **再生プレフィックスは常に「情報集合上の手順」である。** プレフィックスは選択インデックス列であり、
   どのシナリオでも同一の列を使う。あるシナリオでのみ合法な列が出てきた場合は
   `STATE_MISMATCH` として棄却する（そこは情報集合が割れている証拠なので、chance ノードとして
   扱い直す）。

この 4 点が守られているかは、§9.1 の不変性テストと、`InfoKey` の単体テスト、
および「`Scenario` 型が `outcome.py` 以外から import されていない」ことの静的チェックで確認する。

### 2.3 chance ノードの実装（2 つの実行戦略）

chance ノード = 「新情報が公開される地点」= 姉妹文書の MacroAction 境界と**同一物**。
検出は原設計 §2.2 通り、直前 step の `logs` に `DRAW` / `COIN` / `SHUFFLE` が現れたか、
`select.deck != None`、`state.looking != None` で行う。

outcome を適用する方法は 2 つあり、どちらもコストが違う。

| 戦略 | やり方 | コスト | 使う場面 |
|---|---|---|---|
| **A: ルート再開** | outcome を実現する順列で `search_begin` → プレフィックス再生 | `0.44 + 0.32d` ms/outcome | 汎用。実装が単純 |
| **B: シナリオ先読み** | ルートで山札全体の順列を決め打ちし、その 1 本で深さ方向を進む | 追加コスト 0 | 同じ outcome 系列を何度も辿る場合 |

初期実装は **A に統一する**（B は「同じシナリオで複数の分岐を辿る」ときに
情報集合の制約を破りやすく、バグが致命的になるため）。B は Step 3 の最適化候補。

### 2.4 outcome の等価類と粗視化（RelevanceModel）

`k` 枚ドローの outcome を「引いたカードの multiset」で分類すると、
`k=7`（博士の研究相当）では組合せが爆発する。原設計 §5.2 の「同値統合」を
実装可能にするため、**目的に対する等価類**へ粗視化する。

```python
class RelevanceModel:
    """Goal に対してカードを等価類へ写す。既定は恒等写像（＝厳密）。"""
    def class_of(self, card_id: int) -> Hashable: ...
```

粗視化が**健全**（＝厳密列挙と同じ結果を返す）と言えるのは、同一クラスのカードが

1. マクロ生成器が出す候補集合を変えない、
2. 打点計算を変えない（フーディンの**手札枚数比例打点**は枚数のみ依存なので保存される）、
3. 後続 chance ノードの確率分布を変えない（multiset が同じなら自動的に満たす）、

を全て満たすときに限る。満たせない場合は**粗視化しない**（＝ Phase 2 は `UNKNOWN`、
Phase 3 は上下界／サンプリングへ落とす）。

確率は多変量超幾何分布で厳密に計算する（原設計 §5.2）。
`Fraction` で計算し、各 chance ノードで**確率の総和が厳密に 1** であることを assert する。

粗視化の健全性は §9.3 の exhaustive oracle テストで、恒等写像の厳密列挙と
突き合わせて検証する（これが粗視化を入れる前提条件）。

### 2.5 実行時の検証規則（推論しない）

§1.2 のクラス分類は「1 デッキ・限定サンプルでの観測」に基づく。
観測を仕様として扱わないため、**実行時に検証してから主張する**規則を置く。

| 規則 | 内容 | 根拠 |
|---|---|---|
| **R3** | クラス C を主張する前に、その chance ステップを **2 回再生して同一結果**であることを確認する。不一致ならクラス S へ降格。SHUFFLE ログの有無だけでは判定しない | §1.2 補足（457 プレフィックスの観測は仕様ではない） |
| **R4** | デッキが公開された経路（`select.deck != None` を通過した）で、SHUFFLE を挟まずにドローが発生したら、その枝を `UNSUPPORTED_EFFECT` で棄却する | B1（§11） |
| **R5** | クラス S の確率は我々が仮定せず、同一プレフィックスの再生でエンジンにサンプリングさせる。`EstimateKind.SAMPLED` と信頼区間を必ず付ける | §1.2（一様性は未証明） |
| **R6** | `Q_try` / `Q_base` のロールアウトは合法性チェック・step 上限・例外捕捉付き。1 つでも失敗したらその候補の価値を `UNAVAILABLE` にし、採用ゲートを通さない | §1.6（300 本成功は一般保証ではない） |
| **R7** | critic 入力は葉の Observation のみ。`yourIndex != me` の葉では `1 - p` を使い、`Q_try` と `Q_base` で同一規約を適用する | §1.3 / §7.2 |

R3 のコスト: chance ノード 1 個につき再生 1 回分（`0.44 + 0.32d` ms）。
Phase 2 の証明にのみ必須とし、Phase 3 のサンプリング枝では省略してよい（そこは元々 `SAMPLED`）。

#### 完全列挙できるケース／できないケースの分離

| 区分 | 条件（すべて満たすこと） | Phase 2 | Phase 3 |
|---|---|---|---|
| **列挙可能** | ① 経路上で SHUFFLE 未発生 ② デッキ未公開の局面から開始 ③ R3 の再生一致検査に合格 ④ outcome が自分の山札 multiset から決まる ⑤ コインは単発（1 select 1 フリップ） | 全 outcome 列挙で `PROVEN_WIN` を主張可 | 厳密重み（`Fraction`）で `EXACT` |
| **列挙不可** | 上のいずれかを満たさない（SHUFFLE 後 / デッキ公開局面から開始 / 再生不一致 / 多コイン / 相手選択） | `UNKNOWN` + 理由（`SHUFFLE_ENCOUNTERED` / `UNSUPPORTED_EFFECT` 等） | サンプリング or 上下界（`SAMPLED` / `BOUNDED`） |

---

## 3. Phase の定義（実測に合わせた再定義）

原設計 §1 の枠を維持しつつ、判定条件を §1.2 の 3 クラスで書き直す。

### 3.1 Phase 1: 決定的な確定リーサル

- 経路上に chance 事象が**一切無い**線のみ。ドロー系マクロは展開しない
  （確定サーチはドローを伴わないので Phase 1 で扱える。ただしサーチ後の
  `SHUFFLE` は経路を汚さない ── 攻撃までに再ドローしないなら安全）。
- 探索はヒューリスティック付き IDDFS（原設計 §7）。マクロ順序は姉妹文書の Goal 依存優先度。
- `PROVEN_WIN` の線は、**ランダムな隠れ情報 N 本で再生して全て勝つ**ことを確認する
  （既存 `verify_shuffles` を N>=2 に強化。chance が無いなら本来不要だが、
  マクロ境界検出のバグに対する安全網として残す）。

### 3.2 Phase 2: 全 outcome での確定リーサル

- chance 事象がクラス C / M のみで構成される線を対象に、3値 AND-OR 探索を行う。
- chance ノード: 非 0 確率の**全 outcome クラスを列挙**し、全てで `PROVEN_WIN` なら勝ち。
  1 つでも `PROVEN_NO_WIN` ならその候補は不成立。それ以外に `UNKNOWN` が残れば `UNKNOWN`。
- クラス S（`SHUFFLE` 後のドロー）が経路に現れた時点で、その枝は
  `UNKNOWN / OUTCOMES_NOT_ENUMERABLE` として Phase 3 へ委譲する。
- 相手選択ノードは AND（全ての相手選択で勝てること）。列挙できなければ `UNSUPPORTED_EFFECT`。
- コイン: `manual_coin=True` で表・裏の両枝を必ず展開する。
  **Phase 2 では `manual_coin=True` を常用する**（Phase 1 も同様。コインが出た時点で
  Phase 1 は打ち切り、Phase 2 に渡す）。
  ただし 1 つの `COIN_HEAD` select に対して**複数のコインログが出た場合**は、
  混合 outcome へ到達できない可能性があるため `UNSUPPORTED_EFFECT` で棄却する（B8）。
- クラス C を主張する各 chance ノードで、規則 R3（2 回再生して一致）を必ず通す。
  デッキ公開経路では規則 R4 のガードを通す。

#### 相手選択ノード（B4）の扱い — Step 1-4 で実装した内容

実測（Step 1-4）: 自分のターン中に相手へ手番が移るのは、**KO 後の
`TO_ACTIVE`（次のバトルポケモンを選ぶ）** と、効果による `DISCARD` の 2 種類。
保存盤面 26 件の浅い探索で `TO_ACTIVE` 50 件（うち **24 件は選択肢が 2 つ以上**）、
`DISCARD` 7 件（選択肢 7 個）。エンジンは相手の選択肢も `obs.select.option` として
提示するので、**こちらから列挙して 1 つずつ適用できる**。

したがって原設計 §5.3 の通り AND ノードとして実装する（緩和はしない）。

| 場面 | 判定 |
|---|---|
| 全ての合法な相手選択が `PROVEN_WIN` | `PROVEN_WIN` |
| 1 つでも `PROVEN_NO_WIN` | `PROVEN_NO_WIN`（相手はそれを選べる） |
| `UNKNOWN` が混じる | `UNKNOWN` |
| 相手の選択肢を全部列挙できない（組合せ上限に当たった） | `PROVEN_WIN` を**主張しない**（`UNKNOWN`） |
| 相手の選択が新情報を公開する | Phase 1 では `UNKNOWN`（飛ばすと AND が成立しないため） |

- 確率平均しない。相手が我々に有利な選択をする前提も置かない。
- Phase 1 でも同じ AND 規則を使う（「1 本の勝ち手順」を相手の都合で選ばせない）。
- **Phase 3 では同じノードを最悪値（min）で評価する**（未実装。原設計 §9.4）。
- ターン終了（`TURN_END` ログ、またはターン番号の変化）と、ターン継続中の
  相手選択は明確に区別する。前者は終端、後者は AND ノード。

### 3.3 Phase 3: 確率リーサル

- Expectimax（原設計 §9.4）。決定ノードは `InfoKey` で正規化（§2.2）。
- chance ノード:
  - クラス C / M → 厳密重み（`Fraction`）
  - クラス S → **エンジンサンプリング**（同一プレフィックスを N 回再生して経験分布を得る）。
    seed・試行数・Wilson 信頼区間を記録し `EstimateKind.SAMPLED` を付ける。
- 未処理 outcome 質量 `m` は原設計 §9.5 の通り上下界へ反映（`L=Σp_iL_i`, `U=Σp_iU_i+m`）。
- 失敗葉 = 「今ターン中に勝てないと確定した最初の意思決定状態」。そこで通常方策へ handoff し、
  critic で `V_fail` を得る（§7）。
- 採用は `Q_try` vs `Q_base` の**区間比較**（§7.3）。`P_lethal` 単独では採用しない。

### 3.4 起動条件（precheck）

原設計 §1.1 に従い、Phase 定義とは分離する。現行ゲート（`残りサイド <= 2` かつ自分番）を
維持しつつ、姉妹文書 §3.8 の安全な早期棄却のみを追加する。

- 棄却してよい: 必要サイド枚数 > 今ターンに取り得るサイド上限
  （相手の場に ex / メガex が無く、複数 KO 手段も無い等）
- 棄却してはいけない: 現在打点不足 / グスト札が手札に無い / エネ不足 / 攻撃役不在
  （ドロー・サーチ・進化で解決しうる）

`build_dummy_search_state()` が `None` を返す局面は探索不能。
`select.effect` 由来の 26% 取りこぼし（姉妹文書 §2.4(d)）は
作業ツリーの `search_state_stub.py` で修正済み。**この修正の回帰テストを Step 1 で固定する。**

---

## 4. モジュール構成（実配置）

`cg/` は変更しない。`lethal_simple.py` も変更しない。新規パッケージを足す。

```
sample_submission/ptcg_ai/search/
  lethal_simple.py              # 変更なし（既定のまま）
  lethal/
    __init__.py
    entry.py          # search(state, legal_actions, context) 契約。selector から呼ばれる
    types.py          # Proof / EstimateKind / NodeKind / StopReason / 区間 / Budget / Candidate
    adapter.py        # cg.api ラッパ。search_begin/step/release/end、資源のライフサイクル管理
    infokey.py        # 情報集合キー生成（山札順を含めないことを保証する唯一の場所）
    goal.py           # LethalGoal 分析（姉妹文書 §3.4）
    macro.py          # MacroAction と境界判定（reveal boundary == chance node）
    macro_generator.py# Goal 依存の候補生成・順序付け
    outcome.py        # chance 事象の分類（C/M/S）・outcome クラス列挙・シナリオ順列生成
    probability.py    # 多変量超幾何（Fraction）・区間演算・Wilson 区間
    phase1.py         # IDDFS（chance 無し）
    phase2.py         # 3値 AND-OR
    phase3.py         # Expectimax + 失敗葉収集
    value.py          # ValueEvaluator 契約 / ValueModel 実装 / Q_base ロールアウト
    gate.py           # 採用ゲート
    transposition.py
    budget.py         # 試合時計トラッカと予算配分
    diagnostics.py
```

配線:
- `selector._SEARCH_MODULES` に `"lethal_phase123": lethal.entry` を 1 行追加
- `configs/rule_lethal_phase12.json`（Phase 3 OFF）と
  `configs/rule_lethal_phase3.json`（実験用）を新規追加。既存 `rule_lethal.json` は据え置き

---

## 5. 型（原設計 §4 をリポジトリ規約に合わせたもの）

原設計 §4 の型をそのまま採用する。追加・変更は以下のみ。

```python
class ChanceClass(Enum):
    CONTROLLED_DRAW = auto()   # クラス C: SHUFFLE 前のドロー
    CONTROLLED_COIN = auto()   # クラス M: manual_coin
    ENGINE_SAMPLED = auto()    # クラス S: SHUFFLE 後
    OPPONENT_CHOICE = auto()

@dataclass(frozen=True)
class Outcome:
    label: Hashable                 # 等価類（引いたカードの multiset / 表裏）
    probability: Fraction           # 厳密（クラス S では推定値 + 区間）
    scenario: tuple[int, ...] | None  # これを実現する your_deck 順列（クラス C のみ）
```

`StopReason` に `SHUFFLE_ENCOUNTERED` を追加（クラス S 遭遇による Phase 2 打ち切り理由）。
確率・価値は `[0,1]` を assert、NaN・質量欠落・視点反転は fallback 対象（原設計 §4 末尾）。

---

## 6. 予算管理

### 6.1 「余地がある」と「使うべき」を分離する

§1.4 / §1.7 の実測が示すのは**余地**であって、推奨値ではない。

| 事実（実測） | そこから言えること | 言えないこと |
|---|---|---|
| 現行エージェントは 2〜3 秒/試合しか使っていない（p99 38 ms/手） | 予算を増やす余地は確かにある | 増やせば強くなる、とは言えない |
| 自分の意思決定は 72〜136 回/試合 | 均等割りなら 1 手 4〜5 秒が理論上限 | それを 1 手で使ってよいとは言えない（B2: Kaggle の 1 手上限が不明） |
| ゲート発火は 0〜16 回/試合とばらつく | 総消費は発火回数に強く依存する | 「5.6 回/試合」を一定値として計画に使えない |

したがって**既定値は先に決めない**。手順は次の通り。

```
Step A: 現行と同じ 100 ms のまま新実装を動かし、§1.4 の指標一式を再計測
Step B: 発見数・偽陽性・p95/p99・1試合総時間の 4 点を見て、100 → 200 → 400 ms と段階的に上げる
Step C: A/B で最終勝率の改善が確認できた値のみを既定値として採用する
```

計測項目（Step A 以降、毎回同じ形式で取る）:
1 試合の総エージェント時間 / 1 手平均 / p50・p95・p99 / 最大 /
ゲート発火回数 / ゲート 1 回あたりの時間 / Phase 別内訳（Phase 1・2・3・critic）。

### 6.2 Phase 3 の予算は Phase 1・2 と独立させる

Phase 3 が Phase 1・2 の予算を食う構造にしない（原設計 §11.4 の優先順位を時間面でも守る）。

```
phase12_deadline = now + phase12_ms                     # Phase 1・2 専用
phase3_deadline  = now + phase12_used + phase3_ms       # Phase 1・2 が使い切った後から
                                                        # 追加で phase3_ms だけ
```

Phase 3 を無効化しても Phase 1・2 の挙動が 1 ms も変わらないこと（同一の探索順序・
同一の結果）をテストで固定する。

### 6.3 設定

```python
@dataclass
class LethalSearchConfig:
    enabled: bool = True
    module: str = "lethal_phase123"
    max_remaining_prizes: int = 2
    phase1_enabled: bool = True
    phase2_enabled: bool = True
    phase3_enabled: bool = False         # 原設計 §18.4: critic 未検証のため既定 OFF

    # 時間予算: 既定は現行と同値から始める（Step A）。以降は計測に基づいてのみ変更する
    phase12_ms: int = 100                # 現行 lethal_simple の time_limit_ms と同じ
    phase3_ms: int = 0                   # Phase 3 有効時に実験で設定（Phase 1・2 とは独立）
    per_decision_cap_ms: int = 200       # 保守的な上限。B2 解決まで引き上げない
    match_budget_ms: int = 600_000       # 公式 10 分
    match_safety_ms: int = 180_000       # 使い残す安全マージン（30%、B2 未解決のため厚め）

    max_nodes: int = 10_000              # 現行と同値から
    max_primitive_depth: int = 20
    max_chance_depth: int = 2
    max_outcomes_per_event: int = 16
    replay_check_for_proof: bool = True  # 規則 R3
    phase3_sampling_enabled: bool = False
    phase3_samples_per_event: int = 64
    adoption_margin: float = 0.0         # 検証後に設定
```

**試合時計トラッカ** (`budget.py`):

```python
consumed_ms      # agent() 内で自分が消費した累積時間（perf_counter 積算）
decisions_seen   # 自分の意思決定回数
remaining_ms     = match_budget_ms - match_safety_ms - consumed_ms
budget_for_now   = min(per_decision_cap_ms, remaining_ms / expected_remaining_decisions)
```

`expected_remaining_decisions` は実測（72〜136 回/試合）から
`max(20, 140 - decisions_seen)` で保守的に見積もる。
トラッカはプロセスが再起動した場合に**残り時間を少なく見積もる側**へ倒す（B2）。
残り時間が閾値を割ったら **Phase 3 → Phase 2 の順に停止**し、Phase 1 小予算 or 通常方策へ戻す。

---

## 7. critic / `Q_try` / `Q_base` / 採用ゲート

### 7.1 `ValueEvaluator` 契約

```python
class ValueEvaluator(Protocol):
    def evaluate_leaf(self, node, me: int) -> ValueEstimate:
        """葉（通常方策へ戻す時点）の、自分視点の最終勝率。"""
```

初期実装は `ValueModel.predict_win_prob`。将来 Transformer value head へ差し替え可能。

### 7.2 葉の定義と視点（★取り違えやすい）

§1.3 の実測通り、END 後の Observation は**相手視点**である。よって葉の評価規約を
**1 つに固定**する:

> 葉は「自分のターンが終了した直後の Observation」とし、
> `state.yourIndex != me` なら `V = 1 - predict_win_prob(obs)` を使う。

この規約を `Q_try` の失敗葉と `Q_base` のロールアウト葉の**両方**に同一適用する。
実測でも `V(END前, 自視点)=0.564` と `1 - V(END後)=0.665` は一致しないので、
規約を混ぜると 0.1 規模のバイアスが混入する。**混ぜないこと自体が安全条件**。

なお END 後の局面には我々が捏造した相手手札（ダミー基本エネ）が含まれる。
critic の入力には相手手札の中身は入らない（`hand` は `None` で見える）が、
`handCount` 経由の特徴は本物なので問題ない。ただし**この規約での校正は未測定**であり、
Phase 3 を有効化する前に §9.2 の校正評価を通すこと。

### 7.3 `Q_try` と `Q_base`

```
Q_try(s,a)  = P(win now | s,a) + Σ_f P(f | s,a) · V(f)
Q_base(s)   = Σ_o P(o) · V( 通常方策でターン終了まで進めた葉 )      # §1.6、約 5 ms/本
```

`Q_base` は**同じ outcome 重み**（同じシナリオ集合）で平均する。
これにより「山札の並びが良いシナリオだけで Q_try を計算する」偏りが打ち消される。

採用条件（原設計 §11.1）:

```
Q_try.lower > Q_base.upper + δ      # 区間が使えるとき（既定）
Q_try.mean  > Q_base.mean  + δ      # 点推定しか無いときの暫定
```

`δ` は検証データから決める（初期値 0.0 はプレースホルダで、Step 5 まで Phase 3 は OFF）。
`P_lethal` 単独の固定閾値ゲート（原設計 §11.3）は**実験フラグに隔離**し、既定では使わない。

優先順位（原設計 §11.4）:
```
Phase 1 PROVEN_WIN → 無条件採用（critic に拒否権を与えない）
Phase 2 PROVEN_WIN → 無条件採用
Phase 3 candidate  → ゲート通過時のみ
その他             → router.route()
```
ただし採用直前の合法性チェック（`selector.is_valid_action` 相当）は常に通す。

---

## 8. 実行・再探索・診断

原設計 §12 の通り、**最初の 1 手だけ実行**する。`selector.select_action()` は
毎選択ごとに呼ばれるので、この構造は既存実装のまま維持される
（＝条件付き方策木を保持して実行し続ける機構は作らない）。

再探索時の整合チェック: ルートの公開指紋を保存し、次の Observation で一致を確認する。
不一致は `STATE_MISMATCH` として診断に残し、Phase 1 から再探索する。

診断（原設計 §14）は `diagnostics.py` に集約し、既存 `get_stats()` 互換のキーを保ちつつ
`phase`, `proof`, `p_lethal_lower/upper`, `estimate_kind`, `failure_mass`, `v_fail`,
`q_try`, `q_base`, `delta`, `adopted`, `stop_reasons`, `chance_class_counts`,
`engine_steps`, `outcomes_expanded`, `elapsed_ms`, `budget_ms` を追加する。
**非公開カード実体・山札順・全状態 dump は記録しない。**

---

## 9. テスト計画

### 9.1 情報漏洩（最優先・原設計 §2.1）

**T1: 隠れた山札順に対する不変性**（Phase 1 の実装より先に書く）

```
test_root_action_invariant_to_hidden_deck_order:
  for seed in 0..N-1:
      hs = build_dummy_search_state(obs, deck, rng=Random(seed))
      assert Counter(hs["your_deck"]) が seed に依らず同一   # 同一 multiset の別順列
      r[seed] = run_search(obs, hs)                          # Phase 1 / 2 / 3 それぞれ
  assert 全 seed で r[seed].first_action が一致
  assert 全 seed で r[seed].proof        が一致
  assert 全 seed で r[seed].p_lethal(下限・上限) が一致
         # クラス S を含む局面のみ、p_lethal はサンプリング誤差の範囲に緩める。
         # その場合も first_action と proof は厳密一致を要求する
```

対象は姉妹文書の 78 局面コーパス（`select.effect` 修正後は全 78 件が探索可能）。

**T2: `InfoKey` の単体テスト** — 山札順を変えてもキー不変／手札・場・サイドを変えるとキー変化／
相手の非公開手札がキーに現れない。

**T3: 型分離の静的チェック** — `Scenario` 型が `outcome.py` 以外から import されていないこと
（規則 R2）。CI で grep 相当の検査を回す。

**T4: critic 入力の検査** — critic へ渡す `Observation` が探索の内部状態（順列・未処理質量など）
を含まないこと。`State` に順序情報が無いことを固定するテストを含む（規則 R2a）。

**T5: 変異テスト** — 意図的に `InfoKey` へシナリオを混ぜた実装で T1 が**落ちる**ことを確認する
（テストが実際に検出力を持つことの確認）。

### 9.2 Phase 1・2 の安全性

- 確定勝利の全葉を SDK で再生し `state.result == me` に到達
- Phase 2 の全非 0 outcome を方策木が覆う（質量合計が厳密に 1）
- クラス S 遭遇時に `PROVEN_WIN` を返さない
- 列挙不能・時間切れは `UNKNOWN`
- 偽 `PROVEN_WIN` 0 件 / 違法手 0 件 / fallback 100%

### 9.3 exhaustive oracle（原設計 §15.3）

小さい人工局面（山札 6〜10 枚、選択肢 3〜4 個）で全プリミティブ行動 × 全 outcome を
列挙する oracle を作り、`P_lethal`・選択 root action・失敗状態分布・`Q_try` を比較する。
**粗視化（§2.4）の健全性はここでのみ担保できる**ので、粗視化を入れる Step 3 の前提とする。

### 9.4 確率の単体テスト

- 超幾何分布の手計算と一致（`Fraction` 厳密比較）
- 各 chance ノードで確率総和 == 1
- 完全探索時に `P_lethal.lower == upper` かつ `EstimateKind.EXACT`
- 未処理質量が上下界に正しく反映される
- クラス S のサンプリング推定が Wilson 区間内に真値（oracle 値）を含む被覆率 ≈ 95%

### 9.5 critic 校正（Phase 3 の前提条件）

- Phase 3 が生成した失敗葉のデータセットを self-play から作り、
  Brier / log loss / calibration curve(ECE) を実戦局面と比較
- §7.2 の視点規約で評価したときに符号反転が起きないこと
- 校正用・閾値選択用・最終評価用のデータを分離

### 9.6 ベンチと A/B

姉妹文書の 78 局面コーパス（`lethal_simple`: 発見 8/58, timeout 76%, 平均 81 ms）を
**そのままベースライン**として使う。比較軸:

| 設定 | 目的 |
|---|---|
| `lethal_simple`（現行） | ベースライン |
| Phase 1+2 / 300 ms | 確定リーサルの発見率と偽陽性 0 の維持 |
| Phase 1+2 / 700 ms | 予算増の効果 |
| Phase 1+2+3 固定閾値 | 実験のみ |
| Phase 1+2+3 critic gate | 最終候補 |

評価は**リーサル発見率ではなく最終勝率**（held-out self-play）。
デッキ別・先後別・残りサイド別・`P_lethal` 帯別に集計し、
p50/p95/p99 レイテンシと 1 試合の総探索時間も測る。

---

## 10. 実装順序

各 Step の終わりに「対象ファイル・理由・テスト結果・残る制約」を報告する（原設計 §18.9）。

| Step | 内容 | 合格条件 |
|---|---|---|
| **1** | Step 0 の確定作業（**実装前**）: プローブを `tools/lethal_bench/` へ移設し数値を再現、78 局面ベンチ再現、`select.effect` 修正の回帰テスト固定、§9.1 の漏洩テスト T1〜T5 を先に書く、capability report §9 の S1-4〜S1-12（B1 ガード / 深さ別・複数デッキ / 多コイン / サイド / 相手選択 / 非決定性の拡張 / ロールアウト拡大 / 視点規約 / レイテンシ計測）を実施 | ベースライン数値が再現。S1-4 が 0 件。既存テスト全緑 |
| **2** | `adapter` / `infokey` / `outcome`（C/M/S 分類）/ `probability`。Phase 1 を新パッケージで再実装 | 78 局面で `lethal_simple` と**同等以上**の発見数、偽陽性 0 |
| **3** | Phase 2（AND-OR + クラス C/M 全列挙 + manual_coin） | oracle 一致。`verify_shuffles` より偽陰性が減ることを 78 局面で確認 |
| **4** | 探索効率化: transposition 共有・Pareto 候補・上界枝刈り・粗視化（oracle で健全性確認） | 同一予算で展開ノード数が減り、発見数が落ちない |
| **5** | Phase 3 確率エンジン（Expectimax・失敗葉・区間・サンプリング） | §9.3/§9.4 全通過。既定 OFF のまま |
| **6** | critic 校正 → `Q_try`/`Q_base` → 採用ゲート | §9.5 の校正が閾値内。ゲートが区間を跨ぐ場合に必ず fallback |
| **7** | 統合 A/B・レイテンシ・提出判定 | held-out self-play で通常方策に非劣化、採用版は改善 |

Step 5 までは Phase 3 を既定 OFF、`configs/rule_lethal_phase12.json` のみを本番候補にする。

---

## 11. ブロッカー・未確定事項

状態の一次情報は [`step0-capability-report.md`](./step0-capability-report.md) §7 にある。

| ID | 内容 | 影響 | 状態と対処 |
|---|---|---|---|
| **B1** | `select.deck != None`（デッキを見て選ぶ）局面では `search_begin` が我々の `your_deck` を**無視して実物のデッキを使う**（`cg/api.py:553`）。実物の**順序**まで使われるなら情報漏洩になる | Phase 2/3 の健全性 | **緩和済み（条件付き）**。実測: `your_deck` 無視 14/14。後続ドロー 430 サンプルは**全て SHUFFLE 後**で非決定的、listing 先頭を引く率も一様期待値と同程度（16 vs 14.3）。実対戦の end-to-end 漏洩テストでも予測一致 0/19。ただし**保証ではない**ので規則 R4 のガードを実装必須にし、Step 1 で「SHUFFLE を挟まないドロー 0 件」を回帰テスト化する |
| **B2** | Kaggle 側の **1 手あたり**時間上限が不明（公式は 1 試合 10 分のみ明記）。`agent()` プロセスが試合中永続するかも未確認 | 予算設計（§6） | **未解決**。§6.1 の「計測してから段階的に上げる」方針に変更し、既定値を先に決めない。`per_decision_cap_ms` は 200 ms 据え置き、安全マージン 30%。トラッカはプロセス再起動時に残り時間を少なく見積もる側へ倒す |
| **B3** | critic が Phase 3 の失敗葉で校正されているか未測定。§7.2 の視点規約での評価も未検証 | Phase 3 全体 | 既定 OFF。§9.5 通過まで有効化しない |
| **B4** | 自ターン中に相手が選ぶ効果の**実例と頻度**が未計測（`yourIndex` が一時的に相手へ移るケース） | Phase 2 の証明健全性 | Step 2 で self-play 中に検出カウンタを回す。0 件でないなら AND/min 実装を Step 3 に含める |
| **B5** | 粗視化（§2.4）の健全性はデッキ依存。フーディンは手札枚数比例打点なので条件 2 は満たすが、他デッキで自明ではない | Phase 2 の列挙規模 | 既定は恒等写像（厳密）。粗視化はデッキプロファイル側の opt-in にし、oracle テストを必須にする |
| **B6** | `attack_features.resolve_damage()` がダメカン配置にも弱点/抵抗を適用している（姉妹文書 §1.4 注） | Goal 分析の打点見積り | 姉妹文書のスコープ。本設計は Goal を**順序付けにのみ**使い、枝刈りには使わないので致命的ではない |
| **B7** | Transformer policy の value head / policy スコアの推論経路が未整備（`policy_weights.npz` 未コミット） | 将来の critic 品質・マクロ順序 | `ValueEvaluator` 契約で差し替え可能にしておく。今回は使わない |
| **B8** | 多コイン効果（「コインを2回投げる」等）で 1 select 1 フリップになるか未検証。現デッキに該当カードが無い | Phase 2 のコイン列挙の完全性 | **未検証**。検出したら Phase 2 は `UNSUPPORTED_EFFECT` で棄却（§3.2）。Step 1 で該当カードを含むデッキで確認 |
| **B9** | サイド取得（`your_prize` の供給順で取るカードが決まるか）が未検証。300 ロールアウトでサイド取得事象が 0 件だった | Phase 3 の失敗枝の精度（勝敗判定自体には影響しない） | **未検証**。Step 1 で計測。列挙できないなら失敗枝はクラス S 扱い |

---

## 12. 既存への影響

| 変更 | 種別 | 影響 |
|---|---|---|
| `ptcg_ai/search/lethal/**` 新規 | 追加 | なし |
| `selector._SEARCH_MODULES` に 1 行 | 追加 | config 未指定なら従来通り `lethal_simple` |
| `configs/rule_lethal_phase12.json` / `rule_lethal_phase3.json` 新規 | 追加 | 既存 `rule_lethal.json` は据え置き |
| `tools/lethal_bench/**` 新規 | 追加 | 提出物に影響しない |
| `hidden_information/search_state_stub.py` | **既に作業ツリーで修正済み**（`select.effect` 計上） | 探索可能局面 +26%。回帰テストを Step 1 で追加 |
| `ptcg_ai/search/lethal_simple.py` | **変更なし** | — |
| `cg/**`, `data/**` | **変更なし** | — |

---

## 13. 原設計の完了条件との対応

| 原設計 §17 | 本設計での担保 |
|---|---|
| 1. Phase 1・2 の確定勝利安全性 | §3.1/§3.2 + §9.2。クラス S 遭遇で証明を打ち切る |
| 2. 追加ドロー後も合法行動を再生成 | chance ノード = マクロ境界（§2.3）。outcome ごとに候補再生成 |
| 3. `P_lethal` と critic 価値を別値で出力 | §5 の型 + §8 の診断キー |
| 4. 完全探索/上下界/サンプリングの区別 | `ChanceClass` で表現し、**実行時検証**（規則 R3/R4）に合格したものだけを列挙可能として扱う（§2.5） |
| 5. 失敗状態から通常方策へ戻した価値 | §7.2 の葉規約 + §1.6 のロールアウト |
| 6. `Q_base` と比較して採用 | §7.3（同一 outcome 重みで平均） |
| 7. 不確実なら保守的区間比較 or fallback | §7.3 |
| 8. 新情報境界を越える固定マクロが無い | §2.3 + 姉妹文書 §3.3 |
| 9. 非公開情報を使わない | §2.2 の `InfoKey` + 規則 R1/R2/R2a/R2b + §9.1 の T1〜T5。B1 は実測で緩和済み、規則 R4 のガードで担保 |
| 10. 最初の 1 手だけ実行・毎回再検証 | §8 |
| 11. 違法手 0・誤検出 0・fallback 100% | §9.2 + `selector` の既存二重チェック |
| 12. held-out self-play と時間計測で判断 | §9.6 / Step 7 |

---

## 付録 Z: 本番接続条件（Step 1-16/1-17 で確定）

### Z.1 Phase 1 を production Agent の action path へ入れる条件

以下を**すべて**満たさない限り、production config で Phase 1 を有効化しない。

| 条件 | 状態 |
|---|---|
| false `PROVEN_WIN` = 0 | ✅ 維持 |
| 不正行動 = 0 | ✅ 維持 |
| replay verification 100% | ✅ 維持 |
| 失敗時 fallback 100% | ✅ 維持 |
| resource leak = 0 | ✅ 維持 |
| **`RNG_NON_INTERFERENCE_VERIFIED`** | ❌ **False** |

`RNG_NON_INTERFERENCE_VERIFIED` は
`tests/integration/lethal/test_rng_noninterference.py` に定数として置く。
定義は「search の有無だけで本番 random outcome が変化しないことを**検証できた**」。
現状は検証手段（seed 設定 / state clone / RNG state 取得）が SDK に無いため `False`。
根拠と再評価条件は `step0-capability-report.md` §11。

**これは "confirmed unsafe" ではなく "not verified" である。**
Phase 1 の proof そのものに問題が見つかったわけではない。
問題は、実戦環境で「探索の有無だけ」を比較できないという評価・安全性上の制約。

### Z.2 現在の既定

```
Phase 1 : default OFF （_SEARCH_MODULES への登録は残す）
Phase 2 : default OFF
Phase 3 : default OFF、critic gate 未接続
R2      : OFF
B8      : deferred
```

`_SEARCH_MODULES` への登録を残す理由: config 一行で offline 解析へ再接続できるようにするため。
登録されていても、既定 config は `lethal_simple` を選ぶので本番では一度も呼ばれない
（`test_default_config_never_invokes_phase1` で固定）。

### Z.3 offline-only fallback（Phase 1 の現在の位置づけ）

Phase 1 は削除しない。**offline research feature** として次の用途に限定する。

```
Phase 1 lethal search
    └─→ offline analysis / saved states / benchmark
            ├─ proof correctness の検証
            ├─ search efficiency の測定
            ├─ capability expansion の評価
            └─ fixture analysis
```

production Agent の action path には入れない。
self-play の勝率比較を本番採用の根拠にしない（§Z.4）。

### Z.4 評価手法の制約

seed control / state clone が存在しないため、**paired self-play は成立しない**。
過去の A/B（24 試合 62.5%→87.5%、held-out 100 試合 74%→76%、rule-based 53.3%→43.3%）は
すべて採用判断の根拠から除外する。誤った測定値ではなく、比較手法が成立していない。

---

## 付録 Y: Phase 1 / Phase 2 の凍結 baseline（Step 1-33）

```
Phase 1:
    correctness = good
    capability  = limited
    production  = NOT SAFE（RNG non-interference 未検証）

Phase 2:
    correctness = good
    capability  = meaningful offline（Phase 1 に無い chance/reveal 系の追加能力）
    scalability = engine-API limited（materialization bound）
    production  = NOT SAFE

Phase 3: not implemented / not evaluated
R2     : safe mechanism confirmed, default OFF
B1     : unsupported by information boundary
Shuffle: unsupported after order becomes hidden
RNG    : non-interference NOT VERIFIED

max_outcomes     = 24
max_combinations = 128
```

### Y.1 `max_outcomes = 24` の意味

**「24 より大きい outcome が危険」という意味ではない。** 正しくは:

> 現在の materialization architecture では、大きな outcome 集合を処理すると
> 他の探索枝へ使える budget を圧迫し、**総合的な `PROVEN_WIN` 数が低下する**ため、
> 24 を baseline として採用する。

実測（Step 1-31、valid 185 件、Phase 2 `PROVEN_WIN` 数）:

| max_outcomes | 100ms | 500ms | 2000ms |
|---|---|---|---|
| **24** | **100** | **104** | **106** |
| 128 | 96 | 103 | 105 |
| 512 | 91 | 98 | 102 |

「24 が理論最適」ではなく **現行 engine API・corpus・budget における測定上の baseline**。

### Y.2 情報境界の正式仕様

```
DRAW → SHUFFLE     = 許可（そのステップのドローは供給順どおり）
SHUFFLE → DRAW     = UNKNOWN（以降の順序は未知）
経路上で shuffle 済み → 以降の order-dependent draw は UNKNOWN
root で select.deck != None（B1） = UNKNOWN（実デッキ由来の新情報）
```

`KnownRootListingDecision` は**作らない**（Step 1-27: 供給デッキを変えても
root listing が 12/12 で不変 = 実デッキ由来と実証済み）。

### Y.3 「できない」の 3 分類

UNKNOWN を一括で「未実装」と扱わない。

| 分類 | 例 | 対処 |
|---|---|---|
| **Information-boundary limitation** | B1、shuffle 後の hidden order | **緩和しない**（安全性のための正しい拒否） |
| **Engine API limitation** | state clone なし、outcome injection なし、shared prefix 不可 | SDK 側の改善待ち（capability report §14.1） |
| **Solver capability limitation** | 予算切れ・深さ上限 | 改善余地あり。ただし実測では 22% のみ |

---

## 付録 X: UNKNOWN の正式 taxonomy（Step 1-34 で凍結）

**UNKNOWN を「solver が弱い」と一括で表現しない。** 4 つの異なる原因が混在する。

```
UNKNOWN
├─ Information Boundary        26 件 (38%)  ← 緩和しない
│   ├─ B1 root deck reveal         21
│   └─ post-shuffle hidden order    5
├─ Engine / API Limitation     18 件 (26%)  ← SDK 改善待ち
│   └─ max_outcomes（materialization bound 由来）
├─ Safety Limit                 3 件 (4%)   ← 意図的な上限
│   └─ max_combinations
├─ Experimental capability      6 件 (9%)   ← R2 ON で 0 になる
│   └─ known listing decision（既定 OFF）
└─ Solver Capability           15 件 (22%)  ← 唯一の改善余地
    ├─ chance depth 上限            6
    ├─ 予算切れ                     6
    └─ 深さ上限                     3
```

計測条件: valid 185 件 / R2 OFF / budget 500ms / `max_depth=8` / `max_chance_depth=1`。
Phase 2 の内訳は `PROVEN_WIN` 101 / `PROVEN_NO_WIN` 16 / `UNKNOWN` 68。

**UNKNOWN を減らすこと自体を目的にしない。** 78% は情報境界・API 制約・
意図的な安全上限であり、減らそうとすると安全性を損なうか
（B1 / shuffle）、実測上むしろ能力が下がる（`max_outcomes`）。
判断基準は「**安全な確定能力が増えるかどうか**」。

---

## 付録 W: 凍結 baseline 一覧（Step 1-34）

| component | correctness | capability | limitation | production |
|---|---|---|---|---|
| **Phase 1** | good（false PW/PNW とも 0） | 公開を伴わない確定手順。exact-positive 80 件中 68 件（500ms） | 公開枝を展開しない設計。反復深化コストで p50 26ms | **OFF / NOT SAFE** |
| **Phase 2** | good（同上、NO_WIN 契約 green） | Phase 1 に無い chance/reveal 系の追加確定能力 **9 件**（9/9 replay green） | **materialization bound**（engine API の構造的制約） | **OFF / NOT SAFE** |
| **R2** | safe（115/115 applicable、replay 3/3） | `UNSUPPORTED_EFFECT` 47→0、新規 WIN 3 | 中間 listing のみ。root には適用不可 | **OFF**（実装済み・既定 OFF） |
| **B1** | – | – | **information boundary**。root listing は実デッキ由来（12/12 実証） | 緩和しない |
| **Shuffle** | – | `DRAW→SHUFFLE` は許可 | `SHUFFLE→DRAW` は hidden order（拒否 1,886/1,886 が正当） | 緩和しない |
| **Phase 3** | – | 未実装・未評価 | materialization bound を回避しない | **OFF** |
| **B8** | – | engine 表現は確認済み | positive fixture 0 で評価不能 | deferred |

```
max_outcomes     = 24    （現行 corpus / budget / architecture における実測 baseline）
max_combinations = 128   （意図的な安全上限。correctness blocker ではない）
RNG_NON_INTERFERENCE_VERIFIED = False
```
