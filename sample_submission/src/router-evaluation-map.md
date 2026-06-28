# router.py の MAIN / ATTACK 評価メモと後続分岐

`sample_submission/src/decision/router.py` から見た、現在実装されている `MAIN` と `ATTACK` の評価フローと、その後に発生する代表的な後続分岐の入口をまとめたメモです。

- 対象は「今の実装で実際に行っていること」のみ
- `MAIN` は「どの行動カテゴリを採用するか」と「そのカテゴリ内でどの候補を選ぶか」の 2 段階
- `ATTACK` は `MAIN` の攻撃評価をそのまま再利用している
- `MAIN` や `ATTACK` でカードやワザを決めたあと、追加の対象選択やサーチ先選択が必要なら、別の `SelectContext` で再度 `router.py` に入る

---

## 関連ファイル構成

`src/` 配下の現在の構成は次のとおりです（`__init__.py` / `__pycache__` / `tests/` は省略）。

```text
sample_submission/src/
├── agent.py                       # main.py からの入口。初回はデッキ、通常時は router へ
├── router-evaluation-map.md       # このメモ
├── decision/                      # 「何を選ぶか」を決める層
│   ├── router.py                  # obs.select.context を見て担当ハンドラへ振り分ける
│   ├── fallback.py                # 合法手から無難に選ぶ最終フォールバック
│   ├── handlers/                  # SelectContext ごとの入口（薄いディスパッチャ）
│   │   ├── main_turn.py           #   MAIN: 行動カテゴリ比較の本体
│   │   ├── attack_turn.py         #   ATTACK: 攻撃候補の比較
│   │   ├── setup_turn.py          #   初期配置（バトル場・ベンチ）
│   │   ├── switch_turn.py         #   入れ替え・きぜつ後の復帰
│   │   ├── evolution_turn.py      #   進化・退化の対象選択
│   │   ├── energy_tool_turn.py    #   エネ/どうぐの付け替え・トラッシュ
│   │   ├── effect_choice_turn.py  #   特性/効果の順番、封じるワザ
│   │   ├── damage_target_turn.py  #   ダメージ/ダメカン/回復の対象
│   │   ├── count_turn.py          #   枚数・個数の数値選択
│   │   ├── special_condition_turn.py #   状態異常の付与/回復対象
│   │   ├── yes_no_turn.py         #   Yes/No 系の分岐
│   │   └── card_move_turn.py      #   カード移動系を card_move/ へ振り分ける
│   ├── card_move/                 # カード移動先（手札/山札/サイド/トラッシュ/場）の選択ロジック
│   │   ├── common.py              #   移動候補の共通解析・ユーティリティ
│   │   ├── bench_field.py         #   TO_BENCH / TO_FIELD
│   │   ├── hand_like.py           #   TO_HAND
│   │   ├── hidden_zone.py         #   TO_DECK / TO_DECK_BOTTOM / TO_PRIZE
│   │   ├── discard.py             #   DISCARD
│   │   ├── not_move_or_look.py    #   NOT_MOVE / LOOK
│   │   ├── to_deck.py             #   山札へ戻す系の評価補助
│   │   └── to_hand_eval.py        #   手札に加える価値の評価
│   ├── evaluation/                # 盤面・攻撃・エネを数値化する低レベル評価プリミティブ
│   │   ├── attack_features.py     #   ワザ情報の解決・打点算出
│   │   ├── attack_profiles.py     #   既知ワザの効果プロファイル参照
│   │   ├── board_features.py      #   アタッカー評価・危険度などの盤面特徴
│   │   ├── energy_requirements.py #   必要エネと不足分の計算
│   │   └── switch_eval.py         #   入れ替え/逃げ先の評価（retreat/switch 共通）
│   └── main_turn_parts/           # MAIN ハンドラの内部実装
│       ├── buckets.py             #   option を行動カテゴリに仕分け
│       ├── proposals.py           #   提案データ構造と最良提案の選択
│       ├── weights.py             #   カテゴリ間比較の基本重み
│       ├── energy_eval.py         #   エネ貼り先の評価
│       └── priorities/            #   カテゴリごとの「今やるべきか」提案
│           ├── draw.py
│           ├── board.py
│           ├── ability.py
│           ├── energy.py
│           ├── retreat.py
│           ├── attack.py
│           └── end_turn.py
└── knowledge/                     # カードの静的知識（状態に依存しない参照データ）
    ├── card_cache.py              #   CardData / Attack のロードとキャッシュ
    ├── deck_profiles.py           #   ポケモン/ワザ/カードのプロファイル
    └── meta_decks.py              #   環境デッキ定義
```

### 各フォルダ/サブパッケージにまとめているもの

- `decision/handlers/`
  - **「いま何を選ばせているか」(`SelectContext`) ごとの入口**を集約。
  - 各 `*_turn.py` は薄いディスパッチャで、実際の評価は `card_move/` `evaluation/` `main_turn_parts/` などへ委譲する。
  - `router.py` はここのハンドラへ振り分けるだけ。新しい `SelectContext` を扱うときは、まずここに入口を足す。
- `decision/card_move/`
  - **カードの「移動先」を選ぶ処理のファミリ**（手札・山札・サイド・トラッシュ・ベンチ/場）。
  - もともと `decision/` 直下に `card_move_*` として散らばっていたものを 1 パッケージに集約。`card_move_turn.py`（振り分け）は `handlers/` 側に置く。
- `decision/evaluation/`
  - **状態を数値スコアに変換する低レベル部品**。打点・必要エネ・盤面の危険度・逃げ先評価など。
  - 特定の `SelectContext` に依存せず、`handlers/` や `card_move/` や `main_turn_parts/` から共通して呼ばれる。
- `decision/main_turn_parts/`
  - **MAIN ハンドラ専用の内部実装**。カテゴリ仕分け（`buckets`）・重み（`weights`）・提案（`priorities/`, `proposals`）・エネ評価（`energy_eval`）。
  - `handlers/main_turn.py` から使われる。MAIN の挙動を変えるときの主戦場。
- `knowledge/`
  - **盤面状態に依存しないカードの静的知識**。カードデータのキャッシュ、プロファイル、環境デッキ。
  - `decision/` 配下の各所から参照される、最下層の参照データ。

> 依存の向きは原則「上位 → 下位」で一方向。
> `handlers/` → (`card_move/`, `main_turn_parts/`, `evaluation/`, `knowledge/`, `fallback`) → `evaluation/` → `knowledge/`。
> 逆向き（`evaluation/` から `handlers/` を import するなど）は作らない。

---

## ルーティングの入口

処理の流れは次の通りです。

```text
main.py
  -> src/agent.py
    -> src/decision/router.py
      -> MAIN のとき    src/decision/handlers/main_turn.py
         -> 必要なら後続の SelectContext で router.py に戻る
      -> ATTACK のとき  src/decision/handlers/attack_turn.py
         -> 必要なら後続の SelectContext で router.py に戻る
```

`router.py` は `obs.select.context` を見て分岐します。

- `SelectContext.MAIN`
  - `choose_main_action(obs)` を呼ぶ
- `SelectContext.ATTACK`
  - `choose_attack_action(obs)` を呼ぶ

このメモでは `MAIN` / `ATTACK` の評価を中心に扱い、後続分岐は「どこへ流れるか」の入口だけ補足します。

---

## MAIN / ATTACK の後に来る代表的な分岐

`MAIN` で Supporter / Item / Tool / Ability を使ったあとや、`ATTACK` を選んだあとに追加選択が必要なら、次のような `SelectContext` で再度 `router.py` が呼ばれます。

- サーチ先や戻し先、見たカードの処理
  - `TO_HAND` / `TO_DECK` / `TO_DECK_BOTTOM` / `TO_PRIZE` / `DISCARD` / `LOOK` / `NOT_MOVE`
  - `decision/handlers/card_move_turn.py`
- ベンチ・場に出す先の選択
  - `TO_BENCH` / `TO_FIELD`
  - `decision/handlers/card_move_turn.py`
- ダメージ先や回復先などの対象選択
  - `DAMAGE_COUNTER` / `DAMAGE_COUNTER_ANY` / `DAMAGE` / `REMOVE_DAMAGE_COUNTER` / `HEAL` / `EFFECT_TARGET`
  - `decision/handlers/damage_target_turn.py`
- 特性や効果の順番、封じるワザの選択
  - `SKILL_ORDER` / `DISABLE_ATTACK`
  - `decision/handlers/effect_choice_turn.py`
- エネルギーやどうぐの付け替え先
  - `ATTACH_FROM` / `ATTACH_TO` / `DETACH_FROM` / `DISCARD_ENERGY_CARD` / `DISCARD_TOOL_CARD` / `SWITCH_ENERGY_CARD` / `DISCARD_CARD_OR_ATTACHED_CARD` / `DISCARD_ENERGY` / `TO_HAND_ENERGY` / `TO_DECK_ENERGY` / `SWITCH_ENERGY`
  - `decision/handlers/energy_tool_turn.py`
- 枚数やダメカン個数の選択
  - `DRAW_COUNT` / `DAMAGE_COUNTER_COUNT` / `REMOVE_DAMAGE_COUNTER_COUNT`
  - `decision/handlers/count_turn.py`

補足:

- ここで挙げた後続分岐は、`MAIN` / `ATTACK` のようなカテゴリ比較ではなく、「その追加選択でどれを選ぶか」を処理する担当です。
- まだ TODO が多いハンドラもあり、現状は `choose_random_legal_action(obs)` に落ちるものがあります。

---

## MAIN 全体像

`main_turn.py` の役割は次の 3 ステップです。

1. `bucket_main_options(obs)` で `obs.select.option` を行動カテゴリごとに分ける
2. 各カテゴリの `propose_*()` に「このカテゴリを今やるべきか」を提案させる
3. `choose_best_proposal()` で `score` 最大の提案を採用する

### MAIN の比較は 2 段階

`MAIN` では次の 2 段階の比較が入っています。

1. 各カテゴリの中で「どの option を使うか」を決める
2. そのカテゴリを今ターンの行動として採用するかを、カテゴリ間で `score` 比較する

例:

- `attack.py` は「どの攻撃が一番よいか」を細かく比較する
- ただし `MAIN` のカテゴリ比較では、攻撃カテゴリ全体の点数は固定で `30`

つまり、攻撃の中で一番よい攻撃を選べても、`draw` や `evolve` の方が高得点ならそちらが優先されます。

### MAIN の基本重み

`weights.py` の基本重みは次の通りです。

| key | score |
| --- | ---: |
| `draw_or_search` | 70 |
| `pokemon_play` | 62 |
| `evolve` | 60 |
| `ability` | 58 |
| `board_item` | 50 |
| `stadium` | 48 |
| `tool` | 46 |
| `energy` | 44 |
| `retreat` | 40 |
| `attack` | 30 |
| `end` | 5 |

### 同点時の優先順位

`choose_best_proposal()` は `score` だけで `max()` を取っています。  
Python の `max()` は同点なら先に出てきた要素を返すので、同点時は `main_turn.py` に並んでいる順がそのまま優先順位です。

同点時の優先順:

1. `propose_draw_or_search_action`
2. `propose_pokemon_or_evolve_action`
3. `propose_board_item_action`
4. `propose_ability_action`
5. `propose_energy_action`
6. `propose_retreat_action`
7. `propose_attack_action`
8. `propose_end_action`

---

## MAIN の各 proposal が見ているもの

### 1. `propose_draw_or_search_action`

ファイル:

- `decision/main_turn_parts/priorities/draw.py`

まず手札枚数で「今ドロー系を欲しがるか」を決めます。

- 手札 `0-2` 枚: `hand_bonus = 30`
- 手札 `3-4` 枚: `hand_bonus = 15`
- 手札 `5-6` 枚: `hand_bonus = 0`
- 手札 `7` 枚以上: 提案しない

次に `supporter_play` をテキストで分類します。

- `draw`
  - `"draw"` を含む
- `search`
  - `"search"` または `"look at"` を含む
- `discard_draw`
  - `"draw"` と `"discard"` または `"shuffle"` を含む
- `other`
  - 上記以外

Supporter の優先順は次の通りです。

1. `draw`
2. `search`
3. `discard_draw`
4. `other`

Supporter を採用する場合の加点:

- `draw`: `+10`
- `search`: `+5`
- `discard_draw`: `+0`

さらに `discard_draw` については、

- ほかに使える非ドロー系カードがある
  - `pokemon_play`
  - `evolve`
  - `item_play`
  - `tool_play`
  - `stadium_play`
  - `ability`
  - `attach`

このどれかがあると `-20` の追加ペナルティが入ります。

最終スコア:

- Supporter: `70 + hand_bonus + type_bonus`
- Item の `draw/search`: `70 + hand_bonus - 5`

要するに、

- 手札が細いほどドローを強く優先
- 同じドロー系でも `draw > search > discard_draw`
- 捨てて引く系は、まだ他にやることがあるなら後ろに回す

### 2. `propose_pokemon_or_evolve_action`

ファイル:

- `decision/main_turn_parts/priorities/board.py`

ここはかなり単純です。

- `evolve` があれば最優先で最初の 1 件を返す
  - `score = 60 + 20 = 80`
- そうでなく `pokemon_play` があり、ベンチに空きがあれば最初の 1 件を返す
  - `score = 62 + min(bench_space, 2) * 2 + 20`

つまり今の実装では、

- 進化は中身を比較せず「あるなら強い」
- ポケモン展開も中身比較はせず、ベンチ空きがあるほど少しだけ加点

### 3. `propose_board_item_action`

ファイル:

- `decision/main_turn_parts/priorities/board.py`

このカテゴリは `stadium` / `item` / `tool` をまとめて扱います。

優先順は次の通りです。

1. にげるための Tool
2. Stadium
3. Item
4. Tool

#### 3-1. にげるための Tool

次の条件をすべて満たすと、逃げコスト軽減 Tool を最優先します。

- まだ `retreated` していない
- `obs.logs` から、今のバトルポケモンが前ターンに
  - `"can't use"` / `"cannot use"`
  - かつ `"next turn"`
  を含む攻撃を使っていた
- ベンチに、現在の active より「今出せる最大打点」が高いポケモンがいる
- `tool_play` の中に、`"retreat cost"` と
  - `"less"`
  - `" 0"`
  - `"no "`
  - `"free"`
  - `"reduc"`
  のどれかを含む Tool がある

このときのスコア:

- `46 + 30 = 76`

#### 3-2. Stadium

- `stadium_play` がある
- かつ現場にまだ Stadium が出ていない

このとき:

- `score = 48 + 10 = 58`

#### 3-3. Item / Tool

上の特殊条件がなければ先頭を返します。

- `item_play`: `score = 50`
- `tool_play`: `score = 46`

### 4. `propose_ability_action`

ファイル:

- `decision/main_turn_parts/priorities/ability.py`

ここは最も単純です。

- `ability` があれば最初の 1 件を返す
- `score = 58`

Ability の内容比較はまだしていません。

### 5. `propose_energy_action`

ファイル:

- `decision/main_turn_parts/priorities/energy.py`
- `decision/main_turn_parts/energy_eval.py`

まず次の条件なら提案しません。

- すでにこのターン `energyAttached` 済み
- `attach` 候補がない

候補がある場合は `choose_best_attach_option()` で最善の貼り先を選びます。  
ここでの評価は attach 候補同士の比較です。

加点:

- この貼りで active が今すぐ攻撃可能になる: `+30`
- この貼りで主力技まであと 1 エネになる: `+20`
- 貼り先が主力アタッカー: `+15`
- 主力技の必要エネや今出せる打点が改善する: `+10`
- active の逃げコスト支払いにも近づく: `+5`

減点:

- 貼り先が次ターンに倒されやすい: `-20`
- 主力でなく、貼っても主力技まで遠いベンチ: `-10`
- すでに十分攻撃でき、貼っても打点改善がない: `-5`

採用条件:

- attach 候補評価の `score > 0`

ただし `MAIN` のカテゴリ比較に戻すと、Energy カテゴリ全体の点数は固定です。

- `score = 44`

つまり「どこに貼るか」は細かく評価する一方で、「今 Energy を貼るかどうか」は `draw` や `evolve` などより低い固定重みで比較されます。

### 6. `propose_retreat_action`

ファイル:

- `decision/main_turn_parts/priorities/retreat.py`
- `decision/evaluation/switch_eval.py`

まず次の条件なら提案しません。

- `retreat` 候補がない
- まだ攻撃候補 `attack` がある

つまり今の実装では、攻撃できるなら `retreat` proposal 自体を出しません。

`choose_best_retreat_option()` では、実際には「今の active と、逃げた先の最良候補の差」を見ています。

逃げ先ポケモンの評価では主に次を見ます。

- 今すぐ攻撃できるか
  - `+50 + best_damage_now // 10`
- 主力技まであと 1 エネか
  - `+25`
- 主力技まで 2 エネ不足か
  - `+10`
- 主力技まで 3 エネ以上足りないか
  - `-8`
- `attacker_priority_score(...) // 8`
- HP に応じた加点
  - `+ hp // 20`
- 次ターンに倒されやすいか
  - `-18`

そのうえで retreat 候補の評価は次の形です。

- `score = 逃げ先評価 - 現active評価 - retreat_cost * 10`

追加補正:

- 逃げ先が今すぐ攻撃できる: `+15`
- 逃げ先が次ターン主力技に届き、今の active はまだ遠い: `+8`
- 今の active は危険で、逃げ先は危険でない: `+10`
- 今の active がすでに次ターン主力技に届く: `-12`

採用条件:

- `score_gap > 0`
- 位置改善がある
- `score > 0`
- かつ「重すぎる retreat」ではない
  - `retreat_cost >= 3`
  - かつ逃げ先が今攻撃不可
  - かつ次ターン主力技にも届かない

`MAIN` のカテゴリ比較では Retreat カテゴリ全体の点数は固定です。

- `score = 40`

### 7. `propose_attack_action`

ファイル:

- `decision/main_turn_parts/priorities/attack.py`

まず `choose_best_attack_option()` で attack 候補の中から最善手を選びます。  
ここは `MAIN` 用でも `ATTACK` 用でも同じ関数を使います。

#### 攻撃候補ごとの評価値

各攻撃は次の情報に分解されます。

- `knock_out`
  - 相手 active を倒せるか
- `immediate_damage`
  - 今すぐ入る打点
- `overflow_damage`
  - 倒す場合の過剰打点
- `effect_score`
  - 追加効果の評価

#### 攻撃の並べ方

KO できる攻撃は、次の順で優先されます。

1. KO できる
2. `effect_score` が高い
3. `overflow_damage` が少ない
4. `immediate_damage` が高い
5. `option_index` が若い

KO できない攻撃は、次の順です。

1. `pressure_score = immediate_damage + effect_score`
2. `effect_score`
3. `immediate_damage`
4. `option_index` が若い

#### `immediate_damage`

基本は印字打点そのままです。  
ただし `deck_profiles.py` 側に攻撃プロファイルがあり、

- `fails_without_requirement = True`
- かつ必要条件を満たしていない

このときは `immediate_damage = 0` になります。

#### `effect_score`

既知プロファイルがある攻撃は、テキスト直読みではなくプロファイルを使います。

主な加点:

- ベンチダメージ: `+6` から最大 `+12`
- 特殊状態付与: `+6`
- ドロー: 最大 `+6`
- 相手を入れ替える: `+6`
- ベンチへのエネ加速: `+8` から最大 `+14`

主な減点:

- コイン依存: `-6`
- 自分のエネ discard: `-8 * 枚数`
- 自傷: `-max(4, self_damage // 10)`
- 次ターン攻撃不可: `-14`

未知攻撃はテキスト正規化後にキーワード評価します。

主な加点キーワード:

- `"your opponent's benched pokemon"`: `+10`
- `"is now paralyzed"`: `+14`
- `"is now asleep"` / `"is now confused"`: `+10`
- `"is now poisoned"` / `"is now burned"`: `+8`
- `"discard an energy from your opponent's active pokemon"`: `+10`
- `"draw "`: `+6`
- `"search your deck"`: `+6`
- `"attach"` かつ `"energy"`: `+8`

主な減点キーワード:

- `"during your next turn, this pokemon can't use attacks"`
- `"during your next turn, this pokemon can't use "`
- `"discard all energy from this pokemon"`
- `"discard (a|数) ... energy from this pokemon"`
- `"this pokemon does X damage to itself"`

最終的に `MAIN` へ返す proposal のカテゴリ点は固定です。

- `score = 30`

### 8. `propose_end_action`

ファイル:

- `decision/main_turn_parts/priorities/end_turn.py`

ここも単純です。

- `end` 候補があれば最初の 1 件を返す
- `score = 5`

---

## ATTACK でやっていること

ファイル:

- `decision/handlers/attack_turn.py`
- `decision/main_turn_parts/priorities/attack.py`

`ATTACK` コンテキストでは `choose_attack_action(obs)` が呼ばれます。

やっていることは次の通りです。

1. `choose_best_attack_option(obs)` を呼ぶ
2. 最善の攻撃 index が返ればそれを使う
3. 返らなければ `choose_random_legal_action(obs)` に落とす

重要なのは、`ATTACK` では `MAIN` と違ってカテゴリ比較がないことです。

- `MAIN`
  - 攻撃カテゴリは固定 `30` 点で他カテゴリと競合する
- `ATTACK`
  - 攻撃候補同士だけを `choose_best_attack_option()` で比較する

つまり `ATTACK` の中身自体は `MAIN` の攻撃評価と同じですが、`ATTACK` では純粋に「どの攻撃が一番よいか」だけを見ています。

---

## ざっくりした現在の優先傾向

今の実装をざっくり言うと次の傾向です。

- 手札が薄いと `draw/search` がかなり強い
- `evolve` / `pokemon_play` は中身比較より「できるなら前向き」に寄っている
- `ability` はまだ内容を見ず固定重み
- `energy` / `retreat` / `attack` はカテゴリ内評価は細かいが、カテゴリ間では固定重み
- `attack` は `MAIN` ではかなり後ろ寄りで、今すぐ勝ち筋がない限り他行動に負けやすい

---

## 補足

このメモは、今後 `MAIN` の proposal が増えたときに更新が必要です。特に次を変えたら追記対象です。

- `main_turn.py` の proposal 順
- `weights.py` の重み
- `energy_eval.py` / `switch_eval.py` / `attack.py` の採点式
- `draw.py` / `board.py` のテキスト分類ルール
