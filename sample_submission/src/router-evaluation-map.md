# router.py の MAIN / ATTACK 評価メモ

`sample_submission/src/decision/router.py` から見た、現在実装されている `MAIN` と `ATTACK` の評価フローをまとめたメモです。

- 対象は「今の実装で実際に行っていること」のみ
- `MAIN` は「どの行動カテゴリを採用するか」と「そのカテゴリ内でどの候補を選ぶか」の 2 段階
- `ATTACK` は `MAIN` の攻撃評価をそのまま再利用している

---

## 関連ファイル構成

`MAIN` / `ATTACK` を追うときに主に見るファイルは次のとおりです。

```text
sample_submission/src/
├── agent.py
├── router-evaluation-map.md
├── decision/
│   ├── router.py
│   ├── main_turn.py
│   ├── attack_turn.py
│   ├── switch_eval.py
│   ├── evaluation/
│   │   ├── attack_features.py
│   │   ├── board_features.py
│   │   └── energy_requirements.py
│   └── main_turn_parts/
│       ├── buckets.py
│       ├── proposals.py
│       ├── weights.py
│       ├── energy_eval.py
│       └── priorities/
│           ├── draw.py
│           ├── board.py
│           ├── ability.py
│           ├── energy.py
│           ├── retreat.py
│           ├── attack.py
│           └── end_turn.py
└── knowledge/
    ├── card_cache.py
    └── deck_profiles.py
```

---

## ルーティングの入口

処理の流れは次の通りです。

```text
main.py
  -> src/agent.py
    -> src/decision/router.py
      -> MAIN のとき    src/decision/main_turn.py
      -> ATTACK のとき  src/decision/attack_turn.py
```

`router.py` は `obs.select.context` を見て分岐します。

- `SelectContext.MAIN`
  - `choose_main_action(obs)` を呼ぶ
- `SelectContext.ATTACK`
  - `choose_attack_action(obs)` を呼ぶ

このメモではこの 2 つだけを扱います。

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
- `decision/switch_eval.py`

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

- `decision/attack_turn.py`
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
