# 新しいデッキに対応するときのルール

`ptcg_ai/`（担当B: 汎用ロジック）を一切変更せず、`decks/`配下にデータを追加・差し替えるだけで
別デッキに対応できるように設計されている。このドキュメントは、次に別デッキ（または現デッキの
再構築）に取り組む人向けの手順書。背景の設計思想は
[new-deck-src-structure.md](../plans/rule_based/new-deck-src-structure.md)、
プロファイルの型定義そのものは `ptcg_ai/shared/profile_types.py` を正とする
（このガイドは「使い方」のまとめであり、型の一次情報源ではない）。

---

## 0. 大前提: 担当A/Bの境界ルール

- **担当B（`ptcg_ai/`配下）はカードID・カード名を直接書かない。** 判断材料は必ず
  `ptcg_ai.shared.profile_registry`経由で、担当Aが用意した数値・真偽値・関数として受け取る。
- **担当A（`decks/<deck_name>/`配下）はカードID・カード名を自由に書いてよい。** 条件判定の
  ロジック（`usage_condition`などの関数）もここに書く。「Bはデータの形だけ知っていて、中身の
  意味（どのカードが何なのか）は一切知らない」を保つのが唯一のルール。
- 判断に新しい種類の情報が必要になったら、まず`ptcg_ai/shared/profile_types.py`の型を
  拡張してから、`decks/<deck_name>/`側でデータを埋める。逆方向（Aが勝手にBのファイルを
  読みに行く、Bが特定カードIDをハードコードする）はやらない。

---

## 1. ディレクトリ構成とデッキの切り替え方

```text
decks/
├── active.py              ← デッキ切り替えの唯一の変更箇所
└── <deck_name>/            ← デッキごとに1ディレクトリ（例: new_deck）
    ├── deck_plan.py         デッキ方針（⑥）
    ├── pokemon_profiles.py  ポケモン別プロファイル（⑦）
    ├── attack_profiles.py   ワザ別プロファイル（⑦）
    ├── item_profiles.py     グッズ別プロファイル（⑦）
    ├── supporter_profiles.py サポート別プロファイル（⑦）
    ├── tool_profiles.py     どうぐ別プロファイル（⑦）
    ├── stadium_profiles.py  スタジアム別プロファイル（⑦）
    └── energy_profiles.py   エネルギー別プロファイル（⑦）
```

新しいデッキに対応する手順:

1. `decks/<new_deck_name>/`を作り、上記8ファイルを用意する（中身の書き方は2〜5節）。
2. `sample_submission/deck.csv`を新デッキの60枚に差し替える（1行1カードID、合計60行）。
   デッキルール（4枚制限・ACE SPEC 1枚・Basicポケモン必須）を満たしているか確認すること。
3. `decks/active.py`のimport元を新しいディレクトリ名に差し替える。**変更箇所はここだけ**で、
   `ptcg_ai/`配下は一切触らない。
4. 動作確認（7節）を行う。

---

## 2. `deck_plan.py`（クラスタ⑥ デッキ方針）

### 必須データ

`ptcg_ai.shared.profile_registry.get_deck_plan()`が読みに行く、最低限必要なもの
（`_build_deck_plan()`参照）。データ形式は自由（`PriorityEntry`のような理由付きの
独自データクラスでも、単純な`list[int]`でも良い）だが、以下の**名前の変数**として
モジュールに存在すれば`profile_registry`側が自動で拾う。

| 変数名 | 型のイメージ | 内容 |
|---|---|---|
| `MAIN_ATTACKER` | `card_id`を持つオブジェクト | 主力アタッカー1体 |
| `SUB_ATTACKERS` | `card_id`を持つオブジェクトのlist | サブアタッカー |
| `OPENING_BENCH_PRIORITY` | `card_id`を持つオブジェクトのlist | 初手・展開の優先順位（上が優先） |
| `EVOLUTION_PRIORITY` | 同上 | 進化の優先順位 |
| `ENERGY_TARGET_PRIORITY` | 同上 | エネルギーを付けたいポケモンの優先順位 |
| `SEARCH_PRIORITY` | 同上 | サーチで最初に探すべきカードの優先順位 |
| `PROTECT_CARDS` | 同上 | 捨てたくないカードの集合 |

### 任意データ（無ければ使われないだけで、エラーにはならない）

| 変数名 | 内容 | 使う判断ロジック |
|---|---|---|
| `ENERGY_REQUIRED_COUNT` | `dict[card_id, int]`。攻撃に必要なエネルギー総数の上限 | `energy_eval.py`（これ以上エネルギーを付けない） |
| `ENERGY_CARD_PRIORITY_RULES` | `list[EnergyPriorityRule]`（3節参照） | `priorities/energy.py`・`energy_tool_turn.py`（複数エネルギー種から選ぶ） |
| `KO_REPLACEMENT_PRIORITY` | `card_id`を持つオブジェクトのlist（重複可） | `pokemon_value.py`（きぜつ後の後継選び） |
| `ENERGY_RECYCLE_TARGET_CARD_ID` / `ENERGY_RECYCLE_CARD_ID` / `ENERGY_RECYCLE_BACKUP_ITEM_ID` | それぞれ`card_id` | `energy_eval.py`（ACE SPECエネルギー等の周回コンボ、4節参照） |

`profile_registry._build_deck_plan()`は`getattr(plan, "XXX", None)`でこれらを読むため、
無いデッキでは自動的に「その機能を使わない」扱いになる（クラッシュしない）。

### やってはいけないこと

- `win_condition_by_prize`は自由記述の文字列（サイド枚数の日本語レンジ→勝ち筋メモ）のままで
  よい。どの判断ロジックからも参照されない前提のメモ欄。
- `KO_REPLACEMENT_PRIORITY`で同じ`card_id`を複数回登場させて良いのは、**エネルギー充足度で
  優先度を変えたい**場合だけ（`pokemon_value._ko_replacement_rank`が
  `ENERGY_REQUIRED_COUNT`との差分で自動的にどちらのエントリを使うか判定する）。
  それ以外の理由での重複は入れない。

---

## 3. `*_profiles.py`（クラスタ⑦ カード別プロファイル）

`ptcg_ai/shared/profile_types.py`の各`XxxProfile`データクラスに沿って、
`card_id -> Profile`（`attack_profiles.py`だけは`attack_id -> AttackProfile`）の
辞書`PROFILES`を埋める。

| ファイル | 型 | 主なフィールド |
|---|---|---|
| `pokemon_profiles.py` | `PokemonProfile` | `role`/`role_score`/`bench_value`/`combo_with`/`has_ability`/`ability_category`/`ability_priority` |
| `attack_profiles.py` | `AttackProfile` | `bench_snipe`/`inflicts_special_condition`/`draws_cards`/`disables_next_attack` |
| `item_profiles.py` | `ItemProfile` | `category`/`priority`/`usage_condition`（任意） |
| `supporter_profiles.py` | `SupporterProfile` | 同上 |
| `tool_profiles.py` | `ToolProfile` | `category`/`priority`（`usage_condition`は無し） |
| `stadium_profiles.py` | `StadiumProfile` | `category`/`priority`/`usage_condition`（任意） |
| `energy_profiles.py` | `EnergyProfile` | `category`（`"basic"` / `"special"`） |

`category`（`EffectCategory`）は`profile_types.py`で列挙されている7語彙
（`search`/`draw`/`heal`/`disruption`/`setup`/`lock`/`other`）から選ぶ。迷ったら`other`。

### `usage_condition`（「今このカードを使うべきか」の判定関数）

ItemProfile/SupporterProfile/StadiumProfileが持てる任意フィールド。
`Callable[[UsageContext], bool]`型で、`UsageContext`（`profile_types.py`で定義、
`board_evaluation.usage_context.build_usage_context`が盤面から組み立てる）だけを見て
`bool`を返す関数を書く。**「単に手札にあれば使う」程度のカードには付けなくてよい**
（`None`のままなら常に使用可扱いになる）。条件が要るカードだけ書く。

```python
# item_profiles.py の例
def _rare_candy_condition(ctx: UsageContext) -> bool:
    return (
        ctx.own_active_id == _CASEY_ID
        and _KADABRA_ID not in ctx.own_hand_ids
        and _ALAKAZAM_ID in ctx.own_hand_ids
    )

PROFILES: dict[int, ItemProfile] = {
    1079: ItemProfile(category="setup", priority=0.9, usage_condition=_rare_candy_condition),
}
```

`UsageContext`に無い情報が必要になったら、`profile_types.py`にフィールドを足して
`board_evaluation/usage_context.py`の`build_usage_context`を拡張する（担当Bへ相談）。
勝手に`Observation`/`State`を直接読みに行く関数を書かない。

---

## 4. 条件付きロジックの3パターンの使い分け

このコードベースには、Aが「データ」ではなく「小さな判定ロジック」を書くパターンが3つある。
どれも「Bはコンテキストの型だけ定義し、中身の判定はAが書く」という同じ思想。

| パターン | 対象 | 型 | 呼び出し元 |
|---|---|---|---|
| `usage_condition` | グッズ/サポート/スタジアム | `UsageContext -> bool` | `priorities/board.py`・`draw.py` |
| `EnergyPriorityRule.condition` | エネルギーカードの選択順 | `EnergyCardContext -> bool` | `priorities/energy.py`・`energy_tool_turn.py` |
| `combo_with`（`PokemonProfile`） | ポケモンの相棒判定 | `list[int]`（宣言的、関数ではない） | `pokemon_value.py` |

新しい種類の「条件で判断を変えたい」ニーズが出てきたとき、まずこの3パターンで表現できないか
検討する。表現できないなら、`profile_types.py`に似た形（Context型 + 条件関数）で新しいパターンを
足すのが、これまでの設計と一貫性がある。

---

## 5. エネルギー周回コンボ（`energy_recycle_*`）を使うかどうかの判断

「山札に戻る特性を持つポケモンに、希少なエネルギー（ACE SPECなど）を一時的に付けて
使い回す」ようなコンボがデッキにある場合だけ、`deck_plan.py`に
`ENERGY_RECYCLE_TARGET_CARD_ID`（受け皿ポケモン）・`ENERGY_RECYCLE_CARD_ID`
（周回させたいエネルギー）・`ENERGY_RECYCLE_BACKUP_ITEM_ID`（主力への代替供給手段、
無ければ`None`のままでよい）を定義する。無いデッキでは何も設定しなければ良いだけで、
`energy_eval.py`側は自動的にこの機能を無視する。

---

## 6. `deck.csv`とデッキ選択

- `sample_submission/deck.csv`は1行1カードID、合計60行。担当Aが用意した60枚と
  一致していることを必ず確認する（過去に無関係な旧デッキのまま放置されていたことがあった）。
- Basicポケモンが最低1枚、同名カード上限4枚（ACE SPECのみ1枚）を満たすこと。
- `deck.csv`の中身と`decks/active.py`が指すデッキ（`pokemon_profiles.py`等）が
  **同じ60枚を指しているか**は誰も自動チェックしていない。手動で整合を確認すること。

---

## 7. 動作確認の手順

新しいデッキに切り替えたら、必ず以下を確認する。

```powershell
# 型・import エラーが無いか
python -m py_compile decks/<deck_name>/*.py

# 既存の汎用テスト（デッキ非依存のロジック）が壊れていないか
python -m pytest tests/unit tests/integration -q

# 実際に対戦が完走するか（エラーが出ないか）
python tests/local_sim/test_local_game.py

# 自己対戦でのおおまかな勝率確認（対ランダム）
python tests/local_sim/test_local_game_advanced.py --games 20 --opponent random
```

`get_deck_plan()`や各`get_xxx_profile()`が実際に呼ばれることを、Pythonの対話環境から
軽く叩いて確認するのも有効（`profile_registry.get_deck_plan().opening_priority`が
期待通りのcard_id列になっているか、など）。

---

## 8. 過去にハマった落とし穴（新デッキでも起こりうる）

- **`Attack.damage`が0の可変ダメージ技**: 「手札枚数×2」のような技は`cg.api`の
  静的データでは`damage=0`になる。`board_evaluation/attack_features.py`が
  `Attack.text`から正規表現で推定するが、**「N damage counters」という表記は
  N×10ダメージ**（公式ルールの単位）である点に注意。新しい言い回しの可変ダメージ技が
  出てきたら、必ず`data/EN_Card_Data.csv`の実際のテキストで検証してからパターンを足す。
- **`Option.cardId`はほとんどのOptionTypeで`None`**: `cg.api.py`のコメント上、
  `cardId`が直接入るのは`OptionType.SKILL`のみ。他は`area`/`index`（+`toolIndex`/
  `energyIndex`）経由でしか特定できない。`ptcg_ai/rule_based/card_move/common.py`の
  `resolve_card_id`/`resolve_pokemon`を必ず経由し、`option.cardId`を直接読まない。
- **2ステップ選択（例: `ATTACH_FROM`/`ATTACH_TO`）の順序は保証されない**: 実際のリプレイで
  確認したところ、名前から連想する順序と逆に発生するケースがあった。「前のステップの選択を
  覚えておく」設計は避け、**都度、実際に提示された選択肢（`obs.select.option`）の範囲内だけで
  再評価する**実装にする（`energy_eval.energy_target_value`のように、対象を絞り込まず
  「候補ごとのスコア関数」として公開しておくと、どちらの文脈からも使い回せる）。
- **貪欲法（1手ずつ独立評価）の限界**: このAIは「今この瞬間の最良の1手」を毎回選ぶだけで、
  数手先を計画する仕組みは無い。多くの「コンボ」は実際には各ステップが独立に良い手として
  評価されれば自然に繋がる（Issue #5がそうだった）。本当に先読みが無いと成立しない
  コンボ（例: 今は損に見えるが後で得をする）は、このアーキテクチャでは表現できないので、
  無理に再現しようとせず見送るという判断も選択肢に入れる。
- **`weights.py`は`integration`に合流するたびに初期値が失われることがあった**: 複数ブランチで
  並行して重み調整をしていると、マージ時に片方の変更が消えることがある。合流後は必ず
  `CATEGORY_BASE_WEIGHT`が意図した値になっているか確認する。
