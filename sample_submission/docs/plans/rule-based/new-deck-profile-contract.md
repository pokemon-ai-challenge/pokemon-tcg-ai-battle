# カードプロファイル契約（担当A向け共有資料）

`sample_submission/ptcg_ai/shared/profile_types.py` で確定した型定義のリファレンスです。
担当Aが `sample_submission/decks/new_deck/*_profiles.py` にデータを埋める際は、
このドキュメントに沿ってください。

対象読者: 担当A（Bが型を作り、Aがデータを埋める）。
コードの場所や全体設計は [new-deck-src-structure.md](new-deck-src-structure.md)、
どのファイルを誰が触るかは [new-deck-file-assignment.md](new-deck-file-assignment.md) を参照。

---

## なぜこの契約があるか

担当Bの判断ロジック（`ptcg_ai/action_selection/`・`ptcg_ai/rule_based/` 配下）は、カードIDやカード名を直接書きません。
「このカードは主力アタッカーか」「このグッズはサーチ系か」といった判断材料を、
**すべて数値・真偽値・決まった語彙のプロファイルとして** Aから受け取ります。

そのため、Aが `*_profiles.py` に書くデータの**型・フィールド名・語彙**が固定されていないと
Bのコードが動きません。このドキュメントがその固定された「契約」です。

フィールドを増やしたい／表現しきれないケースが出てきた場合は、**Aの判断で書き換えず**、
Bと相談して `profile_types.py` を一緒に更新してください（両方のコードが同じ型を見ているため）。

---

## EffectCategory（共通語彙）

グッズ・サポート・どうぐ・スタジアム・特性の効果を分類する共通の語彙です。
自由記述ではなく、必ずこの7つのいずれかを使ってください。

| 値 | 意味 | 例 |
|---|---|---|
| `"search"` | デッキ/手札からカードを探す | ネストボール、はかせの研究の一部効果 |
| `"draw"` | ドローによる手札補充 | 博士の研究、ボスの指令ではない純粋ドロー |
| `"heal"` | 回復・ダメカン除去 | ポケモンいれかえのポーション枠、キズぐすり |
| `"disruption"` | 相手の手札/デッキ/場を妨害する | ボスの指令、相手の手札を見て捨てさせる系 |
| `"setup"` | 展開・ベンチ強化・エネルギー加速 | ハイパーボールでの展開、エネルギー加速特性 |
| `"lock"` | 相手の選択肢を制限する | 特性ロック、攻撃制限、にげられなくする効果 |
| `"other"` | 上記に当てはまらないもの | 上記以外 |

迷ったら `"other"` にして、あとでBと相談してカテゴリを見直してください。

---

## PokemonProfile（`decks/new_deck/pokemon_profiles.py`）

ポケモン1種ごとの役割データ。

| フィールド | 型 | 意味 |
|---|---|---|
| `role` | `str` | 役割。目安: `"main_attacker"` / `"sub_attacker"` / `"wall"` / `"support"` |
| `role_score` | `float` | 主力度。0.0〜1.0の目安（1.0が最も主力） |
| `bench_value` | `float` | ベンチに置いておく価値（進化前で温存したい、コンボパーツなど） |
| `combo_with` | `list[int]` | コンボ関係にある相棒カードのID一覧（省略可、デフォルト空リスト） |
| `has_ability` | `bool` | 特性を持つか（省略可、デフォルト `False`） |
| `ability_category` | `EffectCategory \| None` | 特性の効果分類（特性が無ければ `None`） |
| `ability_priority` | `float` | 特性を使うべき優先度の目安。0.0なら基本使わない（省略可、デフォルト `0.0`） |

**特性（Ability）と技（Attack）は別物です。** 特性の情報はこの `PokemonProfile` に書きます
（`AttackProfile` には書きません）。

### 記入例

```python
from ptcg_ai.shared.profile_types import PokemonProfile

PROFILES: dict[int, PokemonProfile] = {
    12345: PokemonProfile(
        role="main_attacker",
        role_score=0.9,
        bench_value=0.3,
        combo_with=[23456],  # 相棒のどうぐ/サポートのカードID
        has_ability=True,
        ability_category="setup",
        ability_priority=0.7,
    ),
    23456: PokemonProfile(
        role="support",
        role_score=0.2,
        bench_value=0.6,
    ),
}
```

---

## AttackProfile（`decks/new_deck/attack_profiles.py`）

技1つごとの追加効果分類（キーは `attack_id`。`Attack.attackId` に対応、`card_id` ではない点に注意）。

| フィールド | 型 | 意味 |
|---|---|---|
| `bench_snipe` | `bool` | ベンチ狙撃効果があるか |
| `inflicts_special_condition` | `bool` | 状態異常を付与するか |
| `draws_cards` | `bool` | ドロー効果があるか |
| `disables_next_attack` | `bool` | 次ターン攻撃不可などのロック効果があるか |

### 記入例

```python
from ptcg_ai.shared.profile_types import AttackProfile

PROFILES: dict[int, AttackProfile] = {
    501: AttackProfile(
        bench_snipe=False,
        inflicts_special_condition=True,
        draws_cards=False,
        disables_next_attack=False,
    ),
}
```

---

## ItemProfile / SupporterProfile / ToolProfile / StadiumProfile

グッズ・サポート・どうぐ・スタジアムは全て同じ形です（ファイルはカード種別ごとに分かれています）。

| フィールド | 型 | 意味 |
|---|---|---|
| `category` | `EffectCategory` | 上記の共通語彙から1つ選ぶ |
| `priority` | `float` | 同カテゴリ内での使用優先度（tie-break用）。省略可、デフォルト `0.0` |

`priority` は「サーチ系グッズが手札に2枚あるとき、どちらを優先するか」のような
同カテゴリ内での比較にのみ使います。デッキ全体としての優先順位（何を最初に探すか等）は
`deck_plan.py`（`search_priority` など）の役目なので、そちらと役割を混同しないでください。

### 記入例（グッズ）

```python
from ptcg_ai.shared.profile_types import ItemProfile

PROFILES: dict[int, ItemProfile] = {
    701: ItemProfile(category="search", priority=0.8),
    702: ItemProfile(category="disruption", priority=0.5),
}
```

---

## EnergyProfile（`decks/new_deck/energy_profiles.py`）

| フィールド | 型 | 意味 |
|---|---|---|
| `category` | `Literal["basic", "special"]` | 基本エネルギーか特殊エネルギーか |

---

## Bのコードがこれをどう使うか（参考）

Aはこの先を気にする必要はありませんが、参考までに：`ptcg_ai/action_selection/`・`ptcg_ai/rule_based/` 配下は
`ptcg_ai.shared.profile_registry` の `get_pokemon_profile(card_id)` /
`get_attack_profile(attack_id)` / `get_item_profile(card_id)` などを呼び、
戻り値としてこれらのプロファイルを受け取ります。Aが `PROFILES` 辞書に値を追加するだけで、
B側のロジックは自動的にその値を参照できるようになります。

---

## 更新履歴

- 初版: `PokemonProfile` に特性関連フィールド（`has_ability`/`ability_category`/`ability_priority`）、
  `EffectCategory` 共通語彙、Item/Supporter/Tool/Stadium への `priority` フィールドを追加した版。
