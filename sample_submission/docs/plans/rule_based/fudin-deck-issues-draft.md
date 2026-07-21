# Issue化用ドラフト: フーディンデッキ実装ギャップ

[fudin-deck-implementation-gaps.md](fudin-deck-implementation-gaps.md)の内容をIssue化するための下書き。
GitHubで新規Issue作成時に、タイトル・本文をそのままコピペして使ってください。

---

## Issue 1（最優先・バグ）

### タイトル（案）

```
[bug] ハンドパワーの可変ダメージ計算が実際の1/10になっている
```

### 本文（案）

```markdown
## 概要

`board_evaluation/attack_features.py`の可変ダメージ推定（`_estimate_variable_damage`）が、
フーディンの「ハンドパワー」（Attack.text: "Place 2 damage counters on your opponent's
Active Pokémon for each card in your hand."）を **2 × 手札枚数** として計算しているが、
公式ルール上 damage counter 1個 = 10ダメージ のため、正しくは **20 × 手札枚数** である。

`data/EN_Card_Data.csv`（cardId=743）でも "Place 2 damage counters ... for each card in
your hand" とだけ書かれており、直接ダメージ点数を書く技（例: Cruel Arrow "does 100
damage"）とは表記が異なる点を確認済み。

## 影響範囲

- `priorities/attack.py`のattackカテゴリのscore
- `attack_features.can_ko`によるきぜつ判定
- `board_features.is_likely_ko_next_turn`（相手の次ターン攻撃予測）

いずれも主砲であるハンドパワーの評価が実際の1/10になっており、
本来リーサルが取れる場面を見逃す/相手の脅威を過小評価する可能性が高い。

## 対応方針

`_PER_HAND_CARD_PATTERN`がマッチした際、抽出した数値をそのまま使うのではなく
×10（damage counter → ダメージ点数の変換）する。今後同種の「N damage counters」
表記の技が出てきた場合にも対応できるよう、単位の区別をコード上明示する。

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション1
```

---

## Issue 2

### タイトル（案）

```
[feature] deck_plan.py の KO_REPLACEMENT_PRIORITY / ENERGY_REQUIRED_COUNT / ENERGY_CARD_PRIORITY_RULES を判断ロジックに接続する
```

### 本文（案）

```markdown
## 概要

`decks/new_deck/deck_plan.py`には以下のデータが定義されているが、
`profile_registry.get_deck_plan()`（`_build_deck_plan()`）が変換していないため、
どの判断ロジックからも参照されていない。

| 変数名 | 内容 |
|---|---|
| `KO_REPLACEMENT_PRIORITY` | きぜつ後の後継優先順位（`evolution_priority`とは別軸） |
| `ENERGY_REQUIRED_COUNT` | ポケモンごとの必要エネルギー総数上限（これ以上は付けない） |
| `ENERGY_CARD_PRIORITY_RULES` | 条件付きの「どのエネルギーカードを使うか」優先順位 |

## 対応方針

- `DeckPlan`（`ptcg_ai/shared/profile_types.py`）に対応するフィールドを追加するか、
  専用のアクセサ（例: `get_ko_replacement_priority()`）を`profile_registry.py`に追加する
- `switch_eval`/`pokemon_value.py`（後継選び）、`energy_eval.py`（対象・上限・カード選定）
  それぞれの該当箇所で参照するよう配線する

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション2
```

---

## Issue 3

### タイトル（案）

```
[design] グッズ/サポート/スタジアムの USAGE_NOTES（自由記述の使用条件）を構造化データに落とす方針を決める
```

### 本文（案）

```markdown
## 概要

`deck_plan.py`の`ITEM_USAGE_NOTES`/`SUPPORTER_USAGE_NOTES`/`STADIUM_USAGE_NOTES`には、
カードごとの詳しい使用条件が担当Aによって自由記述で書かれている
（例: 改造ハンマーは「相手のバトル場に特殊エネルギーが付いている場合」、
ボスの指令は「相手ベンチのHPがバトルポケモンより低い場合」など）。

現状これらは自由記述の文字列（`UsageNote.note`）のままで、
`ItemProfile`/`SupporterProfile`/`StadiumProfile`の`category`+`priority`だけでは
表現しきれず、判断ロジックからは一切参照されていない。

## 論点（要相談）

- 自由記述のまま条件判定に使うのは無理があるため、条件ごとに構造化フィールドを
  追加する必要がある（例: `requires_opponent_special_energy: bool`,
  `requires_opponent_bench_lower_hp: bool` 等、カードごとに個別の条件フラグを増やす形）
- カードが増えるたびに条件の種類が増えていく可能性があり、担当A・担当Bで
  どこまで構造化するか、方針をすり合わせたい

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション2
```

---

## Issue 4

### タイトル（案）

```
[feature] ノココッチ×リッチエネルギーの周回コンボに対応する
```

### 本文（案）

```markdown
## 概要

戦略記事（https://note.com/smasakichi/n/nd5426f7cf1dc ）より:
「リッチエネルギーを貼ったノココッチは、必ず攻撃前に特性を使ってデッキに戻しましょう」

リッチエネルギー(card_id=13, ACE SPEC・1枚)をノココッチ(66)に付け、特性「にげあしドロー」
（3ドロー後、自身と付いているカード全てを山札に戻す）でリッチエネルギーごと山札に戻し、
トウコ等の「エネルギーをサーチする」効果で回収し直す、というACE SPECの使い回しコンボ。

## 現状

- `deck_plan.ENERGY_TARGET_PRIORITY`はノココッチを明示的に除外している
- `pokemon_value.py`は`bench_value`の高さでノココッチへのエネルギー付与を減点する設計
- `search_priority`にリッチエネルギー(13)が含まれておらず、山札に戻った後に
  優先して探しに行く理由もない

「ノココッチにエネルギーを付ける」という発想自体がコード上に存在しない。

## 対応方針（要検討）

- リッチエネルギーをノココッチに付けるべきタイミング（毎ターンではなく、
  他に優先すべきエネルギー付与先が無い場合など）の条件をどう表現するか
- 山札に戻った直後、トウコ等でリッチエネルギーを優先回収する判断をどう入れるか

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション3
```

---

## Issue 5

### タイトル（案）

```
[feature] ワンダーパッチの多段コンボ（先読みが必要）への対応を検討する
```

### 本文（案）

```markdown
## 概要

戦略記事より: 「ベンチにノココッチを置いてから、ベンチのフーディンにエネルギーを付け、
ノココッチをにげさせて充電済みのアタッカーにアクセスする」

複数アクションにまたがる意図的な手順（配置→エネルギー付与→交代）が必要なコンボ。

## 現状

現在の意思決定は「今この瞬間、どのカテゴリのどの1手が一番スコアが高いか」を
毎回独立に選ぶ貪欲法（greedy）で、数手先を見越した計画を行う仕組みが構造的に無い。
優先度リストの調整だけでは再現できない。

## 対応方針（要検討）

- 優先度ベースの現行アーキテクチャで無理に表現しようとせず、専用の状態機械/
  シーケンス検出を追加する価値があるか、費用対効果を含めて要議論
- 優先度は他のIssueより低め（実装コストが大きい割に発生頻度が限定的なため）

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション4
```

---

## Issue 6（優先度低）

### タイトル（案）

```
[feature] retreat の判断で撤退コスト（残エネルギー）を確認していない
```

### 本文（案）

```markdown
## 概要

`priorities/retreat.py`は、にげるコスト（`CardData.retreatCost`）に対して、
実際に手放せる分のエネルギーが足りているかを確認していない。
エンジンが合法手のみ選択肢として出す前提に乗っているだけで、致命的ではないが、
撤退可否の見積もりには使えていない。

## 対応方針

`board_evaluation`側にエネルギー充足チェックのヘルパーを追加し、
撤退判断のスコアリングに組み込むかを検討する。優先度は低め。

## 参考

- [fudin-deck-implementation-gaps.md](sample_submission/docs/plans/rule_based/fudin-deck-implementation-gaps.md) セクション7
```
