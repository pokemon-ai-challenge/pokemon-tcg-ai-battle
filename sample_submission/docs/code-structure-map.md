# コード構成メモ

`sample_submission` 配下の「どこからどこを呼んでいるか」を、実装時に追いやすくするためのメモです。  
README は入口だけに保ち、このファイルでは構成と責務をまとめます。

---

## まず見る場所

- `main.py`
  - Kaggle 提出時の入口
- `src/agent.py`
  - 初回デッキ返却と通常ターンの分岐
- `src/decision/router.py`
  - 通常ターンでどの処理に渡すかを決める
- `src/decision/main_turn.py`
  - メインフェーズの司令塔

---

## 呼び出しの流れ

提出時やローカル対戦では、最終的に `main.py` の `agent(obs_dict)` が呼ばれます。

```text
main.py
  └─ src/agent.py
      ├─ obs.select is None
      │   └─ read_deck_csv() で deck.csv を返す
      └─ 通常ターン
          └─ src/decision/router.py
              ├─ MAIN
              │   └─ src/decision/main_turn.py
              ├─ SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON
              │   └─ src/decision/setup_turn.py
              ├─ ATTACK
              │   └─ src/decision/attack_turn.py
              └─ それ以外
                  └─ src/decision/fallback.py
```

ポイントは次の通りです。

- `main.py` は提出入口として薄く保つ
- `agent.py` は「デッキ返却か、通常ターンか」を分ける
- `router.py` は「どの文脈の選択か」を見て分岐する
- 実際の判断ロジックは `src/decision/` 以下で持つ

---

## フォルダごとの役割

### `main.py`

- 提出用の入口
- `agent(obs_dict)` を公開する
- 初回デッキ用の `read_deck_csv()` を持つ

### `cg/`

- コンペ提供のゲームエンジン
- `api.py` に観測データや Enum の定義がある
- 原則として編集しない

### `src/agent.py`

- `obs_dict` を `Observation` に変換する
- `obs.select is None` ならデッキを返す
- そうでなければ `router.py` に渡す

### `src/decision/`

- ターン中の選択ロジック置き場
- 現在は次のファイルに分かれている

```text
src/decision/
├─ router.py
├─ main_turn.py
├─ setup_turn.py
├─ attack_turn.py
└─ fallback.py
```

### `src/knowledge/`

- カード情報や共有キャッシュ置き場
- 現在は `card_cache.py` で `all_card_data()` と `all_attack()` をまとめている

### `src/tests/`

- 将来の固定シナリオや確認コードの置き場
- まだ薄くても問題ない

---

## `router.py` の責務

`router.py` は薄く保つ前提です。

- `obs.select.context` を見る
- `MAIN` か `SETUP` か `ATTACK` かを判定する
- 対応する処理へ渡す

ここには盤面評価やカード個別ロジックを大量に書かず、呼び分けだけを置くのが基本です。

---

## `main_turn.py` の責務

`main_turn.py` はメインフェーズの司令塔です。  
いまは「候補を集めて、仮の重みづけで最善手を選ぶ」形になっています。

```text
main_turn.py
  ├─ bucket_main_options(obs)
  ├─ propose_draw_or_search_action(...)
  ├─ propose_pokemon_or_evolve_action(...)
  ├─ propose_board_item_action(...)
  ├─ propose_ability_action(...)
  ├─ propose_energy_action(...)
  ├─ propose_retreat_action(...)
  ├─ propose_attack_action(...)
  ├─ propose_end_action(...)
  └─ choose_best_proposal(...)
```

---

## `main_turn_parts/` の責務

`main_turn.py` を複数人で触りやすくするために、下請け処理を分けています。

```text
src/decision/main_turn_parts/
├─ buckets.py
├─ proposals.py
├─ weights.py
└─ priorities/
   ├─ draw.py
   ├─ board.py
   ├─ ability.py
   ├─ energy.py
   ├─ retreat.py
   ├─ attack.py
   └─ end_turn.py
```

### `buckets.py`

- `obs.select.option` を大まかな用途ごとに分類する
- 例:
  - `pokemon_play`
  - `supporter_play`
  - `item_play`
  - `ability`
  - `attach`
  - `attack`

### `weights.py`

- 仮の重みをまとめる
- 「どの行動をどれくらい優先するか」の初期値を置く

### `proposals.py`

- `MainActionProposal` を定義する
- `action`
- `score`
- `label`
- それらを比べて一番よい候補を返す

### `priorities/`

- 行動カテゴリごとの提案処理を分割する
- 各ファイルは「この状況ならこの行動をこの重みで出したい」を返す

---

## チームで触るときの分け方

分担しやすい単位は次の通りです。

- `router.py`
  - 文脈ごとの分岐担当
- `main_turn_parts/buckets.py`
  - 選択肢の整理担当
- `main_turn_parts/weights.py`
  - 仮重みの調整担当
- `main_turn_parts/priorities/draw.py`
  - 手札補充やサーチ担当
- `main_turn_parts/priorities/board.py`
  - 展開、進化、盤面整備担当
- `main_turn_parts/priorities/ability.py`
  - 特性担当
- `main_turn_parts/priorities/energy.py`
  - エネルギー担当
- `main_turn_parts/priorities/retreat.py`
  - にげる担当
- `main_turn_parts/priorities/attack.py`
  - 攻撃担当

`main_turn.py` 本体は司令塔にして、できるだけ複数人で同時に大きく触らない方が競合しにくいです。

---

## ローカル確認の流れ

ローカルでは `main.py` が自分でループするのではなく、テストスクリプト側が何度も `agent(obs_dict)` を呼びます。

```text
local_test.py / local_test_advanced.py
  └─ battle_start(...)
      └─ agent(obs_dict)
          └─ 行動 index を返す
              └─ battle_select(action)
                  └─ 次の obs_dict を受け取る
```

つまり、盤面が変わるたびに `agent()` と `main_turn.py` は再度呼ばれ、その時点の `obs` を見て優先順位を毎回再計算します。

---

## 編集の目安

- 提出入口を確認したい
  - `main.py`
- デッキ返却と通常ターンの境界を見たい
  - `src/agent.py`
- 通常ターンの文脈分岐を見たい
  - `src/decision/router.py`
- メインフェーズの全体方針を見たい
  - `src/decision/main_turn.py`
- メインフェーズの個別担当を見たい
  - `src/decision/main_turn_parts/priorities/`
- カード種類や攻撃データを見たい
  - `src/knowledge/card_cache.py`
  - `cg/api.py`

---

## 補足

- `main.py` は提出入口なので、別ファイル化しても「最終的にここから呼ぶ」形は保つ
- `cg/` は原則変更しない
- README は入口ガイドに保ち、詳細な構成説明はこのファイルに寄せる
