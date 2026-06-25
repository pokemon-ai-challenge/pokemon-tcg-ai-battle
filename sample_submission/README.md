# 提出エージェントの実装

このフォルダには、Kaggle提出用のポケモンカードAIエージェントを置きます。

この README は実装の入口を共有するためのガイドです。完全なレビューはまだ終わっていないため、細部の仕様確認や実装時の最終判断は `main.py` と `cg/` 配下のプログラムを直接見て進めてください。

自分たちが主に編集するのは次の2つです。

- `main.py`：対戦中の行動を決めるAI本体
- `deck.csv`：使用する60枚デッキのカードID

`cg/` はコンペ提供のゲームエンジンです。原則として変更しません。

---

## 提出手順

提出時に実行されるプログラムは `main.py` です。  
対戦エンジンは `main.py` 内の `agent(obs_dict)` を呼び出します。  
`deck.csv` は初回のデッキ返却で使い、`cg/` は実行に必要なゲームエンジンです。

提出用アーカイブ `submission.tar.gz` には、次の3つを入れます。

```text
submission.tar.gz
├── main.py
├── deck.csv
└── cg/
```

PowerShell で作る場合は、`sample_submission` フォルダで次を実行します。

```powershell
tar -czvf submission.tar.gz main.py deck.csv cg
```

作成後は、中身を確認します。

```powershell
tar -tzf submission.tar.gz
```

`main.py` がアーカイブ直下にあり、`sample_submission/main.py` のように1段深く入っていないことを確認してください。

その後、Kaggle の Simulation コンペページを開きます。  
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle

`Submit Agent` から `submission.tar.gz` を選択し、提出します。

---

## `main.py` の役割

提出時には、次の関数を実装します。

```python
def agent(obs_dict: dict) -> list[int]:
```

対戦エンジンはこの関数を繰り返し呼び出します。

### 初回：デッキを返す

初回は `obs.select` が `None` です。  
このときは、使用するデッキとしてカードIDを60個返します。

```python
if obs.select is None:
    return read_deck_csv()
```

### 対戦中：行動を返す

2回目以降は、`obs.select.option` に現在選べる合法手が入っています。

返す値はカードIDではなく、**`obs.select.option` 内の選択肢インデックス**です。

```python
# 0番目の選択肢を選ぶ
return [0]

# 複数の選択肢を選ぶ
return [1, 3]
```

返すリストは次の条件を満たす必要があります。

```python
obs.select.minCount <= len(result) <= obs.select.maxCount
```

- 各要素は `0 <= index < len(obs.select.option)`
- 同じインデックスを重複して選ばない
- 任意選択で何も選ばない場合は `[]` を返せる

---

## `cg` の使い方

`main.py` では主に `cg.api` を使います。

```python
from cg.api import (
    Observation,
    OptionType,
    SelectContext,
    to_observation_class,
)
```

受け取った辞書形式の観測情報を、扱いやすい `Observation` に変換します。

```python
obs: Observation = to_observation_class(obs_dict)
```

主に見る値は次の通りです。

```python
obs.current          # 現在の盤面
obs.select           # 現在必要な選択
obs.select.option    # 選択できる合法手の一覧
obs.logs             # 直前の選択以降に起きたイベント
```

### 選択肢の種類

```python
option.type == OptionType.ATTACK  # ワザを使う
option.type == OptionType.ATTACH  # エネルギーなどを付ける
option.type == OptionType.EVOLVE  # 進化する
option.type == OptionType.PLAY    # 手札からカードを使う
option.type == OptionType.ABILITY # 特性を使う
option.type == OptionType.RETREAT # にげる
option.type == OptionType.END     # ターン終了
```

通常のメイン行動を選ぶ場面かどうかは、次のように判定します。

```python
obs.select.context == SelectContext.MAIN
```

### カード・ワザ情報を取得する

カードIDをカード名や効果に変換したい場合は、以下を使用します。

```python
from cg.api import all_card_data, all_attack

cards = all_card_data()
attacks = all_attack()
```

### 先読み探索を使う

必要になった場合は、仮想局面を作って候補手を比較できます。

```python
from cg.api import search_begin, search_step, search_release, search_end
```

`search_begin(...)` には、`agent()` に渡された `Observation` をそのまま渡します。  
また、自分と相手の山札・サイド・手札などの非公開情報については、必要枚数ぶんの予測カードID列を与える必要があります。  
相手のバトル場が伏せポケモンの場合は、`opponent_active` も予測して渡します。

| 関数 | 用途 |
|---|---|
| `search_begin(...)` | 現在局面を基に探索を開始する |
| `search_step(...)` | 仮想局面で選択を1回進める |
| `search_release(search_id)` | 不要な探索局面を削除する |
| `search_end()` | 探索全体を終了する |

まずは `obs.select.option` を使ったルールベースAIを作り、必要になった段階で探索を追加する。

---

## ローカルで対戦を確認する場合

ローカルで対戦を開始・進行・可視化したい場合のみ、`cg.game` を使います。

まずは `sample_submission` フォルダに移動してから実行します。`main.py` は相対パスで `deck.csv` を読むため、別フォルダで実行すると失敗しやすいです。

```powershell
cd C:\dev\pokemon-tcg-ai-battle\sample_submission
```

### 最小の動作確認

`local_test.py` は、現在の `main.py` を使って 1 試合だけ回す最小の確認用スクリプトです。

```powershell
python .\local_test.py
```

`errorType` と `result` が表示されます。

### 複数試合や比較

`local_test_advanced.py` は、複数試合の実行や `random` 相手との比較に使います。

```powershell
python .\local_test_advanced.py
python .\local_test_advanced.py --opponent self --games 10
python .\local_test_advanced.py --opponent random --games 100
python .\local_test_advanced.py --opponent random --games 10 --verbose
```

- `--opponent self` は `main.py` 同士で対戦します
- `--opponent random` はランダム行動の相手と対戦します
- `--games N` は N 試合まとめて実行します
- `--verbose` は各ターンの選択内容も表示します

```python
from cg.game import battle_start, battle_select, battle_finish, visualize_data
```

- `battle_start(deck0, deck1)`：2つの60枚デッキで対戦を開始する
- `battle_select(select_list)`：選択肢インデックスを渡して1手進める
- `visualize_data()`：表示用の対戦データを取得する
- `battle_finish()`：対戦を終了してメモリを解放する

`cg.sim`、`cg.utils`、`libcg.so` / `cg.dll` は内部実装のため、通常は直接使用・変更しません。
