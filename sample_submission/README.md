# 提出エージェントの実装

このフォルダには、Kaggle提出用のポケモンカードAIエージェントを置きます。

この README は実装の入口を共有するためのガイドです。完全なレビューはまだ終わっていないため、細部の仕様確認や実装時の最終判断は `main.py` と `cg/` 配下のプログラムを直接見て進めてください。

自分たちが主に編集するのは次の2つです。

- `main.py`：対戦中の行動を決めるAI本体
- `deck.csv`：使用する60枚デッキのカードID

`cg/` はコンペ提供のゲームエンジンです。原則として変更しません。

開発の中長期方針は [docs/ai-development-roadmap.md](docs/ai-development-roadmap.md) にまとめています。

---

## フォルダ構成

`sample_submission` 配下は、提出に必須な最小構成と、ローカル開発用の AI モジュール群に分けて整理しています。

```text
sample_submission/
├── main.py                  # Kaggle 提出時のエントリーポイント
├── deck.csv                 # 使用する60枚デッキ
├── cg/                      # コンペ提供エンジン（提出に含める）
├── ptcg_ai/
│   ├── core/                # AI 全体の入口、config 読み込み、モジュール切り替え
│   ├── action_selection/    # 最終行動選択、fallback、legal action 確認、選択ルーター
│   ├── rule_based/          # 手書きルールベース AI
│   ├── search/              # リーサル探索、MCTS、ISMCTS、rollout などの探索系
│   ├── opponent_modeling/   # 相手に関する情報の記録・推定
│   ├── hidden_information/  # 自分や相手の非公開情報の推定
│   ├── board_evaluation/    # 盤面評価、行動評価、サイド価値、テンポ評価
│   ├── state_view/          # Observation を AI 向けの形に変換
│   ├── learning/            # 強化学習や機械学習モデルの推論・学習
│   └── shared/              # 複数機能で使う共通処理
├── decks/                   # デッキ固有データ（方針・カード別プロファイル）
├── configs/                 # AI 構成を切り替える設定
├── tests/                   # 単体テスト、import 確認、ローカルシミュレーション
├── docs/                    # 開発メモ、設計メモ
└── results/                 # 実験結果や比較結果の置き場
```

### ディレクトリの責務

- `ptcg_ai/core/`
  AI 全体の入口、config 読み込み、モジュール切り替えなどを置きます。
- `ptcg_ai/action_selection/`
  最終的な行動選択、fallback、legal action 確認、行動選択ルーターなどを置きます。
- `ptcg_ai/rule_based/`
  手書きルールベース AI を置きます。
- `ptcg_ai/search/`
  リーサル探索、MCTS、ISMCTS、rollout などの探索系を置きます。
- `ptcg_ai/opponent_modeling/`
  相手に関する情報の記録・推定を置きます。
- `ptcg_ai/hidden_information/`
  自分の山札・サイドや相手側も含む非公開情報の推定を置きます。
- `ptcg_ai/board_evaluation/`
  盤面評価、行動評価、サイド取得価値、テンポ評価などを置きます。
- `ptcg_ai/state_view/`
  コンペの `Observation` を AI 側で扱いやすい形に変換する処理を置きます。
- `ptcg_ai/learning/`
  強化学習や機械学習モデルの推論・学習関連を置きます。
- `ptcg_ai/shared/`
  複数の機能で使う共通処理を置きます。
- `decks/`
  デッキ固有のデータ（デッキ方針、ポケモン・ワザ・グッズなどカード別プロファイル）を置きます。
  `ptcg_ai/` 側はカードIDやカード名を直接書かず、`decks/active.py` 経由でここを参照します。
- `configs/`
  AI 構成を切り替えるための config を置きます。
- `tests/`
  単体テストや import 確認用のテストを置きます。
- `results/`
  実験結果や比較結果を記録します。まだない場合は必要になったタイミングで作成します。

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

PowerShell で作る場合は、リポジトリ直下から次を実行します。

```powershell
cd sample_submission
tar -czvf submission.tar.gz main.py deck.csv cg
```

`sample_submission` フォルダに移動してから実行することで、`main.py`、`deck.csv`、`cg/` を正しい位置からまとめられます。

作成後は、同じフォルダで中身を確認します。

```powershell
tar -tzf submission.tar.gz
```

`main.py` がアーカイブ直下にあり、`sample_submission/main.py` のように1段深く入っていないことを確認してください。

その後、Kaggle の Simulation コンペページを開きます。  
https://www.kaggle.com/competitions/pokemon-tcg-ai-battle

`Submit Agent` から `submission.tar.gz` を選択し、提出します。

### CLI から提出する場合

ブラウザを開かずに `kaggle` CLI からも提出できる(動作確認済み)。事前に
`kaggle.json`(APIトークン)が `~/.kaggle/` に配置され、認証済みであること
(`kaggle competitions submissions -c pokemon-tcg-ai-battle` が一覧を返せば認証済み)。

複数ファイル(`main.py` 単体ではなく `deck.csv`/`cg/`/`ptcg_ai/`/`configs/` 一式)を
提出する場合は tar.gz にまとめてから提出する
([Kaggle CLI公式ドキュメント](https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md)
のシミュレーションコンペ向け手順と同じ形):

```powershell
cd sample_submission
tar -czvf submission.tar.gz main.py deck.csv cg configs ptcg_ai decks
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "提出内容の説明"
```

> **重要（`decks/` を必ず含める）**: 現在の agent は import 連鎖で `decks/`（`ptcg_ai/shared/profile_registry.py` → `from decks import active`）に依存する。`decks` を tar に含め忘れると、Kaggle 側で agent が **import すらできず即 `SubmissionStatus.ERROR`** になる（ローカルはリポジトリ全体があるため気づけない。実例: 提出 54876982）。
> 提出前の確認として、tarball を**空のディレクトリに展開して**（＝Kaggle と同じ「tarball の中身しか無い」状態）1ゲーム走らせると、この種の欠落を事前に検出できる。

提出後の確認:

```powershell
# 提出一覧・スコア(publicScore)の確認。提出直後はスコアがまだ収束していないことがある
kaggle competitions submissions -c pokemon-tcg-ai-battle

# 実際の対戦ログ(勝敗)を取得して分析する場合は kaggle_replays/fetch_my_episodes.py を使う
# (自分の直近N件の提出に紐づくエピソードをダウンロードし、replays/ + index/episodes_master.jsonl に追記する)
cd ../kaggle_replays
python fetch_my_episodes.py --submissions 1 --max-episodes 500
```

`fetch_my_episodes.py` が出力するリプレイJSON(`replays/episode-<id>-replay.json`)の
`info.TeamNames` で対戦相手のチーム名、`rewards`(player_index順、勝ち側が正の値)で
勝敗が分かる。両陣営とも自分のチーム名の場合はミラー戦(相手プールに自分しかいない等の
理由で発生することがある)なので、実際の対戦相手との勝率を見る場合は除外すること。

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

まずは、同梱しているローカル確認用スクリプトを使うのが簡単です。

### 1ゲームだけ動かす

`local_test.py` は、自分の `main.py` を使って同じ `deck.csv` 同士で 1 ゲーム回す最小確認用です。

```powershell
python .\tests\local_sim\test_local_game.py
```

- `battle_start(...)` の開始エラー
- `agent(obs_dict)` が最後まで合法手を返せるか
- 対戦が最後まで進むか

をざっくり確認できます。

### 複数試合やランダム対戦で確認する

`local_test_advanced.py` は、試合数・相手方針・詳細ログを切り替えられる確認用スクリプトです。

```powershell
# 自分同士で3試合
python .\tests\local_sim\test_local_game_advanced.py --games 3

# ランダム相手に10試合
python .\tests\local_sim\test_local_game_advanced.py --games 10 --opponent random

# 1手ごとの選択も表示
python .\tests\local_sim\test_local_game_advanced.py --games 1 --verbose
```

- `--opponent self`：両プレイヤーとも `main.agent`
- `--opponent random`：相手は合法手からランダム選択
- `--games N`：N 試合まとめて実行
- `--verbose`：各ターンの選択内容を表示

### 内部で使っている API

上のスクリプトは内部で `cg.game` を使っています。

まずは `sample_submission` フォルダに移動してから実行します。`main.py` は相対パスで `deck.csv` を読むため、別フォルダで実行すると失敗しやすいです。

```powershell
cd C:\dev\pokemon-tcg-ai-battle\sample_submission
```

### 最小の動作確認

`local_test.py` は、現在の `main.py` を使って 1 試合だけ回す最小の確認用スクリプトです。

```powershell
python .\tests\local_sim\test_local_game.py
```

`errorType` と `result` が表示されます。

### 複数試合や比較

`local_test_advanced.py` は、複数試合の実行や `random` 相手との比較に使います。

```powershell
python .\tests\local_sim\test_local_game_advanced.py
python .\tests\local_sim\test_local_game_advanced.py --opponent self --games 10
python .\tests\local_sim\test_local_game_advanced.py --opponent random --games 100
python .\tests\local_sim\test_local_game_advanced.py --opponent random --games 10 --verbose
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

独自の検証スクリプトを書きたい場合だけ、これらを直接使う想定です。

`cg.sim`、`cg.utils`、`libcg.so` / `cg.dll` は内部実装のため、通常は直接使用・変更しません。
