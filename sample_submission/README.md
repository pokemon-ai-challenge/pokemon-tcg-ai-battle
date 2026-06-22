# sample_submission

コンペが提供するサンプル提出コードです。チームの AI エージェント実装の起点として使用します。

## フォルダ構成

```
sample_submission/
├── main.py       # エージェント実装 (ここを編集する)
├── deck.csv      # 使用デッキ: カード ID を 60 行で記載
└── cg/           # コンペ提供のゲームエンジン (変更不要)
    ├── api.py    # Observation クラスなど型定義
    ├── game.py   # ゲームロジック
    ├── sim.py    # シミュレーター
    └── utils.py
```

## エージェントの実装方法

`main.py` の `agent(obs_dict)` 関数を実装します。

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)

    if obs.select is None:
        # 初回のみ: 60 枚のカード ID リストを返す (デッキ選択)
        return read_deck_csv()

    # 通常ターン: 選択肢のインデックスリストを返す
    # 返すリストの長さは obs.select.minCount 以上 obs.select.maxCount 以下
    # インデックスは 0 以上 len(obs.select.option) 未満、重複不可
    return [...]
```

## デッキの編集

`deck.csv` にカード ID を 1 行 1 枚、合計 60 行で記載します。  
使用可能なカード ID は `data/EN_Card_Data.csv` / `data/JP_Card_Data.csv`、またはカード一覧 PDF で確認できます。

カードリスト参照・印刷ツールを使うと効率よくデッキを組めます。  
→ [cardlist_referenced/README.md](../cardlist_referenced/README.md)
