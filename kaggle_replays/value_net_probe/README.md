# value_net_probe — 自己対戦による局面データの生成

本ブランチには**生成側**だけがある。解析側(`early_game_signal.py` など)と測定結果は
`feature/top-replay-utilization` にある。

分けた理由: 生成側は `kaggle_replays/rl/collect_pool.py` に依存し、それがさらに
`collect_parallel.py`・`league/run_league.py`・`meta_analysis/archetype_decks/`・
アーキタイプ別の重み31個に依存する。これらは `distributed-selfplay` 系にしか無い。
解析側は sklearn と numpy だけで動くのでどちらのブランチでも動く。

## ファイル

| ファイル | 内容 |
|---|---|
| `selfplay_positions.py` | 相手プールと自己対戦し `(X=166次元の状態特徴, y=勝敗, turn, game_id, opponent, learner)` を npz に落とす |
| `calibrate_learner.py` | 相手プールに対して勝率が50%近辺になる学習側を探す |

## なぜ較正が要るのか

勝率が極端な学習側を使うと、勝敗がターン0のデッキ相性でほぼ決まってしまい、
「序盤の局面が勝敗を予測するか」という問いの答えが相性識別に汚染される。

実測(各48試合、相手プールは alakazam / crustle / marnie_grimmsnarl_ex / archaludon_ex):

| 学習側 | 温度0.05 | 温度1.0 |
|---|---:|---:|
| crustle | 0.604 | 0.667 |
| alakazam(production) | 0.521 | 0.271 |
| mega_lucario_ex | 0.521 | 0.375 |
| marnie_grimmsnarl_ex | 0.479 | 0.354 |
| archaludon_ex | 0.479 | 0.271 |
| **dragapult_ex** | **0.167** | **0.062** |

`--learners` の既定は50%近辺の4種。dragapult_ex は使わない。

温度を上げると全モデルが不利になるのは、**学習側だけが softmax サンプリングで
相手は argmax 固定**だから(`collect_parallel._play_one` の仕様)。温度1.0 は
`train_v3` の既定であり、PPO が実際に収集する分布でもある。

## 使い方

```
python kaggle_replays/value_net_probe/selfplay_positions.py \
    --games 4000 --temperature 0.05 --seed0 200000 \
    --out kaggle_replays/value_net_probe/selfplay_t005.npz
```

`--games` は「学習側の数 × 相手4種 × 2(席)」の倍数にすると端数が出ない。
8ワーカーで1試合あたり約0.2秒(4,000試合で約13分)。

## 注意

- **同一シードでも再現しない。** `lib.BattleStart` にシード引数が無く、`cg/` 内に
  seed/srand の設定も無い。Python の `random.seed()` は方策のサンプリングしか決めない。
  同一シード4回で勝率 3/6/14/7 of 48。共通乱数による対応ありの比較はできない
- **記録されるのは学習側の decision のみ**(`collect_parallel._play_one` の仕様)。
  両視点の漏洩は起きないが、同一試合内の step は相関するので分割は `game_id` 単位にする
- `*.npz` は gitignore 対象(数十MB、再生成可能)
