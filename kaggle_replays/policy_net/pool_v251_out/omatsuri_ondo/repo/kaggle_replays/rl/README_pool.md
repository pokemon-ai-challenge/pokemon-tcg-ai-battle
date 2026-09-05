# kaggle_replays/rl — 相手プール対応の追加分

`rl/` にはチーム既存の資産と、我々が足した分が混在している。**どれが誰のものかを最初に把握すること。**

## ファイルの区別

### チーム既存（変更しない）

| ファイル | 役割 |
|---|---|
| `collect_parallel.py` | 並列トラジェクトリ収集。**学習側1 vs 相手1固定** |
| `train_v3.py` | PPO + Critic + GAE。`--learner-arch` / `--opponent-arch` で1対1固定 |
| `torch_policy.py` | torch 側の方策（pure-Python `PolicyModel` と数値一致） |
| `rollout.py` `collect_field.py` `eval_field.py` `eval_m3.py` `train_v2.py` `train_field.py` `train_dragapult_poc.py` | 各種実験 |
| `test_parallel.py` `test_parity.py` `test_rollout.py` | 既存のテスト |
| `distributed/` | Colab / Kaggle へ試合生成を分散する基盤（`push_kaggle.py` ほか） |

### 我々の追加

| ファイル | 役割 |
|---|---|
| `collect_pool.py` | **相手プール**に対する収集。`collect_parallel` を変更せず、その `_play_one` を import して使う |
| `pools.py` | 学習側・相手プールのレジストリ（重みとデッキの対応） |
| `train_pool.py` | プール対応の PPO。`train_v3` からアルゴリズムを import し、収集だけ差し替える |
| `eval_matrix.py` | モデル × 相手 の勝率行列とオラクル振り分けの計算 |
| `test_collect_pool.py` | 実収集48試合での検証（相手均等・席均等・学習側のみ記録） |
| `test_collect_pool_tasks.py` | タスク列生成の単体テスト（試合を回さない） |
| `distributed/notebooks/push_train_pool.py` | `train_pool.py` を Kaggle Notebook で走らせる |

## 依存関係

```
train_pool.py
  ├── train_v3.py        (Critic / build_padded / compute_gae / policy_logp_entropy / export_temp / wilson_lo)
  ├── collect_pool.py
  │     └── collect_parallel.py   (_play_one / _W)
  ├── pools.py
  └── league/run_league.py        (read_deck_csv_file)

実行に必要な外部資産:
  sample_submission/cg/                                (libcg.so / cg.dll)
  sample_submission/ptcg_ai/learning/policy_weights*.json
  kaggle_replays/meta_analysis/archetype_decks/<arch>/01.csv
```

**アルゴリズムを再実装していない。** `train_v3` から import している。固定相手アームも `train_pool.py` に
「プールの要素数1」を渡して走らせる。別実装で比較すると実装差が交絡するため。

## collect_pool.py が保証すること（テスト済み）

1. **相手が均等に当たる** — 4種すべて games=12（48試合時）
2. **相手ごとに席が均等** — 4種すべて seat0=6, seat1=6
3. **学習側の decision のみ記録される** — 3,562 step すべてが learner の logprob と一致（不一致0）

3番は「`cur.yourIndex == learner_index` だから大丈夫」で済ませず、記録された各 step の logprob を
learner モデルで**再計算して**確認している。さらに同じ step を相手モデルでも計算し、
65.2% が相手とは不一致であることを示している。これが無いと「両モデルの出力が同じで比較が空振り」に気づけない。

2番は**回帰テスト**。以前、相手の周期4と席の周期2が同期して archetype1/3 が常に先攻になるバグを踏んだ。
`build_tasks` のテストは `(opp_idx, learner_index)` の**同時分布**を数える。マージナルだけ見ても検出できない。

## 落とし穴

- **cg の乱数は Python の seed で制御できない。** `lib.BattleStart` にシード引数が無く、`cg/` 内に
  seed/srand の設定も無い。同一シード4回で勝率 3/6/14/7 of 48（`workers=1` でも再現しない）。
  **共通乱数による対応ありの比較は成立しない。** A/B は対応なしの統計量で扱う
- **並列化はプロセスのみ。** `cg` はプロセス全体に1つの対局状態を持つ。スレッド不可
- **温度は非対称。** 学習側だけが softmax サンプリングで、相手は argmax 固定（`_play_one` の仕様）。
  温度を上げるほど学習側が不利になる（alakazam の対プール勝率は 0.521 → 0.271）
- **学習側の選び方で結果が壊れる。** 勝率が極端だと勝敗がデッキ相性でほぼ決まる。
  `value_net_probe/calibrate_learner.py` で 50% 近辺のものを選ぶこと。dragapult_ex は 0.167 / 0.062 で使えない
- **`--games-per-iter` は「相手の数 × 2(席)」の倍数**にすると端数が出ない

## 使い方

```bash
# 固定相手（プール要素数1）
python kaggle_replays/rl/train_pool.py --learner alakazam \
    --train-opponents alakazam --eval-fixed alakazam --iters 20 --tag fixed

# 相手プール
python kaggle_replays/rl/train_pool.py --learner alakazam --iters 20 --tag pool

# 評価行列（担当表は事前登録。列ごとの事後選択は禁止）
python kaggle_replays/rl/eval_matrix.py \
    --models "BC=policy_weights.json,SPEC=policy_weights_alakazam_pool_armA_fixed.json" \
    --games-per-cell 400 \
    --routing "alakazam=SPEC,crustle=BC,marnie_grimmsnarl_ex=SPEC,archaludon_ex=SPEC"
```

**学習用プール（`--train-opponents`）と評価用プール（`--eval-opponents`）は別引数。**
評価プールを凍結しないと、勝率が上がったのか相手が弱くなったのかを区別できない。

ローカル8コアで 20イテレーション（10,240試合 + 評価）約51〜65分。
