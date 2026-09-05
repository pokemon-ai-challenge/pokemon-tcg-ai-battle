# (ii) 戦略残差NN 最小実験の設計 — self-play RL で②③レイヤーを学習

作成: 2026-08-01 / 前提: [direction-and-rationale-2026-08.md](direction-and-rationale-2026-08.md)
/ 較正評価: `kaggle_replays/_eval_current_meta.py` / メタ: `kaggle_replays/meta_analysis/current_meta_shares_2026-08-01.json`

## 0. なぜこの形か（設計判断の根拠）

- **ルールは限界的**（本セッションで deckout/Boss/key-card の3診断が「模倣は単一決定ではまともに打てる」を示した）。**模倣を超えるには学習で、教師信号は勝敗（self-play RL）**（模倣で学習させても床の再現）。
- ただし**模倣は強い床**。だから**床を凍結し、その上に小さな"残差"だけを RL で学習**する（residual RL）。これは (a) 床を壊さず安全、(b) 小容量で学習が速い、(c) アーキ的に「②③戦略が④行動をバイアスする」を最小実装、の3点で最短。
- **目的関数は較正済み評価**：`climb_600_699` 加重勝率（我々の現在地の帯）。**flat RL(i) と同じ物差しで測る**ので直接比較できる。

## 1. アーキテクチャ（最小）

```
状態 obs ──encoder(既存)──▶ 特徴 z
                            │
   [凍結] 模倣PolicyModel ─▶ logits_imit(各option)
                            │
   [学習] 戦略残差NN g(z, belief) ─▶ bias(各option or optionカテゴリ)
                            │
   最終行動 = argmax( logits_imit + α · bias )   (α は温度/スケール, 小さく始める)
```

- **凍結**: 既存 `PolicyModel`（`policy_weights.json`）はそのまま、勾配を流さない。
- **残差NN g**: 小さい MLP。入力 = 既存 encoder の状態特徴 z ＋ **belief（相手アーキ推定 `opponent_tracker.current_prediction()`、無ければ観測ワンホット）**。出力 = option（またはOptionTypeカテゴリ）への bias ベクトル。**容量は小さく**（隠れ 32-64、1-2層）。
- **接続点**: `ml_policy_agent` の既存 config-gated パターン（`_try_deck_sustain` と同じ流儀）で `_try_strategy_residual` を1つ足す。既定 config には無し＝本番不変。残差は logits を「バイアスするだけ」で、床の合法手マスク・リーサル探索は据え置き。

## 2. 学習（self-play RL）

- **手法**: REINFORCE / PPO（既存 `kaggle_replays/rl` の rollout/collect/train 基盤を流用）。**g のみ更新、模倣は凍結**。
- **報酬**: 勝敗（+1/-1）。必要なら PBRS（既存 densems の密報酬設計を流用）で分散低減。
- **対戦相手（field）**: **現在メタ climb帯で加重**（`_eval_current_meta.py` の `climb_600_699`）。特に**足枷の crustle / rocket_mewtwo を厚めに**サンプリング（弱点を直接学習させる）。ミラー(alakazam)も入れる。
- **デッキ**: `deck.csv`（= Plan A）固定。

## 3. 評価と採用ゲート

- **物差し**: `_eval_current_meta.py` の `climb_600_699` 加重勝率（＋各対面）。
- **採用条件**（[[project_measurement_protocol]] 準拠）:
  - climb_600_699 加重が **有意に上昇**（Wilson CI / SPRT, 十分 n）。
  - **強い対面（dragapult/mirror/lucario）が非リグレッション**（残差が床を壊していない）。
  - 目安: 弱点 crustle/rocket_mewtwo が改善し、全体 climb 加重 +数pt。
- ダメなら**残差の効果ゼロ＝模倣床が最適**の証拠として記録し、(i) flat RL に一本化。

## 4. 正直なリスク / スコープ限定

- **一部の弱点はデッキ由来でありRLで直らない**：特に **crustle（ミル/グラインド）は Plan A がグラインド札を削った結果の可能性**。これは**残差NNでなく deck-tech（Enhanced Hammer/Wondrous Patch/回収を戻す）で直すべき**。→ 残差NNは「policy由来で直る弱点」に限定し、crustle は別途 deck-tech で。
- **残差が flat RL(i) と冗長になる恐れ**：g が結局 flat RL と同じものを学ぶだけなら、階層の利点は薄い。**(i) を主・(ii) を小実験**の位置づけ（[[project_decksustain_scaffold]] の結論と整合）。
- **本番の探索(pipeline)との相互作用**：pipeline のロールアウトも凍結模倣を使うので、残差を rollout policy にも入れるか要検討（まずは root 行動のみに適用して単純化）。

## 5. 実装ステップ（最小 → 検証）

1. `_try_strategy_residual`（config-gated, 既定OFF）＋残差NN g のスケルトン（forward だけ, 重みランダム）。unit test で「本番config不変・logits+bias の argmax」を確認。
2. self-play rollout に g を差し込む（`kaggle_replays/rl` の rollout を流用, 模倣凍結）。
3. climb帯 field（crustle/rocket_mewtwo厚め）で REINFORCE/PPO を少数 iter 回し、`_eval_current_meta.py` で climb_600_699 を測る。
4. 採用ゲート判定 → 通れば config で ON にした提出候補、ダメなら negative 記録して (i) に集約。

## 6. 最初の一手

Step 1（残差NNスケルトン＋config-gated接続＋unit）から。crustle は並行して deck-tech（グラインド札の一部復帰＝Plan C 候補）で対処し、`_eval_current_meta.py` で両方測る。
