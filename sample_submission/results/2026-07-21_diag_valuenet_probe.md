# 診断Step1: value網の摂動probe(真犯人(A)の判定+山札切れ感度)

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [pimc-null-diagnosis-implementation-plan.md](../plans/individual/shogo/pimc-null-diagnosis-implementation-plan.md) Step1
- コード変更: なし(`ptcg_ai`/`league`/`configs` は無変更)。分析専用スクリプト
  `kaggle_replays/_diag_valuenet_probe.py`(読み取り専用、リプレイのみ参照)。

## 方法

自チーム(`MORIOKA Tsuoi`)の既存リプレイ(`kaggle_replays/replays/`、364試合分が対象)から
`turn >= 4`の中盤〜終盤の実局面を200件サンプルし、`encoder.encode_state_from_state` で
特徴ベクトル化した上で、`ValueModel.predict_win_prob_from_features` に対し以下4種の
単一特徴摂動を行った:

| 摂動 | 内容 | 期待方向 |
|---|---|---|
| `self_deck_count` | 実測値 → 0 | 勝率**低下** |
| `self_prize_remaining` | 実測値 → -2(下限0) | 勝率**上昇**(自分がサイドを取る想定) |
| `opp_prize_remaining` | 実測値 → -2(下限0) | 勝率**低下**(相手がサイドを取る想定) |
| `prize_diff` | 自分に有利な方向へ-2 | 勝率**上昇** |

あわせて、150試合分の実トラジェクトリで (turn, self_deck_count, predict_win_prob) を記録し、
試合内の Pearson 相関(self_deck_count vs win_prob)を勝敗別・山札切れ負け別に集計した。

**既知の注意点**: 単一特徴の摂動は非現実的な盤面(deck_count だけ減って discard_count 等が
整合しない)を作るため、学習分布外の入力になる。ここでの判定は「その方向への感度が
そもそも存在するか」という純粋な単調性チェックであり、定量的な効果量の推定ではない。

## 結果

### 摂動probe(n=200局面)

| 摂動 | 平均Δ(勝率) | 期待方向と一致した局面数 |
|---|---|---|
| `self_deck_count`(高→0) | **+0.0048**(ほぼ0、符号も期待と逆) | **92/200(46.0%)** — コイントス以下 |
| `self_prize_remaining`(-2) | +0.0451 | 187/200(93.5%) |
| `opp_prize_remaining`(-2) | -0.0575 | 188/200(94.0%) |
| `prize_diff`(有利方向へ-2) | +0.0578 | **200/200(100%)** |

### 実トラジェクトリ相関(n=150試合、self_deck_count vs win_prob)

| 区分 | n | 平均相関係数 |
|---|---|---|
| 勝ち試合 | 69 | -0.610 |
| 負け試合 | 80 | +0.555 |
| うち「山札切れ負け」(自分のdeckCountが試合中に一度でも0を記録) | 14 | +0.536 |

山札切れ負け(n=14)の相関(+0.536)は、山札切れ以外を含む負け試合全体(+0.555)と
**ほぼ同じ**であり、山札切れに特有のシグナルは見られない。

## 判定

**value網は `self_prize_remaining` / `opp_prize_remaining` / `prize_diff` には明確に素直な
方向で反応するが、`self_deck_count`(山札枚数)にはほぼ反応しない(コイントス以下の
的中率)。** これは「摂動probeがそもそも機能しているか」への疑いを晴らす対照実験としても
機能している(サイド系3特徴は高い的中率で応答しており、probe自体は健全)。

トラジェクトリ相関で「負け試合では deck_count と win_prob が正相関(山札が減るにつれ
win_prob も下がる)」という一見それらしい傾向が出るが、これは**山札切れ特有ではなく、
負けている試合全般に共通する傾向**(時間経過とともに盤面が悪化し、山札も自然に減っていく
という交絡)であり、山札切れリスクを value 網が特別に検知しているわけではないと判断する。
山札切れ負け(n=14)の相関が他の負け(n=66)と有意に異ならない点がこれを裏付ける。

**結論: value網は山札枚数の減少を勝率低下として明確には捉えていない(No)。**

## (A)(B)(C) 判定への反映

計画のルーティング表(§Step5)に照らすと、本結果は **「value網が deck_count/サイドに
素直に反応しない」→ 真犯人(A)** に部分的に該当する: サイド系特徴には健全に反応するため
value網が全面的に壊れているわけではないが、**山札切れという実戦の主要敗因
([[project_deckout_loss_cause]]、29%)に対して value網が盲目である**ことが明確になった。
探索(PIMC/lethal)の目的関数がこの盲点を継承している以上、探索をどれだけ作り込んでも
山札切れ絡みの判断は改善しない。

**山札切れ対処の方向性への示唆**: 山札切れは「value網の感度不足」が一因である可能性が
高い(deck_count を素直な特徴として学習していない)。単純なルールガード(deck_count が
閾値以下でドロー系カードの使用を抑制する等)よりも、value網の特徴量・学習データに
山札切れリスクをより明示的に反映させる方が筋が良い可能性がある。ただし本診断は
Step4(山札切れ敗因の内訳)と合わせて判断すべきで、デッキ構築側の問題(ドロー過多)が
主因であれば value網改修だけでは解決しない。

## 次のステップとの関係

- [Step2: determinizationスイープ](./2026-07-21_diag_determinization_sweep.md)(実施中/予定)は
  (B)(strategy fusion)の切り分け。
- [Step4: 山札切れ敗因の拡大標本分類](./2026-07-21_diag_deckout_classification.md)(実施予定)は、
  本Step1の「value網は山札切れに盲目」という結果を前提に、敗因がエージェント判断か
  デッキ構築かを判定する。
