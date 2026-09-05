# 模倣学習v4 別デッキ提出記録（イワパレス / マリィのオーロンゲex）

作成: 2026-07-24
目的: アーキタイプ別に量産した模倣ポリシーを実 Kaggle で測るためのデータ取り。
関連: `docs/architecture/current-algorithm-overview.md`（v4 の実装）、
`results/2026-07-22_attackplan_kaggle_submission.md`（v4 本体の提出記録）、
`kaggle_replays/policy_net/run_archetype_pipeline.py`（アーキ別学習）、
`league/round_robin.py`（対フィールド評価）、`results/round_robin/_matrix.json`。

## 提出履歴

| ref | 提出日時(UTC) | 説明 | publicScore |
|---|---|---|---|
| 54934237 | 2026-07-23 18:30:12 | 模倣学習v4 イワパレスデッキ | **665.9** |
| 54934240 | 2026-07-23 18:30:20 | 模倣学習v4 マリィのオーロンゲexデッキ | **601.0** |

参考(直近): v4 alakazam 749.8/596.0(同一tarball, 154点差) / v3 744.0 / v5(hidden128) 658.9。
→ crustle 665.9・marnie 601.0 はいずれも既存帯(596〜750)内。**両者の64点差は分散の範囲内で有意でない**
(同一tarballで154点ぶれる)。crustleのローカル対フィールド1位(70.9%)は単発Kaggleスコアには突出せず
= ローカル評価は単発スコアを予測しない(実相手プール≠ローカル8アーキ, かつ分散大)。
判定には各デッキ3〜5回の再提出が必要。

## アルゴリズムは v4 と同一・デッキと重みのみ差し替え

- v4 = `AGENT_TYPE=ml_policy` + config `ml_lethal_attackplan_v0only`（確定リーサル探索 +
  attack_plan v0、attack_hybrid OFF）+ PolicyModel 構成C（concentrated=上位順位を強く重み付け）。
- 今回の2提出は **推論・config・学習レシピをすべて v4 と同一**にし、以下だけ差し替えた:
  - `deck.csv` ← 各アーキタイプ代表デッキ（`kaggle_replays/meta_analysis/archetype_decks/<arch>/01.csv`）
  - `ptcg_ai/learning/policy_weights.json` ← 各アーキタイプ専用重み（`run_archetype_pipeline.py` で
    生成、build_features `--weight-scheme concentrated`）。
- `decks/`（手書きプロファイル）は alakazam 用のままだが、**ml_policy の意思決定経路は
  `decks/`/`profile_registry` を使わない**（使うのは rule_based / action_selection のみ、v4 では不使用）
  ため打ち回しに影響しない。import 可能性のためだけに同梱。
- crustle: 対フィールド期待勝率 1位(70.9%)、対 ex 全勝（dragapult 97% 等）、alakazam にも 61%。
  marnie_grimmsnarl_ex: フィールド3位(53.7%)、Kaggle上位100位での採用率2位（強豪好み）。

## 検証

- 各 tarball を空ディレクトリへ展開し `main.agent` で自己対戦1ゲーム完走を確認
  （crustle 32ターン / marnie 16ターン、エラー0）。tarball 構造は v4 と同一
  （ルート直置き: main.py, deck.csv, cg/, configs/, decks/, ptcg_ai/）。

## 注意（スコアの読み方）

- publicScore は**同一 tarball でも大きくばらつく**（v4 本体で 596.0 ↔ 749.8）。相手プールとの
  マッチング運の分散が大きいので、**1回の値で強さは判断しない**。複数回/期間を置いた再提出が必要。
- ローカルの対フィールド評価（round_robin）は模倣ポリシー同士の相性で、相手の練度差が混入する
  （ex を下手に回すと crustle 有利）点にも留意。
