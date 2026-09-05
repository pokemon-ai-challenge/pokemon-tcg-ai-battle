# アーキタイプ分布スナップショット(2026-07-19)

3,898 リプレイ(7,796 デッキ、201〜2000位の深層取得を含む)を
`kaggle_replays/deck_predictor/label_decks.py` で自動ラベリングした結果。
元データ: `kaggle_replays/deck_predictor/output/deck_labels.jsonl` / `label_report.md`

## 全21クラスの分布

| アーキタイプ | 件数 | 割合 |
|---|---:|---:|
| alakazam | 2554 | 32.76% |
| mega_lucario_ex | 1161 | 14.89% |
| archaludon_ex | 975 | 12.51% |
| crustle | 617 | 7.91% |
| **other(未分類)** | 541 | 6.94% |
| marnie_grimmsnarl_ex | 525 | 6.73% |
| dragapult_ex | 475 | 6.09% |
| mega_starmie_ex | 442 | 5.67% |
| shirona_garchomp_ex | 160 | 2.05% |
| mega_abomasnow_ex | 96 | 1.23% |
| rocket_honchkrow | 57 | 0.73% |
| omatsuri_ondo | 51 | 0.65% |
| n_zoroark_ex | 40 | 0.51% |
| mega_froslass_ex | 30 | 0.38% |
| ogerpon_teal_ex | 19 | 0.24% |
| yadoking | 18 | 0.23% |
| kamitsuorochi_ex | 14 | 0.18% |
| takeruraiko_ex | 14 | 0.18% |
| oliva_ex | 4 | 0.05% |
| gekkouga_ex | 2 | 0.03% |
| toxtricity | 1 | 0.01% |

合計: 7,796 デッキ

## 傾向・所見

- **上位4アーキタイプ(alakazam / mega_lucario_ex / archaludon_ex / crustle)で全体の67.9%**を占める、かなり偏った分布。
- `other` は 6.94% で健全(ラベリングロジックの異常なし。90%等に張り付いていれば要疑うが、そうなっていない)。
- 件数1〜4件のクラス(toxtricity, gekkouga_ex, oliva_ex)は統計的にほぼ学習できておらず、予測しても信頼できない。
  データが増えるまでは参考程度に扱うこと。
- **719件(上位帯のみ)時点では7クラスしか観測されていなかった**が、201〜2000位の深層データを足したことで
  14クラスが新規に出現した。特に mega_lucario_ex・archaludon_ex は合計で27%を占める大きな勢力。
- **母集団依存の偏りに注意**: 上記は「取得した3,898リプレイ全体」での分布であり、実戦で当たる分布とは異なる。
  `adjust_prior.py` のターゲット窓(直近14日×上位200位、600デッキ)で見ると:
  - alakazam が 58.45% まで上昇(全体では32.76%)
  - mega_lucario_ex・archaludon_ex はほぼ消失(各0.32%程度)
  → **mega_lucario_ex・archaludon_ex は主に中〜下位帯(201〜2000位)で使われており、上位帯にはほとんど生き残っていない**。
  実戦(自分のレート帯)で当たる分布を見るときは、この全体分布ではなくチューニング後の重み(`deck_predictor_weights.json` の `meta.target_window.pi_prime`)を参照すること。

## 参照

- 集計元スクリプト: `kaggle_replays/deck_predictor/label_decks.py`
- 全体設計: [ml-predictor-plan.md](ml-predictor-plan.md) / [ml-predictor-phase2-scaling.md](ml-predictor-phase2-scaling.md)
