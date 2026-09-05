# バリューネットワーク学習データ 監査レポート

- 対象リプレイディレクトリ集計対象件数: 4698
- データセット出力: `C:\dev\pokemon-tcg-ai-battle\kaggle_replays\training_data\value_positions.jsonl.gz`

## エピソード数

- 総数: 4698
- 採用: 4690 (99.83%)
- スキップ計: 8

| スキップ理由 | 件数 |
|---|---:|
| 引き分け(rewards 同値) (draw) | 3 |
| rewards 欠損 (missing) | 0 |
| rewards 不正な形 (malformed) | 5 |
| JSON パースエラー / 例外 (parse_error) | 0 |

## 局面数

- 総局面数: 614209
- 1エピソードあたり平均局面数: 130.96

## ターン帯別の局面数分布

| ターン帯 | 件数 | 割合 |
|---|---:|---:|
| 1-2 | 96443 | 15.70% |
| 3-5 | 153015 | 24.91% |
| 6-10 | 210313 | 34.24% |
| 11+ | 154438 | 25.14% |
| 不明 | 0 | 0.00% |

## ラベルバランス

両プレイヤー視点の局面を含むため、リプレイのペア構造上ほぼ50/50になるはず(大きくずれている場合はラベル付与ロジックのバグを疑う)。

| label | 件数 | 割合 |
|---|---:|---:|
| 勝ち(1) | 323673 | 52.70% |
| 負け(0) | 290536 | 47.30% |

## rank_at_fetch の分布

- あり: 436186
- なし(null): 178023

| 順位帯 | 件数(rank_at_fetch ありのうち) | 割合 |
|---|---:|---:|
| 1-50 | 45140 | 10.35% |
| 51-200 | 762 | 0.17% |
| 201-1000 | 179567 | 41.17% |
| 1001+ | 210717 | 48.31% |

## アーキタイプ分布・マッチアップペア

`kaggle_replays/deck_predictor/output/deck_labels.jsonl`(既存の `label_decks.py` 出力)を episode_id + player_index で結合して集計。

### アーキタイプ分布(採用エピソード内、両プレイヤー視点で計上)

| アーキタイプ | 件数 | 割合 |
|---|---:|---:|
| alakazam | 2810 | 29.96% |
| mega_lucario_ex | 1253 | 13.36% |
| archaludon_ex | 1078 | 11.49% |
| crustle | 734 | 7.83% |
| other | 679 | 7.24% |
| dragapult_ex | 623 | 6.64% |
| marnie_grimmsnarl_ex | 590 | 6.29% |
| mega_starmie_ex | 486 | 5.18% |
| shirona_garchomp_ex | 181 | 1.93% |
| ogerpon_teal_ex | 142 | 1.51% |
| kamitsuorochi_ex | 139 | 1.48% |
| mega_abomasnow_ex | 138 | 1.47% |
| yadoking | 126 | 1.34% |
| oliva_ex | 100 | 1.07% |
| mega_froslass_ex | 75 | 0.80% |
| rocket_honchkrow | 72 | 0.77% |
| omatsuri_ondo | 58 | 0.62% |
| takeruraiko_ex | 49 | 0.52% |
| n_zoroark_ex | 41 | 0.44% |
| toxtricity | 4 | 0.04% |
| gekkouga_ex | 2 | 0.02% |

### マッチアップペア上位20(両プレイヤーのアーキタイプが判明しているエピソードのみ)

| マッチアップ | 件数 |
|---|---:|
| alakazam vs alakazam | 431 |
| alakazam vs mega_lucario_ex | 387 |
| alakazam vs archaludon_ex | 324 |
| alakazam vs crustle | 226 |
| alakazam vs other | 194 |
| alakazam vs dragapult_ex | 184 |
| alakazam vs marnie_grimmsnarl_ex | 176 |
| alakazam vs mega_starmie_ex | 155 |
| archaludon_ex vs mega_lucario_ex | 127 |
| crustle vs mega_lucario_ex | 105 |
| dragapult_ex vs mega_lucario_ex | 100 |
| archaludon_ex vs archaludon_ex | 93 |
| mega_lucario_ex vs other | 92 |
| mega_lucario_ex vs mega_lucario_ex | 85 |
| archaludon_ex vs dragapult_ex | 77 |
| archaludon_ex vs other | 76 |
| marnie_grimmsnarl_ex vs mega_lucario_ex | 72 |
| mega_lucario_ex vs mega_starmie_ex | 69 |
| archaludon_ex vs marnie_grimmsnarl_ex | 65 |
| archaludon_ex vs crustle | 65 |

## 出力ファイルサイズ

- `C:\dev\pokemon-tcg-ai-battle\kaggle_replays\training_data\value_positions.jsonl.gz`: 144.84 MB
