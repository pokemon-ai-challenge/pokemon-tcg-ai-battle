# バリューネットワーク学習データ 監査レポート

- 対象リプレイディレクトリ集計対象件数: 300
- データセット出力: `C:\Users\rinnz\Documents\pokemon\pokemon-tcg-ai-battle\kaggle_replays\training_data\value_positions.jsonl.gz`

## エピソード数

- 総数: 300
- 採用: 300 (100.00%)
- スキップ計: 0

| スキップ理由 | 件数 |
|---|---:|
| 引き分け(rewards 同値) (draw) | 0 |
| rewards 欠損 (missing) | 0 |
| rewards 不正な形 (malformed) | 0 |
| JSON パースエラー / 例外 (parse_error) | 0 |

## 局面数

- 総局面数: 47282
- 1エピソードあたり平均局面数: 157.61

## ターン帯別の局面数分布

| ターン帯 | 件数 | 割合 |
|---|---:|---:|
| 1-2 | 6272 | 13.27% |
| 3-5 | 12487 | 26.41% |
| 6-10 | 19961 | 42.22% |
| 11+ | 8562 | 18.11% |
| 不明 | 0 | 0.00% |

## ラベルバランス

両プレイヤー視点の局面を含むため、リプレイのペア構造上ほぼ50/50になるはず(大きくずれている場合はラベル付与ロジックのバグを疑う)。

| label | 件数 | 割合 |
|---|---:|---:|
| 勝ち(1) | 25311 | 53.53% |
| 負け(0) | 21971 | 46.47% |

## rank_at_fetch の分布

- あり: 32918
- なし(null): 14364

| 順位帯 | 件数(rank_at_fetch ありのうち) | 割合 |
|---|---:|---:|
| 1-50 | 32918 | 100.00% |
| 51-200 | 0 | 0.00% |
| 201-1000 | 0 | 0.00% |
| 1001+ | 0 | 0.00% |

## アーキタイプ分布・マッチアップペア

TODO: アーキタイプ分布は deck_predictor 連携で別途(`kaggle_replays/deck_predictor/output/deck_labels.jsonl` が見つかりませんでした)。

## 出力ファイルサイズ

- `C:\Users\rinnz\Documents\pokemon\pokemon-tcg-ai-battle\kaggle_replays\training_data\value_positions.jsonl.gz`: 11.20 MB
