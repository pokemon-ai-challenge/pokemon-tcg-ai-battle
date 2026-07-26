# meta_analysis — 環境メタ分析 (#1)

Kaggle リプレイから「アーキタイプ・シェアが上位と全体で差があるか」「未登録の高シェア
アーキタイプはあるか」「各アーキタイプにどのカードが入っているか」を出す一式。
既存の `deck_predictor/output/{deck_db,deck_labels}.jsonl` と `index/episodes_master.jsonl`
を結合するだけで、新規データ収集は不要。

## 実行

```bash
cd kaggle_replays/meta_analysis
python analyze_meta.py --top-rank 100      # 集計 -> output/meta_report.{json,md}
python extract_archetype_decks.py --top-decks 5  # 上位デッキ -> archetype_decks/<arch>/NN.csv
python build_viewer.py                     # ビュアー -> output/meta_viewer.html
```

`output/meta_viewer.html` はブラウザで開くだけで動く自己完結HTML（外部依存なし）。

## 出力

- `output/meta_report.json` / `.md` … シェア(上位/field/使用者数)、`other`内訳、採用率。
- `archetype_decks/<archetype>/NN.csv` … 上位パイロットの複数デッキ(deck.csv互換, 60行=カードID)。
  自己対戦(リーグ)のスパーリング相手用。`manifest.json` にpilot/rank/score/出典episode。

## 集計上の注意（結論に効く）

- **シェアは「出現数」と「使用者数(distinct player)」で分けて出す。** 少数のグラインダーが
  同じ型を連投して出現数を膨らませる（例: oliva_ex 101出現/3人）。self-play の相手抽選
  重みは使用者数側を使う方が実環境に近い。
- rank は `deck_db.rank_at_fetch` を優先し、無ければ `episodes_master` で補完。約2,873件は
  rank不明（主に相手側 player_index=1）で上位/field比較からは除外、全体シェアには算入。
- 上位(rank≤100)サンプルは少数の上位プレイヤーに偏る。`top_pct` は使用者数と併読する。

## カード画像表示（フェーズ2・未実装）

`pdf_card_editor` の画像キャッシュは PDF ページ単位で、カードID→画像の対応表が無い。
採用率テーブルへの画像添付は、カードID→画像のインデックスを別途用意してから着手する。
