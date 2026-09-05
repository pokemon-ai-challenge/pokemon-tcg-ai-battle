# フェーズC/D 実装方針(序盤過信改善の続き)

作成日: 2026-07-19
親ドキュメント: `early-confidence-improvement-plan.md`(背景・フェーズA/Bの結果はそちら)
進め方: **C → D の順**で実施。

## フェーズC: 予測サマリの共通モジュール化 + ビュアー表示改善

### 要件(ユーザー指示)

「未確定判定」と「根拠」はビュアー表示だけの機能にしない。将来、エージェント本体
(例: `search_begin` に渡す相手デッキの決定、MatchupPolicy の適用判断)や ML 側から
同じ基準を再利用するので、**判定ロジックはランタイムライブラリ側に置き、ビュアーは
その結果を表示するだけ**にする。

### C-1. 共通モジュール `sample_submission/ptcg_ai/opponent_modeling/prediction_summary.py`(新規)

純Python(外部依存なし)。提出環境でも動く。

```python
UNCERTAIN_THRESHOLD_DEFAULT = 0.6  # 判定基準の一元管理。利用側はこの定数を参照する

def summarize_prediction(predictor, observed_cards, turn, *,
                         uncertain_threshold=UNCERTAIN_THRESHOLD_DEFAULT,
                         top_n=3) -> dict
```

戻り値スキーマ(ビュアー・エージェント共通の契約。docstring に明記):

```json
{
  "status": "confident" | "uncertain" | "unready",
  "top": [{"deck_type": "...", "probability": 0.xx}, ...],   // top_n件・確率降順
  "top1_probability": 0.xx,
  "uncertain_threshold": 0.6,
  "evidence_count": 3,            // 予測器が evidence_count を持つ場合。無ければ null
  "explanation": {...} | null     // 予測器が explain() を持つ場合のみ(下記 C-2)
}
```

- `status`: `is_ready=False → "unready"`、top-1 確率 < しきい値 → `"uncertain"`、以外 `"confident"`。
- 予測器は duck-typing で受ける(`MLDeckPredictor` / `NBDeckPredictor` / `HybridDeckPredictor`
  のどれでも動く。`explain` / `evidence_count` は `getattr` で存在チェック)。
- テスト: `sample_submission/tests/unit/test_prediction_summary.py`。

### C-2. `HybridDeckPredictor.explain()` の追加

NB 側には実装済みの `explain()`(観測カード別の各クラス log 尤度寄与)を、ハイブリッドにも
パススルー + ブレンド文脈付きで公開する:

- 戻り値: `{"mode": ..., "weight_nb": <このevidence数で使ったw>, "evidence_count": ...,
  "nb_explanation": <NBのexplain()結果> | null}`。
- `lr_only` モード(NB 重みなし等)では `nb_explanation: null`(根拠なしを明示)。
- LR 単体(`MLDeckPredictor`)には explain を追加しない(線形係数×温度の寄与分解は
  誤解を招きやすく、根拠表示の主目的は NB 側で足りるため)。

### C-3. ビュアー表示

- `battle_review_viewer/ml_prediction_debug.py` を `summarize_prediction()` 利用に変更。
  既存 payload キー(`is_ready` / `turn` / `ranked`)は**後方互換のため残し**、
  `status` / `uncertain_threshold` / `evidence_count` / `explanation` を追加する
  (過去にエクスポート済みのリプレイ JSON には新キーが無いので、app.js は欠損を許容する)。
- `battle_review_viewer/web/app.js` ほか: ML 予測サブタブで
  - `status == "uncertain"` のとき: 1位を断定表示せず「未確定」バッジ + 候補列挙表示
  - `explanation` があるとき: 観測カード別の寄与(どのカードがどのアーキタイプを
    押し上げたか)を上位数件のクラスに絞って表示
  - 表示名は既存の `archetype_display_names.json` を利用
- ビュアーの既存機能(rough_predictor 表示等)には手を入れない。追加的変更のみ。

### C-4. 検証

- `pytest tests/unit/` 全パス(新規 test_prediction_summary.py 含む)
- Python 側スモーク: HybridDeckPredictor + summarize_prediction で
  evidence 0 / 汎用1枚 / 専用1枚 の3ケースの status・explanation を確認
- ビュアー: リプレイを1件エクスポートして payload に新キーが入ることを確認

### C: 実施結果(2026-07-19 完了)

- `prediction_summary.py` + テスト17件、`HybridDeckPredictor.explain()`、ビュアー表示
  (未確定バッジ + 候補列挙、カード別寄与の日本語表示)まで実装。ユニットテスト 86 passed。
- スモーク(デプロイ済みハイブリッド): `{}` → uncertain(top1 58.5%)/ Ultra Ball 1枚 →
  uncertain(21.2%)/ Alakazam 1枚 → confident(99.0%、根拠に Alakazam の寄与)。
- 後方互換を確認: 旧エクスポート済みリプレイ(新キーなし)は従来表示にフォールバック。
  実リプレイのエクスポートで新キー(status / evidence_count / explanation)の出力も確認済み。
- 補足: `summarize_prediction` の既定 top_n は 3(ビュアーは従来どおり8を明示指定)。
  エージェント側から使うときは必要に応じて top_n を指定する。

## フェーズD: 少数クラスのデータ収集 + 再学習・再フィット

### 背景

NB の尤度はクラス別デッキ数 N_c からの頻度推定。現状 toxtricity N=1、gekkouga_ex N=2、
oliva_ex N=4、kamitsuorochi_ex/takeruraiko_ex N=14、yadoking N=18、ogerpon_teal_ex N=19、
mega_froslass_ex N=30 と極小クラスが多く、これらの予測は当てにならない
(train_nb.py が N<30 に警告を出す)。また `fit_hybrid.py` の 50/50 安定性チェックで
evidence≥1 の全バケットが不安定(半分あたり約90エピソードでは w の選択がブレる)。

### D-1. 少数クラスを狙ったリプレイ収集

- 新スクリプト `kaggle_replays/fetch_minority_archetype_episodes.py`:
  1. `deck_predictor/output/deck_labels.jsonl` からクラス別デッキ数を集計し、
     N < しきい値(既定50)のアーキタイプを「少数クラス」と判定
  2. 少数クラスのデッキを使っていた (episode_id, team_name) を逆引きし、
     そのチームの他エピソードを `index/episodes_master.jsonl` / リーダーボード履歴から列挙
  3. 未取得分を既存の `_common.py` のダウンロード機構で `replays/` に追加取得
     (取得件数上限 `--max-episodes` 既定200、API 負荷への配慮で既存スクリプトと同じ待機を入れる)
- 同じチームが同じデッキを使い続けるとは限らないため、取得後のラベリングで実際に
  少数クラスが増えたかを確認する(増えなければ対象チームを広げて再実行)。

### D-2. パイプライン再実行と再フィット

```
extract_decks → label_decks → build_dataset → train → adjust_prior → calibrate
→ train_nb → fit_hybrid → compare_nb / evaluate --all --error-dump
```

- **fit_hybrid の採用方針(慎重側の原則、feedback: 確信度は強気にしない)**:
  50/50 安定性チェックを**通過したバケットのみ**新しいフィット値 w を採用する。
  不安定のままのバケットは現行デプロイ値(0→1.0 / 1→0.3 / 2-3→0.55 / 4+→0)を維持。
  特に evidence≥4 は、安定化して明確に log loss 改善が出ない限り w=0 のまま。
- デプロイ(`sample_submission/` への重みコピー + `--deploy`)は比較レポートで
  改善を確認してから行う。

### D-3. 成功基準

- 少数クラスの学習デッキ数が増える(目安: 各クラス N≥30)
- 全体評価(`--all`)で少数クラスの f1 改善(ogerpon 0.36 / takeruraiko 0.51 / oliva 0.00 が基準値)
- 固定 validation の全体 log loss(現行 hybrid 0.0356)を悪化させない
- fit_hybrid の安定バケットが増える(現状は evidence 0 のみ安定)

### D: 実施結果(2026-07-19)

**D-1 収集**: `fetch_minority_archetype_episodes.py` 新規実装。少数クラスを使っていたチームの
未取得エピソードを800件追加取得(3,898→4,698リプレイ)。

クラス別デッキ数(before→after): oliva_ex 4→101、kamitsuorochi_ex 14→139、yadoking 18→126、
ogerpon_teal_ex 19→142、mega_froslass_ex 30→75、takeruraiko_ex 14→49(僅かにN=50未達)。
**toxtricity(1→4)・gekkouga_ex(2→2)は未解決** — 唯一のデッキ所有チームの `team_id` が
`episodes_master.jsonl` 上で解決できず(相手側にしか登場せず取得済みリーダーボード範囲外)、
対象化できなかった。名前ベースのチーム逆引きが別途必要(未着手)。

**D-2 パイプライン再実行**: 615,067サンプルで再構築、fit_hybrid の安定バケットが1→3に改善
(evidence 0 / 7-10 / 11+ が安定)。採用された w は現行デプロイ値と完全一致
(baseline 維持ロジックが正しく働いた)。

**クラス別F1(全体評価)**: ogerpon_teal_ex 0.36→**0.899**、takeruraiko_ex 0.51→**0.722**、
oliva_ex 0.00→**0.894**。目標を大きく上回る改善。

**⚠️ 固定 validation の全体 log loss が悪化**: 0.0356 → **0.0438**(top-1 も 99.09%→98.50%)。
D-3 成功基準の「悪化させない」を満たしていない。原因の仮説(未検証):
`train.py` の train/valid 分割が `random.Random(seed).shuffle()` を**全エピソードリストに対して**
行っているため、データプールが増えると新規分だけでなく**既存エピソードの割り当ても全面的に
再シャッフル**される。「固定 validation」ウィンドウは約90エピソードしかなく、実質的な母集団の
入れ替えに近い比較になっている可能性がある(旧 split.json が保存されておらず確証は取れていない)。
bucket 11+(サンプルの大半)は誤答0件のままで log loss だけ微増(0.0003→0.0045、過信さを
抑えた側の変化)なのに対し、小さいバケット(1, 2-3, 4-6)は正解率自体が落ちている。

→ **この回で作られた LR/NB 重みは `sample_submission/` へデプロイしていない**(据え置き)。
split の決定性を直してから(episode_id ベースのハッシュ分割等)再評価し、
「モデル自体が悪化したのか、split の入れ替わりノイズか」を切り分けてから判断する。

### D: split 決定性修正後の再評価(2026-07-19、結論)

`train.py` の `split_episodes()` を、リスト全体シャッフルから **episode_id 単位のハッシュ分割**
(`sha256(f"{seed}:{episode_id}")` の先頭ビットで閾値判定)に修正。決定性を検証:
プール拡大前後で旧方式は既存エピソードの **31.6%が train/valid 反転**していたのに対し、
新方式は **0%(反転なし)**。仮説どおり shuffle バグは実在した。

しかし、これで再評価しても固定 validation の log loss は **0.0603**(shuffle バグ込みの
0.0438よりさらに悪化、元の 0.0356 とは乖離大)。原因を掘り下げたところ、**shuffle ノイズとは
別のもう一つの問題**が判明:

- 「固定 validation」(直近14日 × 相手上位200位)ウィンドウは現状わずか **93〜97 エピソード**
  しかない。
- そのうち evidence=0 バケットは 199 行(≒93エピソード×turn0〜数手)しかなく、
  **全体の重み付き log loss の約74%をこのバケット単体が占める**(1エピソードの序盤誤答が
  結果を大きく揺らす、サンプル数由来の高分散)。
- しかもこのウィンドウ(上位ランカー)には**少数アーキタイプがほぼ登場しない**
  (alakazam/marnie/crustle 等の主要メタが中心)。つまり「固定 validation」は少数クラス改善の
  効果を測れておらず、逆に主要メタでの僅かなブレをそのまま増幅して見せている可能性が高い。

一方、**全体評価(フィルタなし、960エピソード・126,340行)ではクラス別 F1 の改善が再確認**された:
ogerpon_teal_ex 0.36→**0.886**、takeruraiko_ex 0.51→**0.846**、oliva_ex 0.00→**0.961**
(いずれも split 修正後でも維持・上回っている)。fit_hybrid は baseline と同一の w を採用
(安定バケットは 0 / 7-10 / 11+ の3つ、shuffle バグ修正前の知見と一致)。

**結論・判断(慎重側)**: デプロイ基準(固定 validation log loss ≤ 0.0356 を維持)は**満たされていない**
ため、**今回の再学習重みはデプロイしない**。本番(`sample_submission/`)は
フェーズD開始前の重み(log loss 0.0356 / top-1 99.09%)のまま据え置く
(`ls -la` で更新日時 20:23/19:19 のまま touch されていないことを確認済み)。

split バグは実在し修正済みなので今後の学習には活きるが、「固定 validation」という評価プロトコル
自体が**現状のデータ量では小標本すぎて意思決定に使うには分散が大きすぎる**という新たな課題が
残った。少数クラスの改善という当初の目的(D-3成功基準の一つ)は達成しているので、
データ・コード資産(800件の追加リプレイ、決定的split、NB/hybrid実装一式)は無駄にはならない。

**次にやるなら(未着手、優先度低〜中)**:
- 固定 validation のウィンドウを広げる(例: 直近30日 × 上位500位)か、複数 seed で split を
  繰り返し平均を取り、分散を抑えた形で改めて before/after 比較する
- それでも改善しなければ、少数クラスは「全体評価」を判断基準に据え直すか、
  上位ランカー向けと汎用向けでモデルを分けることも検討

### スコープ外(今回やらない)

- LR のエピソード単位サンプル重み付け・クラスバランス学習(効果はあるはずだが、
  まずデータ増で解決するか見る。残る場合の次候補として記録)
- 温度キャリブレーションの再設計(ハイブリッドで役割が縮小したため)
