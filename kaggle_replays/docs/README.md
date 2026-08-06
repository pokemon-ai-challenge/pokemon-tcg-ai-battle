# kaggle_replays 作業ドキュメント 索引

上位リプレイを使った**模倣学習（BC）と強化学習（RL）**の作業記録。
チームの設計文書（`sample_submission/docs/plans/` 配下、100件超）とは別管轄で、
こちらは我々の実験の記録と設計検討を置く。

各文書の冒頭に **状態**（有効／過去の測定値／設計のみ／古い）を明記してある。
古い数値を現行と取り違えないよう、まずそこを見ること。

---

## 1. いま何が本当か（2026-08-03 時点）

現行の重みと、その素性:

| 種別 | ファイル | 素性 |
|---|---|---|
| **提出中** | **`policy_weights.json`** | **2026-07-21 の `policy_weights_configC.json` と md5 一致。全デッキ共通の1本、n_train 147,705、test top-1 0.5806。以降**一度も更新されていない** |
| BC（模倣） | `policy_weights_{marnie_grimmsnarl_ex,alakazam,crustle}_bc2.json` ほか | 2026-08-03 再学習。旧BC比 +8〜11pt。**提出には入っていない** |
| BC（旧） | `bc_backup_2026-08-02/` | 比較用に退避 |
| 凍結プール | `policy_weights_{marnie_grimmsnarl_ex,crustle}.json` ほか | **旧BC のまま**。過去の全数値と比較可能にするため意図的に固定 |
| RL | `policy_weights_*_pool_k60.json`、`league_runs/league1/` | **旧BC を起点に育てたもの**。新BC はまだ反映されていない |

**提出エージェントの経路**（2026-08-06 に確認）:

```
main.py → core/agent.py (AGENT_TYPE = "ml_policy") → ml_policy_agent
  config = abl_5_full（PTCG_AI_ML_CONFIG 未設定時の既定）
  重み    = policy_weights.json（abl_5_full に policy_weights_path が無いので PolicyModel の既定）

  リーサル探索(lethal_simple) ← 確定詰めがあれば BC を無視して上書き
    ↓ 無ければ
  BC でスコアリング → top1 が 0.9 以上なら即決
    ↓ そうでなければ
  top_k=4 に絞る → 8世界で決定化 → ロールアウト → 末端評価は "handcrafted"
```

**BC の役割は「候補を4つに絞る」ところまで**で、その中の選択は手書き評価関数が決めている。
BC を差し替えたときに勝率がどう動くかは、この構造を踏まえて解釈すること。
なお `*_bc2.json` は configC と同じアーキテクチャ・同じ `--weight-scheme concentrated` で、
アーキタイプ別に分けてデータ量を増やしたもの。**系譜としては configC の発展形**（別物ではない）。

決着済みの論点:

- **BC 再学習は有効。** 3アーキタイプすべてで有意差（→ [imitation/bc-retrain-2026-08-03.md](imitation/bc-retrain-2026-08-03.md)）
- **ただしデータ増はもう伸びない。** 倍加あたりの利得が +24.3 → +14.4 → +8.4 Elo と半減し続け、
  **無限に集めても上限 +12 Elo**。過去の +127.7 Elo は曲線の急な部分にいたためで再現しない
  （→ [imitation/data-scaling-2026-08-06.md](imitation/data-scaling-2026-08-06.md)）
- **モデル容量の拡大は無効。** hidden 32 / 64 / 128 で差なし。32 を維持（→ 同上 §5.5）
- **相互鍛錬ループ（世代を回す）は効果が出なかった。** 6世代8時間で +1.7pt、p=0.355
- **特徴量の小追加（にげるコスト・サイド価値）は効果が出なかった。** 勝率 +0.9pt、p=0.50。実装は revert 済み（→ [imitation/feature-retreat-prize-2026-08-04.md](imitation/feature-retreat-prize-2026-08-04.md)）
- **切り順（sequencing）特徴は Top-1 を +0.84pt 改善したが、勝率は動かなかった**（p=0.80）。
  → **Top-1 一致率を採否の指標に使わない。** 判断は勝率のみ。Top-1 は診断用
  （→ [imitation/sequencing-features-2026-08-05.md](imitation/sequencing-features-2026-08-05.md)）
- **skill concentration は構成C を採用済み**（→ [rl/eval-matrix-skillconc-2026-07-30.md](rl/eval-matrix-skillconc-2026-07-30.md)）

未決着で、実装だけ済んでいる論点:

- **outcome-aware weighting**（`train.py --outcome-weighting`）。旧データでは Top-1 が下がったが、
  この手法は敗北局面を軽く扱うので Top-1 が下がるのは想定内。**勝率で測らないと判定できない**
- **consequence 特徴**（`train.py --consequence-fields`）。旧データでは最良でも +0.2pt で CI 内
- **非公開ゾーンを入力に足す案**。設計のみ（→ [imitation/hidden-zone-features-design-2026-07-30.md](imitation/hidden-zone-features-design-2026-07-30.md)）

---

## 1.5 進捗報告の置き場所

**進捗報告はこのリポジトリではなく `C:\Users\rinnz\Documents\pokemon\進捗報告\` に置く。**
命名は `YYYY-MM-DD_タイトル.md`（日付は作業の開始日）。

- `2026-07-31_BC再学習と模倣学習の改善検証.md` — 07-31〜08-04 のまとめ。
  何が効いて何が効かなかったかを簡潔に。数値の詳細はこの docs/ 配下の各文書へ。

このディレクトリ（`kaggle_replays/docs/`）には、実験の詳細記録と設計検討だけを置く。

---

## 1.7 実験を始める前に読むもの

- [measurement-plan-2026-08-04.md](measurement-plan-2026-08-04.md) — **特徴量実験の測定計画**。
  対照群の作り方・必要シード数・必要試合数と、踏んだ罠のチェックリスト。
  試合レベルの統計（SPRT・δ_min）は `sample_submission/docs/plans/measurement-protocol/design.md` が権威。

---

## 2. 読む順番

初めて読むなら上から。

1. [imitation/bc-retrain-2026-08-03.md](imitation/bc-retrain-2026-08-03.md) — 現行 BC がどう作られたか。手順・数値・限界
2. [imitation/improvement-survey-2026-07-30.md](imitation/improvement-survey-2026-07-30.md) — 模倣学習の改善余地の調査。一部は実測で決着済み
3. [rl/eval-matrix-2026-07-30.md](rl/eval-matrix-2026-07-30.md) — モデル×相手の勝率行列。測り方の定義はここ
4. [imitation/hidden-zone-features-design-2026-07-30.md](imitation/hidden-zone-features-design-2026-07-30.md) — 入力を増やす案。未実装

---

## 3. 全文書

### imitation/ — 模倣学習（BC）

| 文書 | 状態 | 中身 |
|---|---|---|
| [bc-retrain-2026-08-03.md](imitation/bc-retrain-2026-08-03.md) | 有効 | BC 3アーキ再学習の記録。新旧比較、容量アブレーション再検証、未実施の理由 |
| [improvement-survey-2026-07-30.md](imitation/improvement-survey-2026-07-30.md) | 一部更新済み | 改善余地の調査。容量案は否定、BC再学習は有効と判明 |
| [hidden-zone-features-design-2026-07-30.md](imitation/hidden-zone-features-design-2026-07-30.md) | 設計のみ | 山札・サイド・トラッシュを方策の入力に入れる案 |
| [prize-lock-2026-08-03.md](imitation/prize-lock-2026-08-03.md) | 有効 | サイド落ち頻度の実測。非公開ゾーン案の前哨戦 |
| [feature-retreat-prize-2026-08-04.md](imitation/feature-retreat-prize-2026-08-04.md) | 有効 | にげるコスト・サイド価値の追加＝効果なし。不採用の記録 |
| [sequencing-features-2026-08-05.md](imitation/sequencing-features-2026-08-05.md) | 有効 | 切り順特徴＝Top-1 +0.84pt だが勝率は動かず。**Top-1 を採否に使わない**根拠 |
| [data-scaling-2026-08-06.md](imitation/data-scaling-2026-08-06.md) | 有効 | データ量スケーリング曲線（12本×1,600試合）。**リプレイ収集は尽きた**（上限 +12 Elo） |

### rl/ — 強化学習と評価

| 文書 | 状態 | 中身 |
|---|---|---|
| [eval-matrix-2026-07-30.md](rl/eval-matrix-2026-07-30.md) | 過去の測定値 | モデル×相手の勝率行列（400試合/セル） |
| [eval-matrix-skillconc-2026-07-30.md](rl/eval-matrix-skillconc-2026-07-30.md) | 過去の測定値 | skill-concentration 各構成の対戦比較 |
| [blunder-metrics-2026-07-30.md](rl/blunder-metrics-2026-07-30.md) | 過去の測定値 | 勝敗を使わない密な診断指標（KO逸失・過剰エネ・デッキ切れ） |
| [crustle-matchup-2026-08-06.md](rl/crustle-matchup-2026-08-06.md) | 有効 | 最悪マッチアップの構造分析。山札切れ負け17.6%の原因特定 |

### archive/ — 役目を終えたもの

| 文書 | 中身 |
|---|---|
| [handoff-2026-07-30.md](archive/handoff-2026-07-30.md) | 当時の引き継ぎ。記載の「次の一手」は消化済み |
| [resume-2026-07-30.md](archive/resume-2026-07-30.md) | 電源断からの復旧手順。該当実行は終了済み |

---

## 4. コードの README（移動していない）

コードの隣に置くべきものはそのままにしてある。

| 場所 | 中身 |
|---|---|
| `kaggle_replays/README.md` | リプレイ取得スクリプト全般 |
| `kaggle_replays/rl/README_pool.md` | `rl/` のどれがチーム資産でどれが我々の追加か |
| `kaggle_replays/rl/distributed/README.md` | 試合生成を外部マシンへ分散する基盤 |
| `kaggle_replays/rl/distributed/notebooks/README_train_pool.md` | Kaggle Notebook で学習を回す手順 |
| `kaggle_replays/value_net_probe/README.md` | 自己対戦で局面データを作るツール |
| `kaggle_replays/meta_analysis/README.md` | メタ分析・アーキタイプ別デッキ |

自動生成されるレポート（内容は実行のたびに変わる）:

- `kaggle_replays/policy_net/audit_report.md` — 特徴量ビルドの監査
- `kaggle_replays/deck_predictor/output/label_report.md` — デッキラベル付けの結果

---

## 5. この索引の更新について

新しく文書を足したら、**§3 の表と §1 の「いま何が本当か」を必ず更新する。**
古くなった文書は消さずに冒頭へ状態を書き、必要なら `archive/` へ移す。
数値を引用するときは、それがどの時点のモデルのものかを必ず併記すること
（BC 再学習の前後で数値が大きく動いており、取り違えが起きやすい）。
