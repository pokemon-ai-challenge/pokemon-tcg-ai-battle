# kaggle_replays

Kaggleコンペ [pokemon-tcg-ai-battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) の
公開対戦リプレイを集めて、機械学習の学習データ(`(observation, action)` ペア)に変換するためのツール群。
`sample_submission/` の提出物とは無関係で、Kaggleへの提出内容には一切影響しない。

## ディレクトリ構成

```
kaggle_replays/
├── README.md                       このファイル
├── _common.py                      fetch スクリプトが共有するヘルパー
├── fetch_top_episodes.py           リーダーボード上位N チームのリプレイを取得(全量)
├── fetch_deep_decks.py             リーダーボード深い順位帯からチームあたり少数のリプレイを取得(デッキリスト収集用)
├── fetch_my_episodes.py            自分のチームのリプレイを取得
├── extract_training_data.py        replays/ + index/ を学習用JSONLに変換
│
├── replays/                        [Git管理外] リプレイ本体(episode-<id>-replay.json)
│                                    全runで共有するフラットなプール。episode_idで重複排除。
│
├── index/
│   ├── episodes_master.jsonl       [Git管理] エピソード単位の累積インデックス
│   └── leaderboard_history/
│       └── leaderboard-<run_id>.json   [Git管理] 実行ごとのリーダーボードのスナップショット
│
└── training_data/
    └── pairs.jsonl                 [Git管理外] 学習用の (observation, action) ペア(生成物)
```

## 何がGit管理下で、何がそうでないか

| 場所 | Git管理 | 理由 |
|---|---|---|
| `*.py` | ○ | コード |
| `index/episodes_master.jsonl` | ○ | 軽量(数千エピソードでも数MB程度)。かつ**再現不可能**——「そのエピソードを最初に見つけた時点で両チームが何位・何点だったか」はKaggleのリーダーボードが日々変動するため後から再取得できない |
| `index/leaderboard_history/*.json` | ○ | 同上。過去のリーダーボードの状態そのものが記録であり、再現不可能 |
| `replays/*.json` | ✗ | 1エピソードあたり約3〜4MB。件数が増えるとすぐに数百MB〜数GBになる。`fetch_*_episodes.py` を再実行すれば同じ episode_id は何度でも再取得できるので、Gitに含める価値が低い |
| `training_data/pairs.jsonl` | ✗ | `replays/` と `index/` から `extract_training_data.py` でいつでも再生成できる派生物 |

`.gitignore` はこの方針で設定済み。`replays/` と `training_data/` は空でも `.gitkeep` だけはコミットされる。

## 使い方

```bash
cd kaggle_replays

# 上位20チームのリプレイを取得(リーダーボードは日々変わるので、定期的に実行する想定)
python fetch_top_episodes.py --top 20

# 201〜2000位の深い順位帯から、チームあたり最新2件だけリプレイを取得
python fetch_deep_decks.py --rank-from 201 --rank-to 2000 --episodes-per-team 2

# 自分のチームの直近5提出のリプレイを取得
python fetch_my_episodes.py --submissions 5

# 集めたリプレイを学習用JSONLに変換(rank_at_fetch 等の重み付け情報つき)
python extract_training_data.py
```

いずれも既にダウンロード済みの `episode_id` はスキップするので、同じコマンドを何度実行しても安全(差分だけ取得・追記される)。

### 二層データ取得戦略(`fetch_top_episodes.py` と `fetch_deep_decks.py` の役割分担)

背景・設計原理の詳細は
[`sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md`](../sample_submission/docs/plans/opponent-deck-predictor/ml-predictor-phase2-scaling.md)
の「二層データ取得戦略」節を参照。

| スクリプト | 層 | 範囲 | チームあたり取得量 | 目的 |
|---|---|---|---|---|
| `fetch_top_episodes.py` | 上位層(狭く深く) | 〜200位 + 自チーム対戦分 | 全量 | 観測シーケンスの量・実戦で当たる相手の分布 |
| `fetch_deep_decks.py` | 深層(広く浅く) | 200位超(既定 201〜2000位) | 1〜2件 | デッキリスト収集(ラベル分布・事前分布・多様性把握) |

1チームはほぼ同じデッキを使い続けるため、深層側は対局数を稼ぐ必要がない。
そのかわりチーム数(=ユニークデッキ数)を広く稼ぐことを優先し、1チームあたり
1〜2エピソードだけ取得する。両スクリプトとも同じ `replays/` フラットプールと
`index/episodes_master.jsonl` を共有するので、取得後は区別なく1つの学習データセットとして扱える。

#### 実装メモ: 200位を超えるリーダーボードの取得方法

`kaggle competitions leaderboard <comp> -s` は `--page-size` の上限が200のため、
1回の呼び出しでは最大200位までしか取れない。実機検証の結果、以下の経路で
200位より深い順位も取得できることを確認した:

- レスポンスの先頭に(`--format json` を付けても)非JSON行
  `Next Page Token = <token>` が出力される。このトークンを次回呼び出しの
  `--page-token` に渡すとページが連続する(公式ドキュメントには載っていないが
  `kaggle competitions leaderboard --help` に `--page-token` オプションが存在し、
  実際にランクが連番で続くことを確認済み)。
- ページトークンは「直前のページの続き」しか指せず、任意の順位へ
  ジャンプすることはできない。そのため `fetch_deep_decks.py` は1位からページを
  送り、`--rank-from` 未満のページは中身を捨てて読み進める
  (1ページ200件なので `--rank-to 2000` でも高々10回のAPI呼び出しで済む)。
- `kaggle competitions team-submissions <teamId>` / `kaggle competitions episodes <submissionId>`
  は順位に関係なく同じ形式で使える(201位付近のチームでも動作確認済み)ため、
  `fetch_top_episodes.py` と同じ `_common.fetch_episodes()` をそのまま流用している。

`fetch_deep_decks.py` の主なオプション:

```bash
python fetch_deep_decks.py \
  --rank-from 201 --rank-to 2000 \
  --episodes-per-team 2 \
  --sleep 0.3
```

- `--rank-from` / `--rank-to`: 対象順位範囲(両端含む、1始まり)。既定 201〜2000。
- `--episodes-per-team`: 1チームあたり何件(最新のものから)取得するか。既定2。
- 高速化・冪等性: `episodes_master.jsonl` を見て、そのチームが既に
  `--episodes-per-team` 件以上プールにあればAPI呼び出し自体をスキップする。
  チームを処理するたびに即座にマスターインデックスへ追記するため、
  Ctrl+C で中断しても再実行すれば続きから進められる。

### エピソード件数に注意

1チームの1提出だけで **COMPLETED エピソードが1000件近くある**ことがある(実測値)。
`--top 20` で全チーム分を無制限に取得しようとすると、重複排除後でも数千件・数GB〜十数GBのダウンロードになり、
非常に時間がかかる。そのため `--max-episodes` の既定値を **300**(episode_idが大きい=直近のものを優先)にしてある。
もっと絞りたい/広げたい場合は明示的に指定する。

遅さの主因は `--sleep` ではなく **呼び出し回数そのもの**。実測では `kaggle` コマンドは1回の呼び出しにつき
CLI自体の起動だけで約1秒、リプレイダウンロード(数MB)は通信込みで1回2〜3秒かかる。`--sleep`(既定0.3秒)を
0にしても大きくは速くならないため、体感を変えたいなら `--max-episodes` を絞る方が効果的。

```bash
python fetch_top_episodes.py --top 20 --max-episodes 100   # もっと絞る
python fetch_top_episodes.py --top 20 --max-episodes 0     # 無制限(非推奨、ディスク容量に注意)
```

## 生データ(`replays/`)のバックアップについて

`replays/` はGit管理外かつローカルディスクにしかない。以下の点に注意:

- **再取得できる保証はない** — Kaggleが古いエピソードのリプレイをいつまで公開し続けるかは不明(競技終了後に取得できなくなる可能性がある)。長期的に学習に使いたいなら、このフォルダ自体を定期的にどこかへバックアップしておくのが安全。
- 現状の実測値は1エピソードあたり約3.7MB。`--top` の人数や取得頻度次第で、数ヶ月後には数GB〜数十GBに育つ可能性がある。

バックアップ先の選択肢(必要になったら相談してください):

1. **クラウドドライブに同期するだけ**(OneDrive/Google Driveの同期フォルダ配下に置く)— 追加ツール不要で最も手軽。バージョン管理はできないが「消えない」ことが目的ならこれで十分。
2. **`rclone` などで定期的にオブジェクトストレージ(S3/GCS等)へミラー** — 同期スクリプト1本で運用でき、容量が増えても安価。
3. **DVC (Data Version Control)** — Gitと連携して「このモデルはどの時点のデータセットで学習したか」まで再現したい場合に向く。`replays/` をDVC管理にすると、Git側には数百バイトのポインタファイルだけが残り、実データはリモート(S3/Google Drive等)に置ける。学習の再現性を厳密に求めるようになったら検討する価値がある。

まずは1で運用し始めて、必要になったら2・3を検討する、くらいで十分だと思われる。

## `episodes_master.jsonl` のスキーマ

```jsonc
{
  "episode_id": "86776342",
  "competition": "pokemon-tcg-ai-battle",
  "episode_create_time": "...",   // 対戦が実際に行われた日時(Kaggle側の記録)
  "episode_end_time": "...",
  "discovered_run_id": "20260718T233100Z",   // このエピソードを初めて記録したrunのID
  "discovered_at": "...",                     // 上記runの実行時刻(UTC)
  "players": [
    {
      "player_index": 0,
      "team_name": "...",
      "team_id": 12345678,           // 特定できなかった場合は null
      "rank_at_fetch": 3,             // discovered_at 時点の順位。null もあり得る
      "leaderboard_score_at_fetch": 1190.1
    },
    { "player_index": 1, "...": "..." }
  ]
}
```

同じ `episode_id` は最初に記録された内容を保持し続け、再実行しても上書きされない(追記のみ)。
