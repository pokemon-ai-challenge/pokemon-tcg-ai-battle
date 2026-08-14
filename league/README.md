# league

エージェント同士(`rule_based` / `ml_policy` など)を自動対戦させ、勝率と信頼区間を測るための
対戦リーグ基盤。`sample_submission/` の提出物とは無関係で、Kaggleへの提出内容には一切影響しない。

## 背景

`ptcg_ai/learning/`(Step1: バリューネットワーク、Step2: 模倣ポリシー)のオフライン評価は
「上位プレイヤーの選択とどれだけ一致するか」しか測れず、実際の対戦強度(勝率)は別に確認する
必要がある(`ml-agent-plan.md` の「足りない部品4: 評価基盤」)。本ディレクトリはその不足を
埋めるために 2026-07-21 に新設した。

## ディレクトリ構成

```
league/
├── README.md          このファイル
├── run_match.py        1試合を cg エンジンで直接駆動する中核ロジック(play_match)
├── run_league.py        CLI。play_match を N 回呼び、先手/後手バイアスを排除した
│                        勝率・Wilson信頼区間を算出する
├── tests/
│   └── test_run_match.py
└── results/            [Git管理外] run_league.py の出力(勝敗ログ入りJSON)
```

## 仕組み

`cg.game.battle_start` / `battle_select` / `battle_finish`(Kaggle シミュレーション基盤を経由
しない、ネイティブエンジンの低レベルAPI)を直接呼んで1試合を最初から最後まで進める。
`battle_review_viewer/live_match.py`(人間 vs CPU のライブ対戦セッション)と同じAPIを使うが、
あちらが1手ずつUIからの入力を待つのに対し、こちらは両陣営ともプログラムのエージェント関数
(`agent(obs) -> list[int]`)で自動的に手を進め、通して1試合を終わらせる。

**プロセス内シングルトンである点に注意。** `cg.sim.Battle.battle_ptr` はモジュールグローバルな
ネイティブ対戦ポインタなので、1プロセスでは常に1試合しか同時に進行できない。`run_league.py` は
逐次実行(並列化なし)。実測で1試合あたり0.2〜0.8秒程度のため、500試合でも10分弱で終わる規模感
であれば逐次で十分と判断した(数千試合規模が必要になったら、試合単位でプロセスを分ける並列化を
検討する)。

## 使い方

```bash
cd sample_submission   # deck.csv 等の相対パス依存があるため、cwd はここが前提
                        # (run_league.py 自体は起動後に自動で os.chdir する)

python ../league/run_league.py \
  --agent-a rule_based --agent-b ml_policy \
  --games 500

# デッキを変える場合(相手アーキタイプを変えた対戦などに再利用できる)
python ../league/run_league.py \
  --agent-a rule_based --agent-b ml_policy \
  --deck-a ../decks/some_deck.csv --deck-b ../decks/other_deck.csv \
  --games 500
```

主な引数:

| 引数 | 既定値 | 内容 |
|---|---|---|
| `--agent-a` / `--agent-b` | `rule_based` / `ml_policy` | `AGENT_REGISTRY`(`run_league.py` 内)に登録済みの名前 |
| `--games` | 500 | 総試合数(奇数は先手/後手が偏らないよう偶数に切り詰める) |
| `--deck-a` / `--deck-b` | `sample_submission/deck.csv` | 各エージェントのデッキCSV |
| `--seed-start` | 0 | 試合 i には `seed_start + i` を渡す |
| `--out` | `league/results/<a>_vs_<b>_<timestamp>.json` | 結果JSONの出力先 |

新しいエージェントを比較対象に追加する場合は、`run_league.py` の `AGENT_REGISTRY` に
`{"名前": "モジュールパス"}` を1行足すだけでよい(そのモジュールが `agent(obs) -> list[int]`
を公開していること)。

## 設計上の要点

### 先手/後手バイアスの排除

このゲームには先手・後手の概念があり、`battle_start(deck0, deck1)` の `deck0` 側が
「先攻側」として明示的に扱われる(`cg/game.py` の docstring)。単純に「エージェントAを
player0固定」で回すと、勝率差が「強さの差」なのか「先手/後手の差」なのか切り分けられない。
そのため `run_league.py` は試合の半分を `(agent0=A, agent1=B)`、残り半分を
`(agent0=B, agent1=A)` で実行し、勝敗は player_index ではなく**エージェント名**単位で集計する
(先手/後手別の内訳も別途出力する)。

### Wilson score interval

勝率の95%信頼区間には正規近似(Wald区間)ではなく Wilson score interval を使っている。
小標本・勝率が0や1に近いケースでも区間が `[0, 1]` の外にはみ出さず安定するため。

### 1試合の異常はリーグ全体を止めない

`play_match`(`run_match.py`)はエージェントの例外・不正な選択・異常な手数超過を
`MatchResult.error` として返し、決して例外を送出しない設計。`run_league.py` はエラーが出た
試合を勝率計算から除外しつつ件数を報告する(エラー率5%超で警告)。

## 何がGit管理下で、何がそうでないか

| 場所 | Git管理 | 理由 |
|---|---|---|
| `*.py` / `tests/` | ○ | コード |
| `results/*.json` | ✗ | `run_league.py` を再実行すればいつでも再生成できる生成物(ただし対戦結果は乱数シードに依存し、リプレイ厳密性は保証しない。恒久的な記録として残したい結果は個別にコミットするか、`step2-offline-evaluation.md` のようなレポートに数値として書き写すこと) |

## これまでの結果(記録)

- 2026-07-21: `ml_policy`(Step2 模倣ポリシー) vs `rule_based`、同一デッキ(フーディン)
  ミラー戦500試合 → **`ml_policy` 勝率86.4%**(`rule_based` 13.6%、95%CI [10.9%, 16.9%]、
  エラー0件)。詳細と解釈は
  [`../sample_submission/docs/plans/ml-value-network/step2-offline-evaluation.md`](../sample_submission/docs/plans/ml-value-network/step2-offline-evaluation.md)
  §8 を参照。
