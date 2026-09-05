# Stage2 コスト対策 Step0: 実際のKaggle予算に対する正確な消費量の確定

- 日付: 2026-07-21
- ブランチ: `experiment/pimc-hidden-info-integration`
- 対象: [stage2-cost-control-implementation-plan.md](./stage2-cost-control-implementation-plan.md) Step0
- コード変更: なし(`pimc.py`/`lethal_simple.py`/`selector.py`等は無変更。測定はすべてスクラッチパッド上の外部スクリプトから`agent(obs)`呼び出しをラップして行った)

## (a) 実際のKaggle予算(出典付き)

`kaggle_replays/replays/episode-85169987-replay.json` の `specification.observation` フィールド
(実際の対戦データに埋め込まれた本物の競技スキーマ)を直接読み、以下を確認した:

```json
"actTimeout": {"default": 0, "description": "Maximum runtime (seconds) to obtain an action from an agent."},
"remainingOverageTime": {
  "default": 600,
  "description": "Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0.",
  "shared": false
}
```

同ファイルの実際の対局データ(`configuration`: `{"actTimeout": 0, "runTimeout": 2000, ...}`、`steps`配列)でも、各ステップの`observation.remainingOverageTime`が2エージェントそれぞれ独立に600から減っていく様子を確認済み(例: 115ステップの対局で agent0 は600→598.3、agent1 は600→596.5に減少)。

**予算は「1エージェントあたり合計600秒、対局を通じての累積消費」であり、`shared: false` のため両陣営は独立の残高を持つ。** これがStage2実装計画・コスト対策計画で前提としていた数値そのものであることを、一次情報(実際のリプレイJSON)で再確認した。

## (b) 1エージェントあたりの実測消費時間

自己対戦(両陣営とも`rule_pimc`、`hidden_state_source="estimated"`、`time_limit_ms=300`、`num_determinizations=4`)15試合を実行し、`agent(obs)`呼び出し自体を`obs.current.yourIndex`ごとにラップして計測した(`pimc.search()`内部だけでなく、`match_context.update`等を含む1エージェントの意思決定全体の時間)。両陣営が同一configのため、player0/player1それぞれの1試合あたり消費時間を合わせて「1エージェントの1試合あたり消費時間」の30サンプルとして扱った。

| 指標 | 値 |
|---|---|
| サンプル数(15試合 × 2陣営) | 30 |
| 平均消費時間/試合/エージェント | 13.3秒 |
| 最大消費時間/試合/エージェント | 24.3秒 |
| 最小消費時間/試合/エージェント | 0.8秒 |

## (c) 予算に対する使用率

| 指標 | 値 |
|---|---|
| 平均使用率(13.3秒 / 600秒) | **2.22%** |
| 最大使用率(24.3秒 / 600秒) | **4.04%** |

## 結論

**Stage2 Step4/5(`results/2026-07-21_stage2_pimc_timing.md` / `results/2026-07-21_stage2_pimc_ab.md`)の「Kaggleの持ち時間制約に対して危険域」という結論は不正確だった。** 原因は、「1試合平均21〜28秒」という数値が**自己対戦を1プロセスで駆動した際の対局全体の壁時計時間(両陣営の思考時間の合計)**であり、実際の予算(1エージェントあたり600秒、対局を通じての累積)と比較すべき「1エージェント分の消費時間」ではなかったこと。両者を混同していた。

実際に1エージェント視点で測り直すと、平均使用率2.22%・最大でも4.04%(30サンプル中)であり、計画のStep0完了条件に記載された「使用率が十分低ければ(例: 10%未満)、Step1-3は安全マージンを取るためのベストプラクティス程度の優先度に格下げしてよい」という基準に照らして**十分低い**。

**したがって、コスト対策計画のStep1-3(無駄な探索起動の削減・プライズゲート追加・パラメータ再調整)は、本番投入のブロッカーではなく、安全マージンを積み増すためのベストプラクティス程度の優先度に格下げしてよいと判断する。** ただし以下の留保点は残る:

- サンプル数(30)は小さく、`runTimeout`(対局全体2000秒、こちらも実測の対局所要時間21〜42秒なら余裕がある)や、対局数が多い大会形式でのばらつきまでは検証していない。
- 最大値(24.3秒/試合、4.04%)は1試合分であり、複数戦をこなす形式であれば累積使用率は上がる(ただし`remainingOverageTime`は「1エピソード=1対局」単位でリセットされる前提であれば問題にならない。この前提自体はStep0のスコープ外で未検証)。
- `pimc.get_stats()`(このスクリプトでも副次的に取得): searches=1833, found=616(33.6%), determinizations_run=3820, determination_timeouts=1225(32.1%)。Step4の傾向(半数近くが締切までに1件もdeterminizationを完走できないケースがある)と概ね整合。時間予算自体に余裕があるとわかった以上、この`found`率の低さは「危険」ではなく「探索の質を上げる余地がある」という改善機会として読み替えられる。

## 次のステップ

- 本Step0の結果により、コスト対策計画のStep1-3は「対応不要」ではなく「優先度を下げてよい」と判定。ユーザーと相談の上、Step1-3に進むか、他の優先課題(`match_context`のプレイヤー分離によるhead-to-head評価、policy prior統合等)に進むかを決める。
- [2026-07-21_stage2_pimc_timing.md](2026-07-21_stage2_pimc_timing.md) の「危険域」という記述は本ファイルの結果により訂正が必要(該当ファイルの冒頭に訂正注記を追記した)。
