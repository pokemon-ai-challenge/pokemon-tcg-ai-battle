# opponents/ — 対戦相手(sparring partner)エージェント

こちらの提出物(`sample_submission/`)を評価するための「相手役」を置く場所。
外部/ベースラインのルールベースエージェントを収め、`league/` から対戦相手として使う。

## 収録エージェント

| 名前(registry) | ファイル | デッキ | 出典 |
|---|---|---|---|
| `dragapult_rule` | `dragapult_rule_agent.py` | `dragapult_ex_deck.csv` | Kaggle "A Sample Rule-Based Agent Dragapult ex Deck"(著者: kiyotah) |

`dragapult_rule` は原典のロジックを改変せず取り込んだもの。対戦基盤に載せるための差分は
`dragapult_rule_agent.py` 冒頭 docstring の2点のみ(デッキ読み込み元と入力の dict/Observation 両対応)。

## 使い方(league)

`league/run_league.py` の `AGENT_REGISTRY` に登録済みなので、そのまま `--agent-a/--agent-b` で選べる。
**相手のデッキは `--deck-b`(または `--deck-a`)で必ず対応する CSV を渡すこと**
(エージェント内部の山札/サイド推定が実際に使うデッキ構成と一致している必要があるため)。

```bash
# 例: 我々の production(ml_policy)を、ドラパルトex ルールベース相手に計測
cd sample_submission
python ../league/run_league.py \
  --agent-a ml_policy --agent-b dragapult_rule \
  --deck-b opponents/dragapult_ex_deck.csv \
  --games 300 --workers 12
```

先手/後手バイアスは `run_league` が自動で排除する(半数ずつ入れ替え)。
`--deck-a` 未指定なら A は `sample_submission/deck.csv`(現行デッキ)を使う。

## 新しい相手を足すとき

1. `opponents/<name>_agent.py` に `agent(obs: Observation) -> list[int]` を実装(dict 入力も受けるなら Kaggle 基盤でも動く)。
2. デッキ CSV を同ディレクトリに置く。
3. `league/run_league.py` の `AGENT_REGISTRY` に1行、`PLAIN_AGENTS`(config/重みを取らない素の agent)に名前を追加。
