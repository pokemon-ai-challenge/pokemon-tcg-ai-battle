# opponents/ — 対戦相手(sparring partner)エージェント

こちらの提出物(`sample_submission/`)を評価するための「相手役」を置く場所。
外部/ベースラインのルールベースエージェントを収め、`league/` から対戦相手として使う。

## 収録エージェント

| 名前(registry) | ファイル | デッキ | 出典 |
|---|---|---|---|
| `dragapult_rule` | `dragapult_rule_agent.py` | `dragapult_ex_deck.csv` | Kaggle "A Sample Rule-Based Agent Dragapult ex Deck"(著者: kiyotah) |
| `grimmsnarl_rule_01` | `grimmsnarl_rule_01.py`(本体: `grimmsnarl_core.py`) | `grimmsnarl_ex_deck_01.csv` | マリィのオーロンゲex 標準ユキメノコ型(Luca, LB 1242.3)。設計: [grimmsnarl-rule-implementation-design.md](../sample_submission/docs/plans/opponent-training/grimmsnarl-rule-implementation-design.md) |
| `grimmsnarl_rule_02` | `grimmsnarl_rule_02.py`(本体: `grimmsnarl_core.py`) | `grimmsnarl_ex_deck_02.csv` | マリィのオーロンゲex 非メノコ純ビート型(bono, LB 1150.8)。設計: 同上 |
| `grimmsnarl_rule_03` | `grimmsnarl_rule_03.py`(本体: `grimmsnarl_core.py`) | `grimmsnarl_ex_deck_03.csv` | マリィのオーロンゲex 標準+ハンディサーキュレーター妨害型(bono, LB 1150.8)。設計: 同上 |
| `grimmsnarl_rule_04` | `grimmsnarl_rule_04.py`(本体: `grimmsnarl_core.py`) | `grimmsnarl_ex_deck_04.csv` | マリィのオーロンゲex 標準+クセロシキ妨害型(jiatu.l, LB 1116.5)。設計: 同上 |
| `grimmsnarl_rule_05` | `grimmsnarl_rule_05.py`(本体: `grimmsnarl_core.py`) | `grimmsnarl_ex_deck_05.csv` | マリィのオーロンゲex テンポ型(monnosuke, LB 1102.6)。設計: 同上 |
| `lucario_rule_01` ... `lucario_rule_05` | `lucario_rule_01.py` ... `_05.py` (shared `lucario_core.py`) | `lucario_ex_deck_01.csv` ... `_05.csv` | Mega Lucario ex replay-derived sparring partners; implementation notes: [lucario-rule-implementation-design.md](../sample_submission/docs/plans/opponent-training/lucario-rule-implementation-design.md) |
| `lucario_rule_official` | `lucario_rule_official.py` (shared `lucario_core.py`) | `lucario_ex_deck_official.csv` | Mega Lucario ex official 60-card deck; uses per-attack KO scoring and official-deck card profile. |
`dragapult_rule` は原典のロジックを改変せず取り込んだもの。対戦基盤に載せるための差分は
`dragapult_rule_agent.py` 冒頭 docstring の2点のみ(デッキ読み込み元と入力の dict/Observation 両対応)。

`grimmsnarl_rule_01`〜`_05` は新規実装(python-engineer 作成、rule-engine-expert の実装契約
[grimmsnarl-rule-implementation-design.md](../sample_submission/docs/plans/opponent-training/grimmsnarl-rule-implementation-design.md)
に準拠)。5つは同一ロジック(`grimmsnarl_core.py`)を、上位デッキ5種
(`kaggle_replays/meta_analysis/archetype_decks/marnie_grimmsnarl_ex/{01..05}.csv` からコピー)
それぞれの `build_profile()` 結果で動かす薄いラッパ。dragapult 同様、`opponents/` の
対戦相手でありこちらの提出物・production の rule_based とは無関係。

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
