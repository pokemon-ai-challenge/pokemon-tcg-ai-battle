
## 対応Issue

| Issue | タイトル | 状態 |
|---|---|---|
| #38 | ①選択振り分けの実装 | ✅ 完了（`action_selection/`） |
| #39 | ②盤面評価の実装 | ✅ 完了（`board_evaluation/`） |
| #40 | ③メイン行動の意思決定の実装（重みは共同） | ✅ 完了（`buckets.py`/`proposals.py`実装、`weights.py`初期値設定まで。値の継続調整はIssue本文の通り#34側） |
| #41 | ④カード移動・対象選択の実装 | ✅ 完了（`card_move/`） |
| #42 | ⑤カード知識アクセスの実装 | ✅ 完了（`card_cache.py`/`profile_registry.py`） |
| #43 | ⑨-B 検証（汎用基盤）の実装 | ⬜ 未着手（`tests/`は`pytest.skip("TODO")`のプレースホルダのまま。別Issue/PRで対応） |
| #34 | [共同] 重みチューニング | 🔶 一部着手（初期値設定+簡易な自己対戦確認のみ。継続的な自己対戦チューニングは未完了） |
| #33 | [fukuda] 汎用ロジックサイドの実装（親） | 🔶 子Issueの#43が未完了のため据え置き |
| #21 | フーディンデッキのルールベースAIの作成（全体親） | 🔶 担当A側・テスト・重み調整も継続中のため据え置き |

PR本文には `Closes #38`, `Closes #39`, `Closes #40`, `Closes #41`, `Closes #42` を記載してクローズし、#43・#34・#33・#21はオープンのまま残す想定。

## 概要

担当B（汎用ロジック）として `ptcg_ai/action_selection/`・`ptcg_ai/rule_based/`・`ptcg_ai/board_evaluation/`・`ptcg_ai/shared/` を実装し、担当A（Inadaさん）が用意したデッキ固有データ（`decks/new_deck/`、フーディン ハンドパワーデッキ）と接続、`main.py` から実際に呼び出されるところまで配線した。

## 実装

| ファイル/ディレクトリ | 内容 |
|---|---|
| `ptcg_ai/action_selection/router.py`, `fallback.py`, `handlers/`（11ファイル） | `SelectContext`ごとの振り分けと、未知のcontext・handler例外時の安全網 |
| `ptcg_ai/rule_based/main_turn_parts/` | MAIN選択の意思決定本体。7カテゴリ（draw/board/ability/energy/retreat/attack/end）ごとの提案 + `weights.py`によるカテゴリ間比較 |
| `ptcg_ai/rule_based/card_move/` | サーチ・捨て札・ベンチ配置などカード移動系の選択 |
| `ptcg_ai/board_evaluation/` | ダメージ計算（弱点/抵抗力反映）、エネルギー充足判定、盤面評価、交代先評価 |
| `ptcg_ai/shared/` | `card_cache`（cg.apiの生データキャッシュ）、`profile_registry`（担当Aのデータへの唯一の参照経路） |
| `decks/new_deck/deck_plan.py` 他 | 担当Aによるデッキ方針・カード別プロファイル（フーディン ハンドパワーデッキ） |
| `main.py`, `ptcg_ai/core/agent.py` | `main.agent()` を `ptcg_ai.core.agent.agent()` に配線。`deck.csv` を実際のデッキ内容に更新 |

## 主な工夫点

- **可変ダメージ技への対応**: `cg.api`の`Attack.damage`は「手札枚数×2」のような可変ダメージ技では0を返す（実際の計算はエンジン側で実行時に行われるため）。`Attack.text`（英語の効果テキスト）から既知の言い回し（"does N damage" / "N damage counters ... for each card in your hand"）を正規表現で拾うフォールバックを追加し、カード固有のIDに依存せず汎用的に対応。
- **担当Aのデータスキーマ変更への追従**: `decks/new_deck/deck_plan.py`が単純な`DeckPlan`データクラスから、理由付きメモを持つ独自のリッチなスキーマ（`AttackerPlan`/`PriorityEntry`など）に書き換えられたため、`profile_registry.get_deck_plan()`側でその変換を吸収し、他の判断ロジックには影響が出ないようにした。
- **`weights.py`の初期値設定**: 全カテゴリ0だと、スケールの大きいattack（ダメージ量そのまま）が常に他カテゴリを圧倒してしまうため、「同じターン中にattackと両立できる下準備」であるdraw/board/ability/energyに基礎点を持たせ、自然な行動順（下準備→攻撃）になるよう調整。

## 動作確認

- `python -m pytest tests/unit tests/integration` — 11件（デッキ非依存の汎用シナリオ、担当A/Bのデータが揃うまでの`skip`プレースホルダ含む）
- `python tests/local_sim/test_local_game.py` — エラーなし
- `python tests/local_sim/test_local_game_advanced.py --games 20 --opponent random` — エラーなし。`weights.py`調整前後で対ランダム勝率 11/20 → 16/20（参考値、サンプル数は少ない）

## 備考

- `weights.py`のカテゴリ間重みは自己対戦を見ながらの初期値。今後Aさんと一緒にチューニングしていく前提。
- `AttackProfile`（ベンチ狙撃・状態異常・ドロー・次ターン攻撃封じ）の判定は実装済みだが、個別カードごとの複雑なコンボ（例: 特定ポケモンへのエネルギー付与→特性トリガーでの回収など）はまだ未対応。
- `ptcg_ai/shared/card_cache.py`・`profile_registry.py`は担当Bの管轄だが、`decks/new_deck/deck_plan.py`の契約変更に合わせて`ptcg_ai/shared/profile_types.py`に`DeckPlan`型を追加している（元は担当Aのファイル側にあった型）。
