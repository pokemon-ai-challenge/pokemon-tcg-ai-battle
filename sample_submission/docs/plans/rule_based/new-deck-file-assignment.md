# 新デッキ ptcg_ai/ 担当ファイル一覧

[new-deck-src-structure.md](new-deck-src-structure.md) のクラスタ分担（①〜⑨）を、実際に
`sample_submission/ptcg_ai/`・`sample_submission/decks/`・`sample_submission/tests/` に作成した
ファイルへ対応させたものです。「誰がどのファイルを触るか」
だけを素早く確認したいときに見てください。設計の背景・考え方は new-deck-src-structure.md を参照。

---

## 担当A（デッキ知識）が編集するファイル

`sample_submission/decks/new_deck/` 配下の8ファイルのみ。

| ファイル | 内容 |
|---|---|
| `deck_plan.py` | 主力/サブアタッカー、初手優先順位、進化・エネルギー優先順位、サーチ優先順位、守りたいカード、サイド枚数ごとの勝ち筋 |
| `pokemon_profiles.py` | ポケモンごとの役割（主力度、ベンチ価値、コンボ相手） |
| `attack_profiles.py` | ワザごとの追加効果分類（ベンチ狙撃、状態異常、ドロー、次ターン攻撃不可など） |
| `item_profiles.py` | グッズの効果分類 |
| `supporter_profiles.py` | サポートの効果分類 |
| `tool_profiles.py` | ポケモンのどうぐの効果分類 |
| `stadium_profiles.py` | スタジアムの効果分類 |
| `energy_profiles.py` | 使用する基本/特殊エネルギーの種類 |

加えて、③（`main_turn_parts/`）の実装がある程度固まったあとに着手:

| ファイル | 内容 |
|---|---|
| `sample_submission/tests/integration/new_deck/test_ability_priority.py` | 特性の使いどころの固定シナリオ |
| `sample_submission/tests/integration/new_deck/test_attack_priority.py` | 攻撃選択の固定シナリオ |
| `sample_submission/tests/integration/new_deck/test_deck_profiles.py` | deck_plan と各 profiles の整合性チェック |
| `sample_submission/tests/integration/new_deck/test_main_turn_integration.py` | メイン行動の統合シナリオ |
| `sample_submission/tests/integration/new_deck/test_retreat_priority.py` | 撤退判断の固定シナリオ |

やることは基本的に「`ptcg_ai/shared/profile_types.py` で決まった型（dataclass）に沿って、
`card_id -> Profile` の辞書を埋める」作業。**カードIDや効果の中身以外は触らない。**

---

## 担当B（汎用ロジック）が編集するファイル

`sample_submission/ptcg_ai/action_selection/`・`sample_submission/ptcg_ai/rule_based/`・
`sample_submission/ptcg_ai/board_evaluation/`・`sample_submission/ptcg_ai/shared/` の全部、および `agent.py`。

| 場所 | ファイル数 | 内容 |
|---|---|---|
| `ptcg_ai/core/agent.py` | 1 | 入口（デッキ返却 or router呼び出し） |
| `ptcg_ai/action_selection/router.py`, `fallback.py` | 2 | SelectContext振り分けと安全網 |
| `ptcg_ai/action_selection/handlers/` | 11 | SelectContextごとの入口ハンドラ |
| `ptcg_ai/rule_based/card_move/` | 6 | どのカード/対象を選ぶか |
| `ptcg_ai/board_evaluation/` | 4 | 盤面評価の計算部品（ダメージ、エネ充足、盤面スコア、交代先評価） |
| `ptcg_ai/rule_based/main_turn_parts/`（本体） | 4 | メイン行動の意思決定の骨格（buckets, proposals, weights, energy_eval） |
| `ptcg_ai/rule_based/main_turn_parts/priorities/` | 7 | カテゴリ別の行動提案（draw/board/ability/energy/retreat/attack/end_turn） |
| `ptcg_ai/shared/` | 3 | プロファイルの型定義・キャッシュ・引き方 |
| `decks/active.py` | 1 | 雛形のみ（中身の re-export 先は担当Aが差し替え） |
| `tests/conftest.py`, `helpers_decision.py` | 2 | テスト基盤 |
| `tests/unit/` | 3 | デッキ非依存の汎用テスト |

ここでは**カードIDやカード名を直接書かない**のが唯一のルール。カード固有の情報が要る場所は
必ず `ptcg_ai/shared/profile_registry.py` の `get_xxx_profile()` / `get_deck_plan()` を呼ぶ。

---

## 共同で触る場所

| ファイル | 内容 |
|---|---|
| `ptcg_ai/rule_based/main_turn_parts/weights.py` | カテゴリ基礎点。Bが初期値（仮に0）を仮置きし、自己対戦の結果を見ながら二人で調整 |
| `tests/integration/new_deck/`（5ファイル） | Aがシナリオを下書きし、Bと「今の③のスコア設計で本当にその動きになるか」をすり合わせる |

---

## 進める順番の目安

1. Bが `ptcg_ai/shared/profile_types.py` の型（フィールド構成）を確定・共有する
2. Aはその型に沿って `decks/new_deck/*` を埋め始める（新デッキ60枚確定後）
3. Bは並行して `ptcg_ai/action_selection/`・`ptcg_ai/rule_based/`・`ptcg_ai/board_evaluation/` 配下（①〜④）の中身を実装する
4. `main_turn_parts/proposals.py` が動くようになったら、Aが `tests/decks/new_deck/` のシナリオを
   下書きし、Bとすり合わせる
5. 両方揃ったら統合し、`weights.py` を共同でチューニングする
