# ORCHESTRATOR — ポケモンTCG AI 開発マスター指示書

このファイルは **Claude Code への実行指示書**です。  
新しいセッションの開始時に、まずこのファイルを読み込んでください。

> **⚡ セッション開始時は `docs/STATUS.md` を最初に読む。**  
> 「今すぐやること」「現在の勝率」「バックログ」が一覧できる。  
> 人間から「続きをやって」と言われたら STATUS.md の「今すぐやること」を実行する。  
> 人間から新タスクを指示されたら、実装後に STATUS.md を更新して次の「今すぐやること」を記録する。

---

## このファイルの使い方

1. **`docs/STATUS.md` を最初に読む**（現在状態・次のタスク）
2. `## 現在の作業状態` でチェックが入っていない最初のタスクを探す
3. そのタスクの「担当エージェント」を確認し、対応する `agents/` ファイルを読む
4. 指定モデルに切り替えて作業を実行する
5. 完了したらこのファイルのチェックボックスに `x` を入れて保存する
6. 次のタスクに進む

---

## モデル選択ガイド（コスト最適化）

| タスク種別 | 推奨モデル | モデルID | 理由 |
|-----------|----------|---------|------|
| 戦術設計・アーキテクチャ決定 | Sonnet | `claude-sonnet-4-6` | 複雑な推論が必要 |
| 実装コーディング（中規模） | Sonnet | `claude-sonnet-4-6` | バランス型 |
| 単純なボイラープレート生成 | Haiku | `claude-haiku-4-5-20251001` | 安価・高速 |
| コードレビュー・合理性チェック | Haiku | `claude-haiku-4-5-20251001` | 安価で十分 |
| カードタグ付け（大量処理） | Haiku | `claude-haiku-4-5-20251001` | 単純分類は安価で十分 |
| 対戦ログのデバッグ分析 | Sonnet | `claude-sonnet-4-6` | 因果推論が必要 |
| 評価関数の重み設計 | Sonnet | `claude-sonnet-4-6` | ゲーム理論的推論が必要 |
| ベンチマーク結果の集計 | Haiku | `claude-haiku-4-5-20251001` | データ集計は単純処理 |
| project.json → meta_decks.py 変換 | Haiku | `claude-haiku-4-5-20251001` | データ変換は単純処理 |
| アーキタイプ別対策方針の策定 | Sonnet | `claude-sonnet-4-6` | 戦術的推論が必要 |

> **Opus は使わない**。Sonnet で対応できないほど複雑なタスクはこの段階では発生しない。

---

## エージェント一覧

| エージェント | ファイル | 役割 | 使うタイミング |
|------------|---------|------|-------------|
| 戦術評価AI | `docs/agents/tactics-agent.md` | デッキ戦略・選択方針の適合性評価 | 実装前の方針決定 |
| コーディングAI | `docs/agents/coding-agent.md` | Pythonコードの実装 | 全フェーズ |
| 合理性チェックAI | `docs/agents/review-agent.md` | 実装の正しさ・一貫性の検証 | 実装後の確認 |
| カードタグ付けAI | `docs/agents/card-tagger-agent.md` | カードへの意味タグ付与・card_tags.py 生成 | タスク 0-2 の前に実施 |
| デバッグAI | `docs/agents/debug-agent.md` | 対戦ログから悪手の原因を特定 | 挙動がおかしいとき |
| 評価関数設計AI | `docs/agents/evaluator-design-agent.md` | 盤面スコアリングの重み設計 | Level 3 着手前 |
| ベンチマーク分析AI | `docs/agents/benchmark-agent.md` | 対戦成績の集計・改善効果の定量評価 | 各Level完了後 |
| メタデッキ管理AI | `docs/agents/meta-deck-agent.md` | Tierデッキの読み込み・相手デッキ推定テーブル生成 | Level 2着手前・更新時 |
| 提出後改善AI | `docs/agents/improvement-agent.md` | Kaggleログ分析 → 優先度判断 → 改善サイクル管理 | 提出後・各改善ループ |
| **Kaggle改善サイクル** | **`docs/kaggle-improvement-cycle.md`** | **Kaggleログが渡されたときに自動実行するステップ定義** | **ログが提供されたとき（最優先で読む）** |

---

## 常時参照ファイル（毎セッション読む）

- `cg/api.py` — ゲームエンジンAPI（SelectContext/OptionType の全値確認）
- `docs/strategy-knowledge.md` — 戦略判断の記録（実装前に必ず読む）
- `docs/ptcg-ai-architecture.md` — 汎用プレイングロジック方針（ARC-* タスクに着手する前に必ず読む）

## タスク別参照ファイル（該当タスクのときだけ読む）

| タスク | 読むファイル | 読む箇所 |
|-------|-----------|---------|
| 0-1, 0-4 | `main_minimal_snapshot.py` | 現在のベースコード |
| 1-1 | `docs/ai-development-roadmap.md` | Level 2 の DeckPlan フィールド定義・デッキ状態機械 |
| 1-2, 1-3 | `docs/ai-development-roadmap.md` | Level 1「まだやらないこと」・初心者AIの基本方針 |
| 2-1 | `data/JP_Card_Data.csv` + `data/EN_Card_Data.csv` | カード名解決 |
| 3-1 | `docs/ai-development-roadmap.md` | Level 3 評価関数の式・評価要素一覧 |
| 4-1, 4-2 | `docs/ai-development-roadmap.md` | Level 4 探索パラメータ・不確実情報の扱い |
| メタデッキ系 | `cardlist_referenced/tier_deck_data/name_resolution_cache.json` | 既存の名前解決キャッシュ |

> `docs/ai-development-roadmap.md` は全体を毎回読む必要はない。上記の該当箇所だけ読めば十分。

## その他参照先（必要な場合のみ）

- `data/JP_Card_Data.csv` — 日本語カードデータ（カード名解決の第一参照先）
- `data/EN_Card_Data.csv` — 英語カードデータ（IDから英語名・HP・技名を引く際に使用）
- `cardlist_referenced/tier_deck_data/resolved/*.json` — デッキ単位の解決済みレシピキャッシュ

---

## 現在の作業状態

### フェーズ 0：基盤構築（着手前にここから始める）

#### タスク 0-1：コード構造のリファクタリング
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `main.py` の `choose_main_action` を「分類」と「優先順位判断」に分離する
- [x] OptionType / SelectContext を数値ではなく Enum で扱うよう修正する
- [x] `SelectContext` ごとにルーター関数 `choose_action` を作る

#### タスク 0-2：カード情報辞書の作成
**担当:** コーディングAI　**モデル:** Haiku

- [x] `src/knowledge/card_database.py` を作成する
- [x] `all_card_data()` から `card_id -> CardData` 辞書を構築する
- [x] `all_attack()` から `attack_id -> Attack` 辞書を構築する

#### タスク 0-3：DecisionTrace ログの実装
**担当:** コーディングAI　**モデル:** Haiku

- [x] `src/knowledge/` に `decision_trace.py` を作成する
- [x] ターン・コンテキスト・候補・選択・理由を記録する関数を作る
- [x] `main.py` の主要選択箇所でログを出力する

#### タスク 0-4：コンテキスト別選択の実装
**担当:** 戦術評価AI（方針決定）→ コーディングAI（実装）　**モデル:** Sonnet

- [x] 戦術評価AIで各コンテキストの選択基準を決定する
- [x] `SETUP_ACTIVE_POKEMON` の選択ロジックを実装する
- [x] `SETUP_BENCH_POKEMON` の選択ロジックを実装する
- [x] `ATTACH_FROM / ATTACH_TO` の選択ロジックを実装する
- [x] `EVOLVES_FROM / EVOLVES_TO` の選択ロジックを実装する
- [x] `TO_BENCH / TO_HAND / DISCARD / SWITCH / ATTACK` の選択ロジックを実装する

#### タスク 0-5：固定シナリオテストの作成
**担当:** 合理性チェックAI（ケース定義）→ コーディングAI（実装）　**モデル:** Haiku

- [x] `src/tests/scenarios/` ディレクトリを作成する
- [x] 初手バトルポケモン選択シナリオのテストを作る
- [x] 進化選択シナリオのテストを作る
- [x] 即倒しできる攻撃選択シナリオのテストを作る

#### タスク 0-6：Level 0 完了確認
**担当:** 合理性チェックAI　**モデル:** Haiku

- [x] ランダム先頭候補選択が主要な選択処理に残っていないことを確認する
- [x] 固定シナリオで意図したカードを選べることを確認する
- [x] 不正選択・例外が発生しないことをテストで確認する
- [x] 選択理由がログから確認できることを確認する

---

### フェーズ 1：初心者AI（Level 0 完了後に着手）

#### タスク 1-1：DeckPlan 設計
**担当:** 戦術評価AI　**モデル:** Sonnet

- [x] 戦術評価AIで現在の `deck.csv` を分析しデッキコンセプトを整理する
- [x] `src/knowledge/deck_plan.py` の `DeckPlan` クラスを定義する
- [x] 現在のデッキの `DeckPlan` インスタンスを作成する

#### タスク 1-2：メインアクション強化
**担当:** コーディングAI　**モデル:** Sonnet

- [x] メインアクションの優先順位を DeckPlan に基づいて実装する
- [x] エネルギー付け先の判断ロジックを実装する
- [x] 進化先の選択ロジックを実装する

#### タスク 1-3：サポート・サーチ判断
**担当:** コーディングAI　**モデル:** Sonnet

- [x] サーチカードの対象選択ロジックを実装する
- [x] 捨て札の選択ロジックを実装する
- [x] サポートカードの使用判断ロジックを実装する

#### タスク 1-4：Level 1 完了確認
**担当:** 合理性チェックAI　**モデル:** Haiku

- [x] 初手バトルポケモン選択が妥当かをログで確認する
- [x] 明らかにおかしい行動（重要カードを無意味に捨てるなど）が消えたことを確認する
- [x] 固定シナリオテストの正答率を記録する

---

### フェーズ 2：メタデッキ統合とDeckPlan強化（Level 2、フェーズ 1 完了後）

#### タスク 2-1：Tierデッキデータの取り込み
**担当:** メタデッキ管理AI　**モデル:** Haiku（カード解決に曖昧さがある場合は Sonnet）

> **前提作業（pdf_card_editor への登録）：**  
> `cardlist_referenced/pdf_card_editor` の UI でデッキを登録するか、または  
> **メタデッキ管理AI が `data/EN_Card_Data.csv` と既知の Tier 情報をもとに直接 `meta_decks.py` を生成することもできる。**  
> pdf_card_editor への登録がない場合は、AI が「環境 Tier-S デッキの一般的な構成」を `EN_Card_Data.csv` で検索して構築する。  
> ラベル `Tier-S` / `Tier-A` / `arch:<アーキタイプ名>` は AI がカード名から自動付与する。

- [x] `pdf_card_editor/projects/*/project.json` があれば読み込む（なければ後続ステップで直接構築）
- [x] Tier-S 候補デッキの構成を `EN_Card_Data.csv` で確認し、同名異技カードの解決を行う
- [x] 解決できなかったカードは `# AMBIGUOUS` コメントを付けて記録する
- [x] `src/knowledge/meta_decks.py` を生成する（DECK_CATALOG / DECK_RECIPES / DECK_TIERS / CARD_TO_DECKS）
- [x] `estimate_deck_candidates()` 関数を実装する
- [x] Tier-S・Tier-A デッキのレシピが60枚揃っていることを確認する（`AMBIGUOUS` は人間確認を促すコメントで残す）

#### タスク 2-2：Tier上位デッキとのベンチマーク対戦
**担当:** ベンチマーク分析AI　**モデル:** Haiku

- [x] `meta_decks.tier_s_decks()` から対戦相手デッキを取得する仕組みを作る
- [x] 各Tierデッキ相手に 30 試合以上の自己対戦を実施する
- [x] 勝率・敗因・パターンをベンチマークレポートとして出力する

#### タスク 2-3：アーキタイプ別対策方針の策定
**担当:** 戦術評価AI　**モデル:** Sonnet

- [x] Tier-S 各アーキタイプの脅威（攻撃パターン・妨害手段）を整理する
- [x] アーキタイプ別に「警戒すべき局面」「対策できる選択肢」を言語化する
- [x] DeckPlan の `recovery_rules` / `win_conditions` に対アーキタイプ判断を追加する

---

### フェーズ 3：盤面評価の導入（Level 3、フェーズ 2 完了後）

#### タスク 3-1：評価関数の設計
**担当:** 評価関数設計AI　**モデル:** Sonnet

- [x] 現在のデッキと Tier-S 相手の対戦パターンをもとに評価要素と初期重みを設計する
- [x] `src/decision/evaluator.py` の設計仕様をドキュメントとして出力する

#### タスク 3-2：評価関数の実装
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `src/decision/evaluator.py` の `BoardEvaluator` クラスを実装する
- [x] サポート・サーチ・妨害の「使うタイミング」判断に評価値を組み込む

#### タスク 3-3：重みのキャリブレーション（繰り返しタスク）
**担当:** ベンチマーク分析AI → 評価関数設計AI　**モデル:** Haiku / Sonnet

- [x] Tier-S 相手に 50 試合実施し結果を記録する
- [x] 主要敗因を分析し重み調整案を出す
- [!] 重みを更新して再測定する（勝率差 5% 以上の改善が確認できるまで繰り返す）
  > **ゲート未達成（2026-06-26）**: 評価器+探索(73%) がルールのみ(72-77%) を一貫して下回る。
  > depth=3/5/8・random/rule-based rollout・bench_count 調整など複数試行。改善なし。
  > **対策**: main.py を `choose_action`（ルールのみ）に戻す。Phase 5 で相手デッキ推定精度が上がれば再有効化を検討。

---

### フェーズ 4：複数手先探索（Level 4、フェーズ 3 完了ゲート通過後）

> **⚠ フェーズ 3 完了ゲートが通過できていない場合、このフェーズには絶対に入らない。**

#### タスク 4-1：相手情報推定戦略の文書化
**担当:** 戦術評価AI　**モデル:** Sonnet

- [x] 公開情報から推測できる相手のデッキ・手札を文書化する（`search_begin` に渡す値の設計）
- [x] 推測が外れた場合の影響を評価し、保守的な仮定を定める

#### タスク 4-2：探索APIの統合
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `src/decision/search_policy.py` を作成する
- [x] 評価関数で候補を上位 3〜5 に絞ってから `search_begin`/`search_step` で探索する
- [x] 探索深さ・幅の初期値（幅 5、深さ 4〜8）を設定する

---

### フェーズ 5：自己改善（Level 5、フェーズ 4 完了後）

#### タスク 5-1：相手デッキ推定の組み込み
**担当:** コーディングAI　**モデル:** Sonnet

- [x] バトル中の公開情報（相手の使用カード・バトル場）から `estimate_deck_candidates()` を実行する
- [x] 推定結果をボスの指令ターゲット選択に反映する（`src/knowledge/opponent_model.py` 新規作成）
  > ARCH_BOSS_TARGETS: アーキタイプ別の優先ターゲット定義（hydrapple→Meganium等）
  > choose_switch: playerIndex で相手ターゲットを検出しボスターゲット優先順を使用
  > 効果: hydrapple_ex 57%→87%（Meganiumターゲット戦略が機能）
  > 全10デッキ平均 72% は変化なし（分散±9%のため300試合では誤差範囲内）

#### タスク 5-2：デッキ評価・最適化
**担当:** 戦術評価AI → ベンチマーク分析AI　**モデル:** Sonnet / Haiku

- [x] 現行デッキとの代替デッキ（ドラパルト等）を Tier 観点で比較する
- [x] デッキ変更の場合は DeckPlan を追加して `deck.csv` を入れ替え、ベンチマーク比較する
- [x] 変更後の勝率が改善していれば次の提出でそのデッキを採用する
  > **結論: Lucario 継続（72%）。Dragapult ex は 51% と大幅劣位。**
  > 理由: Dragapult は Rare Candy スキップ進化・Blaziken エネ循環・ファントムダイブ散布など
  > 固有ロジックが必要で、汎用ルールベースでは性能が出ない。
  > DRAGAPULT_PLAN は deck_plan.py に保持（将来の専用ロジック追加時に使用可能）。
  > deck_lucario.csv / deck_dragapult.csv は参照用に保持。

#### タスク 5-3：アーキタイプ別戦略オーバーライド（MatchupPolicy）
**担当:** コーディングAI　**モデル:** Sonnet  
**前提:** フェーズ 5-1 の `opponent_model.py` が動作していること（完了済み）

- [x] `src/knowledge/matchup_policy.py` を新規作成する
  - `MatchupOverride` データクラス（`max_bench_size: int`, `boss_rush: bool` など）
  - `MATCHUP_OVERRIDES: dict[str, MatchupOverride]` を定義
  - `get_matchup_override(arch: str | None) -> MatchupOverride` 関数
- [x] `target_policy.py:choose_to_bench()` を修正
  - `get_matchup_override()` で `max_bench_size` を取得
  - ベンチ現在枚数 ≥ max_bench_size のとき `min_count` 分のみ選択（追加しない）
- [x] `main_policy.py:_pick_best_play()` を修正
  - ベンチ展開ポケモンのスコアを max_bench_size 超過時に 0 に下げる
- [!] vs dragapult_ex 30試合ベンチマーク（基準 82%、目標 85%+）を実施
  > **ゲート未達成 (2026-06-26)**: bench=2で81%(90試合)、bench=3で73%(30試合)、制限なしで80%(30試合)。
  > 統計的に差なし（30試合分散 ±7%）。bench=2 が最良設定として採用。85%目標は未達。
  > 詳細設計: `docs/axis_card_analysis.md` の「崩し方」表 / `docs/STATUS.md` B-02

#### タスク 5-4：全アーキタイプ勝率の再計測（マリィ版）
**担当:** ベンチマーク分析AI　**モデル:** Haiku

- [x] 現デッキ（マリィのオーロンゲex）で全10アーキタイプ × 30試合を実施
  > 新規計測 7 arch: raging_bolt_ex 100%・mega_lucario_ex 63%・olivia_ex 93%・ogerpon_bullet 83%・maries_obstagoon_ex 73%・alakazam 90%・crustle 73%
  > 前回計測 3 arch: dragapult_ex 81%・hydrapple_ex 74%・terasta_bullet 84%
  > 全体平均 81%。最弱: mega_lucario_ex 63%（格闘→悪弱点）、最強: raging_bolt_ex 100%
- [x] `docs/STATUS.md` の「勝率ベースライン」を更新する
- [x] `docs/strategy-knowledge.md [META-001]` の勝率表を更新する
- [x] Lucario デッキで全 10arch × 30 試合を再計測（MatchupPolicy 実装後）
  > 2026-06-26 実施。hydrapple_ex 57%→80%(+23% boss_rush効果)、dragapult_ex 77%→60%(-17% bench=2阻害疑い)
  > maries_obstagoon_ex 73%→33% 異常値（Z≈-4.9）→要再計測。全体 72%→66%（異常値含む）

#### タスク 5-6：Lucario maries_obstagoon_ex 再計測 + bench=2 デッキ依存調査
**担当:** ベンチマーク分析AI　**モデル:** Haiku

- [x] Lucario で vs maries_obstagoon_ex を 30 試合再計測（33% 異常値の確認）
  > 67% — 33% は外れ値。正常値に戻った。
- [x] Lucario で vs dragapult_ex を 30 試合再計測（60% の確認）
  > bench=2 なし: 73%（旧 60%）。bench=2 が Lunatone+Solrock エネサイクルを阻害していた。
- [x] bench=2 が Lucario のエネサイクルを阻害している場合は、`matchup_policy.py` をデッキ識別対応に改修
  > `MatchupOverride(only_for_decks=frozenset({"maries_obstagoon_ex"}))` でマリィ専用化。
  > dragapult_ex / mega_lucario_ex の bench=2 はマリィにのみ適用される。

#### タスク 5-5b：バッチベンチマーク実装
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `src/tests/run_bench_suite.py` 新規作成（複数デッキ一括計測 + 比較テーブル出力）
- [x] `tier_s_benchmark.py` に `--my-deck` 引数追加（deck.csv 書き換え不要）
- [x] `deck_plan.py` に `set_active_plan()` 追加
- [x] `docs/ORCHESTRATOR.md` にバッチ実行コマンド集を追加

#### タスク 5-6c：vs crustle EX 免疫対策（ex_immune.py 接続）
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `src/knowledge/ex_immune.py` に `has_non_ex_on_bench()` を追加
- [x] `target_policy.py:choose_attach_from()` — EX 免疫時は `non_ex_energy_priority` を使用
- [x] `target_policy.py:choose_switch()` — 自分交代時に EX 免疫なら `non_ex_active_priority` を使用
- [x] `main_policy.py:_pick_best_attach()` — EX 免疫時は `non_ex_energy_priority` を使用
- [x] `main_policy.py:choose_main_action()` — EX アクティブ + ベンチ非 EX あり → 強制リトリート
- [ ] ベンチマーク: 両デッキで vs crustle 30試合（boss_rush + ex_immune 統合）

---

### フェーズ 6：汎用プレイングロジックの実装（フェーズ 5 完了後）

> 設計方針の詳細は `docs/ptcg-ai-architecture.md` を参照。各タスク着手前に必ず読む。

#### タスク ARC-4：要求値スコアの evaluator 実装
**担当:** 評価関数設計AI → コーディングAI　**モデル:** Sonnet

- [x] `docs/ptcg-ai-architecture.md` の「2-1. 要求値（Opponent's Requirement）」を読む
- [x] `src/decision/evaluator.py` に `opponent_requirement_score()` メソッドを追加する
- [x] `score()` に `opponent_requirement` ウェイトで統合済み
- [x] マリィ全10arch: **86%**（旧81%）+5%改善確認

#### タスク ARC-5：妨害札（手札干渉）の撃ちどき判断
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `docs/ptcg-ai-architecture.md` の「3-3. 妨害札の撃ちどき」を読む
- [x] `src/knowledge/opponent_model.py` に `estimate_hand_quality(obs) -> float` を追加
  - `obs.logs` から LogType.PLAY で相手カードプレイ数を推定
  - `opp.handCount` と組み合わせてスコア化
- [x] `src/decision/main_policy.py:_pick_best_play()` でイオナ(IONO_ID=1198)のスコアを動的調整
  - hand_quality < 0 → score=3（使わない）/ >= 2 → score=35（最優先）
- [x] マリィ全10arch: **86%** maries_obstagoon_ex ミラーで +17%（73%→90%）が顕著

#### タスク ARC-7：逃げエネ0前出し戦略（Lazy Evaluation）
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `docs/ptcg-ai-architecture.md` の「4-1. 逃げエネ0ポケモンの前出し戦略」を読む
- [x] `src/decision/target_policy.py:choose_switch()` に TO_ACTIVE 判定 + 逃げエネ0優先ロジックを追加
  - `_any_bench_attacker_ready()` で攻撃準備確認
  - アタッカー未準備 + free_retreat pokemon → score=5000 で最優先
- [x] `src/decision/main_policy.py:choose_main_action()` に `_active_is_buffer()` + buffer_retreat_lazy_eval トリガーを追加
- [x] マリィ全10arch: **86%** で改善確認

#### タスク ARC-3：残りリソース確率推定
**担当:** コーディングAI　**モデル:** Sonnet

- [x] `docs/ptcg-ai-architecture.md` の「2-2. リソース残量の確率推定」を読む
- [x] `src/knowledge/opponent_model.py` に `estimate_remaining(card_id: int) -> float` を追加
  - `DECK_RECIPES[arch]` から recipe_count を取得し `_revealed` の出現数を引く
- [!] `choose_main_action` へのリスク判断接続は未実装（ARC-5/7の改善で十分な効果が出たため後回し）

---

#### タスク 5-7：Kaggle ログ駆動の継続改善サイクル
**担当:** 提出後改善AI　**モデル:** Sonnet  
**実行手順:** `docs/kaggle-improvement-cycle.md` を読んでステップ 1 から順番に実行する

- [x] 改善サイクルのプロセスを `docs/kaggle-improvement-cycle.md` に定義した
- [ ] Kaggle 試合ログを読み込み、主要敗因をステップ 1〜2 の手順で分類する
- [ ] 優先度ルール（ステップ 3）に従い改善内容を 1 つ決定・実装する
- [ ] ローカルベンチマーク（ステップ 5）で確認し、採否を判定する
- [ ] STATUS.md / ORCHESTRATOR.md を更新して次サイクルへ（ステップ 6〜7）

---

---

## 既知の技術的注意点（AI が実装前に必ず読む）

### SelectContext は 48 値ある（ロードマップ記載の 12 値より多い）

`cg/api.py` の実際の列挙は 0〜47。以下は通常のインデックス選択と扱いが異なる：

| コンテキスト | 型 | 実装上の注意 |
|------------|-----|-----------|
| `IS_FIRST (41)` | YES/NO | DeckPlan.prefer_go_first に従う。未設定なら YES（先攻）。戦術評価AIがデッキ分析時に設定する |
| `MULLIGAN (42)` | YES/NO | 手札に Basic ポケモンがない場合のみ YES。それ以外は NO |
| `COIN_HEAD (46)` | YES/NO | 常に YES（コイン表を選ぶ） |
| `ACTIVATE (43)` | YES/NO | 原則 YES。ただし DeckPlan で特定カード効果の不使用を定義できる |
| `FIRST_EFFECT (44)` | YES/NO | 原則 YES |
| `MORE_DEVOLVE (45)` | YES/NO | 原則 NO（過度な退化は避ける） |
| `DRAW_COUNT (38)` | COUNT | options に数値候補が並ぶ。最大値を選ぶ |
| `DAMAGE_COUNTER_COUNT (39/40)` | COUNT | 最大値または戦術方針に従って選択 |

### テスト環境（ローカルで対戦シミュレーション可能）

`sample_submission/local_test_advanced.py` を使ってフル対戦を実行できる：

```bash
cd sample_submission
python local_test_advanced.py --games 30 --opponent random
python local_test_advanced.py --games 1 --verbose
```

- `result=0` → player0（自分）の勝ち
- `result=1` → player1（相手）の勝ち

---

### バッチ実行コマンド集（deck.csv を書き換えずに計測できる）

> **ポイント:** `--my-deck` / `--my-decks` を指定すれば `deck.csv` の上書きも `get_deck_plan()` 変更も不要。

#### よく使うコマンド

```bash
cd sample_submission

# ① 両デッキ × 全Tier-Sアーキタイプ（フル計測）
python src/tests/run_bench_suite.py --games 30

# ② 特定アーキタイプのみ両デッキ計測（例: 再計測）
python src/tests/run_bench_suite.py --games 30 --decks maries_obstagoon_ex dragapult_ex

# ③ Lucario のみ全アーキタイプ計測
python src/tests/run_bench_suite.py --games 30 --my-decks lucario

# ④ マリィ1デッキで特定アーキタイプのみ
python src/tests/tier_s_benchmark.py --games 30 --my-deck maries --decks crustle alakazam

# ⑤ Lucario で全アーキタイプ（旧来の単発実行と同等）
python src/tests/tier_s_benchmark.py --games 30 --my-deck lucario
```

#### タスク別推奨コマンド

| やること | コマンド |
|---------|---------|
| 汎用改善後の両デッキ影響確認 | `run_bench_suite.py --games 30` |
| 特定アーキタイプの再計測 | `run_bench_suite.py --games 30 --decks <arch1> <arch2>` |
| 新アーキタイプ対策の効果測定 | `tier_s_benchmark.py --games 30 --my-deck maries --decks <arch>` |
| Kaggle提出前の全力計測 | `run_bench_suite.py --games 60` |

#### 出力例（比較テーブル）

```
COMPARISON TABLE
Opponent                       maries     lucario    diff
----------------------------------------------------------------------
dragapult_ex                      81%         60%     -21%
hydrapple_ex                      88%         80%      -8%
...
AVERAGE                           84%         66%
```

### src/ の import は Kaggle 提出でも動作確認済み

`main_minimal_snapshot.py`（`src/` から import するパターン）を実際に Kaggle 提出して動作することを確認済み。  
新規に `src/` を使う場合は `main.py` 先頭に以下を入れる：

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
```

### デッキ構成（複数デッキ対応済み）

現在の `deck.csv`：**マリィのオーロンゲexデッキ（Tier A、全体 79% / Tier-S 80%）**

```
メインアタッカー: マリィのオーロンゲex (648) × 3  ← ベロバー(646)×4 → ギモー(647)×2 → ふしぎなアメ可
サブアタッカー:   マシマシラ (112) × 4            — ベンチユーティリティ
サポートライン:   ユキワラシ(103)×2 → ユキメノコ(104)×2
エネルギー:      基本悪 (7) × 10
サポーター:      リリィの決心 (1227)×4, ロケット団のラムダ (1219)×4, ポケパッド (1152)×4
アイテム:        ふしぎなアメ(1079)×2, ハイパーボール(1121)? なかよしポフィン(1086)×3 ← ベロバー検索
```

**デッキ候補一覧（deck_plan.py に全 DeckPlan 定義済み）**:
| ファイル | DeckPlan | 全体勝率 | Tier-S |
|---------|----------|---------|--------|
| deck_maries_obstagoon.csv | MARIES_OBSTAGOON_PLAN | **79%** | **80%** |
| deck_hydrapple.csv | HYDRAPPLE_PLAN | 77% | 70% |
| deck_lucario.csv | LUCARIO_PLAN | 72% | 71% |
| deck_dragapult.csv | DRAGAPULT_PLAN | 51% | — |
| deck_raging_bolt.csv | RAGING_BOLT_PLAN | 24% | — |

切り替え方: ベンチマーク時は `--my-deck lucario` 等を指定するだけ（ファイル変更不要）。  
Kaggle 提出用に恒久切り替えする場合のみ `deck.csv` 差し替え + `get_deck_plan()` return 変更。

### Level 4（探索API）の重要な前提条件

`search_begin()` は相手の非公開情報（デッキ・手札・伏せポケモン）を引数に取る。  
**評価関数が不十分な状態で Level 4 に入ると、間違った推測で間違った探索をするだけになる。**  
Level 3 の完了確認（中級者AI に対して勝率差が出る）を必ず確認してから Level 4 に進む。

---

## 改善履歴（提出後に更新）

| 提出回 | 主な変更 | Tier-S ローカル勝率 | Kaggle スコア |
|-------|---------|------------------|-------------|
| v1 | ランダム選択（初期状態） | — | — |
| v2 | フェーズ0完了：コンテキスト別選択・DeckPlan・DecisionTrace | ランダムAI 21/30（70%） | 未提出 |
| v3 | フェーズ1完了：カードタイプ別PLAY優先度・_pick_best_play追加 | ランダムAI 25/30（83%） | 未提出 |
| v4 | フェーズ2完了：meta_decks.py生成・Tier-Sベンチマーク・DeckPlan対アーキタイプ方針追加 | Tier-S平均66%（架空デッキ相手）| 未提出 |
| v5 | フェーズ3完了：BoardEvaluator実装・HP低下時リトリート判断・should_retreat統合 | Tier-S平均73%（架空デッキ相手）| 未提出 |
| v6 | フェーズ4完了：search_policy.py統合。meta_decks.pyを実データ10デッキに差し替え（pokeka+torecamap 1626デッキ）。フェーズ2/3を実データデッキ相手に再実施。 | Tier-S **72%**（90試合: dragapult 77%・hydrapple 57%・terasta 83%）| 未提出 |
| v7 | Phase 3 ゲート未達：評価器+探索(73%)がルールのみ(72-77%)を下回るため main.py を choose_action（ルールのみ）に戻す。evaluator.py(bench_count=3.0)/search_policy.py はコードとして保持し Phase 5 で再活用予定。 | Tier-S **72%**（30試合: dragapult 70%・hydrapple 80%・terasta 67%）| 未提出 |
| v8 | Phase 5-1: opponent_model.py 新規作成。choose_switch にアーキタイプ別ボスターゲット判断を追加。hydrapple_ex 57%→87%（Meganium ターゲット戦略が機能）。全10デッキ 72%。 | Tier-S **71%**（150試合: dragapult 72%・hydrapple 66%・terasta 72%）| 未提出 |
| v9 | Phase 5-2 デッキ比較結果: Lucario 72%・Dragapult 51%・Raging Bolt 24%・Hydrapple 77%・Maries Obstagoon 79%。マリィのオーロンゲexデッキに切り替え。deck_plan.py に全5デッキのDeckPlan保持。 | Tier-S **80%**（150試合: dragapult 82%・hydrapple 74%・terasta 84%）| 未提出 |
| v10 | Phase 5-3: MatchupPolicy 実装。matchup_policy.py 新規作成。vs dragapult_ex: bench=2 上限（choose_to_bench/main_policy 両方に適用）。bench=2が最良設定(81% 90試合)。85%目標は分散内で未達。 | Tier-S **81%**（90試合: dragapult 81%）| 未提出 |
| v11 | Phase 6 ARC: ptcg-ai-architecture.md 策定 → ARC-3(estimate_remaining), ARC-4(opponent_requirement_score), ARC-5(Iono撃ちどき判断), ARC-7(逃げエネ0 Lazy Evaluation) を実装。 | Tier-S(マリィ) **88%**（dragapult ~80%・hydrapple 93%・terasta 80%）。全10arch **86%**（旧81%）+5%改善 | 未提出 |

---

## フェーズ完了ゲート（次のフェーズに進む前に必ず確認）

**このルールは最重要。完了条件が満たせない場合、次のフェーズには進まない。**

| フェーズ → 次 | 進める条件（すべて満たすこと） |
|-------------|--------------------------|
| 0 → 1 | ・不正選択・例外がゼロ<br>・全 SelectContext にハンドラが存在する<br>・固定シナリオテストが通る<br>・DecisionTrace がログに出力される |
| 1 → 2 | ・`local_test_advanced.py` でランダムAIに 30 試合中 20 勝以上<br>・明らかにおかしい行動がログに見られない |
| 2 → 3 | ・Tier-S デッキ相手に 30 試合実施済み（勝率問わず）<br>・ベンチマークレポートが存在する<br>・DeckPlan に対アーキタイプ方針が記載されている |
| 3 → 4 | ・評価関数あり/なしで Tier-S デッキ相手の勝率差が 5% 以上<br>・**`search_begin` の相手情報推定戦略が文書化されている** |<br>**[!] 未達成 (2026-06-26)**: 評価器+探索(73%) vs ルールのみ(72-77%) — 差なし。Phase 4 コードは実装済みだが main.py ではルールのみを使用中。 |
| 4 → 5 | ・探索ありが探索なし Level 3 より明確に強い（勝率差 5% 以上） |

**ゲートを通過できないとき：**
1. 該当チェックを `[!]` にして理由を記録する
2. 現在のフェーズ内で原因を修正する
3. 修正後に再度ゲート確認を行う
4. 同じフェーズで 3 回以上詰まった場合は、デバッグAI に分析を依頼する

---

## チェックボックスの更新ルール

- 作業を開始したら `- [ ]` → `- [~]` （進行中）
- 完了したら `- [~]` → `- [x]` （完了）
- ブロックされたら `- [ ]` → `- [!]` と理由をコメントで追記

---

## マルチデッキ開発方針（恒久ルール）

### 各デッキの位置づけ

| デッキ | 位置づけ | 理由 |
|-------|---------|------|
| **マリィのオーロンゲex** ★ | メイン開発（常時） | 現在最高勝率 (79%)。全改善で必ず計測する |
| **メガルカリオex** ★ | メイン開発（常時） | 全アーキタイプ計測済みで比較基準が明確。全改善で必ず計測する |
| **カミツオロチex** | 定期スポットチェック | 77% と高水準。汎用改善が一段落した区切りで全10アーキタイプを計測 |
| **ドラパルトex** | 保留（専用ロジック待ち） | 汎用ルールでは 51%。Rare Candy 進化・ファントムダイブ専用ロジックが必要になってから着手 |
| **タケルライコex** | 保留（DeckPlan 見直し待ち） | 24% は DeckPlan 設定ミスの可能性大。原因究明後に再評価 |

### 改善実装後の計測ルール

```
汎用改善を実装する（MatchupPolicy・評価器改善 等）
    ↓
1. マリィ版でベンチマーク（影響するアーキタイプ、または全10アーキタイプ）
2. Lucario 版でベンチマーク（同上）
3. 両方で改善が確認できたらコミット・STATUS.md の勝率表を更新
4. Hydrapple は「大きな区切り」でのみスポットチェック（下記タイミング参照）
```

### Hydrapple スポットチェックのタイミング

- MatchupPolicy が一通り実装し終わったとき
- Kaggle 提出直前の最終確認
- 結果が大きく想定外だったとき（デバッグの手がかりとして）

---

## 禁止事項（全エージェント共通）

- `cg/` フォルダのファイルを変更しない
- `data/` フォルダのファイルを変更しない
- `main.py` の `agent(obs_dict: dict) -> list[int]` シグネチャを変更しない
- このファイルの「モデル選択ガイド」を無視してOpusを使用しない
- フェーズ完了ゲートを確認せずに次のフェーズに進まない
- 完了条件を満たせていないタスクのチェックを `[x]` にしない
- 非自明な判断・設計決定を `docs/strategy-knowledge.md` に記録せずにコードだけ書かない
- `strategy-knowledge.md` の `状態:` を証拠なく `検証済み` にしない
