# 現在の作業状態（AI セッション開始時に必ず読む）

**このファイルの使い方:**
- AI は新しいセッション開始時にまずこのファイルを読む（ORCHESTRATOR.md より先に）
- AI は「今すぐやること」を完了したら、このファイルを更新して「次の予定タスク」を「今すぐやること」に昇格させる
- 人間は「続けて」と言えば「今すぐやること」を実行する。「次にこれをやって」と言えばバックログに追記またはタスクを差し込む

最終更新: 2026-06-26 (ARC-3/4/5/7 実装。マリィ全10arch 86%（旧81%）+5%改善。Lucario 全10arch 70% 再計測完了)
最終作業者: Claude Sonnet 4.6

---

## 今すぐやること（次の 1 タスク）

> **タスク: Kaggle 再提出 → ogerpon_bullet boss_rush 効果検証（B-09）**

**背景:**
- 2026-06-26 改善サイクル実行済み。ogerpon_bullet boss_rush 追加（matchup_policy.py）
- ローカル検証: 83% 維持（60試合）。実戦 Kaggle での効果確認が次ステップ

**実施手順:**
1. `deck.csv` が `deck_maries_obstagoon.csv` の内容と一致しているか確認
2. `main.py` の `get_deck_plan()` が `MARIES_OBSTAGOON_PLAN` を返すことを確認
3. Kaggle に提出
4. **実戦ログが得られたら `docs/kaggle-improvement-cycle.md` に従って自動分析・改善を実行**
5. 特に ogerpon_bullet 戦績が 0% から改善しているか確認

**参照:**
- `deck_maries_obstagoon.csv` — マリィデッキ
- `main.py` — エントリーポイント
- `docs/kaggle-improvement-cycle.md` — ログ提供時の自動実行プロセス

---

## 次の予定タスク（「今すぐやること」完了後の順番）

> AI は「今すぐやること」完了後、ここの **#1 を「今すぐやること」に移動** してから作業を始める。
> 人間が「続けて」と言えば自動的に #1 に進む。割り込みタスクがあれば人間が指示する。
> **各タスクのベンチマークは原則マリィ + Lucario 両方で実施する。**

| 順番 | タスク | 両デッキの計測対象 | ねらい |
|-----|--------|-----------------|-------|
| #1 | **MatchupPolicy 拡張**: vs crustle — ボスでイシズマイ引き出しを徹底 | マリィ・Lucario 両方で vs crustle 30試合 | 現在マリィ73%・Lucario60%、目標80%+ |
| #2 | **MatchupPolicy 拡張**: vs alakazam — 対フーディン戦略 | マリィ・Lucario 両方で vs alakazam 30試合 | 現在マリィ90%・Lucario 63%、Lucario 対策未実装 |
| #3 | **Hydrapple スポットチェック**（全10アーキタイプ × 30試合） | Hydrapple のみ | MatchupPolicy の恩恵を確認。77% から上がっていれば並行開発を本格化 |
| #4 | **Kaggle 提出** → 実戦スコア確認 | — | ローカル勝率との乖離を把握。次改善サイクルの起点 |

---

## 勝率ベースライン

### デッキ別総合勝率（最新測定値）

現在使用中: ★ **マリィのオーロンゲex**

| デッキ | DeckPlan 定数 | 総合勝率 | Tier-S勝率 | 計測日 | 計測条件 |
|-------|-------------|---------|-----------|--------|---------|
| ★ deck_maries_obstagoon.csv | MARIES_OBSTAGOON_PLAN | **86%** | **88%** | 2026-06-26 | 全10arch×30試合（ARC実装後） |
| deck_hydrapple.csv | HYDRAPPLE_PLAN | **80%** | **80%** | 2026-06-26 | 全10arch×30試合（MatchupPolicy反映後） |
| deck_lucario.csv | LUCARIO_PLAN | **70%** | **87%** | 2026-06-26 | 全10arch×30試合（ARC実装後） |
| deck_dragapult.csv | DRAGAPULT_PLAN | 51% | — | 2026-06-26 | 限定計測 |
| deck_raging_bolt.csv | RAGING_BOLT_PLAN | 24% | — | 2026-06-26 | 限定計測 |

> デッキ切り替え方法: `deck.csv` を差し替え + `main.py` の `get_deck_plan()` return 値を変更

---

### アーキタイプ別勝率詳細

`–` = 未計測 / `?` = 参考値（試合数が少なく誤差大）

| 相手アーキタイプ | Tier | ★マリィ | Lucario | Hydrapple | 備考 |
|---------------|------|--------|---------|-----------|------|
| dragapult_ex | S | **~80%** (60試+分散) | **87%** (30試/ARC後) | — | Lucario ARC後 +14%改善 |
| hydrapple_ex | S | **93%** (30試/ARC後) | **50%** (30試/ARC後) | — | Lucario 最低値。boss_rush未設定？ |
| terasta_bullet | S | **80%** (30試/ARC後) | **70%** (30試/ARC後) | — | 分散内 |
| raging_bolt_ex | A | **93%** (30試/ARC後) | **80%** (30試/ARC後) | — | 安定 |
| mega_lucario_ex | A | **80%** (30試/ARC後) | **73%** (30試/ARC後) | — | ミラー。同等 |
| olivia_ex | A | **87%** (30試/ARC後) | **60%** (30試/ARC後) | — | Lucario 低値。boss_rush調査要 |
| ogerpon_bullet | A | **93%** (30試/ARC後) | **60%** (30試/ARC後) | — | Lucario 対策未完 |
| maries_obstagoon_ex | A | **90%** (30試/ARC後) | **73%** (30試/ARC後) | — | ARC後改善 |
| crustle | A | **73%** (30試/ARC後) | **70%** (30試/ARC後) | — | Lucario +7%、B-04継続課題 |
| alakazam | B | **83%** (30試/ARC後) | **80%** (30試/ARC後) | — | boss_rush有効 |
| **平均** | | **86%** (10arch/30試) | **70%** (10arch/30試) | **77%** (全体) | 2026-06-26 ARC-3/4/5/7 実装後 |

> **マリィ全10arch完了**（**86%**）。**Lucario 全10arch 完了**（**70%**、旧72%近傍で誤差内）。  
> ARC実装内容: ARC-3(残りリソース推定), ARC-4(要求値スコア), ARC-5(Ionoタイミング), ARC-7(逃げエネ0 Lazy Eval)  
> Lucario の hydrapple_ex 50% / olivia_ex 60% が低値。boss_rush 設定の追加が次の改善候補。  
> bench=2 は `only_for_decks={"maries_obstagoon_ex"}` でマリィ専用化済み。

---

## アクティブバックログ（発見された未完了タスク）

> このセクションに人間が「次やること」を追記してよい。

### 優先度: 高

| # | タスク | 背景・理由 |
|---|--------|----------|
| B-01 | **全アーキタイプのベンチマーク再計測**（マリィ版） | デッキ変更後の正確な勝率が不明。まず現状把握が必要 |
| B-02 | ~~**MatchupPolicy: vs dragapult ベンチ上限**~~ ✓完了 | bench=2上限実装済み。81%(90試)で baseline と同等。85%目標未達だが最良設定 |
| B-03 | **MatchupPolicy: vs hydrapple/olivia ボスラッシュ強化** | メガニウムKOが最優先なのに他のカードを先に使うケースがある |
| B-09 | **Kaggle 再提出 → ogerpon_bullet boss_rush 効果検証** | ローカルでは中立(83%維持)。実際の対戦で0%→改善するか確認 |

### 優先度: 中

| # | タスク | 背景・理由 |
|---|--------|----------|
| B-04 | **イワパレス(Crustle)対策の完成** | EX免疫でEXダメージ0、ボスでイシズマイ引き出し戦略は機能するが66%どまり。詳細は `docs/crustle_counter_plan.md` |
| B-05 | **フーディン(Alakazam)対策** | パワフルハンド（手札枚数×20ダメ）に対する対策未検討 |
| B-06 | **マリィのオーロンゲex デッキの DeckPlan 精査** | Lucario 用に作られた共通ロジックがそのまま使われている部分がある |

### アーキテクチャ改善（docs/ptcg-ai-architecture.md より）

> 詳細設計・背景は `docs/ptcg-ai-architecture.md` を必ず先に読むこと。

| # | タスクID | タスク | 優先度 |
|---|---------|--------|--------|
| ARC-4 | ARC-4 | **要求値スコア実装** — `evaluator.py` に `opponent_requirement_score()` 追加 | 高 |
| ARC-5 | ARC-5 | **妨害札タイミング判断** — `opponent_model.py` に `estimate_hand_quality()` 追加、`_pick_best_play` に組み込み | 高 |
| ARC-7 | ARC-7 | **逃げエネ0前出し戦略（Lazy Evaluation）** — `DeckPlan.free_retreat_ids` 追加、`choose_switch` 改修 | 高 |
| ARC-3 | ARC-3 | **残りリソース推定** — `OpponentModel.estimate_remaining(card_id)` 実装（ボス残り枚数推定） | 中 |
| ARC-6 | ARC-6 | **汎用壁役戦略** — `MatchupOverride.use_wall` フィールド追加と壁役ロジック | 中 |
| ARC-1 | ARC-1 | **サーチ優先度調整** — アイテム系サーチをサポーターと同等以上にする | 中 |
| ARC-2 | ARC-2 | **手札干渉後の空打ちシャッフル** — `obs.logs` から干渉検知 → 空打ちロジック | 中 |

### 優先度: 低

| # | タスク | 背景・理由 |
|---|--------|----------|
| B-07 | **Kaggle提出と実戦スコア確認** | ローカル勝率との乖離を確認する |
| B-08 | **1626デッキ軸カードの JP名→ID 解決精度向上** | キャッシュ更新時に自動解決できるよう名前正規化を改善 |

---

## 最近の作業ログ（最新 5 件）

### 2026-06-26 ⑬ Lucario 全10アーキタイプ再計測（ARC実装後）

**やったこと:**
- ARC-3/4/5/7 実装後の Lucario デッキで全10アーキタイプ×30試合を計測
- `run_benchmark.py` を新規作成（deck.csv 読み込みで全アーキタイプ計測）

**結果（Lucario ARC後）:**

| Arch | WR |
|------|----|
| dragapult_ex | 87% |
| raging_bolt_ex | 80% |
| alakazam | 80% |
| mega_lucario_ex | 73% |
| maries_obstagoon_ex | 73% |
| terasta_bullet | 70% |
| crustle | 70% |
| ogerpon_bullet | 60% |
| olivia_ex | 60% |
| hydrapple_ex | 50% |
| **平均** | **70%** |

**知見:**
- dragapult_ex 87%（旧73%）は ARC-7 Lazy Eval + ARC-5 Iono タイミングが効いた可能性
- hydrapple_ex 50% が最低。boss_rush（メガニウムKO優先）の Lucario 用設定が未適用の可能性
- olivia_ex 60% も低値。同様に boss_rush 調査が有効
- 次の改善候補: Lucario 向け hydrapple_ex / olivia_ex の boss_rush 設定確認

### 2026-06-26 ⑫ Kaggleログ改善サイクル実行 — ogerpon_bullet boss_rush追加

**やったこと:**
- 改善前の Kaggle ログ13試合（GameID 81976680〜81995133）を `kaggle-improvement-cycle.md` に従って分析
- アーキタイプ別 Kaggle 結果: dragapult_ex 100%(6試) / ogerpon_bullet **0%(3試)** / crustle 50%(2試)
- ogerpon_bullet 0% → Priority Rule#1 適用（≥3試合で0%） → Step 4b
- TRACE ログ精査（3試合）: SWITCH score=0 多発 + 複数ターン攻撃不能がパターン
  - マシマシラ(112) が相手のエネルギー転送役 → KO すれば多タイプアタッカー戦略を崩せる
  - ARCH_BOSS_TARGETS["ogerpon_bullet"] = [112, 184, 108, 96] 既定義済みだが boss_rush=False だったため未使用
- `matchup_policy.py`: `"ogerpon_bullet": MatchupOverride(boss_rush=True)` 追加

**ベンチマーク結果（マリィ vs ogerpon_bullet）:**
- 30試合: 80%（ベースライン83%との-3%は誤差範囲 → cycle doc「±3%→60試合」基準適用）
- 60試合: **83%**（ベースラインと同等、悪化なし）

**判断:** ローカルランダムAIでは効果測定困難（マシマシラがランダムに動く）。戦略的根拠（エネルギー転送エンジン破壊）が有効のため「条件付き採用 → Kaggle で実証」。

**Kaggle結果との比較:**
- Kaggle 0%(3試) は「改善前」コードでの結果。boss_rush追加後の再提出で改善見込み

### 2026-06-26 ⑪ Crustle 非EXアタッカー交代戦略

**やったこと:**
- `skip_attack_ex_immune`（EX が Crustle に殴って 0 → ターン終了）を削除
- 新方針: Crustle 対面でエネルギーを非EXアタッカーへ回す + 準備完了後に引き込んで攻撃
- `main_policy.py`: `_non_ex_bench_ready()` ヘルパー追加、Crustle 引き込みトリガー（攻撃ブロック前）
- `main_policy.py::_pick_best_attach()`: Crustle 対面は `non_ex_energy_priority` 使用
- `target_policy.py::choose_attach_from()`: 同上
- `target_policy.py::choose_switch()`: Crustle 対面は `non_ex_active_priority` 使用
- `deck_plan.py::LUCARIO_PLAN`: `non_ex_active_priority=[674]`（Solrock 除外）、`non_ex_energy_priority=[674, 677]`

**結果（60試合）:**
- マリィ vs crustle: 64% → **68%**（参考: 30試合では 73%）
- Lucario vs crustle: 64% → **63%**（誤差内、変化なし）
- マリィ 全10arch: 79% → **81%**（30試合 × 10arch）

**知見:**
- `opponent_active_blocks_ex()` が False の間（boss_rush でイシズマイが場に）は EX が普通に攻撃する → 正しい
- Solrock(676) は HP 低く Crustle 120 ダメで 1 撃 → `non_ex_active_priority` から除外

### 2026-06-26 ⑩ Hydrapple スポットチェック（全10アーキタイプ）

**やったこと:**
- Hydrapple デッキで全10アーキタイプ×30試合計測
- 結果: 全体 **80%**（従来77%から+3%）、Tier-S **80%**（従来70%から+10%）

**アーキタイプ別（30試合）:**

| Arch | WR |
|------|-----|
| raging_bolt_ex | 97% |
| mega_lucario_ex | 90% |
| hydrapple_ex | 87% |
| dragapult_ex | 83% |
| olivia_ex | 83% |
| maries_obstagoon_ex | 77% |
| alakazam | 77% |
| ogerpon_bullet | 73% |
| terasta_bullet | 70% |
| crustle | 60% |

**知見:** boss_rush / bench=2 のMatchupPolicy がHydrappleにも恩恵（mega_lucario 90%、alakazam 77%）。Crustle 60% は依然として最低だが boss_rush のみ適用済み。

### 2026-06-26 ⑨ vs crustle / vs alakazam boss_rush 完成

**やったこと:**
- vs crustle: ex_immune 全変更リバート（forced retreat + energy priority switch が逆効果と判明）
- vs crustle 90試合で安定計測: マリィ **64%**、Lucario **64%**
- `matchup_policy.py` に `"alakazam": MatchupOverride(boss_rush=True)` を追加
- vs alakazam 60試合計測: マリィ **87%**、Lucario **63% → 78%**（+15%）

**知見:**
- Crustle 対面で非EX ポケモン（イベルタル/ギモー）を優先するとCrustleの120ダメで1撃される逆効果
- Alakazam対面の boss_rush（Abra 50HP を1撃KO）は Lucario に顕著な効果（+15%）

### 2026-06-26 ⑧ バッチベンチマーク実装 + bench=2 デッキ依存化
**やったこと:**
- `src/tests/run_bench_suite.py` 新規作成（複数デッキ一括計測 + 比較テーブル出力）
- `tier_s_benchmark.py` に `--my-deck` 引数追加（deck.csv 書き換え不要でデッキ切り替え）
- `deck_plan.py` に `set_active_plan()` 追加（プログラマティックなデッキ切り替え）
- `matchup_policy.py` に `only_for_decks` フィールドを追加、bench=2 をマリィ専用に変更

**結果（Lucario 再計測）:**
- maries_obstagoon_ex 再計測: 33% → **67%**（33% は外れ値だった）
- dragapult_ex 再計測（bench制限なし）: 60% → **73%**（+13%）
- mega_lucario_ex 再計測（bench制限なし）: 70% → 70%（変化なし）

**知見:** bench=2 制限の dragapult_ex 対策は マリィ専用。Lucario は Lunatone+Solrock エネサイクルが必要なため bench=2 が逆効果だった。

### 2026-06-26 ⑦ Lucario 全アーキタイプ再計測（MatchupPolicy 実装後）
**やったこと:**
- Lucario デッキに一時切り替えして 10arch × 30試合を計測
- 計測後にマリィデッキに復元

**結果:**
- hydrapple_ex: 57%→**80%** (+23%) ← boss_rush が Lucario にも有効
- dragapult_ex: 77%→60% (-17%) ← bench=2 がエネサイクル阻害の疑い
- maries_obstagoon_ex: 73%→33% ← 極端な低値、要再計測
- 全体平均: 72%→66%（maries_obstagoon_ex 異常値含む）

**知見:** bench=2 制限は Lucario の Lunatone(675)+Solrock(676) エネサイクルに 3 スロット必要なため逆効果になる可能性。マリィ専用の対策として分離する検討が必要。

### 2026-06-26 ⑥ boss_rush ボスの指令優先化（vs hydrapple/olivia）
**やったこと:**
- `main_policy.py` に `BOSS_ORDERS_ID = 1182` と `_boss_target_on_bench()` ヘルパーを追加
- `_pick_best_play()` に boss_rush_active 判定を追加（ターゲットがベンチにいる場合スコア=40で最優先）
- vs hydrapple_ex 2×30試合: run1 87% / run2 90% → **平均 88%**（基準 74%、+14%）
- vs olivia_ex 2×30試合: run1 83% / run2 93% → **平均 88%**（基準 93%(30試)、分散内）

**結果:** hydrapple_ex で **+14%** の明確な改善。olivia_ex は分散内で悪化なし。

**知見:** boss_rush はターゲット(710=メガニウム)がベンチにいる場合のみ発動するため、不要なターンにボスを使う副作用はない。

### 2026-06-26 ⑤ vs mega_lucario_ex MatchupPolicy bench=2 ベンチマーク
**やったこと:**
- `matchup_policy.py` に `"mega_lucario_ex": MatchupOverride(max_bench_size=2)` を追加
- vs mega_lucario_ex 計 60 試合ベンチマーク: run1 70% (21/30), run2 83% (25/30)

**結果:** 平均 **77%**（46/60）。基準 63% から **+14%** 改善（Z ≈ 2.5 で統計的有意）。

**知見:** ベンチを2体に絞ることで格闘攻撃の被弾面積を削減。bench=2 は dragapult_ex 対策と同じ理由で有効。

### 2026-06-26 ④ MatchupPolicy 実装（タスク 5-3）
**やったこと:**
- `src/knowledge/matchup_policy.py` 新規作成（MatchupOverride データクラス + MATCHUP_OVERRIDES）
- `target_policy.py:choose_to_bench()` に MatchupPolicy 適用（max_bench_size 超過時は min_count のみ）
- `main_policy.py:_pick_best_play()` に MatchupPolicy 適用（ベンチ上限時は全ポケモン score=0）
- vs dragapult_ex ベンチマーク: bench=2 → 81%(90試)、bench=3 → 73%(30試)、制限なし → 80%(30試)

**結果:** bench=2 が最良設定。85%目標は未達（30試合分散 ±7% のため統計的に確認不能）。

**知見:** MatchupPolicy ベンチ制限はベースライン(82%)と統計的同等。bench=3 は進化ライン阻害で逆効果。

### 2026-06-26 ③ ドキュメント整備
**やったこと:**
- `docs/opponent_model.md` 新規作成（盤面推定ロジックの全体説明）
- `docs/axis_card_analysis.md` 新規作成（1626デッキ軸カード分析 + 再抽出手順）
- `docs/strategy-knowledge.md` に [META-006], [META-007] 追記
- `docs/STATUS.md`（このファイル）新規作成

**結果:** ドキュメント整備完了。AIセッション再開時の参照効率向上。

---

## 「やってはいけないこと」（過去の失敗から）

| 禁止 | 理由 |
|-----|------|
| Crustle対策で非EXポケモンを優先する | イベルタル(110HP)・ギモー(100HP)がCrustleの120ダメで1撃。逆効果 |
| 評価器 + 探索(search_policy)を main に組み込む | ルールのみより下回る(73% vs 77%)。Phase 4 コードはそのまま保持 |
| ベンチに全進化ラインを出す（vs Dragapult） | ファントムダイブで全体に10ダメ×6散布される |
| ドラパルトex デッキへの切り替え | 汎用ルールベースでは51%。専用ロジック未実装 |

---

## 重要ファイルマップ（AI 向け）

| 役割 | ファイル |
|-----|---------|
| エージェント本体 | `main.py` |
| デッキ設定 | `deck_maries_obstagoon.csv`（現在使用中） |
| デッキ戦略定義 | `src/knowledge/deck_plan.py` |
| 行動決定ルーター | `src/decision/router.py` |
| メイン行動決定 | `src/decision/main_policy.py` |
| ターゲット選択 | `src/decision/target_policy.py` |
| 相手デッキ推定 | `src/knowledge/opponent_model.py` |
| メタデッキ知識 | `src/knowledge/meta_decks.py` |
| 盤面評価（未接続） | `src/decision/evaluator.py` |
| EX免疫検出・非EX交代 | `src/knowledge/ex_immune.py` |
| ベンチマーク実行 | `local_test_advanced.py` |
| フェーズ管理 | `docs/ORCHESTRATOR.md` |
| 戦略知識ベース | `docs/strategy-knowledge.md` |
| 軸カード分析 | `docs/axis_card_analysis.md` |
| 相手推定ロジック | `docs/opponent_model.md` |
