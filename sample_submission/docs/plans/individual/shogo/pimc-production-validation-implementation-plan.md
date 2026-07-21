# 実装計画: PIMC 本番改善の検証（ml_policy に対する効果測定）

作成: 2026-07-21
状態: 実装計画
前提: [stage2-match-context-separation-implementation-plan.md](./stage2-match-context-separation-implementation-plan.md) 完了。head-to-head で `rule_pimc` > `rule_lethal_estimated`（61% vs 39%、100試合）が実証済み（[2026-07-21_stage2_head_to_head.md](../../../results/2026-07-21_stage2_head_to_head.md)）
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## この計画が答える問い

Stage2の head-to-head は「PIMCは**lethal探索**より強い」を示した。しかし比較した2つ（`rule_pimc` / `rule_lethal_estimated`）は**どちらも本番エージェントではない**。現在Kaggleに提出中の本番は `ml_policy`（`ptcg_ai/core/agent.py:18` の `AGENT_TYPE = "ml_policy"`）で、`ml_policy + lethal_simple + dummy hidden state`（config `ml_lethal`）で動いている。

**したがって未回答の問いは「PIMC（および estimated hidden state）を本番 `ml_policy` に入れたら、提出物は強くなるのか？」** これに答えを出すのが本計画の目的。勝てば即提出候補（すぐスコアになる）、負ければ「研究では勝ったのに本番では勝たない」理由の分析材料になり、v1（policy prior等）投資の前提が変わる。

---

## スコープ

**やること**: `ml_policy` フレームワークの上で、現行提出 `ml_lethal`（lethal + dummy）を基準に、PIMC化・estimated hidden state化の効果を head-to-head で測る。

**やらないこと（Non-goals）**:
- 多様な相手デッキでの評価 → follow-up。理由: `match_context._load_own_deck_ids()` が常に `deck.csv` を読む設計のため、estimated構成で別デッキを両陣営に割り当てると自分デッキの推定が壊れる（別途インフラ対応が要る）。本計画は Stage1/2 と同じ**ミラーマッチ（同一 `deck.csv`）**に留め、本番改善の最初のシグナルを得ることに集中する。
- policy prior 統合・相手ターンをまたぐ探索（v1） → 本計画の結果を見てから
- `AGENT_TYPE` の切り替えや実際のKaggle提出 → 人間の判断。本計画は「提出すべきかの判断材料」を出すところまで
- `lethal_simple.py` の変更 → 一切触らない

---

## 事前確認済みの契約（Contracts）

### 1. 【最重要】config衝突の問題と解決策

`ml_policy_agent.agent()` は `_get_config()`（モジュールグローバル `_config_cache` に `load_config(_CONFIG_NAME)` をキャッシュ、`_CONFIG_NAME = os.environ.get("PTCG_AI_ML_CONFIG", "ml_lethal")`）を使い、`_select_action(obs)` は `_try_lethal(obs)` に**config を渡していない**（`ml_policy_agent.py:166`）。

**帰結**: 1プロセス内で `ml_lethal` と `ml_pimc` の2エージェントを対戦させようとしても、`_config_cache` が両者で共有されるため**両方が同じconfigになってしまう**（実際には head-to-head ではなく自己対戦になる）。Stage2の rule_based head-to-head が `selector.select_action(obs, full_deck, config=...)` を明示configで直接呼んでこれを回避したのと同じ問題が、ml_policy 側にもある。cgエンジンはプロセス内シングルトン（`league/run_match.py` docstring）なので、両エージェントの意思決定関数は必ず同一プロセスで呼ばれる → プロセス分離では回避できない。

**解決策（推奨）**: `ml_policy_agent` に **config注入点**を追加する（後方互換）。`selector.select_action` が既に持つ `config` 引数の ml_policy 版。

```python
# 変更後（すべて default 引数で後方互換。core/agent.py・main.py からの agent(obs) 呼び出しは不変）
def agent(obs: Observation, config: dict | None = None) -> list[int]: ...
def _select_action(obs: Observation, config: dict | None = None) -> list[int]: ...
# _try_lethal は既に config 引数を持つ（ml_policy_agent.py:115）。_select_action から明示的に渡すだけ。
```

`config` が `None` のときは従来どおり `_get_config()` にフォールバック（本番・既存テストの挙動は完全に不変）。value_shadow ゲートも同じ config を見るようにする。

**代替策（agentを変更したくない場合）**: 使い捨てスクリプト側で `agent()` の本体（`match_context.update` → value_shadow → `_try_lethal(obs, config=明示config)` → PolicyModel の maxCount 分岐）を複製する。ただし複製は壊れやすいため、再利用可能な注入点を作る推奨策を採る。

### 2. config行列（すべて `ml_policy` フレームワーク、deck は共通 `deck.csv`）

| config名 | search | hidden state | 位置づけ | 既存/新規 |
|---|---|---|---|---|
| `ml_lethal` | lethal_simple | dummy | **現行提出（基準）** | 既存 |
| `ml_lethal_estimated` | lethal_simple | estimated | hidden info 単独の効果 | 既存 |
| `ml_pimc` | pimc | estimated | **本命候補（full）** | 新規作成 |
| `ml_pimc_dummy` | pimc | dummy | pimc単独の効果（任意） | 新規作成（任意） |

`ml_pimc` は `configs/rule_pimc.json` と同じ `lethal_search` セクション（`module: pimc`, `hidden_state_source: estimated`, `num_determinizations: 4`, `time_limit_ms: 300`）を持つ。`ml_policy_agent._SEARCH_MODULES` には Stage2で既に `pimc` が登録済み（`ml_policy_agent.py:37`）なので、config を置くだけで動く。

### 3. head-to-head の方法論（Stage2から踏襲）

`league/run_match.play_match(agent0, agent1, deck0, deck1, seed)` を使い、`agent0`/`agent1` は §1の注入点で別configを束縛したクロージャ。先手/後手を半々に入れ替え、試合ごとに `match_context.reset()`。Wilson 95% CI。エラー・実行時間も記録。`match_context` は Stage2でプレイヤー分離済みなので混線しない。

---

## ステップ

### Step 1: config注入点の追加と新規config作成

- `ml_policy_agent.agent(obs, config=None)` / `_select_action(obs, config=None)` を追加し、`_try_lethal` へ config を明示的に渡す（§1推奨策）。`config=None` で従来挙動。
- `configs/ml_pimc.json` を作成（§2）。`ml_pimc_dummy.json` は任意。
- 完了条件: 既存テスト全green（`agent(obs)` の1引数呼び出しが不変であることの回帰確認）。新規に「`agent(obs, config=A)` と `agent(obs, config=B)` が同一プロセス内で独立して別configを使う」ことを検証する unit test を1件追加（config衝突の再発防止）。`test_local_game_advanced.py --games 10` を `ml_pimc` 相当で完走（`PTCG_AI_ML_CONFIG=ml_pimc`）。

### Step 2: 主比較 — 現行提出 `ml_lethal` vs 本命 `ml_pimc`

- 100試合（先手/後手半々、ミラー）。勝率・95% CI・エラー・実行時間を記録。
- 完了条件: `results/2026-07-21_pimc_prod_validation_main.md` に記録。**`ml_pimc` の勝率CIが0.5を含むか否か**を明記（これが「提出すべきか」の一次判定）。

### Step 3: 分離比較 — hidden info 単独の効果（`ml_lethal` vs `ml_lethal_estimated`）

- 同じ方法論で100試合。目的: PIMCを入れなくても、Stage1で配線した estimated hidden state **だけ**で本番が改善するかを見る（改善するなら、探索コストゼロのより安い提出候補になる）。Stage1では自己対戦しか測っておらず、この head-to-head は未実施。
- 完了条件: `results/2026-07-21_pimc_prod_validation_hiddeninfo.md` に記録。

### Step 4: （任意）full attribution — pimc単独の効果

- 時間に余裕があれば `ml_lethal` vs `ml_pimc_dummy`（探索だけ変え hidden は dummy）を測り、Step2/3と合わせて「hidden info の寄与」と「pimc の寄与」を分離する2×2を完成させる。
- 完了条件: 実施した場合のみ `results/` に記録。省略可（DoDでは必須にしない）。

### Step 5: 判断のとりまとめ

- Step2/3（+任意4）を1つの結論ドキュメントにまとめ、**明示的な判断ルール**で締める:
  - `ml_pimc` が `ml_lethal` に有意に勝ち越し（CI が0.5を除外） → **提出候補**。次アクションは「切り出し・統合＋tsuoimorikaさん共有」を経て `AGENT_TYPE`/config 切替を人間が判断。
  - `ml_lethal_estimated` が有意に勝ち越し、かつ `ml_pimc` がそれ以下 → **より安い提出候補**（探索コストなしで hidden info だけ入れる）を優先検討。
  - どれも `ml_lethal` に有意差なし → 研究の勝ち（vs lethal）が本番に転移しなかった。理由分析（例: `ml_policy` は既に PolicyModel が強く lethal/pimc の寄与が小さい、ミラーだと opponent modeling の利得が出にくい 等）を記し、v1 の前提を見直す。
- 完了条件: `results/2026-07-21_pimc_prod_validation_summary.md` に判断を記載。

---

## Definition of Done

- [ ] `ml_policy_agent` に後方互換の config 注入点があり、既存の `agent(obs)` 呼び出しは不変
- [ ] 同一プロセス内で別configが独立して効くことの回帰テストが追加済み
- [ ] `ml_lethal` vs `ml_pimc`（主比較）の結果が `results/` に記録済み
- [ ] `ml_lethal` vs `ml_lethal_estimated`（hidden info単独）の結果が `results/` に記録済み
- [ ] 明示的な判断ルールに基づく結論が `results/` の summary に記載済み
- [ ] `ml_policy_agent.py`/`selector.py` への累積の追記内容を tsuoimorikaさんに共有・合意（Stage1からの持ち越し。本計画完了時にまとめて共有を推奨）

この結果次第で「提出する（→切り出し・統合フェーズへ）」「hidden infoだけ提出」「v1へ研究続行」「方向転換」のいずれかが、感覚ではなく数値で選べるようになる。
