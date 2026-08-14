# Engine / SDK 改善要求（Step 1-34）

Step 1-1 〜 1-33 の実測に基づく、native engine API への改善要求。
**推測は含まない。** すべて `cg.dll` の export 解析と実測値が根拠。

現在の export は 13 個のみ:

```
GameInitialize BattleStart BattleFinish GetBattleData Select AgentStart
VisualizeData AllCard AllAttack SearchBegin SearchStep SearchEnd SearchRelease
```

---

## Priority 1: State Clone / Snapshot & Restore

| | 内容 |
|---|---|
| **現在の制約** | 探索状態を複製・保存・復元する入口が無い。chance outcome を作るには毎回 `SearchBegin(your_deck=<順序>)` からやり直し、prefix を replay する必要がある |
| **必要な API** | `SearchClone(searchId) -> newSearchId`、または `SearchSnapshot(searchId) -> blob` / `SearchRestore(blob) -> searchId` |
| **測定根拠** | `SearchBegin` + 初期化/解放 = **0.243 ms**、`search_step` = 0.093 ms。prefix 長 3 の materialization = 0.522 ms のうち **`SearchBegin` が 46%** |
| **期待効果** | outcome あたり **0.61 ms → 約 0.1 ms（4〜5 倍）**。`max_outcomes` を上げても他の枝の budget を圧迫しなくなる（現在は 24 → 512 で `PROVEN_WIN` が 104 → 98 と減少する） |
| **研究上できるようになること** | 大規模 chance branching（65〜4,066 outcome）の scalability を初めて実測できる。現在は上限 24 で入口を閉じているため未測定 |

> 改善余地は **測定した component cost からの見積もり**であって、
> 「実装できる」という主張ではない。

---

## Priority 2: Deterministic Outcome Injection

| | 内容 |
|---|---|
| **現在の制約** | `search_step(search_id, select)` の引数は**選択肢インデックスのみ**。「どのカードを引くか」「コインの表裏」を指定できない。outcome を制御する唯一の入口が `search_begin` の `your_deck` の並び |
| **必要な API** | `SearchStep(..., forced_draw=[cardId...])` / `SearchStep(..., forced_coin=[bool...])` のように、chance の結果を明示指定できる引数 |
| **測定根拠** | Step 1-32: 同一 parent から `search_step` を 3 回連続実行して parent は生存する（branch 自体は可能）。しかし**分岐先で何を引くかは `search_begin` 時点で決まっている**ため、共通 prefix から複数 outcome へ分岐できない |
| **期待効果** | Priority 1 と同等。prefix replay が 1 回で済む |
| **研究上できるようになること** | B8（多コイン）の outcome 列挙が現実的になる。現在は capability 確認のみで実装保留 |

---

## Priority 3: Shared Prefix Branching

| | 内容 |
|---|---|
| **現在の制約** | 同時に 2 つの探索セッションを保持できない（`SessionNestingError: agent_ptr is shared`）。「parent を保持したまま別 outcome を構築する」実装が不可能 |
| **必要な API** | 同一 native state から複数 child へ分岐できる入口。Priority 1 か 2 のどちらかがあれば代替可能 |
| **測定根拠** | Step 1-32 の実測。`agent_ptr` が共有されているため、ネストしたセッションは例外になる |
| **期待効果** | Priority 1 / 2 と同じ |

---

## Priority 4: RNG Seed / State Control

| | 内容 |
|---|---|
| **現在の制約** | seed 設定・RNG 状態の取得/設定の入口が無い。`BattleStart` の引数はデッキ 120 枚のみ。同一デッキ・同一固定方策でも対局は再現しない（別プロセスの「最初の 1 戦」でも 4/4 が別軌跡） |
| **必要な API** | `BattleStart(cards, seed)`、または `GetRandomState` / `SetRandomState` |
| **測定根拠** | Step 1-16。export 13 個に該当なし。供給デッキを変えても root listing が 12/12 で不変 = 実デッキ由来 |
| **期待効果（性能ではなく評価）** | ・reproducible battle<br>・**paired A/B が初めて成立する**（現在は seed 制御も state fork も無いため、対応比較が原理的に不可能）<br>・reproducible benchmark<br>・**`RNG_NON_INTERFERENCE_VERIFIED` を検証可能にする** |
| **研究上できるようになること** | **これが無い限り Phase 1/2 の本番投入判断ができない。** 「探索の有無だけで本番 random outcome が変化しないこと」を証明する手段が存在しないため、production safety は永久に NOT VERIFIED のまま |

> Priority 4 は性能ではなく **評価可能性**の問題。
> 性能改善は Priority 1〜3、本番採用判断の前提条件は Priority 4。

---

## まとめ

| API feature | current | requested | expected benefit |
|---|---|---|---|
| State Clone / Snapshot | **unavailable** | `SearchClone` / `SearchSnapshot` + `SearchRestore` | materialization 0.61ms → 約 0.1ms（4〜5 倍） |
| Outcome Injection | **unavailable** | `SearchStep(..., forced_draw/forced_coin)` | 同上。B8 の実装が現実的になる |
| Shared Prefix Branching | **unavailable** | 同一 native state からの多分岐 | 同上（1 or 2 で代替可） |
| RNG Seed / State Control | **unavailable** | `BattleStart(cards, seed)` / RNG state R/W | paired A/B の成立、production safety の検証 |

**Priority 1〜3 が無い限り、Phase 2 の大規模 chance branching は
solver 側の工夫では改善できない**（Step 1-31 で `max_outcomes` の
拡大が逆効果であることを実測済み）。
