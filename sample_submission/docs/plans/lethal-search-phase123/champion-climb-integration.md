# Phase 1/2 リーサル探索器 × champion climb(フーディンBC→RL)の統合

`feature/lethal-phase2-alakazam-climb` ブランチの内容。`integration` で作った
Phase 1/2 確定リーサル探索器（`ptcg_ai/search/lethal/`）と、
`feature/gen2-bc-data-update` の **champion climb**（Plan A アラカザム(フーディン)デッキ +
BC→PPO ポリシー、Kaggle publicScore 823.5）を1つのブランチに載せた。

---

## 1. 何をどう繋いだか

| 層 | 変更 |
|---|---|
| `ml_policy/ml_policy_agent.py` | `_SEARCH_MODULES` に `lethal_phase1`（= `search/lethal/entry`）を登録。`_try_lethal` の context に `full_deck` を追加（Phase 2 の outcome 列挙に必要。無いと必ず `UNKNOWN`） |
| `action_selection/selector.py` | マージ解決。`lethal_simple` / `lethal_phase1` / `pimc` の3モジュールを登録し、gen2 側の `hidden_state_source`（dummy / estimated）切り替えを維持したまま `full_deck` を渡す |
| `configs/abl_5_full_lethalphase2.json` | 新規。champion の `abl_5_full` の `pipeline` セクションはそのまま、`lethal_search` だけ Phase 1/2 に差し替えた opt-in config |
| `ptcg_ai/learning/policy_weights.json` | 当初は champion の `policy_weights_alakazam_rl_climb.json`（sha256 `395b0248…`）。**2026-08-14 に 2026-08-02 生成の再学習BC `policy_weights_alakazam_bc2.json`（sha256 `8f5b4bf6…`、RL なし、同一 test split の top1 0.5778→0.6643）へ差し替え。** climb 側の重みはファイルとして残してあるので復帰可能 |
| `deck.csv` | 変更不要。マージ後の時点で既に Plan A アラカザム(フーディン)デッキ（`models/climb_lb823/deck.csv` と改行を除いて一致） |

意思決定パイプラインは champion と同じ 3 段のままで、1段目だけが入れ替わる:

```
(1) 確定リーサル探索   lethal_simple  →  lethal_phase1(Phase 1 → Phase 2)
(2) 浅い PIMC 前読み   （変更なし: top_k=4 / N=8 / opponent_depth=1 / 400ms）
(3) Policy フォールバック（変更なし: climb 重み）
```

## 2. 既定 = 提出構成（2026-08-14 に切り替え）

`ml_policy_agent._CONFIG_NAME` の既定値を `abl_5_full_lethalphase2` にしてある。
Kaggle 提出時は環境変数を設定できないため、提出したい config はここに直接書く必要がある。
**この状態で tar を作れば Phase 1/2（Phase 2 含む）が有効なまま提出される。**

ローカルで別 config を試すときだけ環境変数で上書きする:

```bash
PTCG_AI_ML_CONFIG=abl_5_full python league/run_league.py    # 従来の champion(lethal_simple)
```

元に戻すのは `_CONFIG_NAME` の既定値を `"abl_5_full"` に書き戻すだけ。

### 承知の上での未検証項目

`design.md` 付録 Z.1 の production 接続条件6つのうち、5つ（誤 `PROVEN_WIN` 0 / 不正行動 0 /
replay 検証 100% / 失敗時 fallback 100% / リソースリーク 0）は満たすが、
`RNG_NON_INTERFERENCE_VERIFIED` は **False** のまま。「探索の有無だけを変えて本番の乱数結果を
比較する」手段（seed 設定 / state clone / RNG 状態取得）が SDK に無いため *not verified* で、
証明ロジック側に問題が見つかったわけではない（付録 Z.1 / `step0-capability-report.md` §11）。

なお `rule_based` 側の既定（`configs/rule_lethal.json`）は `lethal_simple` のままで、
`test_feature_flag_safety.py::test_default_config_never_invokes_phase1` がそれを固定している。

### 時間予算

Phase 1/2 が動くのは precheck を通った局面（残りサイド ≤ 2 など）だけで、その場合の
1手あたり最悪時間は `phase12_ms=400` + pipeline（動的予算、`max_ms=2000`）。
疎通確認した2試合では 123 select 中 8 回の発火だった。

## 3. パラメータの根拠

`abl_5_full_lethalphase2.json` の `lethal_search` は、champion 側の値ではなく
**Phase 1/2 が実測された範囲**（`rule_lethal_phase1.json` / 付録 W）に合わせてある。

- `max_remaining_prizes: 2` — champion の `lethal_simple` は 3 だが、Phase 1/2 の
  能力コーパスは 2 で測っている。3 へ上げるのは再測定してから。
- `phase12_ms: 400`（2026-08-14 に 500 から変更） — champion の lethal 段は 100ms
  だったので、1手あたりの最悪時間は +300ms 増える（`pipeline.time_budget` の
  `max_ms: 2000` の範囲内）。500 → 400 で確定数がどう変わるかは下記の実測どおり。
- `max_depth: 8` / `max_chance_depth: 1` — 実測 baseline のまま。

### `phase12_ms` 500 → 400 の実測（能力コーパス 205 局面 / valid 185）

**本番と同じ予算配分**（Phase 1 が先に走り、Phase 2 は `phase12_ms - Phase1経過` で走る。
`entry._run`）で比較した結果:

| subset | 400ms 確定 | 500ms 確定 | 失った確定 |
|---|---|---|---|
| exact-positive (n=80) | 64 | 64 | **0** |
| chance-positive (n=40) | 23 | 23 | **0** |
| all-valid (n=185) | 86 | 88 | **2**（いずれも Phase 1 側） |

主評価の exact-positive では損失なし。all-valid で減った 2 件は Phase 1 の深い証明が
時間切れになったもので、Phase 2 の追加確定は 400/500 とも 1 件で変わらない。

### 注意: 共有予算だと Phase 2 はほとんど時間をもらえない

付録 W の「Phase 2 が Phase 1 に無い確定を +9 件」は、**各 phase に満額の予算を与えた**
能力比較（`tools/compare_phases.py`）の値。本番の `entry._run` は `phase12_ms` を
両者で分け合うため、Phase 1 が証明に失敗する局面ではほぼ全額を使い切り、Phase 2 に
残るのは数 ms しかない。上表のとおり **本番配分での Phase 2 の追加確定は 1 件**に留まる。

Phase 2 の能力を活かすなら Phase 1 の取り分に上限を設ける（例: `phase1_ms` を別キーで
切る）改修が要る。現状はその改修を入れていないので、Phase 2 の寄与は限定的と見るべき。

## 4. 既知の未解決事項（コーパスとデッキの不一致）

`tests/fixtures/lethal_capability.jsonl` / `lethal_positions.jsonl` は
**マージ前のデッキで採取した局面**なので、Plan A アラカザムデッキと組み合わせると
`tests/integration/lethal/` の 9 テストが落ちる:

- `test_capability_matrix.py`（golden 再現 / false PROVEN_WIN / replay）3件
- `test_deck_belief_and_listing.py` 5件
- `test_entry_harness.py::test_proven_win_chains_to_an_actual_engine_win` 1件

マージ前のデッキに戻すと 38/38 green になることを確認済みで、**探索器のロジック回帰ではない**。
`read_deck_csv()`（=現在のデッキ）と記録済み局面が食い違うことによる失敗。

解消するには新デッキでコーパスを採り直す:

```bash
python tools/build_capability_corpus.py
```

これは付録 W の凍結 baseline（`max_outcomes = 24`、Phase 2 の追加確定 9 件 など）も
新デッキ基準で取り直すことを意味するため、実行するかどうかは測定方針の判断が要る。

## 5. 疎通確認

`abl_5_full_lethalphase2` の ml_policy エージェント vs champion（`abl_5_full`）で 2 試合:

- エラー 0 / 不正選択 0、36 手・185 手で正常終了
- Phase 1/2 探索器の発火 8 回（precheck で見送り 115 回）、`PROVEN_NO_WIN` 5 / `UNKNOWN` 3、
  誤 `PROVEN_WIN` 0

勝率の評価には足りない試合数。強さの比較は league の gauntlet で別途。

### 提出パッケージ単体での動作確認

README の手順で作った tar を展開し、**展開先だけを `sys.path`/cwd にして** `main.agent` を呼び、
リポジトリ本体に依存していないことを確認した（`decks/` 入れ忘れのような事故の検出）:

```bash
cd sample_submission
tar -czf submission.tar.gz main.py deck.csv cg ptcg_ai decks configs
```

- 展開先で読まれた config = `abl_5_full_lethalphase2`（`module=lethal_phase1` / `phase2=True`）
- Policy 重みロード `is_ready=True`、デッキ選択 60 枚、通常 select は範囲内の有効な選択
- `ptcg_ai/search/lethal/` は stdlib + `cg.api` + `ptcg_ai` 内部しか import しない
  （`tools/` や `tests/` に依存しないので tar に入れる必要はない）
