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
| `ptcg_ai/learning/policy_weights.json` | champion の `policy_weights_alakazam_rl_climb.json` に差し替え（sha256 `395b0248…` = `models/climb_lb823/MANIFEST.json` の提出時パッケージと一致） |
| `deck.csv` | 変更不要。マージ後の時点で既に Plan A アラカザム(フーディン)デッキ（`models/climb_lb823/deck.csv` と改行を除いて一致） |

意思決定パイプラインは champion と同じ 3 段のままで、1段目だけが入れ替わる:

```
(1) 確定リーサル探索   lethal_simple  →  lethal_phase1(Phase 1 → Phase 2)
(2) 浅い PIMC 前読み   （変更なし: top_k=4 / N=8 / opponent_depth=1 / 400ms）
(3) Policy フォールバック（変更なし: climb 重み）
```

## 2. 既定では OFF

`design.md` 付録 Z.1 の production 接続条件のうち `RNG_NON_INTERFERENCE_VERIFIED` が
**False** のままなので、既定 config（`abl_5_full`）は従来どおり `lethal_simple` を選ぶ。
Phase 1/2 を使うのは明示的に config を切り替えたときだけ:

```bash
PTCG_AI_ML_CONFIG=abl_5_full_lethalphase2 python league/run_league.py
```

Kaggle 提出でこれを既定にする場合は `ml_policy_agent._CONFIG_NAME` の既定値を
`abl_5_full_lethalphase2` に変える（環境変数は提出時に設定できないため）。
その際は付録 Z.1 の未検証項目を承知の上で行うこと。

## 3. パラメータの根拠

`abl_5_full_lethalphase2.json` の `lethal_search` は、champion 側の値ではなく
**Phase 1/2 が実測された範囲**（`rule_lethal_phase1.json` / 付録 W）に合わせてある。

- `max_remaining_prizes: 2` — champion の `lethal_simple` は 3 だが、Phase 1/2 の
  能力コーパスは 2 で測っている。3 へ上げるのは再測定してから。
- `phase12_ms: 500` — 付録 W の Phase 2 能力（Phase 1 に無い追加確定 9 件）を測った予算。
  champion の lethal 段は 100ms だったので、1手あたりの最悪時間は +400ms 増える
  （`pipeline.time_budget` の `max_ms: 2000` の範囲内）。
- `max_depth: 8` / `max_chance_depth: 1` — 実測 baseline のまま。

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
