# 調査書: Open #9 — ml_policy own-deck / belief 依存追跡と deck injection 可否

作成日: 2026-07-26
種別: **調査書(read-only 実コード追跡・設計判断)**。実装・コード変更なし。
前提: [reference-pool-field-gate-design.md](./reference-pool-field-gate-design.md) の Open #9(field opponent の deck.csv belief 制約)。
ステータス: **調査のみ。production/cg/main.py/deck.csv/weights/configs/shared league/measurement harness 変更なし。**

---

## 1. Problem

Field Evaluation では `battle_start(deck0, deck1)` で opponent に別デッキ(例 crustle)を渡す。だが ml_policy は
自分の 60 枚を **cwd の `deck.csv`** から取得し belief/hidden-information/search に使う。opponent に別デッキを持たせても
内部 own-deck が `deck.csv`(Champion=alakazam)のままなら「実対戦 deck=crustle / 内部 belief=alakazam」の不整合が起き、
その opponent の意思決定(自山札推定・サイド落ち・search・value/policy 特徴)が壊れ、Field Gate 結果を信用できない。

**結論(先出し)**: 問題は **opponent が「別デッキ」かつ「own-deck を意思決定に使う」場合のみ**。**Candidate/Champion は
deck.csv を実際に使う=own-deck 正しい**。opponent を **policy-only ml_policy** か **dragapult_rule(自前デッキ)** にすれば
**own-deck を意思決定に使わない/正しく読む**ので、**production 変更なしの measurement-only で v1 は実装可能**(下記)。

---

## 2. deck.csv reference inventory(read-only grep + 到達経路追跡)

| file:func | deck.csv を直接読むか | 呼び出し元 | cache | 意思決定影響 | 別deck時の危険度 |
|---|---|---|---|---|---|
| `rule_based_agent.read_deck_csv()` | **YES**(cwd deck.csv、fallback /kaggle_simulations) | 下記全て | ― | 起点 | ― |
| `ml_policy_agent._get_deck()` | 経由(read_deck_csv) | `_model_hidden_state_factory`/`_try_lethal`(dummy)/`_try_attack_plan`(dummy) | **`_deck_cache`(単一プロセスglobal, reset無)** | dummy hidden state の your_deck | 高(単一globalで per-player 不可) |
| `match_context._load_own_deck_ids()` | 経由 | `get_own_state`/`update` | **`_own_states[player_index]`(player別dict, reset は obs.select=None時のみ)** | estimated hidden state の your_deck | 高だが **player別=注入余地** |
| `rule_based_agent.get_full_deck()` | 経由 | rule_based lethal stub | `_DECK_CACHE` | rule_based 側 | (opponent=rule_based時) |
| `own_hidden_state.OwnHiddenState(deck_card_ids)` | 受領 | match_context | 対局内 state | **prize/サイド落ち/自山札推定(zone_math)** | Critical |
| `search_adapter.to_search_begin_kwargs` | own_state.sample→your_deck | lethal/pipeline/attack_plan(estimated) | ― | search 決定化の your_deck | Critical |
| `build_dummy_search_state(obs, full_deck)` | full_deck 受領 | lethal/pipeline/attack_plan(dummy) | ― | search の your_deck | Critical |
| `search/{lethal_simple,pimc,pipeline,attack_plan}` | hidden_state["your_deck"] | ― | ― | 詰み探索/決定化の自山札 | Critical |
| `board_evaluation/consequence.py` | hidden["your_deck"] | PolicyModel consequence 特徴 | ― | option 特徴(**既定重みは未使用**、§7) | 条件付 |
| `encoder.py "self_deck_count"` | **NO**(State の deckCount=枚数、identity非依存) | policy 特徴 | ― | 枚数のみ正しい | 無 |

---

## 3. Dependency graph(production ml_policy full pipeline)

```
deck.csv (cwd)
  └ read_deck_csv()
      ├ ml_policy._get_deck()  ──[_deck_cache 単一global]──┐
      │                                                     ├→ build_dummy_search_state(full_deck) → your_deck
      │   (lethal_search "dummy" / policy_model "dummy")    │      → lethal_simple / pimc / pipeline / attack_plan / consequence
      └ match_context._load_own_deck_ids()                  │
          → OwnHiddenState(deck_ids) [_own_states[pidx]] ───┘  (estimated)
              → search_adapter.sample() → your_deck → 同上 search
```
- **hidden_state_source が分岐点**(config): `"estimated"`→match_context(**player別**) / `"dummy"`(既定)→`_get_deck()`(**単一global**)。
- **abl_5_full の実設定**(確認済): `pipeline.hidden_state_source="estimated"`(match_context) ＋ `lethal_search`は hidden_state_source 無=**"dummy"**(`_get_deck()`)。→ **両経路を併用**。

---

## 4. Lifecycle / caching(重要)

- `_deck_cache`(ml_policy): **プロセス内で1度だけ**(初回 `_get_deck()`)、以降不変。**per-match reset 無し**。単一値=player 区別なし。
- `match_context._own_states/_opponent_states/_knowledge`: **player_index 別 dict**。`reset()` は公開だが **`update()` 内では
  `obs.select is None`(デッキ選択ターン)でのみ呼ばれる**。**ローカル `battle_start(deck0,deck1)` はデッキ選択ターンを通らない**
  (module docstring 明記)→ **measurement harness では match_context が games を跨いで reset されない**(実測: measurement/*.py は
  `match_context.reset()` を呼んでいない)。→ **games/strata 間で opponent modeling・own state が carryover(§9)**。
- `_model_cache_by_weights_path`/`_model`: weights 別/既定モデルを使い回し(deck とは無関係、問題なし)。

---

## 5. Affected modules 分類

| module | own-deck identity 依存 | severity |
|---|---|---|
| `OwnHiddenState`(自山札残/サイド落ち zone_math) | **YES**(`deck_card_ids` で pool 構成) | **A. Critical** |
| search(`lethal_simple`/`pimc`/`pipeline`/`attack_plan`) via `your_deck` | **YES**(決定化・詰み探索の自山札) | **A. Critical** |
| `consequence` 特徴(hidden["your_deck"]) | 条件付(**既定重みは consequence_fields 空=未使用**、§7) | B→実質 C(既定重み) |
| PolicyModel scoring(state/option 特徴) | **NO**(deckCount 枚数のみ、identity 非依存) | **C. Unaffected** |
| opponent modeling(相手デッキ予測) | own-deck 非依存(観測カードから予測)。ただし carryover 影響(§9) | 別問題 |

---

## 6. Observation からの own-deck 復元可否

`OwnHiddenState` は初期 60 枚(`deck_card_ids`)を要する。Observation/State には **自分の初期 60 枚 identity は含まれない**
(hand + board + discard + prize は見えるが、**deck は枚数のみ・prize は伏せ**)。`search_state_stub._visible_own_card_ids` も
「見えている自カード」しか復元しない。→ **「環境から自動取得すればよい」は不可**。own-deck は外部(deck.csv 等)から与える必要がある。

---

## 7. policy-only(abl_2_policy_only)は安全か → **Safe(既定重み)**

- config: `lethal_search.enabled=false` / `pipeline.enabled=false`。→ `_try_lethal`/`_try_pipeline` は即 None。
- 経路: `_select_action` → `model.select_option(obs, model_factory, deadline)`。`model_factory=_model_hidden_state_factory(obs,config)`
  は **"dummy"** なので内部で `full_deck=_get_deck()` を **eager に呼ぶ**(=`_deck_cache` を populate)。
- **だが結果 lambda(build_dummy_search_state(full_deck))は既定重みでは呼ばれない**: `PolicyModel.score_options` は
  `if self._consequence_fields:` の時だけ factory を呼ぶ(policy_model.py:165)。**既定 `policy_weights.json` の meta に
  `consequence_fields` 無し(実測: meta keys に無い)→ `_consequence_fields=[]`→factory 未呼出**。
- ∴ policy-only は **`_get_deck()` を呼ぶ(副作用のみ)が own-deck を意思決定に使わない**。match_context も `update()` で計算
  されるが policy scoring は参照しない。→ **DECISION は own-deck 不整合の影響を受けない = Safe**(既定重み前提。consequence 特徴
  入り重みを使う場合は Unsafe に変わるので注意)。

## 7b. Candidate/Champion(abl_5_full)は安全か → **Safe(自分は deck.csv を実際に使う)**

Candidate は `deck.csv`(alakazam)を実プレイ。`_get_deck()`=deck.csv・`match_context.get_own_state(自idx)`=deck.csv=**正しい**。
∴ Candidate/Champion 側は own-deck 整合。**field opponent が別デッキの時だけが問題**。

---

## 8. dedicated agent 分析

| agent | 自 deck の取得 | deck.csv/match_context/_get_deck | 判定 |
|---|---|---|---|
| **dragapult_rule** | `os.path.join(module_dir, "dragapult_ex_deck.csv")`(module 相対、cwd 非依存) | **使わない**(grep 0 件) | **Safe**(自前 deck、Field runner から `--deck-b` に同 CSV を渡す) |
| `rule_based`(汎用) | `read_deck_csv()`(cwd deck.csv) | 使う | 別デッキ時 Unsafe(own-deck=deck.csv)。ただし弱く pool 不採用 |
| `first_choice`(measurement) | deck 非依存(先頭 minCount 選ぶだけ) | 使わない | Safe だが calibration 専用 |
| ml_policy heavy(lethal/pipeline) on 別デッキ | `_get_deck`/match_context=deck.csv | 使う | **Unsafe**(own-deck 誤り) |

---

## 9. parallel worker implications(carryover)

- 現 measurement worker: init で agent 1回 build → 複数 game 再利用。**match_context は module global で games を跨いで reset されない**
  (§4)。→ game1(crustle)の opponent-modeling/own-state が game2(marnie)へ carryover=**誤り**。同一 stratum(同デッキ)なら被害小、
  strata 切替で顕著。
- **fix(measurement 側・production 非改変)**: **field runner が各 game 前に `match_context.reset()` を呼ぶ**(既存 public API)。
  加えて ml_policy pipeline カウンタ(`_match_start_perf`/`_selects_seen`)は own-deck 非依存だが、厳密には match 境界で新試合として
  扱うのが望ましい(agent は `obs.select is None` で reset するが local は通らない=measurement 側で対処)。
- **worker 固定方針(推奨)**: stratum(opponent+deck)ごとに worker を固定 or task を stratum 連続にし、model は再利用・
  match_context のみ per-game reset。model 再 build は不要(deck.csv=Candidate 側は不変、opponent policy も weights 再利用)。
- **Track A への遡及注意**: Track A も match_context reset 無しで実行(carryover 有)。ただしミラー(両者 deck.csv 同一)で
  own-deck は両者正しく、opponent-model は同デッキ観測の蓄積で概ね整合 → 決定的 PROMOTE 結論は保つと判断。**field eval は
  必ず per-game reset**。

---

## 10. measurement-only injection options 評価

| option | 内容 | 可否 |
|---|---|---|
| **A. config で注入** | `config["own_deck_ids/path"]` を既存コードが読む | **不可**(そんな config キーは存在しない。`_get_deck`/`_load_own_deck_ids` は引数を取らない) |
| **B. match_context を pre-populate** | game 前に `match_context._own_states[opp_idx]=OwnHiddenState(opp_deck)` を注入 | **部分可**: pipeline(estimated)経路は player別 `_own_states` なので効く。**だが lethal(dummy)経路は `_get_deck()`=単一global で効かない**。両経路併用の heavy opponent には不十分。private global 直叩き=脆い |
| **C. worker ごとに仮想 cwd/deck.csv 切替** | worker cwd=対応デッキの dir | **不可(head-to-head では)**: Candidate と opponent が**同一プロセス**で走り cwd は1つ。両者に別 deck.csv を同時に読ませられない。かつ `_deck_cache` は単一 global で先に読んだ deck を共有 |
| **D. monkey patch** | `_get_deck`/`_load_own_deck_ids` を差し替え | 技術的には可だが **どの agent の呼び出しか区別できない**(単一 global)ため per-player 注入は不可。信頼性を下げる=非推奨 |

**結論**: heavy opponent を「別デッキで own-deck 正しく」measurement-only 注入するのは **B/C/D いずれも同一プロセス head-to-head の
単一 global 制約で困難**。→ **heavy opponent を使わない設計(policy-only + dragapult_rule)にするのが唯一の measurement-only 解**。

---

## 11. Recommended solution(measurement-only、production 非改変)

**Field opponent を「own-deck を意思決定に使わない/自前 deck を正しく読む」agent に限定する**:
- **policy-only ml_policy(abl_2_policy_only, 既定重み)** on 各 archetype deck(crustle/rocket_mewtwo/marnie/lucario/archaludon)
  = §7 で Safe(own-deck 未使用)。opponent は engine が配る archetype deck のカードを state ベース policy で普通にプレイ。
- **dragapult_rule** = §8 Safe(自前 deck)。
- **Candidate/Champion = abl_5_full(deck.csv)** = §7b Safe。
- **field runner が各 game 前に `match_context.reset()`**(§9、public API、measurement 側)。
- **production/configs/weights/shared/deck.csv 変更ゼロ。deck injection 不要。**

**Reference Pool v1(6 メンバー)はこのまま実装可能**(Open #9 解除)。ただし **限界**: opponent は policy-only=弱 AI。
field eval は「Candidate vs 弱-policy-archetype」を測る(強 AI 相性ではない)=既知の worthy-opponent ギャップ。デッキ相性の
robustness は測れる。

---

## 12. Fallback solution(強 heavy opponent が必要になった場合のみ・最小 production 変更・未実装)

policy-only では不十分(archetype を強くプレイする opponent が欲しい)なら、**own-deck の explicit injection point を production に
1つ足す**のが最小案(**今回は実装しない・要担当者確認**):
- 案: `ml_policy_agent.agent(obs, config=None, own_deck_ids: list[int] | None = None)` を追加、`_get_deck`/match_context の
  own-deck をこの引数で上書き可能に(**default None → 現行 deck.csv fallback = byte/意味等価**)。config 経由(`config["own_deck_ids"]`)でも可。
- 要件: 既定 production 挙動は完全維持 / optional 注入時のみ新挙動 / test 追加可 / measurement・self-play で再利用可 / 変更範囲最小。
- 影響ファイル(**変更が必要なら停止・報告**): `ptcg_ai/ml_policy/ml_policy_agent.py`(_get_deck/agent、production)、
  `ptcg_ai/hidden_information/match_context.py`(get_own_state に deck 注入、production)。→ **他者/production ownership。要担当者確認。**

---

## 13. Ownership concerns

- **推奨解(§11)は production/shared を一切触らない**(measurement 側で安全な opponent 選択 + public `match_context.reset()`)。
- Fallback(§12)を採る場合のみ `ml_policy_agent.py` / `match_context.py`(production・他者担当の可能性)を触る → **実装前に担当者確認、勝手に変更しない**。

---

## 14. Impact on Reference Pool v1 / Open #9 blocker

- **Open #9 は blocker 解除**(measurement-only で v1 実装可)。
- 6 メンバー: dragapult_rule(強・Safe) + crustle/rocket_mewtwo/marnie/lucario/archaludon(policy-only・Safe・弱 AI)。
- 設計 §11 の Δ_field/guard/予算はそのまま有効。opponent が policy-only=弱い点は「デッキ相性 robustness を測る」用途として整合。
- **要検証(実装時)**: `_matrix.json` の既存 opponent が policy-only か heavy か(heavy なら過去 matchup 数値に own-deck 汚染の疑い。
  ただし v1 は自前で policy-only opponent を回すので新規に正しく測り直す)。

---

## 15. Open decisions(実装前にユーザー確定)

1. Field opponent を **policy-only + dragapult_rule に限定**する(推奨・measurement-only)か、heavy opponent のために
   **Fallback §12(最小 production 変更・要担当者確認)**に進むか。
2. per-game `match_context.reset()` を field runner に入れる(推奨・measurement 側)。
3. worker 割当(stratum 固定 or per-game reset)。
4. `_matrix.json` opponent の config 種別確認を実装時に行う(過去数値の扱い)。

---

## 15b. 更新(2026-07-26): match_context carryover 修正 + Track A 再検証【解決済】

§9 の carryover を実測・修正・再検証した(measurement のみ、production 非改変)。

- **carryover 実証(Step 0)**: 現行 runner で `OwnHiddenState` が game 間で **同一 id=再利用**(reset されず残留)、
  deck_count も継続。`match_context.reset()` で作り直しを確認(`_diag_carryover.py`)。
- **修正**: `runner.play_game` 冒頭(battle_start 前)に `match_context.reset()`(既存 public API)を追加(`_reset_match_state`)。
  serial/parallel とも play_game を通るため同一契約。**Field runner はこの runner.play_game を再利用するので毎 game reset される**。
- **契約(Field runner 必須)**: `worker init → model/agent build → [各 game: match_context.reset() → battle_start → loop → finish]`。
  model/weights は worker 再利用、match_context のみ per-game reset。
- **副次修正**: `run_sprt_ab` は `resume=False` でも既存 outcomes.jsonl に追記していた footgun → resume=False 時 truncate。
- **テスト**: `test_reset.py` Test A-E(carryover 解消 / player0-1 両クリア / **`_config_cache`/`_model`/`_model_cache_by_weights_path`/
  `_deck_cache`/`_predictor` 非影響** / serial reset / parallel integrity)。全 23 measurement テスト green(desktop でも 5/5)。
- **Track A 再裁定(reset 入り、desktop workers=15、A-0 同期確認済)**: **PROMOTE @N=199, winrate 0.638, CI[0.569,0.702], errors 0/0,
  integrity 全 OK**。旧(no-reset)PROMOTE @217/0.627 と**実質同一**(win-condition・P0/P1 も同傾向)。→ **前回 PROMOTE は carryover artifact でない。
  Champion(abl_5_full)採用根拠を reset 契約込みの正式結果として維持**。Open #9 は解決、Field Gate へ進行可。

## 16. 今回やらないこと
deck injection 実装 / field runner / Reference Pool manifest / Gate2 / Champion manager / self-play / 新 agent / 新 deck /
production 改変 / measurement 改変 / shared league 改変。**本調査は read-only + doc のみ。** production/cg/main.py/deck.csv/
weights/configs/shared/他者コード 変更なし。git add/commit/push なし。
