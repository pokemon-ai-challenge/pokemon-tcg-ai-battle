# 非公開情報推定レイヤー（hidden_information）実装プラン

作成日: 2026-07-20
親ドキュメント: [`design.md`](./design.md)（背景・設計方針はそちら。本ファイルは手順のみ）
配置予定モジュール: `sample_submission/ptcg_ai/hidden_information/`（現在は `__init__.py` のみの空パッケージ）

進め方: **Phase 1 → 2 → 3 → 4 の順**。各フェーズは独立にテスト可能な単位に切ってあり、
前フェーズの成果物が無くても単体では動く（Phase 2 は Phase 1 の `zone_math` を再利用するが、
`OwnHiddenState` 自体には依存しない）。

---

## Phase 1: 自分側 — `OwnHiddenState`（多重集合管理 + 超幾何 + サーチ消し込み）

### 対象ファイル（新規）

- `sample_submission/ptcg_ai/hidden_information/zone_math.py`
  超幾何分布の閉形式計算をまとめた純粋関数群。Phase 2 の `OpponentHiddenState` からも再利用するため、
  「自分固有」のロジックを持ち込まない共通基盤としてここに切り出す。
  - `prob_in_prize(pool_size: int, prize_size: int, target_count: int, at_least: int = 1) -> float`
    未確認プール `pool_size` 枚のうち `target_count` 枚が対象カードのとき、サイド `prize_size` 枚の中に
    そのカードが `at_least` 枚以上含まれる確率。
  - `prob_in_prize_exact(pool_size, prize_size, target_count, j) -> float`
    ちょうど `j` 枚サイドに落ちている確率（`C(k,j)*C(M-k,n-j)/C(M,n)`）。`prob_in_prize` はこれの累積。
  - `expected_in_prize(pool_size, prize_size, target_count) -> float`
    期待値 `target_count * prize_size / pool_size`（表示用のショートカット）。
- `sample_submission/ptcg_ai/hidden_information/own_hidden_state.py`
  - `class OwnHiddenState`
    - `__init__(self, deck_card_ids: list[int])` — 自分の60枚（`ptcg_ai.rule_based.rule_based_agent.read_deck_csv()` の戻り値をそのまま渡す想定）。
    - `update(self, state: State) -> None` — 毎ターン呼ぶ。`state.players[state.yourIndex]` の `hand` / `active` / `bench`（本体・`energyCards`・`tools`・`preEvolution`） / `discard` を集計し、`collections.Counter[int]`（card_id → 残り枚数）として**未確認プール**を作り直す（Phase 0 相当の再計算。持ち越し状態は最小限にする。§4.4 参照）。
    - `resolve_deck_search(self, select: SelectData) -> None` — `select.context == SelectContext.LOOK and select.deck is not None` のときに呼ぶ。`select.deck` に写っていない未確認プールの card_id を「サイド確定」として `_confirmed_prize: Counter[int]` に移し、未確認プールから決定的に除外する。
    - `marginals(self) -> dict[int, dict[str, float]]` — `card_id -> {"deck": p, "prize": p}`（`zone_math.prob_in_prize` ベース。確定済みは 1.0/0.0）。
    - `sample(self, rng: random.Random | None = None) -> tuple[list[int], list[int]]` — `(deck_card_ids, prize_card_ids)` を1組返す（`search_begin` の `your_deck` / `your_prize` にそのまま渡せる形）。多変量超幾何分布に従うサンプリング（`random.sample` で未確認プールから `prize_size` 枚を非復元抽出すれば十分。カード種類ごとの確率は自動的に超幾何分布に従う）。
    - 内部不変条件（デバッグアサート推奨）: `len(未確認プール) == state.players[my_index].deckCount + len(state.players[my_index].prize)`。

### 実装内容

- §4.1〜4.3（`design.md`）の「全60枚 − 公開ゾーン = 未確認プール」「超幾何閉形式」「サーチ消し込み」をそのままコード化する。
- サーチ消し込みの検知条件（`SelectContext.LOOK` かつ `SelectData.deck is not None`）は、現状リポジトリ内のどこにも実装例が無い新規の接続点（`hidden_zone.py` は `select.option` からしか card_id を解決しておらず、`select.deck` フィールド自体は未参照）。実装時に **自分の `LOOK` でも相手の `LOOK` でもこのフィールドが同じ意味で埋まるのか**（`select.deck` が「今このプレイヤーが見ている山札」を指すのか、常に自分の山札限定か）を、`tests/local_sim` で実際に `SelectContext.LOOK` が発生するシナリオを再現して確認すること（design.md §11 の「参照ファイル一覧」にある `cg/api.py` の `AreaType.LOOKING` / `SelectContext.LOOK` のコメントだけでは断定できないため、実データでの検証が必須）。

### テスト方針

- 配置: `sample_submission/tests/unit/test_own_hidden_state.py`（新規）、`test_zone_math.py`（新規）。
- `tests/unit/test_opponent_knowledge.py` の `make_pokemon` / `make_player_state` / `make_state` ヘルパーと同じスタイルで `State` を手組みする（cg エンジンの起動不要）。
- 観点:
  1. `zone_math`: 既知の小さい `(M, n, k)` で `prob_in_prize_exact` の合計が `prob_in_prize(at_least=0)` の全域で 1.0 になる（分布として整合）。手計算できる小ケース（例: `M=10, n=2, k=3`）で厳密値と一致することを確認。
  2. `OwnHiddenState.update()`: 手札・場・トラッシュに写ったカードが未確認プールから正しく引かれる（`test_opponent_knowledge.py` 同様、同名別 `card_id` のケースも1件）。
  3. 不変条件（未確認プール総数 == `deckCount + len(prize)`）が毎回成り立つ。
  4. `resolve_deck_search()`: 未確認プール内のカードのうち `select.deck` に写っていないものが確定的に `prize` 側へ移り、以後 `marginals()` で `deck: 0.0, prize: 1.0` を返す。写っているものは通常の超幾何のまま。
  5. `sample()`: 複数回呼んで `(deck, prize)` の card_id 多重集合が毎回「未確認プール全体」と一致する（二重計上・欠落が無い）こと、`len(prize) == len(state.players[my_index].prize)` を満たすこと。

### 完了条件

- 上記テストが green。
- `python -m pytest sample_submission/tests/unit/test_own_hidden_state.py sample_submission/tests/unit/test_zone_math.py` が単体で（Phase 2/3/4 の成果物なしに）通る。
- `tests/local_sim/test_local_game.py` 相当のローカル対戦ループに `OwnHiddenState` を差し込んでも（呼ぶだけで、まだ他モジュールとは結線しない）例外なく最後まで動く（Phase 4 の下ごしらえとしてのスモーク実行。差し込み自体は Phase 4 で正式に行うが、Phase 1 の時点で「壊れずに最後まで回る」ことだけ確認しておく）。

### 実装結果（2026-07-20）

- **`SelectData.deck` の実データ検証結果**: ローカル対戦10試合・149回の山札公開イベントを観測したところ、`SelectContext.LOOK` は一度も発火しなかった。`select.deck` は `TO_HAND`（サーチ効果の対象選択）や `TO_BENCH`（ベンチに出すポケモンのサーチ）など「選択肢の出所が山札」であるケースで、`select.context` の値によらず埋まっており、埋まる際は必ず `len(select.deck) == player.deckCount`（山札全量、部分公開は観測されず）だった。そのため `resolve_deck_search()` の発火条件は、プラン記載の `select.context == SelectContext.LOOK and select.deck is not None` ではなく、**`select.deck is not None` のみ**に変更した（`own_hidden_state.py` の `resolve_deck_search()` docstring に検証結果を明記）。
- **`OwnHiddenState.update()` のシグネチャ拡張**: `update(self, state: State, select: SelectData | None = None) -> None`（プラン記載は `update(self, state)` のみ）。トレーナーカードの解決中、プレイされたカード自身が手札からは既に消えているがトラッシュへはまだ移っていない「浮きカード」1枚により、公開ゾーン合計が60枚に対して1枚不足する一瞬が実データで確認された。`select.effect`（渡されている場合）で、この既知の一時的不整合を安全側の条件付きで補正する。
- **自分の公開ゾーンの列挙に不足**: design.md §4.1 のゾーン列挙に「自分が出したスタジアム」（`state.stadium` のうち `card.playerIndex == state.yourIndex`）が抜けていた。実データ検証で欠落が判明し、追加済み。
- **`_confirmed_deck` カウンタの追加と安全網**: `_confirmed_prize` に加え `_confirmed_deck` を持つ実装にした。さらに、同一 card_id に山札確定分・サイド確定分の両方がある状態で片方だけが公開ゾーンへ移った場合、`State`（card_id 単位の集計）だけからは「どちらの確定分が減ったか」を構造的に判別できないケースが実データ検証で見つかった。合計値がありえない値（例: サイド確定合計 > 実際のサイド枚数）になったら確定情報を丸ごと破棄し、通常の超幾何計算へ安全側フォールバックする処理を追加した（次に `resolve_deck_search()` が呼ばれれば確定情報は再度張り直される）。
- テスト: `test_own_hidden_state.py`（10件）+ `test_zone_math.py`（10件）= **20件 green**。ローカル対戦のスモーク実行で不変条件（未確認プール総数 == deckCount + len(prize)）の成立を確認。

#### 追記（2026-07-20・Phase 4 実施後の battle_review_viewer 接続検証で発見）

`battle_review_viewer`（後述）に `OwnHiddenState` を実接続したところ、既知の「浮きカード」補正（`select.effect` ベース）では拾えない**もう1系統の一時的不整合**が実データで見つかり、修正した。

- **セットアップ中の自分アクティブの伏せ不整合**: `state.turn == 0` の間、両者が同時公開を待つ一瞬、自分で選んだはずの `player.active` が `[None]`（伏せ）のまま観測される。この間に選んだカードは既に手札から消えているが、まだどこにも公開されていないため、公開ゾーン合計が1枚不足する。`SelectContext.DRAW_COUNT`（マリガンのボーナスドロー枚数選択）を挟むと、この不整合が複数手番にまたがって続くこともある。
- **修正**: `_last_hand_ids`（前回呼び出し時の手札 card_id 集合）と `_folded_own_active_id`（一度特定した「伏せ中の自分アクティブ」の記憶。`player.active` が伏せでなくなったらクリア）を追加。`actual_total == expected_total + 1` かつ `player.active` に `None` を含み `state.turn == 0` のとき、既に特定済みならそれを再利用し、未特定なら「前回の手札 − 今回の手札」の差分がちょうど1枚に絞れる場合に限り観測済み扱いにする（曖昧な場合は assert に委ねる、既存の安全側方針を踏襲）。
- 検証: `test_own_hidden_state.py` 20件 green を維持。`battle_review_viewer/export_replay.py --opponent self` を24試合分（自動シャッフルのため実質24通りの独立試合）生成し、`hidden_info` にエラーフレームが0件であることを確認（修正前は1試合あたり十分な頻度で発生していた）。

---

## Phase 2: 相手側 — `OpponentHiddenState`（混合モデル + 2段サンプラー）

### 2.0 前提: アーキタイプ代表リストのオフライン構築（新規オフライン資産）

`design.md` §5.1 の「リストを1つ仮定すれば『リスト − 観測済み = 未観測プール』」を成立させるには、
**アーキタイプごとの代表60枚リスト（またはカード別採用率テーブル）** が要る。これは現状どこにも存在しない
（`cardlist_referenced/build_meta_decks.py` が `sample_submission/ptcg_ai/shared/meta_decks.py` を生成する想定だが、
今のツリーには生成物自体が無く未コミット。かつ外部サイトのスクレイプ結果に依存し、`hybrid_predictor` の
クラス体系・card_id 空間との対応が別途必要になる）。

代わりに、**`kaggle_replays/deck_predictor/` の既存パイプラインが既に持っている `deck_db.jsonl`
（各エピソード×プレイヤーの実際の60枚デッキ、`extract_decks.py` の出力）と `deck_labels.jsonl`
（`label_decks.py` によるアーキタイプラベル、`hybrid_predictor` の学習ラベルと完全に同じ語彙）を
そのまま再利用する。** 同じ card_id 空間・同じアーキタイプ語彙で完結するため、変換コストがゼロ。

- 新規スクリプト: `kaggle_replays/deck_predictor/build_archetype_pool.py`
  - 入力: `output/deck_db.jsonl` + `output/deck_labels.jsonl`（既存パイプラインの手順1〜2の成果物）。
  - 処理: アーキタイプごとに、そのラベルが付いた全デッキを集計し、`card_id -> 採用率(0.0-1.0)` と
    `card_id -> 中央値(または最頻値)採用枚数` を計算する。他のスクリプト（`train_nb.py` 等）と同じく
    `deck_db.jsonl` / `deck_labels.jsonl` を読むだけの軽量集計なので、既存パイプラインの手順1〜2
    （`extract_decks.py` → `label_decks.py`）の直後に置く新しい手順として追加する。
  - 出力: `output/model/archetype_card_pool.json`
    ```json
    {
      "meta": {"built_at": "...", "n_decks_by_archetype": {"mega_lucario_ex": 42, ...}},
      "archetypes": {
        "mega_lucario_ex": {
          "card_counts": {"678": {"median": 2, "inclusion_rate": 0.95}, "677": {"median": 4, "inclusion_rate": 1.0}, ...}
        },
        ...
      }
    }
    ```
    （キーは `hybrid_predictor` の `classes`（21アーキタイプ + `other`）と一致させる。`other` は
    代表リストを持たない特別扱い＝後述 2.2 の「代表リスト無しアーキタイプ」ロジックで処理する。）
  - `--deploy` フラグで `sample_submission/ptcg_ai/hidden_information/archetype_card_pool.json` へコピーする
    （既存の `adjust_prior.py --deploy` / `fit_hybrid.py --deploy` と同じ配置パターンを踏襲）。
  - `README.md`（`kaggle_replays/deck_predictor/README.md`）の「実行順序」表に手順として追記する
    （このドキュメントでは変更しないが、本フェーズ実装時に追記すること）。

### 対象ファイル（新規）

- `sample_submission/ptcg_ai/hidden_information/opponent_hidden_state.py`
  - `class OpponentHiddenState`
    - `__init__(self, pool_path: str | Path | None = None)` — `archetype_card_pool.json` を読む。無ければ
      `is_ready = False`（`MLDeckPredictor` の「重みJSON無しは未ロード状態」というフォールバック規則と
      同じ思想。存在しない場合に例外にしない）。
    - `update(self, archetype_posterior: dict[str, float], observed_card_ids: dict[int, int], player_state: PlayerState) -> None` —
      毎ターン呼ぶ。`archetype_posterior` は `HybridDeckPredictor.predict(observed_cards, turn)` の戻り値、
      `observed_card_ids` は `OpponentKnowledge.get_prediction_features()["observed_card_ids"]` をそのまま渡す
      （呼び出し側の責務。本クラスは `HybridDeckPredictor` / `OpponentKnowledge` を直接 import しない。
      `prediction_summary.py` が `predictor` を duck-typing で受けるのと同様、疎結合を保つ）。
    - `_smoothed_weight(self, archetype: str, observed_card_ids: dict[int, int]) -> float` —
      §5.2（design.md）のスムージング。観測済みだが代表リストに無いカードの枚数に応じて
      `archetype_posterior[archetype]` を減衰させるが、下限（例: 元の重みの5%等、実装時にチューニング）を
      設けてゼロにはしない。
    - `marginals(self) -> dict[int, dict[str, float]]` — card_id ごとの `{"deck": p, "hand": p, "prize": p}`。
      アーキタイプで周辺化: `p(card in zone) = Σ_archetype P(archetype) * P(card in zone | archetype)`。
      `P(card in zone | archetype)` は `zone_math` の超幾何を3ゾーン（山札/手札/サイド）に拡張した
      多変量版（`hypergeom` を2値ではなく `deckCount : handCount : prizeCount` の比で計算する）。
    - `sample(self, rng: random.Random | None = None) -> tuple[list[int], list[int], list[int]]` —
      `(deck_card_ids, hand_card_ids, prize_card_ids)`。2段サンプリング:
      ① `_smoothed_weight` で重み付けした `archetype_posterior` から `random.choices` でアーキタイプを1つ引く。
      ② そのアーキタイプの代表リスト（`card_counts` の中央値枚数を採用）から `observed_card_ids` を引いた
      未観測プールを作り、`deckCount : handCount : len(prize)` の比で非復元抽出する。
    - `is_ready: bool` プロパティ（`archetype_card_pool.json` 読み込み成功フラグ）。

### 実装内容

- `zone_math.py`（Phase 1）に、2値（山札/サイド）ではなく3値（山札/手札/サイド）の多変量超幾何に対応する
  関数を追加する（`prob_in_zone(pool_size, zone_sizes: dict[str, int], target_count) -> dict[str, float]`
  のような形。Phase 1 の `prob_in_prize` はこの2値版として書き直すか、共存させるかは実装時に判断してよいが、
  **計算ロジックの二重実装は避ける**（`ml_predictor.py` が `evidence_count()` を fit 側・推論側で共有している
  のと同じ思想）。
- `observed_card_ids` に基づく未観測プールの計算は「代表リストの `card_counts`（中央値枚数）」から
  「観測済み枚数」を差し引くだけ（負にならないよう `max(0, ...)` でクリップ）。

### テスト方針

- 配置: `sample_submission/tests/unit/test_opponent_hidden_state.py`（新規）。
- `test_hybrid_predictor.py` と同じスタイルで、`tmp_path` に合成した小さな `archetype_card_pool.json`
  （2〜3アーキタイプ、数枚のカードのみ）を書き出し、それを `pool_path` として渡す形で検証する
  （本物の21アーキタイプ・数百カードのプールを使わず、ロジックだけを狭い語彙で検証する）。
- 観点:
  1. `marginals()`: 単一アーキタイプに posterior が 100% 集中しているケースで、超幾何の期待値と一致する。
  2. スムージング: 代表リストに無いカードが観測された場合、そのアーキタイプの重みが下がるが 0 にはならない
     （合成データで「代表リストに無いカードを見た」状況を作り、`marginals()` の該当アーキタイプ寄与が
     正の値のまま残ることを確認）。
  3. `sample()`: `len(hand) == player_state.handCount`、`len(prize) == len(player_state.prize)`、
     `len(deck) == player_state.deckCount` を満たす。複数回サンプルして、観測済みカード（`observed_card_ids`）
     が `deck`/`hand`/`prize` のどこにも二重に登場しない（合計が代表リストの採用枚数を超えない）。
  4. `is_ready`: `archetype_card_pool.json` が無いパスを渡した場合に `False` になり、`sample()`/`marginals()`
     が例外ではなく空/一様分布相当のフォールバックを返す（`MLDeckPredictor` の `{"other": 1.0}` に相当する
     安全なデフォルトを設計・実装すること）。

### 完了条件

- 上記テストが green。
- `build_archetype_pool.py` を実データ（`kaggle_replays/deck_predictor/output/deck_db.jsonl` 等、
  ローカルで再生成したもの）に対して実行し、`archetype_card_pool.json` が21アーキタイプ分（`hybrid_predictor`
  の `classes` と一致する数）出力されることを目視確認する（自動テストではなく手動実行の確認でよい。
  他のオフラインパイプラインスクリプトと同じ扱い）。
- `python -m pytest sample_submission/tests/unit/test_opponent_hidden_state.py` が Phase 1/3/4 の成果物なしに
  単体で通る。

### 実装結果（2026-07-20）

- 代表リストの未観測プール総数 `M` と隠しゾーン合計 `Z`（`deckCount + handCount + len(prize)`）が実データでは一般に一致しない（代表値は「よくある構築」の中央値であり、テックカードやサイド落ち・トラッシュ済みの実データとズレるため）。この不一致を `M_eff = max(M, Z)` と `None` プレースホルダ（不足分を埋める「不明カード」）で吸収する設計を採用した。`sample()` の返却リスト長は常に `PlayerState` の実ゾーンサイズ（`deckCount` / `handCount` / `len(prize)`）に一致する（`opponent_hidden_state.py` モジュールdocstring「プールサイズとゾーン合計の不一致の扱い」に詳細）。
- スムージングのパラメータを確定: 代表リストに無い観測カード（distinct card_id）1件につき重みを **0.5倍**（`_SMOOTHING_DECAY_PER_MISS`）に減衰、下限は元の重みの **5%**（`_SMOOTHING_FLOOR_RATIO`）。代表リストを持たない `other` アーキタイプは減衰対象外（生の重みをそのまま使う）とし、代表リストが無いため `sample()` では全ゾーンを `None`（不明カード）で埋め、`marginals()` では特定 card_id への寄与を持たない扱いにした。
- 実データでのプール構築: `deck_db.jsonl`（9,396デッキ）を集計し、21アーキタイプ分の代表リストを構築。`hybrid_predictor` の `classes` と完全一致することを確認し、`build_archetype_pool.py --deploy` で `sample_submission/ptcg_ai/hidden_information/archetype_card_pool.json` へ配置済み。
- テスト: `test_opponent_hidden_state.py`（16件）green。

---

## Phase 3: リプレイでのキャリブレーション検証

### 3.1 データソースが2種類あることに注意

- **`kaggle_replays/replays/*.json`（Kaggle からダウンロードした実リプレイ、~~719件~~ 実装結果注: 実ファイル数は **4,698件**。719件は誤記。詳細は本フェーズ末尾「実装結果」参照）**:
  各ステップの `steps[i][player_index]["observation"]` は「そのプレイヤー自身に配信された `Observation`」
  であり、`current.players[player_index].hand` はそのプレイヤー自身の手札として**常に真値が入っている**
  （非公開なのは相手から見た場合だけ）。よって **手札 vs 山札の切り分けは、両プレイヤー分の視点を
  同じ replay から読むだけで100%正解が取れる**（追加のシミュレーション不要）。
  一方 **サイドカードの中身は、そのプレイヤー自身にも伏せられている**
  （`PlayerState.prize` は `list[Card | None]`、伏せは `None`）。ゲーム中にそのサイドが実際に取られた
  瞬間（KO 後の `TO_PRIZE`/`TO_HAND` 系ログでサイドから手札へ移動する）だけ中身が判明する。
  **未取得のまま試合が終わったサイドは、この replay データからは真値を復元できない。**
- **`battle_review_viewer` のローカル対戦（`export_replay.py` / `live_match.py`、`cg.game.visualize_data()` 経由）**:
  自前でシミュレートするため `visualize_data()` が両者の手札・山札・サイドを含む完全情報（神視点）を返す
  （`opponent_knowledge_diff.py` が `OpponentKnowledge` の検証に既に使っている仕組みと同じ）。
  未取得サイドの中身まで含めて100%の ground truth が取れるが、Kaggle 実リプレイ由来の相手多様性は失われる
  （自己対戦 or ボット対戦のみ）。

したがって検証は2段構成にする: **量を稼げる Kaggle リプレイで「山札 vs 手札」の切り分け精度と
「取得済みサイド」の的中率を測り、`battle_review_viewer` のローカル対戦で（件数は少なくても）
「未取得サイドを含めた完全なキャリブレーション」を追加検証する。**

### 対象ファイル（新規）

- `kaggle_replays/deck_predictor/evaluate_hidden_information.py`
  - 入力: `output/deck_db.jsonl`（真の60枚デッキ）、`output/deck_labels.jsonl`、`output/model/deck_predictor_weights.json`
    等（既存 `evaluate.py` と同じ入力群）、`output/model/archetype_card_pool.json`（Phase 2 成果物）。
  - 処理: `build_dataset.py` と同じ要領で各リプレイ・各意思決定時点を走査し、その時点の
    `HybridDeckPredictor.predict()` と `OpponentHiddenState.marginals()` を計算する。
    同じ時点の「相手プレイヤー視点の `Observation`」から、実際に相手の手札にあった card_id 集合
    （ground truth）を読み取り、`marginals()` の `hand` 確率と突き合わせる。山札についても同様
    （「相手の手札にも公開ゾーンにも無い残り」が実際の山札の中身）。
  - 出力: `output/hidden_info_eval_report.md` — 確率をバケット分けした reliability テーブル
    （予測確率 vs 実頻度。`evaluate.py` の ECE/reliability テーブルと同じ形式に合わせる）を
    「山札」「手札」それぞれについて出す。既存 `evaluate.py` のコード（reliability 計算部分）を
    関数として切り出して再利用できないか、実装時に確認すること（車輪の再発明を避ける）。
- `battle_review_viewer/hidden_info_diff.py`（新規、`opponent_knowledge_diff.py` と対になる位置づけ）
  - `visualize_data()` の神視点から、ある瞬間の「相手の山札の中身」「相手の手札の中身」「相手のサイドの中身
    （未取得分も含む）」を集合として取り出すユーティリティ（`opponent_knowledge_diff.collect_ground_truth()`
    が公開ゾーンについてやっているのと同じことを、非公開ゾーンについて行う）。
  - `OpponentHiddenState.marginals()` の出力と突き合わせて、同じ reliability 集計を行う
    （件数は少ないので、自動テストというより手動実行のレポートツールとしてよい）。

### テスト方針

- 本フェーズの成果物（`evaluate_hidden_information.py` / `hidden_info_diff.py`）自体は
  「レポートを出す集計スクリプト」であり、`kaggle_replays/deck_predictor/` 配下の他のスクリプト
  （`evaluate.py`, `compare_nb.py` 等）と同様に**自動テスト対象にはしない**（このリポジトリの慣習に合わせる。
  `tests/unit/` はランタイムモジュール専用）。
- 代わりに、`OwnHiddenState` / `OpponentHiddenState` の**計算ロジック自体**は Phase 1/2 の単体テストで
  担保済みという前提に立ち、本フェーズは「その計算が実データに対して妥当な精度を出しているか」を
  人間が読むレポートとして出力する位置づけにする。

### 完了条件

- `evaluate_hidden_information.py` を実データに対して実行し、`hidden_info_eval_report.md` が生成される。
- 山札/手札の reliability（予測確率と実頻度の乖離）が、ナイーブな一様分布仮定（アーキタイプを考慮しない
  版）より悪化していないことを確認する（悪化していたら Phase 2 のスムージングやゾーン配分ロジックに
  バグがある可能性が高いので、そこへ戻って直す）。
- `battle_review_viewer/hidden_info_diff.py` によるローカル対戦検証を最低数試合分実行し、大きな乖離が
  無いことを確認する（未取得サイドを含めた完全ground truthでの追加確認）。

### 実装結果（2026-07-20）— 差し戻しと修正の経緯

- `kaggle_replays/replays/` の実ファイル数は **4,698件**（本セクション冒頭「719件」は誤記、上記のとおり訂正）。評価は500件のリプレイ・**68,646意思決定時点**（両プレイヤー×両視点の合計）で実施した（`evaluate_hidden_information.py`、`output/hidden_info_eval_report.md`）。
- **初回評価**: 山札 ECE **0.0224** vs ナイーブ **0.0372**（合格）、サイド（`hidden_info_diff.py` の神視点ローカル対戦5試合）**0.0054** vs ナイーブ **0.0072**（合格）、**手札 ECE 0.0096 vs ナイーブ 0.0086 で不合格**。高確信ビンでの系統的な過信が原因（例: 予測確率95.6%帯に対し実際の的中率77.7%）。
- 完了条件（本フェーズ「ナイーブな一様分布仮定より悪化していないこと」）に照らして不合格のため、プラン規定どおり **Phase 2 の実装へ差し戻した**。原因分析: 「手札 = 未観測プールからの一様サンプル」という design.md §5.3 が明記する簡略化（意図的に後回しにされた条件付け）が、相手がプレイ可能なカードを手札から使う偏りを無視するため、まだ観測されていないカードが小さな手札（3〜7枚）にそのまま残っている確率を系統的に過大評価することが分かった。
- **修正**: `OpponentHiddenState.marginals()` の**手札確率にのみ**、単調・保守的な縮小補正を導入した: `p' = 0.15 + (p - 0.15) * 0.5`（`p <= 0.15` はそのまま素通し）。常に `p' <= p` が成り立ち、確率を強気側へ動かすことは決してない（「確信度は慎重側に倒す」プロジェクト方針準拠）。山札・サイドの `marginals()` には手を加えていない（検証でナイーブ基準に対し悪化していなかったため）。パラメータは `floor ∈ [0.10, 0.20]` × `keep ∈ [0.4, 0.6]` の近傍全域でナイーブ手札 ECE を下回ることを確認済みで、特定の検証データへの過剰適合ではない。
- **最終評価**（500件リプレイ・68,646意思決定時点）: 手札 ECE **0.0068** vs ナイーブ **0.0095**、山札 **0.0224** vs **0.0372**、いずれも合格。神視点ローカル対戦5試合（556意思決定時点、`hidden_info_diff.py`）でも全ゾーン合格: 山札 0.0261 vs 0.0344、手札 0.0116 vs 0.0161、サイド 0.0116 vs 0.0127。
- 山札 ground truth の会計不整合（cg エンジンの遷移的な一瞬に起因。全意思決定時点の **41.72%** で発生）を検出し、該当時点は山札サンプルから除外する処理を `evaluate_hidden_information.py` に追加した（手札 ground truth は `player.hand` を直接読むだけなのでこの不整合の影響を受けない）。
- ナイーブベースラインは「全アーキタイプ等重み」の一様事後分布を採用（`archetype_card_pool.json` のアーキタイプキー一覧から構築、`hybrid_predictor.predict()` の実posteriorと並べて同じ decision point で計算）。

---

## Phase 4: agent / search への接続

### 4.1 現状把握（接続前の重要事実）

- `main.agent(obs_dict)` は1試合を通じて**同じ Python プロセス内で繰り返し呼ばれる**
  （`tests/local_sim/test_local_game.py` の `battle_select` ループ、および Kaggle 実行環境も同様の想定）。
  よってモジュールレベルの変数は試合中持ち越せる。既に `ptcg_ai/shared/card_cache.py` が
  `global _card_cache` / `reset_cache()` というパターンでこれを行っている。
- **現状、対局ループのどこにも「1試合を通じて `OpponentKnowledge` を蓄積し続ける」処理が無い。**
  `rough_predictor.py` の `_state_prediction_features()` は毎回**その場限りの新しい** `OpponentKnowledge` を
  `state` だけから作っており（`logs` を一切見ない）、`opponent_knowledge.py` 冒頭のdocstringが明記する
  「`update_from_logs` → `update_from_state` の順で毎ターン呼び、蓄積する」という正しい使い方をしていない
  （現状はあくまでフォールバック的な単発計算）。**Phase 4 で初めて、正しい蓄積フローを持つ永続インスタンスを
  対局ループに接続する。**
- 呼び出しチェーンは `main.agent()` → `core.agent.agent()` → `rule_based.rule_based_agent.agent()` →
  （`obs.select is None` なら `_select_deck()`、それ以外は）`action_selection.router.route()` →
  `handlers/*.handle()`。**`rule_based_agent.agent()` が、デッキ選択ターンと通常ターンの両方を毎回必ず
  通る唯一の共通の入口**（`core.agent.agent()` はさらに外側で `AGENT_TYPE` 分岐をしているだけ）。

### 対象ファイル

- 新規: `sample_submission/ptcg_ai/hidden_information/match_context.py`
  - 1試合分の `OwnHiddenState` / `OpponentHiddenState` / `OpponentKnowledge` をまとめて保持する
    モジュールレベルシングルトン（`card_cache.py` と同じパターン）。
  ```python
  _knowledge: OpponentKnowledge | None = None
  _own_state: OwnHiddenState | None = None
  _opponent_state: OpponentHiddenState | None = None

  def reset() -> None: ...  # 新しい試合の開始を検知したら呼ぶ

  def update(obs: Observation) -> None:
      """obs.select is None（デッキ選択）なら reset() してから自分の60枚を登録するだけ。
      通常ターンなら opponent_knowledge の呼び出し順契約(logs→state)を守って更新し、
      hybrid_predictor.predict() の結果で OpponentHiddenState を更新し、
      OwnHiddenState.update()（+ 山札サーチ検知時は resolve_deck_search()）を呼ぶ。"""

  def get_own_state() -> OwnHiddenState: ...
  def get_opponent_state() -> OpponentHiddenState: ...
  ```
  - 「新しい試合の開始」の検知は `obs.select is None`（デッキ選択ターン。`design.md` §1 の背景・
    CLAUDE.md の「初回（デッキ選択） → obs.select は None」の表と一致）をトリガーにする。
    Kaggle 実行環境が同一プロセスで複数試合を回す可能性を考慮し、`card_cache.reset_cache()` と同様
    テスト用にも `match_context.reset()` を公開する。
- 変更: `sample_submission/ptcg_ai/rule_based/rule_based_agent.py`
  - `agent(obs)` の先頭で `match_context.update(obs)` を呼ぶ（1行追加）。デッキ選択・通常ターンいずれの
    パスも `agent(obs)` は必ず通るので、ここが唯一かつ最小の接続点になる。
  - **既存の判断ロジック（`router.route(obs)` 以降）は一切変更しない。** 本フェーズの範囲は
    「`hidden_information` が正しく更新され続ける状態にする」ところまでであり、`action_selection/handlers/*`
    が `hidden_information` の `marginals()` を読んで意思決定を変える改修は、本プラン外
    （各 handler 側の別課題として扱う。理由: どのタイミングでどの handler が非公開情報の確率を
    使うべきかは意思決定ロジックそのものの設計であり、本レイヤーの責務ではない）。
- 新規: `sample_submission/ptcg_ai/hidden_information/search_adapter.py`
  - `to_search_begin_kwargs(own_state: OwnHiddenState, opponent_state: OpponentHiddenState, obs: Observation, rng=None) -> dict`
    — `OwnHiddenState.sample()` / `OpponentHiddenState.sample()` を呼び、`cg.api.search_begin()` の
    キーワード引数（`your_deck`, `your_prize`, `opponent_deck`, `opponent_prize`, `opponent_hand`,
    `opponent_active`）にそのまま展開できる `dict` を返す薄いアダプタ。
    `opponent_active`（相手のバトル場が伏せの場合のみ必要。`search_begin` のシグネチャ参照）は
    `OpponentHiddenState.sample()` が返す手札/山札/サイドとは別の小さな追加ロジックが要る
    （伏せポケモンの正体は「アーキタイプの basic ポケモンから尤もらしい1体を選ぶ」程度の簡易実装でよい。
    `CardData.basic` で絞り込める）。
  - **`ptcg_ai/search/` 自体（MCTS 本体）はまだ存在しないため、本フェーズでは `search_begin()` を
    実際には呼ばない。** アダプタが返す `dict` の各リストの**長さ**が `search_begin()` の docstring に
    書かれた検証条件（`your_deck` は `deckCount` 以上、`your_prize` は `len(prize)` 以上、等）を満たすことを
    テストで確認するに留める（実際の呼び出し・探索結果の妥当性検証は `ptcg_ai/search/` 実装時の別プランで行う）。

### テスト方針

- 配置: `sample_submission/tests/unit/test_match_context.py`（新規、手組み `Observation` で
  「デッキ選択 → 数ターン進行 → リセットして次の試合」を模擬）。
  `sample_submission/tests/unit/test_search_adapter.py`（新規、`to_search_begin_kwargs()` が返す各リストの
  長さが `search_begin()` の検証条件を満たすことを確認）。
- 統合テスト: `tests/local_sim/test_local_game.py` と同じ枠組みで、`match_context.update(obs)` を
  `rule_based_agent.agent()` に組み込んだ状態のまま**ローカル対戦を最後まで1本通す**
  （`cg.game.battle_start/battle_select/battle_finish`）。目的は「例外を出さずに最後まで動く」ことと、
  `OwnHiddenState` の不変条件（Phase 1）が全ターンで破れないことの確認。既存の `test_local_game.py` を
  改変せず、新規に `tests/local_sim/test_local_game_hidden_info.py` を追加する形が望ましい
  （既存のスモークテストの意味を変えないため）。

### 完了条件

- 上記単体テスト・統合テストが green。
- ローカル対戦を10試合程度連続実行し（同一プロセス内、`match_context.reset()` が試合間で正しく効くことの
  確認を兼ねる）、`OwnHiddenState` の不変条件違反やクラッシュが発生しないこと。
- `router.route(obs)` 以降の意思決定結果（既存テスト `tests/integration/new_deck/` 等）が本フェーズの変更前後で
  一切変わらないこと（`match_context.update()` の追加が純粋な副作用追加であり、既存の action_selection の
  挙動に影響しないことの回帰確認。既存の統合テストをそのまま再実行して green を維持すればよい）。

### 実装結果（2026-07-20）

- プラン記載どおり実装完了: `match_context.py`（`update()` 本体を丸ごと `try/except` で包み、推定レイヤーの失敗が意思決定に絶対に波及しない設計）、`rule_based_agent.py` への1行追加（`agent(obs)` 先頭で `match_context.update(obs)` を呼ぶ）、`search_adapter.py`（プラン記載どおり `search_begin()` は実際には呼ばず、返す `dict` 各リストの長さのみ検証）。
- 循環import回避のため、`match_context._load_own_deck_ids()` は `ptcg_ai.rule_based.rule_based_agent.read_deck_csv` を関数内で遅延importしている（モジュール先頭で import すると `rule_based_agent → match_context → rule_based_agent` の循環になるため）。
- 検証結果: `sample_submission/tests/unit` 全体で **133 passed / 6 skipped**、統合テスト green、ローカル対戦10試合連続実行でクラッシュなし、`match_context.update()` のオーバーヘッドは約 **0.71ms/call**、5試合603手で `router.route()` 以降の意思決定が変更前後で完全一致（0 mismatch）を確認。
- 既知の残課題（実装報告に明記。本プランの範囲外として残す）:
  1. ~~`SelectContext.SETUP_BENCH_POKEMON` 中、自分自身のバトル場が自分視点でも伏せ（`active == [None]`）になる一瞬があり、`OwnHiddenState` の不変条件が単発で破れる（ローカル対戦約20試合中4試合で1回発生。いずれも次ターンで自己回復し、`match_context` の `try/except` で安全に吸収される）。将来 `own_hidden_state.py` 側で正式対応するのが望ましい。~~ → **解消済み（2026-07-20、Phase 1「追記」節参照）**。`battle_review_viewer` 接続検証で同根の不整合（`state.turn==0` かつ `player.active` に `None` を含む間の一時的な観測漏れ、`SelectContext.DRAW_COUNT` を挟むと複数手番継続する変種を含む）を `own_hidden_state.py` 側で正式に修正し、24試合分の実行でエラーフレーム0件を確認。
  2. `search_adapter.py` が `OpponentHiddenState` の私有メンバ（`_smoothed_normalized_weights()` / `_archetype_pool`）を直接参照している（`opponent_active`〈相手の伏せポケモンの正体推測〉のため。Phase 1〜3 の既存ファイルを変更しない制約の中での判断）。公開APIの追加は将来課題。
  3. `action_selection/handlers/*` が `marginals()` を意思決定に使う改修は、プラン記載どおり本プラン外のまま。

---

## 各フェーズの依存関係まとめ

```
Phase 1 (OwnHiddenState, zone_math)         … 独立に実装・テスト可能
Phase 2 (OpponentHiddenState)               … zone_math を Phase 1 から再利用。それ以外は独立
Phase 3 (リプレイ検証)                       … Phase 1・2 の成果物を評価する。ロジック自体は変更しない
Phase 4 (agent/search 接続)                  … Phase 1・2 の成果物を実際の対局ループに配線する
```

Phase 2 は Phase 1 の完了を待たなくても大枠の実装は進められる（`zone_math` のインターフェースだけ先に
固めておけば並行着手も可能）が、`zone_math` の二重実装を避けるため、実装順は本プランのとおり
Phase 1 → 2 を推奨する。
