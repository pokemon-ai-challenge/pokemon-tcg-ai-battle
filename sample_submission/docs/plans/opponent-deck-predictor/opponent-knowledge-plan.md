# 相手観測情報の記録器（OpponentKnowledge）実装計画

対象 Issue: 相手デッキ予測器サブ Issue（`opponent_knowledge.py` 本体）
配置予定モジュール: `sample_submission/ptcg_ai/opponent_modeling/opponent_knowledge.py`
親計画: [plan.md](./plan.md)（④「opponent_knowledge 連携」に相当）

> ステータス: 方針ドラフト（未確定）。実装着手前の合意用メモ。

---

## 1. ゴール（このIssueの射程）

相手について**実際に観測できたカード情報**（見えたカード・見えた場所・見えた枚数）を
1試合を通じて蓄積し、相手デッキ予測器や将来の山札推定が使いやすい形で取り出せるようにする。

- **やること**: 公開領域・`logs` から見えたカードを `observed_cards` として記録し、
  `get_prediction_features()` で取り出す。
- **やらないこと**（別Issue）:
  - 相手デッキそのものの予測（`rough_predictor.py` の役目）。
  - 山札・手札・サイド落ちなど**非公開領域の内容推定**（`inferred_cards`）。
  - `rough_predictor.json` のような特徴量 config の作成。

この分離が本Issueの核。**「実際に見えた情報（observed）」と「推定した情報（inferred）」を混ぜない。**
本器が持つのは observed のみ。inferred は将来レイヤーが本器の出力を入力として別途組み立てる。

---

## 2. 現状把握（着手前の重要事実）

`cg/api.py` を確認した結果、記録に使える情報源は次の2系統。

### 2.1 盤面スナップショット（`State`、`update_from_state` で使う）

相手 = `state.players[1 - state.yourIndex]`（`PlayerState`）。公開ゾーン:

| Issueの領域 | 取得元 | `AreaType` | 備考 |
|-------------|--------|-----------|------|
| バトル場 | `player.active[0]`（`Pokemon \| None`） | `ACTIVE` | 伏せ中は `None`（記録しない） |
| ベンチ | `player.bench`（`list[Pokemon]`） | `BENCH` | |
| トラッシュ | `player.discard`（`list[Card]`） | `DISCARD` | 採用確認に有効 |
| 付いているエネルギー | `Pokemon.energyCards`（`list[Card]`）/ `Pokemon.energies` | `ENERGY` | カード実体＋色ヒント |
| 付いている道具 | `Pokemon.tools`（`list[Card]`） | `TOOL` | |
| 進化元 | `Pokemon.preEvolution`（`list[Card]`） | `PRE_EVOLUTION` | 進化ラインの露出 |
| スタジアム | `state.stadium`（`list[Card]`、共有領域） | `STADIUM` | 所有者は判定要（後述） |
| 一時公開 | `state.looking`（`list[Card \| None] \| None`） | `LOOKING` | 見えている間だけ。伏せは `None` |

**不可視**（本Issueでは扱わない）: `hand`（相手は `None`）、`prize`（伏せは `None`）、`deckCount`/`handCount`（枚数のみ）。

各カードは `id`（= `CardData.cardId`, int）と `serial`（試合内で一意）を持つ。
**盤面のカードは名前を持たない** → 名前は `all_card_data()` の `CardData.name` から `cardId` 逆引きで解決する。

### 2.2 イベント履歴（`Observation.logs`、`update_from_logs` で使う）

`logs` は前回選択以降のイベント列。相手のカードが**一瞬だけ公開される**（プレイ→即トラッシュ等）
ケースは、スナップショットだけでは取りこぼすため logs で補う。相手が関与するもの（`playerIndex == 相手index`）
のうち、`cardId != None`（表向き）のものを記録対象にする。

| `LogType` | 主フィールド | 記録内容 | 記録ゾーン |
|-----------|-------------|---------|-----------|
| `PLAY`(10) | `cardId, serial` | 手札からプレイされたカード（サポート/グッズ等が一時公開） | `LOOKING`（一時公開）→ 実質「デッキに入っていた証拠」 |
| `EVOLVE`(12) | `cardId, serial` / `cardIdTarget` | 進化して出たカード | `ACTIVE`/`BENCH`（`state` 側で確定するので補助） |
| `DEVOLVE`(13) | `cardId, serial` | 退化で戻ったカード | — |
| `MOVE_CARD`(6) | `cardId, serial, fromArea, toArea` | 公開移動（例: 場→トラッシュ） | `toArea` |
| `ATTACH`(11) | `cardId, serial`（+ `cardIdTarget`） | エネ/道具の付与 | `ENERGY` or `TOOL`（カード種別で判定） |
| `SWITCH`(8) / `CHANGE`(9) | `cardIdActive/Bench`, `cardIdBefore/After` | 入れ替え・変化したポケモン | `ACTIVE`/`BENCH` |
| `MOVE_ATTACHED`(14) | `cardId, serial` | 付属カードの移動 | 付替え先 |

- `MOVE_CARD_REVERSE`(7) は**伏せ移動**で `cardId` を持たない → 記録しない。
- `DRAW_REVERSE`(5) も内容不明 → 記録しない。
- logs の `serial` は `state` の `serial` と同一空間なので、**両系統を serial で突き合わせできる**。

---

## 3. データモデル

### 3.1 観測レコード（serial を主キーにする）

各実カードは試合内で一意な `serial` を持つ。これを**同一カード追跡の主キー**にすることで、
「同じ1枚が場→トラッシュへ移動」しても二重カウントせず、ゾーン移動として追跡できる（完了条件の
「公開領域間で移動した場合 serial で同じカードとして追跡」を満たす）。

`card_id` は同名でも印刷違い（再録・別テキスト・別HP）ごとに異なる値を持つため、
**枚数カウントの単位は常に `card_id`**（`serial` を distinct カウントの重複排除キーとして使う）。
`name` は複数 `card_id` を束ねうる**表示・集計用のラベル**であり、カウントの単位にはしない
（山札推定で card_id ごとの残り枚数を引き算する際に、name に丸めた時点で情報が失われるため）。

```python
@dataclass
class ObservedCard:
    serial: int | None       # 試合内一意ID（主キー。logsで欠落時のみ None）
    card_id: int             # CardData.cardId
    name: str                # all_card_data() から解決した名前
    zones_seen: set[str]     # これまで見えた全ゾーン（"active"/"bench"/"discard"/"energy"/"tool"/"pre_evolution"/"stadium"/"revealed"）
    current_zone: str | None # 直近スナップショットで見えたゾーン（消えたら None）
    first_seen_turn: int | None
    last_seen_turn: int | None
    is_energy: bool          # CardType による分類（features 用）
    is_tool: bool
    is_pokemon: bool
```

- **serial が取れないケース**（logs で `serial=None` 等）は、`(card_id, zone)` を代替キーにした
  ベストエフォート記録にフォールバックする。重複が疑わしいものは「これ以上枚数を増やさない」
  保守側に倒す（過大カウント防止 > 取りこぼし防止）。

### 3.2 コンテナ

```python
class OpponentKnowledge:
    _by_serial: dict[int, ObservedCard]         # serial → レコード（主索引）
    _no_serial: list[ObservedCard]              # serial 不明レコードの退避
    _id_to_name: dict[int, str]                 # all_card_data() から一度だけ構築
    _id_to_type: dict[int, CardType]            # energy/tool/pokemon 分類用
    _opponent_index: int | None                 # 記録対象プレイヤー
```

- **1試合＝1インスタンス**（試合内メモリのみ。永続化しない）。
- `all_card_data()` は初期化時に1回だけ呼び、`cardId → name / type` の辞書を作る。

### 3.3 ゾーン名（内部表現）

`AreaType` の生値ではなく、features で扱いやすい**小文字文字列キー**に正規化する:
`"active" / "bench" / "discard" / "energy" / "tool" / "pre_evolution" / "stadium" / "revealed"`。
`revealed` = 一時公開（`LOOKING` や PLAY ログ由来）。`AreaType ↔ 文字列`の対応表を1箇所に持つ。

---

## 4. 公開 API（契約）

**呼び出し順序は固定: 新しい `Observation` を受け取るたびに `update_from_logs(obs.logs)` →
`update_from_state(obs.current)` の順で呼ぶ。** `logs` は「この `state` に至るまでの出来事」なので
時系列としては先に処理し、盤面の完全スキャンである `update_from_state` を最後に当てて現在ゾーンを
確定させる。逆順で呼ぶと、盤面スキャンが正しく特定した現在ゾーンを、logs 側の暫定タグ
（`revealed` や、SWITCH ログ由来の active/bench 等）が後から上書きしてしまう。
（`battle_review_viewer` を使った実データ検証（§8-6参照）で実際に踏んだ不具合。呼び出し側の責務として
ここに明記する。）

```python
class OpponentKnowledge:
    def __init__(self, opponent_index: int | None = None): ...

    def update_from_state(self, state: State) -> None:
        """盤面スナップショットから相手の公開カードを記録する。
        opponent_index 未設定なら state.yourIndex から相手を確定する。冪等（毎回呼んでよい）。
        update_from_logs の後に呼ぶこと（このメソッドが現在ゾーンの最終確定を行う）。"""

    def update_from_logs(self, logs: list[Log]) -> None:
        """logs から相手がプレイ/進化/トラッシュ移動したカードを記録する。
        update_from_state より前に呼ぶこと。"""

    def observe_card(self, card_id: int, zone: str,
                     turn: int | None = None, serial: int | None = None) -> None:
        """観測1件を記録する最小単位（上記2メソッドが内部で呼ぶ低レベルAPI）。
        serial があれば既存レコードを更新、無ければ (card_id, zone) でベストエフォート追加。"""

    def get_observed_cards(self) -> list[ObservedCard]:
        """これまでに観測した全カード（履歴。現在盤面から消えたものも含む）。"""

    def get_zone_cards(self) -> dict[str, list[ObservedCard]]:
        """現在の公開ゾーンごとのカード一覧（current_zone でグルーピング）。"""

    def get_prediction_features(self) -> dict:
        """相手デッキ予測器が使いやすい形式の観測特徴量を返す（下記スキーマ）。"""
```

### 4.1 `get_prediction_features()` の返り値スキーマ

**カウントの正は `card_id` 単位。`name` 単位は必ずそこから合算して導出する（逆はしない）。**
理由は次の2つの用途で必要な粒度が違うため:

- **山札推定（超幾何分布）** は「その `card_id` の残り枚数 = デッキ採用枚数 − 公開領域で見えた枚数」
  という **card_id 単位の引き算**で成立する。同名でも別 `card_id`（再録・別テキスト・別HP）は
  別カードなので、name に丸めた時点でこの引き算ができなくなる（`rough_predictor.json` にも
  `"リオル": [333, 677, 974]` のように1名に複数IDがぶら下がる例がある）。
- **アーキタイプ照合・デッキ構築の「同名最大4枚」制約** は **name 単位**で見る。
  推定器はこの上限を prior として使うため name 集計も要る。

よって **同名の別カードも数える。ただし card_id ごとに別々に数え、name はその合算ビューとして
別キーで両方持たせる**（`observed_cards` を name 主体からは降格し、`observed_card_ids` を正にする）。

```python
{
    # --- 正: card_id 単位（distinct serial 数）。山札推定の引き算はこちらを使う ---
    "observed_card_ids": {345: 2, 970: 1, 121: 1, ...},

    # --- 派生ビュー: card_id を name で合算しただけ（アーキタイプ照合・4枚上限チェック用）---
    "observed_cards":    {"ヒトカゲ": 3, "リザードンex": 1},   # 345:2 + 970:1 の合算例

    # --- 同名内訳（name→{card_id: count}）。山札推定が「どの print が何枚か」を辿るのに使う ---
    "name_to_card_ids":  {"ヒトカゲ": {345: 2, 970: 1}},

    "zone_cards":        {zone: [name, ...], ...},  # 現在ゾーンごとの名前一覧
    "observed_pokemon":  [name, ...],               # is_pokemon なカード（重複除去）
    "observed_energies": [name, ...],               # is_energy なカード（付属エネルギー等）
    "observed_tools":    [name, ...],                # is_tool なカード
    "energy_types":      [EnergyType, ...],          # Pokemon.energies から集めた色（色ヒント）
}
```

- **枚数カウントは distinct serial ベース**。同じ物理カード（同一 `serial`）を何度見ても
  対応する `card_id` のカウントは +1 のまま。別 `serial`（同名・別 `card_id` を含む）が
  見えるたびに、その `card_id` のエントリにだけ +1 する。
  → 完了条件「同じカードを重複して数えすぎない」「カードごとの観測枚数」を、
  card_id 粒度で満たす（name 粒度はその合算なので自動的に満たされる）。
- `rough_predictor.py`（親計画②/④）は name 照合が主なので `observed_cards` / `name_to_card_ids` を、
  将来の山札推定は `observed_card_ids` を直接引き算に使う。**name → card_id へ逆変換する
  （lossy な合算から戻す）処理は行わない** — 常に card_id 側の生データを持ち回る。

---

## 5. 処理フロー

### 5.1 `update_from_state(state)`

1. `opponent_index` を確定（`1 - state.yourIndex`）。
2. 前ターンの `current_zone` を一旦クリア（今スナップショットに無いカードは「現在ゾーン None」にする）。
3. 相手 `PlayerState` を走査:
   - `active[0]`（`None` でなければ）→ ポケモン本体＋ `energyCards`/`tools`/`preEvolution` を各ゾーンで記録。
   - `bench[*]` → 同上。
   - `discard[*]` → `DISCARD` で記録。
   - `Pokemon.energies`（色配列）→ `energy_types` ヒントとして集約（カード実体は `energyCards` 側で記録）。
4. `state.stadium` → スタジアムは共有領域。**所有者が相手のときのみ**記録（判定できない場合は
   `stadium` ゾーンとして記録しつつ、予測器側で扱いを弱める。要検討 §8）。
5. `state.looking` → `revealed` で記録（`None` 要素は伏せなのでスキップ）。
6. 各カードは `serial` をキーに `_upsert()`：既存なら `zones_seen` に追加＋`current_zone`/`last_seen_turn` 更新、
   新規なら追加。`turn = state.turn`。

### 5.2 `update_from_logs(logs)`

1. 各 `log` について `playerIndex == opponent_index` かを確認（違えば無視）。
2. `LogType` ごとに §2.2 の表に従いカードとゾーンを取り出し、`observe_card()` を呼ぶ。
3. `serial` があるものは主索引に乗り、後続の `update_from_state` と自動的に突き合う。

### 5.3 `observe_card(card_id, zone, turn, serial)`

- `name = _id_to_name.get(card_id, str(card_id))`、type から `is_*` を決定。
- `serial is not None`: `_by_serial` に upsert。
- `serial is None`: `_no_serial` を `(card_id, zone)` で検索し、無ければ追加（重複追加しない）。

---

## 6. 実装順序

| 順 | 作業 | 依存 | 完了で満たす条件 |
|----|------|------|-----------------|
| ① | `ObservedCard` / `OpponentKnowledge` の骨格・`_id_to_name` 構築・`observe_card` | なし | クラス実装・serial 管理の土台 |
| ② | `update_from_state`（公開ゾーン走査） | ① | active/bench/discard/energy/tool、ゾーン別取得、枚数 |
| ③ | `update_from_logs`（PLAY/EVOLVE/MOVE_CARD/ATTACH…） | ① | logs からの記録、一時公開の記録 |
| ④ | `get_observed_cards` / `get_zone_cards` / `get_prediction_features` | ②③ | 取り出しAPI・features スキーマ |
| ⑤ | 単体テスト（模擬 `State`/`Log` フィクスチャ） | ①〜④ | ローカル動作確認・非破壊確認 |

---

## 7. テスト・検証（⑤）

- 配置: `tests/unit/`（cg 実行不要。`State`/`Pokemon`/`Card`/`Log` を最小構成で手組み）。
- 観点:
  1. 相手のバトル場/ベンチ/トラッシュのカードが `observed_cards` に入る。
  2. 同一 `card_id` を別 serial で2枚見た → `observed_card_ids[card_id] == 2`、
     同一 serial 再観測 → 1 のまま。
  2b. 同名だが `card_id` が異なる2種を1枚ずつ見た →
      `observed_card_ids` は別エントリで各1、`observed_cards[name] == 2`（合算）、
      `name_to_card_ids[name] == {id_a: 1, id_b: 1}`。
  3. 付属エネルギー・道具が `observed_energies` / `observed_tools` に分類される。
  4. カードが `ACTIVE → DISCARD` へ移動しても serial 追跡で1枚扱い（重複カウントしない）。
  5. logs の `PLAY` で一瞬見えたサポートが `revealed` として残る。
  6. 伏せカード（`active[0] is None`、`looking` の `None`、`MOVE_CARD_REVERSE`）は記録されない。
  7. 自分のカードは記録されない（`opponent_index` フィルタ）。
- **非破壊確認**: 本器は独立モジュール。`main.py` / 既存の意思決定・探索から呼ばない
  （親計画④で予測器に繋ぐのは後続）。import 追加だけで既存挙動が変わらないことを確認。

---

## 7.5 実データ検証（`battle_review_viewer` との自動 diff）

手組みフィクスチャの単体テスト（§7）だけでは、実際の cg エンジンが吐く `logs` の組み合わせ・
順序に対する想定漏れは検出できない。そこで既存の `battle_review_viewer/`（ローカル対戦のリプレイ
ビュアー）を拡張し、実データでの検証手段を用意した。

- `battle_review_viewer/export_replay.py`: `run_match` 内で player0（提出エージェント）視点の
  `obs.current`/`obs.logs` を毎フレーム `OpponentKnowledge` に食わせ、`get_prediction_features()`
  のスナップショットを各 frame に埋め込む。
- `battle_review_viewer/opponent_knowledge_diff.py`: `visualize_data()` が返す神視点（両者の手札まで
  見える完全情報）の盤面から、同じ瞬間の相手の公開ゾーンを独立に再集計し、`OpponentKnowledge` の
  観測結果と `serial` 単位で突き合わせて `missing`/`extra`/`mismatched` を機械的に検出する。
  - `missing`: 神視点では見えているのに観測できていない（取りこぼしバグ）
  - `extra`: 観測側は見えていると思っているが神視点では見えない（クリア漏れ・所有者取り違え）
  - `mismatched`: 両方にあるが `card_id`/ゾーンが食い違う（分類バグ）
  - セットアップ中の伏せバトル場ポケモン（神視点は伏せの中身まで見せるが実際の `Observation` は
    `None`）は `real_state` 側の facedown 判定で ground truth から除外し、誤検出を防いでいる。
- ビュアー側（`web/`）には `Debug: Opponent Knowledge` というトグル1つを toolbar に追加し、
  デフォルト非表示。ON にすると inspector サイドバーに観測特徴量と diff 結果が表示される
  （常時表示にせず、必要な時だけ切り替えられる設計）。

**この検証で実際に見つかった2件のバグ（修正済み）:**

1. **`SWITCH` ログのフィールド意味の取り違え**: `cg/api.py` の `cardIdActive`/`cardIdBench` は
   フィールド名と実際の意味が逆（`cardIdActive` = 元アクティブ＝ベンチへ移動する側、
   `cardIdBench` = 元ベンチ＝アクティブへ移動する側）。実データで検証して確定した
   （`_handle_switch` 修正済み、回帰テスト追加済み）。
2. **呼び出し順序による現在ゾーンの上書き**: `update_from_state` を先、`update_from_logs` を後に
   呼ぶと、盤面スキャンが正しく特定した現在ゾーンを logs 側の暫定タグが後から上書きしてしまう。
   正しい順序は `update_from_logs` → `update_from_state`（§4 に契約として明記、
   `export_replay.py` の呼び出し順序も修正済み）。

いずれも手組みの単体テストだけでは（想定自体が誤っていたため）検出できず、独立に実装した
ground truth との自動 diff で初めて見つかった。単体テストは「ロジックが意図通り動くか」、
この実データ diff は「意図（想定）自体が現実のエンジンと合っているか」を検証するもので、
補完関係にある。

---

## 8. 未確定・要判断（着手前に潰したい点）

1. **スタジアムの所有者判定**: `state.stadium` は共有領域で所有者フィールドが無い。
   `logs` の `PLAY`/`MOVE_CARD`（playerIndex）から所有者を辿るか、`stadium` ゾーンは
   参考情報に留めるか。
2. **serial 欠落ログの扱い**: `serial=None` のログをどこまで信用して枚数に反映するか
   （過大カウント防止を優先し、原則 §3.1 のベストエフォートに留める案）。
3. **`current_zone` のリセット粒度**: 毎 `update_from_state` で全クリアするか、
   「今回のスナップショットに写っていた serial だけ更新」にするか（ベンチ落ち・トラッシュ移動の表現）。
4. **features の枚数上限**: 同名カードは最大4枚（基本エネは無制限）。观測が4を超える異常値の扱い。
5. **`observed_card_ids` を予測器がどう食べるか**: 親計画 §6 の「serial で現盤面と重複しないものだけ追加」
   ロジックを予測器側に置くか、本器の features を最終形として渡すか（責務境界の確定）。
   → カウント単位は card_id・name 両方を features に含める方針で確定済み（§4.1）。
   残る論点は「予測器側でどちらを主に読むか」のみ（`rough_predictor.py` は name ベース、
   将来の山札推定器は card_id ベースを使う想定）。
