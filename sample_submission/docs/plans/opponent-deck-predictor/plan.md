# 相手デッキ予測器（rough_predictor）実装計画

対象 Issue: #25（MVP 本体）／ サブ #26（特徴量 config）ほか
配置予定モジュール: `sample_submission/ptcg_ai/opponent_modeling/`

> ステータス: 方針ドラフト（未確定）。実装着手前の合意用メモ。

---

## 1. ゴール

相手の**公開情報だけ**から、相手デッキが大まかにどのアーキタイプかを判定する
ルールベース／スコアリング型の予測器 MVP を作る。高精度 ML 分類は目指さない。

- 作るファイル: `opponent_modeling/rough_predictor.py`
- 作る関数: `predict(state, opponent_knowledge=None) -> dict`
- config: `opponent_modeling/rough_predictor.json`
- 返り値: `deck_type` / `display_name` / `confidence` / `score` / `evidence` / `candidates`
  （判定不能時は `unknown`）

---

## 2. 現状把握（着手前の重要事実）

- `ptcg_ai/` はまだスケルトン（各 `__init__.py` のみ）。`opponent_modeling/` も空。
  → ほぼ新規実装。
- メモリ記載の `src/inference`（ベイズ推定）・`meta_decks.py` は現ツリーに**存在しない**
  （`comparison_tmp/` の旧スナップショットのみ）。今回の予測器とは独立に新規で作る。
- **盤面のカードは `cardId`（int）で識別され、名前を持たない。**
  カード名は `all_card_data()` の `CardData.name` から引く必要がある。
  → config は名前ベースだが、照合には `card_id ↔ name` 解決が実質必須。
- 相手 = `state.players[1 - state.yourIndex]`。
- 相手の**公開ゾーン**（`cg/api.py` 確認済み）:
  | ゾーン | 取得元 | 備考 |
  |--------|--------|------|
  | バトル場 | `player.active[0]`（`Pokemon | None`） | 伏せ中は None |
  | ベンチ | `player.bench`（`list[Pokemon]`） | |
  | トラッシュ | `player.discard`（`list[Card]`） | 採用確認に有効 |
  | スタジアム | `state.stadium` | 共有領域 |
  | 付属エネルギー | `Pokemon.energyCards` / `Pokemon.energies` | エネルギー色ヒント |
  | 進化元 | `Pokemon.preEvolution`（`list[Card]`） | 進化しても進化ラインが露出する |
- **不可視**: `hand`（`None`）・`prize`（伏せ `None`）。枚数のみ `handCount` / `deckCount`。
  → 本 Issue では山札・手札・サイド落ちの詳細推定は扱わない。

---

## 3. 返り値スキーマ（契約）

```python
{
    "deck_type": "charizard_ex",       # 最有力アーキタイプの内部キー / 不能時 "unknown"
    "display_name": "リザードンex",
    "confidence": 0.72,                # 0.0〜max_confidence
    "score": 12.4,                     # 最有力の合計スコア
    "evidence": [                      # 根拠カード（最有力デッキ分）
        {"card": "リザードンex", "zone": "active", "role": "anchor",
         "weight": 12.0, "reason": "デッキの主軸カード"},
        # ...
    ],
    "candidates": [                    # スコア降順の候補一覧
        {"deck_type": "charizard_ex", "display_name": "リザードンex",
         "score": 12.4, "confidence": 0.72},
        # ...
    ],
}
```

判定不能時:

```python
{"deck_type": "unknown", "display_name": "unknown",
 "confidence": 0.0, "score": 0.0, "evidence": [], "candidates": []}
```

---

## 4. 実装順序（推奨）

依存が少ない土台から積む。**#26 config から着手**。

| 順 | 作業 | 対応 Issue | 依存 | 完了で満たす |
|----|------|-----------|------|--------------|
| ① | `rough_predictor.json`（特徴量 config） | #26 | なし | role/zone/archetypes/combo/generic/threshold/confidence |
| ② | `predict()` MVP（state 公開ゾーンのみ、`opponent_knowledge=None`） | #25 | ① | スコアリング・evidence・candidates・unknown |
| ③ | `card_id ↔ name` 解決ヘルパ | #25 | なし（②に内包可） | 名前照合と将来の card_id 優先 |
| ④ | `opponent_knowledge.py` と連携（観測履歴の蓄積・活用） | 別サブ | ②③ | opponent_knowledge 経路 |
| ⑤ | テスト（主要デッキ例）・`results/` 記録 | 別サブ | ② | 主要デッキ例の期待結果確認 |

**なぜ #26 が先か**
1. 依存ゼロ。返り値スキーマ・role・しきい値という「他サブ Issue が全部参照する契約」を先に固定できる。
2. config が固まれば `predict()` は「読んでスコアを足す」だけに単純化できる。
3. `opponent_knowledge.py`（④）は精度向上の補助であって MVP 必須経路ではない
   （`predict(state)` 単体で #25 の完了条件を満たせる）。最後で良い。

---

## 5. スコアリング設計（②）

```
for archetype in config.archetypes:
    score = 0
    seen_serials = set()  # 同じ実カード（serial）の重複加点を防ぐ
    for 見えているカード in 相手公開ゾーン(active, bench, discard, attached, revealed...):
        if カード.serial in seen_serials:
            continue
        if カード in archetype.cards:
            score += role_weights[role]  # ace_spec なら ace_spec_bonus も掛ける
            evidence.append(...)
            seen_serials.add(カード.serial)
    score += エネルギー色一致ボーナス（補助的）
    score += combo_rules 加点（all のカードが全て見えていれば bonus）
```

- **role ごとの共通スコア**（`role_weights`）を使う。カード個別点は持たない。
- **ゾーンは加点に使わない**: `Observation` に載るカードは常に表向き確定情報なので、
  バトル場/ベンチ/トラッシュのどこで見えても「デッキに入っていた」確からしさは変わらない。
  ゾーンによる脅威度の重み付けは `board_evaluation` の役目であり、このスコアリングには混ぜない。
  代わりに `serial`（実カードの一意ID）で重複観測を弾く。`evidence` にはゾーンを参考情報として残してよい。
- **汎用カードは加点しない**（`generic_cards` / role=`generic`, weight=0）。
- **判定ロジック**:
  - 最有力 `score >= unknown_threshold.min_score` かつ
    2位との差 `>= unknown_threshold.min_score_gap` を満たさなければ `unknown`。
  - `confidence` = 正規化（例: `min(score / confident_score, max_confidence)`）。
    2位との差で減衰させても良い（後で調整）。

### card_id ↔ name 解決（③）
- 起動時に `all_card_data()` から `name -> [card_id, ...]` と `card_id -> name` を1回構築。
- 照合優先順位: config に `card_id` があればそれで一致 → なければ `name` で解決。
  → #26 の「将来 card_id 優先」を自然に満たす。
- 名前は表記ゆれ・複数 cardId（再録）に注意。1名 → 複数 cardId を許容する。

---

## 6. opponent_knowledge 連携（④・後回し）

- 役割分担（Issue #26 メモ準拠）:
  - `opponent_knowledge.py`: 見えたカード・ゾーン・枚数・serial の**履歴**を蓄積
    （例: 一度トラッシュに落ちて盤面から消えたカードも「観測済み」として保持）。
  - `rough_predictor.json`: 特徴カード・role・スコア・しきい値の定義。
  - `rough_predictor.py`: 観測 + config でスコアリングし結果を返す。
- `predict()` は `opponent_knowledge` が渡されたら、現盤面に加えて観測履歴カードも
  スコア対象に含める（`serial` で現盤面と重複しないものだけ追加する）。
- MVP（②）は `opponent_knowledge=None` でも完結する設計を維持する。

---

## 7. 初期アーキタイプ（#26 で定義）

最低限（Issue 例）:
- `charizard_ex`（リザードンex）: ヒトカゲ/リザード/リザードンex + ふしぎなアメ, FIRE
- `gardevoir_ex`（サーナイトex）: ラルトス/キルリア/サーナイトex, PSYCHIC
- `mega_lucario_ex`（メガルカリオex）: リオル/ルカリオ/メガルカリオex + ソルロック/ルナトーン, FIGHTING
- `miraidon_ex`（ミライドン系）: ミライドンex, LIGHTNING

combo_rules 例: `ヒトカゲ+ふしぎなアメ` / `ソルロック+ルナトーン` / `ラルトス+キルリア` / `主軸+対応エネルギー`。

---

## 8. テスト・検証（⑤）

- 単体テスト: `tests/unit/` に主要デッキの模擬盤面 → 期待 `deck_type` を assert。
- 盤面フィクスチャは `State`/`Pokemon` を最小構成で組む（cg 実行不要）。
- 判定不能ケース（汎用カードのみ／情報僅少）で `unknown` を返すことを確認。
- 必要に応じて確認結果を `results/` に記録。

---

## 9. 未確定・要判断（着手前に潰したい点）

1. **アーキタイプ一覧の確定**（環境デッキの取捨。Tier 上位を優先するか）。
2. **card_id の埋め込み時期**（MVP は `card_id: null` で名前照合、後で埋める前提で良いか）。
3. **confidence 計算式**（単純正規化 vs 2位差減衰）。
4. **`opponent_knowledge.py` の永続範囲**（1試合内メモリのみで良いか）。
5. **既存 config 読み込み規約**: 現状 `configs/` は空・共通ローダ未整備。
   config を `opponent_modeling/` 直下に置くか `configs/` 集約かを決める。
