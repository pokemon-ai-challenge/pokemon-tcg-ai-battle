# PR #90 レビュー対応方針（2026-07-25）

対象: [PR #90 "Feature/rule based fix"](https://github.com/pokemon-ai-challenge/pokemon-tcg-ai-battle/pull/90)（`feature/rule-based-fix`）へのレビューコメント2件。
レビュー本文中のリンクは `#1`/`#2` になっているが、実体はこのPR内の指摘事項1・2（Issue/PR番号ではない）。

## 前提として確認した事実

レビュー指摘の技術的根拠はコード読解で両方とも裏付けが取れた。加えて、指摘1については
**レビューが引用した「PR本文の結論」自体が、直近のコミットで既に古くなっている**ことが判明した。

- 最新コミット `65f83f2`（2026-07-25、branch HEAD）: "Wire draw-engine bench reservation and
  fire Run Away Draw every turn"。コミットメッセージに **"Per deck-owner direction"**
  **"Shipping per deck-owner decision"** と明記されており、"ミラーでデッキアウト率〜82%"を
  承知の上で `_DRAW_ABILITY_MIN_DECK` の既定を `14`→`0` に変更したのは**デッキ担当者の意図的判断**。
- しかし [PR #90 の本文](https://github.com/pokemon-ai-challenge/pokemon-tcg-ai-battle/pull/90)は
  このコミットより前の状態（"全実験を破棄し `floor=14` に復元" で締めた 2026-07-24 時点の記述）の
  ままで、**最新コミットの内容を反映して更新されていない**。
- `ability.py` 内のコメント（34行目付近）にも同じ2026-07-25の方針転換が書かれている
  ([ability.py:22-33](../../../ptcg_ai/rule_based/main_turn_parts/priorities/ability.py)）。

つまりレビューが引用した「PR本文の結論（floor=14）」は、レビュー時点で既に**PR本文が古いだけ**で、
デッキ担当者の最新の意図は「floor=0（毎ターン発動、ミラー82%デッキアウトのリスクは承知の上で採用）」
で確定している。ただしレビューの技術的指摘——**「env変数のデフォルト値」という置き場所そのものが
Kaggle採点ハーネスには効かず、コミットされる既定値がどこにも明示されていない**——は、
どちらの値を採用するにせよ有効な指摘であり、修正が必要。

## 指摘1（🔴要対応）: 出荷既定値がコミットされておらず不透明

### 現状
```python
# sample_submission/ptcg_ai/rule_based/main_turn_parts/priorities/ability.py:34-35
_DRAW_ABILITY_MIN_DECK = int(os.environ.get("PTCG_DRAW_ABILITY_MIN_DECK", "0"))
_DRAW_ABILITY_MAX_HAND = int(os.environ.get("PTCG_DRAW_ABILITY_MAX_HAND", "999"))
```
Kaggle採点ハーネスは `PTCG_DRAW_ABILITY_MIN_DECK` を設定しないため、提出物は常に
このPythonソース中の `"0"` という文字列リテラルで実際の挙動が決まる。grep で
`_DRAW_ABILITY_MIN_DECK` を検索しても、実際に効く値がどこにも「設定」として存在せず、
`os.environ.get` の第2引数に埋もれている。`deck_plan.py` 側（担当A・デッキ固有知識の置き場）
には対応する値が無く、`OPPONENT_EFFECT_LOCK_ENERGY_IDS` などの既存パターンと非対称。

### 採用する方針: **Option B（floor=0を意図として明示）**
上記の通りデッキ担当者の最新判断は floor=0 なので、これを「隠れたenv既定値」ではなく
`deck_plan.py` に**コミットされた宣言**として明示する。`RESERVED_BENCH_SLOTS_FOR_DRAW_ENGINE`
と全く同じ配線パターン（`deck_plan.py` → `profile_registry.py` の `getattr` アクセサ →
利用側)を踏襲する。

### 修正案（diff）

**`sample_submission/decks/new_deck/deck_plan.py`** — 末尾（エネルギー周回コンボの節の後ろ）に追記:
```python
# ---------------------------------------------------------------------------
# にげあしドロー（ドロー系特性）の発動可否の歯止め
# ---------------------------------------------------------------------------
#
# 山札がこの枚数以下ならドロー系特性（にげあしドロー等）を温存する。
# 【2026-07-25、デッキ担当判断】毎ターン欠かさず発動する（floor=0）を採用。
# 根拠: にげあしドローは発動のたびにノコッチ+ノココッチ(+付属カード)が山札へ戻るため、
# 山札1枚からでも2枚に補充される（実測: deck 1->2）。RESERVED_BENCH_SLOTS_FOR_DRAW_ENGINE
# による予備ノコッチのベンチ確保とセットで運用する。
# 既知のリスク（ミラー自己対戦・40戦実測）: floor=0はデッキアウト率82.5%
# （floor=14なら40%）。原因はループ維持率が~45%止まりで、稼働不十分なまま毎ターン
# 撃つと安定域(山札2枚)に速く到達し穴が致命傷になるため。ミラーは両者同時消耗するため
# 非対称マッチアップでの真価は測れておらず、これを承知の上でfloor=0を出荷する。
# 保守的な値に戻したい場合は env `PTCG_DRAW_ABILITY_MIN_DECK=14` で上書きできる
# （このデッキ内固定値ではなく "任意の実行時オーバーライド" に格下げ）。
DRAW_ABILITY_MIN_DECK = 0
DRAW_ABILITY_MAX_HAND = 999
```

**`sample_submission/ptcg_ai/shared/profile_registry.py`** — `get_opponent_effect_lock_energy_ids`
の並びに追記:
```python
def get_draw_ability_deck_floor() -> int:
    """decks.active.deck_plan.DRAW_ABILITY_MIN_DECK を返す。

    ドロー系特性（にげあしドロー等）を温存する山札残枚数の閾値。デッキが未定義なら0。
    """
    return int(getattr(active.deck_plan, "DRAW_ABILITY_MIN_DECK", 0))


def get_draw_ability_max_hand() -> int:
    """decks.active.deck_plan.DRAW_ABILITY_MAX_HAND を返す。デッキが未定義なら999（実質無効）。"""
    return int(getattr(active.deck_plan, "DRAW_ABILITY_MAX_HAND", 999))
```

**`sample_submission/ptcg_ai/rule_based/main_turn_parts/priorities/ability.py`** — 既存の
env読み取り部分を置き換え:
```python
# env は「任意の実行時オーバーライド」。コミットされる既定値は
# decks/new_deck/deck_plan.py の DRAW_ABILITY_MIN_DECK / DRAW_ABILITY_MAX_HAND
# を参照（提出物の実挙動はここで確定する）。
_env_min_deck = os.environ.get("PTCG_DRAW_ABILITY_MIN_DECK")
_env_max_hand = os.environ.get("PTCG_DRAW_ABILITY_MAX_HAND")
_DRAW_ABILITY_MIN_DECK = (
    int(_env_min_deck) if _env_min_deck is not None else profile_registry.get_draw_ability_deck_floor()
)
_DRAW_ABILITY_MAX_HAND = (
    int(_env_max_hand) if _env_max_hand is not None else profile_registry.get_draw_ability_max_hand()
)
```
（`profile_registry` は既にこのファイルで import 済み。）

### 副次的な確認事項（改善案B、優先度中）
現状 `board.py` の `_RESERVED_BENCH_SCORE=30 > _EVOLUTION_BASE_SCORE=20` は常時有効。
`floor=0` を正式採用するならこのままで整合する（毎ターン発動する設計なので予備ノコッチの
ベンチ確保は必須）。ただし `PTCG_DRAW_ABILITY_MIN_DECK=14` で保守的値に env オーバーライド
した場合、ベンチ予約はそのまま有効に残り「安全floor＋攻めのベンチ予約」という中途半端な
組み合わせになる（レビュー指摘の懸念そのもの）。実害は限定的（予備を余分に展開するだけ）だが、
気になるならベンチ予約側も同じ `deck_plan.py` の値を見て `floor > 0` の時は
`RESERVED_BENCH_SLOTS_FOR_DRAW_ENGINE=0` 相当に倒す、等の連動は次の課題として切り出す
（今回のPRスコープ外）。

### PR本文の更新
`gh pr edit 90 --body-file <更新後の本文>` で、末尾に以下のセクションを追記する（要約）:

> ## 追記（2026-07-25）: floor=0 を正式採用・出荷確定
>
> 上記「実験は全破棄しfloor=14に復元」から方針転換。デッキ担当判断により、
> ミラー自己対戦でのデッキアウト率82.5%（floor=14なら40%）を承知の上で
> `DRAW_ABILITY_MIN_DECK=0`（毎ターン欠かさず発動）を `decks/new_deck/deck_plan.py` に
> コミットされた既定値として採用した。理由: ミラーは両者同時消耗するため
> 「対ex火力ビルド」戦略の真価を構造的に過小評価する（non-mirror環境での検証は
> 別ブランチのmeta-deckベンチが必要、現状未検証）。保守的な値に戻す場合は
> env `PTCG_DRAW_ABILITY_MIN_DECK=14` で上書き可能。

### テスト
`tests/unit/` に `test_draw_ability_deck_floor.py`（仮）を追加し、
(1) env未設定時は `deck_plan.DRAW_ABILITY_MIN_DECK` の値（0）が使われる、
(2) env設定時はenvが優先される、の2点を回帰テスト化する（現状この値の直接テストが無い）。

---

## 指摘2（🟡推奨）: `proposals.py` のモジュールグローバル状態

### 現状（確認済み）
```python
# sample_submission/ptcg_ai/rule_based/main_turn_parts/proposals.py:56-57, 85, 90-92, 95-96
_setup_turn: int | None = None
_setup_count = 0

def decide(obs: Observation) -> list[int]:
    global _setup_turn, _setup_count
    ...
    turn = obs.current.turn if obs.current is not None else None
    if turn != _setup_turn:
        _setup_turn = turn
        _setup_count = 0
    ...
```
レビュー指摘の通り、`turn != _setup_turn` で新ターンごとに自己修復するため単一ゲーム内の
実害は無いが、1プロセス内で複数ゲームを回す自己対戦・pytest では、ゲームBの1ターン目が
ゲームAの最終ターン番号と一致すると誤ってリセットされない（ターン番号は1から再スタートする
ため通常は早期に自己修復するが、テストの決定的初期化・並行実行には向かない）。

### 修正案
レビュー提案の `_SetupTurnBudget` dataclass をそのまま採用する。

```python
# sample_submission/ptcg_ai/rule_based/main_turn_parts/proposals.py
from dataclasses import dataclass

@dataclass
class _SetupTurnBudget:
    turn: int | None = None
    count: int = 0

    def take(self, turn: int | None, cap: int) -> bool:
        if turn != self.turn:
            self.turn, self.count = turn, 0
        if self.count >= cap:
            return False
        self.count += 1
        return True

    def reset(self) -> None:
        self.turn, self.count = None, 0


_setup_budget = _SetupTurnBudget()


def reset_turn_state() -> None:
    """新規ゲーム開始時に呼ぶ。自己対戦・テストでのプロセス間状態汚染を防ぐ。"""
    _setup_budget.reset()


def decide(obs: Observation) -> list[int]:
    proposals = collect_proposals(obs)
    if SETUP_BEFORE_ATTACK:
        turn = obs.current.turn if obs.current is not None else None
        setup = [p for p in proposals if p.category in _SELF_DEPLETING_SETUP]
        if setup and _setup_budget.take(turn, _SETUP_ACTION_CAP):
            return max(setup, key=_total_score).select
    best = max(proposals, key=_total_score)
    return best.select
```

### 呼び出し元の配線
`rule_based_agent.py` の `agent()` が新規ゲーム開始点（`obs.select is None` でデッキ選択を
返す分岐、[rule_based_agent.py:29-30](../../../ptcg_ai/rule_based/rule_based_agent.py)）に
該当するので、ここで `reset_turn_state()` を呼ぶ:

```python
# rule_based_agent.py
from ptcg_ai.rule_based.main_turn_parts import proposals

def agent(obs: Observation) -> list[int]:
    if obs.select is None:
        proposals.reset_turn_state()
        return _select_deck()
    return selector.select_action(obs, _full_deck())
```
（`selector.select_action` → ルールベースルート → `proposals.decide` への呼び出し経路が
既にあるはずなので、import追加のみで完結する想定。実装時に経路を要確認。）

挙動は現状と等価（`turn != self.turn` 判定は既存の `turn != _setup_turn` と同一）で、
回帰リスクはほぼゼロ。回帰テストとして「ゲーム跨ぎで `_setup_budget` が明示リセットされる」
ケースを1本追加する。

---

## 実装チェックリスト

- [ ] `deck_plan.py` に `DRAW_ABILITY_MIN_DECK=0` / `DRAW_ABILITY_MAX_HAND=999` を追加
- [ ] `profile_registry.py` に `get_draw_ability_deck_floor()` / `get_draw_ability_max_hand()` を追加
- [ ] `ability.py` の既定値取得を `deck_plan` 経由に変更（env は上書き専用に格下げ）
- [ ] 回帰テスト `test_draw_ability_deck_floor.py` を追加
- [ ] PR #90 本文に「2026-07-25: floor=0 を正式採用」セクションを追記
- [ ] `proposals.py` に `_SetupTurnBudget` を導入し `reset_turn_state()` を公開
- [ ] `rule_based_agent.py` の `obs.select is None` 分岐で `reset_turn_state()` を呼ぶ
- [ ] `pytest` フル実行で回帰なしを確認

## 確認したい点（ユーザー判断）

1. 指摘1の対応方針は「floor=0を正式採用してPR本文を更新」（Option B）でよいか。
   コミットメッセージから読み取れる最新意図はこちらだが、念のため確認。
2. 指摘2（`proposals.py` のリファクタ）も同じPRでまとめて対応するか、別PRに分けるか。
