# 戦略改善 実装計画

> 前提: `issue_analysis.md` の分析に基づく  
> 作成日: 2026-06-26

---

## アーキテクチャ方針

各問題を独立した「判断モジュール」として実装し、`main_policy.py` / `setup_policy.py` からルール呼び出し形式で差し込む。

```
src/decision/
  risk_evaluator.py      ← 新規: KOリスク・エネルギー上限・ベンチ価値評価
  main_policy.py         ← 改修: _pick_best_attach, choose_main_action
  setup_policy.py        ← 改修: choose_setup_bench
```

---

## Issue A-1: KOリスクを無視したエネルギー付与（優先度: 高）

### 担当ファイル
- **実装**: `src/decision/risk_evaluator.py`（新規）
- **呼び出し修正**: `src/decision/main_policy.py:_pick_best_attach`

### やること

```python
# risk_evaluator.py に追加する関数

def will_be_ko_next_turn(state: State, pokemon: Pokemon) -> bool:
    """相手が次ターンにこのポケモンをKOできるかを推定する。
    
    判定ロジック:
    1. 相手アクティブの最大ダメージを取得
    2. pokemon.hp <= 相手最大ダメージ なら True
    3. 弱点補正は簡易的に2倍で計算
    """
```

### `_pick_best_attach` への組み込み

```python
# エネルギーを付けるターゲットがアクティブの場合、次ターンKOされるなら
# priority_score を大幅に下げてベンチのポケモンを優先する

if target_is_active and will_be_ko_next_turn(state, active_poke):
    score = priority_score * 2  # 大幅減点（通常は priority_score * 10）
```

### 検証方法
- ベンチマークで「エネルギー喪失ターン数」を追跡
- KOされる直前ターンにベンチへエネルギーが回ったかのログ確認

---

## Issue A-2: KOデメリットの理解不足（優先度: 高）

### 担当ファイル
- **実装**: `src/decision/evaluator.py`（改修）

### やること

1. **`prize_risk` の改善**: HP消耗ベースから「次ターンKO確率」ベースへ変更

```python
# 現行
damage_taken_ratio = 1.0 - (my_active.hp / my_active.maxHp)
total -= prizes_on_ko * damage_taken_ratio * self.w.prize_risk

# 改善案
from src.decision.risk_evaluator import will_be_ko_next_turn
if will_be_ko_next_turn(state, active):
    total -= prizes_on_ko * self.w.prize_risk * 1.5  # KO確定ならフルペナルティ
```

2. **ベンチ消滅リスク**: アクティブがKOされた後ベンチが空になると負けが確定するリスクを加算

```python
bench_count = len(my.bench)
if bench_count == 0 and will_be_ko_next_turn(state, active):
    total -= 100.0  # ほぼ負け確定シグナル
```

3. **エネルギー喪失コスト**: KOされるポケモンについているエネルギー枚数 × ペナルティ

```python
if will_be_ko_next_turn(state, active):
    energy_loss = len(active.energies)
    total -= energy_loss * 8.0  # エネルギー1枚喪失=8点ペナルティ
```

---

## Issue B-1: ベンチの戦略的な埋め方（優先度: 中）

### 担当ファイル
- **改修**: `src/decision/setup_policy.py:choose_setup_bench`
- **改修**: `src/decision/main_policy.py:_pick_best_play`

### やること

#### setup_policy.py

```python
# 補完時も bench_priority に入っているポケモン優先、それ以外は出さない
# min_count を満たせない場合のみやむを得ず追加

# 追加ロジック: bench_priority 外のポケモンは min_count が要求する場合のみ
```

#### main_policy.py

`_pick_best_play` の POKEMON スコアを修正:

```python
if card_data.cardType == CardType.POKEMON:
    if bench_full:
        score = 0
    elif card_id in plan.bench_priority:
        score = BENCH_PRIORITY_POKEMON_SCORE
    else:
        score = 0  # ← 5 → 0 に変更（bench_priority 外は出さない）
```

### 注意点
- `min_count` が 1 以上の SETUP フェーズでは bench_priority 外でも出す必要がある（変更なし）
- メインフェーズの任意プレイ時のみスコア 0 にする

---

## Issue B-2: 進化前ポケモンのアクティブ継続問題（優先度: 中）

### 担当ファイル
- **改修**: `src/decision/main_policy.py:_active_is_buffer`
- **改修**: `src/knowledge/deck_plan.py:DeckPlan`

### やること

#### DeckPlan に `buffer_pokemon_ids` を追加

```python
@dataclass
class DeckPlan:
    ...
    # 逃げエネを問わず「バッファ専用」として扱うポケモン ID
    # 進化先アタッカーが準備できたら優先的に引っ込める
    buffer_pokemon_ids: list[int] = field(default_factory=list)
```

例（MARIES_OBSTAGOON_PLAN）:
```python
buffer_pokemon_ids=[646, 103],  # マリィのベロバー, スノーバー
```

#### `_active_is_buffer` の修正

```python
def _active_is_buffer(state, plan) -> bool:
    ...
    # 逃げエネ0 OR buffer_pokemon_ids に含まれる、の両方を対象にする
    active_card = card_db.get(active.id)
    is_zero_retreat = active_card and active_card.retreatCost == 0
    is_declared_buffer = active.id in plan.buffer_pokemon_ids
    if not (is_zero_retreat or is_declared_buffer):
        return False
    ...
```

#### リトリートコスト支払い判断

buffer_pokemon_ids のポケモンがアクティブで逃げエネを持っている場合、
エネルギーが手持ちにあればリトリートを許可するロジック追加（router.py で呼ぶ）。

---

## Issue B-3: エネルギー過剰付与（優先度: 中）

### 担当ファイル
- **改修**: `src/decision/main_policy.py:_pick_best_attach`
- **参照**: `src/decision/risk_evaluator.py`（新規）

### やること

```python
# risk_evaluator.py に追加

def energy_cap(card_id: int) -> int:
    """攻撃 + 逃げコストの合計（= 最大有効エネルギー枚数）を返す。
    
    例: オーロンゲex(攻撃2 + 逃げ2) → cap=4
    この枚数以上付けても無駄
    """
    card = card_db.get(card_id)
    if not card:
        return 3
    attack_cost = _get_energy_needed_for_attack(card_id)
    retreat_cost = card.retreatCost or 0
    return attack_cost + retreat_cost
```

`_pick_best_attach` での適用:

```python
from src.decision.risk_evaluator import energy_cap

# エネルギーが cap に達しているポケモンへの付与はスコアを大幅減点
cap = energy_cap(target_card_id)
if energy_count >= cap:
    score = 0  # これ以上つけても無駄
```

---

## 進化軸の共通化（メモ）

マリィのオーロンゲex と他デッキで「進化アタッカーの育成ロジック」は共通化できる。

現行は DeckPlan の `evolution_priority` + `evolution_lines` が同じ構造なので、
`main_policy._pick_best_evolve` はそのまま流用可能。デッキ固有のロジックは DeckPlan の設定値で吸収している（追加実装不要）。

---

## 実装順序

```
Phase 6-1: risk_evaluator.py 新規作成
  └─ will_be_ko_next_turn()
  └─ energy_cap()

Phase 6-2: evaluator.py 改修
  └─ prize_risk を次ターンKO確率ベースに変更
  └─ ベンチ消滅リスク追加
  └─ エネルギー喪失コスト追加

Phase 6-3: main_policy.py 改修
  └─ _pick_best_attach に KOリスク判定組み込み
  └─ _pick_best_attach に energy_cap 上限追加
  └─ _pick_best_play でベンチ外ポケモンのスコアを 0 に

Phase 6-4: deck_plan.py + setup_policy.py 改修
  └─ buffer_pokemon_ids フィールド追加
  └─ _active_is_buffer の条件拡張

Phase 6-5: ベンチマーク計測
  └─ 全 Tier デッキ対 30 試合ずつ再計測
  └─ 基準: 改修前 vs 改修後の勝率比較
```

---

## 各 Issue を担当するエージェントへの依頼文テンプレート

### Agent: risk_evaluator 実装（Phase 6-1）

```
sample_submission/src/decision/risk_evaluator.py を新規作成してください。

実装する関数:
1. will_be_ko_next_turn(state: State, pokemon: Pokemon) -> bool
   - 相手アクティブの最大攻撃ダメージを取得
   - pokemon.hp <= 最大ダメージ（弱点2倍を加味）なら True
   - get_attack_db(), get_card_db() を使用
   
2. energy_cap(card_id: int) -> int
   - 攻撃コスト（最小エネルギー）+ 逃げコスト の合計を返す
   - get_card_db(), get_attack_db() を使用

参照: src/decision/evaluator.py の _get_energy_needed_for_attack() と同じパターン
```

### Agent: evaluator.py 改修（Phase 6-2）

```
sample_submission/src/decision/evaluator.py の BoardEvaluator.score() を改修してください。

変更点 (issue_analysis.md 問題4の対応):
1. prize_risk 計算を「次ターンKO確率」ベースに変更
   - from src.decision.risk_evaluator import will_be_ko_next_turn を追加
   - will_be_ko_next_turn が True なら prize_risk をフルペナルティ適用
   - False の場合は現行の damage_taken_ratio ベースを継続

2. ベンチ消滅リスクスコアを追加
   - bench が空 AND アクティブが次ターンKO → -100.0

3. エネルギー喪失コストを追加  
   - will_be_ko_next_turn が True → active のエネルギー枚数 × 8.0 減点
```

### Agent: main_policy.py 改修（Phase 6-3）

```
sample_submission/src/decision/main_policy.py を改修してください。

変更点 1 (_pick_best_attach の KOリスク対応):
- from src.decision.risk_evaluator import will_be_ko_next_turn, energy_cap を追加
- アクティブへのエネルギー付与で next_turn_ko=True なら score を priority_score * 2 に減点
- エネルギーが energy_cap(target_card_id) に達していたら score = 0

変更点 2 (_pick_best_play のベンチ制限):
- bench_priority 外 POKEMON の score を 5 → 0 に変更
```

### Agent: buffer_pokemon_ids 実装（Phase 6-4）

```
以下の 2 ファイルを改修してください。

1. sample_submission/src/knowledge/deck_plan.py
   - DeckPlan に buffer_pokemon_ids: list[int] フィールドを追加（デフォルト空リスト）
   - MARIES_OBSTAGOON_PLAN: buffer_pokemon_ids=[646, 103]
   - 他デッキはデフォルト（空）のまま

2. sample_submission/src/decision/main_policy.py の _active_is_buffer()
   - 条件を「逃げエネ0 OR buffer_pokemon_ids に含まれる」に拡張
   - どちらかを満たせばバッファとして判定してベンチのアタッカー準備完了時に引っ込める
```
