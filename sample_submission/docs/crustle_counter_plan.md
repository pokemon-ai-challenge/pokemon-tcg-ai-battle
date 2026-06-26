# イワパレス(Crustle)対策 実装方針（全デッキ共通）

## 1. 問題の確認

### Crustle (ID:345) のアビリティ
**「ふしぎないわのやど（Mysterious Rock Inn）」**
> 相手のポケモン{ex}の攻撃で、**このポケモン**に乗るダメージをすべて防ぐ。

→ `{ex}` がついたポケモンの攻撃が **Crustle (345) にだけ** 無効化される。  
→ Dwebble (344) や他のベンチには **普通にダメージが入る**。

### 現状の問題（全 EX デッキ共通）
- AI は常に `energy_priority[0]`（EX アタッカー）に向けて攻撃
- Crustle がバトル場にいても EX で攻撃し続ける → **0 ダメージ**
- 相手ターンに Crustle の Superb Scissors (120 ダメージ) で削られ続ける

| デッキ | 現状 Crustle 勝率 |
|-------|----------------|
| マリィのオーロンゲex | 64-68% |
| カミツオロチex | 64-68% |
| メガルカリオex | 64-68% |

---

## 2. 対策の全体像（2 段階）

### Strategy A：ボスの司令でデンジュモク(Dwebble)を引き出す（主力）

Crustle のアビリティは Crustle 自身のみに適用される。  
**ベンチの Dwebble (344, HP:70) は EX で普通に倒せる。**

```
ボスの司令 → Dwebble をバトル場に → EX アタッカーで KO（70HP = 1 撃圏内）
```

- 既に `ARCH_BOSS_TARGETS["crustle"] = [344, 756]` で Dwebble が優先ターゲット
- しかし現在 AI は「相手アクティブが Crustle で攻撃が無効」でも EX 攻撃を続ける  
  → ボスを手札に持っていても攻撃優先でボスを使わない問題がある

### Strategy B：非 ex アタッカーで Crustle を直接突破（サブ）

ボスがない場合や Dwebble が既に倒れている場合のフォールバック。  
各デッキの非 ex アタッカーを使って Crustle (HP:150) を削る。

| デッキ | 非 ex アタッカー | ダメージ | コスト | 必要回数 |
|-------|---------------|---------|-------|--------|
| マリィのオーロンゲex | Yveltal (689) | 110 | {D}{D}● | 2 回 |
| マリィのオーロンゲex | Morgrem (647) | 60 | {D}{D} | 3 回 |
| メガルカリオex | Solrock (676)※ | 70 | {F} | 3 回 |
| メガルカリオex | Hariyama (674) 能力 | — | — | Dwebble 引き出し |
| カミツオロチex | Tapu Bulu (920) | 220 | {G}{G}●● | **1 回**（Crustle 瞬殺） |
| カミツオロチex | Meganium (710) | 0 | — | 能力のみ |

※ Solrock は Lunatone がベンチにいる条件あり

---

## 3. 実装方針（汎用・デッキ非依存）

### 3-1. EX 免疫検出（共通ヘルパー）

`src/knowledge/ex_immune.py` を新規作成：

```python
from cg.api import all_card_data, State

# Crustle のみ（将来追加の可能性あり）
EX_IMMUNE_CARD_IDS: frozenset[int] = frozenset({345})

# EX カード ID のキャッシュ
_ex_card_ids: set[int] | None = None

def get_ex_card_ids() -> set[int]:
    global _ex_card_ids
    if _ex_card_ids is None:
        _ex_card_ids = {
            c.cardId for c in all_card_data()
            if c.name.lower().endswith(' ex')
        }
    return _ex_card_ids

def is_ex_pokemon_id(card_id: int) -> bool:
    return card_id in get_ex_card_ids()

def opponent_active_blocks_ex(state: State) -> bool:
    """相手のバトルポケモンが ex からのダメージを防ぐ場合 True"""
    opp_idx = 1 - state.yourIndex
    for poke in state.players[opp_idx].active:
        if poke and poke.cardId in EX_IMMUNE_CARD_IDS:
            return True
    return False
```

### 3-2. DeckPlan への非 ex フォールバック追加

`src/knowledge/deck_plan.py` の `DeckPlan` に 2 フィールドを追加：

```python
@dataclass
class DeckPlan:
    ...
    # Crustle など EX 無効化相手に向けた非 ex 優先順位（デッキごとに設定）
    non_ex_energy_priority: list[int] = field(default_factory=list)
    non_ex_active_priority: list[int] = field(default_factory=list)
```

各デッキの設定値：

```python
# マリィのオーロンゲex
MARIES_OBSTAGOON_PLAN = DeckPlan(
    ...
    non_ex_energy_priority=[689, 647, 646],   # Yveltal → Morgrem → Impidimp
    non_ex_active_priority=[689, 647, 646],
)

# メガルカリオex
LUCARIO_PLAN = DeckPlan(
    ...
    non_ex_energy_priority=[676, 673, 674],   # Solrock → Makuhita → Hariyama
    non_ex_active_priority=[676, 674, 673],   # Solrock 優先、Hariyama で Dwebble 引き出し
)

# カミツオロチex
HYDRAPPLE_PLAN = DeckPlan(
    ...
    non_ex_energy_priority=[920, 709, 149],   # Tapu Bulu → Bayleef → Applin
    non_ex_active_priority=[920, 709],        # Tapu Bulu 最優先（220 ダメ一撃）
)
```

### 3-3. エネルギー付与の動的切り替え

`src/decision/card_play_policy.py` の `choose_energy_attachment` を修正：

```python
from src.knowledge.ex_immune import opponent_active_blocks_ex

def choose_energy_attachment(obs, state, plan):
    effective_priority = plan.energy_priority  # デフォルト
    if state and opponent_active_blocks_ex(state) and plan.non_ex_energy_priority:
        effective_priority = plan.non_ex_energy_priority  # EX 免疫時は非 ex 優先
    # 以降は effective_priority を使って付与先を決定
    ...
```

### 3-4. アクティブ選択の動的切り替え

`src/decision/target_policy.py` の `choose_active` を修正：

```python
from src.knowledge.ex_immune import opponent_active_blocks_ex

def choose_active(obs, state, plan):
    effective_priority = plan.active_priority  # デフォルト
    if state and opponent_active_blocks_ex(state) and plan.non_ex_active_priority:
        effective_priority = plan.non_ex_active_priority  # EX 免疫時は非 ex 優先
    ...
```

### 3-5. にげる判断（EX → 非 ex 交代）

アクティブが EX で相手が EX 免疫の場合、逃げコストが払えるなら積極的に逃げる：

```python
from src.knowledge.ex_immune import opponent_active_blocks_ex, is_ex_pokemon_id

def choose_retreat(obs, state, plan):
    if state and opponent_active_blocks_ex(state):
        my_active = state.players[state.yourIndex].active[0]
        if my_active and is_ex_pokemon_id(my_active.cardId):
            # EX がアクティブ → 非 ex を出すために強制逃げ
            # 逃げコストが支払えるなら最優先でにげる
            ...
```

---

## 4. 実装ファイル一覧

| ファイル | 変更種別 | 内容 |
|---------|---------|------|
| `src/knowledge/ex_immune.py` | **新規** | EX 検出・EX 免疫検出ヘルパー |
| `src/knowledge/deck_plan.py` | 修正 | DeckPlan に 2 フィールド追加、各プランに設定値 |
| `src/decision/card_play_policy.py` | 修正 | `choose_energy_attachment` の動的切り替え |
| `src/decision/target_policy.py` | 修正 | `choose_active`・`choose_retreat` の動的切り替え |

---

## 5. 期待効果

| デッキ | 現状 Crustle 勝率 | 改善後見込み |
|-------|----------------|------------|
| マリィのオーロンゲex | 64-68% | 72-78% |
| メガルカリオex | 64-68% | 72-78% |
| カミツオロチex | 64-68% | **78-85%**（Tapu Bulu 一撃あり） |

全体勝率の向上幅は Crustle の対戦頻度（Tier A）に依存するが、
各デッキ平均 +5-10% 改善を想定。

---

## 6. 実装優先順位

1. `src/knowledge/ex_immune.py` 新規作成（検出ロジック）
2. `DeckPlan` フィールド追加 + 各プランに値設定
3. `choose_energy_attachment` 修正
4. `choose_active` 修正
5. `choose_retreat` 修正
6. ベンチマーク（30 試合 × Crustle デッキ × 3 デッキ）
7. 効果確認後、デッキ枚数調整を検討（Yveltal 2 枚化など）

---

## 7. 注意点

- EX 免疫の検出は「相手アクティブが Crustle かどうか」のみで OK（現状 Crustle だけ）
- Crustle が逃げてオーロンゲex 系が出てきたらすぐ通常モードに戻す
- Strategy A（ボスで Dwebble ）のほうが強力なので、ボス + 非 ex の両輪で戦う
- Munkidori (112) のダメカン移動は Crustle のアビリティに阻まれず使えるが補助的活用に留める
