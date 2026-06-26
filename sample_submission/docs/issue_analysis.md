# 戦略的問題点の分析レポート

> 対象バージョン: Phase 5-3 完了時点  
> 検証日: 2026-06-26

---

## 問題 1: ベンチを戦略的に埋めていない

### 根拠コード

**`setup_policy.py:51-58`** — `bench_priority` に含まれないポケモンは最低人数を満たすために何でも出してしまう：

```python
# 最低限の選択数を満たすために残りを補完
if len(chosen) < min_count:
    for i in range(len(options)):
        if i not in used:
            chosen.append(i)  # ← bench_priority 外のカードでも追加
```

**`main_policy.py:237-243`** — `bench_priority` 外のポケモンは `score=5`（デフォルト）で出てしまう：

```python
if card_data.cardType == CardType.POKEMON:
    if bench_full:
        score = 0
    elif card_id in plan.bench_priority:
        score = BENCH_PRIORITY_POKEMON_SCORE  # 30
    else:
        score = SCORE.get(card_data.cardType, 5)  # ← 5 のままベンチに出る
```

### 判定: ✅ 問題あり

`bench_priority` に含まれないポケモン（例: 余分な基本ポケモン）が SCORE=5 でカードプレイされてしまう。ベンチに出しても進化できないポケモンや、ドラパルトのベンチ散布の的になるだけのポケモンを出してしまう可能性がある。

---

## 問題 2: KOリスクを無視してエネルギーを付けている

### 根拠コード

**`main_policy.py:63-101`** — エネルギー付与先の選択に「次ターンKOされるか」の評価がない：

```python
def _pick_best_attach(obs, attach_options):
    # energy_priority + 現在のエネルギー枚数でスコア計算
    score = priority_score * 10 - energy_count
    # ← 相手が次ターンに KO できるかどうかは一切考慮していない
```

**`evaluator.py:121-128`** — `prize_risk` はHP消耗割合×サイド価値だが、「次ターン1撃KO」の判定ではない：

```python
damage_taken_ratio = 1.0 - (my_active.hp / my_active.maxHp)
total -= prizes_on_ko * damage_taken_ratio * self.w.prize_risk
```

### 判定: ✅ 問題あり

例: アクティブのHPが残り少なく、相手が次ターンにKOできる状況でも、アクティブへのエネルギー付与を止める判断をしない。エネルギーをベンチ控えに回す判断ができていない。

---

## 問題 3: 進化前にバトル場に出てしまう（オーロンゲ例）

### 根拠コード

**`deck_plan.py:213`** — `active_priority=[646, 103]` でマリィのベロバーをセットアップ時に出す設定：

```python
MARIES_OBSTAGOON_PLAN = DeckPlan(
    active_priority=[646, 103],  # Marnie's Zigzagoon(646), Snover(103)
    ...
)
```

**`main_policy.py:311-314`** — バッファ判定の条件が「逃げエネ0」のみ：

```python
def _active_is_buffer(state, plan) -> bool:
    active_card = card_db.get(active.id)
    if not active_card or active_card.retreatCost != 0:
        return False  # ← 逃げエネ0でなければバッファとして引っ込めない
```

### 判定: ✅ 問題あり（条件付き）

マリィのベロバー（逃げエネ1）がアクティブにいる場合、`_active_is_buffer` の条件を満たさず引っ込められない。ギモーに進化していない状態でもアタッカーとして戦わされ、エネルギーが無駄になるケースがある。

---

## 問題 4: KOされたときのデメリットの理解不足

### 根拠コード

**`evaluator.py`** — `prize_risk` の計算はHP消耗ベースだが、KO後のベンチ消滅（ポケモンが全滅するリスク）は評価していない：

- KOされたときにエネルギーが失われる（=次のアタッカーに引き継がれない）ことのペナルティなし
- KOされたポケモンが EX の場合、相手が2枚サイドを取ることへの動的評価が弱い
- アクティブがKOされた後の「ベンチにポケモンがいない = 負け」リスクの重み不足

### 判定: ✅ 問題あり

特に EX ポケモンをアクティブに出す際の「2枚サイドを失うリスク」の評価が、単純な HP 割合計算に止まっている。

---

## 問題 5: メインポケモンへのエネルギー過剰付与

### 根拠コード

**`main_policy.py:93`** — エネルギーが少ないほど優先するが、上限チェックがない：

```python
score = priority_score * 10 - energy_count
# ← 攻撃に必要な枚数 + 逃げコストを超えても付け続ける
```

例: オーロンゲex（攻撃コスト2エネ、逃げコスト2エネ）の場合、4枚で十分だが5枚目以降も付けようとする。

### 判定: ✅ 問題あり

攻撃に必要なエネルギー + 逃げコストを上限として、超過分はベンチのサブアタッカー育成に回すべき。

---

## まとめ

| # | 問題 | 判定 | 影響度 | 優先度 |
|---|------|------|--------|--------|
| 1 | ベンチの戦略的な埋め方 | ✅ 問題あり | 中 | B |
| 2 | KOリスクを無視したエネ付与 | ✅ 問題あり | 高 | A |
| 3 | 進化前ポケモンをアクティブ継続 | ✅ 問題あり | 中 | B |
| 4 | KOデメリットの理解不足 | ✅ 問題あり | 高 | A |
| 5 | エネルギー過剰付与 | ✅ 問題あり | 中 | B |

すべての指摘は現行コードに根拠があり、実装上の問題として確認できた。
