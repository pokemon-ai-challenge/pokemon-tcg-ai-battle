# Pokemon TCG AI Principles

このドキュメントは、現在のディレクトリ構成やファイル名に強く依存しない形で、

- ポケカ AI を動かすうえで強かった考え方
- どういう行動が弱く、どういう行動が強いか
- 実装するときにどうプログラムへ落とすか

をまとめたものです。

このファイルの目的は「今のコードの説明」ではなく、
将来フォルダ構成が変わっても使い回せる設計メモを残すことです。

---

## 1. まず大きな考え方

強いポケカ AI は、単純に「今いちばん打点が高い手」を選ぶだけでは足りません。

特に重要なのは次の 4 つです。

1. 次のターンも攻撃できる盤面を作る
2. 相手の勝ち筋の供給源を止める
3. 不要なリスクを増やさない
4. 行動順を安定させる

言い換えると、
「このターンの一手」より「攻撃が途切れない形」と「相手の加速を止める形」を優先する方が強い、
という知見がかなり多く見つかっています。

---

## 2. 共通で強かった行動原則

### 2-1. 攻撃より前に盤面を整える

強かった順序の基本形は次です。

```text
進化 -> エネルギー付与 -> 展開/サポート -> アビリティ -> 攻撃
```

理由:

- 進化してからエネルギーを付けた方が、進化後の要求に沿って育成できる
- 攻撃前に使えるカードを使うと、選択肢が増える
- 先に殴ると、そのターンに使えたはずのリソースを捨てることがある

短い実装例:

```python
def choose_main_action(ctx):
    if ctx.can_evolve():
        return ctx.best_evolve()
    if ctx.can_attach():
        return ctx.best_attach()
    if ctx.can_play_card():
        return ctx.best_play()
    if ctx.can_use_ability():
        return ctx.best_ability()
    if ctx.can_attack():
        return ctx.best_attack()
    return ctx.end_turn()
```

注意:

- 例外はある
- ただし最初の土台としては、この順序を固定した方が挙動が安定しやすい

### 2-2. 「今殴れる相手」より「次の 2 ターンを作る」

強い AI は、
1 回だけ強く殴るより、
毎ターン攻撃が続く形を作る方を優先しやすいです。

典型例:

- 既に完成しているアタッカーより、次のアタッカーを育てる
- このターンの 10 点より、次ターンの確定攻撃を優先する
- 1 体に過剰投資せず、攻撃のバトンを用意する

短い実装例:

```python
def score_attach_target(priority_rank, current_energy):
    return priority_rank * 10 - current_energy
```

ポイント:

- 優先度だけでなく、現在どれだけ育っているかを見る
- 「足りないところを埋める」発想が強い

### 2-3. 倒される場所に資源を積みすぎない

弱い行動の代表は、
次ターン倒されそうなアクティブにさらにエネルギーや手間をかけることです。

強い行動:

- 危ないアクティブへの投資を減らす
- ベンチの後続を育てる
- 壁役や無料 retreat 役で 1 ターン稼ぐ

短い実装例:

```python
def adjusted_attach_score(base_score, target):
    if target.will_be_ko_next_turn:
        return base_score * 0.2
    return base_score
```

### 2-4. ベンチは必要な分だけ使う

ベンチは広げれば強いわけではありません。

弱い行動:

- 使わないポケモンを何となく出す
- spread ダメージの的を増やす
- 進化しない置物を増やす

強い行動:

- 明確な役割があるポケモンだけ置く
- 相手の spread や狙撃に応じて展開数を抑える
- ただし自分のエンジンに必要な枠は確保する

短い実装例:

```python
def should_bench(card_id, state, bench_priority, max_bench_size):
    if state.bench_count >= max_bench_size:
        return False
    return card_id in bench_priority
```

### 2-5. 相手の主力よりエンジン役を止める

ポケカでは、前にいるアタッカーそのものより、
そのアタッカーを支えているカードを止めた方が強い場面が多いです。

狙いたい対象:

- 進化元
- エネ加速役
- ドロー役
- 手札補充役
- エネルギー転送役

つまり「相手の今の盤面」ではなく、
「次のターンに何をしてくるか」を止める発想です。

短い実装例:

```python
def score_boss_target(card_id, priority_targets):
    if card_id not in priority_targets:
        return 0
    return 10000 - priority_targets.index(card_id) * 1000
```

### 2-6. 妨害札は刺さるタイミングで使う

手札干渉や妨害サポートは、あるから使うのではなく、
刺さるタイミングで使う方が強いです。

弱い行動:

- 相手手札が少ないのに干渉を打つ
- 相手の事故を直してしまう

強い行動:

- 相手が準備完了しているときに干渉する
- 相手の次ターンの最大出力を下げる

短い実装例:

```python
def score_disruption(opponent_hand_count, opponent_setup_score):
    if opponent_hand_count <= 2:
        return 3
    if opponent_setup_score >= 2:
        return 35
    return 18
```

### 2-7. 重要札は雑に捨てない

これもかなり共通的です。

弱い行動:

- 進化ラインを早い段階で捨てる
- 主力や後続を失う
- 「今の 1 枚の都合」で将来の勝ち筋を消す

強い行動:

- 捨てたくないカードの集合を持つ
- 捨てるなら代替の利く札から捨てる

短い実装例:

```python
def discard_priority(card_id, protected_ids):
    return 1 if card_id in protected_ids else 0
```

0 のカードから先に捨てる、という使い方がしやすいです。

---

## 3. 共通で避けたい弱い行動

AI 実装でまず消したいのは次のような挙動です。

- 進化できるのに進化せず殴る
- 必要のないベンチ展開をする
- 完成済みの 1 体にだけエネルギーを集める
- 次ターン倒されるアクティブに過剰投資する
- 重要な進化ラインをコストで切る
- 相手の主力だけを見て、加速役や進化元を放置する
- 干渉札を無条件で即打ちする
- スコアが同じときに意味のないランダム行動をする

これらを消すだけでも、かなり挙動が安定します。

---

## 4. 実装に落とすときの考え方

### 4-1. まず「優先順位」と「禁止条件」を分ける

実装では、次の 2 種類を分けると整理しやすいです。

1. 何を優先するか
2. 何をやらないか

例:

- 優先: 主力アタッカーにエネを付けたい
- 禁止: 既に十分エネがある相手には付けない

短い実装例:

```python
def score_target(target):
    if target.energy >= target.energy_cap:
        return 0
    return target.priority_rank * 10 - target.energy
```

### 4-2. ルールは 3 層に分けると保守しやすい

ファイル名は将来変わってよいですが、概念としては次の 3 層に分けると扱いやすいです。

#### 層 1. 共通ルール

例:

- 行動順
- エネ付与の基本式
- 攻撃選択の基本
- discard の基本

#### 層 2. デッキ固有ルール

例:

- どの Basic を先に出したいか
- どの進化ラインを主軸にするか
- どのカードを守るか

#### 層 3. 対面固有ルール

例:

- spread 対面ではベンチを絞る
- 特定の加速役をボスで引く
- EX 無効対面では非 EX に寄せる

この 3 層を混ぜすぎない方が、後で壊れにくいです。

### 4-3. スコアは「固定点 + 補正」が扱いやすい

複雑な完全数式に寄せるより、

- まず基礎点
- そのあと少数の補正

の方が調整しやすいです。

短い実装例:

```python
def score_play(card):
    score = BASE_SCORE_BY_TYPE[card.type]

    if card.is_bench_pokemon:
        score = 30 if card.is_in_bench_priority else 0

    if card.is_boss and card.boss_rush_active:
        score = 40

    if card.is_disruption:
        score = adjust_disruption_score(score, card)

    return score
```

### 4-4. 迷ったら 0 点に落とす

実装を安定させるコツは、
「まあ選んでもいいか」を減らすことです。

特に強かったのは、
不要な行動を 0 点に落とす設計です。

例:

- bench priority 外のポケモンは 0
- 過剰エネ先は 0
- 明らかに不利な妨害は低得点

この設計は、事故的な変な行動をかなり減らします。

### 4-5. 理由をログに残す

後から改善するなら、
「何を選んだか」だけでなく
「なぜそれを選んだか」が必要です。

短い実装例:

```python
def choose_with_trace(label, candidates):
    best = max(candidates, key=lambda x: x.score)
    trace(label, best.index, f"score={best.score} reason={best.reason}")
    return best.index
```

ログがないと、
Kaggle の試合ログやローカル対戦を見ても改善しにくくなります。

---

## 5. 最小構成の実装テンプレート

以下は、かなり短くした実装イメージです。

```python
def choose_main_action(state):
    actions = classify_actions(state)

    if actions.evolves:
        return max(actions.evolves, key=score_evolve)

    if actions.attaches:
        return max(actions.attaches, key=score_attach)

    if actions.plays:
        return max(actions.plays, key=score_play)

    if actions.abilities:
        return actions.abilities[0]

    if should_retreat_first(state):
        return best_retreat(state)

    if actions.attacks:
        return max(actions.attacks, key=score_attack)

    if actions.retreats:
        return best_retreat(state)

    return actions.end[0]
```

このテンプレートだけでも、
「進化しない」「展開しない」「何となく殴る」型の弱い AI を避けやすくなります。

---

## 6. 改善するときの順番

改善の順番も重要です。

おすすめの順番:

1. 合法手を壊していないか直す
2. 明らかに弱い行動を 0 点化する
3. 行動順を安定させる
4. デッキ固有の優先順位を入れる
5. 対面固有の例外を入れる
6. 最後に探索や深い評価を試す

経験上、探索より前に

- 行動順
- エネ配分
- ベンチ制御
- エンジン妨害

を整えた方が伸びやすいです。

---

## 7. 実装メモとして使うときのチェックリスト

新しいルールを足す前に確認したいこと:

- これは共通ルールか、デッキ固有か、対面固有か
- 優先順位で表現できるか
- 0 点にすべき禁止行動は何か
- 次ターンの攻撃継続に寄与するか
- 相手の供給源を止める行動か
- spread や狙撃に対してリスクを増やしていないか
- 重要札を無駄に捨てる副作用はないか
- ログに理由を残せるか

---

## 8. ひとことでまとめると

ポケカ AI を強くするうえで共通的に効きやすいのは、

- 先に盤面を整える
- 攻撃が続く形を作る
- 倒される場所に積みすぎない
- ベンチを無駄に広げない
- 相手の主力よりエンジン役を止める
- 妨害を刺さるタイミングで使う
- やってはいけない行動を 0 点に落とす

という原則です。

実装では、これを

- 共通ルール
- デッキ固有ルール
- 対面固有ルール

に分けて持つと、改善しやすくなります。
