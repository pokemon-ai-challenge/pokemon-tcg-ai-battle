# 攻撃を通すための準備行動探索(Attack-Enabling Search)— 設計 + 実装計画

作成日: 2026-07-22 / 対象: `ptcg_ai/ml_policy/ml_policy_agent.py` に新レイヤを1つ追加(新規 `ptcg_ai/search/attack_plan.py`)/
状態: **v1(2026-07-22)— codex(gpt-5.6-sol)によるレビューを経て、深さ2-3 DFS + value網の初版設計から、
shadow計測ゲート付きの段階的因果探索(veto→1-prep→将来のdepth2)へ縮小。詳細は §0。**

関連ドキュメント:
- [step2-lethal-hybrid.md](../ml-value-network/step2-lethal-hybrid.md) — 既存の確定リーサル探索ハイブリッド。本レイヤはこの直後に差し込む
- [step2-design.md](../ml-value-network/step2-design.md) — 模倣ポリシー本体・担当領域境界(rule_based/action_selection を変更しない方針)
- 参照実装(変更しない): [`../../../ptcg_ai/search/lethal_simple.py`](../../../ptcg_ai/search/lethal_simple.py) / value網 [`../../../ptcg_ai/learning/value_model.py`](../../../ptcg_ai/learning/value_model.py)

---

## 0. v1改訂の経緯(なぜ縮小したか)

初版(深さ2〜3のDFS + `value_model.predict_win_prob_from_state` で葉を採点)を、実装着手前に
codex(gpt-5.6-sol、CLI経由)へ批判的レビューを依頼した。実装コード(`value_model.py` /
`encoder.py` / `lethal_simple.py` / `attack_features.py`)を突き合わせて検証した結果、
以下は**実装不可能または致命的な欠陥**と判明し、初版のまま実装するのは見送った。

1. **勝率の視点反転(致命的)。** `value_model.predict_win_prob_from_state(state)` は
   `state.yourIndex` を「自分」として勝率を返す([value_model.py:119-129](../../../ptcg_ai/learning/value_model.py)、
   `encoder.encode_state_from_state` が `state.yourIndex` 基準で特徴を組む
   [encoder.py:296](../../../ptcg_ai/learning/encoder.py))。一方、攻撃を打つとターンが相手に移り
   `yourIndex` が反転する。既存の `lethal_simple.py` はこれを検知して探索を打ち切っている
   ([lethal_simple.py:298](../../../ptcg_ai/search/lethal_simple.py): `if child_state.yourIndex != me: continue`)。
   初版のまま攻撃後の葉 State を無条件で `predict_win_prob_from_state` に渡すと、
   **相手視点の勝率を「自分の勝率」として最大化してしまう**(高いほど悪い手を選ぶ)。
2. **receding horizon が MAIN 限定トリガと両立しない。** 改造ハンマー等は
   「MAINでPLAY → 捨てる対象を選択 → MAINへ復帰」のように複数の非MAINコールバックを挟む。
   初版は「先頭選択だけ返して次の選択点で再探索」という lethal と同じ規律を想定していたが、
   lethal は次の非MAIN選択でも lethal 自身が再起動できるのに対し、AttackPlan は
   MAIN限定トリガのため次の非MAINコールバックで発火せず、探索で見つけた線を最後まで実行できない。
3. **`deckCount` 不変判定は事前に実装不可能かつ不正確。** `deckCount` は行動の結果であり
   候補生成の時点(実行前)では分からない。またシャッフルして同枚数引く効果や、対象選択の
   途中経過など、`deckCount` の差分だけでは「デッキに触れたか」を正しく分類できない
   (§3.2 に正しい判定方法を記載)。
4. **到達条件が弱すぎる。** 初版の採否基準は「ATTACK実行に到達」するだけで、
   準備前より実際にダメージが改善したことを要求していなかった。value の誤差次第で
   「無関係な札を消費しただけで依然0ダメージ」の線を採用しうる。
5. **トリガの「ダメージ0」判定が誤検知しうる。** `cg.api` の `Attack.damage` は
   手札枚数などに依存する可変ダメージ技では常に `0` を返す仕様([attack_features.py:1-12](../../../ptcg_ai/board_evaluation/attack_features.py))。
   静的な `Attack.damage` だけで「ダメージ0」を判定すると、こうした主力技を誤って
   トリガ対象にしてしまう。
6. **検証計画に実装前の頻度計測が無かった。** 「0ダメージ攻撃を続けてしまう」問題が
   実戦でどれだけ起きているか(頻度・1-prepで救済できる率)を計測せずに、
   depth2〜3のDFS新設から着手する計画になっていた。

これらを踏まえ、**「後検証型の0ダメージvetoから始め、shadow計測で効果が確認できた場合のみ
段階的に因果探索を深くする」方針(§2以降)に縮小した。** 元の深さ2〜3 DFS + value網の設計は
§8(将来の一般化)の v2 候補として残す。

---

## 1. 動機・背景(なぜ必要か)

### 1.1 現象:ダメージ0でも攻撃を続けてしまう

現在の意思決定チェーン([ml_policy_agent.py](../../../ptcg_ai/ml_policy/ml_policy_agent.py) `_select_action`)は

```
_try_lethal(確定リーサル探索) → _try_attack_hybrid(rule_baseゲート) → PolicyModel(模倣)
```

で、大半の非リーサル局面は **PolicyModel(模倣学習)** が決めている。模倣ポリシーは
「人間ならこの選択肢を選ぶ」を再現するだけで、**選んだ結果どうなるかをシミュレートしない**。
そのため、**相手ポケモンにダメージが 0 になる盤面でも、平然と攻撃を選び続けてしまう。**

### 1.2 具体例:相手の闘エネルギー(ロックエネルギー系)

分かりやすい実例が、相手のポケモンに **闘エネルギー等の「ダメージを軽減/無効化する特殊エネルギー」**
が付いているケース(例: ルカリオ系デッキ)。この状態で技を撃つとダメージが 0 に抑えられる。

- **人間プレイヤーなら**、まず **改造ハンマー(Enhanced Hammer)** などで相手の特殊エネルギーを
  剥がしてから攻撃する。剥がせば技が本来のダメージ(例: 200)で通る。
- **しかし現在の AI はこれができない。** 模倣ポリシーは打点計算も「剥がしてから殴る」という
  因果の読みも内在的に持たないため、特殊エネが付いたまま 0 ダメージの攻撃を選んでしまう。

同種の軽減札はほかにもある(ミストエネルギー等)し、今後カードが追加される可能性もある。
**この一件(0ダメージ攻撃をそのまま撃つ)を、カード個別対応ではなく一般化した仕組みで潰す** のが
本レイヤの目的である。

### 1.3 なぜ既存の2レイヤで拾えないか

- **確定リーサル探索(lethal_simple)** は「このターンで勝てるか(`result == me`)」という
  終端条件でしか発火しない([lethal_simple.py:294](../../../ptcg_ai/search/lethal_simple.py))。
  「リーサルではないが攻撃が 0」という中間状態は対象外。
- **PolicyModel(模倣)** は前述の通り結果を見ない。

→ 「非リーサル & 攻撃が無効化されている」という帯域が、結果を見ずに手を打っている空白地帯になっている。

---

## 2. 中核アイデア:「後検証」で無意味な攻撃を止め、因果が確認できた分だけ手を打つ

初版の「事前にDFSで最良線を探す」から、**「既存チェーンが選んだ攻撃を仮実行し、無意味だった場合だけ
段階的に因果を確認して手を打つ」後検証型**に変更する。カードを教えないという中核方針(§2 初版と同じ)
は維持する:「改造ハンマー = 特殊エネを剥がす札」とはコードに一切書かない。

```
既存チェーン(_try_lethal → PolicyModel等)が選んだ行動を取得
        ↓
その行動が ATTACK か? かつ 仮実行結果が「本来ダメージを与える意図なのに直接ダメージ0
  かつ KO/有益な副作用なし」か?  ← v0 の判定対象(§4)
        ↓ Yes                              ↓ No
   v0: その選択肢だけマスクして         そのまま既存チェーンの選択を採用
   残り選択肢で再選択
        ↓
   まだ0ダメージ攻撃しか残らない場合のみ
   v1: 「デッキに触れない行動を1つだけ、
        効果が完全に解決するまで仮実行 →
        直後の攻撃が改善するか」を確認
        ↓ 改善する                          ↓ 改善しない
   その先頭行動を採用                    既存チェームの選択(0ダメージ攻撃)へフォールバック
```

これにより:
- 闘エネでも、ミストエネでも、将来の未知の軽減札でも、**軽減を外す札がデッキにあれば無改修で対応。**
- v0/v1 はいずれも「実行結果」だけを見て判定するため、**value網のバグ(視点反転等)を持ち込まない。**
- イワパレス的な「EX にダメージが通らない」構造的無効も同枠(改善しない → 委譲。v1 では未対応、§7)。

v2(将来・§8)で depth2〜3 の探索と value網を導入する場合も、本レイヤで確立する
「実行結果に基づく因果判定」の基盤(§3.2 の deck-touch 判定、§3.3 の transaction 定義)をそのまま使う。

---

## 3. 設計の要点

### 3.1 別ファイルにする(lethal_simple は触らない)

`lethal_simple.py` は他メンバーの担当ファイルであり、コンフリクト源になるため**大幅改変しない**
(config 変更すらしない)。新レイヤは **新規ファイル `ptcg_ai/search/attack_plan.py`** として作り、
lethal_simple の内部関数ではなく **公開ゲームAPI `cg.api`(`search_begin/step/end/release`)を独立に叩く。**

- 共有するのはコンペ公式APIであって lethal_simple 内部ではない → **結合ゼロ・コンフリクト面ゼロ。**
- DFS 骨格・`_state_key` 等は多少コピーになるが、**境界を守るための意図的な複製**。
  前例として ml_policy_agent は `_is_valid_action` を「action_selection への依存をゼロに保つため」
  意図的に複製している([ml_policy_agent.py:156-168](../../../ptcg_ai/ml_policy/ml_policy_agent.py))。
- ml_policy 側に `_try_attack_plan` を1レイヤ追加。ただし §2 の通り「後検証」なので、
  配線位置は `_try_lethal` の直後ではなく **既存チェーン全体が1つ選択を出した後**(§5.2)。

### 3.2 「デッキに触れたか」の判定 — `deckCount` 差分ではなく Log 種別で行う

初版の `deckCount` 不変判定(§0-3)を、**仮実行中に発生した `cg.api.LogType` を見る方式**に置き換える。
以下のいずれかが transaction 中(§3.3)に1件でも発生したら「デッキに触れた」として候補から除外する。

| LogType | 意味 | 除外理由 |
|---|---|---|
| `SHUFFLE` (0) | 山札シャッフル | 非決定的 |
| `DRAW` (4) / `DRAW_REVERSE` (5) | ドロー(自分/相手) | 非決定的・deckCount減少 |
| `MOVE_CARD` (6) / `MOVE_CARD_REVERSE` (7) で `fromArea == DECK` または `toArea == DECK` | デッキへ/から移動 | サーチ・戻す効果など |
| `COIN` (22) | コイントス | **非決定的(§3.4)。deckは触らないが v0/v1 では別理由で除外** |
| transaction 中のいずれかの `Observation.select` が `SelectContext` 的にデッキ由来の選択(`select.deck` 相当) | サーチ選択中 | 非公開情報依存 |

`deckCount` の単純比較は**主判定ではなく取りこぼし検出用の補助アサーション**として残す
(上記Logで拾えないケースがあれば `deckCount` が変わっているはずなので、そこで検出して除外する安全網)。

### 3.3 「transaction」の定義(1手 = カード効果が完全に解決するまで)

v0/v1 でいう「1つの行動」は、**MAIN で選んだ1つの非ATTACK行動と、そこから派生する全コールバックの
連鎖**を指す(改造ハンマーの「PLAY→捨てる対象選択→MAIN復帰」を1単位として扱う)。
`search_step` を繰り返し呼び、以下のいずれかに到達するまでを1 transaction とする:

- **成立**: 同一ターン・同一 `yourIndex` で MAIN の選択に戻ってきた(= 効果が解決し切った)。
- **成立(稀)**: そのまま終局した。
- **失格**: ターンが終わる、または `yourIndex` が変わる(相手に手番が移る)。DFS候補から除外する
  (§0-1 の視点反転を踏まえ、v0/v1 では相手手番に及ぶ線を一切評価しない)。

攻撃側への対象選択(ハンマーの捨て先、進化先など)も同じ transaction の一部として最後まで解決する。
「先頭選択だけ返して次の選択点で再探索」という lethal 型の規律は、**transaction が成立するまでは
崩さない**(= 1 transaction 分は計画をコミットして実行し切る。次の MAIN 選択点でまた最新盤面から
やり直す、という意味での receding horizon は維持)。

### 3.4 乱数線(コイントス)は不採用 — 高精度側に倒す

1回の仮実行結果だけで採否を決めるため、**transaction 中またはその直後の攻撃解決中に `COIN` ログが
発生した線は「判定不能」として不採用**にする(表が出ただけの偶然の改善を「因果」として学習しない)。
コイントス由来の技(それ自体が攻撃側でも、prepの対象選択側でも)を伴う候補は、生成段階で
優先度を下げるか、判定不能として除外する。

### 3.5 v0: 事後veto(既存チェーンの選択を検証・マスク)

既存チェームが選んだ選択が ATTACK で、仮実行した結果が

- 直接ダメージが 0(§4 の判定方法。可変ダメージ技の誤検知を除外)
- かつ KO・状態異常・ベンチダメージ・エネルギー除去・ドロー等の**有益な副作用が無い**
  (ホワイトリストで判定。「有益な盤面変化」という曖昧な基準にしない)
- かつ本来ダメージを与える意図の技である(サポート/回復専用技等は対象外)

の場合、**その選択肢だけを `-inf` 相当のスコアにしてマスクし、`PolicyModel.score_options()` の
残りの選択肢から再選択する。** `obs.select.option` 配列自体は縮めない(返却する index と
配列の対応を壊さないため)。マスク後もなお別の0ダメージ攻撃が最上位に来る場合は、
上位から順に再検証してすべて除外してよい。全選択肢が0ダメージ攻撃になった場合のみ v1 に進む。

### 3.6 v1: 1-prep 因果カウンターファクタル

v0 で「無意味な0ダメージ攻撃」と判定され、かつ手札に §3.2 の意味で「デッキに触れない行動」が
存在する場合のみ発火する。候補行動を1つずつ(§3.3 の transaction 単位で)仮実行し、
その直後に同じ攻撃を再仮実行して、以下のハード条件のいずれかを満たすか確認する:

1. 直接ダメージが 0 → 正の値に変化
2. KO が成立する
3. 自分のサイドが減る(相手を倒してサイドを取る)、または即座に勝利する

満たす候補が複数ある場合の優先順位(タイブレーク。value網は使わない):

```
即座に勝利 > サイド獲得 > KO > ダメージ改善量(大きい順) > 元のPolicyModelスコア > option index
```

満たす候補が無ければ `None` を返し、既存チェームの選択(= v0でマスクした0ダメージ攻撃)へ
フォールバックする(**「1-prepで救えなければ既存へ委譲」を厳守**。深追いしない)。

---

## 4. 「ダメージ0」の判定方法(可変ダメージ技の誤検知を避ける)

`cg.api` の `Attack.damage` フィールドは、手札枚数などで変動する可変ダメージ技では
**常に `0` を返す仕様**であり([attack_features.py:1-12](../../../ptcg_ai/board_evaluation/attack_features.py))、
これをそのまま「ダメージ0」と判定すると主力技を誤ってトリガにしてしまう。

したがって判定は静的な `Attack.damage` ではなく、**`search_step` で実際にその攻撃を実行し、
結果としてエンジンが記録した相手への直接ダメージ(ログ or 相手 Pokemon の HP 減少)を見る。**
これは §3.5 の v0 が「既存チェームが選んだ攻撃を仮実行して検証する」設計そのものなので、
追加の特別扱いは不要(仮実行する時点で可変ダメージ技も実際の値が出る)。

将来 v2 でトリガを「攻撃前の事前フィルタ」として広げる場合は、この誤検知源を再考する必要がある
(§8)。

---

## 5. 実装方針

### 5.1 新規ファイル `ptcg_ai/search/attack_plan.py`

lethal_simple と同じ **team 共通インターフェース**を踏襲する:

```python
def search(state: State, legal_actions: list, context: dict) -> list[int] | None:
    """既存チェームが選んだ行動が無意味な0ダメージ攻撃だった場合、
    v0(マスク再選択)→ v1(1-prep因果探索)の順で代替行動を探す。無ければ None。

    context キー:
      - "observation" (必須): エージェントに渡った元 Observation。
      - "chosen_action" (必須): 既存チェーン(_try_lethal→_try_attack_hybrid→PolicyModel)が
        選んだ選択(v0/v1 の検証対象)。
      - "config": agent config の "attack_plan" セクション(欠損キーは DEFAULTS)。
      - "hidden_state_factory": search_begin 用の隠れ情報 dict を返す0引数callable
        (lethal_simple と同じもの)。
      - "policy_scores": PolicyModel.score_options() の生スコア(v0 のマスク再選択に使う)。
    """
```

内部関数:
- `_resolve_attack_damage(node, attack_selection) -> AttackOutcome` — 攻撃を仮実行し、
  直接ダメージ・KO有無・状態異常等の副作用・`COIN` ログの有無を読む(§3.4/§3.5)。
- `_is_deck_touching(logs: list[Log]) -> bool` — §3.2 の Log 種別判定。
- `_run_transaction(node, first_selection) -> TransactionResult | None`
  — §3.3 の定義に従い、MAIN復帰まで仮実行を進める。相手手番に及んだら None。
- `_candidate_prep_selections(select, state, config) -> Iterator[list[int]]`
  — `_is_deck_touching` で候補生成段階から deckに触れる手を除外(取りこぼしは `deckCount` 補助判定で検出)。
- `_try_v0_mask(chosen_action, obs, config) -> list[int] | None`
- `_try_v1_one_prep(obs, config) -> list[int] | None`

DEFAULTS(案):
```python
DEFAULTS = {
    "enabled": False,          # 既定 off(既存 config は挙動不変)
    "time_limit_ms": 100,      # lethal/PIMC等と合算した1意思決定あたりの共有予算に将来統合(§5.4)
    "max_root_actions": 8,     # v1で試すprep候補の上限(max_nodesではなく実効コストで制限)
    "zero_damage_threshold": 0,
    "shadow_only": True,       # Step0(§6)期間中は判定結果をログするだけで行動は既存チェーンのまま
}
```

### 5.2 `ml_policy_agent.py` への配線(自分の担当領域)

`_select_action` の**末尾**(既存チェーンが最終選択を確定した後)に検証を挟む:

```python
def _select_action(obs, config=None):
    lethal_action = _try_lethal(obs, config=config)
    if lethal_action is not None:
        return lethal_action

    attack_hybrid_action = _try_attack_hybrid(obs, config=config)
    baseline_action = attack_hybrid_action if attack_hybrid_action is not None else _policy_model_select(obs, config)

    attack_plan_action = _try_attack_plan(obs, baseline_action, config=config)   # 新規・後検証
    if attack_plan_action is not None:
        return attack_plan_action

    return baseline_action
```

`_try_attack_plan` は `_try_lethal` と同型(config から `attack_plan` セクションを読み、
`attack_plan.search(...)` を呼び、contract 検証 `_is_valid_action` を通す。例外は握って `None`)。
`shadow_only=True` の間は、判定結果(§6 Step0 の計測項目)をログするだけで返り値は常に `None`
(= 既存チェームの選択を変えない)。

### 5.3 config

ml_policy 専用 config(`PTCG_AI_ML_CONFIG`、既定 `ml_lethal`)に `attack_plan` セクションを追加。
既定は付けない(= 無効、挙動不変)。A/B 用に `ml_lethal_attackplan` のような別 config を用意して切替。

### 5.4 探索予算の共有(将来の統合ポイント)

v0/v1 単体では `time_limit_ms=100` 程度で収まる想定だが、`_try_lethal`(と将来PIMCを使う場合)も
それぞれ独立の deadline を持つため、1回の意思決定あたりの合計予算が積み上がりうる。
v1 まではリスクは小さい(候補数が手札の非デッキ操作行動だけに絞られるため)が、
**v2(depth2以降)に進む際は lethal/PIMC/AttackPlan で1つの共有 deadline を持たせる**
(§8)。

---

## 6. Step 分割と完了条件

### Step 0 — shadow計測(実装より前・最優先)

`_try_attack_plan` を **`shadow_only=True` で常時有効**にし、既存チェームの選択を変えずに
以下をログするだけの計測を数百試合分走らせる(このログ収集自体が1-prep探索を実際に走らせるため、
所要時間・timeout率も同時に記録する)。

- 全MAIN選択数のうち、既存チェームがATTACKを選んだ数
- そのうち仮実行結果が直接ダメージ0(§4の意味)だった数
- さらにそのうち KO/有益な副作用(§3.5ホワイトリスト)も無い「真の無駄攻撃」だった数
- 手札に deck-touch しない行動が存在し、1-prepで0→正ダメージ/KOに改善できた数
- 同一試合内で無駄攻撃を繰り返した回数
- v0/v1 判定にかかった p50/p95/p99 時間、timeout率

**合格条件**: 「真の無駄攻撃」の頻度と1-prep救済率が、投資に見合う水準であることを確認できること。
ここで頻度が無視できるほど低ければ、本レイヤの実装(v0配線を `shadow_only=False` にすること)を
見送る。

### Step 1 — v0(veto)配線 + 既存テストgreen確認

- 既定 off で挙動不変を確認(既存テスト green)。
- v0 単体(マスク再選択のみ、v1は未接続)をローカル対戦で on にし、非発火局面での既存action
  一致率が100%であること(vetoが発火しない大多数の局面を壊していないこと)を確認。

### Step 2 — v1(1-prep)配線 + head-to-head検証

- `_try_v1_one_prep` を接続。
- **アブレーション比較**: `ml_lethal`(現行) vs `ml_lethal_attackplan_v0only` vs
  `ml_lethal_attackplan_v0v1` を同一プロセス内で対戦(既存の候補重み head-to-head の仕組みを流用)。
  段階ごとの増分効果を計測する。
- 特にルカリオ系(闘)相手での勝率改善を見る。有意差が出れば本番 config に昇格を検討。
- 非発火局面での既存actionとの完全一致率をリグレッションガードにする
  (attack_hybrid が広く撃って一致率23.2%まで劣化した教訓 [ml_policy_agent.py:47-57](../../../ptcg_ai/ml_policy/ml_policy_agent.py) の再発防止)。

### 必須テスト(Step 1/2 共通)

- 改造ハンマーの「PLAY→対象選択→MAIN復帰」を実際に複数コールバック通すテスト。
- `yourIndex` が相手に移った場合に候補が失格になるテスト(§3.3 の視点反転回避)。
- 0ダメージだが有益な技(状態異常・ベンチダメージ・エネルギー除去等)がvetoされないテスト。
- コイントス技が判定不能としてスキップされるテスト(§3.4)。
- `deckCount` は同じだがシャッフル/ドローを伴う効果が deck-touch として除外されるテスト(§3.2)。
- `PolicyModel` 未ロード時のフォールバックテスト(v0のマスク対象が index 0 のケースを含む)。
- timeout時に部分結果を採用せず `None` を返すテスト。
- 非発火局面で既存 `_select_action` の出力と完全一致するテスト。

---

## 7. 非目標(v1 でやらないこと)

- **デッキ全体を使えると仮定した探索**(実行不能な空想線で行動を選ぶのは不健全)。
- **ドロー/サーチで掘る線の探索**(PolicyModel に委譲)。
- **イワパレス等の構造的ダメージ無効への special-case**(自然に fall through。§4)。
- **攻撃者の入れ替え探索**(深く/広くなり1ターンで完結しにくい。必要性を計測後に判断)。
- **value網を使った採否判定**(§0-1 の視点反転バグが未検証のうちは持ち込まない。v2 の課題)。
- **コイントスを含む線の因果判定**(§3.4。判定不能として扱う)。

## 8. 将来の一般化(v2 以降、Step0計測で「1-prepでは救えない実例」が十分溜まってから)

- 深さ2〜3の探索(連言ケース: 「エネ1枚 + ハンマー」の両方が要る局面など)。
- 採否判定に value網を使う場合は、**まず root player 視点に固定する
  `value(state, root_player_index)` を新設し、視点反転のユニットテストを通してから**導入する
  (§0-1)。value は主根拠にせず、§3.6 のハード条件を満たした複数候補のタイブレークに限定する。
- `max_nodes` ではなく `max_root_actions` / `max_leaf_evals` / `beam_width` で実コストを制限する。
- lethal / PIMC / AttackPlan で1回の意思決定あたりの探索予算(deadline)を共有する(§5.4)。
- 決定的チューター(deck は減るが結果が決定的)を prep に許可する案の検討。
- トリガを「非0だが弱い攻撃 → 準備で KO」まで拡張(ハード条件ガード前提)。

いずれも **Step0の計測結果を見てから** 投資する(過剰一般化を避ける)。

---

## 付録: レビュー経緯

本ドキュメントの v1 は、codex(gpt-5.6-sol、Codex CLI経由)による2回のレビューを踏まえて
初版から縮小・修正したもの。

- 1回目レビューで、初版設計(深さ2〜3 DFS + value網で葉を採点)に対する6件の指摘(§0参照)を受け、
  実装コード(`value_model.py` / `encoder.py` / `lethal_simple.py` / `attack_features.py`)で
  すべて事実確認した。
- その指摘を踏まえた縮小案(shadow計測 → v0 veto → v1 1-prep → v2 depth2、value網はv2まで不使用)
  を2回目のレビューに提示し、「方向性は妥当」との合意を得た上で、5件の追加修正
  (transaction定義の明文化、deck-touch判定の拡張、コイントス除外、v0のoption配列非破壊、
  ハード条件とタイブレークの分離)を反映した。
