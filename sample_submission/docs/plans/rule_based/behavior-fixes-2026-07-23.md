# ルールベースAI 挙動修正まとめ（2026-07-23）

フーディン（Alakazam）デッキの自己対戦観戦で見つかった、明らかに不合理な3つの挙動。
それぞれ独立した GitHub Issue として起票できるよう、現象・根拠・原因・要件（受け入れ条件）を整理する。

> **実装ステータス（2026-07-23 実装済み、同日ユーザ指摘で追加改修も反映）**
> 3件とも実装・検証完了。
> - Issue 1: 超エネのノコッチ系への誤付与 2→**0**（`energy_eval.is_attach_eligible` で付与対象をアタッカー
>   ＋周回条件成立時のノコッチ系列に限定）／ポフィン条件を系列単位に是正。
>   **追加改修**: 周回受け皿を66単独→ノコッチ系列(65/66)全体に一般化し、リッチエネはノコッチ(65)にも
>   付けられるようにした（超エネは引き続き付かない、実機12戦で確認）。
>   回帰テスト `tests/unit/test_energy_target_eligibility.py`（6テスト）。
> - Issue 2: 改造ハンマーの破壊対象がミストエネ(11)を最優先（`discard.py`＋`OPPONENT_EFFECT_LOCK_ENERGY_IDS`）。
>   回帰テスト `tests/unit/test_discard_mist_energy.py`。※ミラー自己対戦にミストエネは出ないため単体テストで担保。
> - Issue 3: エネを捨てる非緊急逃げを **136→96（約29%減）**、勝率53%（回帰なし）。
>   `retreat.py` に逃げコストのエネ損失マージンを追加。
> - `pytest`: 75 passed（新規含む）/ 0 failed。
> 以下は起票時の要件定義（記録として残す）。

対象ブランチ: `feature/rule-based-fix` / 対象デッキ: フーディン（`decks/new_deck/`）

---

## Issue 1: ノコッチにエネルギーを付け、ポフィンでケーシィを持ってくる無意味な動き

### 現象
バトル・展開の下準備で、**ノコッチ（Dunsparce, card_id 65）系に超エネルギー（【超】＝基本超5／テレパス超19）を付与**し、
さらに**なかよしポフィン（card_id 1086）でケーシィ（Abra, 741）を持ってくる**という順序のおかしい動きが観測された。

**訂正（2026-07-23 ユーザ指摘）— 問題はエネルギーの“種類”と“順序”:**
- ノコッチ系に**リッチエネルギー（Enriching Energy, 13）を付けるのはOK**。リッチエネは無色1個ぶんで、
  フーディンのハンドパワーのコスト【超】を満たせない一方、ノココッチのにげあしドローで山札に戻して
  周回できる（#78 の周回コンボ）。よって「余ったリッチエネの置き場」としてノコッチ系は妥当。
- 一方**超エネ（【超】: 5 / 19）をノコッチ系に付けるのはダメ**。超エネはフーディン（アタッカー）の
  ハンドパワー始動に必須なので、そちらに温存すべき。ノコッチ系に付けると死に札になる。
- さらに**順序がおかしい**: ケーシィ（＝フーディンの起点）をサーチできるカードが手札にあるのに、
  ケーシィを展開してエネの正しい付け先（フーディン系列）を作る前に、先に（付け先が無いからと）
  非アタッカーへ超エネを付けてしまっていた。**先にアタッカーを展開してからエネを付ける**べき。

### 根拠（再現）
- 自己対戦8ゲームで **ATTACH 対象がノコッチ(65) だったケースを2回**確認
  （`scratchpad/probe_issues.py`）。
- ノコッチ(65)は攻撃 `atk 74`(10dmg)/`atk 75`(30dmg・エネルギー要) を持つ。
- なかよしポフィンの使用条件 `_buddy_buddy_poffin_condition` は
  「ケーシィ/キチキギスex/ノコッチのいずれかが場にも手札にも無い場合」に発火する
  （`decks/new_deck/item_profiles.py:35`）。

### 原因
1. **エネ付与先の評価がノコッチを候補に含めてしまう。**
   `energy_eval.energy_target_value()` は「そのポケモンが攻撃に必要なエネルギーが不足していれば
   加点（shortage_bonus）」する。ノコッチは atk 75 にエネルギーが要るため、
   `board_evaluation.energy_requirements.energy_shortfall` が不足を返し、加点されてしまう。
   デッキ方針上ノコッチは決して攻撃しないのに、カードが技を持つだけで「エネを付ける価値あり」と
   誤判定している（`ptcg_ai/rule_based/main_turn_parts/energy_eval.py:49-75`）。
2. **ポフィンのサーチ先/使用タイミングが盤面充足を見ていない。**
   すでにフーディン系列が十分展開できている場面でも、条件に合致すると発火し、
   直接勝ち筋に絡まないケーシィを持ってくる（`item_profiles.py:35-38`）。

### 要件（受け入れ条件）— 訂正後
- [x] **超エネ（【超】: 基本超5 / テレパス超19）を非アタッカー（ノコッチ系・シェイミ等）に付けない。**
      超エネはフーディン系列のハンドパワー始動に温存する。
      → 実装済み: `energy_eval.is_attach_eligible`（`ENERGY_TARGET_PRIORITY` のアタッカー＋周回条件成立時の
      ノココッチのみを付与対象にする）。
- [x] **リッチエネ（13, 無色）はノコッチ系（65/66どちらでも）に付けてよい。** ユーザ指示
      「ノコッチループがこのデッキの肝」を受けて対応。`ENERGY_RECYCLE_TARGET_CARD_ID`（単数, 66のみ）を
      `ENERGY_RECYCLE_TARGET_CARD_IDS = frozenset(DUNSPARCE_LINE_CARD_IDS)`（複数, 65/66）へ一般化し、
      `DeckPlan.energy_recycle_target_ids` 経由で `is_attach_eligible`/`_energy_recycle_bonus` を
      単一ID比較→集合メンバーシップ判定に変更。既存の周回ゲート条件（手札にリッチエネ＋主力エネ充足 or
      代替供給手段）はそのまま両段階に適用（進化前に付けたエネは進化で引き継がれるため理屈上も妥当）。
      超エネがノコッチ系列に付かない保証は、対象ポケモンの許可ではなく既存の `ENERGY_CARD_PRIORITY_RULES`
      （`_otherwise` 規則がリッチエネを最優先にする tie-break）に委ねる設計。
      実機12戦self-playで確認: Dunsparce(65)へのATTACH = リッチエネ13回・超エネ0回、
      Dudunsparce(66)へは同6回・0回。回帰テスト `tests/unit/test_energy_target_eligibility.py`
      （周回条件成立/不成立/state無し/65・66両方をカバー、6テスト）。pytest 75 passed（全体回帰なし）。
- [x] #78 のリッチエネ×ノココッチ周回（66 への意図的付与）は維持する（`is_attach_eligible` の周回例外で担保）。
- [x] **順序**: 付け先アタッカーが場に無いときは `best_energy_target` が None を返し超エネを浪費しない
      → setup-first でポフィン等の展開が先に走り、ケーシィ→フーディン系列を立ててからエネを付ける流れになる。
- [x] ポフィンは「系列単位でまだ確保できていない主要パーツがある時だけ」使う（`_buddy_buddy_poffin_condition` 是正済み）。
- [x] 回帰テスト: `tests/unit/test_energy_target_eligibility.py`。

### 関連ファイル
- `ptcg_ai/rule_based/main_turn_parts/energy_eval.py`（`best_energy_target`/`energy_target_value`）
- `ptcg_ai/rule_based/main_turn_parts/priorities/energy.py`
- `decks/new_deck/item_profiles.py`（`_buddy_buddy_poffin_condition`）
- `decks/new_deck/deck_plan.py`（`ENERGY_TARGET_PRIORITY` / `ENERGY_REQUIRED_COUNT`）

---

## Issue 2: 改造ハンマーでミストエネルギー（技の効果を無効化する特殊エネ）を優先的に壊す

### 現象
相手が**ミストエネルギー**（付いているポケモンが「ワザの効果を受けない」＝こちらのワザの効果を
無効化する特殊エネルギー）を貼っていても、改造ハンマー（Enhanced Hammer, card_id 1081）が
それを狙って壊しに行かない。結果、こちらのワザ効果が通らないまま放置される。

### 根拠
- 改造ハンマーの使用条件 `_enhanced_hammer_condition` は
  「相手の**バトル場**のポケモンに**何らかの**特殊エネルギーが付いていれば発火」
  （`decks/new_deck/item_profiles.py:30-32`、`ctx.opponent_active_has_special_energy`）。
- `usage_context._has_special_energy()` は特殊エネルギーを**種類で区別せず**、
  `cardType == SPECIAL_ENERGY` を一括判定しているだけ（`ptcg_ai/board_evaluation/usage_context.py:49-55`）。
- 破壊対象の選択は `discard.choose()` に流れるが、これは**自分の保護カードを避けるだけ**で、
  相手のどの特殊エネを壊すか（ミストエネ優先）を選別していない
  （`ptcg_ai/rule_based/card_move/discard.py:18-28`）。

### 原因
1. ミストエネルギーを他の特殊エネルギー（例: 攻撃補助だけの特殊エネ）と**区別できていない**。
   使用条件も破壊対象選択も「特殊エネ＝一律」で扱っている。
2. 相手の**ベンチ**の特殊エネは見ていない（`opponent_active_has_special_energy` はアクティブのみ）。

### 要件（受け入れ条件）— 訂正後
- [x] ミストエネ系を card_id で識別（`decks/new_deck/deck_plan.py: OPPONENT_EFFECT_LOCK_ENERGY_IDS = {11}`）。
- [x] 破壊対象の選択で**ミストエネ系を最優先**で壊す（`discard.py` の `_SCORE_EFFECT_LOCK`、複数特殊エネの tie-break）。
      回帰テスト `tests/unit/test_discard_mist_energy.py`。
- [x] 改造ハンマーの使用条件は「相手アクティブに特殊エネがあれば発火」（`_enhanced_hammer_condition`）＝
      ミストが付いていれば発火し、setup-first で**フーディンの攻撃前に**破壊対象＝ミストを壊す。
      → **訂正後の理解ではこの“無条件発火”は妥当**（ミストはハンドパワーを完全に止めるので、迷わず壊すべき）。
      当初「効果付きワザを通したい時だけ発火」に絞る案は**撤回**（ハンドパワー自体が効果扱いなので絞ると逆効果）。
- [ ] **【未対応・任意】相手ベンチのミストエネは未考慮**（`opponent_active_has_special_energy` はアクティブのみ）。
      相手がベンチのポケモンにミストを貼って後で前に出す動きには後手になる。優先度低（まず前のミストを割れれば十分）。
- [ ] **【未対応・任意】改造ハンマーを2種類の特殊エネに対して無駄撃ちしない**。現状は非ミストの特殊エネにも
      発火するので、相手の特殊エネがミスト系のみ／ダメージに絡む時に重み付けする改善余地。優先度低。

### 補足・要確認（調査済み）
- **ミストエネルギー = card_id 11**（特殊エネルギー、`data/EN_Card_Data.csv`/`JP_Card_Data.csv` で確認）。
  効果全文: 「このカードをつけているポケモンは、相手のポケモンが使うワザの**効果**を受けない。
  （すでに受けている効果は、なくならない。**ダメージは効果ではない**。）」
- **【最重要・訂正 2026-07-23（ユーザ指摘）】ミストエネはこのデッキに対しては“ダメージそのもの”を止める。**
  一般にミストエネは「効果のみ無効化・ダメージは通す」だが、**フーディンの唯一のワザ ハンドパワー(attackId 1072)は
  「ダメージカウンターを置く（Place 2 damage counters ... for each card in your hand）」＝“ダメージ”ではなく
  “効果”**である（TCGルール上、ダメージカウンターを置くのは効果でありダメージ計算ではない）。
  したがって**ミストエネが付いた相手にはフーディンのハンドパワーが一切通らない（0点）**。
  → 改造ハンマーでミストエネを壊す価値は**極めて高い（このデッキの生命線）**。相手アクティブにミストエネが
  付いている場合は、フーディンで攻撃する前に必ず改造ハンマーで壊してから殴る、が正しい方針。
  （旧記述「ミストはダメージを防がないので改造ハンマーの価値は低い」は**誤り**だったので撤回する。この誤りを
  信じて改造ハンマーの優先度を下げると、ミスト下でフーディンが一切ダメージを出せなくなる。）
- 一般化する場合、「ワザの効果を受けなくする特殊エネ」を card_id 集合（`OPPONENT_EFFECT_LOCK_ENERGY_IDS`）で持ち、
  ミストエネ(11)以外（将来の類似カード）にも対応できるようにする。
- 補足: キチキギスex 等、ダメージカウンター配置以外の効果を持つワザも同様にミストで無効化されるため、
  改造ハンマーでの解除価値は総じて高い。

### 関連ファイル
- `decks/new_deck/item_profiles.py`（`_enhanced_hammer_condition`）
- `ptcg_ai/board_evaluation/usage_context.py`（`_has_special_energy` / `UsageContext`）
- `ptcg_ai/rule_based/card_move/discard.py`（破壊対象の選択）
- `ptcg_ai/shared/profile_types.py`（`UsageContext` フィールド追加が必要な場合）

---

## Issue 3: 意味のない逃げでエネルギーを無駄にする

### 現象
相手に倒されそうでもないのに**不要不急の交代（逃げ）**を行い、逃げコストとして
バトル場のエネルギーを捨てて無駄にしている。エネルギーを付けて育てたアタッカーを、
わずかな評価差で下げてしまう。

### 根拠（再現）
- 自己対戦8ゲームで **逃げ16回中11回（約69%）が非緊急**（`board_features.is_likely_ko_next_turn`＝False）。
- 逃げ時にバトル場に付いていたエネルギー総数は合計17（逃げコストで捨てられうる投資）
  （`scratchpad/probe_issues.py`）。

### 原因
`priorities/retreat.py` の非緊急交代は
`switch_target_value(bench) > active_value(active) + _IMPROVEMENT_MARGIN(3.0)` で発火するが、
**「逃げるとバトル場のエネルギーを retreatCost 枚ぶん捨てる」コストを評価に入れていない**。
`can_afford_retreat` で「払えるか」は見るが（`retreat.py:40`）、
「払う価値があるか（捨てるエネの損失 > 交代の得か）」は見ていない
（`ptcg_ai/rule_based/main_turn_parts/priorities/retreat.py:52-57`）。
このため、エネが乗ったアタッカーでも僅差の交代先に釣られて逃げ、投資エネを捨ててしまう。

### 要件（受け入れ条件）
- [ ] 非緊急交代の判定に**逃げで失うエネルギーのコスト**を織り込む。
      例: バトル場ポケモンに付いているエネが多い／育っているほど、交代に必要な
      改善マージンを大きくする（`_IMPROVEMENT_MARGIN` をエネ投資量に応じて増やす）。
- [ ] 緊急（次ターンKOされる）でない限り、**エネが乗った主力アタッカーは基本下げない**。
- [ ] 逃げコストで捨てるエネが、次の番に無駄になる（付け直せない）場合はさらに抑制する。
- [ ] 逃げそのものが盤面改善に繋がらない（交代先も同等・劣る）ケースを確実に None にする。
- [ ] 回帰テスト: 「エネ2個付きのフーディンが、僅差の交代先候補があっても非緊急では逃げない」テスト追加。
      既存の `_IMPROVEMENT_MARGIN` テスト（Issue #83）と整合させる。

### 関連ファイル
- `ptcg_ai/rule_based/main_turn_parts/priorities/retreat.py`
- `ptcg_ai/rule_based/main_turn_parts/pokemon_value.py`（`active_value` / `switch_target_value`）
- `ptcg_ai/board_evaluation/energy_requirements.py`（`can_afford_retreat`）
- `ptcg_ai/board_evaluation/switch_eval.py`

---

## 検証方法（共通）
各修正後は以下で回帰を確認する:
- `python -m pytest -q`（ユニット/統合テスト）
- 自己対戦A/B（`scratchpad/ab_bench.py` 相当。setup-first を含む現行版に対して勝率が下がらないこと）
- 該当挙動の再現プローブ（`scratchpad/probe_issues.py`）で対象挙動が消えたことを数値で確認
  - Issue 1: ノコッチへの ATTACH 回数 = 0
  - Issue 2: ミストエネ付き相手に対する改造ハンマーの破壊対象がミストエネ
  - Issue 3: 非緊急逃げの回数・逃げで捨てるエネ総数が減少

## 注意（測定の限界）
自己対戦はフーディン・ミラー（低HP同型）なので、対高HP ex 特有の挙動（改造ハンマーでミストエネ破壊の価値など）は
過小評価される可能性がある。meta-deck ベンチのあるブランチでの確認が望ましい。
