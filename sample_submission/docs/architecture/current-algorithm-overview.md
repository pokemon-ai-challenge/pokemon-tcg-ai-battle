# 現在のアルゴリズム概要(2026-07-22 時点)

対象: `AGENT_TYPE = "ml_policy"`([`ptcg_ai/core/agent.py`](../../ptcg_ai/core/agent.py))、
既定config `ml_lethal_attackplan_v0only`([`configs/ml_lethal_attackplan_v0only.json`](../../configs/ml_lethal_attackplan_v0only.json)、
`ptcg_ai/ml_policy/ml_policy_agent.py` の `_CONFIG_NAME` 既定値)。
Kaggle提出中の模倣学習v4(ref 54883922, publicScore 749.8)の実装に対応する。
関連記録: [`results/2026-07-22_attackplan_kaggle_submission.md`](../../results/2026-07-22_attackplan_kaggle_submission.md)。

---

## 1. 全体の意思決定フロー

```mermaid
flowchart TD
    OBS["Observation\n(main.py -> core.agent -> ml_policy_agent.agent)"]
    OBS --> HI["match_context.update(obs)\n非公開情報の推定を毎ターン更新(副作用のみ)"]
    HI --> SELNULL{"obs.select is None?\n(初回デッキ選択)"}
    SELNULL -- Yes --> DECK["deck.csv を60枚返す\n(read_deck_csv)"]
    SELNULL -- No --> LETHAL

    subgraph SA["_select_action"]
        LETHAL{"1) _try_lethal\n確定リーサル探索\n(lethal_search.enabled)"}
        LETHAL -- "詰みが見つかった" --> ACT["この選択を返す"]
        LETHAL -- 見つからない --> HYBRID

        HYBRID{"2) _try_attack_hybrid\nルールベースATTACKゲート\n(既定config では無効=OFF)"}
        HYBRID -- "有効かつ採用" --> BASE1["baseline_action = rule_baseのATTACK選択"]
        HYBRID -- 無効/非採用 --> POLICY

        POLICY["3) PolicyModel(模倣学習, 構成C重み)\nscore_options→argmax\n(maxCount>1は貪欲フォールバック)"]
        POLICY --> BASE2["baseline_action"]

        BASE1 --> PLAN
        BASE2 --> PLAN

        PLAN{"4) _try_attack_plan (v0)\nbaseline_actionが『0ダメージ&\n無益な攻撃』なら代替を探索\n(attack_plan.enabled=true, 既定ON)"}
        PLAN -- "代替が見つかった" --> ACT2["その代替行動を返す"]
        PLAN -- "見つからない/対象外" --> ACT3["baseline_action をそのまま返す"]
    end

    ACT2 --> RESULT["最終的な選択インデックス列を\ncg engine へ返す"]
    ACT3 --> RESULT
    ACT --> RESULT

    VALUE["ValueModel(Step1)\nvalue_shadow_logging=true のときのみ\nログ目的で勝率予測(意思決定には未使用)"]
    OBS -.既定configではOFF.-> VALUE

    PIMC["PIMC + hidden_information推定\n(search/pimc.py, hidden_information/match_context)\n研究用config(ml_pimc等)でのみ使用。\n既定config では lethal/attack_planとも\nhidden_state_source=\"dummy\"(推定を使わない)"]
    HI -.既定configでは意思決定に不接続.-> PIMC
```

### フロー上の要点

| 段階 | 有効/無効(既定config) | 役割 |
|---|---|---|
| 1. 確定リーサル探索(`lethal_simple`) | **有効** | このターンで勝てる手順があれば最優先で採用。DFS、`max_depth=20`・`max_nodes=10000`・`time_limit_ms=100`・`max_remaining_prizes=3`・`verify_shuffles=1` |
| 2. ATTACK専用ハイブリッド(`attack_hybrid`) | **無効(既定OFF)** | rule_baseの提案から「他に展開すべきことが無い」局面のATTACKだけ採用する予定だった仕組み。600試合で有意差なし(51.8%)のため保留中([`project_attack_rulebased_hybrid`](../../results) 系の記録参照) |
| 3. PolicyModel(模倣学習) | **有効(既定・主力)** | 通常ターンの大半を決める本体。詳細は §2 |
| 4. attack_plan v0(攻撃部分探索) | **有効(既定ON)** | 0ダメージ&無益な攻撃だけを事後的に検出し、デッキに触れない代替行動があれば差し替える。詳細は §3.2 |
| ValueModel(Step1バリューネットワーク) | 無効(shadow logging用、意思決定には不使用) | §3.3 |
| PIMC・hidden_information推定 | 無効(既定configでは`hidden_state_source="dummy"`) | §3.4 |

---

## 2. 模倣学習(PolicyModel)の特徴量一覧

実装: [`ptcg_ai/learning/encoder.py`](../../ptcg_ai/learning/encoder.py)(特徴抽出)、
[`ptcg_ai/learning/policy_model.py`](../../ptcg_ai/learning/policy_model.py)(推論)。

### 2.1 モデルの入出力

- **入力**: ある意思決定点(`obs.select`)の「盤面特徴(状態特徴)」+「各選択肢の特徴」+「対象カードのidentity埋め込み」
- **出力**: 各選択肢のスコア(生の実数値、確率ではない)。スコア最大(argmax)の選択肢を採用
- 選択肢が複数選択(`maxCount > 1`)の場合は学習スコープ外のため、独立スコアリング後に上位から貪欲に選ぶフォールバック

### 2.2 状態特徴(state features, 166次元)— 盤面全体で1回だけ計算し、全選択肢で共有

**ポケモン1体あたり11特徴 × 12スロット(自分/相手 × active + ベンチ5枠) = 132次元**

| 特徴名 | 内容 |
|---|---|
| `present` | そのスロットにポケモンが存在するか |
| `hp_ratio` | 残りHP割合 |
| `remaining_hp` | 残りHP(絶対値) |
| `damage_counters` | 乗っているダメカン数 |
| `energy_count` | 付いているエネルギー枚数 |
| `best_attack_damage` | 現状撃てる最大ダメージ技の与ダメージ |
| `can_ko_defender` | その技で相手をKOできるか |
| `min_energy_shortfall` | 技を撃つのに足りないエネルギー数(最小) |
| `has_ready_attack` | 今すぐ撃てる技があるか |
| `attacker_score` | 攻撃適性の合成スコア |
| `is_likely_ko_next_turn` | 次ターン中にKOされそうか |

**枚数系(自分8種+相手5種 = 13次元)**

`self_hand_count` / `self_hand_pokemon` / `self_hand_trainer` / `self_hand_energy` / `self_deck_count` /
`self_prize_remaining` / `self_discard_count` / `self_bench_count` /
`opp_hand_count` / `opp_deck_count` / `opp_prize_remaining` / `opp_discard_count` / `opp_bench_count`

(自分の手札は内訳まで、相手は公開情報である枚数のみ。**相手の非公開情報(手札の中身・山札の並び)は一切使わない**)

**特殊状態(自分/相手 × 5種 = 10次元)**: `poisoned` / `burned` / `asleep` / `paralyzed` / `confused`

**ゲーム進行(7次元)**: `turn` / `is_first_player` / `supporter_played` / `stadium_played` /
`energy_attached` / `retreated` / `has_stadium`

**集約(4次元)**: `prize_diff`(サイド差) / `bench_count_diff`(ベンチ枚数差) /
`self_energy_on_board` / `opp_energy_on_board`

### 2.3 選択肢特徴(option features, 65次元)— 選択肢ごとに計算

| グループ | 内容 |
|---|---|
| `opttype_*`(18) | 選択肢の`OptionType`のone-hot(PLAY/ATTACH/EVOLVE/ATTACK/END等 + other) |
| `seltype_*`(11) | この意思決定点の`SelectType`のone-hot(MAIN/CARD/ATTACK等 + other) |
| その他スカラー(5) | `is_own` / `number_norm` / `count_norm` / `option_position_norm` / `n_options` |
| 対象ポケモン特徴(12) | `has_target_pokemon` + 上記ポケモン11特徴を対象1体分 |
| 対象カード特徴(11) | `has_target_card` + カード種別one-hot(7種) + `hp_norm`/`is_basic`/`is_stage1`/`is_stage2`/`is_ex` |
| 対象技特徴(5) | `has_attack` / `attack_damage_norm` / `attack_can_ko` / `attack_min_shortfall_norm` / `attack_has_ready` |

### 2.4 カード識別embedding(2026-07-20 追加、8次元)

選択肢特徴だけでは「博士の研究 と ハイパーボール」のような同種別内の個別カードを区別できないため、
`CardData.cardId` をキーにした学習済み埋め込み(dim=8、ゲーム全カード対象、`card_id`直書きなし)を
状態特徴・選択肢特徴と連結して使う。学習データに出現しないカードの埋め込みは初期値のまま(ゼロ近傍)。

### 2.5 モデル構造・学習

- 入力次元: `166(state) + 65(option) + 8(card embedding) = 239`
- 構造: 全結合層(ReLU) → … → 最終層(活性化なし、生スコア)。純Python実装(numpy/torch非依存)
- 学習データ: Kaggle公開リプレイ、train 147,705 / val 17,359 / test 22,626 件
- **「構成C」= 上位パイロット技量集中**: `rank_at_fetch`(取得時点の順位)に基づき上位パイロットの
  リプレイへリウェイトして学習し直したデータ構成。オフライン一致率の伸びは小さい(+1.5pt程度)が、
  現行重みとのミラー対戦では58%対42%で有意に勝ち越し、既定重み(`policy_weights.json`)として採用済み
  ([`policymodel-skill-concentration-implementation-plan.md`](../plans/individual/shogo/policymodel-skill-concentration-implementation-plan.md))
- test top1一致率: **58.06%**(選択肢特徴のみのベースライン54.14%比 +3.9pt)

---

## 3. その他のアルゴリズムに関する情報

### 3.1 確定リーサル探索(`ptcg_ai/search/lethal_simple.py`)

- このターン中に勝てる行動列(prep→lethal)をDFSで探索し、見つかれば最優先で採用
- パラメータ(既定config): `max_depth=20` / `max_nodes=10000` / `time_limit_ms=100` /
  `max_remaining_prizes=3`(相手の残りサイドがこの枚数以下のときだけ探索) /
  `verify_shuffles=1`(非公開情報の引き直し再現検証)
- `hidden_state_source`(既定`"dummy"`): 相手の非公開情報を仮のダミー値で埋めるか、
  `"estimated"`にすると`hidden_information.match_context`の推定値を使う(実験config限定)

### 3.2 attack_plan v0(`ptcg_ai/search/attack_plan.py`)— 2026-07-22追加

- 既存チェーン(lethal→attack_hybrid→PolicyModel)が選んだ行動を**事後検証**する後付けレイヤー
- 対象: 選ばれた行動がATTACKで、実行結果が「直接ダメージ0 かつ KO/状態異常/ベンチダメージ/
  エネルギー除去/ドローのいずれも無い(=真に無意味)」場合のみ
- 該当する場合、その選択肢だけをマスクしてPolicyModelスコア上位の残り選択肢から再選択(v0)。
  既定config(`ml_lethal_attackplan_v0only`)では `max_root_actions=0` によりv0のみ有効
  (深さ1以上の準備行動探索=v1は無効)
- 設計文書: [`docs/plans/attack-enabling-search/design-and-implementation-plan.md`](../plans/attack-enabling-search/design-and-implementation-plan.md)
- 実戦検証(87試合・357攻撃): ダメージ0の攻撃の69.6%は正しく「有益」と判定されvetoされない。
  真に無意味な攻撃は全体の15.4%まで残存(大半は代替行動が無く委譲した結果と推定)。
  詳細は [`results/2026-07-22_attackplan_kaggle_submission.md`](../../results/2026-07-22_attackplan_kaggle_submission.md)

### 3.3 バリューネットワーク(`ptcg_ai/learning/value_model.py`, Step1)

- 状態特徴(2.2節と同じ166次元)から勝率を予測するMLP(標準化 + ターン帯別温度較正つき)
- **現在は意思決定に一切使われていない。** `config["value_shadow_logging"]=true` のときだけ
  `value_shadow_log.record(obs)` でログに残す(既定configではこのキー自体が無く、既定false=無効)
- attack_planの初期設計案では葉ノード評価に使う予定だったが、`state.yourIndex`視点反転の
  未検証バグがレビューで発覚したため不採用(§0参照、design-and-implementation-plan.md)

### 3.4 hidden_information / PIMC(既定では意思決定に不接続)

- `hidden_information.match_context`: 相手デッキ推定・自分の山札/サイド推定を毎ターン更新
  (`agent()`冒頭で常時呼ぶが、これ自体は副作用のみで意思決定には影響しない)
- `search/pimc.py`(PIMC探索): 研究用config(`ml_pimc`等)でのみ使用。rule_based上では
  有意に勝ち越したが、`ml_policy`本番上では有意差なしと確認済みのため現在は既定OFF
  ([`project_pimc_prod_validation`](../../results) 系の記録参照)
- 既定config(`ml_lethal_attackplan_v0only`)では、lethal探索・attack_planとも
  `hidden_state_source`が指定されておらず既定の`"dummy"`(仮の非公開情報)を使う。
  つまり**相手の非公開情報の実推定値は、現在の意思決定には使われていない**

### 3.5 ルールベース(`ptcg_ai/rule_based/`)

- 独立した手書きルールAI。`AGENT_TYPE="rule_based"`で切替可能だが、現在の提出は`ml_policy`
- `main_turn_parts.proposals`/`weights`のみ、attack_hybrid用に読み取り専用でimportされる
  (attack_hybrid自体は既定OFF)
