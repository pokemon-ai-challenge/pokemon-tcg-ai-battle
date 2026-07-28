# 実装計画: ATTACK専用ハイブリッド(案B) — rule_baseの「攻撃すべき」判定をゲートにする

作成: 2026-07-21
状態: 実装計画
前提: [attack-rulebased-hybrid-strategy.md](./attack-rulebased-hybrid-strategy.md) で案B決定済み
対象ブランチ: `experiment/pimc-hidden-info-integration`

---

## この計画の目的

`ml_policy`(configC重み)の意思決定経路に、非リーサルATTACKの選択だけ rule_based の判定に委ねるゲートを追加する。ミラー対戦で configC 単体からさらに勝率を積み増せるかを検証する。

**境界の制約(ユーザー確認済み):** `ptcg_ai/rule_based/main_turn_parts/` 配下(`attack.py` / `proposals.py` / `weights.py`、担当B領域)は **一切変更しない**。既存の公開関数を読み取り専用で import・呼び出しするだけに限定する。変更するのは `ptcg_ai/ml_policy/ml_policy_agent.py` とその周辺(config・テスト・診断スクリプト)のみ。

---

## スコープ

**やること:**
- `ml_policy_agent.py` に、rule_baseの「このターンはattackカテゴリを選ぶべきか」判定をゲートにしたハイブリッド関数を追加。
- config フラグ(`attack_hybrid.enabled`)で既定OFF、明示的に有効化した場合のみ挙動が変わる(既存の`lethal_search.enabled`等と同じ後方互換パターン)。
- configC重み固定でON/OFFのミラーhead-to-head評価スクリプトを追加。
- ユニットテスト追加(`_try_lethal`系テストと同型)。

**Non-goals:**
- `rule_based/main_turn_parts/*.py` の変更(境界制約)。
- `AGENT_TYPE`切り替え・本番config(`ml_lethal`)の既定値変更・Kaggle再提出。
- 非ミラー・複数アーキタイプ評価(有意な勝ち越しが出たあとの次ステップ)。
- Step0(オフライン事前検証)以外のPolicyModel再学習・アーキ変更。

---

## Step0: オフライン事前検証(ゲートの質の確認)

**目的:** 「rule_baseの`proposals.decide()`がattackカテゴリを選んだ回」に絞ったとき、その回のATTACK一致率が94.9%からどう変わるかを確認する。ゲートそのものの質を対戦前に安く見積もる。

**やること:**
- 既存のオフライン評価データセット(`step2-offline-evaluation.md`で使ったもの、`kaggle_replays/policy_net/` 配下の該当jsonl/npz)から、ATTACK型選択肢を含む意思決定点を抽出。
- 各点について `rb_proposals.collect_proposals(obs)` を実行し、`max(score + weights.CATEGORY_BASE_WEIGHT[cat])` のカテゴリが `"attack"` になった点だけを部分集合とする。
- その部分集合における「rule_base推奨インデックス」と「実際の(上位プレイヤーの)選択」の一致率を計算。

**完了条件:**
- 一致率(件数付き)を `results/2026-07-21_attack_hybrid_gate_offline.md` に記録する。
- 一致率が94.9%から大きく劣化していない(目安: 5pt以上落ちていない)ことを確認できれば Step1 以降へ進む。大きく劣化していれば、ゲート条件の見直し(例: `proposals.decide()`のスコア差が一定以上ある場合のみ採用する等)を検討してから進める。

**実施結果(2026-07-21、`results/2026-07-21_attack_hybrid_gate_offline.md`):**

素朴なゲート(`collect_proposals(obs)`の最高スコアカテゴリが"attack"かどうかだけを見る)は
一致率23.2%まで劣化し**不採用**。原因は、ATTACKのKOボーナス(1000点)がboard等の得点
(20点程度)を常に上回るため、進化を複数回してから最後に攻撃するような同一ターン内の
連続したMAIN選択の**途中**でも即攻撃を選んでしまう欠陥(rule_baseがPLAY/ABILITY/ATTACHで
一致率が低いのと同根)。

**ゲート条件を修正して採用:** `draw`/`board`/`ability`/`energy`のいずれの提案も存在しない
(=もう他にやるべき展開が残っていない)行に限定したところ、一致率89.7%(発火350件/対象
368件)まで回復した。recallは20%と低い(発火頻度自体が狭い)が、これは意図した安全側の
挙動(曖昧な局面ではPolicyModelに委ねる高精度・低頻度介入)。**以降のStep1はこの
`_BLOCKING_CATEGORIES = {"draw", "board", "ability", "energy"}` 条件を組み込んだ設計とする。**

---

## Step1: `ml_policy_agent.py` にゲート関数を追加

### 変更ファイル

`sample_submission/ptcg_ai/ml_policy/ml_policy_agent.py`

### 追加import(読み取り専用、rule_based側は無変更)

```python
from cg.api import OptionType
from ptcg_ai.rule_based.main_turn_parts import proposals as rb_proposals
from ptcg_ai.rule_based.main_turn_parts import weights as rb_weights
```

### 関数シグネチャ

```python
def _try_attack_hybrid(obs: Observation, config: dict | None = None) -> list[int] | None:
    """rule_base が『このターンは attack カテゴリを選ぶべき』と判定した場合のみ、
    そのATTACK選択のインデックスを返す。それ以外(rule_baseが他カテゴリを選ぶ/
    ATTACK型選択肢が無い/draw・board・ability・energyのいずれかが提案されている/
    config無効)は None を返し、呼び出し側はPolicyModelに委ねる。

    `rule_based/main_turn_parts` (attack.py, proposals.py, weights.py) を読み取り専用で
    import・呼び出しするだけで、ファイル自体は変更しない(担当領域を尊重するため。
    attack-rulebased-hybrid-strategy.md §4.2)。draw/board/ability/energyのいずれかが
    提案されている行を除外するのは、Step0のオフライン実測(results/2026-07-21_
    attack_hybrid_gate_offline.md)で判明した「他の展開が残っている途中でも即攻撃を
    選んでしまう」欠陥を避けるため(一致率23.2%→89.7%に回復)。

    Args:
        obs: core.agent から渡される Observation。obs.select は maxCount==1 の
            MAIN選択(PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/ATTACK/ENDが混在)を想定。
        config: `_try_lethal` と同じ注入パターン。`config["attack_hybrid"]["enabled"]`
            が truthy のときだけ動作する(既定 False、既存挙動を変えない)。

    Returns:
        list[int] | None: rule_base が選んだATTACKの選択(1要素)。条件を満たさない/
            例外時/contract違反時は None。
    """
```

### 実装方針(擬似コード)

```python
def _try_attack_hybrid(obs, config=None):
    if obs.current is None:
        return None
    if config is None:
        config = _get_config()
    hybrid_config = (config or {}).get("attack_hybrid") or {}
    if not hybrid_config.get("enabled", False):
        return None

    select = obs.select
    if not any(opt.type == OptionType.ATTACK for opt in select.option):
        return None  # ATTACK型の選択肢が無ければゲートを検討する必要が無い

    try:
        rb_props = rb_proposals.collect_proposals(obs)
    except Exception:
        return None
    if not rb_props:
        return None

    # Step0実測(results/2026-07-21_attack_hybrid_gate_offline.md): draw/board/ability/energy
    # のいずれかが提案されている(=まだ他にやるべき展開が残っている)行では、ATTACKのKOボーナス
    # (1000点)が常に他カテゴリの生スコアを上回ってしまい、進化等の途中でも即攻撃を選んでしまう
    # (一致率23.2%まで劣化)。このため、これらの提案が1つも無い場合に限定する
    # (一致率89.7%まで回復、代わりに発火頻度は絞られる=安全側の低recall高精度介入)。
    categories_present = {p.category for p in rb_props}
    if categories_present & _BLOCKING_CATEGORIES:
        return None

    def total_score(p):
        return p.score + rb_weights.CATEGORY_BASE_WEIGHT.get(p.category, 0.0)

    best = max(rb_props, key=total_score)
    if best.category != "attack":
        return None  # rule_baseも「今は攻撃すべきでない」と判定 → PolicyModelに委ねる

    if not _is_valid_action(best.select, select):
        return None
    return best.select
```

```python
# _try_attack_hybrid と同じモジュールレベルに定義(rule_based側は変更しない、ml_policy_agent.py側の定数)。
_BLOCKING_CATEGORIES = {"draw", "board", "ability", "energy"}
```

### `_select_action` への組み込み

`_try_lethal` の直後、モデルスコアリングの直前に挿入する(確定リーサル > ATTACKゲート > PolicyModel の優先順)。

```python
def _select_action(obs, config=None):
    select = obs.select

    lethal_action = _try_lethal(obs, config=config)
    if lethal_action is not None:
        return lethal_action

    attack_action = _try_attack_hybrid(obs, config=config)
    if attack_action is not None:
        return attack_action

    model = _get_model(config)
    if select.maxCount == 1:
        idx = model.select_option(obs)
        return [idx if idx is not None else 0]
    return _greedy_multi_select(obs, model, select)
```

**完了条件:**
- `config`未指定時(既定 `ml_lethal`、`attack_hybrid`キー無し)は `_try_attack_hybrid` が常に `None` を返し、既存の挙動(configC単体と同一)が一切変わらないこと。
- `rule_based/main_turn_parts/*.py` に diff が無いこと(`git diff --stat` で確認)。
- モジュールdocstring(冒頭の「rule_based/action_selectionは一切変更・呼び出しせず独立に完結させる」という記述)を、今回の例外を明記する形に更新する。

---

## Step2: configとテスト

### config

新規configファイルは作らない。`league`側の診断スクリプトが `load_config("ml_lethal")` をコピーし、
```python
config_candidate["policy_weights_path"] = str(configC_weights_path)
config_candidate["attack_hybrid"] = {"enabled": True}
```
のように上書きする(`_diag_skillconc_head_to_head.py` が `policy_weights_path` だけ上書きしているのと同じパターン)。baseline側は同じ `policy_weights_path`(configC)を使い、`attack_hybrid`キーは付けない(既定False)。

### ユニットテスト

`sample_submission/tests/unit/test_ml_policy_agent.py` に追記(`_try_lethal`系テストと同型、`monkeypatch`で`rb_proposals.collect_proposals`をスタブ化):

- `test_try_attack_hybrid_returns_none_when_disabled`: config無指定/`attack_hybrid`無しでは常に`None`。
- `test_try_attack_hybrid_returns_none_when_no_attack_option`: `obs.select.option`にATTACK型が無ければ`rb_proposals`を呼ばずに`None`(呼ばれていないことを`monkeypatch`+フラグで確認)。
- `test_try_attack_hybrid_returns_action_when_rule_base_picks_attack`: `collect_proposals`スタブが`category="attack"`の提案を最高スコアで返すとき、その`select`を採用する。
- `test_try_attack_hybrid_returns_none_when_rule_base_picks_other_category`: 最高スコアが`"attack"`以外なら`None`(PolicyModelに委ねる)。
- `test_try_attack_hybrid_swallows_exceptions`: `collect_proposals`が例外を投げても`None`(クラッシュしない)。
- `test_try_attack_hybrid_rejects_illegal_action`: rule_base提案のインデックスが範囲外なら`None`。
- `agent()`経由の統合テスト: `attack_hybrid.enabled=True` かつ上記スタブ条件で、`agent(obs, config=...)`が rule_base側の選択をそのまま返すこと。

**完了条件:** 上記すべてがPASS。既存の`_try_lethal`系テストも無変更でPASSし続ける(回帰なし)。

**実施結果(2026-07-21):** Step0の結果を受け、`_BLOCKING_CATEGORIES`条件込みで実装。
上記7テストに加え`test_try_attack_hybrid_returns_none_when_blocking_category_present`を追加し
(draw/board/ability/energyそれぞれで検証)、`tests/unit/test_ml_policy_agent.py` 全17件PASS
(既存10件も回帰無し)。`git diff --stat -- sample_submission/ptcg_ai/rule_based/main_turn_parts`
は空、境界制約も満たしている。

---

## Step3: 診断スクリプト(configC baseline vs configC+ATTACK委譲)

### 新規ファイル

`league/_diag_attack_hybrid_head_to_head.py`(`_diag_skillconc_head_to_head.py`を土台に複製・改変)

### 差分

- `--candidate-weights`は不要(両側とも`policy_weights_configC.json`固定)。代わりに`--weights-path`(既定`kaggle_replays/policy_net/policy_weights_configC.json`)を受け取る。
- `config_baseline`: `load_config("ml_lethal")` + `policy_weights_path=configC` + `attack_hybrid`キー無し。
- `config_candidate`: 上記に加え `attack_hybrid = {"enabled": True}`。
- 出力JSONに `error_reasons`・先後別勝率に加え、可能であれば「ATTACKゲートが発火した回数」も記録できるとよい(`_try_attack_hybrid`にオプションのカウンタ/ログフックを追加するかは軽量に倒す。必須ではない)。

### 実行(想定)

```
python league/_diag_attack_hybrid_head_to_head.py \
    --weights-path kaggle_replays/policy_net/policy_weights_configC.json \
    --candidate-name attack_hybrid_on --games 200 --seed-start 40000 \
    --out league/results/_diag_attack_hybrid_configC.json
```

**完了条件:**
- 200試合以上、エラー0件(または現行同等)、先手後手半々。
- Wilson 95% CIを含む結果JSONと、`results/2026-07-21_attack_hybrid_headtohead.md`への要約記録。

---

## 判定基準(strategy doc §3.4と同一)

- CI下限が0.5を上回る: 有意な勝ち越し → 非ミラー評価(Priority 0.2相当)へ進める候補にする。
- 点推定は上がるがCIが0.5を含む: 追加試合または保留。
- 50%付近: 効果なし、この軸はクローズしてPriority 2(評価設計)等へ戻る。

---

## 完了条件(この計画全体)

1. [x] Step0のオフライン結果が記録されている(`results/2026-07-21_attack_hybrid_gate_offline.md`)。
2. [x] `rule_based/main_turn_parts/*.py` に diff が無いことを確認済み(`git diff --stat`)。
3. [x] Step2のユニットテストが全てPASS(17件、既存分含め回帰無し)。
4. [x] Step3のhead-to-head結果(600試合、200+400の2バッチ)が`results/2026-07-21_attack_hybrid_headtohead.md`に記録されている。
5. [x] 上記いずれかの段階でゲート品質や勝率が明確に不十分と判明した場合は、その時点で中止し理由を記録する(次の一手を無理に進めない)。

**結論(2026-07-21): 中止。** 600試合合算で勝率51.8%(Wilson 95% CI [47.8%, 55.8%])、
0.5をほぼ中心に挟んでおり有意差なし。オフラインでのゲート精度(89.7%)は高かったが、
発火頻度がMAIN行の約2.5%と狭く、対戦全体の勝率に効果が現れるほどではなかったと考えられる。
詳細は`results/2026-07-21_attack_hybrid_headtohead.md`。非ミラー評価には進まない。
`_try_attack_hybrid`実装自体は既定OFFのままコードに残す(将来の再検討の土台)。
