"""オーガポン戦略Q-critic専用のstate・option特徴生成(design.md §7、Phase3 item1)。

既存の166次元state encoder(``ptcg_ai.learning.encoder``)をそのまま再利用しつつ、
戦略比較(``EX_TEMPO`` vs ``SINGLE_PRIZE_ROTATION``)に必要な特徴だけを追加する。
既存の ``encoder.py`` / 既存重みJSONの契約は変更しない(この専用encoderは完全に別ファイル)。

## 出力の3分割(``collect_ogerpon_counterfactuals.py`` のJSONLスキーマと対応)

- ``continuous_features``: 既存166次元 + 盤面集約特徴(§7.3)。連続値。
- ``slot_card_ids``: 場の12スロット(自分 active+bench5、相手 active+bench5)の
  card_id。埋め込みテーブルへの生の整数インデックス(標準化しない)。
- ``option_features``: ``ogerpon_strategy.StrategyCandidate`` 1件に対応するoption特徴
  (§7.4)。EX_TEMPOとSINGLE_PRIZE_ROTATIONそれぞれについて別々に計算する。

## 前提

``encode_continuous_features``/``encode_slot_card_ids`` は「``state.yourIndex`` が
学習側(learner)である局面」で呼ぶこと(Strategy Window Builderの発火局面は必ず
学習側自身の意思決定点なので、収集パイプラインではこの前提が自然に満たされる)。
"""

from __future__ import annotations

from cg.api import OptionType, SelectType, State

from ptcg_ai.learning import encoder
from ptcg_ai.ml_policy import ogerpon_planner as P
from ptcg_ai.ml_policy import ogerpon_strategy as STRAT

# このencoderの契約バージョン。特徴の名前・順序・本数を変えたら必ず上げる。
# ogerpon_strategy_model.py の実行時ローダーが、export済みJSONのfeature_contractと
# 現在のこの値・SLOT_NAMES/CONTINUOUS_FEATURE_NAMES/OPTION_FEATURE_NAMESを突き合わせて
# 完全一致を検証する(外部レビュー指摘: 次元数が同じままの並び替えを検出できていなかった)。
ENCODER_VERSION = 1

# 場のスロット順序: 自分 active + bench5、相手 active + bench5。
SLOT_COUNT = 12
_BENCH_SLOTS = 5
SLOT_NAMES: list[str] = (
    ["self_active"] + [f"self_bench{i}" for i in range(_BENCH_SLOTS)]
    + ["opp_active"] + [f"opp_bench{i}" for i in range(_BENCH_SLOTS)]
)

# 未使用/不明なスロット・埋め込みのID(学習時に「空」として扱う)。
UNKNOWN_CARD_ID = 0

_TRIGGER_KINDS = [
    STRAT.TRIGGER_MAIN_ATTACH, STRAT.TRIGGER_PROMOTE,
    STRAT.TRIGGER_RETREAT, STRAT.TRIGGER_CONTINUE,
    "terminate",  # design.md §7.4 が挙げる5種のうち、Phase2時点では検出未実装
]
_OPTION_NAME_ORDER = [STRAT.EX_TEMPO, STRAT.SINGLE_PRIZE_ROTATION]
_FIRST_ACTION_TYPE_BUCKETS = ["attach", "retreat", "card", "other"]

AGGREGATE_FEATURE_NAMES: list[str] = [
    "required_kos_against_me", "required_kos_against_opp",
    "ready_ex_count", "ready_non_ex_count", "attacker_chain_length",
    "bulu_location", "bulu_hp_ratio", "bulu_energy_count", "bulu_true_shortfall",
    "active_can_attack_now", "active_ko_risk_tier",
    "ogerpon_extra_accel_available", "ogerpon_extra_accel_unknown",
    "damaged_ex_count", "damaged_ex_prize_at_risk",
    "opp_active_current_damage", "opp_active_next_turn_damage",
    "opp_active_variable_damage_flag",
]

OPTION_FEATURE_NAMES: list[str] = (
    [f"option_is_{n}" for n in _OPTION_NAME_ORDER]
    + [f"trigger_is_{k}" for k in _TRIGGER_KINDS]
    + [f"first_action_type_{b}" for b in _FIRST_ACTION_TYPE_BUCKETS]
    + [
        "target_shortfall_before", "target_shortfall_after",
        "target_can_attack_before", "target_can_attack_after",
        "target_best_damage_before", "target_best_damage_after",
        "target_can_ko_opp_active_before", "target_can_ko_opp_active_after",
        "ready_attacker_count_delta",
        "required_ko_gain", "turns_until_ready",
        "safety_legal", "safety_risks_losing_active_attack", "safety_target_certain_ko",
    ]
)

CONTINUOUS_FEATURE_NAMES: list[str] = list(encoder.FEATURE_NAMES) + list(AGGREGATE_FEATURE_NAMES)


def _ko_risk_tier(risk: str) -> float:
    return {P.SAFE: 0.0, P.POSSIBLE_KO: 1.0, P.LIKELY_KO: 2.0, P.CERTAIN_KO: 3.0}.get(risk, 0.0)


def _slot_pokemon(state: State, me: int) -> list:
    mine = state.players[me]
    opp = state.players[1 - me]

    def _side(player):
        active = next((s for s in (player.active or []) if s is not None), None)
        bench = [s for s in (player.bench or []) if s is not None]
        slots = [active]
        for i in range(_BENCH_SLOTS):
            slots.append(bench[i] if i < len(bench) else None)
        return slots

    return _side(mine) + _side(opp)


def encode_slot_card_ids(state: State, me: int) -> list[int]:
    """場の12スロット(自分 active+bench5、相手 active+bench5)のcard_id。空スロットは0。"""
    return [(s.id if s is not None else UNKNOWN_CARD_ID) for s in _slot_pokemon(state, me)]


def _board_aggregate_features(state: State, me: int) -> list[float]:
    mine = state.players[me]
    opp = state.players[1 - me]
    active = next((s for s in (mine.active or []) if s is not None), None)
    opp_active = next((s for s in (opp.active or []) if s is not None), None)
    bench = [s for s in (mine.bench or []) if s is not None]

    ready_ex = sum(1 for s in [active] + bench if s is not None and P.is_ex(s) and P.can_attack_now(s))
    ready_non_ex = sum(
        1 for s in [active] + bench if s is not None and not P.is_ex(s) and P.can_attack_now(s))
    chain = sum(1 for s in [active] + bench if s is not None and P.energy_shortfall(s) <= 1)

    bulu = next((s for s in bench if P.is_bulu(s)), None)
    bulu_location = 0.0
    if active is not None and P.is_bulu(active):
        bulu, bulu_location = active, 2.0
    elif bulu is not None:
        bulu_location = 1.0

    cfg = P.main_config(None)
    active_ready = P.can_attack_now(active) if active is not None else False
    active_risk, _ = P.ko_risk(active, opp_active, cfg) if active is not None else (P.SAFE, {})

    damaged_ex = [s for s in [active] + bench
                 if s is not None and P.is_ex(s) and s.maxHp and s.hp < s.maxHp * 0.5]
    opp_tiers = P.damage_tiers(opp_active, active, cfg)

    return [
        P.kos_needed_against(mine), P.kos_needed_against(opp),
        float(ready_ex), float(ready_non_ex), float(chain),
        bulu_location,
        (float(bulu.hp) / float(bulu.maxHp) if (bulu is not None and bulu.maxHp) else 0.0),
        (float(len(P.energies_of(bulu))) if bulu is not None else 0.0),
        (float(P.energy_shortfall(bulu)) if bulu is not None else 0.0),
        1.0 if active_ready else 0.0, _ko_risk_tier(active_risk),
        0.0, 1.0,  # オーガポン特性の追加加速: Observationだけでは判定できないため常に「不明」
        float(len(damaged_ex)), float(sum(P.prize_value(s) for s in damaged_ex)),
        float(opp_tiers["current_payable_damage"]), float(opp_tiers["credible_next_turn_damage"]),
        1.0 if opp_tiers["any_variable_damage"] else 0.0,
    ]


def encode_continuous_features(state: State, me: int) -> list[float]:
    """既存166次元 + 盤面集約特徴(§7.3)。長さは ``len(CONTINUOUS_FEATURE_NAMES)``。"""
    return encoder.encode_state_from_state(state) + _board_aggregate_features(state, me)


def _first_action_type_bucket(select, first_action) -> str:
    if not first_action or select is None or not (0 <= first_action[0] < len(select.option)):
        return "other"
    otype = int(getattr(select.option[first_action[0]], "type", -1))
    if otype == int(OptionType.ATTACH):
        return "attach"
    if otype == int(OptionType.RETREAT):
        return "retreat"
    if int(select.type) == int(SelectType.CARD):
        return "card"
    return "other"


def encode_option_features(candidate, state: State, me: int, select=None) -> list[float]:
    """``StrategyCandidate`` 1件のoption特徴(§7.4)。長さは ``len(OPTION_FEATURE_NAMES)``。"""
    mine = state.players[me]
    opp = state.players[1 - me]
    opp_active = next((s for s in (opp.active or []) if s is not None), None)
    active = next((s for s in (mine.active or []) if s is not None), None)

    feats: list[float] = [1.0 if candidate.option_name == n else 0.0 for n in _OPTION_NAME_ORDER]
    feats += [1.0 if candidate.trigger_kind == k else 0.0 for k in _TRIGGER_KINDS]

    bucket = _first_action_type_bucket(select, candidate.first_action)
    feats += [1.0 if bucket == b else 0.0 for b in _FIRST_ACTION_TYPE_BUCKETS]

    target = None
    if candidate.target_serial is not None:
        for s in [active] + [b for b in (mine.bench or []) if b is not None]:
            if s is not None and getattr(s, "serial", None) == candidate.target_serial:
                target = s
                break

    if target is not None:
        shortfall_before = P.energy_shortfall(target)
        can_attack_before = P.can_attack_now(target)
        damage_before = P.best_damage(target)
        opp_hp = float(opp_active.hp) if opp_active is not None else 0.0
        can_ko_before = 1.0 if (can_attack_before and damage_before >= opp_hp and opp_hp > 0) else 0.0
        is_active_target = (active is not None
                            and getattr(target, "serial", None) == getattr(active, "serial", None))
        if bucket == "attach" and not is_active_target:
            # 手貼り後の色は候補からは分からない(first_actionのcardIdはこの時点で未解決)ため、
            # 「1エネ分だけ shortfall が縮まる」近似で after を計算する(実際の色不一致は
            # ogerpon_planner._attach_is_productive が別途弾いているので、ここでは楽観近似)。
            shortfall_after = max(0, shortfall_before - 1)
            can_attack_after = shortfall_after <= 0
            damage_after = damage_before  # 打点は攻撃側のエネ色構成に依存し近似できないため据え置き
        else:
            shortfall_after, can_attack_after, damage_after = shortfall_before, can_attack_before, damage_before
        can_ko_after = 1.0 if (can_attack_after and damage_after >= opp_hp and opp_hp > 0) else 0.0
        feats += [
            float(shortfall_before), float(shortfall_after),
            1.0 if can_attack_before else 0.0, 1.0 if can_attack_after else 0.0,
            float(damage_before), float(damage_after),
            can_ko_before, can_ko_after,
        ]
        ready_before = sum(1 for s in [active] + [b for b in (mine.bench or []) if b is not None]
                           if s is not None and P.can_attack_now(s))
        ready_after = ready_before + (1 if (not can_attack_before and can_attack_after) else 0)
        feats.append(float(ready_after - ready_before))
    else:
        feats += [0.0] * 9

    feats += [candidate.required_ko_gain, float(candidate.turns_until_ready)]
    feats += [
        1.0 if candidate.safety_flags.get("legal") else 0.0,
        1.0 if candidate.safety_flags.get("single_prize_risks_losing_active_attack") else 0.0,
        1.0 if candidate.safety_flags.get("single_prize_target_faces_certain_ko") else 0.0,
    ]
    return feats


def encode_strategy_pair(obs, me: int, candidates) -> dict:
    """``ogerpon_strategy.build_candidates`` が返す(EX_TEMPO, SINGLE_PRIZE_ROTATION)の
    ペアから、両者に共通のcontinuous/slot特徴と、候補ごとのoption特徴を組み立てる。
    """
    state = obs.current
    select = obs.select
    continuous = encode_continuous_features(state, me)
    slots = encode_slot_card_ids(state, me)
    per_option = {
        c.option_name: encode_option_features(c, state, me, select) for c in candidates
    }
    return {"continuous_features": continuous, "slot_card_ids": slots, "option_features": per_option}
