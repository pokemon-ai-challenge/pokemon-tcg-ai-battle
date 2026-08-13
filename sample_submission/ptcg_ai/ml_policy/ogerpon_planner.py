"""オーガポンデッキ専用のリソースプランナー(カプ・ブルル中継戦略)。

## なぜ必要か

`pipeline.search`(PIMC前読み)は ``SelectType.MAIN`` かつ ``maxCount == 1`` にしか適用されない
(pipeline.py の適用範囲)。実測(60試合/4790選択)では:

    MAIN|MAIN            60.9%   <- PIMC が関与する
    CARD|ATTACH_TO       15.6%   <- エネを誰に付けるか。Policy の argmax のみ
    CARD|TO_ACTIVE        1.4%   <- 気絶後に誰を前に出すか。Policy の argmax のみ
    CARD|SWITCH           1.2%   <- 交代先。Policy の argmax のみ

つまり「エネ配分」「昇格」「交代先」は探索も評価関数も一切関与しておらず、`leaf_eval.py` の
係数をどう変えても挙動は変わらない(実測でカプ・ブルルのバトル場起用率が
24.0% -> 24.0% と不変だったのはこのため)。

このモジュールは、その3つの選択コンテキストに限定して Policy スコアへ補正を掛ける。

## 安全性の原則

- **既定 OFF**。config で明示的に有効化しない限り一切発火せず、従来と同一挙動。
- **オーガポンデッキ以外では発火しない**(自分の60枚にオーガポンex のカードIDが無ければ無効)。
- **確定リーサル・今ターンの攻撃を妨害しない**(呼び出し側で lethal の後にのみ適用する)。
- カード名の文字列比較をしない。技コスト・にげる・ex 判定はすべて **エンジンのカードデータ** から引く。
- 例外時は必ず「補正なし」を返し、既存 Policy / PIMC の選択をそのまま通す。
- 永続状態を持たない(毎回盤面から再計算する)ので、試合間で状態が混ざらない。
"""

from __future__ import annotations

import re

from cg.api import (AreaType, CardType, Observation, OptionType, SelectContext,
                    SelectData, SelectType)

from ptcg_ai.shared import card_cache

# このプランナーが対象とするデッキの識別子(カード名ではなくカードID)。
# オーガポン みどりのめんex。自分の60枚にこれが含まれるときだけ有効化する。
OGERPON_EX_CARD_ID = 96

# 既定係数。config の "ogerpon_planner" で上書きできる。
DEFAULTS: dict = {
    "enabled": False,
    # ATTACH_TO: 控えの1枚アタッカーを完成させる手貼りを押す強さ
    "attach_backup_bonus": 2.0,
    # ATTACH_TO: 現在のバトル場が攻撃可能になっていないときは、そちらを優先する強さ
    "attach_active_first_bonus": 4.0,
    # TO_ACTIVE / SWITCH: 直ちに攻撃できる候補を押す強さ
    "promote_ready_bonus": 4.0,
    # TO_ACTIVE / SWITCH: サイド1枚で済む候補を押す強さ(サイドパリティで意味があるときのみ)
    "promote_one_prize_bonus": 2.0,
    # TO_ACTIVE / SWITCH: 攻撃不能な候補を下げる強さ(にげるコストが重いほど強く下げる)
    "promote_not_ready_penalty": 3.0,
    # 交代先として「削れた ex」を温存する(=選ばない)強さ
    "preserve_damaged_ex_bonus": 1.5,
    # 相手の残サイドが偶数のときだけサイドパリティのボーナスを適用する
    "require_even_prize_for_parity": True,
}


# ---------------------------------------------------------------------------
# エンジン由来のドメイン知識(カード名に依存しない)
# ---------------------------------------------------------------------------

def _card(card_id):
    return card_cache.get_card(int(card_id))


def is_ex(pokemon) -> bool:
    """このポケモンが「ルールを持つポケモン」(気絶時にサイド2枚)か。"""
    try:
        return bool(_card(pokemon.id).ex)
    except Exception:  # noqa: BLE001
        return False


def prize_value(pokemon) -> int:
    """気絶したとき相手が取るサイド枚数。判定できないときは安全側の 1。"""
    return 2 if is_ex(pokemon) else 1


def retreat_cost(pokemon) -> int:
    try:
        return int(_card(pokemon.id).retreatCost or 0)
    except Exception:  # noqa: BLE001
        return 0


def attack_costs(pokemon) -> list[list[int]]:
    """このポケモンが持つ全ワザのエネルギーコスト(EnergyType の値の並び)。"""
    try:
        c = _card(pokemon.id)
        out = []
        for aid in (c.attacks or []):
            a = card_cache.get_attack(aid)
            if a.energies:
                out.append(list(a.energies))
        return out
    except Exception:  # noqa: BLE001
        return []


def _satisfies(cost: list[int], attached: list[int]) -> bool:
    """付いているエネルギー ``attached`` がコスト ``cost`` を満たすか。

    コストのうち 0(COLORLESS)は任意のエネルギーで払える。色指定は同じ色で払う。
    エンジンの特殊エネルギー(複数色を兼ねる等)までは解釈しないので、
    「満たせるのに満たせないと見なす」安全側に倒れることがある。
    """
    pool = list(attached)
    # 先に色指定を消費する。
    for need in cost:
        if need == 0:
            continue
        if need in pool:
            pool.remove(need)
        else:
            return False
    # 残りは無色コスト。何で払ってもよい。
    colorless = sum(1 for c in cost if c == 0)
    return len(pool) >= colorless


def energies_of(pokemon) -> list[int]:
    out = []
    for e in (pokemon.energies or []):
        try:
            out.append(int(e))
        except Exception:  # noqa: BLE001
            continue
    return out


def can_attack_now(pokemon) -> bool:
    """今ついているエネルギーで、いずれかのワザを撃てるか。"""
    if pokemon is None:
        return False
    attached = energies_of(pokemon)
    for cost in attack_costs(pokemon):
        if _satisfies(cost, attached):
            return True
    return False


def energy_shortfall(pokemon) -> int:
    """一番安いワザを撃つのにあと何個エネルギーが要るか(撃てるなら 0)。"""
    if pokemon is None:
        return 99
    attached = len(energies_of(pokemon))
    costs = attack_costs(pokemon)
    if not costs:
        return 99
    if can_attack_now(pokemon):
        return 0
    return max(0, min(len(c) for c in costs) - attached)


def best_damage(pokemon) -> int:
    """撃てるワザの中での最大打点(撃てないなら 0)。"""
    if pokemon is None:
        return 0
    attached = energies_of(pokemon)
    best = 0
    try:
        c = _card(pokemon.id)
        for aid in (c.attacks or []):
            a = card_cache.get_attack(aid)
            if a.energies and _satisfies(list(a.energies), attached):
                try:
                    best = max(best, int(a.damage or 0))
                except (TypeError, ValueError):
                    continue
    except Exception:  # noqa: BLE001
        return best
    return best


# ---------------------------------------------------------------------------
# サイド勘定
# ---------------------------------------------------------------------------

def kos_needed_against(player) -> float:
    """``player`` の残サイドを取り切るのに必要な KO 回数(相手にとって最短)。

    leaf_eval._kos_needed_against と同じ考え方。1枚ポケモンを混ぜると増える。
    """
    import math

    remaining = len(player.prize or [])
    if remaining <= 0:
        return 0.0
    values = []
    for slot in (player.active or []):
        if slot is not None:
            values.append(prize_value(slot))
    for slot in (player.bench or []):
        if slot is not None:
            values.append(prize_value(slot))
    if not values:
        return float(remaining)
    values.sort(reverse=True)
    kos = taken = 0
    for v in values:
        if taken >= remaining:
            break
        taken += v
        kos += 1
    if taken < remaining:
        kos += math.ceil((remaining - taken) / 2)
    return float(kos)


def _kos_needed_for_values(remaining: int, values: list[int]) -> float:
    """残サイド ``remaining`` を、サイド価値 ``values`` の集合から取り切る最短KO回数。"""
    import math

    if remaining <= 0:
        return 0.0
    if not values:
        return float(remaining)
    vs = sorted(values, reverse=True)
    kos = taken = 0
    for v in vs:
        if taken >= remaining:
            break
        taken += v
        kos += 1
    if taken < remaining:
        kos += math.ceil((remaining - taken) / 2)
    return float(kos)


def required_ko_delta(player, candidate) -> float:
    """``candidate``(1枚ポケモン)を場に置くことで、相手の必要KO回数が何回増えるか。

    「挟んだ場合」と「挟まなかった場合」の必要KO回数を実際に両方計算して差を取る。
    偶数サイドかどうかのような代理条件ではなく、**実差**で判定するための関数。

    戻り値が 1 以上なら挟む価値が大きい。0 なら挟んでも相手の手間は増えないので
    過剰に優先してはいけない。
    """
    try:
        remaining = len(player.prize or [])
        if remaining <= 0:
            return 0.0
        on_field = []
        for slot in list(player.active or []) + list(player.bench or []):
            if slot is not None:
                on_field.append(slot)
        cand_serial = getattr(candidate, "serial", None)
        with_values = [prize_value(p) for p in on_field]
        # candidate を除いた盤面(=挟まなかった場合)。除いた枠は ex(2枚)が入ると仮定する
        # (このデッキの控えはオーガポンex なので、それが最も現実的な対比になる)。
        without_values = [prize_value(p) for p in on_field
                          if getattr(p, "serial", None) != cand_serial]
        if len(without_values) == len(with_values):
            return 0.0  # candidate が場に見つからない = 比較できない
        without_values.append(2)
        return max(0.0, _kos_needed_for_values(remaining, with_values)
                   - _kos_needed_for_values(remaining, without_values))
    except Exception:  # noqa: BLE001
        return 0.0


def parity_gain_from_one_prize(player) -> float:
    """自分の盤面が「全部 ex」だった場合と比べて、相手の必要KO回数が何回増えているか。

    0 なら1枚ポケモンを挟む意味が無い(過大評価を避けるためのゲート)。
    """
    import math

    remaining = len(player.prize or [])
    if remaining <= 0:
        return 0.0
    all_ex_kos = math.ceil(remaining / 2)
    return max(0.0, kos_needed_against(player) - all_ex_kos)


# ---------------------------------------------------------------------------
# 有効化判定
# ---------------------------------------------------------------------------

def planner_config(config: dict | None) -> dict:
    cfg = dict(DEFAULTS)
    user = (config or {}).get("ogerpon_planner") or {}
    if isinstance(user, dict):
        cfg.update(user)
    return cfg


def deck_is_ogerpon(deck_ids) -> bool:
    """自分の60枚がオーガポンデッキか(カードIDで判定。名前文字列は使わない)。"""
    try:
        return OGERPON_EX_CARD_ID in set(int(c) for c in (deck_ids or []))
    except Exception:  # noqa: BLE001
        return False


def is_active(config: dict | None, deck_ids) -> bool:
    cfg = planner_config(config)
    return bool(cfg.get("enabled")) and deck_is_ogerpon(deck_ids)


# ---------------------------------------------------------------------------
# 局面分類
# ---------------------------------------------------------------------------

NORMAL = "NORMAL"
BULU_SETUP = "BULU_SETUP"
BULU_READY = "BULU_READY"
BULU_ACTIVE = "BULU_ACTIVE"
NEXT_OGERPON_SETUP = "NEXT_OGERPON_SETUP"


def classify(state, me: int) -> str:
    """盤面から局面を分類する(永続状態を持たず毎回再計算する)。"""
    try:
        mine = state.players[me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        bench = [s for s in (mine.bench or []) if s is not None]
        if active is not None and not is_ex(active):
            # 1枚アタッカーが前に出ている = 中継中。次のアタッカーを作る局面。
            ready_backup = any(is_ex(b) and can_attack_now(b) for b in bench)
            return BULU_ACTIVE if not ready_backup else NEXT_OGERPON_SETUP
        one_prize_bench = [b for b in bench if not is_ex(b)]
        if not one_prize_bench:
            return NORMAL
        if any(can_attack_now(b) for b in one_prize_bench):
            return BULU_READY
        return BULU_SETUP
    except Exception:  # noqa: BLE001
        return NORMAL


# ---------------------------------------------------------------------------
# 選択肢スコアへの補正
# ---------------------------------------------------------------------------

def _resolve_option_pokemon(obs: Observation, option, me: int):
    """選択肢が指しているポケモン(自分の場)を返す。解決できなければ None。

    Option.inPlayArea / inPlayIndex を優先し、無ければ area/index を使う。
    エリア定義の解釈はエンジン依存なので、解決できないときは黙って None を返す
    (補正を掛けない = 従来挙動)。
    """
    try:
        state = obs.current
        mine = state.players[me]
        on_field = [s for s in (list(mine.active or []) + list(mine.bench or [])) if s is not None]

        # 1) serial 一致が最も確実(エリア定義の解釈に依存しない)。
        serial = getattr(option, "serial", None)
        if serial is not None:
            for slot in on_field:
                if getattr(slot, "serial", None) == serial:
                    return slot

        # 2) area/index でのフォールバック。ACTIVE/BENCH の並びはエンジン依存なので、
        #    解決できなければ黙って None を返す(補正を掛けない = 従来挙動)。
        for area, index in ((getattr(option, "inPlayArea", None), getattr(option, "inPlayIndex", None)),
                            (getattr(option, "area", None), getattr(option, "index", None))):
            if area is None or index is None:
                continue
            # AreaType.ACTIVE=4 / BENCH=5。`area == 0` は AreaType に存在しない値なので、
            # 以前の実装は ACTIVE 指定の選択肢まで bench 側へ誤って解決していた。
            a = int(area)
            if a == int(AreaType.ACTIVE):
                slots = list(mine.active or [])
            elif a == int(AreaType.BENCH):
                slots = list(mine.bench or [])
            else:
                continue
            i = int(index)
            if 0 <= i < len(slots) and slots[i] is not None:
                return slots[i]
    except Exception:  # noqa: BLE001
        return None
    return None


def score_adjustments(obs: Observation, me: int, config: dict | None,
                      deck_ids) -> list[float] | None:
    """選択肢ごとのスコア加算値を返す。適用外・判定不能なら None。

    呼び出し側は Policy のスコアにこれを足してから argmax を取る。None のときは
    一切触らない(従来挙動)。
    """
    try:
        if not is_active(config, deck_ids):
            return None
        select: SelectData | None = obs.select
        state = obs.current
        if select is None or state is None or not select.option:
            return None
        # 単一選択のみを対象にする(複数選択は貪欲フォールバックに任せる)。
        if select.maxCount != 1:
            return None
        ctx = int(select.context)
        stype = int(select.type)
        if stype != int(SelectType.CARD):
            return None

        cfg = planner_config(config)
        mine = state.players[me]
        opp = state.players[1 - me]
        n = len(select.option)
        adj = [0.0] * n

        # エネルギーの「付与先ポケモン」を選ぶのは ATTACH_FROM(area=ACTIVE/BENCH)。
        # ATTACH_TO は「手札のどのエネルギーカードを使うか」(area=HAND)で、付与先ではない。
        # 実測(20試合/相手4アーキ): ATTACH_TO|HAND 359件、ATTACH_FROM|BENCH 56件、
        # ATTACH_FROM|ACTIVE 15件。名前の語感と逆なので取り違えやすい。
        if ctx in (int(SelectContext.ATTACH_FROM), int(SelectContext.ATTACH_TO)):
            active = next((s for s in (mine.active or []) if s is not None), None)
            active_ready = can_attack_now(active)
            for i, opt in enumerate(select.option):
                target = _resolve_option_pokemon(obs, opt, me)
                if target is None:
                    continue
                is_active_slot = (active is not None
                                  and getattr(target, "serial", None) == getattr(active, "serial", None))
                if not active_ready:
                    # 今ターンの攻撃を成立させるのが最優先。バトル場へ寄せる。
                    if is_active_slot:
                        adj[i] += float(cfg["attach_active_first_bonus"])
                    continue
                # バトル場は既に攻撃可能。控えの1枚アタッカーを完成させに行く。
                if is_active_slot:
                    continue
                if is_ex(target):
                    continue
                short = energy_shortfall(target)
                if short <= 0:
                    continue  # 既に完成しているので追加は不要
                # 完成が近いほど強く押す(遠いほど投資が無駄になりやすい)。
                # 必要KO回数の実差で判定する(偶数サイドは特徴として見るが最終判定には使わない)。
                if required_ko_delta(mine, target) > 0:
                    adj[i] += float(cfg["attach_backup_bonus"]) / float(short)
            return adj

        if ctx in (int(SelectContext.TO_ACTIVE), int(SelectContext.SWITCH)):
            for i, opt in enumerate(select.option):
                target = _resolve_option_pokemon(obs, opt, me)
                if target is None:
                    continue
                ready = can_attack_now(target)
                if ready:
                    adj[i] += float(cfg["promote_ready_bonus"])
                else:
                    # 攻撃できない候補は下げる。にげるコストが重いほど「前で止まる」危険が大きい。
                    rc = retreat_cost(target)
                    adj[i] -= float(cfg["promote_not_ready_penalty"]) * (1.0 + 0.5 * rc)
                if not is_ex(target):
                    # サイド1枚で済む候補。**必要KO回数の実差**で判定する
                    # (偶数サイドという代理条件ではなく、挟んだ場合と挟まない場合を
                    #  両方計算した差。0 なら相手の手間が増えないので加点しない)。
                    delta = required_ko_delta(mine, target)
                    if delta > 0 and ready:
                        adj[i] += float(cfg["promote_one_prize_bonus"]) * min(delta, 2.0)
                else:
                    # 削れた ex は温存したい(=昇格させたくない)。
                    try:
                        if target.maxHp:
                            hp_ratio = float(target.hp) / float(target.maxHp)
                            if hp_ratio < 0.5:
                                adj[i] -= float(cfg["preserve_damaged_ex_bonus"]) * (1.0 - hp_ratio)
                    except Exception:  # noqa: BLE001
                        pass
            return adj

        return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# MAIN(にげる判断)への合成用リソース評価
# ---------------------------------------------------------------------------

# MAIN 用の既定係数。DEFAULTS とは別に持ち、config の同じ "ogerpon_planner" 下で上書きする。
MAIN_DEFAULTS: dict = {
    # PIMC スコア(0..1 の勝ち見込み)へ足す補助項の重み。控えめにする。
    "main_enabled": False,
    "main_shadow_only": False,
    # 削れた ex を逃がして守れる期待サイド損失の重み
    "w_expected_prize_loss": 0.06,
    # カプ・ブルルを挟むことで増える相手の必要KO回数の重み
    "w_required_ko_increase": 0.05,
    # 交代後に攻撃できないことのテンポ損失
    "w_attack_tempo_loss": 0.10,
    # にげるために失うエネルギーの重み(1個あたり)
    "w_retreat_energy_loss": 0.015,
    # ベンチへ逃がした ex がボスの指令等で狙われる危険
    "w_boss_target_risk": 0.02,
    # 次のアタッカー(新しいオーガポン)が準備できていることの価値
    "w_next_attacker_ready": 0.03,
    # KO 危険度の判定に使う想定被ダメージ(相手バトルポケモンの最大打点が引けないときの既定)
    "assumed_incoming_damage": 180,
}

# 直近の bonus 内訳(shadow 診断用。意思決定には使わない)。
LAST_BREAKDOWN: dict = {}

# KO 危険度の段階
SAFE = "SAFE"
POSSIBLE_KO = "POSSIBLE_KO"
LIKELY_KO = "LIKELY_KO"
CERTAIN_KO = "CERTAIN_KO"


def main_config(config: dict | None) -> dict:
    cfg = dict(MAIN_DEFAULTS)
    user = (config or {}).get("ogerpon_planner") or {}
    if isinstance(user, dict):
        cfg.update({k: v for k, v in user.items() if k in MAIN_DEFAULTS})
    return cfg


# 「〜につき N ダメージ追加」型の可変打点を検出する。静的な Attack.damage だけを
# 信じると打点を大きく取り違える(例: オーガポンの技は damage=30 だが実際は
# 「両者のバトルポケモンについているエネルギーの数 x 30」が加算される)。
_ENERGY_SCALING_RE = re.compile(
    r"(\d+)\s+more damage for each Energy attached to (both|your|the opponent)",
    re.IGNORECASE)


def attack_damage_estimate(attacker, defender, attack) -> tuple[int, bool]:
    """``attack`` の実効打点と、可変打点かどうかを返す。

    公開情報(場のポケモンについているエネルギー)だけで計算する。解釈できない
    効果文の場合は静的な ``Attack.damage`` を返しつつ variable=True を立て、
    呼び出し側が「過小評価かもしれない」と扱えるようにする。
    """
    try:
        base = int(attack.damage or 0)
    except (TypeError, ValueError):
        base = 0
    text = attack.text or ""
    m = _ENERGY_SCALING_RE.search(text)
    if m:
        per = int(m.group(1))
        scope = m.group(2).lower()
        n = 0
        if scope == "both":
            n = len(energies_of(attacker)) + len(energies_of(defender))
        elif scope == "your":
            n = len(energies_of(attacker))
        else:
            n = len(energies_of(defender))
        return base + per * n, True
    # 効果文があるのに解釈できない = 打点が変わりうる可能性を残す。
    variable = bool(text) and ("damage" in text.lower())
    return base, variable


def damage_tiers(opp_active, my_active, cfg: dict) -> dict:
    """相手バトルポケモンの打点を3段階で見積もる(すべて公開情報ベース)。

    - current_payable_damage : 今ついているエネルギーで払える技の最大打点
    - credible_next_turn_damage : 手貼り1回(+1エネ)で払えるようになる技を含めた最大打点
    - theoretical_max_damage : コストを無視した最大打点(参考値)
    """
    out = {"current_payable_damage": 0, "credible_next_turn_damage": 0,
           "theoretical_max_damage": 0, "any_variable_damage": False,
           "opponent_attached_energy": 0}
    if opp_active is None:
        return out
    try:
        attached = energies_of(opp_active)
        out["opponent_attached_energy"] = len(attached)
        # 手貼り1回を想定した仮のエネルギープール(色は相手のカードの型に合わせる)。
        try:
            extra = int(_card(opp_active.id).energyType)
        except Exception:  # noqa: BLE001
            extra = 0
        next_turn_pool = attached + [extra]
        c = _card(opp_active.id)
        for aid in (c.attacks or []):
            a = card_cache.get_attack(aid)
            dmg, variable = attack_damage_estimate(opp_active, my_active, a)
            out["any_variable_damage"] = out["any_variable_damage"] or variable
            out["theoretical_max_damage"] = max(out["theoretical_max_damage"], dmg)
            cost = list(a.energies or [])
            if _satisfies(cost, attached):
                out["current_payable_damage"] = max(out["current_payable_damage"], dmg)
            if _satisfies(cost, next_turn_pool):
                out["credible_next_turn_damage"] = max(out["credible_next_turn_damage"], dmg)
    except Exception:  # noqa: BLE001
        return out
    return out


def ko_risk(active, opp_active, cfg: dict) -> tuple[str, dict]:
    """次の相手ターンにバトル場が倒される危険。段階値と根拠を返す。

    公開情報(相手の場のエネルギー)で「今払える技」を判定する。カード記載の最大打点を
    常に撃てる前提にすると不要な交代が増えるため、段階を分ける。

      CERTAIN_KO  : 今の公開情報で攻撃コストを満たし、KO できる
      LIKELY_KO   : 手貼り1回など通常の加速で KO 可能になる
      POSSIBLE_KO : 追加のカードや条件が必要(理論最大打点なら届く)
      SAFE        : 現実的な KO 手段を確認できない
    """
    reason: dict = {}
    if active is None:
        return SAFE, reason
    try:
        hp = int(active.hp)
        tiers = damage_tiers(opp_active, active, cfg)
        reason = {"remaining_hp": hp, **tiers}
        if opp_active is None:
            return SAFE, reason
        if tiers["current_payable_damage"] >= hp:
            reason["basis"] = "current_payable_damage >= remaining_hp"
            return CERTAIN_KO, reason
        if tiers["credible_next_turn_damage"] >= hp:
            reason["basis"] = "credible_next_turn_damage >= remaining_hp"
            return LIKELY_KO, reason
        if tiers["theoretical_max_damage"] >= hp:
            reason["basis"] = "theoretical_max_damage >= remaining_hp"
            return POSSIBLE_KO, reason
        reason["basis"] = "no known attack reaches remaining_hp"
        return SAFE, reason
    except Exception:  # noqa: BLE001
        return SAFE, reason


def intended_retreat_target(player):
    """MAIN で「にげる」を選んだ場合に、後続の SWITCH で選ぶことになる交代先の想定。

    MAIN の RETREAT 選択肢は行き先を持たない(交代先は後続の ``CARD/SWITCH`` で決まる)。
    その SWITCH も同じプランナーが制御するので、ここでは「プランナーが選ぶであろう
    ベンチ個体」を想定先として使う:

      1. 直ちに攻撃できる非ex(=カプ・ブルル中継の本命)
      2. 直ちに攻撃できる ex(次のアタッカー)
      3. それも無ければ None(交代しても攻撃できないので補正しない)
    """
    try:
        bench = [b for b in (player.bench or []) if b is not None]
        ready_non_ex = [b for b in bench if not is_ex(b) and can_attack_now(b)]
        if ready_non_ex:
            return ready_non_ex[0]
        ready_ex = [b for b in bench if is_ex(b) and can_attack_now(b)]
        if ready_ex:
            # HP の高い個体(=削れていない新しいオーガポン)を優先する。
            return max(ready_ex, key=lambda b: (b.hp or 0))
        return None
    except Exception:  # noqa: BLE001
        return None


def _retreat_target_of(option, obs, me):
    """RETREAT 候補が指す交代先。MAIN では行き先を持たないので想定先を使う。"""
    target = _resolve_option_pokemon(obs, option, me)
    if target is not None:
        return target
    try:
        return intended_retreat_target(obs.current.players[me])
    except Exception:  # noqa: BLE001
        return None


def main_candidate_bonus(obs: Observation, me: int, candidate_indices: list[int],
                         config: dict | None, deck_ids) -> dict[int, float]:
    """MAIN の各候補に足す補助スコア(PIMC の勝ち見込みと同じ 0..1 スケール想定)。

    「にげる」と「現在のポケモンで続行(攻撃・END 等)」を **同じ scored の上で** 比較する。
    したがって PIMC の判断を捨てず、リソース面の考慮だけを上乗せする形になる。
    """
    out: dict[int, float] = {}
    breakdown: dict[int, dict] = {}
    LAST_BREAKDOWN.clear()
    try:
        if not is_active(config, deck_ids):
            return out
        cfg = main_config(config)
        if not cfg.get("main_enabled"):
            return out
        select = obs.select
        state = obs.current
        if select is None or state is None:
            return out
        mine = state.players[me]
        opp = state.players[1 - me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        opp_active = next((s for s in (opp.active or []) if s is not None), None)
        if active is None:
            return out

        risk, _reason = ko_risk(active, opp_active, cfg)
        risk_weight = {SAFE: 0.0, POSSIBLE_KO: 0.4, LIKELY_KO: 0.8, CERTAIN_KO: 1.0}[risk]
        bench = [s for s in (mine.bench or []) if s is not None]
        # カプ・ブルル戦闘中に次のオーガポンを準備できるか
        next_attacker_ready = any(is_ex(b) and can_attack_now(b) for b in bench)

        for i in candidate_indices:
            if i >= len(select.option):
                continue
            opt = select.option[i]
            otype = int(getattr(opt, "type", -1))
            if otype != int(OptionType.RETREAT):
                continue  # 「続行」側は素の PIMC スコアのまま(基準点)

            target = _retreat_target_of(opt, obs, me)
            if target is None:
                continue

            bonus = 0.0
            # (1) 削れた ex を逃がして守れる期待サイド損失
            bonus += cfg["w_expected_prize_loss"] * risk_weight * prize_value(active)
            # (2) カプ・ブルルを挟むことで増える相手の必要KO回数(実差)
            # 行動条件付きの差(stay_line vs retreat_line)。ベンチにいるだけでは 0。
            ko_delta = required_ko_delta_for_retreat(
                mine, active, target, len(opp.prize or []))
            bonus += cfg["w_required_ko_increase"] * ko_delta
            # (3) 交代後に攻撃できないテンポ損失
            if not can_attack_now(target):
                bonus -= cfg["w_attack_tempo_loss"]
            # (4) にげるために失うエネルギー
            bonus -= cfg["w_retreat_energy_loss"] * retreat_cost(active)
            # (5) ベンチへ逃がした ex がボス等で狙われる危険
            if is_ex(active):
                bonus -= cfg["w_boss_target_risk"] * risk_weight
            # (6) 次のアタッカーが準備できているなら中継の価値が上がる
            if next_attacker_ready and not is_ex(target):
                bonus += cfg["w_next_attacker_ready"]
            out[i] = bonus
            breakdown[i] = {
                "expected_prize_loss": cfg["w_expected_prize_loss"] * risk_weight * prize_value(active),
                "required_ko_increase": cfg["w_required_ko_increase"] * ko_delta,
                "attack_tempo_loss": (0.0 if can_attack_now(target) else -cfg["w_attack_tempo_loss"]),
                "retreat_energy_loss": -cfg["w_retreat_energy_loss"] * retreat_cost(active),
                "boss_target_risk": (-cfg["w_boss_target_risk"] * risk_weight if is_ex(active) else 0.0),
                "next_attacker_ready": (cfg["w_next_attacker_ready"]
                                        if (next_attacker_ready and not is_ex(target)) else 0.0),
                "ko_risk": risk, "ko_delta": ko_delta,
                "target_can_attack": can_attack_now(target),
                "target_is_ex": is_ex(target),
            }
    except Exception:  # noqa: BLE001
        return {}
    LAST_BREAKDOWN.update(breakdown)
    return out


def required_ko_delta_for_retreat(player, active, retreat_target,
                                  opponent_prize_remaining: int) -> float:
    """**行動条件付き**の必要KO回数の差。

    「ベンチにカプ・ブルルがいるか」ではなく「次に倒される対象を入れ替えること」の効果を測る。

        stay_line   : 削れた active(ex なら2枚)が次に倒される
                      -> その後、相手が勝つまでに必要な KO 回数
        retreat_line: 交代した retreat_target(1枚なら1枚)が次に倒される
                      -> その後、相手が勝つまでに必要な KO 回数(active はベンチへ回る)

        delta = required_KOs(retreat_line) - required_KOs(stay_line)

    正なら「交代すると相手はより多くの KO を要する」。ベンチにいるだけでは 0 になる。

    サイドは**倒した側が自分の山から取る**ので、基準になるのは相手の残サイド枚数
    (``opponent_prize_remaining``)であって自分の残サイドではない。
    """
    try:
        if active is None or retreat_target is None:
            return 0.0
        remaining = int(opponent_prize_remaining)
        if remaining <= 0:
            return 0.0
        on_field = [s for s in (list(player.active or []) + list(player.bench or []))
                    if s is not None]
        a_ser = getattr(active, "serial", None)
        t_ser = getattr(retreat_target, "serial", None)
        others = [prize_value(p) for p in on_field
                  if getattr(p, "serial", None) not in (a_ser, t_ser)]

        # stay: active が最初に倒れる。その後は残り(retreat_target 含む)から取られる。
        stay = 1.0 + _kos_needed_for_values(
            max(0, remaining - prize_value(active)),
            others + [prize_value(retreat_target)])
        # retreat: retreat_target が最初に倒れる。active はベンチへ退避して残る。
        retreat = 1.0 + _kos_needed_for_values(
            max(0, remaining - prize_value(retreat_target)),
            others + [prize_value(active)])
        return retreat - stay
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# MAIN/OptionType.ATTACH(通常の手貼り)への候補注入と評価
# ---------------------------------------------------------------------------
#
# 09:00 JST 訂正: SelectType.MAIN 内の OptionType.ATTACH が通常の手貼りで、
# 1つの option に「付けるカード(area/index)」と「付与先ポケモン
# (inPlayArea/inPlayIndex)」の両方が入っている。ATTACH_FROM/ATTACH_TO(SelectType.CARD)
# とは別物(調査: investigate_main_attach.py、85/85件でATTACH logと一致確認済み)。

ATTACH_DEFAULTS: dict = {
    "attach_enabled": False,
    "attach_shadow_only": False,
    "attach_bonus_cap": 0.08,
    "w_backup_progress": 0.02,
    "w_completion_bonus": 0.05,
    "w_required_ko_increase": 0.03,
    "w_current_active_preservation": 0.01,
    "w_attack_tempo_loss": 0.06,
    "w_energy_stranding_risk": 0.02,
    "w_boss_target_risk": 0.01,
    "w_match_end_risk": 0.02,
    "require_active_ready_for_positive": True,
}


def attach_config(config: dict | None) -> dict:
    cfg = dict(ATTACH_DEFAULTS)
    user = (config or {}).get("ogerpon_planner") or {}
    if isinstance(user, dict):
        cfg.update({k: v for k, v in user.items() if k in ATTACH_DEFAULTS})
    return cfg


def _energy_stage_bonus(shortfall_before: int, cfg: dict) -> float:
    if shortfall_before <= 0:
        return 0.0
    stage = {4: 0.4, 3: 0.6, 2: 0.8, 1: 1.2}.get(shortfall_before, 0.3)
    bonus = cfg["w_backup_progress"] * stage
    if shortfall_before == 1:
        bonus += cfg["w_completion_bonus"]
    return bonus


def _has_existing_investment(pokemon) -> bool:
    return len(energies_of(pokemon)) > 0


def _resolve_attach_target(state, me, option):
    try:
        in_play_area = getattr(option, "inPlayArea", None)
        in_play_index = getattr(option, "inPlayIndex", None)
        if in_play_area is None or in_play_index is None:
            return None
        player = state.players[me]
        a = int(in_play_area)
        i = int(in_play_index)
        if a == int(AreaType.ACTIVE):
            slots = list(player.active or [])
        elif a == int(AreaType.BENCH):
            slots = list(player.bench or [])
        else:
            return None
        return slots[i] if 0 <= i < len(slots) else None
    except Exception:  # noqa: BLE001
        return None


def select_strategic_attach_candidate(obs: Observation, me: int, config: dict | None,
                                      deck_ids) -> int | None:
    """MAIN の raw option の中から、控えアタッカー完成に最も価値のある ATTACH 候補を1件選ぶ。

    優先順位: 1.あと1枚で攻撃可能 2.あと2枚 3.既に投資済み 4.最大打点が高い
    """
    try:
        acfg = attach_config(config)
        if not acfg.get("attach_enabled"):
            return None
        if not is_active(config, deck_ids):
            return None
        select = obs.select
        state = obs.current
        if select is None or state is None or int(select.type) != int(SelectType.MAIN):
            return None

        best_idx, best_key = None, None
        for i, o in enumerate(select.option):
            if int(getattr(o, "type", -1)) != int(OptionType.ATTACH):
                continue
            target = _resolve_attach_target(state, me, o)
            if target is None or is_ex(target):
                continue
            shortfall = energy_shortfall(target)
            if shortfall <= 0:
                continue
            dmg = best_damage(target) or 1
            key = (shortfall, -1 if _has_existing_investment(target) else 0, -dmg)
            if best_key is None or key < best_key:
                best_key, best_idx = key, i
        return best_idx
    except Exception:  # noqa: BLE001
        return None


def attach_candidate_bonus(obs: Observation, me: int, candidate_indices: list[int],
                           config: dict | None, deck_ids) -> dict[int, float]:
    """MAIN の ATTACH 候補に足す補助スコア(PIMC の 0..1 スケール想定、上限あり)。"""
    out: dict[int, float] = {}
    try:
        acfg = attach_config(config)
        if not acfg.get("attach_enabled"):
            return out
        if not is_active(config, deck_ids):
            return out
        select = obs.select
        state = obs.current
        if select is None or state is None:
            return out
        mine = state.players[me]
        opp = state.players[1 - me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        active_ready_now = can_attack_now(active) if active is not None else False
        opp_active = next((s for s in (opp.active or []) if s is not None), None)
        opp_prize = len(opp.prize or [])
        cap = float(acfg["attach_bonus_cap"])

        for i in candidate_indices:
            if i >= len(select.option):
                continue
            o = select.option[i]
            if int(getattr(o, "type", -1)) != int(OptionType.ATTACH):
                continue
            target = _resolve_attach_target(state, me, o)
            if target is None:
                continue
            is_active_slot = (active is not None
                              and getattr(target, "serial", None) == getattr(active, "serial", None))

            bonus = 0.0
            if is_active_slot:
                bonus += acfg["w_current_active_preservation"]
                out[i] = bonus
                continue
            if is_ex(target):
                continue

            shortfall_before = energy_shortfall(target)
            if shortfall_before <= 0:
                continue

            bonus += _energy_stage_bonus(shortfall_before, acfg)
            ko_gain = required_ko_delta(mine, target)
            bonus += acfg["w_required_ko_increase"] * ko_gain

            if not active_ready_now:
                bonus -= acfg["w_attack_tempo_loss"]
                if acfg.get("require_active_ready_for_positive"):
                    bonus = min(bonus, 0.0)

            opp_dmg_tiers = damage_tiers(opp_active, target, main_config(config))
            if opp_dmg_tiers["current_payable_damage"] >= (target.hp or 0):
                bonus -= acfg["w_energy_stranding_risk"]

            bench_damaged_ex = [b for b in (mine.bench or [])
                                if b is not None and is_ex(b) and b.maxHp
                                and b.hp < b.maxHp * 0.5]
            if bench_damaged_ex:
                bonus -= acfg["w_boss_target_risk"]

            if opp_prize <= max(1, shortfall_before):
                bonus -= acfg["w_match_end_risk"]

            bonus = max(-cap, min(cap, bonus))
            out[i] = bonus
    except Exception:  # noqa: BLE001
        return {}
    return out


# ---------------------------------------------------------------------------
# finish_only: 0エネから無理に育てず、2/3エネまで育ったブルルだけを完成させる
# ---------------------------------------------------------------------------
#
# 上の attach_candidate_bonus / select_strategic_attach_candidate / score_adjustments
# (交代側)とは完全に独立したモジュール。ogerpon_planner.enabled(過去の全補正の
# 共通ゲート)には一切依存しない専用の config キー "ogerpon_finish_only" を使うことで、
# 「finish_only 単体の効果」を他の補正から切り離して測れるようにする。
#
# 方針: ブルルの現在のエネルギー枚数(0/1/2/3/4+)だけで扱いを変える。
#   0,1エネ : 何もしない(候補注入もbonusも無し)。0エネから無理に育てない。
#   2エネ   : 手貼り候補を最大1件注入。小さい正のbonus(上限 cap_tier2)。
#   3エネ   : 4エネ完成候補を最大1件注入。強めの正のbonus(上限 cap_tier3)。
#   4エネ以上: 追加の手貼りを優先しない。bonus は 0 以下。

FINISH_ONLY_DEFAULTS: dict = {
    "enabled": False,
    "shadow_only": False,
    "cap_tier2": 0.03,
    "cap_tier3": 0.08,
    "w_tier2": 0.02,
    "w_tier3": 0.06,
    "w_required_ko_increase": 0.02,
    "switch_bonus": 0.05,
}


def finish_only_config(config: dict | None) -> dict:
    cfg = dict(FINISH_ONLY_DEFAULTS)
    user = (config or {}).get("ogerpon_finish_only") or {}
    if isinstance(user, dict):
        cfg.update({k: v for k, v in user.items() if k in FINISH_ONLY_DEFAULTS})
    return cfg


def _attach_source_energy_type(option) -> int | None:
    """ATTACH option が付けようとしているカードの EnergyType。判定不能なら None。"""
    try:
        card_id = getattr(option, "cardId", None)
        if card_id is None:
            return None
        cd = _card(card_id)
        if int(cd.cardType) not in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY)):
            return None
        et = getattr(cd, "energyType", None)
        return int(et) if et is not None else None
    except Exception:  # noqa: BLE001
        return None


def _true_shortfall(pokemon, attached: list[int]) -> int:
    """色指定まで考慮した実際の不足エネ数(``energy_shortfall`` は総数のみで色を見ない)。"""
    costs = attack_costs(pokemon)
    if not costs:
        return 99
    best = None
    for cost in costs:
        pool = list(attached)
        colored_needed = 0
        for need in cost:
            if need == 0:
                continue
            if need in pool:
                pool.remove(need)
            else:
                colored_needed += 1
        colorless_needed = max(0, sum(1 for c in cost if c == 0) - len(pool))
        total = colored_needed + colorless_needed
        if best is None or total < best:
            best = total
    return best if best is not None else 99


def _attach_is_productive(target, energy_type: int | None) -> bool:
    """安全条件(c): このエネルギーは実際に ``target`` の攻撃コストへ効くか(色込みで判定)。

    判定できない(カード情報が引けない)ときは安全側に倒して False。
    """
    if energy_type is None:
        return False
    attached = energies_of(target)
    before = _true_shortfall(target, attached)
    after = _true_shortfall(target, attached + [energy_type])
    return after < before


def finish_only_select_attach_candidate(obs: Observation, me: int, config: dict | None,
                                        deck_ids) -> int | None:
    """MAIN の raw option から、2エネ/3エネのブルルへ効く手貼りを最大1件だけ選ぶ。"""
    try:
        cfg = finish_only_config(config)
        if not cfg.get("enabled") or not deck_is_ogerpon(deck_ids):
            return None
        select = obs.select
        state = obs.current
        if select is None or state is None or int(select.type) != int(SelectType.MAIN):
            return None
        mine = state.players[me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        if not can_attack_now(active):
            return None  # 安全条件(a): 今の攻撃可能性を崩してまで注入しない

        best_idx, best_key = None, None
        for i, o in enumerate(select.option):
            if int(getattr(o, "type", -1)) != int(OptionType.ATTACH):
                continue
            target = _resolve_attach_target(state, me, o)
            if target is None or is_ex(target):
                continue
            if active is not None and getattr(target, "serial", None) == getattr(active, "serial", None):
                continue  # 安全条件(d): 対象はベンチの控えのみ(バトル場は触らない)
            n_before = len(energies_of(target))
            if n_before not in (2, 3):
                continue  # 0,1エネは注入しない。4エネ以上は優先しない。
            et = _attach_source_energy_type(o)
            if not _attach_is_productive(target, et):
                continue  # 安全条件(c): コストに効かない色は注入しない
            key = (0 if n_before == 3 else 1, -best_damage(target))
            if best_key is None or key < best_key:
                best_key, best_idx = key, i
        return best_idx
    except Exception:  # noqa: BLE001
        return None


def finish_only_attach_bonus(obs: Observation, me: int, candidate_indices: list[int],
                             config: dict | None, deck_ids) -> dict[int, float]:
    """MAIN の ATTACH 候補に足す finish_only 専用の補助スコア(エネ枚数の段階で上限を分ける)。"""
    out: dict[int, float] = {}
    try:
        cfg = finish_only_config(config)
        if not cfg.get("enabled") or not deck_is_ogerpon(deck_ids):
            return out
        select = obs.select
        state = obs.current
        if select is None or state is None:
            return out
        mine = state.players[me]
        active = next((s for s in (mine.active or []) if s is not None), None)
        active_ready = can_attack_now(active) if active is not None else False

        for i in candidate_indices:
            if i >= len(select.option):
                continue
            o = select.option[i]
            if int(getattr(o, "type", -1)) != int(OptionType.ATTACH):
                continue
            target = _resolve_attach_target(state, me, o)
            if target is None or is_ex(target):
                continue
            if active is not None and getattr(target, "serial", None) == getattr(active, "serial", None):
                continue  # 安全条件(d): バトル場自身への手貼りには関与しない
            n_before = len(energies_of(target))
            if n_before >= 4 or n_before in (0, 1):
                continue  # 4エネ以上は優先しない(bonus 0)。0,1エネは無理に育てない(bonus 0)。
            if not active_ready:
                continue  # 安全条件(a)
            et = _attach_source_energy_type(o)
            if not _attach_is_productive(target, et):
                continue  # 安全条件(c)

            if n_before == 2:
                bonus = float(cfg["w_tier2"])
                cap = float(cfg["cap_tier2"])
            else:  # n_before == 3 (=4エネ完成候補)
                bonus = float(cfg["w_tier3"])
                cap = float(cfg["cap_tier3"])
            bonus += float(cfg["w_required_ko_increase"]) * required_ko_delta(mine, target)
            out[i] = max(0.0, min(cap, bonus))
    except Exception:  # noqa: BLE001
        return {}
    return out


def finish_only_switch_adjustments(obs: Observation, me: int, config: dict | None,
                                   deck_ids) -> list[float] | None:
    """CARD/TO_ACTIVE,SWITCH: ブルルが「4エネ完成 かつ 支払い可能 かつ 交代後に攻撃可能」の
    ときだけ小さく後押しする。攻撃不能な壁交代のbonusは無い(=旧 promote_not_ready系はOFF)。
    """
    try:
        cfg = finish_only_config(config)
        if not cfg.get("enabled") or not deck_is_ogerpon(deck_ids):
            return None
        select: SelectData | None = obs.select
        state = obs.current
        if select is None or state is None or not select.option:
            return None
        if select.maxCount != 1:
            return None
        if int(select.type) != int(SelectType.CARD):
            return None
        if int(select.context) not in (int(SelectContext.TO_ACTIVE), int(SelectContext.SWITCH)):
            return None

        n = len(select.option)
        adj = [0.0] * n
        for i, opt in enumerate(select.option):
            target = _resolve_option_pokemon(obs, opt, me)
            if target is None or is_ex(target):
                continue
            # can_attack_now は実際の色込みコストで判定するので、4エネ完成・支払い可能・
            # (交代はエネルギーを変えないので)交代後に攻撃可能、の3条件を同時に満たす。
            if can_attack_now(target):
                adj[i] += float(cfg["switch_bonus"])
        return adj
    except Exception:  # noqa: BLE001
        return None
