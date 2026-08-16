#!/usr/bin/env python3
"""ガードの**発火率・誤爆率・結果メトリクス**を測る計測ハーネス(採用判定の主軸)。

なぜ勝率A/Bではないのか
------------------------
r9/r10 で入れているガード群は、実ラダー31敗中1〜3件という頻度の修正が中心で、勝率A/Bの
ノイズ床(同一エージェント同士でも n=420 で ±7pt、対面別 ±10pt。
`project_local_ab_noise_floor`)では**原理的に判定できない**。よって採用判定は

  (1) 実局面スナップショットで正しい手を選ぶか(`_snapshot_gate_r10.py` / `_snapshot_gate_r8.py`)
  (2) 発火率・誤爆率メトリクス(このスクリプト)

で行い、勝率は「-3pt級の退行が無いこと」の確認にだけ使う(参考値として一緒に出す)。

測るもの
--------
* **発火率**: `ml_policy_agent.get_guard_stats()` の各カウンタを *1 decision ごとに差分取得*
  して、どの decision でどのガードが発火したかを特定する(自陣の decision の前後でだけ
  差分を取るので、相手側エージェントの発火は混ざらない)。
* **誤爆率**: 発火した decision の観測を**ガードの実装とは独立に**再評価する。
    - tool_stadium_guard        : 無効スタジアムでないのに発火した回数(=0であるべき)
    - energy_to_active_first    : アクティブが既に攻撃コストを払えるのに発火した回数(=0)
                                  判定は engine の ATTACK 選択肢の有無**と**カードデータ
                                  (attack.energies vs 付いているエネ)の両方で二重に見る
    - search_pick_pokemon_first : ベンチ満杯なのに発火した回数(=0)
    - low_deck_draw_brake       : 山札が閾値超なのに発火した回数(=0)、
                                  KO可能ターンに発火した回数(=0、`--ko-audit` 時のみ)
    - boss_lethal_gate          : KO可能な対象があるのにボスを却下した回数
                                  (=0、`--ko-audit` 時のみ)
  未実装のガード(config にキーが無い/カウンタが無い)は自動的にスキップする。
* **結果メトリクス**(勝率より鋭敏): 山札切れ敗率 / ベンチ0(アクティブ不在)敗率 /
  0ダメージ攻撃回数 per game / 無効どうぐ装着回数 per game / 攻撃実施率
  (自分の MAIN があったターンのうち、実際に攻撃したターンの割合)。

使い方
------
    # 2アームを同一 seed 列(paired)で比較
    python kaggle_replays/_metrics_gate.py --config abl_5_full_og_r7 abl_5_full_og_r10 \\
        --games 60 --workers 12 --opponents mix --out kaggle_replays/_metrics_gate_r10.json

    # 特定対面だけ(どうぐ無効スタジアムを持つのは dragapult_ex だけ、等)
    python kaggle_replays/_metrics_gate.py --config abl_5_full_og_r10 --games 40 \\
        --opponents dragapult_ex

前提
----
* 自陣デッキ/重みは og 系 A/B の既定(`g2top2_v032.csv` + `..._rl_mixogerpon.json`)。
  `--own-deck` / `--own-weights` で差し替え可能。
* 相手は `meta_analysis/archetype_decks_g2/<arch>/01.csv` + `policy_weights_<arch>_g2.json` +
  `abl_5_full`(=ガードキーを持たない config)。相手側でガードが発火しない構成にしてある。
* `league/run_match.py` は変更せず、per-decision の計測が要るぶんだけ同等のループを持つ
  (`_begin_match_state` と `MAX_STEPS` は run_match から import して二重定義を避ける)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SUB = _ROOT / "sample_submission"
for _p in (str(_ROOT), str(_SUB), str(_ROOT / "league")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402
import run_match  # noqa: E402

from cg.api import (  # noqa: E402
    AreaType,
    EnergyType,
    LogType,
    OptionType,
    SelectType,
    to_observation_class,
)
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

_WDIR = _SUB / "ptcg_ai" / "learning"
_DECKDIR_G2 = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks_g2"
_DECKDIR_J = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"

DEFAULT_OWN_DECK = _ROOT / "kaggle_replays" / "deck_search" / "candidates_ogerpon_stage3" / "g2top2_v032.csv"
DEFAULT_OWN_WEIGHTS = _WDIR / "policy_weights_ogerpon_teal_ex_rl_mixogerpon.json"

# mix フィールド(gen2)。重みは og_r8 の field A/B(`_og_r8_field_ab.py`)の試合数配分を
# そのまま整数比にしたもの。`--opponents mix` のとき、この比で --games を割り振る。
#
# 注意(このハーネスで分かったこと): **どうぐ無効スタジアム(ジャミングタワー1246)を積んで
# いるのは dragapult_ex だけ**なので、`tool_stadium_guard` の発火はこの対面でしか観測できない。
# 発火率を測りたいときは `--opponents dragapult_ex` で焦点を絞ること。
MIX_FIELD: list[tuple[str, int, str]] = [
    ("alakazam", 4, "_g2"),
    ("mega_froslass_ex", 4, "_g2"),
    ("lopunny_megafroslass", 4, "_g2"),
    ("dragapult_ex", 3, "_g2"),   # 唯一ジャミングタワーを積む対面
    ("yadoking", 2, "_g2"),
    ("marnie_grimmsnarl_ex", 2, "_g2"),
    ("mega_lucario_ex", 2, ""),
    ("ogerpon_teal_ex", 1, "_g2"),  # ミラー
    ("archaludon_ex", 1, "_g2"),
    ("crustle", 1, "_g2"),
]

# 「効果を無効化するスタジアム」= 結果メトリクス(無効どうぐ装着回数)の判定に使う。
# 意思決定側の既定(`ml_policy_agent._DEFAULT_NULLIFYING_STADIUM_IDS`)と同じ値だが、
# 計測はガード実装から独立させたいので**ここで別に持つ**(実装が壊れても計測は壊れない)。
NULLIFYING_STADIUM_IDS = frozenset({1246})

# 「自分自身に基本エネを付ける特性」を持つポケモン(オーガポン みどりのめん ex のみどりのまい)。
# ガード側(`ml_policy_agent._TEAL_DANCE_CARD_IDS`)と同じ値をあえて別に持つ(計測の独立性)。
ENERGY_ABILITY_CARD_IDS = frozenset({96})

# Fix-J: テラスタル退避(`terastal_rotation`)計測用。ガード側
# (`ml_policy_agent._DEFAULT_SKIP_BENCH_DAMAGE_IDS`)と同じ値をあえて別に持つ(計測の独立性)。
# ドラパルトex(121): ファントムダイブがダメージカウンタをベンチへ直接置く=テラスタルを貫通する。
TERA_BENCH_DAMAGE_ARCHETYPE_IDS = frozenset({121})

# 誤爆判定に使う「発火カウンタ -> (ガード名)」。ここに無いカウンタは発火数のみ集計する。
FIRE_COUNTERS = {
    "tool_stadium_guard_fired": "tool_stadium_guard",
    "energy_to_active_first_fired": "energy_to_active_first",
    "search_pick_pokemon_first_fired": "search_pick_pokemon_first",
    "low_deck_draw_brake_fired": "low_deck_draw_brake",
    "boss_lethal_gate_fired": "boss_lethal_gate",
    "boss_lethal_gate_redirected": "boss_lethal_gate",
    "briar_gate_fired": "briar_gate",
    "hand_damage_guard_fired": "hand_damage_guard",
    # Fix-I: 「みどりのまい」型の特性ドロー抑制(`ability_draw_brake`)。誤爆判定は
    # 「発火した decision で ATTACK 選択肢が無かった(=絶対例外に違反した)回数」(=0のはず)。
    "ability_draw_brake_fired": "ability_draw_brake",
    # Fix-J: テラスタル退避(`terastal_rotation`)。誤爆判定は「発火した decision で相手の場に
    # ドラパルトex等が居た(=スキップすべきだった)回数」(=0のはず)。
    "terastal_rotation_fired": "terastal_rotation",
}


# ---------------------------------------------------------------------------
# 観測からの独立判定(ガード実装を参照しない)
# ---------------------------------------------------------------------------

def _current_stadium_id(state) -> int | None:
    try:
        stadium = state.stadium or []
        return int(stadium[0].id) if stadium and stadium[0] is not None else None
    except Exception:  # noqa: BLE001
        return None


def _card_type(card_id: int | None):
    if card_id is None:
        return None
    try:
        from ptcg_ai.shared import card_cache
        return card_cache.get_card(int(card_id)).cardType
    except Exception:  # noqa: BLE001
        return None


def _resolve_option_card_id(option, state) -> int | None:
    from ptcg_ai.learning import encoder as _enc
    try:
        return _enc._resolve_card_id(option, state)
    except Exception:  # noqa: BLE001
        return None


def _is_tool_play(option, state) -> bool:
    from cg.api import CardType
    if option.type not in (OptionType.ATTACH, OptionType.PLAY):
        return False
    return _card_type(_resolve_option_card_id(option, state)) == CardType.TOOL


_WILD_ENERGY = {int(EnergyType.RAINBOW), int(EnergyType.TEAM_ROCKET)}


def _can_pay_cost(cost: list[int], have: list[int]) -> bool:
    """`cost`(必要エネの色配列)を `have`(付いているエネの色配列)で払えるか。

    無色(COLORLESS=0)は何でも充当でき、レインボー/ロケット団エネは任意色として扱う
    (「払える」側に倒す = 誤爆判定を厳しくする方向で安全)。
    """
    pool = list(have)
    for need in cost:
        if int(need) == int(EnergyType.COLORLESS):
            continue
        for i, got in enumerate(pool):
            if int(got) == int(need) or int(got) in _WILD_ENERGY:
                pool.pop(i)
                break
        else:
            return False
    colorless = sum(1 for c in cost if int(c) == int(EnergyType.COLORLESS))
    return len(pool) >= colorless


# スタジアムによるワザコストの追加(独立判定を実盤面に合わせるための補正表)。
# 夜の鉱山(1266): 「おたがいの場の『テラスタル』のポケモン全員は、ワザを使うためのエネルギーが、
# それぞれ【無】エネルギー1個ぶん多くなる」。初回計測(60試合)で
# `energy_to_active_first.active_can_pay_attack_cost` が3件立ったのは全てこの効果の見落としで、
# 実際には engine 側が正しく ATTACK 選択肢を出していなかった(=ガードの誤爆ではなく計測側の
# 過検出)。オーガポン みどりのめん ex は tera=True なので草3では攻撃できない。
_STADIUM_COST_SURCHARGE = {1266: "tera_plus_one_colorless"}


def _attack_cost_surcharge(state, mon) -> int:
    """スタジアム効果で増える【無】エネルギーの個数。該当なしは 0。"""
    rule = _STADIUM_COST_SURCHARGE.get(_current_stadium_id(state))
    if rule != "tera_plus_one_colorless":
        return 0
    try:
        from ptcg_ai.shared import card_cache
        return 1 if bool(card_cache.get_card(int(mon.id)).tera) else 0
    except Exception:  # noqa: BLE001
        return 0


def _active_can_pay_any_attack(state, me: int) -> bool | None:
    """自分のバトルポケモンが**いずれかのワザのエネルギーコストを払えるか**。判定不能は None。

    engine の ATTACK 選択肢とは独立の経路(カードデータ)で計算する誤爆判定用の指標。
    特殊状態(ねむり/マヒ)やロック効果までは見ないので「払える」と出ても攻撃できないことは
    あるが、`energy_to_active_first` の誤爆(=攻撃できるのに発火)を**過剰に**検出する方向
    なので安全側。スタジアムによるコスト増(`_attack_cost_surcharge`)だけは、実測で過検出の
    主因だったので補正する。
    """
    try:
        from ptcg_ai.shared import card_cache
        active = state.players[me].active or []
        mon = active[0] if active else None
        if mon is None:
            return None
        card = card_cache.get_card(int(mon.id))
        have = [int(e) for e in (mon.energies or [])]
        surcharge = [int(EnergyType.COLORLESS)] * _attack_cost_surcharge(state, mon)
        for attack_id in (card.attacks or []):
            cost = [int(e) for e in card_cache.get_attack(attack_id).energies] + surcharge
            if _can_pay_cost(cost, have):
                return True
        return False
    except Exception:  # noqa: BLE001
        return None


def _misfire_diagnostics(obs) -> dict:
    """誤爆フラグが立った decision の生データ(後追い調査用。原因の言い訳を作らないため)。"""
    state = obs.current
    me = state.yourIndex
    try:
        active = state.players[me].active or []
        mon = active[0] if active else None
        return {
            "stadium": _current_stadium_id(state),
            "active_id": None if mon is None else int(mon.id),
            "active_energies": [] if mon is None else [int(e) for e in (mon.energies or [])],
            "deck_count": int(state.players[me].deckCount or 0),
            "asleep": bool(state.players[me].asleep),
            "paralyzed": bool(state.players[me].paralyzed),
            "bench": len(state.players[me].bench or []),
            "bench_max": int(state.players[me].benchMax or 0),
        }
    except Exception:  # noqa: BLE001
        return {}


def _bench_full(state, me: int) -> bool | None:
    try:
        player = state.players[me]
        return len(player.bench or []) >= int(player.benchMax or 0)
    except Exception:  # noqa: BLE001
        return None


def _energy_attach_area(option, state):
    """「エネルギーを自分の場のポケモンに付ける手」なら付け先の AreaType、違えば None。

    `ml_policy_agent._energy_attach_target_area` と同じ判定だが、計測をガード実装から
    独立させるため**意図的に別実装**にしている(実装が壊れたら計測が同じように壊れる、
    という事故を避ける)。
    """
    from cg.api import CardType
    try:
        if option.type == OptionType.ATTACH:
            if _card_type(_resolve_option_card_id(option, state)) not in (
                    CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
                return None
            return getattr(option, "inPlayArea", None)
        if option.type == OptionType.ABILITY:
            card_id = _resolve_option_card_id(option, state)
            if card_id is None or int(card_id) not in ENERGY_ABILITY_CARD_IDS:
                return None
            return getattr(option, "area", None)
    except Exception:  # noqa: BLE001
        return None
    return None


def _looking_id(option, state) -> int | None:
    """`area == LOOKING` の選択肢が指すカードID(`state.looking[index]`)。判定不能は None。"""
    try:
        if getattr(option, "area", None) != AreaType.LOOKING:
            return None
        looking = getattr(state, "looking", None)
        index = getattr(option, "index", None)
        if not looking or index is None or not (0 <= index < len(looking)):
            return None
        card = looking[index]
        return int(card.id) if card is not None else None
    except Exception:  # noqa: BLE001
        return None


def _is_basic_pokemon_id(card_id: int | None) -> bool:
    from cg.api import CardType
    if card_id is None:
        return False
    try:
        from ptcg_ai.shared import card_cache
        card = card_cache.get_card(int(card_id))
        return card.cardType == CardType.POKEMON and bool(card.basic)
    except Exception:  # noqa: BLE001
        return False


def _measure_decision(rec: dict, obs, action: list[int]) -> None:
    """**最終的に選ばれた手**を見て「修正対象のバグがまだ起きているか」を数える。

    ガードの発火数(治した回数)とは別に、**残っているバグの回数**を測るのが目的。ガードON側で
    0 に近づき、OFF側で正の値になることが、そのガードが狙い通りに効いている証拠になる
    (勝率より鋭敏で、しかも n が小さくても読める)。
    """
    state, select = obs.current, obs.select
    me = state.yourIndex
    chosen = [i for i in action if 0 <= i < len(select.option)]

    if select.type == SelectType.MAIN:
        has_attack = any(o.type == OptionType.ATTACK for o in select.option)
        active_targets = any(
            _energy_attach_area(o, state) == AreaType.ACTIVE for o in select.option
        )
        for i in chosen:
            option = select.option[i]
            # Fix-F 残存バグ: どうぐ無効スタジアム下でのどうぐ装着。
            if _is_tool_play(option, state) \
                    and _current_stadium_id(state) in NULLIFYING_STADIUM_IDS:
                rec["tool_attach_under_nullifying_stadium"] += 1
            # Fix-G 残存バグ: 攻撃できないアクティブを差し置いてベンチにエネを付けた。
            if (not has_attack) and active_targets \
                    and _energy_attach_area(option, state) == AreaType.BENCH:
                rec["bench_energy_while_active_cannot_attack"] += 1
        return

    # Fix-H: 「山札の上からN枚見て手札に加える」サーチ(全候補が LOOKING で解決できるもの)。
    if select.type == SelectType.CARD and select.option:
        ids = [_looking_id(o, state) for o in select.option]
        if any(cid is None for cid in ids):
            return
        rec["search_selects"] += 1
        if _bench_full(state, me) is True:
            return
        if not any(_is_basic_pokemon_id(cid) for cid in ids):
            return
        rec["search_selects_with_pokemon_candidate"] += 1
        if any(_is_basic_pokemon_id(ids[i]) for i in chosen):
            rec["search_pokemon_taken"] += 1
        else:
            rec["search_pokemon_missed"] += 1


def _misfire_flags(guard: str, obs, action, config: dict, ko_audit: bool) -> list[str]:
    """発火した decision について、そのガードの**誤爆条件**に当たるものを列挙する。"""
    state = obs.current
    me = state.yourIndex
    flags: list[str] = []
    if guard == "tool_stadium_guard":
        if _current_stadium_id(state) not in NULLIFYING_STADIUM_IDS:
            flags.append("stadium_not_nullifying")
    elif guard == "energy_to_active_first":
        if any(o.type == OptionType.ATTACK for o in obs.select.option):
            flags.append("attack_option_present")
        if _active_can_pay_any_attack(state, me) is True:
            flags.append("active_can_pay_attack_cost")
    elif guard == "search_pick_pokemon_first":
        if _bench_full(state, me) is True:
            flags.append("bench_full")
    elif guard == "low_deck_draw_brake":
        threshold = int(((config or {}).get("low_deck_draw_brake") or {}).get("deck_threshold", 6))
        try:
            if int(state.players[me].deckCount or 0) > threshold:
                flags.append("deck_above_threshold")
        except Exception:  # noqa: BLE001
            pass
        if ko_audit and _can_ko_now(obs, config):
            flags.append("fired_on_ko_turn")
    elif guard == "boss_lethal_gate":
        if ko_audit and _boss_target_ko_available(obs, config):
            flags.append("vetoed_with_ko_target")
    elif guard == "ability_draw_brake":
        # 絶対例外(攻撃コスト未充足なら絶対にvetoしない)の実装とは独立に、engine の
        # ATTACK 選択肢の有無だけで判定する(ガード実装が壊れても計測は壊れない)。
        if not any(o.type == OptionType.ATTACK for o in obs.select.option):
            flags.append("fired_without_attack_option")
    elif guard == "terastal_rotation":
        # 条件4(ドラパルトex等が居たら発火しない)だけを独立に再チェックする(=0のはず)。
        # 条件1-3(HP閾値/健康な退避先/KO整合性)は `_terastal_retreat_opportunity` /
        # `--ko-audit` 側の残存バグ計測でカバーする。
        opp = state.players[1 - me]
        opp_ids = {int(p.id) for p in (opp.active or []) if p is not None}
        opp_ids |= {int(p.id) for p in (opp.bench or []) if p is not None}
        if opp_ids & TERA_BENCH_DAMAGE_ARCHETYPE_IDS:
            flags.append("fired_vs_bench_damage_archetype")
    return flags


def _is_tera(card_id: int | None) -> bool:
    if card_id is None:
        return False
    try:
        from ptcg_ai.shared import card_cache
        return bool(card_cache.get_card(int(card_id)).tera)
    except Exception:  # noqa: BLE001
        return False


def _opponent_static_max_damage(state, me: int) -> int | None:
    """相手アクティブの技の静的最大damage。`ml_policy_agent._opponent_max_attack_damage` と
    同じ考え方だが、計測をガード実装から独立させるためあえて別実装にする。"""
    try:
        opp_active = state.players[1 - me].active or []
        if not opp_active or opp_active[0] is None:
            return None
        from ptcg_ai.shared import card_cache
        card = card_cache.get_card(int(opp_active[0].id))
        if not card.attacks:
            return 0
        return max(card_cache.get_attack(int(a)).damage for a in card.attacks)
    except Exception:  # noqa: BLE001
        return None


def _terastal_retreat_opportunity(state, select) -> int | None:
    """Fix-J の条件1・2・4(探索を伴わない条件)だけを独立に再判定し、退避機会があれば
    自分のアクティブの serial を返す(無ければ None)。

    条件3(にげてもKO成否が変わらない)はここでは見ない: 60試合規模の全 decision で
    毎回 `search_begin` を張るのはコストが大きく、かつガード実装(`retreat_safety_eval`)と
    同じ探索を再実装しても「独立検証」にならないため。よってこの関数が返す「機会」は
    Fix-Jが実際に発火する条件の**上位集合**(条件3で見送られるケースも含みうる)。
    「残存バグ」としては安全側(=過大に数える方向)であることに注意。
    """
    try:
        me = state.yourIndex
        if select.type != SelectType.MAIN:
            return None
        if not any(o.type == OptionType.RETREAT for o in select.option):
            return None
        active = state.players[me].active or []
        if not active or active[0] is None:
            return None
        mon = active[0]
        if not _is_tera(mon.id):
            return None
        opp = state.players[1 - me]
        opp_ids = {int(p.id) for p in (opp.active or []) if p is not None}
        opp_ids |= {int(p.id) for p in (opp.bench or []) if p is not None}
        if opp_ids & TERA_BENCH_DAMAGE_ARCHETYPE_IDS:
            return None
        threshold = _opponent_static_max_damage(state, me)
        if threshold is None or mon.hp > threshold:
            return None
        bench = state.players[me].bench or []
        if not any(b is not None and _is_tera(b.id) and b.hp > mon.hp for b in bench):
            return None
        return int(mon.serial)
    except Exception:  # noqa: BLE001
        return None


def _can_ko_now(obs, config: dict) -> bool:
    """`ko_search` でこのターンKOできるか(誤爆監査用。重いので --ko-audit のときだけ呼ぶ)。"""
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mpa
        from ptcg_ai.search import ko_search
        factory = mpa._model_hidden_state_factory(obs, config)
        return bool(ko_search.can_ko_this_turn(obs, factory, {}, time.perf_counter() + 0.2))
    except Exception:  # noqa: BLE001
        return False


def _boss_target_ko_available(obs, config: dict) -> bool:
    """ボスで引きずり出せる相手の中にKO可能な対象があるか(--ko-audit のときだけ)。"""
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mpa
        from ptcg_ai.search import boss_target_eval
        factory = mpa._model_hidden_state_factory(obs, config)
        report: dict = {}
        best = boss_target_eval.best_ko_target(obs, factory, {}, time.perf_counter() + 0.3, report=report)
        return best is not None
    except Exception:  # noqa: BLE001 - モジュール未実装/失敗は「情報なし」= 誤爆にしない
        return False


# ---------------------------------------------------------------------------
# 1試合(計測付き)
# ---------------------------------------------------------------------------

def _play_instrumented(
    agent_me, agent_opp, deck_me: list[int], deck_opp: list[int], my_index: int,
    seed: int, config: dict, ko_audit: bool,
) -> dict:
    """1試合を回しつつ、自陣の decision ごとに計測を行う。

    `run_match.play_match` と同じ順序・同じ安全策(必ず `battle_finish`)だが、
    「自分の手番だけ観測を覗いてメトリクスを取る」ために自前ループにしている。
    """
    import random

    from ptcg_ai.ml_policy import ml_policy_agent as mpa

    random.seed(seed)
    deck0, deck1 = (deck_me, deck_opp) if my_index == 0 else (deck_opp, deck_me)
    run_match._begin_match_state(deck0, deck1)
    mpa.reset_guard_stats()

    rec: dict = {
        "seed": seed, "my_index": my_index, "winner": None, "turns": None, "steps": 0,
        "error": None, "decisions": 0, "own_turns": 0, "attack_turns": 0,
        "attacks": 0, "zero_damage_attacks": 0,
        "tool_attach_under_nullifying_stadium": 0,
        "bench_energy_while_active_cannot_attack": 0,
        "search_selects": 0, "search_selects_with_pokemon_candidate": 0,
        "search_pokemon_taken": 0, "search_pokemon_missed": 0,
        "guard_fires": Counter(), "misfires": Counter(), "fire_events": [],
        "result_reason": None, "result_reason_source": None,
        # Fix-I(ability_draw_brake)の検証用: 試合終了時の自分の山札残枚数(平均最終山札)。
        "final_deck_count": None,
        # Fix-J(terastal_rotation)の検証用: 条件1・2・4(探索なし)だけで独立に見た
        # 「退避すべきだったのに退避しなかった」回数と、そのうち実際に気絶まで至った回数。
        "retreat_opportunity_missed": 0,
        "low_hp_faint_after_missed_retreat": 0,
    }
    my_logs: list = []
    main_turns: set[int] = set()
    attack_turns: set[int] = set()
    # Fix-J: 「退避すべきだったのに退避しなかった」個体の serial を保留し、後続の自陣
    # decision で discard(=気絶して墓地へ)に入っていないかを確認する。
    pending_low_hp_serials: set[int] = set()

    t0 = time.time()
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        rec["error"] = f"battle_start errorType={start_data.errorType}"
        return rec

    try:
        while True:
            obs = to_observation_class(obs_dict)
            if obs.current is None:
                rec["error"] = "current is None mid-match"
                return rec
            if obs.current.result != -1:
                rec["winner"] = obs.current.result
                rec["turns"] = obs.current.turn
                # 終局 observation が自分側のものなら、最後の decision 以降のログ(=勝ちを
                # 決めた攻撃など)もここで取り込む。相手側のものだと自分の decision 起点の
                # ログ列と重複しうるので取り込まない(0ダメージ攻撃の二重計上を防ぐ)。
                if obs.current.yourIndex == my_index:
                    my_logs.extend(obs.logs or [])
                try:
                    rec["final_deck_count"] = int(obs.current.players[my_index].deckCount or 0)
                except Exception:  # noqa: BLE001
                    pass
                # Fix-J: 保留中の「退避すべきだったのに退避しなかった」個体が、終局時点で
                # 墓地(=気絶)に入っていれば「HP僅少のまま気絶した」ものとして数える。
                if pending_low_hp_serials:
                    try:
                        discard_serials = {
                            int(c.serial) for c in (obs.current.players[my_index].discard or [])
                            if c is not None
                        }
                        rec["low_hp_faint_after_missed_retreat"] += \
                            len(pending_low_hp_serials & discard_serials)
                    except Exception:  # noqa: BLE001
                        pass
                _finalize_result(rec, obs, my_index)
                return rec
            if rec["steps"] >= run_match.MAX_STEPS:
                rec["error"] = f"max_steps_exceeded({run_match.MAX_STEPS})"
                rec["turns"] = obs.current.turn
                return rec

            turn_player = obs.current.yourIndex
            if turn_player != my_index:
                obs_dict = battle_select(agent_opp(obs))
                rec["steps"] += 1
                continue

            # --- ここから自陣の decision(計測対象) ---
            my_logs.extend(obs.logs or [])
            before = mpa.get_guard_stats()
            action = agent_me(obs)
            after = mpa.get_guard_stats()
            rec["decisions"] += 1

            state, select = obs.current, obs.select
            turn = int(getattr(state, "turn", 0) or 0)
            if select.type == SelectType.MAIN:
                main_turns.add(turn)
                for i in action:
                    if 0 <= i < len(select.option) \
                            and select.option[i].type == OptionType.ATTACK:
                        attack_turns.add(turn)
            _measure_decision(rec, obs, action)

            # Fix-J: 退避機会(条件1・2・4のみ、独立再判定)を見送っていないか。
            opportunity_serial = _terastal_retreat_opportunity(state, select)
            if opportunity_serial is not None:
                retreated = any(
                    0 <= i < len(select.option) and select.option[i].type == OptionType.RETREAT
                    for i in action
                )
                if not retreated:
                    rec["retreat_opportunity_missed"] += 1
                    pending_low_hp_serials.add(opportunity_serial)
            if pending_low_hp_serials:
                try:
                    discard_serials = {
                        int(c.serial) for c in (state.players[my_index].discard or [])
                        if c is not None
                    }
                    resolved = pending_low_hp_serials & discard_serials
                    if resolved:
                        rec["low_hp_faint_after_missed_retreat"] += len(resolved)
                        pending_low_hp_serials -= resolved
                except Exception:  # noqa: BLE001
                    pass

            fired_guards: set[str] = set()
            for key, guard in FIRE_COUNTERS.items():
                delta = int(after.get(key, 0)) - int(before.get(key, 0))
                if delta > 0:
                    rec["guard_fires"][key] += delta
                    fired_guards.add(guard)
            for key in after:
                delta = int(after.get(key, 0)) - int(before.get(key, 0))
                if delta > 0 and key not in FIRE_COUNTERS:
                    rec["guard_fires"][key] += delta
            for guard in sorted(fired_guards):
                flags = _misfire_flags(guard, obs, action, config, ko_audit)
                for flag in flags:
                    rec["misfires"][f"{guard}.{flag}"] += 1
                event = {
                    "turn": turn, "guard": guard, "select_type": int(select.type),
                    "misfire": flags,
                }
                if flags:
                    event["diagnostics"] = _misfire_diagnostics(obs)
                rec["fire_events"].append(event)

            obs_dict = battle_select(action)
            rec["steps"] += 1
    except Exception as exc:  # noqa: BLE001 - 1試合の異常で計測全体を止めない
        rec["error"] = repr(exc)
        return rec
    finally:
        battle_finish()
        rec["seconds"] = time.time() - t0
        rec["own_turns"] = len(main_turns)
        rec["attack_turns"] = len(attack_turns)
        attacks, zero = _count_attack_damage(my_logs, my_index)
        rec["attacks"] = attacks
        rec["zero_damage_attacks"] = zero
        rec["guard_fires"] = dict(rec["guard_fires"])
        rec["misfires"] = dict(rec["misfires"])


def _finalize_result(rec: dict, obs, my_index: int) -> None:
    """終局 observation から敗因(山札切れ/アクティブ不在/サイド)を取る。

    第一候補は `LogType.RESULT` の `reason`(1: サイド0 / 2: 山札0でターン開始 /
    3: バトル場にポケモンがいない / 4: カード効果)。ログに無い場合は最終盤面から推定する。
    """
    reason = None
    for log in reversed(obs.logs or []):
        if int(getattr(log, "type", -1)) == int(LogType.RESULT):
            reason = getattr(log, "reason", None)
            break
    if reason is not None:
        rec["result_reason"] = int(reason)
        rec["result_reason_source"] = "log"
        return
    try:
        loser = 1 - int(obs.current.result)
        player = obs.current.players[loser]
        if int(player.deckCount or 0) <= 0:
            rec["result_reason"] = 2
        elif not [p for p in (player.active or []) if p is not None]:
            rec["result_reason"] = 3
        else:
            rec["result_reason"] = 1
        rec["result_reason_source"] = "inferred"
    except Exception:  # noqa: BLE001
        rec["result_reason_source"] = "unknown"


def _count_attack_damage(logs: list, my_index: int) -> tuple[int, int]:
    """自分の ATTACK ログのうち、直後に相手へのダメージが1件も無かったものを数える。

    ログのスキーマ(実リプレイで確認): ATTACK{playerIndex, cardId, attackId} の直後に
    HP_CHANGE{playerIndex=<HPが変わったカードの持ち主>, value=<負ならダメージ>} が並ぶ。
    次の ATTACK / TURN_END / TURN_START までを1回の攻撃の解決範囲とみなす。
    """
    attacks = zero = 0
    n = len(logs)
    for i, log in enumerate(logs):
        if int(getattr(log, "type", -1)) != int(LogType.ATTACK):
            continue
        if int(getattr(log, "playerIndex", -1)) != my_index:
            continue
        attacks += 1
        damaged = False
        for j in range(i + 1, n):
            nxt = logs[j]
            ntype = int(getattr(nxt, "type", -1))
            if ntype in (int(LogType.ATTACK), int(LogType.TURN_END), int(LogType.TURN_START)):
                break
            if ntype == int(LogType.HP_CHANGE) \
                    and int(getattr(nxt, "playerIndex", -1)) == 1 - my_index \
                    and int(getattr(nxt, "value", 0) or 0) < 0:
                damaged = True
                break
        if not damaged:
            zero += 1
    return attacks, zero


# ---------------------------------------------------------------------------
# 並列実行(cg エンジンはプロセス内シングルトンなのでプロセス並列)
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(config_name: str, own_deck: str, own_weights: str, opp_specs: dict,
                 ko_audit: bool) -> None:
    os.chdir(_SUB)
    from ptcg_ai.core.config import load_config
    _WORKER["config_name"] = config_name
    _WORKER["config"] = load_config(config_name)
    _WORKER["agent_me"] = run_league.build_agent("ml_policy", own_weights, config_name)
    _WORKER["deck_me"] = run_league.read_deck_csv_file(own_deck)
    _WORKER["opp_specs"] = opp_specs
    _WORKER["opp_cache"] = {}
    _WORKER["ko_audit"] = ko_audit


def _get_opponent(arch: str):
    cached = _WORKER["opp_cache"].get(arch)
    if cached is None:
        spec = _WORKER["opp_specs"][arch]
        agent = run_league.build_agent("ml_policy", spec["weights"], "abl_5_full")
        deck = run_league.read_deck_csv_file(spec["deck"])
        cached = (agent, deck)
        _WORKER["opp_cache"][arch] = cached
    return cached


def _worker_play(task: tuple[int, str, int, int]) -> dict:
    index, arch, seed, my_index = task
    agent_opp, deck_opp = _get_opponent(arch)
    rec = _play_instrumented(
        _WORKER["agent_me"], agent_opp, _WORKER["deck_me"], deck_opp, my_index, seed,
        _WORKER["config"], _WORKER["ko_audit"],
    )
    rec["index"] = index
    rec["arch"] = arch
    return rec


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------

def _wilson(successes: int, n: int) -> list[float]:
    lo, hi = run_league.wilson_interval(successes, n)
    return [round(lo, 4), round(hi, 4)]


def summarize(records: list[dict]) -> dict:
    valid = [r for r in records if r["error"] is None and r["winner"] is not None]
    errors = [r for r in records if r["error"] is not None]
    wins = sum(1 for r in valid if r["winner"] == r["my_index"])
    n = len(valid)

    fires: Counter = Counter()
    misfires: Counter = Counter()
    for r in valid:
        fires.update(r["guard_fires"])
        misfires.update(r["misfires"])

    deckout_losses = sum(1 for r in valid
                         if r["winner"] != r["my_index"] and r.get("result_reason") == 2)
    no_active_losses = sum(1 for r in valid
                           if r["winner"] != r["my_index"] and r.get("result_reason") == 3)
    deckout_wins = sum(1 for r in valid
                       if r["winner"] == r["my_index"] and r.get("result_reason") == 2)
    # Fix-I 検証用: 平均最終山札(残っている試合だけを対象、None は除く)。
    final_decks = [r["final_deck_count"] for r in valid if r.get("final_deck_count") is not None]
    attacks = sum(r["attacks"] for r in valid)
    zero = sum(r["zero_damage_attacks"] for r in valid)
    own_turns = sum(r["own_turns"] for r in valid)
    attack_turns = sum(r["attack_turns"] for r in valid)
    tool_waste = sum(r["tool_attach_under_nullifying_stadium"] for r in valid)
    bench_energy_bug = sum(r["bench_energy_while_active_cannot_attack"] for r in valid)
    search_opps = sum(r["search_selects_with_pokemon_candidate"] for r in valid)
    search_taken = sum(r["search_pokemon_taken"] for r in valid)
    search_missed = sum(r["search_pokemon_missed"] for r in valid)
    # Fix-J(terastal_rotation)検証用。
    retreat_opportunity_missed = sum(r.get("retreat_opportunity_missed", 0) for r in valid)
    low_hp_faints = sum(r.get("low_hp_faint_after_missed_retreat", 0) for r in valid)

    by_arch: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        grouped[r["arch"]].append(r)
    for arch, rs in sorted(grouped.items()):
        w = sum(1 for r in rs if r["winner"] == r["my_index"])
        by_arch[arch] = {
            "games": len(rs), "wins": w, "win_rate": round(w / len(rs), 4),
            "fires": dict(sum((Counter(r["guard_fires"]) for r in rs), Counter())),
            "deckout_losses": sum(1 for r in rs
                                  if r["winner"] != r["my_index"] and r.get("result_reason") == 2),
        }

    return {
        "games": len(records),
        "valid_games": n,
        "errors": len(errors),
        "error_reasons": dict(Counter(r["error"] for r in errors)),
        "win_rate": round(wins / n, 4) if n else None,
        "wins": wins,
        "win_rate_wilson_95ci": _wilson(wins, n),
        "avg_turns": round(sum(r["turns"] or 0 for r in valid) / n, 2) if n else None,
        "avg_seconds": round(sum(r.get("seconds") or 0 for r in valid) / n, 2) if n else None,
        # --- 発火率 ---
        "guard_fires_total": dict(fires),
        "guard_fires_per_game": {k: round(v / n, 4) for k, v in sorted(fires.items())} if n else {},
        "games_with_fire": {
            k: sum(1 for r in valid if r["guard_fires"].get(k, 0) > 0) for k in sorted(fires)
        },
        # --- 誤爆(すべて 0 であるべき) ---
        "misfires_total": dict(misfires),
        "misfires_per_game": {k: round(v / n, 4) for k, v in sorted(misfires.items())} if n else {},
        # --- 結果メトリクス ---
        "outcomes": {
            "deckout_loss_rate": round(deckout_losses / n, 4) if n else None,
            "deckout_losses": deckout_losses,
            "no_active_loss_rate": round(no_active_losses / n, 4) if n else None,
            "no_active_losses": no_active_losses,
            "opponent_deckout_wins": deckout_wins,
            "zero_damage_attacks_per_game": round(zero / n, 4) if n else None,
            "zero_damage_attacks": zero,
            "attacks_per_game": round(attacks / n, 4) if n else None,
            "attack_turn_rate": round(attack_turns / own_turns, 4) if own_turns else None,
            "own_turns_total": own_turns,
            "tool_attach_under_nullifying_stadium_per_game": round(tool_waste / n, 4) if n else None,
            "tool_attach_under_nullifying_stadium": tool_waste,
            # Fix-I(ability_draw_brake)検証用: 試合終了時点の自分の平均山札残枚数。
            "avg_final_deck_count":
                round(sum(final_decks) / len(final_decks), 3) if final_decks else None,
        },
        # --- 残存バグ(=修正対象の事象が最終手にまだ残っている回数。ガードONで 0 に近づくべき) ---
        "residual_bugs": {
            "tool_attach_under_nullifying_stadium": tool_waste,
            "bench_energy_while_active_cannot_attack": bench_energy_bug,
            "bench_energy_while_active_cannot_attack_per_game":
                round(bench_energy_bug / n, 4) if n else None,
            "search_pokemon_missed": search_missed,
            "search_pokemon_opportunities": search_opps,
            "search_pokemon_take_rate": round(search_taken / search_opps, 4) if search_opps else None,
            # Fix-J(terastal_rotation)検証用: 条件1・2・4(探索なし)だけで独立に見た
            # 「退避すべきだったのに退避しなかった」回数と、そのうち実際に気絶まで至った回数
            # (=タスクで要求されている「HP僅少のまま気絶した回数」)。ガードONで両方とも
            # 0に近づくべき(ガードが発火した decision はここでは既に「退避した」側なので
            # 対象から自然に外れる)。
            "retreat_opportunity_missed": retreat_opportunity_missed,
            "retreat_opportunity_missed_per_game":
                round(retreat_opportunity_missed / n, 4) if n else None,
            "low_hp_faint_after_missed_retreat": low_hp_faints,
            "low_hp_faint_after_missed_retreat_per_game":
                round(low_hp_faints / n, 4) if n else None,
        },
        "by_archetype": by_arch,
    }


def build_tasks(opponents: list[str], games: int, seed_base: int) -> list[tuple[int, str, int, int]]:
    """(index, arch, seed, my_index) のタスク列。先手/後手は index の偶奇で入れ替える。"""
    if opponents == ["mix"]:
        weights = [(a, w) for a, w, _g in MIX_FIELD]
    else:
        weights = [(a, 1) for a in opponents]
    total_w = sum(w for _a, w in weights)
    order: list[str] = []
    for arch, w in weights:
        order.extend([arch] * max(1, round(games * w / total_w)))
    # 端数調整(round の積み上げで games とずれる)。
    while len(order) < games:
        order.append(weights[len(order) % len(weights)][0])
    order = order[:games]
    order.sort()  # 同じ対面をまとめる = worker のモデルロードを減らす
    return [(i, arch, seed_base + i, i % 2) for i, arch in enumerate(order)]


def opponent_specs(archs: set[str]) -> dict:
    specs = {}
    gen_by_arch = {a: g for a, _w, g in MIX_FIELD}
    for arch in sorted(archs):
        gen = gen_by_arch.get(arch, "_g2")
        deckdir = _DECKDIR_G2 if gen == "_g2" else _DECKDIR_J
        deck = deckdir / arch / "01.csv"
        weights = _WDIR / f"policy_weights_{arch}{gen}.json"
        if not deck.exists():   # gen2 に無いアーキは7月版へフォールバック
            deck = _DECKDIR_J / arch / "01.csv"
            weights = _WDIR / f"policy_weights_{arch}.json"
        if not deck.exists():
            raise FileNotFoundError(f"デッキが見つかりません: {arch} ({deck})")
        if not weights.exists():
            fallback = _WDIR / f"policy_weights_{arch}.json"
            if not fallback.exists():
                raise FileNotFoundError(f"相手重みが見つかりません: {arch} ({weights})")
            weights = fallback
        specs[arch] = {"deck": str(deck), "weights": str(weights)}
    return specs


def run_arm(config_name: str, tasks: list, specs: dict, own_deck: str, own_weights: str,
            workers: int, ko_audit: bool, progress_every: int) -> list[dict]:
    records: list[dict] = []
    t0 = time.time()
    if workers <= 1:
        _worker_init(config_name, own_deck, own_weights, specs, ko_audit)
        for task in tasks:
            records.append(_worker_play(task))
            if progress_every and len(records) % progress_every == 0:
                print(f"  [{config_name}] {len(records)}/{len(tasks)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
            initargs=(config_name, own_deck, own_weights, specs, ko_audit),
        ) as pool:
            futures = [pool.submit(_worker_play, task) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
                if progress_every and len(records) % progress_every == 0:
                    print(f"  [{config_name}] {len(records)}/{len(tasks)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
    records.sort(key=lambda r: r["index"])
    return records


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_report(summaries: dict[str, dict]) -> None:
    arms = list(summaries)
    width = max(len(a) for a in arms) + 2
    print("\n" + "=" * 100)
    print("METRICS GATE")
    print("=" * 100)

    def row(label: str, getter) -> None:
        cells = "  ".join(f"{_fmt(getter(summaries[a])):>12}" for a in arms)
        print(f"{label:<52}{cells}")

    print(f"{'metric':<52}" + "  ".join(f"{a[-12:]:>12}" for a in arms))
    print("-" * 100)
    row("games (valid)", lambda s: s["valid_games"])
    row("errors", lambda s: s["errors"])
    row("win_rate (参考値。判定には使わない)", lambda s: s["win_rate"])
    row("avg_turns", lambda s: s["avg_turns"])
    print("-- 結果メトリクス ------------------------------------------")
    for key in ("deckout_loss_rate", "no_active_loss_rate", "zero_damage_attacks_per_game",
                "attacks_per_game", "attack_turn_rate",
                "tool_attach_under_nullifying_stadium_per_game", "avg_final_deck_count"):
        row(key, lambda s, k=key: s["outcomes"][k])
    print("-- 残存バグ (ガードONで 0 に近づくべき) --------------------")
    for key in ("tool_attach_under_nullifying_stadium",
                "bench_energy_while_active_cannot_attack",
                "search_pokemon_missed", "search_pokemon_opportunities",
                "search_pokemon_take_rate",
                "retreat_opportunity_missed_per_game",
                "low_hp_faint_after_missed_retreat_per_game"):
        row(key, lambda s, k=key: s["residual_bugs"][k])
    print("-- 発火率 (per game) ---------------------------------------")
    keys = sorted({k for s in summaries.values() for k in s["guard_fires_per_game"]})
    for key in keys:
        row(key, lambda s, k=key: s["guard_fires_per_game"].get(k, 0.0))
    print("-- 誤爆 (すべて 0 であるべき) ------------------------------")
    mkeys = sorted({k for s in summaries.values() for k in s["misfires_total"]})
    if not mkeys:
        print("  (誤爆 0 件)")
    for key in mkeys:
        row(key, lambda s, k=key: s["misfires_total"].get(k, 0))
    print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description="ガード発火率・誤爆率・結果メトリクスの計測ハーネス")
    ap.add_argument("--config", nargs="+", required=True,
                    help="計測する agent config 名(複数指定で同一 seed 列の paired 比較)")
    ap.add_argument("--games", type=int, default=60, help="1アームあたりの試合数")
    ap.add_argument("--workers", type=int, default=1, help="並列プロセス数(0以下=CPU数-1)")
    ap.add_argument("--opponents", default="mix",
                    help="'mix'(既定, MIX_FIELD の比で配分) または カンマ区切りのアーキ名")
    ap.add_argument("--own-deck", default=str(DEFAULT_OWN_DECK))
    ap.add_argument("--own-weights", default=str(DEFAULT_OWN_WEIGHTS))
    ap.add_argument("--seed-base", type=int, default=1100000)
    ap.add_argument("--ko-audit", action="store_true",
                    help="KO関連の誤爆判定(low_deck_draw_brake / boss_lethal_gate)を有効化する"
                         "(発火した decision でだけ ko_search を張るので追加コストは限定的)")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--out", default=str(_HERE / "_metrics_gate_results.json"))
    args = ap.parse_args()

    opponents = ["mix"] if args.opponents == "mix" else [
        a.strip() for a in args.opponents.split(",") if a.strip()
    ]
    tasks = build_tasks(opponents, args.games, args.seed_base)
    specs = opponent_specs({arch for _i, arch, _s, _m in tasks})
    workers = run_league.resolve_workers(args.workers)

    print(f"own deck   : {args.own_deck}")
    print(f"own weights: {args.own_weights}")
    print(f"opponents  : {Counter(arch for _i, arch, _s, _m in tasks)}")
    print(f"workers={workers} games/arm={len(tasks)} ko_audit={args.ko_audit}")

    summaries: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}
    for config_name in args.config:
        print(f"\n=== arm: {config_name} ===", flush=True)
        records = run_arm(config_name, tasks, specs, args.own_deck, args.own_weights,
                          workers, args.ko_audit, args.progress_every)
        all_records[config_name] = records
        summaries[config_name] = summarize(records)
        print(json.dumps({k: v for k, v in summaries[config_name].items()
                          if k in ("valid_games", "errors", "win_rate", "guard_fires_total",
                                   "misfires_total")}, ensure_ascii=False), flush=True)

    print_report(summaries)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "tasks": [{"index": i, "arch": a, "seed": s, "my_index": m} for i, a, s, m in tasks],
        "summaries": summaries,
        "games": all_records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
