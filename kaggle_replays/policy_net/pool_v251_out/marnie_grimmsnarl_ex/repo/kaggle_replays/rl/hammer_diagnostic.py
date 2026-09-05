"""改造ハンマー(Enhanced Hammer, 1081)診断: 「相手を倒せる局面」で本番構成のエージェントが
実際にハンマーを使っているかを数える。

背景(なぜこの診断が要るか)
--------------------------
743 Alakazam の Powerful Hand は damage=0 の可変ダメージ技(手札枚数 x 20 のダメージ
"カウンター"を置く効果ダメージ)。11 Mist Energy / 20 Rock Fighting Energy のような
「ワザの効果のみを防ぐ」特殊エネルギーが付いていると、Powerful Hand は完全に無効化される
(ダメージそのものではなく効果を防ぐ側のため)。1081 Enhanced Hammer で相手の特殊エネを
1枚剥がせばこの無効化は外れ、Powerful Hand が通るようになる。

alakazam デッキは Enhanced Hammer を4枚積んだ唯一のデッキで、crustle(Mist 4 + Rock 4)・
shirona_garchomp_ex(Rock 4)・ogerpon_teal_ex(Mist 2) のような防壁デッキを想定して
組まれている。この関係が実際にゲーム内で機能しているか(ハンマーを撃つべき局面で
本番エージェントが撃っているか)を、本番構成(league/run_league.py 経由の ml_policy +
abl_5_full)のまま、意思決定に一切介入せず観測だけして数える。

対象の決定点(すべて満たすもの)
--------------------------------
1. 自分の手番の SelectType.MAIN
2. 自分のバトルポケモンが Alakazam(743) で、Powerful Hand(attackId=1072) が使える
   (energy_requirements.is_energy_sufficient)
3. いまは倒せない: attack_features.resolve_damage(...) が相手バトルポケモンの残りHPに届かない
4. 相手のバトルポケモンに付いている特殊エネを1枚(dataclasses.replace で複製した仮想の
   Pokemon から)剥がすと、resolve_damage(...) が残りHP以上になる
5. 手札に Enhanced Hammer(1081) があり、MAIN選択の選択肢(OptionType.PLAY)として
   実際にプレイ可能

計測方法
--------
`league/run_league.py` の `build_agent("ml_policy", weights_path, config_base)` が返す
`agent(obs)` 呼び出し可能オブジェクトを**そのまま**薄くラップする。ラッパーは
(1) obs を見て上の条件を判定し、(2) 本物の agent(obs) を呼び、(3) 返ってきた action が
「Enhanced Hammer をプレイする選択肢」を含むかどうかを記録するだけで、選択そのものには
一切介入しない。`ptcg_ai/` 配下のエージェント本体・`run_league.py`/`run_match.py` は変更しない。

対戦は `league/run_league.py` の `play_match`(run_match.py)を直接呼ぶ(collect_parallel.py の
対戦ループは pm._forward() で素の PolicyModel しか呼ばずリーサル探索/pipeline探索を通らないため
使わない、という指示に従う)。

run2 拡張: 「ハンマーを撃つべき局面で探索がそもそも起動しているのか」の切り分け
--------------------------------------------------------------------------------
qualifying な決定点ごとに、追加で次を観測する(いずれも読み取り専用・意思決定には介入しない)。

1. **どの経路が行動を決めたか**: `ml_policy_agent` の `_try_lethal` / `_try_pipeline` /
   `_try_attack_hybrid` をモジュール属性ごと薄いラッパで monkeypatch し(`RouteRecorder`)、
   戻り値を観測してから元の関数の戻り値をそのまま返す。`_select_action` はこれらをモジュール
   グローバルとして毎回名前解決して呼ぶため、この置き換えだけで判断への介入なしに経路が分かる。
2. **pipeline が深い探索(Step3-5)を行わなかった理由**: `ptcg_ai/search/pipeline.py` の
   `search()` は変更せず、診断側 (`compute_step2_and_pipeline_diagnosis`) で同じ前提条件
   (select 形状・手番・スコア取得・top1 ショートカット判定)を再現して分類する。
3. **ハンマーの方策順位**: `pipeline.search()` の Step2 と同じ呼び出し
   (`policy_model.score_options` → softmax → `pipeline._select_candidate_indices`)を
   診断側で再現し、Enhanced Hammer の選択肢が方策スコア降順で何位か・top_k に入っていたか・
   top1 softmax 確率を記録する。この計算は agent_fn(obs) 呼び出しの**後**に行う(本番の
   `_select_action` が実際に見た `match_context` の状態と食い違わないようにするため)。

使用例:
    python kaggle_replays/rl/hammer_diagnostic.py --games 40 --opponents crustle
    python kaggle_replays/rl/hammer_diagnostic.py --games 80 --opponents crustle,shirona_garchomp_ex,ogerpon_teal_ex --workers 8

run3 拡張(問い1): ハンマーの無駄打ちを測る
------------------------------------------
qualifying 判定(cond2〜5)とは独立に、**Enhanced Hammer(1081) を実際にプレイしたすべての
決定**を観測し、各プレイを次の3つに分類する(いずれも `attack_features.resolve_damage` /
`attack_features.damage_prevented` を使い、「特殊エネがあれば無効化」のような自前判定はしない)。

  1_enables_ko    : そのターンに相手バトルポケモンを倒せるようになった
  2_unblocks_no_ko: 自分の攻撃(Powerful Hand)を無効化している特殊エネを剥がした(が倒せない)
  3_unrelated     : 打点に無関係な特殊エネを剥がした(手番のアクティブが Alakazam でない/
                     Powerful Hand のエネルギーが足りない/剥がした相手がバトルポケモンでない、
                     を含む)

分類対象の特定は「MAIN選択でハンマーのPLAYを選んだ直後の局面」を基準(before)にし、実際に
ゲームエンジンが解決した後の相手の場(アクティブ+ベンチ)を`serial`単位で突き合わせて
「どの特殊エネが消えたか」を検出する(pending_hammer 状態機械、`_diff_removed_special_energy`)。
検出できたら、evaluate_decision の cond4 と同じ手法(`_remove_one_energy_card` で仮想的に
1枚だけ除いた局面を作る)で判定する。

さらに、`evaluate_decision` の cond4(剥がせば倒せる)が成立した決定点を、cond5(手札にハンマー
がある)の成否に関わらずすべて記録し(`lethal_via_hammer_events`)、「手札にハンマーが無かった」
試合について、それ以前にそのゲーム内で 2_unblocks_no_ko / 3_unrelated のハンマーを何枚
撃っていたか(=引き運か使い切りかの切り分け)を突き合わせる。

run3 拡張(問い2): 葉評価を値ネットに替えたら直りそうか
--------------------------------------------------------
「逃した」決定点(cond2&3&4&5 成立・実際にはハンマーを使わなかった)ごとに、`search_begin`で
決定化した局面から `search_step` で (a) ハンマーをプレイした後 (b) 実際に選んだ手を打った後、
の2つの1手先の局面を作り、`leaf_eval.build_evaluator({"kind": "handcrafted"})` と
`{"kind": "value"}`(既定重みを `kaggle_replays/value_net/value_weights_wired_2026-08-07.json`
に差し替えたもの。`ValueModelEvaluator(model=ValueModel(weights_path=...))` で直接組み立てる、
`ptcg_ai/` 側は一切変更しない)の両方で評価し、どちらが「ハンマー後」を高く評価したかを記録する。
ロールアウトはしない(1手先だけを評価する)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # kaggle_replays/rl
_ROOT = _HERE.parents[1]                          # repo root
_LEAGUE_DIR = _ROOT / "league"
_SUB_DIR = _ROOT / "sample_submission"
for _p in (str(_LEAGUE_DIR), str(_ROOT), str(_SUB_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_league  # noqa: E402  (league/run_league.py 本番同様の build_agent 等を再利用)
from cg.api import (  # noqa: E402
    AreaType, CardType, Observation, OptionType, SelectType,
    search_begin, search_end, search_release, search_step,
)
from ptcg_ai.board_evaluation import attack_features, energy_requirements  # noqa: E402
from ptcg_ai.core.config import load_config as load_ptcg_config  # noqa: E402
from ptcg_ai.hidden_information import match_context, search_adapter  # noqa: E402  (run3問い2用。観測のみ)
from ptcg_ai.learning.value_model import ValueModel  # noqa: E402  (run3問い2用。観測のみ)
from ptcg_ai.ml_policy import ml_policy_agent  # noqa: E402  (② ③ 診断用。観測のみ、agent 本体は変更しない)
from ptcg_ai.search import leaf_eval  # noqa: E402  (run3問い2用。観測のみ)
from ptcg_ai.search import pipeline as pipeline_search  # noqa: E402  (② ③ 診断用。観測のみ)
from ptcg_ai.shared import card_cache  # noqa: E402

ALAKAZAM_ID = 743
POWERFUL_HAND_ATTACK_ID = 1072
ENHANCED_HAMMER_ID = 1081

ALAKAZAM_DECK_PATH = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam" / "01.csv"
_DECKDIR = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
_WDIR = _SUB_DIR / "ptcg_ai" / "learning"

# 相手(Mist/Rock Energy 系デッキ)。deck + 専用重み(round_robin.py / field_gauntlet_deck.py と
# 同じ既存の慣習: 各アーキタイプは自分専用の policy_weights_<archetype>.json を使う)。
# config は v0only 固定(round_robin.py / field_gauntlet_deck.py の OPP_CONFIG と同じ)。
# 契約側(alakazam)だけ本番既定の abl_5_full を使う。
OPPONENT_SPECS: dict[str, tuple[Path, Path]] = {
    "crustle": (_DECKDIR / "crustle" / "01.csv", _WDIR / "policy_weights_crustle.json"),
    "shirona_garchomp_ex": (
        _DECKDIR / "shirona_garchomp_ex" / "01.csv", _WDIR / "policy_weights_shirona_garchomp_ex.json"
    ),
    "ogerpon_teal_ex": (
        _DECKDIR / "ogerpon_teal_ex" / "01.csv", _WDIR / "policy_weights_ogerpon_teal_ex.json"
    ),
}

CONTENDER_CONFIG = os.environ.get("PTCG_AI_ML_CONFIG", "abl_5_full")
OPPONENT_CONFIG = "ml_lethal_attackplan_v0only"

# run3 問い2用。配線後の特徴量で作り直した最新の値ネット重み(test AUC 0.80、
# `sample_submission/ptcg_ai/learning/value_weights.json`(7月版、test AUC 0.74)より新しい)。
# `ptcg_ai/` 側のコードは変更せず、`ValueModelEvaluator(model=ValueModel(weights_path=...))` で
# 呼び出し側からこのパスを注入するだけ(leaf_eval.build_evaluator の既定重みパスは変えない)。
_WIRED_VALUE_WEIGHTS_PATH = _ROOT / "kaggle_replays" / "value_net" / "value_weights_wired_2026-08-07.json"


# ---------------------------------------------------------------------------
# 決定点の条件判定(観測のみ。obs から読むだけで一切書き換えない)
# ---------------------------------------------------------------------------


def _special_energy_cards(pokemon) -> list:
    """pokemon.energyCards のうち特殊エネルギー(CardType.SPECIAL_ENERGY)のものだけを返す。"""
    out = []
    for card in getattr(pokemon, "energyCards", None) or []:
        try:
            data = card_cache.get_card(card.id)
        except Exception:  # noqa: BLE001
            continue
        if data.cardType == CardType.SPECIAL_ENERGY:
            out.append(card)
    return out


def _remove_one_energy_card(pokemon, card_to_remove):
    """pokemon の複製を作り、energyCards から card_to_remove(serial一致)を1枚だけ除く。

    元の pokemon オブジェクトは変更しない(dataclasses.replace で新しいインスタンスを作る)。
    energies(EnergyType の集計リスト)も対応する1ユニットを除いて整合を取るが、
    resolve_damage/damage_prevented はどちらも defender.energyCards だけを見て
    defender.energies は参照しない(attack_features.py 確認済み)ため、ここでの整合性は
    「壊れたPokemonを返さない」ための保守目的であり、ダメージ計算の結果には影響しない。
    """
    new_energy_cards = []
    removed = False
    for card in pokemon.energyCards:
        if not removed and card.serial == card_to_remove.serial:
            removed = True
            continue
        new_energy_cards.append(card)

    new_energies = list(pokemon.energies)
    try:
        removed_type = card_cache.get_card(card_to_remove.id).energyType
        if removed_type in new_energies:
            new_energies.remove(removed_type)
    except Exception:  # noqa: BLE001
        pass

    return replace(pokemon, energyCards=new_energy_cards, energies=new_energies)


def _hammer_play_indices(obs: Observation) -> list[int]:
    """obs.select.option のうち、手札の Enhanced Hammer(1081) を PLAY する選択肢の index 一覧。

    cond2(Alakazamがアクティブでエネルギー十分)を満たすかどうかに関わらず判定できる
    (run3 拡張①: 実際にハンマーがプレイされた瞬間を cond 判定と独立に検出するために使う)。
    ハンマーは ACE SPEC ではない通常Itemなので手札に複数枚(最大4枚)同時に入り得て、
    その場合 option には手札インデックスの数だけ別々の PLAY オプションが並ぶ。
    """
    if obs.select is None or obs.current is None:
        return []
    state = obs.current
    player = state.players[state.yourIndex]
    hand = player.hand or []
    out: list[int] = []
    for i, option in enumerate(obs.select.option):
        if option.type == OptionType.PLAY and option.index is not None:
            if 0 <= option.index < len(hand) and hand[option.index].id == ENHANCED_HAMMER_ID:
                out.append(i)
    return out


def evaluate_decision(obs: Observation) -> dict:
    """1つの MAIN 選択局面が、ハンマー診断の対象条件をどこまで満たすかを判定する。

    条件は cond2 -> cond3 -> cond4 -> cond5 の順に階層的に評価する(前の条件を満たさない
    限り後の条件は判定不能/無意味なため)。すべて満たせば qualifies=True で、そのときだけ
    target_option_index(Enhanced Hammer をプレイする選択肢のインデックス)が入る。
    """
    result: dict = {"cond2": False, "cond3": False, "cond4": False, "cond5": False, "qualifies": False}

    state = obs.current
    you = state.yourIndex
    player = state.players[you]
    opponent = state.players[1 - you]

    if not player.active or player.active[0] is None:
        return result
    active = player.active[0]
    if active.id != ALAKAZAM_ID:
        return result

    attack = card_cache.get_attack(POWERFUL_HAND_ATTACK_ID)
    if not energy_requirements.is_energy_sufficient(attack, active.energies):
        return result
    result["cond2"] = True

    if not opponent.active or opponent.active[0] is None:
        return result
    defender = opponent.active[0]
    defender_card = card_cache.get_card(defender.id)
    hand_count = player.handCount

    defender_side = [p for p in (opponent.active or []) if p is not None] + list(opponent.bench)
    damage_is_effect = attack_features.damage_is_effect_based(attack)

    current_damage = attack_features.resolve_damage(
        attack, active, defender_card.weakness, defender_card.resistance, hand_count,
        defender=defender, defender_side_pokemon=defender_side,
        defender_is_benched=False, damage_is_effect=damage_is_effect,
    )
    result.update(
        current_damage=current_damage,
        defender_hp=defender.hp,
        defender_max_hp=defender.maxHp,
        defender_name=defender_card.name,
        hand_count=hand_count,
    )

    if current_damage >= defender.hp:
        # 現状ですでに倒せる(ハンマーは不要) -> 対象外。
        return result
    result["cond3"] = True

    specials = _special_energy_cards(defender)
    result["special_energy_names"] = [card_cache.get_card(c.id).name for c in specials]

    qualifying = None
    for card in specials:
        hyp_defender = _remove_one_energy_card(defender, card)
        hyp_side = [hyp_defender if p is defender else p for p in defender_side]
        hyp_damage = attack_features.resolve_damage(
            attack, active, defender_card.weakness, defender_card.resistance, hand_count,
            defender=hyp_defender, defender_side_pokemon=hyp_side,
            defender_is_benched=False, damage_is_effect=damage_is_effect,
        )
        if hyp_damage >= defender.hp:
            qualifying = (card, hyp_damage)
            break
    if qualifying is None:
        return result
    result["cond4"] = True
    result["removed_energy_name"] = card_cache.get_card(qualifying[0].id).name
    result["removed_energy_card"] = qualifying[0]  # run3問い2用: 実カード(serial)。KO成立に使った1枚
    result["hypothetical_damage"] = qualifying[1]

    # Enhanced Hammer(1081)は ACE SPEC ではない通常の Item(EN_Card_Data.csv 確認済み)なので、
    # 手札に複数枚(最大4枚)同時に入り得る。その場合 obs.select.option には手札インデックスの
    # 数だけ別々の PLAY オプションが並ぶ(すべて同じカードだが option インデックスは別)。
    # run1 のスクリプトは「最初に見つかった1つ」だけを target_option_index として記録し、
    # used_hammer を `target_option_index in action` で判定していたため、エージェントが
    # (同じ)ハンマーの別コピーを選んだ場合に "使わなかった" と誤判定するバグがあった
    # (run2 の実装確認中の120試合サンプルで、chosen_desc が "PLAY(Enhanced Hammer)" なのに
    # used_hammer=False という組が2件連続で観測され、発覚した)。target_option_index(最初の1つ)
    # は③のランク計算等の後方互換のために残しつつ、hammer_option_indices(全コピー)を新たに
    # 持たせ、used_hammer 判定はそちらとの積集合で行う(下記 instrument() 側で使用)。
    hammer_option_indices = _hammer_play_indices(obs)
    if not hammer_option_indices:
        return result
    result["cond5"] = True
    result["target_option_index"] = hammer_option_indices[0]
    result["hammer_option_indices"] = hammer_option_indices
    result["qualifies"] = True
    return result


# ---------------------------------------------------------------------------
# run3 拡張(問い1): 実際にプレイされた Enhanced Hammer を①②③に分類する。
#
# 「ハンマーを PLAY する MAIN 選択」が action に含まれた瞬間(before_obs)を pending として
# 記録しておき、以後の wrapped(obs) 呼び出しのたびに相手(アクティブ+ベンチ)の特殊エネの
# 構成を before_obs と突き合わせ、1枚でも消えていれば「解決した」とみなして分類する
# (`_diff_removed_special_energy`)。相手が実際にどのカードを選んで捨てさせられたかを追跡する
# だけで、ハンマー自身の効果解決(捨てるエネルギーを選ぶ側の意思決定)には一切介入しない。
#
# 分類そのものは evaluate_decision の cond3/cond4 と全く同じ手法
# (`_remove_one_energy_card` で実際に消えた1枚だけを仮想的に除去し、
# `attack_features.resolve_damage` / `damage_prevented` で判定)を再利用する。
# 「特殊エネが付いていれば無効化」のような自前判定はしない。
# ---------------------------------------------------------------------------


def _diff_removed_special_energy(before_obs: Observation, obs: Observation):
    """before_obs から obs までの間に、相手(アクティブ+ベンチ)のいずれかの特殊エネが1枚
    消えていれば ``(defender_pokemon_before, removed_card, is_benched)`` を返す。
    まだ解決していなければ(相手の場の特殊エネ構成に変化がなければ) None を返す
    (pending_hammer の状態機械が「まだ解決待ち」として次の呼び出しまで待つのに使う)。

    Pokemon は ``serial``(対戦内で一意)で同一性を追跡するため、ベンチの並び替えなどが
    間に挟まっても取り違えない。ポケモン自身が場からいなくなっていれば(きぜつ等)そのポケモン
    はスキップする(このタイミングでは起こり得ないはずだが、防御的に無視するだけで例外は投げない)。
    """
    if obs.current is None or before_obs.current is None:
        return None
    you = obs.current.yourIndex
    opp_before = before_obs.current.players[1 - you]
    opp_after = obs.current.players[1 - you]

    before_mons = [(p, False) for p in (opp_before.active or []) if p is not None]
    before_mons += [(p, True) for p in (opp_before.bench or []) if p is not None]

    after_by_serial = {}
    for p in (opp_after.active or []):
        if p is not None:
            after_by_serial[p.serial] = p
    for p in (opp_after.bench or []):
        if p is not None:
            after_by_serial[p.serial] = p

    for mon_before, is_benched in before_mons:
        mon_after = after_by_serial.get(mon_before.serial)
        if mon_after is None:
            continue
        before_specials = {c.serial: c for c in _special_energy_cards(mon_before)}
        after_serials = {c.serial for c in _special_energy_cards(mon_after)}
        missing = sorted(set(before_specials) - after_serials)
        if missing:
            return mon_before, before_specials[missing[0]], is_benched
    return None


def classify_hammer_play(before_obs: Observation, defender_before, removed_card, is_benched: bool) -> dict:
    """1回のハンマープレイを①(1_enables_ko)②(2_unblocks_no_ko)③(3_unrelated)に分類する。

    ①②のいずれも、判定対象は「そのターンの自分のアクティブが Alakazam で Powerful Hand の
    エネルギーが足りている」場合に限る(evaluate_decision の cond2 と同じ条件)。それ以外
    (アクティブが Alakazam でない/エネルギー不足/剥がした相手が相手のアクティブでない)は、
    このモデル化の範囲では「今の攻撃とは無関係」として③に分類する(attacker_ready /
    target_is_active フラグで判別できるようにしておく。詳細は報告の「判断に迷った点」参照)。
    """
    state = before_obs.current
    you = state.yourIndex
    player = state.players[you]
    opponent = state.players[1 - you]
    active = player.active[0] if (player.active and player.active[0] is not None) else None

    opp_active_serial = None
    if opponent.active and opponent.active[0] is not None:
        opp_active_serial = opponent.active[0].serial

    out: dict = {
        "category": "3_unrelated",
        "attacker_ready": False,
        "target_is_active": (not is_benched) and opp_active_serial == defender_before.serial,
        "removed_energy_name": card_cache.get_card(removed_card.id).name,
        "defender_name": card_cache.get_card(defender_before.id).name,
        "defender_is_benched": is_benched,
    }

    if active is None or active.id != ALAKAZAM_ID:
        return out
    attack = card_cache.get_attack(POWERFUL_HAND_ATTACK_ID)
    if not energy_requirements.is_energy_sufficient(attack, active.energies):
        return out
    out["attacker_ready"] = True
    if not out["target_is_active"]:
        return out

    defender_card = card_cache.get_card(defender_before.id)
    hand_count = player.handCount
    defender_side_before = [p for p in (opponent.active or []) if p is not None] + list(opponent.bench)
    damage_is_effect = attack_features.damage_is_effect_based(attack)

    current_damage = attack_features.resolve_damage(
        attack, active, defender_card.weakness, defender_card.resistance, hand_count,
        defender=defender_before, defender_side_pokemon=defender_side_before,
        defender_is_benched=False, damage_is_effect=damage_is_effect,
    )
    damage_prevented_before = attack_features.damage_prevented(
        active, defender_before, defender_side_before, False, damage_is_effect,
    )

    hyp_defender = _remove_one_energy_card(defender_before, removed_card)
    hyp_side = [hyp_defender if p is defender_before else p for p in defender_side_before]
    hyp_damage = attack_features.resolve_damage(
        attack, active, defender_card.weakness, defender_card.resistance, hand_count,
        defender=hyp_defender, defender_side_pokemon=hyp_side,
        defender_is_benched=False, damage_is_effect=damage_is_effect,
    )
    damage_prevented_after = attack_features.damage_prevented(
        active, hyp_defender, hyp_side, False, damage_is_effect,
    )

    out.update(
        current_damage=current_damage,
        hypothetical_damage=hyp_damage,
        defender_hp=defender_before.hp,
        damage_prevented_before=damage_prevented_before,
        damage_prevented_after=damage_prevented_after,
    )

    if current_damage < defender_before.hp and hyp_damage >= defender_before.hp:
        out["category"] = "1_enables_ko"
    elif damage_prevented_before and not damage_prevented_after:
        out["category"] = "2_unblocks_no_ko"
    else:
        out["category"] = "3_unrelated"
    return out


# ---------------------------------------------------------------------------
# run3 拡張(問い2): 「逃した」決定点で、ハンマー後 vs 実際に選んだ手の後の局面を
# handcrafted / value の両葉評価器で採点する(1手先だけ。ロールアウトはしない)。
# ---------------------------------------------------------------------------


def _search_begin_from_hidden_state(obs: Observation, hidden_state: dict):
    """pipeline.py の `_begin` / pimc.py の `_begin` と同じ組み立て方(チーム慣例どおり複製)。"""
    return search_begin(
        obs,
        hidden_state["your_deck"],
        hidden_state["your_prize"],
        hidden_state["opponent_deck"],
        hidden_state["opponent_prize"],
        hidden_state["opponent_hand"],
        hidden_state["opponent_active"],
    )


def build_run3_evaluators() -> tuple[dict, str]:
    """{"handcrafted": ..., "value": ...} の評価器2つと、value が実際に使った重みパスを返す。

    handcrafted は `leaf_eval.build_evaluator({"kind": "handcrafted"})`(abl_5_full の既定と同じ)。
    value は既定(`sample_submission/ptcg_ai/learning/value_weights.json`、7月版)ではなく、
    配線後の特徴量で作り直した最新重み(`_WIRED_VALUE_WEIGHTS_PATH`)を
    `ValueModelEvaluator(model=ValueModel(weights_path=...))` として直接組み立てて使う
    (`ptcg_ai/search/leaf_eval.py` 自体は変更しない。build_evaluator に重みパスを渡す口が
    無いため、呼び出し側でモデルを組み立てて注入する)。ファイルが無ければ既定の
    `build_evaluator({"kind": "value"})`(7月版重み)にフォールバックする。
    """
    handcrafted = leaf_eval.build_evaluator({"kind": "handcrafted"})
    if _WIRED_VALUE_WEIGHTS_PATH.is_file():
        value_model = ValueModel(weights_path=str(_WIRED_VALUE_WEIGHTS_PATH))
        value_evaluator = leaf_eval.ValueModelEvaluator(model=value_model)
        weights_used = str(_WIRED_VALUE_WEIGHTS_PATH)
    else:
        value_evaluator = leaf_eval.build_evaluator({"kind": "value"})
        weights_used = "sample_submission/ptcg_ai/learning/value_weights.json (既定・7月版。" \
            "wired weights が見つからずフォールバック)"
    return {"handcrafted": handcrafted, "value": value_evaluator}, weights_used


def _find_energy_card_option(select, obs_ref: Observation, target_serial: int) -> int | None:
    """select.option の中から、``obs_ref``(その時点ではまだ効果未解決の局面)を基準に
    「捨てる特殊エネとして target_serial のカードを指す ENERGY_CARD 型の選択肢」を探す。

    Enhanced Hammer(1081)の効果解決(「捨てる特殊エネを選ぶ」follow-up select)は
    OptionType.ENERGY_CARD(area/index/playerIndex/energyIndex で対象ポケモンとその
    energyCards内の位置を指す)で提示される想定。option 自体はカードIDやserialを直接
    持たないため、``obs_ref`` 側の実際のポケモンの energyCards を参照して逆引きする。
    一致する選択肢が無ければ None(呼び出し側が先頭選択肢にフォールバックする)。
    """
    if select is None or obs_ref.current is None:
        return None
    state = obs_ref.current
    for i, option in enumerate(select.option):
        if option.type != OptionType.ENERGY_CARD:
            continue
        if option.playerIndex is None or option.area is None or option.index is None:
            continue
        if option.energyIndex is None:
            continue
        try:
            area_player = state.players[option.playerIndex]
            if option.area == AreaType.ACTIVE:
                mon = area_player.active[option.index]
            elif option.area == AreaType.BENCH:
                mon = area_player.bench[option.index]
            else:
                mon = None
            if mon is None:
                continue
            card = mon.energyCards[option.energyIndex]
        except (IndexError, AttributeError, TypeError):
            continue
        if card.serial == target_serial:
            return i
    return None


def _resolve_one_action(
    root, first_selection: list[int], target_energy_serial: int | None, obs_ref: Observation,
    max_follow_ups: int = 5,
):
    """root から1つの MAIN 選択(first_selection)を適用し、その効果に伴う follow-up select
    (MAIN以外の中間選択、例: ハンマーで「どの特殊エネを捨てるか」)を、MAIN選択に戻るか
    決着するまで機械的に進めて最終ノードを返す(ロールアウトはしない。あくまで「その1手」の
    効果解決の範囲に限る)。

    ``target_energy_serial`` が指定されていれば、ENERGY_CARD型の follow-up 選択肢のうち
    そのカードを指すものを優先して選ぶ(ハンマーの捨てエネ選択を、cond4 が KO 可能と
    判定した特定の1枚に正しく解決させるため)。指定が無い/一致が見つからない場合は
    先頭の合法選択肢で機械的に進める(実際のエージェントの意思決定を再現するものではない、
    という限界がある。詳細は報告の「判断に迷った点」参照)。

    通過した中間ノードは順次 release する。最終的に返すノードは呼び出し側の責務で release する。
    """
    try:
        node = search_step(root.searchId, first_selection)
    except ValueError:
        return None

    steps = 0
    while steps < max_follow_ups:
        obs = node.observation
        state = obs.current
        if state is None or state.result != -1:
            break
        select = obs.select
        if select is None or not select.option or select.type == SelectType.MAIN:
            break

        idx = 0
        if target_energy_serial is not None:
            found = _find_energy_card_option(select, obs_ref, target_energy_serial)
            if found is not None:
                idx = found

        try:
            next_node = search_step(node.searchId, [idx])
        except ValueError:
            break
        try:
            search_release(node.searchId)
        except Exception:  # noqa: BLE001
            pass
        node = next_node
        steps += 1

    return node


def compute_leaf_eval_comparison(
    obs: Observation,
    hammer_option_indices: list[int],
    removed_energy_card,
    action: list[int],
    evaluators: dict,
) -> dict:
    """「逃した」決定点1件について、ハンマー後/実際手後の1手先局面を両評価器で採点する。

    `search_begin` の隠れ情報は abl_5_full の pipeline 設定(`hidden_state_source: "estimated"`)
    と同じ組み立て方(`search_adapter.to_search_begin_kwargs` + `match_context` の推定)を使う
    (`ml_policy_agent._try_pipeline` の estimated 分岐と同じ呼び出し)。ロールアウトはせず、
    1手進めた直後の局面だけを評価する(ただし、その1手の効果解決に伴う follow-up select は
    `_resolve_one_action` で機械的に進める。例えばハンマーは「PLAYする」選択の直後は
    まだ「どの特殊エネを捨てるか」を選ぶ前の局面で、それを進めないと特殊エネが実際には
    消えておらず、評価が「何も変わっていない」局面のままになってしまうため)。
    search_begin/search_step は本物の意思決定(`agent_fn(obs)` 呼び出し)の**後**に呼ぶので、
    実際のプレイには一切影響しない。
    """
    result: dict = {"ok": False}
    state = obs.current
    if state is None or obs.select is None or not hammer_option_indices:
        return result
    me = state.yourIndex
    hammer_idx = hammer_option_indices[0]
    target_serial = removed_energy_card.serial if removed_energy_card is not None else None

    try:
        hidden_state = search_adapter.to_search_begin_kwargs(
            match_context.get_own_state(me), match_context.get_opponent_state(me), obs,
        )
    except Exception:  # noqa: BLE001
        return result

    try:
        root = _search_begin_from_hidden_state(obs, hidden_state)
    except Exception:  # noqa: BLE001
        return result

    scores_hammer = None
    scores_actual = None
    try:
        child_hammer = _resolve_one_action(root, [hammer_idx], target_serial, obs)
        if child_hammer is not None:
            try:
                s = child_hammer.observation.current
                scores_hammer = {name: ev.evaluate(s, me) for name, ev in evaluators.items()}
            finally:
                try:
                    search_release(child_hammer.searchId)
                except Exception:  # noqa: BLE001
                    pass

        child_actual = _resolve_one_action(root, list(action), None, obs)
        if child_actual is not None:
            try:
                s = child_actual.observation.current
                scores_actual = {name: ev.evaluate(s, me) for name, ev in evaluators.items()}
            finally:
                try:
                    search_release(child_actual.searchId)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        try:
            search_release(root.searchId)
        except Exception:  # noqa: BLE001
            pass
        try:
            search_end()
        except Exception:  # noqa: BLE001
            pass

    if scores_hammer is None or scores_actual is None:
        return result

    result["ok"] = True
    for name in evaluators:
        result[f"{name}_hammer"] = scores_hammer[name]
        result[f"{name}_actual"] = scores_actual[name]
        result[f"{name}_diff"] = scores_hammer[name] - scores_actual[name]
    return result


# ---------------------------------------------------------------------------
# run2 拡張①: どの経路(_try_lethal / _try_pipeline / _try_attack_hybrid / policy 単体)が
# 行動を決めたかを observe する。ml_policy_agent 本体は変更せず、モジュール属性を薄いラッパで
# 置き換えて戻り値を記録するだけ(monkeypatch、判断には介入しない)。
# ---------------------------------------------------------------------------

_NOT_CALLED = object()  # このターンにまだ呼ばれていないことを示す番人(None と区別するため)


class RouteRecorder:
    """1手分の意思決定で `_try_lethal`/`_try_pipeline`/`_try_attack_hybrid` がそれぞれ
    呼ばれたか・何を返したかを記録する。`reset()` を毎手番(agent_fn(obs) 呼び出し直前)に
    呼ぶことで、前の手の値が混ざらないようにする。
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.lethal = _NOT_CALLED
        self.pipeline = _NOT_CALLED
        self.attack_hybrid = _NOT_CALLED


def _wrap_tracked(attr_name: str, orig_fn, recorder: "RouteRecorder"):
    """`orig_fn(obs, config=None)` をそのまま呼び、戻り値を recorder に記録するだけの
    薄いラッパを返す。判断には一切介入せず、orig_fn の戻り値をそのまま返す。
    """

    def wrapped(obs, config=None):
        result = orig_fn(obs, config=config)
        setattr(recorder, attr_name, result)
        return result

    return wrapped


def install_route_instrumentation(recorder: "RouteRecorder") -> None:
    """`ml_policy_agent` モジュールの `_try_lethal`/`_try_pipeline`/`_try_attack_hybrid` を
    monkeypatch し、戻り値を observe する薄いラッパに置き換える。`_select_action` はこれらを
    モジュールのグローバル名として毎回名前解決して呼ぶため、この置き換えだけで意思決定ロジック
    そのものを一切変えずに経路を観測できる。プロセスごとに一度だけ適用する(多重ラップ防止。
    `ProcessPoolExecutor` の各ワーカープロセスは独立した Python インタプリタなので、
    プロセスをまたいだ汚染は起きない)。
    """
    if getattr(ml_policy_agent, "_hammer_diag_route_instrumented", False):
        return
    ml_policy_agent._try_lethal = _wrap_tracked("lethal", ml_policy_agent._try_lethal, recorder)
    ml_policy_agent._try_pipeline = _wrap_tracked("pipeline", ml_policy_agent._try_pipeline, recorder)
    ml_policy_agent._try_attack_hybrid = _wrap_tracked(
        "attack_hybrid", ml_policy_agent._try_attack_hybrid, recorder
    )
    ml_policy_agent._hammer_diag_route_instrumented = True


def classify_route(recorder: "RouteRecorder") -> str:
    """今回の意思決定を最終的に決めた経路を返す。

    `_select_action` の呼び出し順(lethal -> pipeline -> attack_hybrid -> policy 単体)を
    そのまま反映する。`_try_attack_plan` は abl_5_full では config 未指定のため常に無効
    (baseline_action をそのまま返す)なので経路には含めない。
    """
    if recorder.lethal is not _NOT_CALLED and recorder.lethal is not None:
        return "lethal"
    if recorder.pipeline is not _NOT_CALLED and recorder.pipeline is not None:
        return "pipeline"
    if recorder.attack_hybrid is not _NOT_CALLED and recorder.attack_hybrid is not None:
        return "attack_hybrid"
    return "policy_only"


# ---------------------------------------------------------------------------
# run2 拡張② + ③: pipeline.search() が深い探索(Step3-5)を行わずに終わった理由と、
# ハンマーの方策(Step2)順位を診断側で再現する。pipeline.py 自体は一切変更しない。
#
# pipeline.search() が None を返す分岐(コードを読んで洗い出したもの、pipeline.py 2026-08 時点):
#   (a) config["enabled"] が False                          … abl_5_full は常に True のため出現しない
#       (`_try_pipeline` 側で pipeline_config.get("enabled") を先にチェックしており、False なら
#        search() 自体が一度も呼ばれない。abl_5_full は "pipeline.enabled": true 固定)
#   (b) obs / obs.select / obs.current が None              … `_try_pipeline` が事前に
#       obs.current is None or obs.select is None をガードしているため出現しない
#   (c) select.type != MAIN / select.maxCount != 1 / 選択肢なし
#   (d) state.result != -1 (決着済み) / 自分の手番でない (_is_my_turn が False)
#   (e) context.get("policy_model") が None                  … `_try_pipeline` が必ず
#       `_get_model(config)`(常に非None)を渡すため出現しない
#   (f) scores が空、または長さが選択肢数と不一致(score_options の例外を含む)
#   (g) probs[ranked[0]] >= top1_shortcut_prob で top1 を返す分岐: 技術的には None を返す
#       わけではない(top1 が合法なら top1 をそのまま返す)が、Step3-5 の深い探索は一度も
#       走らない。top1 が非合法になるケースは maxCount==1 かつ minCount<=1 の下では
#       起こり得ない(top1 は必ず合法)ため、実質「探索スキップして即返す」分岐として扱う。
#   (h) factory = _hidden_state_factory(context) が None      … `_try_pipeline` は常に
#       lambda を渡すため出現しない
#   (i) N決定化 × 候補の集計 `scored` が空(全世界・全候補で search_begin/search_step が
#       失敗した、またはロールアウトが評価不能だった)
#   (j) 最終選択 best が非合法                                … (g) と同じ理由で実質起こり得ない
#   (k) search() 本体を包む except Exception: return None (未知の例外)
#
# (i)/(j)/(k) は実際に search_begin/search_step を再実行しないと外部から区別できない
# (pipeline.py に計測コードを入れることは禁止されているため)。診断側では
# 「(c)(d)(f)(g) のいずれにも該当せず、かつ実際の _try_pipeline の戻り値が None だった」場合を
# まとめて "full_search_no_result" として扱う(scored空/例外/非合法選択のいずれかだが、
# pipeline.py を変更せずにはこれ以上分解できない)。
# ---------------------------------------------------------------------------


def compute_step2_and_pipeline_diagnosis(
    obs: Observation,
    hammer_option_indices: list[int],
    recorder: "RouteRecorder",
    contender_cfg: dict,
    pipeline_cfg_merged: dict,
) -> dict:
    """②(pipeline early-return 理由)と③(ハンマーの方策順位)をまとめて計算する。

    `agent_fn(obs)` の呼び出しが終わった**後**に呼ぶこと(`match_context` が今回の obs で
    更新済みの状態で `policy_model.score_options` を呼ぶことで、実際の `_try_pipeline` 呼び出し
    が見た状態と一致させる)。ここでの再計算はすべて読み取り専用で、本番の意思決定には
    一切影響しない。

    `hammer_option_indices`: Enhanced Hammer をプレイする選択肢の index のリスト(手札に複数枚
    あれば複数)。順位/top_k判定は「最も有利な(=最も上位の)コピー」を基準にする
    (どれか1枚でも上位候補に入っていれば「候補に入っていた」と数えるのが自然なため)。
    """
    select = obs.select
    state = obs.current

    result: dict = {
        "route": classify_route(recorder),
        "hammer_rank": None,        # 1-based。方策スコア降順で何位か(複数コピーがあれば最良の順位)
        "n_options": len(select.option) if select is not None else None,
        "top1_prob": None,          # softmax 後の top1 確率(ショートカット判定に使う値)
        "hammer_in_topk": None,     # top_k(既定4)候補にいずれかのコピーが入っていたか
        "top_k": int(pipeline_cfg_merged.get("top_k", 4)),
        "pipeline_reason": None,
    }

    # --- ③ pipeline.search() Step2 と同じスコアリングを診断側で再現する ---
    policy_model = ml_policy_agent._get_model(contender_cfg)
    try:
        factory = ml_policy_agent._model_hidden_state_factory(obs, contender_cfg)
        deadline = time.perf_counter() + 1.0
        scores = policy_model.score_options(obs, factory, deadline)
    except Exception:  # noqa: BLE001
        scores = []

    scores_ok = bool(scores) and select is not None and len(scores) == len(select.option)
    top1_shortcut = False

    if scores_ok:
        probs = pipeline_search._softmax(scores)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_by_index = {opt_i: pos + 1 for pos, opt_i in enumerate(ranked)}
        result["hammer_rank"] = min(rank_by_index[i] for i in hammer_option_indices)
        result["top1_prob"] = probs[ranked[0]]
        candidate_indices = pipeline_search._select_candidate_indices(select, ranked, pipeline_cfg_merged)
        result["hammer_in_topk"] = any(i in candidate_indices for i in hammer_option_indices)
        top1_shortcut = probs[ranked[0]] >= pipeline_cfg_merged["top1_shortcut_prob"]

    # --- ② pipeline が深い探索を行わなかった理由 ---
    if result["route"] == "lethal":
        # _try_lethal が先に発火したため、_try_pipeline はこのターン一度も呼ばれていない。
        result["pipeline_reason"] = "not_reached_lethal_fired"
    elif select is None or select.maxCount != 1 or not select.option:
        result["pipeline_reason"] = "select_shape_maxcount_or_empty"
    elif state is None or state.result != -1 or not pipeline_search._is_my_turn(state, state.yourIndex):
        result["pipeline_reason"] = "not_my_turn_or_game_over"
    elif not scores_ok:
        result["pipeline_reason"] = "scores_empty_or_mismatched"
    elif top1_shortcut:
        result["pipeline_reason"] = "top1_shortcut"
    elif recorder.pipeline is not _NOT_CALLED and recorder.pipeline is not None:
        result["pipeline_reason"] = "full_search_fired"
    else:
        result["pipeline_reason"] = "full_search_no_result"

    return result


def _resolve_card_name_in_area(obs: Observation, area, index) -> str | None:
    """行動の説明表示用: area/index からカード名/ポケモン名を引く(失敗時 None)。"""
    if area is None or index is None:
        return None
    state = obs.current
    player = state.players[state.yourIndex]
    try:
        if area == AreaType.HAND:
            return card_cache.get_card(player.hand[index].id).name
        if area == AreaType.ACTIVE:
            pokemon = player.active[index] if index < len(player.active) else None
            return card_cache.get_card(pokemon.id).name if pokemon else None
        if area == AreaType.BENCH:
            pokemon = player.bench[index] if index < len(player.bench) else None
            return card_cache.get_card(pokemon.id).name if pokemon else None
        if area == AreaType.DISCARD:
            return card_cache.get_card(player.discard[index].id).name
    except Exception:  # noqa: BLE001
        return None
    return None


def describe_action(obs: Observation, action: list[int]) -> str:
    """エージェントが実際に選んだ手を人間可読な文字列にする(診断ログ用、判断には使わない)。"""
    if not action:
        return "(empty action)"
    option = obs.select.option[action[0]]
    type_name = OptionType(option.type).name
    detail = None
    try:
        if option.type == OptionType.PLAY:
            player = obs.current.players[obs.current.yourIndex]
            detail = card_cache.get_card(player.hand[option.index].id).name
        elif option.type == OptionType.ATTACK:
            detail = card_cache.get_attack(option.attackId).name
        elif option.type in (OptionType.ATTACH, OptionType.EVOLVE, OptionType.ABILITY, OptionType.DISCARD):
            detail = _resolve_card_name_in_area(obs, option.area, option.index)
    except Exception:  # noqa: BLE001
        detail = None
    text = type_name if detail is None else f"{type_name}({detail})"
    if len(action) > 1:
        text += f" (+{len(action) - 1} more selections)"
    return text


# ---------------------------------------------------------------------------
# 観測用ラッパー: 本物の agent(obs) をそのまま呼び、結果だけを記録する
# ---------------------------------------------------------------------------


class DiagState:
    """1試合分の集計状態。reset() でゲーム開始時にクリアする。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.counts: Counter = Counter()
        self.records: list[dict] = []
        # run3 拡張(問い1)用。
        self.seq: int = 0                       # このゲーム内の MAIN 以外も含む通し番号
        self.pending_hammer: dict | None = None  # 解決待ちのハンマープレイ(state機械)
        self.hammer_plays: list[dict] = []       # 分類済みの全ハンマープレイ(①②③)
        self.lethal_via_hammer_events: list[dict] = []  # cond4 成立点(cond5の成否に関わらず全件)


def instrument(
    agent_fn,
    state: DiagState,
    route_recorder: "RouteRecorder",
    contender_cfg: dict,
    pipeline_cfg_merged: dict,
    evaluators: dict,
):
    """本番の agent(obs) 呼び出し可能オブジェクトを薄くラップする。

    意思決定には一切介入しない: obs を見て条件判定 -> route_recorder をリセット ->
    本物の agent_fn(obs) を呼ぶ(この中で `_try_lethal`/`_try_pipeline`/`_try_attack_hybrid`
    が呼ばれ、route_recorder に戻り値が記録される)-> 返ってきた action と route_recorder を
    見て記録するだけ。action はそのまま(改変せず)呼び出し元に返す。

    ②③の再計算(compute_step2_and_pipeline_diagnosis)は agent_fn(obs) の**後**に行う
    (match_context が今回の obs で更新済みの状態で policy_model.score_options を呼ぶため)。

    run3 拡張(問い1): 呼び出しのたびに、まず「解決待ちのハンマープレイ」(state.pending_hammer)
    が今回の obs で解決した(相手の場の特殊エネ構成が変化した)かを確認する
    (`_diff_removed_special_energy`)。解決していれば分類して `state.hammer_plays` に積む。
    その後 agent_fn(obs) を呼び、返ってきた action が新たにハンマーの PLAY を含んでいれば、
    新しい pending として記録する(次回以降の呼び出しで解決を待つ)。
    """

    def wrapped(obs: Observation) -> list[int]:
        state.seq += 1
        seq = state.seq

        # --- run3 問い1: 解決待ちのハンマープレイがあれば、今回の obs で解決したか確認する ---
        if state.pending_hammer is not None:
            diff = _diff_removed_special_energy(state.pending_hammer["before_obs"], obs)
            if diff is not None:
                defender_before, removed_card, is_benched = diff
                cls = classify_hammer_play(
                    state.pending_hammer["before_obs"], defender_before, removed_card, is_benched
                )
                cls["seq"] = state.pending_hammer["seq"]
                cls["turn"] = state.pending_hammer["turn"]
                state.hammer_plays.append(cls)
                state.pending_hammer = None

        info = None
        if obs.select is not None and obs.current is not None and obs.select.type == SelectType.MAIN:
            info = evaluate_decision(obs)

        # --- run3 問い1: cond4(剥がせば倒せる)成立点を、cond5(手札にハンマーがある)の
        # 成否に関わらず全件記録する(「手札にハンマーが無かった」試合を後段で突き合わせるため)。
        if info is not None and info.get("cond4"):
            state.lethal_via_hammer_events.append({
                "seq": seq,
                "turn": obs.current.turn,
                "had_hammer_in_hand": bool(info.get("qualifies")),
            })

        route_recorder.reset()
        action = agent_fn(obs)

        # --- run3 問い1: 今回の action が新たにハンマーの PLAY を含んでいれば pending 開始 ---
        if obs.select is not None and obs.select.type == SelectType.MAIN:
            hammer_indices = set(_hammer_play_indices(obs))
            if hammer_indices and (set(action) & hammer_indices):
                if state.pending_hammer is not None:
                    # 前回の pending が解決しないまま次のハンマーが来た(想定外・安全弁)。
                    # 診断上は起こらないはず(1枚プレイ→即解決)だが、カウントだけ残して
                    # 前回分は捨てる(解決不能な情報を無理に分類しない)。
                    state.counts["hammer_pending_unresolved"] += 1
                state.pending_hammer = {"before_obs": obs, "seq": seq, "turn": obs.current.turn}

        if info is not None and info["cond2"]:
            state.counts["cond2"] += 1
            if info["cond3"]:
                state.counts["cond2_3"] += 1
                if info["cond4"]:
                    state.counts["cond2_3_4"] += 1
                    if info["qualifies"]:
                        state.counts["qualifying"] += 1
                        # ハンマーは同名複数枚が手札にあり得る(ACE SPEC ではない通常Item)。
                        # どのコピーを選んでも "撃った" ことに変わりないので、
                        # hammer_option_indices(全コピーのオプション index)との積集合で判定する
                        # (target_option_index 単体だけを見ると、別コピーを選んだ場合に
                        # "使わなかった" と誤判定してしまうバグが run1 の判定にあった)。
                        used_hammer = bool(set(info["hammer_option_indices"]) & set(action))
                        if used_hammer:
                            state.counts["used_hammer"] += 1
                        else:
                            state.counts["missed_hammer"] += 1
                        chosen_desc = describe_action(obs, action)
                        chosen_type = (
                            OptionType(obs.select.option[action[0]].type).name if action else "(empty)"
                        )
                        diag = compute_step2_and_pipeline_diagnosis(
                            obs, info["hammer_option_indices"], route_recorder,
                            contender_cfg, pipeline_cfg_merged,
                        )
                        state.counts[f"route_{diag['route']}"] += 1
                        state.counts[f"pipereason_{diag['pipeline_reason']}"] += 1
                        record = {
                            "used_hammer": used_hammer,
                            "hand_count": info["hand_count"],
                            "defender_name": info["defender_name"],
                            "defender_hp": info["defender_hp"],
                            "defender_max_hp": info["defender_max_hp"],
                            "special_energy_names": info["special_energy_names"],
                            "removed_energy_name": info["removed_energy_name"],
                            "current_damage": info["current_damage"],
                            "hypothetical_damage": info["hypothetical_damage"],
                            "chosen_option_type": chosen_type,
                            "chosen_desc": chosen_desc,
                            "route": diag["route"],
                            "pipeline_reason": diag["pipeline_reason"],
                            "hammer_rank": diag["hammer_rank"],
                            "n_options": diag["n_options"],
                            "top1_prob": diag["top1_prob"],
                            "hammer_in_topk": diag["hammer_in_topk"],
                            "top_k": diag["top_k"],
                        }
                        if not used_hammer:
                            # run3 問い2: 「逃した」決定点だけ、ハンマー後/実際手後の1手先局面を
                            # handcrafted/value の両評価器で比較する。removed_energy_card は
                            # cond4 が「これを剥がせばKOできる」と判定した実カード(serial)で、
                            # ハンマーの捨てエネ follow-up select をそれに解決させるために使う。
                            record["q2"] = compute_leaf_eval_comparison(
                                obs, info["hammer_option_indices"], info["removed_energy_card"],
                                action, evaluators,
                            )
                        state.records.append(record)
        return action

    return wrapped


# ---------------------------------------------------------------------------
# 対戦ドライバ(league/run_league.py の build_agent / read_deck_csv_file / play_match を再利用)
# ---------------------------------------------------------------------------

_WORKER_STATE: dict = {}


def _build_contender_agent() -> tuple:
    """本番構成のまま(build_agent 経由)alakazam 側 agent を作り、観測ラッパーで包む。

    ついでに `_try_lethal`/`_try_pipeline`/`_try_attack_hybrid` の経路計測(run2 拡張①)を
    このプロセスに1度だけインストールする。`contender_cfg`/`pipeline_cfg_merged` は
    `_build_contender_agent` 内でだけ読み込んだ「診断用の別インスタンス」であり、
    `run_league.build_agent` が内部で作る本番用 config(cfg)とは別オブジェクトだが、
    どちらも同じ config ファイル(`abl_5_full.json`)を読むだけ(変更を加えない)なので
    内容は一致する。
    """
    diag_state = DiagState()
    route_recorder = RouteRecorder()
    install_route_instrumentation(route_recorder)
    contender_cfg = load_ptcg_config(CONTENDER_CONFIG)
    pipeline_cfg_merged = {**pipeline_search.DEFAULTS, **(contender_cfg.get("pipeline") or {})}
    evaluators, _weights_used = build_run3_evaluators()  # run3 問い2用。プロセスごとに1回だけ構築
    raw_agent = run_league.build_agent("ml_policy", None, CONTENDER_CONFIG)
    wrapped = instrument(raw_agent, diag_state, route_recorder, contender_cfg, pipeline_cfg_merged, evaluators)
    return wrapped, diag_state


def _worker_init(opponent_names: list[str]) -> None:
    os.chdir(_SUB_DIR)
    agent_a, diag_state = _build_contender_agent()
    _WORKER_STATE["agent_a"] = agent_a
    _WORKER_STATE["diag_state"] = diag_state
    _WORKER_STATE["deck_a"] = run_league.read_deck_csv_file(str(ALAKAZAM_DECK_PATH))
    opponents = {}
    for name in opponent_names:
        deck_path, weights_path = OPPONENT_SPECS[name]
        agent_b = run_league.build_agent("ml_policy", str(weights_path), OPPONENT_CONFIG)
        deck_b = run_league.read_deck_csv_file(str(deck_path))
        opponents[name] = (agent_b, deck_b)
    _WORKER_STATE["opponents"] = opponents


def _play_one(task: tuple[str, int, int]) -> dict:
    opponent_name, game_index, seed = task
    s = _WORKER_STATE
    diag_state: DiagState = s["diag_state"]
    diag_state.reset()

    agent_b, deck_b = s["opponents"][opponent_name]
    a_is_player0 = (game_index % 2 == 0)
    if a_is_player0:
        agent0, agent1 = s["agent_a"], agent_b
        deck0, deck1 = s["deck_a"], deck_b
    else:
        agent0, agent1 = agent_b, s["agent_a"]
        deck0, deck1 = deck_b, s["deck_a"]

    result = run_league.play_match(agent0, agent1, deck0, deck1, seed=seed)
    a_player_index = 0 if a_is_player0 else 1
    winner_is_a = (result.winner == a_player_index) if result.winner is not None else None

    if diag_state.pending_hammer is not None:
        # ゲームが終わるまでに解決が観測できなかった(例: ハンマー自体で試合が終わった等)。
        # 分類不能なので捨てるが、頻度が分かるようにカウントだけ残す。
        diag_state.counts["hammer_pending_at_game_end"] += 1

    return {
        "opponent": opponent_name,
        "game_index": game_index,
        "seed": seed,
        "error": result.error,
        "winner_is_a": winner_is_a,
        "turns": result.turns,
        "counts": dict(diag_state.counts),
        "records": diag_state.records,
        "hammer_plays": diag_state.hammer_plays,
        "lethal_via_hammer_events": diag_state.lethal_via_hammer_events,
    }


HAMMER_DECK_COPIES = 4  # alakazam/01.csv に積まれた Enhanced Hammer(1081) の総枚数


def analyze_q1(all_game_hammer_data: list[dict]) -> dict:
    """run3 問い1: ハンマープレイの①②③分類と、「手札にハンマーが無かった」試合の突き合わせ。

    `all_game_hammer_data` の各要素は1試合分の
    ``{"hammer_plays": [...], "lethal_via_hammer_events": [...]}``(いずれも `seq` で時系列順)。

    「倒せたのにハンマー無し」局面が発生した試合について、**その試合内で最初に発生した
    そのような局面**を基準点とし(1試合に複数回起きる場合は最初の1回を代表とする。理由は
    報告の「判断に迷った点」参照)、それより前(`seq` が小さい)に撃たれたハンマーのうち
    ①(1_enables_ko)以外の枚数・そこまでに撃った総枚数(deck全4枚のうち何枚を消費済みか)
    を数える。
    """
    all_plays = [p for g in all_game_hammer_data for p in g["hammer_plays"]]
    category_counts = Counter(p["category"] for p in all_plays)

    # ③(打点に無関係)の内訳。attacker_ready=False(その瞬間 Alakazam のエネルギーが
    # Powerful Hand に足りない/アクティブが Alakazam でない)が③の大半を占め得るため、
    # 「本当に打点と無関係」と「そもそもこのモデル化の対象外」を区別できるようにする。
    unrelated = [p for p in all_plays if p["category"] == "3_unrelated"]
    unrelated_breakdown = {
        "attacker_not_ready": sum(1 for p in unrelated if not p["attacker_ready"]),
        "target_not_active": sum(1 for p in unrelated if p["attacker_ready"] and not p["target_is_active"]),
        "computed_and_unrelated": sum(1 for p in unrelated if p["attacker_ready"] and p["target_is_active"]),
    }

    games_with_miss = []
    for g in all_game_hammer_data:
        no_hand_events = [e for e in g["lethal_via_hammer_events"] if not e["had_hammer_in_hand"]]
        if no_hand_events:
            games_with_miss.append((g, no_hand_events))

    n_games = len(games_with_miss)
    n_with_prior_non1 = 0
    prior_non1_counts: list[int] = []
    n_all_used_up = 0
    for g, events in games_with_miss:
        first_seq = min(e["seq"] for e in events)
        prior_plays = [p for p in g["hammer_plays"] if p["seq"] < first_seq]
        prior_non1 = [p for p in prior_plays if p["category"] != "1_enables_ko"]
        prior_non1_counts.append(len(prior_non1))
        if prior_non1:
            n_with_prior_non1 += 1
        if len(prior_plays) >= HAMMER_DECK_COPIES:
            n_all_used_up += 1

    return {
        "hammer_play_category_counts": dict(category_counts),
        "unrelated_breakdown": unrelated_breakdown,
        "total_hammer_plays": len(all_plays),
        "n_games_with_no_hand_lethal_miss": n_games,
        "n_games_with_prior_non1_hammer": n_with_prior_non1,
        "avg_prior_non1_hammer_count": (
            sum(prior_non1_counts) / len(prior_non1_counts) if prior_non1_counts else None
        ),
        "n_games_all_copies_used_by_then": n_all_used_up,
        "deck_hammer_copies": HAMMER_DECK_COPIES,
    }


def analyze_q2(qualifying_records: list[dict]) -> dict:
    """run3 問い2: 「逃した」決定点で、ハンマー後 vs 実際手後を handcrafted/value で比較する。"""
    missed_q2 = [
        r["q2"] for r in qualifying_records
        if not r["used_hammer"] and r.get("q2", {}).get("ok")
    ]
    n = len(missed_q2)
    out: dict = {
        "n_missed_decisions": sum(1 for r in qualifying_records if not r["used_hammer"]),
        "n_with_valid_comparison": n,
    }
    for kind in ("handcrafted", "value"):
        diffs = [q2[f"{kind}_diff"] for q2 in missed_q2]
        n_gt = sum(1 for d in diffs if d > 0)
        out[kind] = {
            "n_hammer_gt_actual": n_gt,
            "pct_hammer_gt_actual": (n_gt / n * 100.0) if n else None,
            "avg_diff": (sum(diffs) / n) if n else None,
        }
    return out


def run_diagnostic(
    games_per_opponent: int,
    opponent_names: list[str],
    seed_start: int,
    workers: int,
    log=lambda msg: print(msg, file=sys.stderr),
) -> dict:
    for name in opponent_names:
        if name not in OPPONENT_SPECS:
            raise ValueError(f"unknown opponent: {name!r} (choices: {sorted(OPPONENT_SPECS)})")

    tasks: list[tuple[str, int, int]] = []
    seed = seed_start
    for name in opponent_names:
        for gi in range(games_per_opponent):
            tasks.append((name, gi, seed))
            seed += 1

    n_workers = run_league.resolve_workers(workers)
    t0 = time.time()
    all_results: list[dict] = []

    if n_workers <= 1:
        os.chdir(_SUB_DIR)
        agent_a, diag_state = _build_contender_agent()
        deck_a = run_league.read_deck_csv_file(str(ALAKAZAM_DECK_PATH))
        opponents = {}
        for name in opponent_names:
            deck_path, weights_path = OPPONENT_SPECS[name]
            agent_b = run_league.build_agent("ml_policy", str(weights_path), OPPONENT_CONFIG)
            deck_b = run_league.read_deck_csv_file(str(deck_path))
            opponents[name] = (agent_b, deck_b)
        _WORKER_STATE["agent_a"] = agent_a
        _WORKER_STATE["diag_state"] = diag_state
        _WORKER_STATE["deck_a"] = deck_a
        _WORKER_STATE["opponents"] = opponents
        for i, task in enumerate(tasks, 1):
            all_results.append(_play_one(task))
            if i % 10 == 0 or i == len(tasks):
                log(f"[progress] {i}/{len(tasks)} games done ({time.time() - t0:.1f}s)")
    else:
        log(f"[info] running {len(tasks)} games across {n_workers} worker processes")
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init, initargs=(opponent_names,)
        ) as executor:
            futures = [executor.submit(_play_one, task) for task in tasks]
            done = 0
            for fut in as_completed(futures):
                all_results.append(fut.result())
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    log(f"[progress] {done}/{len(tasks)} games done ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0

    total_counts: Counter = Counter()
    per_opponent_counts: dict[str, Counter] = {name: Counter() for name in opponent_names}
    qualifying_records: list[dict] = []
    all_game_hammer_data: list[dict] = []
    errors = 0
    valid_games = 0
    for rec in all_results:
        if rec["error"] is not None:
            errors += 1
            continue
        valid_games += 1
        c = Counter(rec["counts"])
        total_counts += c
        per_opponent_counts[rec["opponent"]] += c
        for r in rec["records"]:
            r2 = dict(r)
            r2["opponent"] = rec["opponent"]
            r2["game_index"] = rec["game_index"]
            qualifying_records.append(r2)
        all_game_hammer_data.append({
            "opponent": rec["opponent"],
            "game_index": rec["game_index"],
            "hammer_plays": rec["hammer_plays"],
            "lethal_via_hammer_events": rec["lethal_via_hammer_events"],
        })

    _, value_weights_used = build_run3_evaluators()  # レポート用にパスだけ確認(集計処理では未使用)
    contender_cfg_for_report = load_ptcg_config(CONTENDER_CONFIG)  # sweep用: 実際に読まれたtie_epsをレポートに残す

    return {
        "contender_config": CONTENDER_CONFIG,
        "contender_tie_eps": contender_cfg_for_report.get("pipeline", {}).get("tie_eps"),
        "games_requested_per_opponent": games_per_opponent,
        "opponents": opponent_names,
        "games_total": len(tasks),
        "valid_games": valid_games,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "workers": n_workers,
        "total_counts": dict(total_counts),
        "per_opponent_counts": {k: dict(v) for k, v in per_opponent_counts.items()},
        "qualifying_records": qualifying_records,
        "q1_hammer_usage": analyze_q1(all_game_hammer_data),
        "q2_leaf_eval_comparison": analyze_q2(qualifying_records),
        "value_weights_used": value_weights_used,
        # 生データ(再集計・目視確認用)。集計値は上の q1_hammer_usage/q2_leaf_eval_comparison を参照。
        "hammer_plays_by_game": all_game_hammer_data,
    }


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------

_ROUTE_ORDER = ("lethal", "pipeline", "attack_hybrid", "policy_only")
_ROUTE_LABELS = {
    "lethal": "lethal",
    "pipeline": "pipeline",
    "attack_hybrid": "attack_hybrid",
    "policy_only": "policy 単体",
}
_PIPELINE_REASON_ORDER = (
    "not_reached_lethal_fired",
    "select_shape_maxcount_or_empty",
    "not_my_turn_or_game_over",
    "scores_empty_or_mismatched",
    "top1_shortcut",
    "full_search_no_result",
    "full_search_fired",
)
_PIPELINE_REASON_LABELS = {
    "not_reached_lethal_fired": "lethal が先に発火(pipeline未到達)",
    "select_shape_maxcount_or_empty": "maxCount!=1 / 選択肢なし",
    "not_my_turn_or_game_over": "自分の手番でない/決着済み",
    "scores_empty_or_mismatched": "スコア取得失敗/選択肢数と不一致",
    "top1_shortcut": "top1集中でスキップ (prob >= top1_shortcut_prob)",
    "full_search_no_result": "深い探索を実行したが結果を得られず(scored空/例外等)",
    "full_search_fired": "深い探索(Step3-5)が実際に走り、結果を採用",
}


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 == 1 else (xs[mid - 1] + xs[mid]) / 2.0


def _print_route_and_pipeline_breakdown(log, records: list[dict], label: str) -> None:
    """run2 拡張①②③の内訳(経路 / pipeline未起動理由 / ハンマーの方策順位)を出力する。"""
    n = len(records)
    log(f"  [{label}] {n} 件")
    if n == 0:
        log("    (0件のため内訳なし)")
        return

    log("    行動を決めた経路")
    route_counts = Counter(r.get("route") for r in records)
    for key in _ROUTE_ORDER:
        c = route_counts.get(key, 0)
        log(f"      {_ROUTE_LABELS[key]:15s} {c:4d} 件（{c / n * 100:.1f}%）")

    log("    pipeline が深い探索(Step3-5)を行わなかった理由の内訳")
    reason_counts = Counter(r.get("pipeline_reason") for r in records)
    for key in _PIPELINE_REASON_ORDER:
        c = reason_counts.get(key, 0)
        log(f"      {_PIPELINE_REASON_LABELS[key]:40s} {c:4d} 件（{c / n * 100:.1f}%）")

    log("    ハンマーの方策順位（Step2 再現）")
    ranks = [r["hammer_rank"] for r in records if r.get("hammer_rank") is not None]
    if not ranks:
        log("      (順位を計算できたレコードが0件)")
    else:
        rank1 = sum(1 for x in ranks if x == 1)
        rank2_4 = sum(1 for x in ranks if 2 <= x <= 4)
        rank5plus = sum(1 for x in ranks if x >= 5)
        log(f"      1位     {rank1:4d} 件 / 2〜4位  {rank2_4:4d} 件 / 5位以下  {rank5plus:4d} 件"
            f"（順位計算できたのは {len(ranks)}/{n} 件）")
        top_k = records[0].get("top_k", 4)
        in_topk = [r["hammer_in_topk"] for r in records if r.get("hammer_in_topk") is not None]
        n_in = sum(1 for v in in_topk if v)
        if in_topk:
            log(f"      top_k={top_k} に入っていた            {n_in:4d} 件（{n_in / len(in_topk) * 100:.1f}%）")
        probs = [r["top1_prob"] for r in records if r.get("top1_prob") is not None]
        median = _median(probs)
        if median is not None:
            log(f"      top1 確率の中央値                {median:.3f}")


_HAMMER_CATEGORY_ORDER = ("1_enables_ko", "2_unblocks_no_ko", "3_unrelated")
_HAMMER_CATEGORY_LABELS = {
    "1_enables_ko": "① そのターンに倒せるようになった",
    "2_unblocks_no_ko": "② 自分の攻撃の無効化を解いた(倒せない)",
    "3_unrelated": "③ 打点に無関係",
}


def _print_q1_report(log, report: dict) -> None:
    q1 = report["q1_hammer_usage"]
    log("=== 問い1: ハンマーの無駄打ちを測る ===")
    log(f"実際にプレイされた Enhanced Hammer  {q1['total_hammer_plays']} 件")
    cat_counts = q1["hammer_play_category_counts"]
    total = q1["total_hammer_plays"] or 1
    for key in _HAMMER_CATEGORY_ORDER:
        c = cat_counts.get(key, 0)
        log(f"  {_HAMMER_CATEGORY_LABELS[key]:32s} {c:4d} 件（{c / total * 100:.1f}%）")
    ub = q1.get("unrelated_breakdown") or {}
    n_unrelated = cat_counts.get("3_unrelated", 0)
    if n_unrelated:
        log(f"    ③の内訳(合計 {n_unrelated} 件):")
        log(f"      その瞬間 Alakazam の Powerful Hand が使えない(未進化/エネ不足)  "
            f"{ub.get('attacker_not_ready', 0):4d} 件")
        log(f"      アクティブ以外(ベンチ)から剥がした                          "
            f"{ub.get('target_not_active', 0):4d} 件")
        log(f"      上記以外(実際に打点計算しても無関係だった)                    "
            f"{ub.get('computed_and_unrelated', 0):4d} 件")
    unresolved = report["total_counts"].get("hammer_pending_unresolved", 0)
    at_end = report["total_counts"].get("hammer_pending_at_game_end", 0)
    if unresolved or at_end:
        log(f"  (分類不能・除外: 解決待ち中に次のハンマー {unresolved} 件 / "
            f"試合終了まで未解決 {at_end} 件)")
    log("")
    n_games = q1["n_games_with_no_hand_lethal_miss"]
    n_prior = q1["n_games_with_prior_non1_hammer"]
    avg_prior = q1["avg_prior_non1_hammer_count"]
    n_used_up = q1["n_games_all_copies_used_by_then"]
    log('「倒せたのにハンマー無し」が発生した試合数           '
        f"{n_games} 件")
    if n_games:
        log(f"  そのうち、それ以前に①以外のハンマーを撃っていた試合   "
            f"{n_prior} 件（{n_prior / n_games * 100:.1f}%）")
        avg_prior_text = f"{avg_prior:.2f} 枚" if avg_prior is not None else "(計算不能)"
        log(f"  平均で、その時点までに撃っていた①以外の枚数        {avg_prior_text}")
        log(f"  デッキのハンマーは{q1['deck_hammer_copies']}枚。使い切っていた試合            "
            f"{n_used_up} 件（{n_used_up / n_games * 100:.1f}%）")
    else:
        log("  (該当試合が0件)")


def _print_q2_report(log, report: dict) -> None:
    q2 = report["q2_leaf_eval_comparison"]
    log("")
    log("=== 問い2: 葉評価を値ネットに替えたら直りそうか ===")
    log(f"value 評価器に使った重み: {report['value_weights_used']}")
    n = q2["n_with_valid_comparison"]
    log(f"逃した局面 {q2['n_missed_decisions']} 件中、比較可能  {n} 件")
    if n == 0:
        log("  (比較可能な局面が0件のため内訳なし)")
        return
    for kind, label in (("handcrafted", "handcrafted"), ("value", "値ネット")):
        k = q2[kind]
        pct = k["pct_hammer_gt_actual"]
        avg = k["avg_diff"]
        log(
            f"  {label:12s} ハンマー後 > 選択手後 だった件数   "
            f"{k['n_hammer_gt_actual']:4d} ({pct:.1f}%)   平均差 {avg:+.3f}"
        )


def print_report(report: dict, log=print) -> None:
    tc = report["total_counts"]
    n_qual = tc.get("qualifying", 0)
    used = tc.get("used_hammer", 0)
    missed = tc.get("missed_hammer", 0)

    log(f"=== hammer diagnostic: alakazam(ml_policy/{CONTENDER_CONFIG}) vs {', '.join(report['opponents'])} ===")
    log(f"tie_eps(pipeline)={report.get('contender_tie_eps')!r} (config={CONTENDER_CONFIG}; "
        f"PTCG_AI_ML_CONFIG env override確認用)")
    log(
        f"games: {report['valid_games']}/{report['games_total']} valid "
        f"({report['errors']} errors), elapsed {report['elapsed_seconds']:.1f}s, workers={report['workers']}"
    )
    log("")
    log(f"条件緩和内訳(階層的。cond2 ⊇ cond2&3 ⊇ cond2&3&4 ⊇ qualifying=cond2&3&4&5):")
    log(f"  cond2(Alakazamでハンマー先発可能なエネルギー)              {tc.get('cond2', 0):5d} 件")
    log(f"  cond2&3(+ いまは倒せない)                                  {tc.get('cond2_3', 0):5d} 件")
    log(f"  cond2&3&4(+ 特殊エネ1枚剥がせば倒せる)                     {tc.get('cond2_3_4', 0):5d} 件")
    log(f"  qualifying = cond2&3&4&5(+ 手札にハンマーがありプレイ可能) {n_qual:5d} 件")
    log("")

    log(f"該当決定点            {n_qual} 件（{report['valid_games']} 試合中）")
    if n_qual == 0:
        log("  (該当決定点が0件のため、以下の成功率・内訳は計算できません)")
    else:
        log(f"  ハンマーを使った     {used} 件（{used / n_qual * 100:.1f}%）")
        log(f"  使わなかった        {missed} 件（{missed / n_qual * 100:.1f}%）")
        if missed:
            miss_types = Counter(
                r["chosen_option_type"] for r in report["qualifying_records"] if not r["used_hammer"]
            )
            log("    代わりに選んだ手の内訳（OptionType 別、上位5件）")
            for opt_type, cnt in miss_types.most_common(5):
                log(f"      {opt_type}: {cnt} 件（{cnt / missed * 100:.1f}%）")

    log("")
    log("--- run2拡張: 経路 / pipeline未起動理由 / ハンマーの方策順位 ---")
    all_records = report["qualifying_records"]
    missed_records = [r for r in all_records if not r["used_hammer"]]
    used_records = [r for r in all_records if r["used_hammer"]]
    _print_route_and_pipeline_breakdown(log, all_records, "該当決定点 全体")
    log("")
    _print_route_and_pipeline_breakdown(log, missed_records, f"使わなかった{len(missed_records)}件のみ")
    log("")
    _print_route_and_pipeline_breakdown(log, used_records, f"使った{len(used_records)}件のみ(比較用)")

    log("")
    log("--- 相手アーキタイプ別 ---")
    for name, c in report["per_opponent_counts"].items():
        nq = c.get("qualifying", 0)
        u = c.get("used_hammer", 0)
        log(
            f"  {name:22s} qualifying={nq:4d}  used={u:4d}"
            + (f"  ({u / nq * 100:.1f}%)" if nq else "  (n/a)")
        )

    misses = [r for r in report["qualifying_records"] if not r["used_hammer"]]
    if misses:
        log("")
        log(f"--- 使わなかった事例（最大3件、全 {len(misses)} 件中）---")
        for i, r in enumerate(misses[:3], 1):
            log(f"  [{i}] opponent={r['opponent']} game={r['game_index']}")
            log(f"      手札枚数: {r['hand_count']}")
            log(f"      相手: {r['defender_name']}  HP {r['defender_hp']}/{r['defender_max_hp']}")
            log(f"      付いている特殊エネ: {r['special_energy_names']}")
            log(
                f"      現状の想定ダメージ: {r['current_damage']}  "
                f"({r['removed_energy_name']} を剥がした場合: {r['hypothetical_damage']})"
            )
            log(f"      実際に選んだ手: {r['chosen_desc']}")

    log("")
    _print_q1_report(log, report)
    _print_q2_report(log, report)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Alakazam(743) の Powerful Hand が、Enhanced Hammer(1081) で相手の特殊エネを"
        "剥がせば倒せる局面で、本番構成のエージェントが実際にハンマーを使っているかを数える。"
    )
    ap.add_argument("--games", type=int, default=40, help="相手アーキタイプ1体あたりの試合数(default: 40)")
    ap.add_argument(
        "--opponents", default="crustle",
        help=f"カンマ区切りの相手アーキタイプ名(選択肢: {', '.join(sorted(OPPONENT_SPECS))}) (default: crustle)",
    )
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--workers", type=int, default=1,
        help="同時プロセス数(default: 1=逐次)。0以下で自動(CPU数-1)",
    )
    ap.add_argument("--out", default=None, help="結果JSONの出力先(default: 出力しない)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    opponent_names = [x.strip() for x in args.opponents.split(",") if x.strip()]
    report = run_diagnostic(
        games_per_opponent=args.games,
        opponent_names=opponent_names,
        seed_start=args.seed_start,
        workers=args.workers,
    )
    print_report(report)

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = _ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nresult written to: {out_path}")


if __name__ == "__main__":
    main()
