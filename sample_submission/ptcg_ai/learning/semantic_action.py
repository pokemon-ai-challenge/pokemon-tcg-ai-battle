#!/usr/bin/env python3
"""Option の位置参照を observation.current で解決し、選択肢集合に依存しない
行動ラベルを得る（要件定義 v2 §4.2 の実装）。

## このモジュールの位置づけ（唯一の実装）

方策の推論(``ptcg_ai.learning.policy_features`` / ``policy_model``、提出側・試合中に動く)と
方策の学習データ生成(``kaggle_replays/policy_prior/build_dataset.py``、オフライン)の
両方が、Option を Semantic Action へ解決するのに**このモジュールだけ**を使う。
train/serve で解決規則がズレると最も見つけにくいバグになるため、コピーを作らないこと。
``kaggle_replays`` 側は ``sample_submission`` を ``sys.path`` に足したうえで
``from ptcg_ai.learning.semantic_action import ...`` としてこれを import する
(既存の ``kaggle_replays/deck_predictor/*.py`` が ``ptcg_ai.opponent_modeling`` を
import しているのと同じ配線パターン)。

## なぜ解決が要るか

`cg/api.py` の `Option` は「カードが何であるか」を持たない。持つのは位置参照
(`area` / `index` / `inPlayArea` / `inPlayIndex` / `playerIndex`) だけである。
リプレイ300件・option 延べ302,791件の実測で、`cardId` / `serial` を持つ Option は
471件(0.16%、`OptionType.SKILL` 専用)しかないことを確認している。

一方 `observation.current` 側のカード実体は `{"id": ..., "serial": ...}` を持つ。
したがって「Option の位置参照 -> current のカード実体」を引く手順が必要になる。
本モジュールがその唯一の実装であり、方策学習・再マッピングの両方がここに乗る。

## index を特徴量にしないこと（v2 §4.3 FR-ACT-006）

`index` はエリア内の位置であり、ドロー・シャッフル・トラッシュで容易に動く。
同じ index が別ターンには別のカードを指すため、そのまま特徴量にすると
モデルが偽の相関を学ぶ。特徴量には解決結果の `card_id` を使うこと。
`serial` は個体追跡には使えるが、試合ごとに振り直される任意の番号なので
特徴量にはしない。

## 検証状況

`SelectContext.MAIN` の全 option 161,866件に対し解決失敗 0件（2026-07-26）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# cg/api.py の AreaType のうち、current.players[] 配下のリストに対応するもの。
# DECK(1) は deckCount しか公開されないためリストが存在しない。
_PLAYER_AREA = {2: "hand", 3: "discard", 4: "active", 5: "bench", 6: "prize"}
_STADIUM = 7

# cg/api.py の OptionType
PLAY, ATTACH, EVOLVE, ABILITY, CARD = 7, 8, 9, 10, 3
RETREAT, ATTACK, END = 12, 13, 14
TOOL_CARD, ENERGY_CARD, ENERGY, SKILL, NUMBER = 4, 5, 6, 15, 0
YES, NO = 1, 2

# 実体を持たない OptionType（解決の対象外。位置参照フィールドも一切持たない）
_NO_ENTITY = {END, RETREAT, YES, NO, NUMBER}


class ResolutionStats:
    """解決の成否を集計する。例外を投げない代わりに件数で品質を可視化する。"""

    def __init__(self) -> None:
        self.ok: Counter = Counter()
        self.fail: Counter = Counter()
        self.invariant_violations: Counter = Counter()

    @property
    def total(self) -> int:
        return sum(self.ok.values()) + sum(self.fail.values())

    @property
    def failure_rate(self) -> float:
        return sum(self.fail.values()) / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "resolved": sum(self.ok.values()),
            "failed": sum(self.fail.values()),
            "failure_rate": round(self.failure_rate, 6),
            "failed_by_option_type": dict(self.fail),
            "invariant_violations": dict(self.invariant_violations),
        }


def lookup_entity(current: dict, player: int, area: int | None, index: int | None):
    """(area, index) の位置参照を current のカード実体へ解決する。

    引けなければ None を返す（例外にしない）。相手の手札は null、サイドは
    [null]*6 として公開されるため、相手領域の参照は原理的に解決できないことがある。
    """
    if area is None or index is None:
        return None
    if area == _STADIUM:
        # ★ current.stadium は単一オブジェクトではなく list。不在時は []。
        #    ここを単一オブジェクトとして扱うと ABILITY の解決が 46% 失敗する。
        area_list = current.get("stadium")
    else:
        key = _PLAYER_AREA.get(area)
        if key is None:
            return None
        players = current.get("players") or []
        if not (0 <= player < len(players)) or not isinstance(players[player], dict):
            return None
        area_list = players[player].get(key)
    if not isinstance(area_list, list) or not (0 <= index < len(area_list)):
        return None
    entity = area_list[index]
    return entity if isinstance(entity, dict) else None


def resolve_option(
    option: dict, current: dict, me: int, stats: ResolutionStats | None = None
) -> dict:
    """1つの Option を Semantic Action（v2 §4.4 のスキーマ）へ変換する。

    Args:
        option: `observation.select.option[i]` の生 dict
        current: `observation.current`
        me: 自分の player_index（`current.yourIndex`）
        stats: 集計器。省略可
    """
    option_type = option.get("type")
    action: dict[str, Any] = {
        "option_type": option_type,
        # --- 位置参照（再マッピング用。特徴量にしない） ---
        "from_area": option.get("area"),
        "from_index": option.get("index"),
        "from_player": option.get("playerIndex"),
        "target_area": option.get("inPlayArea"),
        "target_index": option.get("inPlayIndex"),
        "attack_id": option.get("attackId"),
        "tool_index": option.get("toolIndex"),
        "energy_index": option.get("energyIndex"),
        "count": option.get("count"),
        "number": option.get("number"),
        # --- 解決結果（特徴量用） ---
        "card_id": None,
        "serial": None,
        "target_card_id": None,
        "target_serial": None,
        "resolved": True,
    }

    if option_type in _NO_ENTITY or option_type == ATTACK:
        return action

    if option_type == SKILL:
        # 唯一 Option 自身が実体を持つ型。解決不要。
        action["card_id"] = option.get("cardId")
        action["serial"] = option.get("serial")
        return action

    if option_type == PLAY:
        # api.py: "index (int):Index within the hand." area を持たないが手札固定。
        entity = lookup_entity(current, me, 2, option.get("index"))
    elif option_type in (ATTACH, EVOLVE, ABILITY):
        entity = lookup_entity(current, me, option.get("area"), option.get("index"))
    elif option_type in (CARD, TOOL_CARD, ENERGY_CARD, ENERGY):
        owner = option.get("playerIndex")
        entity = lookup_entity(
            current, me if owner is None else owner, option.get("area"), option.get("index")
        )
    else:
        # 未知の OptionType。v2 §4.7 のとおり例外にせず UNKNOWN として記録する。
        action["resolved"] = False
        if stats is not None:
            stats.fail[option_type] += 1
        return action

    if entity is None:
        action["resolved"] = False
        if stats is not None:
            stats.fail[option_type] += 1
        return action

    action["card_id"] = entity.get("id")
    action["serial"] = entity.get("serial")
    if stats is not None:
        stats.ok[option_type] += 1

    if option.get("inPlayArea") is not None:
        target = lookup_entity(current, me, option["inPlayArea"], option.get("inPlayIndex"))
        if target is not None:
            action["target_card_id"] = target.get("id")
            action["target_serial"] = target.get("serial")

    return action


def action_label(action: dict) -> tuple:
    """選択肢集合に依存しない行動ラベル。

    別の decision・別のエージェント間で「同じ行動か」を比較するための正準形。
    `serial` は試合ごとに振り直されるので含めない（含めると常に不一致になる）。
    """
    return (action["option_type"], action["card_id"], action["target_card_id"])


# --- 測定された不変条件（v2 §4.1 / FR-ACT-005） ---------------------------
# リプレイ300件で成立を確認した性質。破れた場合は例外にせず件数を集計する
# （コンペ期間中のエンジン更新で破れ得るため）。

def check_invariants(select: dict, stats: ResolutionStats) -> None:
    options = select.get("option") or []
    types = [o.get("type") for o in options]

    # MAIN には必ず END が1つ含まれる（END 24,785件 = MAIN decision 数と完全一致）
    if select.get("context") == 0 and types.count(END) != 1:
        stats.invariant_violations["main_has_exactly_one_end"] += 1

    for option in options:
        # ATTACH / EVOLVE の area は全件 HAND(2) だった
        if option.get("type") in (ATTACH, EVOLVE) and option.get("area") != 2:
            stats.invariant_violations["attach_evolve_area_is_hand"] += 1
        # ABILITY の area は ACTIVE / BENCH / STADIUM のみ
        if option.get("type") == ABILITY and option.get("area") not in (4, 5, 7):
            stats.invariant_violations["ability_area_in_play"] += 1
