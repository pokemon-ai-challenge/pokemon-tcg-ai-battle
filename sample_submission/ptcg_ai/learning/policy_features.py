"""方策の Policy Prior（pointwise scoring）向け特徴抽出（唯一の実装）。

``ptcg_ai.action_selection.selector`` (試合中・提出側) と、後続のオフライン学習パイプライン
(``kaggle_replays/policy_prior/`` 配下、このモジュールを import する)の両方が、このモジュール
だけを使って特徴ベクトルを作る。train/serve で特徴抽出がズレるのは最も見つけにくいバグなので、
コピーを作らないこと。

## 入力

- ``state``: 1 decision の状態。``ptcg_ai.learning.observable_state.observable_state()`` の
  戻り値そのもの（学習データ生成側とランタイム側の両方がこの関数を経由するため、形は必ず揃う）。
- ``action``: 1 選択肢の解決済み Semantic Action。
  ``ptcg_ai.learning.semantic_action.resolve_option()`` の戻り値そのもの。

## 出力

``dict[str, float]``（疎な特徴名 -> 値）。存在しない特徴は単に辞書に現れない
（重み側で 0 として扱われる想定。``policy_model.py`` 参照）。

## 禁止事項（要件 FR-ACT-006）

``index`` / ``from_index`` / ``target_index`` / ``serial`` を特徴に使ってはならない。
``index`` はエリア内位置でドローやトラッシュで動くため偽の相関を学ぶ。``serial`` は
試合ごとに振り直される。``resolve_option()`` の戻り値にはこれらのキーがそのまま残っているが、
本モジュールは意図的にこれらを一切参照しない（``tool_index`` / ``energy_index`` も同じ理由で
参照しない）。``from_area`` と ``from_player``(= playerIndex) は安定なので使ってよい。

## カードの静的属性・頻出カード one-hot

全種カード one-hot は次元過多になるため、以下の2本立てにする:

- ``card_attributes``: ``{str(card_id): {attr_name: value, ...}}`` の形の静的属性テーブル。
  ``data/EN_Card_Data.csv`` 由来だが、このモジュールは CSV に直接依存しない
  (提出時に CSV が同梱されない可能性があるため、学習パイプラインが重みJSONの
  ``card_attributes`` にテーブルごと埋め込み、ここには引数として渡ってくる)。属性名は
  学習側が決めるため固定せず、テーブルに存在するキーをそのまま動的に特徴名へ展開する。
- ``frequent_card_ids``: 重みJSONの ``frequent_card_ids`` に記録された、頻出上位N種の
  card_id のみ one-hot にする。

## 交互作用

``option_type × カード属性`` を明示的に外積し、``option_type`` ごとに別の線形項として
学習できるようにする(``option_type_{t}__attr_{name}`` という特徴名)。
"""

from __future__ import annotations

from typing import Any

# --- 状態特徴で使う真偽値フィールド(0.0/1.0 に変換するだけ) ------------------
_STATE_BOOL_FIELDS = (
    "energy_attached",
    "retreated",
    "supporter_played",
    "stadium_played",
)

_SIDE_BOOL_FIELDS = ("asleep", "confused", "paralyzed", "poisoned", "burned")

# in_play() (observable_state.py) が返すポケモン1体分のフィールドのうち、
# 数値特徴に使うもの。card_id はここでは使わない(one-hot ではなく数値特徴に
# 混ぜても意味がないため。カード種の情報が要るなら card_attributes 経由で action 側から入る)。
_POKEMON_NUMERIC_FIELDS = ("hp", "max_hp", "n_energy", "n_tool")


def _num(value: Any) -> float:
    """None/bool を含む任意値を安全に float へ。変換できなければ 0.0。"""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pokemon_features(prefix: str, pokemon: dict | None, features: dict[str, float]) -> None:
    """1体分の in_play() 出力を ``{prefix}_present`` / ``{prefix}_{field}`` へ展開する。"""
    features[f"{prefix}_present"] = 1.0 if pokemon is not None else 0.0
    if pokemon is None:
        return
    for field in _POKEMON_NUMERIC_FIELDS:
        features[f"{prefix}_{field}"] = _num(pokemon.get(field))


def _side_features(prefix: str, side: dict, features: dict[str, float]) -> None:
    _pokemon_features(f"{prefix}_active", side.get("active"), features)
    features[f"{prefix}_bench_count"] = float(len(side.get("bench") or []))
    features[f"{prefix}_n_prize"] = _num(side.get("n_prize"))
    features[f"{prefix}_n_hand"] = _num(side.get("n_hand"))
    features[f"{prefix}_n_deck"] = _num(side.get("n_deck"))
    features[f"{prefix}_discard_count"] = float(len(side.get("discard") or []))
    for field in _SIDE_BOOL_FIELDS:
        features[f"{prefix}_{field}"] = _num(side.get(field))


def state_features(state: dict) -> dict[str, float]:
    """``observable_state()`` の出力(1 decision の状態)を状態特徴へ展開する。

    ``state`` が壊れている(必須キーが欠けている)場合でも例外を投げず、そのフィールドの
    特徴を省く(``.get()`` を通す)。呼び出し側でターンを止めないため。
    """
    features: dict[str, float] = {"turn": _num(state.get("turn"))}
    for field in _STATE_BOOL_FIELDS:
        features[field] = _num(state.get(field))

    own = state.get("own") or {}
    opponent = state.get("opponent") or {}
    _side_features("own", own, features)
    _side_features("opp", opponent, features)

    return features


def _card_attribute_features(
    prefix: str,
    card_id: int | None,
    card_attributes: dict[str, dict[str, float]],
    features: dict[str, float],
) -> dict[str, float]:
    """``card_id`` の静的属性テーブルを ``{prefix}_{attr_name}`` として展開し、
    展開した attr の辞書(interaction 生成用)を返す。属性名は固定せず、
    ``card_attributes`` に実際に載っているキーをそのまま使う。
    """
    if card_id is None:
        return {}
    attrs = card_attributes.get(str(card_id))
    if not attrs:
        return {}
    resolved: dict[str, float] = {}
    for name, value in attrs.items():
        v = _num(value)
        features[f"{prefix}_{name}"] = v
        resolved[name] = v
    return resolved


def action_features(
    action: dict,
    card_attributes: dict[str, dict[str, float]] | None = None,
    frequent_card_ids: list[int] | None = None,
) -> dict[str, float]:
    """1選択肢の解決済み Semantic Action(``resolve_option()`` の戻り値)を行動特徴へ展開する。

    Args:
        action: ``ptcg_ai.learning.semantic_action.resolve_option()`` の戻り値。
        card_attributes: ``{str(card_id): {attr_name: value}}``。省略時は静的属性を使わない。
        frequent_card_ids: 頻出上位N種の card_id。省略時は card_id one-hot を作らない。

    禁止: ``index`` / ``from_index`` / ``target_index`` / ``serial`` は絶対に参照しない
    (FR-ACT-006、モジュール docstring 参照)。
    """
    card_attributes = card_attributes or {}
    frequent_set = set(frequent_card_ids or [])

    features: dict[str, float] = {}

    option_type = action.get("option_type")
    if option_type is not None:
        features[f"option_type_{option_type}"] = 1.0

    from_area = action.get("from_area")
    if from_area is not None:
        features[f"from_area_{from_area}"] = 1.0

    from_player = action.get("from_player")
    if from_player is not None:
        features[f"from_player_{from_player}"] = 1.0

    # count / number は「量」であって位置参照ではないので使ってよい。
    if action.get("count") is not None:
        features["count"] = _num(action.get("count"))
    if action.get("number") is not None:
        features["number"] = _num(action.get("number"))
    if action.get("attack_id") is not None:
        features[f"attack_id_{action['attack_id']}"] = 1.0

    card_id = action.get("card_id")
    target_card_id = action.get("target_card_id")

    if card_id is not None and card_id in frequent_set:
        features[f"card_id_{card_id}"] = 1.0
    if target_card_id is not None and target_card_id in frequent_set:
        features[f"target_card_id_{target_card_id}"] = 1.0

    card_attrs = _card_attribute_features("card_attr", card_id, card_attributes, features)
    _card_attribute_features("target_attr", target_card_id, card_attributes, features)

    # --- 交互作用: option_type × カード属性(外積して明示的な線形項にする) ---
    if option_type is not None:
        for name, value in card_attrs.items():
            features[f"option_type_{option_type}__attr_{name}"] = value

    return features


def extract_features(
    state: dict,
    action: dict,
    card_attributes: dict[str, dict[str, float]] | None = None,
    frequent_card_ids: list[int] | None = None,
) -> dict[str, float]:
    """1 decision の状態 + 1 選択肢の解決済み Semantic Action から特徴 dict を作る。

    ``policy_model.PolicyModel`` から呼ばれる唯一の入口。状態特徴と行動特徴を
    単純にマージするだけ(キーが衝突しない設計になっている)。
    """
    features = state_features(state)
    features.update(action_features(action, card_attributes, frequent_card_ids))
    return features
