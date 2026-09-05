"""末端評価インターフェース(意思決定パイプライン Step4 用)。

`pipeline.py` の先読みの葉ノードで盤面を数値化する部品。spec の要件どおり
**評価関数をインターフェースとして分離**し、後で学習 Value モデルへ差し替えられるように
している。

- `LeafEvaluator.evaluate(state, me) -> float`  : ``me`` 視点の勝ち見込み(0..1)。
- `HandcraftedEvaluator` : サイド差・盤面総合力・エネルギー加速・ベンチ展開数の線形結合
  (係数は :data:`HandcraftedEvaluator.COEFFS` 1箇所に集約)。`board_evaluation` の既存部品
  (`attacker_score`)を読み取り専用で流用する。
- `ValueModelEvaluator` : 既存 `learning.value_model.ValueModel` を ``me`` 視点へ整列して包む。
- `build_evaluator(config)` : ``config["kind"]``("handcrafted"|"value")で選ぶ。既定 handcrafted。

いずれの実装も例外を投げず、評価不能時は中立値 0.5 を返す(先読みを止めないため)。
"""

from __future__ import annotations

import math
from typing import Protocol

from cg.api import State

from ptcg_ai.board_evaluation.board_features import attacker_score

_NEUTRAL = 0.5


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class LeafEvaluator(Protocol):
    """先読みの葉で盤面を ``me`` 視点の勝ち見込み(0..1)に写像する。"""

    def evaluate(self, state: State, me: int) -> float: ...


def _board_strength(player) -> float:
    """自軍全ポケモン(バトル+ベンチ)の attacker_score 合計。"""
    total = 0.0
    for slot in (player.active or []):
        if slot is not None:
            total += attacker_score(slot)
    for slot in (player.bench or []):
        if slot is not None:
            total += attacker_score(slot)
    return total


def _energy_on_board(player) -> int:
    count = 0
    for slot in (player.active or []):
        if slot is not None:
            count += len(slot.energies or [])
    for slot in (player.bench or []):
        if slot is not None:
            count += len(slot.energies or [])
    return count


def _bench_count(player) -> int:
    return sum(1 for slot in (player.bench or []) if slot is not None)


class HandcraftedEvaluator:
    """手作り線形評価。係数はここ1箇所に集約(後で学習 Value に差し替え可能)。"""

    # spec: 「サイド差、盤面の総打点、エネルギー加速状況、ベンチ展開数などの線形結合」。
    # いずれも (自分 - 相手) の差分に対する重み。sigmoid で 0..1 に写す前のロジット係数。
    COEFFS: dict[str, float] = {
        "prize_advantage": 1.20,   # 相手の残サイド - 自分の残サイド(正 = 自分が先に取り切る側)
        "board_strength": 0.05,    # 盤面総合力(HP割合+エネルギー)の差
        "energy_advantage": 0.15,  # 盤面エネルギー総数の差
        "bench_advantage": 0.10,   # ベンチ展開数の差
    }

    def __init__(self, card_advantage_coeff: float = 0.0):
        # 汎用「カードアドバンテージ」項の係数(手札枚数差 self-opp のロジット重み)。
        # 既定 0.0 = 無効(従来挙動そのまま)。>0 にすると「引く/回収する」局面を評価が
        # 高く見るようになる。特定コンボを名指ししないデッキ非依存の項で、ドローエンジン全般
        # (例: リッチエネルギー付与の+4ドローや、にげあしドロー特性の+3ドロー)を自然に
        # 後押しするのが狙い(decision-pipeline の card-advantage 実験)。
        self.card_advantage_coeff = card_advantage_coeff

    def evaluate(self, state: State, me: int) -> float:
        try:
            if state is None:
                return _NEUTRAL
            # 決着済み局面は評価関数を通さず確定値にする。
            if state.result == me:
                return 1.0
            if state.result != -1:
                return 0.0

            mine = state.players[me]
            opp = state.players[1 - me]

            # サイドは「残り枚数が少ない方が勝ちに近い」。相手残 - 自分残 が正なら自分が有利。
            prize_advantage = len(opp.prize or []) - len(mine.prize or [])
            board = _board_strength(mine) - _board_strength(opp)
            energy = _energy_on_board(mine) - _energy_on_board(opp)
            bench = _bench_count(mine) - _bench_count(opp)

            c = self.COEFFS
            logit = (
                c["prize_advantage"] * prize_advantage
                + c["board_strength"] * board
                + c["energy_advantage"] * energy
                + c["bench_advantage"] * bench
            )
            if self.card_advantage_coeff:
                my_hand = getattr(mine, "handCount", 0) or 0
                opp_hand = getattr(opp, "handCount", 0) or 0
                logit += self.card_advantage_coeff * (my_hand - opp_hand)
            return _sigmoid(logit)
        except Exception:
            return _NEUTRAL


# ValueModel はプロセス内で1個だけ作って共有する。`build_evaluator` は `pipeline.search` の
# **中**で毎 select 呼ばれるため、ここでキャッシュしないと重みJSON(166次元MLP)のディスク読み込みが
# 意思決定のたびに走る。handcrafted 経路(本番既定)は ValueModel に触れないので影響を受けない。
_SHARED_VALUE_MODEL = None


def _get_shared_value_model():
    global _SHARED_VALUE_MODEL
    if _SHARED_VALUE_MODEL is None:
        from ptcg_ai.learning.value_model import ValueModel

        _SHARED_VALUE_MODEL = ValueModel()
    return _SHARED_VALUE_MODEL


def value_model_ready() -> bool:
    """学習 Value がロード済みかを返す(評価harness / テスト用の明示的な確認口)。

    `ValueModel` は重みJSONが無くても例外を出さず「未ロード」状態になり、推論は常に 0.5 を返す。
    そのまま `kind="value"`/`"blend"` で走らせると **葉評価が定数になっているのに探索は動いている**
    という気付きにくい失敗になる(勝率だけ見ても分からない)。A/B を回す前にこれで確認する。
    """
    try:
        return bool(getattr(_get_shared_value_model(), "is_ready", False))
    except Exception:  # noqa: BLE001
        return False


class ValueModelEvaluator:
    """既存 ValueModel を ``me`` 視点へ整列して包む差し替え実装。"""

    def __init__(self, model=None):
        self._model = model  # 未指定ならプロセス共有インスタンスを使う(毎select再読込を防ぐ)

    def _get_model(self):
        if self._model is None:
            self._model = _get_shared_value_model()
        return self._model

    def evaluate(self, state: State, me: int) -> float:
        try:
            if state is None:
                return _NEUTRAL
            if state.result == me:
                return 1.0
            if state.result != -1:
                return 0.0
            # predict_win_prob_from_state は state.yourIndex 視点の勝率を返す。
            prob = self._get_model().predict_win_prob_from_state(state)
            if state.yourIndex == me:
                return prob
            return 1.0 - prob
        except Exception:
            return _NEUTRAL


class BlendedEvaluator:
    """``alpha * learned_value + (1 - alpha) * handcrafted`` の線形ブレンド。

    学習 Value を本番探索の葉へ入れる際、handcrafted を検証なしに捨てないための混合口
    (climb v1.5 診断 §5.2 の α スイープ用)。α=0.0 は handcrafted と、α=1.0 は value と
    それぞれ数値的に一致する。

    決着局面は両実装とも 1.0/0.0 を返すため、ブレンドしても確定値は保たれる。
    """

    def __init__(self, alpha: float, handcrafted=None, value=None):
        self.alpha = min(1.0, max(0.0, float(alpha)))
        self._hand = handcrafted if handcrafted is not None else HandcraftedEvaluator()
        self._value = value if value is not None else ValueModelEvaluator()

    def evaluate(self, state: State, me: int) -> float:
        try:
            if state is None:
                return _NEUTRAL
            if state.result == me:
                return 1.0
            if state.result != -1:
                return 0.0
            a = self.alpha
            if a <= 0.0:
                return self._hand.evaluate(state, me)
            if a >= 1.0:
                return self._value.evaluate(state, me)
            return a * self._value.evaluate(state, me) + (1.0 - a) * self._hand.evaluate(state, me)
        except Exception:
            return _NEUTRAL


def build_evaluator(config: dict | None = None) -> LeafEvaluator:
    """``config["kind"]`` に応じた LeafEvaluator を返す(既定 handcrafted)。

    - ``"handcrafted"``(既定): 手作り線形評価。``config["card_advantage_coeff"]``(既定 0.0=
      無効)で汎用のカードアドバンテージ項を有効化できる。
    - ``"value"``: 学習 Value 単体。
    - ``"blend"``: ``config["alpha"]``(既定 0.5)で learned/handcrafted を線形混合。
    """
    cfg = config or {}
    kind = cfg.get("kind", "handcrafted")
    if kind == "value":
        return ValueModelEvaluator()
    if kind == "blend":
        return BlendedEvaluator(
            alpha=float(cfg.get("alpha", 0.5)),
            handcrafted=HandcraftedEvaluator(
                card_advantage_coeff=float(cfg.get("card_advantage_coeff", 0.0))
            ),
        )
    return HandcraftedEvaluator(card_advantage_coeff=float(cfg.get("card_advantage_coeff", 0.0)))
