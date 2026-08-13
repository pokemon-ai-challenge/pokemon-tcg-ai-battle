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
from ptcg_ai.shared import card_cache

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


def _prize_value(pokemon) -> int:
    """このポケモンが気絶したとき相手が取るサイド枚数(ex/ルールを持つポケモン=2、他=1)。

    カード名やデッキを名指ししないデッキ非依存の判定。カードデータが引けないときは
    安全側の 1 を返す(未知カードを過大評価しない)。
    """
    try:
        return 2 if card_cache.get_card(int(pokemon.id)).ex else 1
    except Exception:  # noqa: BLE001
        return 1


def _kos_needed_against(player) -> float:
    """``player`` を倒し切る(=残サイドを取り切る)のに必要な KO 回数の見積り。

    残サイド枚数だけを見る従来の指標は「1体倒すと何枚取れるか」を無視するため、
    サイド2枚の ex だけで戦う盤面と、サイド1枚のポケモンを混ぜた盤面を区別できない。
    ここでは「相手が今見えている自軍ポケモンを、サイド価値の高い順に倒していく」と仮定して
    必要 KO 回数を数える(＝相手にとって最短のサイドの取り方)。盤面のポケモンだけでは
    残サイドを賄えない分は、まだ見えていないポケモンが最良ケース(ex=2枚)であると仮定して
    ceil(不足/2) 回を足す。ここを按分(小数)にするとサイドパリティの肝である
    「あと1枚取るにも KO は1回必要」という整数性が消えるので、切り上げる。

    例: 残サイド6・場が全部 ex(2枚) なら 3回。1枚のポケモンを1体挟むと 4回になり、
    「サイドが偶数のときに1枚ポケモンを立てると相手の勝ち筋が1KO遠のく」効果が値に出る。
    """
    remaining = len(player.prize or [])
    if remaining <= 0:
        return 0.0
    values = []
    for slot in (player.active or []):
        if slot is not None:
            values.append(_prize_value(slot))
    for slot in (player.bench or []):
        if slot is not None:
            values.append(_prize_value(slot))
    if not values:
        return float(remaining)
    # 相手は「1回のKOで多く取れる」ポケモンから狙うのが最短。降順に消費する。
    values.sort(reverse=True)
    kos = 0
    taken = 0
    for v in values:
        if taken >= remaining:
            break
        taken += v
        kos += 1
    if taken < remaining:
        # 場に見えていない分。相手にとって最良(1KO=2枚)を仮定した下限を切り上げで足す。
        kos += math.ceil((remaining - taken) / 2)
    return float(kos)


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

    def __init__(self, card_advantage_coeff: float = 0.0, prize_parity_coeff: float = 0.0):
        # 汎用「カードアドバンテージ」項の係数(手札枚数差 self-opp のロジット重み)。
        # 既定 0.0 = 無効(従来挙動そのまま)。>0 にすると「引く/回収する」局面を評価が
        # 高く見るようになる。特定コンボを名指ししないデッキ非依存の項で、ドローエンジン全般
        # (例: リッチエネルギー付与の+4ドローや、にげあしドロー特性の+3ドロー)を自然に
        # 後押しするのが狙い(decision-pipeline の card-advantage 実験)。
        self.card_advantage_coeff = card_advantage_coeff
        # サイド「価値」の項(既定 0.0 = 無効、従来挙動そのまま)。
        #
        # 既存の prize_advantage はサイドの**枚数**しか見ないため、
        # 「自分が倒されたら相手に何枚渡すか」を区別できない。全部 ex(2枚)の盤面と、
        # 1枚ポケモンを混ぜた盤面が同点になってしまい、サイドが偶数のときに1枚ポケモンを
        # 前に出して相手の勝ち筋を1KO遠のかせる、という基本的なサイドプランを探索が
        # 発見できない。しかも board_strength は HP 割合を含むので、HPの低い1枚ポケモンを
        # 前に出すと評価が下がる方向にバイアスがかかる。
        #
        # この項は「相手が勝つのに必要なKO数 − 自分が勝つのに必要なKO数」を足す。
        # カード名やデッキを名指ししないデッキ非依存の項。
        self.prize_parity_coeff = prize_parity_coeff

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
            if self.prize_parity_coeff:
                # 相手が自分を倒し切るのに必要なKO数 − 自分が相手を倒し切るのに必要なKO数。
                # 正 = 相手の方が多くKOを要する = 自分が有利。
                logit += self.prize_parity_coeff * (
                    _kos_needed_against(mine) - _kos_needed_against(opp))
            return _sigmoid(logit)
        except Exception:
            return _NEUTRAL


class ValueModelEvaluator:
    """既存 ValueModel を ``me`` 視点へ整列して包む差し替え実装。"""

    def __init__(self, model=None):
        self._model = model  # 遅延生成(未指定なら初回 evaluate で作る)

    def _get_model(self):
        if self._model is None:
            from ptcg_ai.learning.value_model import ValueModel

            self._model = ValueModel()
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


def build_evaluator(config: dict | None = None) -> LeafEvaluator:
    """``config["kind"]`` に応じた LeafEvaluator を返す(既定 handcrafted)。

    handcrafted の場合、``config["card_advantage_coeff"]``(既定 0.0=無効)で汎用の
    カードアドバンテージ項を、``config["prize_parity_coeff"]``(既定 0.0=無効)で
    サイド価値(必要KO数)の項を有効化できる。どちらも既定 0.0 なので、
    指定しない限り従来と完全に同一の評価値になる。
    """
    cfg = config or {}
    kind = cfg.get("kind", "handcrafted")
    if kind == "value":
        return ValueModelEvaluator()
    return HandcraftedEvaluator(
        card_advantage_coeff=float(cfg.get("card_advantage_coeff", 0.0)),
        prize_parity_coeff=float(cfg.get("prize_parity_coeff", 0.0)),
    )
