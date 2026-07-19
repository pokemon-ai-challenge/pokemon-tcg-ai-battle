"""相手側の非公開情報（山札・手札・サイドの中身）を、混合モデルで推定する。

design.md §5「相手側の設計（混合モデル）」をそのままコード化したもの。相手の60枚は中身自体が
不明なので、自分側（``OwnHiddenState``、Phase 1）のように「全部見えているゾーンの引き算で確定」は
できない。代わりに2段の確率モデルを組む:

    P(相手の60枚の中身とゾーン配置)
        = Σ_archetype P(archetype | 観測) * P(ゾーン配置 | archetype, 観測)

1. リスト事後分布 ``P(archetype | 観測)``: ``HybridDeckPredictor.predict()`` の出力をそのまま入力に
   使う（本クラスは ``HybridDeckPredictor`` / ``OpponentKnowledge`` を import しない。疎結合を守る）。
2. ゾーン配分 ``P(ゾーン配置 | archetype, 観測)``: アーキタイプを1つ仮定すれば「代表60枚リスト
   − 観測済みカード = 未観測プール」が決まり、あとは ``zone_math`` の超幾何を3ゾーン（山札 /
   手札 / サイド）に拡張した多変量版で配分する。

代表リスト（``archetype_card_pool.json``、``card_id -> {median, inclusion_rate}``）は
``kaggle_replays/deck_predictor/build_archetype_pool.py`` がオフラインで構築する。無ければ
``is_ready = False``（``MLDeckPredictor`` の「重みJSON無しは未ロード」フォールバックと同じ思想）。

## プールサイズとゾーン合計の不一致の扱い（重要）

代表リストの median 枚数の合計から観測済みを引いた「未観測プール枚数 ``M``」は、実際の隠しゾーン
合計 ``Z = deckCount + handCount + prizeCount`` と一般に一致しない（代表値は「よくある構築」の
中央値であり、テックカードやサイド落ち・トラッシュ済みの実データとはズレるため、実戦では必ず起きる）。
本クラスは次の統一ルールで吸収する（``marginals()`` / ``sample()`` で同じ扱い）:

- ``M < Z``（プールがゾーン合計より小さい）: 不足分 ``Z - M`` 枚を「不明カード」
  （プレースホルダ ``card_id = None``）として補い、実効プールサイズを ``Z`` に引き上げる。
  ``sample()`` はこの ``None`` を隠しゾーンに混ぜて返す（相手の未知テックカード等を表す）。
- ``M > Z``（プールがゾーン合計より大きい）: 超過分は「今その相手が持っていないカード」として
  実効プールサイズを ``M`` のままにし、超幾何の「非対象」側に余らせる（``sample()`` では ``Z`` 枚
  だけ非復元抽出し、残りは捨てる）。
- いずれの場合も ``sample()`` が返す各リスト長は ``player_state`` のゾーンサイズ
  （``deckCount`` / ``handCount`` / ``len(prize)``）に必ず一致する（``search_begin`` にそのまま
  渡せる形を最優先する）。

実効プールサイズは全ケースで ``M_eff = max(M, Z)`` にまとめられる。あるカード（未観測 ``k`` 枚）が
特定ゾーン（サイズ ``s``）に少なくとも1枚ある確率は ``prob_in_prize(M_eff, s, k)`` で計算でき、
``M < Z`` のときは ``M_eff = Z`` に増えた分だけプレースホルダが非対象側を埋め、``M > Z`` のときは
超過分が非対象側に余る、という形で自動的に整合する。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from cg.api import PlayerState

from .zone_math import prob_in_prize

_DEFAULT_POOL_FILENAME = "archetype_card_pool.json"

# 代表リストに無い観測カード1枚ごとに、そのアーキタイプの重みへ掛ける減衰率。
# （design.md §5.2「メタ外亜種対応」: 代表リストに無いカードが見えたらアーキタイプの尤度を
# 下げるが、ゼロにはしない。）
_SMOOTHING_DECAY_PER_MISS = 0.5
# 減衰の下限（元の重みに対する比）。何枚外れても元の重みの _SMOOTHING_FLOOR_RATIO 倍より
# 下げない = ゼロを作らない（「確信度は慎重側に倒す」方針）。
_SMOOTHING_FLOOR_RATIO = 0.05

# 代表リストを持たない特別扱いのアーキタイプ（design.md 実装プラン 2.2）。
# 雑多なデッキの寄せ集めであり、median 枚数を集計しても意味のある代表60枚にならないため、
# サンプル時は全ゾーンを「不明カード」(None)で埋め、marginals では特定 card_id に寄与させない。
_NO_REPRESENTATIVE_ARCHETYPES = frozenset({"other"})

# ------------------------------------------------------------------
# 手札(hand)確率の過信を慎重側へ丸める補正（Phase 3 差し戻し対応）
#
# 検証（kaggle_replays/deck_predictor/evaluate_hidden_information.py・68,646時点／神視点5試合の
# 両データソース）で、手札の marginals だけがナイーブ基準より悪化することが一貫して再現した。
# 内訳は「高確信ビンでの系統的な過信」で、確信度 0.15 を超えるあたりから予測確率が実際の的中率を
# 大きく上回る（例: 予測 0.7 帯に対し的中 ~0.40、予測 0.95 帯に対し的中 ~0.66）。一方、確信度が
# 0.15 以下の低確信帯はナイーブと同等に正しく、山札・サイドの marginals は過信しない。
#
# 過信が手札「だけ」に出る理由（design.md §5.3）: 未観測プールから各ゾーンへ一様配分する本モデルは、
# 相手が手札からプレイ可能なカードを使う（=手札を離れて公開ゾーンへ移り観測済みになる）という偏りを
# 無視している。このため「まだ観測されていない特定カードが、小さな手札(3〜7枚)にそのまま残っている」
# 確率を系統的に過大評価する。加えて手札はゾーンが小さいので、代表リスト(median)の誤差や
# アーキタイプ誤判定の相対影響を山札より強く受ける（終盤 M が小さいほど顕著）。
#
# design.md §5.3 は「手札の偏りのモデル化（分布の精緻化）は後回し」と明記しているため、ここでは
# 分布を作り直すのではなく、プロジェクト確立方針「確信度は慎重側に倒す・生の見積もりより強気に
# しない」に沿って、手札確率“だけ”を単調・保守的に割り引く（山札/サイドの合格状態には触れない）:
#
#     p' = FLOOR + (p - FLOOR) * KEEP        (p > FLOOR)
#     p' = p                                  (p <= FLOOR)
#
# - FLOOR 以下（低確信）の手札推定は過信が観測されないのでそのまま残す（唯一ナイーブが分布を持つ
#   低確信帯を歪めない＝ナイーブ比較を不当に有利化しない）。
# - FLOOR を超える“アーキタイプ由来の上乗せ確信”のうち KEEP の割合だけを残し、残りを割り引く。
#   p' <= p が常に成り立ち、確率を強気側へ動かすことは決してない（T>=1 温度スケーリング方針と同思想）。
#
# パラメータは検証データの「悪化の閾値(~0.15)」と「保守的に半分だけ残す」から設定し、
# floor∈[0.10,0.20]・keep∈[0.4,0.6] の広い近傍すべてでナイーブ手札ECEを下回る（特定値への過剰適合
# ではないことを確認済み）。数値そのものへのフィットではなく上記の原理に基づく穏当な既定値とする。
_HAND_CONFIDENCE_FLOOR = 0.15
_HAND_EXCESS_KEEP = 0.5


def _shrink_hand_confidence(prob: float) -> float:
    """手札 marginal を単調・保守的に割り引く（``FLOOR`` 超の“上乗せ確信”を ``KEEP`` 倍に縮める）。

    ``prob`` は周辺化後の「そのカードが手札にある確率」。``FLOOR`` 以下は素通し、超えた分だけ
    割り引くので ``戻り値 <= prob`` が常に成り立つ（強気側へは決して動かさない）。
    """
    if prob <= _HAND_CONFIDENCE_FLOOR:
        return prob
    return _HAND_CONFIDENCE_FLOOR + (prob - _HAND_CONFIDENCE_FLOOR) * _HAND_EXCESS_KEEP


class OpponentHiddenState:
    """相手の山札 / 手札 / サイドの中身を、リスト事後分布 × ゾーン配分の混合モデルで推定する。"""

    def __init__(self, pool_path: str | Path | None = None) -> None:
        if pool_path is None:
            pool_path = Path(__file__).parent / _DEFAULT_POOL_FILENAME
        self._pool_path = Path(pool_path)

        # archetype -> {card_id(int) -> median枚数(int)}。代表リスト（未観測プールの母集団）。
        self._archetype_pool: dict[str, dict[int, int]] = {}
        self._is_ready = False
        self._load_pool()

        # update() で受け取る直近の状態（サンプリング・周辺化に使う）。
        self._posterior: dict[str, float] = {}
        self._observed: dict[int, int] = {}
        self._deck_count = 0
        self._hand_count = 0
        self._prize_count = 0

    @property
    def is_ready(self) -> bool:
        """代表リストJSONの読み込みに成功していれば True。"""
        return self._is_ready

    def _load_pool(self) -> None:
        if not self._pool_path.exists():
            return  # 未ロード（is_ready=False のまま。build_archetype_pool.py 実行後に配置される）。
        with self._pool_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        archetypes = payload.get("archetypes", {})
        parsed: dict[str, dict[int, int]] = {}
        for archetype, entry in archetypes.items():
            card_counts = entry.get("card_counts", {})
            per_card: dict[int, int] = {}
            for card_id_str, stats in card_counts.items():
                median = int(stats.get("median", 0))
                if median <= 0:
                    continue  # median 0（過半数のデッキに入っていない）カードは代表リストに含めない。
                per_card[int(card_id_str)] = median
            parsed[archetype] = per_card
        self._archetype_pool = parsed
        self._is_ready = True

    # ------------------------------------------------------------------
    # 更新

    def update(
        self,
        archetype_posterior: dict[str, float],
        observed_card_ids: dict[int, int],
        player_state: PlayerState,
    ) -> None:
        """毎ターン呼ぶ。相手のアーキタイプ事後分布・観測済みカード・相手の ``PlayerState`` を取り込む。

        - ``archetype_posterior``: ``HybridDeckPredictor.predict(observed_cards, turn)`` の戻り値
          （アーキタイプ名 -> 確率、合計1.0）。
        - ``observed_card_ids``: ``OpponentKnowledge.get_prediction_features()["observed_card_ids"]``
          （card_id -> これまでに相手側で観測できた枚数）。
        - ``player_state``: 相手プレイヤーの ``PlayerState``。``deckCount`` / ``handCount`` /
          ``len(prize)`` からゾーンサイズを取る（相手の ``hand`` は None なので ``handCount`` を使う）。

        呼び出し側が ``HybridDeckPredictor`` / ``OpponentKnowledge`` の生成・更新を担い、その出力だけを
        ここへ渡す（``prediction_summary.py`` が予測器を duck-typing で受けるのと同じ疎結合）。
        """
        self._posterior = dict(archetype_posterior)
        self._observed = dict(observed_card_ids)
        self._deck_count = int(player_state.deckCount)
        self._hand_count = int(player_state.handCount)
        self._prize_count = len(player_state.prize)

    # ------------------------------------------------------------------
    # スムージング（リスト事後分布側）

    def _smoothed_weight(self, archetype: str, observed_card_ids: dict[int, int]) -> float:
        """代表リストに無い観測カードの枚数に応じて、そのアーキタイプの重みを減衰させる。

        design.md §5.2: 観測済みだが代表リストに無いカードが多いほど、そのアーキタイプらしさは
        下がる。ただし下限を設けてゼロにはしない（代表リストに無いカードが1枚見えただけで候補から
        永久に消える、という過剰な推論を避ける。「確信度は慎重側に倒す」方針）。

        減衰率は「代表リストに無い観測カードの distinct な card_id 数」に対して指数的
        （``_SMOOTHING_DECAY_PER_MISS ** miss``）にかけ、``_SMOOTHING_FLOOR_RATIO``（元の重みに対する
        比）を下限とする。代表リストを持たないアーキタイプ（``other``）は減衰させない（そもそも
        「リストに無い」という概念が成り立たないため、生の重みをそのまま使う）。
        """
        raw = float(self._posterior.get(archetype, 0.0))
        if raw <= 0.0:
            return 0.0
        if archetype in _NO_REPRESENTATIVE_ARCHETYPES:
            return raw

        pool = self._archetype_pool.get(archetype)
        if not pool:
            # 代表リストが空 = 何が入っているか分からないので減衰させようがない。生の重みを返す。
            return raw

        miss = sum(1 for card_id, count in observed_card_ids.items() if count > 0 and card_id not in pool)
        factor = max(_SMOOTHING_FLOOR_RATIO, _SMOOTHING_DECAY_PER_MISS ** miss)
        return raw * factor

    def _smoothed_normalized_weights(self) -> dict[str, float]:
        """全アーキタイプに ``_smoothed_weight`` を適用したうえで、合計1.0に再正規化して返す。

        スムージングは重みを不均一に減衰させるので、適用後は合計が1.0からズレる。周辺化・サンプリング
        の前に必ずここで正規化し直す（設計方針: 重みの正規化を忘れない）。
        """
        weights = {
            archetype: self._smoothed_weight(archetype, self._observed)
            for archetype in self._posterior
        }
        total = sum(weights.values())
        if total <= 0.0:
            return {}
        return {archetype: w / total for archetype, w in weights.items() if w > 0.0}

    # ------------------------------------------------------------------
    # 未観測プールの計算

    def _unobserved_pool(self, archetype: str) -> dict[int, int]:
        """アーキタイプの代表リスト（median枚数）から観測済み枚数を引いた未観測プール。

        ``card_id -> 未観測枚数``（負にならないよう ``max(0, ...)`` でクリップ）。代表リストを持たない
        アーキタイプ（``other`` 等）は空 dict を返す。
        """
        pool = self._archetype_pool.get(archetype)
        if not pool or archetype in _NO_REPRESENTATIVE_ARCHETYPES:
            return {}
        result: dict[int, int] = {}
        for card_id, median in pool.items():
            remaining = median - self._observed.get(card_id, 0)
            if remaining > 0:
                result[card_id] = remaining
        return result

    def _zone_sizes(self) -> dict[str, int]:
        return {"deck": self._deck_count, "hand": self._hand_count, "prize": self._prize_count}

    # ------------------------------------------------------------------
    # 周辺化（マージナル確率）

    def marginals(self) -> dict[int, dict[str, float]]:
        """card_id -> {"deck": p, "hand": p, "prize": p}。

        各確率は「この card_id が少なくとも1枚そのゾーンにある確率」を、アーキタイプ事後分布で
        周辺化したもの: ``p(card in zone) = Σ_archetype P(archetype) * P(card in zone | archetype)``。
        ``P(archetype)`` は ``_smoothed_weight`` 適用後に再正規化した重み。
        ``P(card in zone | archetype)`` は、未観測プール ``M`` と隠しゾーン合計 ``Z`` の不一致を
        ``M_eff = max(M, Z)`` で吸収したうえでの2値超幾何 ``prob_in_prize(M_eff, zone_size, k)``。

        なお ``"hand"``（手札）のみ、周辺化後に ``_shrink_hand_confidence`` で過信を慎重側へ丸める
        （モジュール冒頭の解説参照。手札は一様配分仮定のプレイ偏りと代表リスト誤差で系統的に過信する）。
        ``"deck"`` / ``"prize"`` は素の超幾何のまま（検証でナイーブ基準に対し悪化しておらず補正不要）。
        この補正により、1枚しか無いカードでも3ゾーン確率の合計は厳密には1にならなくなる（各ゾーン確率は
        元々「そのゾーンに少なくとも1枚ある独立確率」であり、手札だけ保守的に縮めるため）。

        未ロード（``is_ready=False``）や情報が無いときは空 dict を返す（安全なフォールバック）。
        """
        if not self._is_ready:
            return {}

        weights = self._smoothed_normalized_weights()
        zone_sizes = self._zone_sizes()
        zone_total = sum(zone_sizes.values())
        if not weights or zone_total <= 0:
            return {}

        result: dict[int, dict[str, float]] = {}
        for archetype, weight in weights.items():
            unobserved = self._unobserved_pool(archetype)
            if not unobserved:
                # 代表リスト無し（other 等）は特定 card_id に寄与しない（全ゾーンが「不明カード」）。
                continue
            pool_size = sum(unobserved.values())
            eff_pool = max(pool_size, zone_total)
            for card_id, k in unobserved.items():
                zone_probs = result.setdefault(card_id, {"deck": 0.0, "hand": 0.0, "prize": 0.0})
                for zone, size in zone_sizes.items():
                    if size <= 0:
                        continue
                    zone_probs[zone] += weight * prob_in_prize(eff_pool, size, k, at_least=1)

        # 周辺化後、手札確率だけを慎重側へ丸める（過信補正。モジュール冒頭の解説参照）。
        for zone_probs in result.values():
            zone_probs["hand"] = _shrink_hand_confidence(zone_probs["hand"])

        return result

    # ------------------------------------------------------------------
    # サンプリング（決定化）

    def sample(self, rng: random.Random | None = None) -> tuple[list[int | None], list[int | None], list[int | None]]:
        """(deck_card_ids, hand_card_ids, prize_card_ids) を1組返す。2段サンプリング。

        ① ``_smoothed_weight`` で重み付け・正規化した事後分布からアーキタイプを1つ引く。
        ② そのアーキタイプの未観測プール（代表リストの median − 観測済み）を、``deckCount :
        handCount : prizeCount`` の比で非復元抽出してゾーンへ配分する。

        プールサイズ ``M`` とゾーン合計 ``Z`` の不一致（モジュール docstring 参照）は次で吸収する:
        ``M < Z`` なら不足分を ``None``（不明カード）で埋め、``M > Z`` なら ``Z`` 枚だけ抽出して超過分を
        捨てる。返す各リスト長は必ずゾーンサイズ（``deckCount`` / ``handCount`` / ``len(prize)``）に一致。

        未ロードや重みが全滅のときは、全ゾーンを ``None`` で埋めた長さ整合のリストを返す
        （例外を投げない安全なフォールバック。``search_begin`` の長さ検証を通せる形を保つ）。
        """
        rng_obj: random.Random = rng if rng is not None else random  # type: ignore[assignment]

        deck_slots, hand_slots, prize_slots = self._deck_count, self._hand_count, self._prize_count
        total_slots = deck_slots + hand_slots + prize_slots

        weights = self._smoothed_normalized_weights() if self._is_ready else {}
        if not weights or total_slots <= 0:
            return (
                [None] * deck_slots,
                [None] * hand_slots,
                [None] * prize_slots,
            )

        # ① アーキタイプを1つ引く。
        archetypes = list(weights.keys())
        probs = [weights[a] for a in archetypes]
        chosen = rng_obj.choices(archetypes, weights=probs, k=1)[0]

        # ② 未観測プールを作り、ゾーンへ配分する。
        unobserved = self._unobserved_pool(chosen)
        pool: list[int | None] = []
        for card_id, count in unobserved.items():
            pool.extend([card_id] * count)

        if len(pool) < total_slots:
            # M < Z: 不足分を不明カード(None)で補う。
            pool.extend([None] * (total_slots - len(pool)))
        # M >= Z: 下の rng.sample(pool, total_slots) が Z 枚だけ非復元抽出し、超過分は捨てられる。

        drawn = rng_obj.sample(pool, total_slots)
        deck = drawn[:deck_slots]
        hand = drawn[deck_slots : deck_slots + hand_slots]
        prize = drawn[deck_slots + hand_slots :]
        return deck, hand, prize
