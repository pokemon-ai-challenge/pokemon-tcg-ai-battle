"""opponent_hidden_state.OpponentHiddenState のユニットテスト。

archetype_card_pool.json は kaggle_replays/deck_predictor/build_archetype_pool.py の成果物であり、
本テストでは tmp_path に合成した小さなプールJSON（2〜3アーキタイプ・数枚のカードのみ）を書き出し、
それを pool_path として渡す形で検証する（test_hybrid_predictor.py と同じスタイル。本物の21アーキタイプ・
数百カードのプールは使わず、ロジックだけを狭い語彙で検証する）。

検証観点（実装プラン Phase 2 のテスト方針）:
  1. marginals(): 単一アーキタイプに posterior 100% のとき超幾何の期待値と一致する。
  2. スムージング: 代表リストに無いカードを観測してもアーキタイプ寄与が正のまま残る（0にしない）。
  3. sample(): 各ゾーン長がゾーンサイズと一致し、観測済みカードが二重計上されない。
  4. is_ready: プールJSONが無いとき False になり、sample()/marginals() が安全なフォールバックを返す。
加えて数理面（marginals のゾーン確率が d/M, h/M, p/M に比例し合計が k 枚分で整合すること、
アーキタイプ周辺化が正規化済みの重み付き和であること、プールとゾーン合計の不一致の扱い）を検証する。
"""

from pathlib import Path
import json
import math
import random
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.hidden_information.opponent_hidden_state import (
    OpponentHiddenState,
    _HAND_CONFIDENCE_FLOOR,
    _HAND_EXCESS_KEEP,
    _POSTERIOR_UNIFORM_MIX,
    _SMOOTHING_DECAY_PER_MISS,
    _SMOOTHING_FLOOR_RATIO,
    _shrink_hand_confidence,
)
from ptcg_ai.hidden_information import zone_math


def _hand_calibrated(raw: float) -> float:
    """テスト側で marginals の手札補正（_shrink_hand_confidence）を再現するヘルパー。"""
    if raw <= _HAND_CONFIDENCE_FLOOR:
        return raw
    return _HAND_CONFIDENCE_FLOOR + (raw - _HAND_CONFIDENCE_FLOOR) * _HAND_EXCESS_KEEP


# ----------------------------------------------------------------------
# テスト用ヘルパー


class _StubPlayerState:
    """OpponentHiddenState が参照する3フィールドだけを持つ最小スタブ。

    実コードは deckCount / handCount / len(prize) しか見ないので、フル PlayerState を組む必要はない。
    prize は「伏せカード=None」のリスト（相手のサイドは中身が伏せられているため）。
    """

    def __init__(self, deck_count: int, hand_count: int, prize_count: int) -> None:
        self.deckCount = deck_count
        self.handCount = hand_count
        self.prize = [None] * prize_count


def _write_pool(tmp_path: Path, archetypes: dict[str, dict[int, int]], filename="archetype_card_pool.json") -> Path:
    """archetypes = {アーキタイプ名: {card_id: median枚数}} から合成プールJSONを書き出す。"""
    payload = {
        "meta": {"built_at": "test", "n_archetypes": len(archetypes)},
        "archetypes": {
            name: {
                "card_counts": {
                    str(card_id): {"median": median, "inclusion_rate": 1.0}
                    for card_id, median in cards.items()
                }
            }
            for name, cards in archetypes.items()
        },
    }
    pool_path = tmp_path / filename
    with pool_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return pool_path


# ----------------------------------------------------------------------
# 観点1 + 数理: marginals の超幾何期待値・d/M 比例・合計整合


def test_marginals_single_archetype_matches_hypergeometric(tmp_path):
    # 代表リスト: 10種×1枚 = プール10枚。ゾーン deck=5 / hand=3 / prize=2、合計 Z=10=M。
    cards = {100 + i: 1 for i in range(10)}
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    assert state.is_ready is True

    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(5, 3, 2))
    marg = state.marginals()

    # 1枚しか無いカードは deck/prize が d/M, p/M（素の超幾何）に一致する。手札のみ過信補正が
    # かかるため d/M ではなく _shrink_hand_confidence(3/10) に一致する（Phase 3 差し戻し対応）。
    for card_id in cards:
        m = marg[card_id]
        assert math.isclose(m["deck"], 5 / 10, abs_tol=1e-12)
        assert math.isclose(m["prize"], 2 / 10, abs_tol=1e-12)
        # 手札は素の 3/10=0.3 を FLOOR 超のぶん割り引いた保守値（0.3 > FLOOR=0.15 なので縮む）。
        assert math.isclose(m["hand"], _hand_calibrated(3 / 10), abs_tol=1e-12)
        assert m["hand"] < 3 / 10  # 補正は常に慎重側（生の見積もりより強気にしない）
        # deck は「少なくとも1枚あるゾーン確率」= 超幾何の閉形式と一致（手札補正の影響を受けない）。
        assert math.isclose(m["deck"], zone_math.prob_in_prize(10, 5, 1), abs_tol=1e-12)


def test_hand_confidence_shrink_is_conservative_and_monotone():
    # FLOOR 以下は素通し、超えた分は縮み、常に p' <= p（強気側へ動かさない）。単調増加も保つ。
    assert _shrink_hand_confidence(0.0) == 0.0
    assert math.isclose(_shrink_hand_confidence(_HAND_CONFIDENCE_FLOOR), _HAND_CONFIDENCE_FLOOR)
    prev = -1.0
    for i in range(101):
        p = i / 100
        s = _shrink_hand_confidence(p)
        assert s <= p + 1e-12  # 生の見積もりより強気にしない
        assert s >= prev - 1e-12  # 単調（順位＝識別力を壊さない）
        prev = s
    # FLOOR を超えた点は厳密に縮む。
    assert _shrink_hand_confidence(0.9) < 0.9


def test_marginals_only_hand_is_shrunk_deck_prize_raw(tmp_path):
    # 手札のみ補正がかかり、deck/prize は素の超幾何のままであることを明示的に確認する。
    cards = {100 + i: 1 for i in range(10)}
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(5, 3, 2))
    marg = state.marginals()
    m = marg[100]
    assert math.isclose(m["deck"], 5 / 10, abs_tol=1e-12)  # 無補正
    assert math.isclose(m["prize"], 2 / 10, abs_tol=1e-12)  # 無補正
    assert math.isclose(m["hand"], _hand_calibrated(3 / 10), abs_tol=1e-12)  # 補正あり


def test_marginals_multi_copy_card_deck_probability(tmp_path):
    # 4-of カード(200) + 6枚の埋め草 = プール10枚。ゾーン deck=5/hand=3/prize=2。
    cards = {200: 4}
    cards.update({300 + i: 1 for i in range(6)})
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(5, 3, 2))
    marg = state.marginals()

    # 4枚あるカードが「少なくとも1枚 deck にある確率」= 1 - P(0枚 deck) = 1 - C(6,5)/C(10,5)。
    expected_deck = zone_math.prob_in_prize(10, 5, 4, at_least=1)
    assert math.isclose(marg[200]["deck"], expected_deck, abs_tol=1e-12)
    # 期待枚数はゾーン合計が k(=4) に一致する（保存則）。
    exp = zone_math.expected_in_zone(10, {"deck": 5, "hand": 3, "prize": 2}, 4)
    assert math.isclose(sum(exp.values()), 4.0, abs_tol=1e-12)


def test_zone_math_multivariate_helpers_conserve_count():
    # d+h+p == M のとき、期待枚数の合計は target_count に一致する。
    exp = zone_math.expected_in_zone(20, {"deck": 12, "hand": 5, "prize": 3}, 4)
    assert math.isclose(exp["deck"], 4 * 12 / 20, abs_tol=1e-12)
    assert math.isclose(exp["hand"], 4 * 5 / 20, abs_tol=1e-12)
    assert math.isclose(exp["prize"], 4 * 3 / 20, abs_tol=1e-12)
    assert math.isclose(sum(exp.values()), 4.0, abs_tol=1e-12)
    # prob_in_zone は各ゾーン「そのゾーン vs その他」の2値超幾何に一致する。
    probs = zone_math.prob_in_zone(20, {"deck": 12, "hand": 5, "prize": 3}, 1)
    assert math.isclose(probs["deck"], 12 / 20, abs_tol=1e-12)
    assert math.isclose(probs["hand"], 5 / 20, abs_tol=1e-12)
    assert math.isclose(probs["prize"], 3 / 20, abs_tol=1e-12)


# ----------------------------------------------------------------------
# 観点: アーキタイプ周辺化が正規化済みの重み付き和になっている


def test_marginals_is_normalized_weighted_sum_over_archetypes(tmp_path):
    # A と B は互いに素な代表リスト（各 10枚）。posterior は非正規（合計 != 1）で渡し、
    # 内部で正規化されることを確認する。
    a_cards = {100 + i: 1 for i in range(10)}
    b_cards = {200 + i: 1 for i in range(10)}
    pool_path = _write_pool(tmp_path, {"A": a_cards, "B": b_cards})
    state = OpponentHiddenState(pool_path=pool_path)
    # わざと合計 0.8 の非正規 posterior。正規化後は A=0.75, B=0.25。その後、一様分布へ
    # _POSTERIOR_UNIFORM_MIX だけ混ぜる（K=2 なので各アーキタイプへ mix/2 を上乗せ）。
    state.update({"A": 0.6, "B": 0.2}, observed_card_ids={}, player_state=_StubPlayerState(5, 3, 2))
    marg = state.marginals()

    mix = _POSTERIOR_UNIFORM_MIX
    pa = (1.0 - mix) * 0.75 + mix / 2  # 混合後 P(A)
    pb = (1.0 - mix) * 0.25 + mix / 2  # 混合後 P(B)
    # A固有カードの deck 確率 = P(A)*P(deck|A)。P(deck|A)=5/10。
    assert math.isclose(marg[100]["deck"], pa * (5 / 10), abs_tol=1e-12)
    assert math.isclose(marg[200]["deck"], pb * (5 / 10), abs_tol=1e-12)


# ----------------------------------------------------------------------
# 観点2: スムージング（代表リストに無いカードを観測してもゼロにしない）


def test_smoothing_decays_but_never_zeroes_weight(tmp_path):
    a_cards = {100 + i: 1 for i in range(10)}
    b_cards = {200 + i: 1 for i in range(10)}
    pool_path = _write_pool(tmp_path, {"A": a_cards, "B": b_cards})
    state = OpponentHiddenState(pool_path=pool_path)

    # A・B に posterior 0.5/0.5。観測カード 999 は A にも B にも無い（両方 miss=1 で等しく減衰）。
    observed = {999: 1}
    state.update({"A": 0.5, "B": 0.5}, observed_card_ids=observed, player_state=_StubPlayerState(5, 3, 2))

    wa = state._smoothed_weight("A", observed)
    wb = state._smoothed_weight("B", observed)
    # miss=1 -> factor _SMOOTHING_DECAY_PER_MISS。元の 0.5 から減衰しているが 0 ではない。
    assert math.isclose(wa, 0.5 * _SMOOTHING_DECAY_PER_MISS, abs_tol=1e-12)
    assert wa > 0.0 and wb > 0.0

    # A固有カードの marginal も正のまま残る（アーキタイプが候補から消えていない）。
    marg = state.marginals()
    assert marg[100]["deck"] > 0.0


def test_smoothing_floor_prevents_zero_even_with_many_misses(tmp_path):
    a_cards = {100 + i: 1 for i in range(10)}
    pool_path = _write_pool(tmp_path, {"A": a_cards})
    state = OpponentHiddenState(pool_path=pool_path)

    # 代表リストに無いカードを大量に観測（miss=12）。decay^12 は floor を下回るので floor が効く。
    observed = {900 + i: 1 for i in range(12)}
    state.update({"A": 1.0}, observed_card_ids=observed, player_state=_StubPlayerState(5, 3, 2))
    wa = state._smoothed_weight("A", observed)
    assert _SMOOTHING_DECAY_PER_MISS ** 12 < _SMOOTHING_FLOOR_RATIO  # 前提: floor が効く miss 数
    assert math.isclose(wa, 1.0 * _SMOOTHING_FLOOR_RATIO, abs_tol=1e-12)  # floor まで下がるが 0 ではない
    assert wa > 0.0


def test_other_archetype_is_not_decayed(tmp_path):
    # other は代表リストを持たない特別扱い。observed に何があっても減衰しない（生の重みのまま）。
    pool_path = _write_pool(tmp_path, {"A": {100: 1}, "other": {}})
    state = OpponentHiddenState(pool_path=pool_path)
    observed = {999: 1}
    state.update({"A": 0.5, "other": 0.5}, observed_card_ids=observed, player_state=_StubPlayerState(5, 3, 2))
    assert math.isclose(state._smoothed_weight("other", observed), 0.5, abs_tol=1e-12)


# ----------------------------------------------------------------------
# 観点3: sample() のゾーン長一致・観測二重計上なし


def test_sample_zone_lengths_match_player_state(tmp_path):
    # 代表リスト20枚。ゾーン deck=8/hand=4/prize=2、合計 Z=14 < M=20（超過分は捨てられる）。
    cards = {100 + i: 1 for i in range(20)}
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(8, 4, 2))

    rng = random.Random(0)
    for _ in range(50):
        deck, hand, prize = state.sample(rng)
        assert len(deck) == 8
        assert len(hand) == 4
        assert len(prize) == 2
        # M > Z なので全て実カード（None は入らない）。
        assert all(cid is not None for cid in deck + hand + prize)


def test_sample_does_not_double_count_observed(tmp_path):
    # 4-of カード(200)。観測済み 1枚 -> 未観測プールには 3枚だけ残る。
    cards = {200: 4}
    cards.update({300 + i: 1 for i in range(6)})  # 埋め草6枚 -> 代表計10枚
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    # ゾーン deck=5/hand=3/prize=2 = 10。観測 200 を1枚 -> 未観測プール 9枚 < Z=10 -> 1枚 None 補填。
    state.update({"A": 1.0}, observed_card_ids={200: 1}, player_state=_StubPlayerState(5, 3, 2))

    rng = random.Random(1)
    for _ in range(200):
        deck, hand, prize = state.sample(rng)
        drawn = deck + hand + prize
        # サンプルされた 200 の枚数は未観測残(=3)を超えない。観測1 + サンプル<=3 <= median4。
        assert drawn.count(200) <= 3
        # 各ゾーン長は不変。
        assert (len(deck), len(hand), len(prize)) == (5, 3, 2)


def test_sample_pads_with_none_when_pool_smaller_than_zones(tmp_path):
    # 代表リスト4枚のみ。ゾーン合計 Z=6 > M=4 -> 2枚は不明カード(None)で補填される。
    cards = {100: 1, 101: 1, 102: 1, 103: 1}
    pool_path = _write_pool(tmp_path, {"A": cards})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(3, 2, 1))

    rng = random.Random(2)
    for _ in range(50):
        deck, hand, prize = state.sample(rng)
        drawn = deck + hand + prize
        assert len(drawn) == 6
        assert drawn.count(None) == 2  # Z - M = 2 枚が None
        real = [c for c in drawn if c is not None]
        assert len(real) == 4
        assert sorted(real) == [100, 101, 102, 103]  # 実カード4枚は全て過不足なく現れる


# ----------------------------------------------------------------------
# 観点4: is_ready False のフォールバック


def test_missing_pool_file_is_not_ready(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    state = OpponentHiddenState(pool_path=missing)
    assert state.is_ready is False

    state.update({"A": 1.0}, observed_card_ids={}, player_state=_StubPlayerState(5, 3, 2))
    # marginals は空、sample は長さ整合の None リスト（例外を投げない）。
    assert state.marginals() == {}
    deck, hand, prize = state.sample(random.Random(0))
    assert deck == [None] * 5
    assert hand == [None] * 3
    assert prize == [None] * 2


def test_ready_but_no_matching_posterior_weight_falls_back(tmp_path):
    # プールはロード済みだが posterior が全て 0（重み全滅）。sample は None リストにフォールバック。
    pool_path = _write_pool(tmp_path, {"A": {100: 1, 101: 1}})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update({"A": 0.0}, observed_card_ids={}, player_state=_StubPlayerState(2, 1, 1))
    deck, hand, prize = state.sample(random.Random(0))
    assert deck == [None] * 2 and hand == [None] * 1 and prize == [None] * 1
    assert state.marginals() == {}


def test_sample_before_update_returns_empty(tmp_path):
    pool_path = _write_pool(tmp_path, {"A": {100: 1}})
    state = OpponentHiddenState(pool_path=pool_path)
    # update() を一度も呼んでいない（ゾーンサイズ 0）。
    assert state.sample(random.Random(0)) == ([], [], [])
    assert state.marginals() == {}


# ----------------------------------------------------------------------
# 正規化の健全性


def test_smoothed_weights_normalize_to_one(tmp_path):
    pool_path = _write_pool(tmp_path, {"A": {100: 1}, "B": {200: 1}, "other": {}})
    state = OpponentHiddenState(pool_path=pool_path)
    state.update(
        {"A": 0.5, "B": 0.3, "other": 0.2},
        observed_card_ids={999: 1},
        player_state=_StubPlayerState(5, 3, 2),
    )
    weights = state._smoothed_normalized_weights()
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    assert all(w > 0.0 for w in weights.values())
