"""B1 ガード(規則 R4)の単体テスト。

Step 0 の実測では「デッキ公開 → 取得 → SHUFFLE → 以後のドロー」という順序しか
観測されなかったが、それはエンジンの保証ではない。よって実行時に検査する。

観点:
1. デッキが公開されていない経路では、ドローがあってもガードは発火しない
2. root でデッキが公開されている(B1 の局面)なら、SHUFFLE 前のドローで発火する
3. SHUFFLE を挟めば発火しない
4. 経路の途中で公開された場合も同じ
5. 1ステップ内のログ順序(draw と shuffle のどちらが先か)を正しく見る
6. 再公開されたら「シャッフル済み」はリセットされる
"""

import pytest

from ptcg_ai.search.lethal.chance import (
    RevealState,
    advance_reveal_state,
    check_reveal_guard,
)
from ptcg_ai.search.lethal.engine import COIN, DRAW, SHUFFLE, StepEvents
from ptcg_ai.search.lethal.types import StopReason


def events(*sequence, deck_listed_before=False, drawn=(1,)):
    return StepEvents(
        sequence=tuple(sequence),
        drawn=tuple(drawn),
        deck_listed_before=deck_listed_before,
    )


def test_draw_without_reveal_is_allowed():
    # 観点1: 通常のドロー(供給順序が効いている)は当然許される。
    assert check_reveal_guard([events(DRAW)]) is None


def test_draw_after_root_reveal_without_shuffle_is_rejected():
    # 観点2: B1 の核心。実デッキが使われている状態で、シャッフル前に引いたら止める。
    reason = check_reveal_guard([events(DRAW)], deck_revealed_at_root=True)
    assert reason is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL


def test_shuffle_before_draw_after_reveal_is_allowed():
    # 観点3: 実測で観測された順序。ここは通す。
    assert check_reveal_guard(
        [events(SHUFFLE), events(DRAW)], deck_revealed_at_root=True
    ) is None


def test_reveal_in_the_middle_of_the_path_is_tracked():
    # 観点4: 経路の途中でデッキ公開選択に答えた場合。
    path = [events(), events(DRAW, deck_listed_before=True)]
    assert check_reveal_guard(path) is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL

    path_ok = [events(), events(SHUFFLE, deck_listed_before=True), events(DRAW)]
    assert check_reveal_guard(path_ok) is None


def test_log_order_within_one_step_matters():
    # 観点5: 同じ1ステップでも draw が shuffle より前なら違反。
    assert check_reveal_guard(
        [events(DRAW, SHUFFLE)], deck_revealed_at_root=True
    ) is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL
    assert check_reveal_guard(
        [events(SHUFFLE, DRAW)], deck_revealed_at_root=True
    ) is None


def test_second_reveal_resets_the_shuffled_flag():
    # 観点6: 一度シャッフルされても、再公開されたらまた実順序を見ている状態に戻る。
    path = [
        events(SHUFFLE, deck_listed_before=True),
        events(DRAW, deck_listed_before=True),
    ]
    assert check_reveal_guard(path) is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL


def test_advance_returns_updated_state():
    state = RevealState.at_root(True)
    assert state.deck_revealed and not state.shuffled_since_reveal
    state, violation = advance_reveal_state(state, events(SHUFFLE, COIN, drawn=()))
    assert violation is None
    assert state.shuffled_since_reveal
    state, violation = advance_reveal_state(state, events(DRAW))
    assert violation is None


@pytest.mark.parametrize("revealed", [True, False])
def test_guard_is_pure(revealed):
    # 状態は frozen dataclass。呼んでも入力を書き換えない。
    original = RevealState.at_root(revealed)
    advance_reveal_state(original, events(SHUFFLE))
    assert original == RevealState.at_root(revealed)


# ------------------------- B1: 指示された 2 系列を明示的に回帰テストにする


def test_b1_sequence_reveal_move_shuffle_draw_is_allowed():
    """DECK_REVEAL → MOVE_CARD → SHUFFLE → DRAW（実測された順序）は通す。"""
    path = [
        events(deck_listed_before=True, drawn=()),   # デッキ公開選択に答えた
        events(SHUFFLE, drawn=()),                   # サーチ後のシャッフル
        events(DRAW),                                # その後のドロー
    ]
    assert check_reveal_guard(path) is None


def test_b1_sequence_reveal_move_draw_is_rejected():
    """DECK_REVEAL → MOVE_CARD → DRAW（シャッフル無し）は絶対に通さない。

    実エンジンでは今のところ観測されていないが、**観測されなかったことを
    保証として扱わない**ためのガード。人工的な系列で発火を確認する。
    """
    path = [
        events(deck_listed_before=True, drawn=()),
        events(DRAW),
    ]
    assert check_reveal_guard(path) is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL


def test_b1_sequence_within_one_step_reveal_then_draw_is_rejected():
    """1 ステップ内で「公開 → ドロー」が起きた場合も弾く。"""
    assert check_reveal_guard(
        [events(DRAW, deck_listed_before=True)]
    ) is StopReason.DRAW_BEFORE_SHUFFLE_AFTER_REVEAL
