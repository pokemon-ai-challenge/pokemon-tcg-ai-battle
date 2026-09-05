"""`ml_policy_agent` の「自分の60枚」宣言口(ローカル評価harness専用)のユニットテスト。

背景: 既定の dummy 経路(`build_dummy_search_state(obs, _get_deck())`)は CWD の `deck.csv`
を自分の山札とみなす。ローカルの `run_match.play_match` / `measurement.runner.play_game` は
1プロセスで両陣営を動かしデッキをエンジンへ直接渡すため、deck.csv 以外を持つ側は
hidden_state が必ず None になり、`lethal_search` と `ko_search` 系ガードが一度も発火しない。
`set_own_deck_override` はその宣言口(本番は誰も呼ばない=挙動不変)。

ここで守る性質:
  1. 未宣言(=本番)では `_get_deck_for` は `_get_deck()` と完全に同一のものを返す。
  2. 宣言すると `obs.current.yourIndex` に応じて陣営ごとの60枚を返す。
  3. 宣言していない player_index / `obs.current` が無い obs は既定へフォールバックする。
  4. `clear_own_deck_override()` / `None` 指定で解除でき、既定へ戻る。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from ptcg_ai.ml_policy import ml_policy_agent as mpa


class _FakeState:
    def __init__(self, your_index: int) -> None:
        self.yourIndex = your_index


class _FakeObs:
    """`_get_deck_for` が触るのは `obs.current.yourIndex` だけなので最小のスタブで足りる。"""

    def __init__(self, your_index: int | None) -> None:
        self.current = None if your_index is None else _FakeState(your_index)


@pytest.fixture(autouse=True)
def _isolate_override_and_deck_cache(monkeypatch):
    """モジュールグローバルを汚さない。deck.csv への依存も切る(CWD 非依存にするため)。"""
    monkeypatch.setattr(mpa, "_own_deck_override", {}, raising=True)
    monkeypatch.setattr(mpa, "_deck_cache", [999] * 60, raising=True)
    yield


def test_default_returns_deck_csv_deck():
    """未宣言(=本番)では `_get_deck()` と同一オブジェクトを返す(挙動不変)。"""
    assert mpa._get_deck_for(_FakeObs(0)) is mpa._get_deck()
    assert mpa._get_deck_for(_FakeObs(1)) is mpa._get_deck()


def test_override_is_per_player_index():
    """宣言すると陣営ごとに違う60枚が返る(1プロセス2エージェントの分離)。"""
    deck0 = [11] * 60
    deck1 = [22] * 60
    mpa.set_own_deck_override(0, deck0)
    mpa.set_own_deck_override(1, deck1)

    assert mpa._get_deck_for(_FakeObs(0)) == deck0
    assert mpa._get_deck_for(_FakeObs(1)) == deck1
    # 渡したリストのコピーを保持する(呼び出し側があとで書き換えても影響しない)。
    deck0.append(33)
    assert len(mpa._get_deck_for(_FakeObs(0))) == 60


def test_unknown_player_index_and_missing_state_fall_back():
    """宣言のない陣営・`current` の無い obs は既定(deck.csv)へフォールバックする。"""
    mpa.set_own_deck_override(0, [11] * 60)

    assert mpa._get_deck_for(_FakeObs(1)) is mpa._get_deck()
    assert mpa._get_deck_for(_FakeObs(None)) is mpa._get_deck()
    assert mpa._get_deck_for(None) is mpa._get_deck()


def test_clear_and_none_release_the_override():
    mpa.set_own_deck_override(0, [11] * 60)
    mpa.set_own_deck_override(1, [22] * 60)

    mpa.set_own_deck_override(0, None)
    assert mpa._get_deck_for(_FakeObs(0)) is mpa._get_deck()
    assert mpa._get_deck_for(_FakeObs(1)) == [22] * 60

    mpa.clear_own_deck_override()
    assert mpa._get_deck_for(_FakeObs(1)) is mpa._get_deck()
    assert mpa._own_deck_override == {}
