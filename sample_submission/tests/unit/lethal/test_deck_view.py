"""自分のデッキ情報の境界(Step 1-4 指示 6)。

観点:
1. 構成(multiset)へ落とした時点で**順序は復元できない**
2. 順序を持つ表現(list[int])を探索の内側へ入れられない
3. 診断ログにデッキリストを書けない
4. `repr` に中身が出ない
"""

import pytest

from ptcg_ai.search.lethal.deck_view import KnownDeckComposition, coerce
from ptcg_ai.search.lethal.diagnostics import DiagnosticsLeakError, DiagnosticsRecorder

DECK = [5, 5, 743, 1182, 66, 743]


def test_order_is_discarded_at_construction():
    forward = KnownDeckComposition.from_card_ids(DECK)
    shuffled = KnownDeckComposition.from_card_ids(list(reversed(DECK)))
    assert forward == shuffled
    assert forward.as_counter() == shuffled.as_counter()
    assert forward.total() == len(DECK)


def test_no_public_accessor_returns_an_order():
    composition = KnownDeckComposition.from_card_ids(DECK)
    # counts は (card_id, 枚数) の昇順で、元の並びとは無関係。
    assert composition.counts == ((5, 2), (66, 1), (743, 2), (1182, 1))
    assert not hasattr(composition, "order")
    assert not hasattr(composition, "to_engine_list")


def test_coerce_accepts_list_mapping_and_composition():
    from_list = coerce(DECK)
    from_mapping = coerce({5: 2, 66: 1, 743: 2, 1182: 1})
    assert from_list == from_mapping
    assert coerce(from_list) is from_list
    assert coerce(None) is None


def test_backend_rejects_a_raw_deck_list():
    """順序を持つ表現を探索の内側へ渡そうとしたら型で弾く。"""
    from ptcg_ai.search.lethal.cg_backend import CgBackend

    with pytest.raises(TypeError):
        CgBackend.__init__(
            object.__new__(CgBackend), session=None, obs=None, hidden=None,
            deck_composition=list(DECK),
        )


def test_repr_does_not_expose_the_contents():
    text = repr(KnownDeckComposition.from_card_ids(DECK))
    assert "743" not in text and "1182" not in text
    assert "cards=6" in text


def test_diagnostics_rejects_a_deck_list():
    recorder = DiagnosticsRecorder()
    with pytest.raises(DiagnosticsLeakError):
        recorder.record("search", deck=list(range(60)))
    with pytest.raises(DiagnosticsLeakError):
        recorder.record("search", cards=list(range(60)))
