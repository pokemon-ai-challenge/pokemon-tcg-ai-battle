"""card_vocab.py の単体テスト(T0.1)。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cv():
    try:
        from ptcg_ai.learning import card_vocab as _cv
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"unavailable: {exc}")
    return _cv


def test_pad_and_unk_reserved(cv):
    assert cv.PAD_INDEX == 0
    assert cv.UNK_INDEX == 1


def test_load_committed_vocab_matches_engine(cv):
    """コミット済みcard_vocab.jsonが現在のcgエンジンのカード集合と一致すること
    (競合期間中にカードが増えていないかの検出)。"""
    from cg.api import all_card_data
    vocab = cv.load_vocab()
    engine_ids = sorted({c.cardId for c in all_card_data()})
    assert vocab.card_ids == engine_ids


def test_index_of_known_and_unknown(cv):
    vocab = cv.load_vocab()
    known = vocab.card_ids[0]
    assert vocab.index_of(known) >= 2
    assert vocab.index_of(999999) == cv.UNK_INDEX
    assert vocab.index_of(None) == cv.UNK_INDEX


def test_indices_are_unique_and_within_size(cv):
    vocab = cv.load_vocab()
    indices = [vocab.index_of(cid) for cid in vocab.card_ids]
    assert len(set(indices)) == len(indices)
    assert all(2 <= i < vocab.size for i in indices)


def test_hash_detects_tampering(cv, tmp_path):
    vocab = cv.load_vocab()
    path = tmp_path / "tampered.json"
    payload = {"version": vocab.version, "hash": vocab.hash, "card_ids": vocab.card_ids + [999999]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        cv.load_vocab(path)


def test_missing_file_raises(cv, tmp_path):
    with pytest.raises(FileNotFoundError):
        cv.load_vocab(tmp_path / "missing.json")


def test_extend_vocab_no_new_cards_returns_identical(cv):
    """新規カードが無ければ既存vocabと完全に同じ(バージョン・hashも不変)。"""
    existing = cv.load_vocab()
    extended = cv.extend_vocab()
    assert extended.card_ids == existing.card_ids
    assert extended.version == existing.version
    assert extended.hash == existing.hash


def test_extend_vocab_appends_new_cards_without_reordering(cv, tmp_path, monkeypatch):
    """新規カードは末尾に追加され、既存カードのindexは変わらない(append-only)。"""
    base = cv.CardVocab([5, 1, 3], version=1, vocab_hash=cv._compute_hash([5, 1, 3]))
    path = tmp_path / "v.json"
    cv.save_vocab(base, path)

    class _FakeCard:
        def __init__(self, cid):
            self.cardId = cid

    def fake_all_card_data():
        return [_FakeCard(c) for c in (5, 1, 3, 2, 9)]  # 2, 9 が新規

    import cg.api
    monkeypatch.setattr(cg.api, "all_card_data", fake_all_card_data)

    extended = cv.extend_vocab(path)
    assert extended.card_ids == [5, 1, 3, 2, 9]  # 既存[5,1,3]の並びは不変、新規は末尾に昇順
    assert extended.index_of(5) == base.index_of(5)
    assert extended.index_of(1) == base.index_of(1)
    assert extended.index_of(3) == base.index_of(3)
    assert extended.version == base.version + 1


def test_hash_depends_on_order(cv):
    """並び順が変わればhashも変わる(sorted()を挟まない)。"""
    h1 = cv._compute_hash([1, 2, 3])
    h2 = cv._compute_hash([3, 2, 1])
    assert h1 != h2


def test_save_and_load_round_trip(cv, tmp_path):
    vocab = cv.CardVocab([5, 1, 3], version=1, vocab_hash=cv._compute_hash([5, 1, 3]))
    path = tmp_path / "v.json"
    cv.save_vocab(vocab, path)
    reloaded = cv.load_vocab(path)
    assert reloaded.card_ids == [5, 1, 3]
    assert reloaded.index_of(5) == 2
    assert reloaded.index_of(1) == 3
    assert reloaded.index_of(3) == 4
