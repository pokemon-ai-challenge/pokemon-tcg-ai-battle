"""Card ID embedding vocabulary(T0.1/T1.1)。学習時・提出時で同一の語彙を読み込むための
唯一の定義元。board側・option側の両方でこの語彙を使う(T1.1でraw card idを直接
Embedding indexとして使う設計をやめ、盤面・選択肢の両方をこの語彙で統一した)。

**raw Card IDが連続値であるという仮定は置かない。** ``card_vocab.json`` に
card idのリスト(順序が意味を持つ)を固定して持つ。0=PAD・1=UNKを予約し、
リストの位置 + 2 が embedding index になる。

**append-only。既存indexは永久に変わらない。** ``card_vocab.json`` の
``card_ids`` は**並べ替えない**。新しいカードが増えたときは ``extend_vocab()`` が
既存リストの末尾に新規カードだけを追加する(既存カードの位置は一切動かさない)。
これにより、一度学習・提出した重みのembedding indexが、カードデータの更新後も
永久に有効であることを保証する。

hashは**リストの並び順を含めて**計算する(順序が変わればhashも変わる。
中身の集合が同じでも順序が違えば別物として検出するため、``sorted()``を挟まない)。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAD_INDEX = 0
UNK_INDEX = 1
_RESERVED = 2  # PAD, UNK

_DEFAULT_PATH = Path(__file__).resolve().parent / "card_vocab.json"
_VERSION = 1


class CardVocab:
    def __init__(self, card_ids: list[int], version: int, vocab_hash: str):
        self.card_ids = list(card_ids)
        self.version = version
        self.hash = vocab_hash
        self._index = {cid: i + _RESERVED for i, cid in enumerate(self.card_ids)}

    @property
    def size(self) -> int:
        """embeddingテーブルのサイズ(PAD/UNK込み)。"""
        return len(self.card_ids) + _RESERVED

    def index_of(self, card_id: int | None) -> int:
        """raw card id -> embedding index。``None``/未登録は ``UNK_INDEX``。"""
        if card_id is None:
            return UNK_INDEX
        return self._index.get(int(card_id), UNK_INDEX)


def _compute_hash(card_ids: list[int]) -> str:
    """並び順を含めてhashする(``sorted()``しない。順序変化を検出するため)。"""
    payload = json.dumps(list(card_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_vocab_from_engine() -> CardVocab:
    """cgエンジンの ``all_card_data()`` から**初回生成専用**の語彙を作る(ファイルI/O無し)。

    既存の ``card_vocab.json`` がある場合はこちらではなく ``extend_vocab()`` を使うこと
    (このヘルパーは順序を昇順ソートで新規に決めるため、既存ファイルの上書きに使うと
    append-only保証が壊れる)。
    """
    from cg.api import all_card_data
    ids = sorted({c.cardId for c in all_card_data()})
    return CardVocab(ids, _VERSION, _compute_hash(ids))


def save_vocab(vocab: CardVocab, path: str | Path | None = None) -> Path:
    path = Path(path) if path else _DEFAULT_PATH
    payload = {"version": vocab.version, "hash": vocab.hash, "card_ids": vocab.card_ids}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_vocab(path: str | Path | None = None) -> CardVocab:
    """固定されたJSONファイルから語彙を読み込む(cgエンジン不要。学習・提出両方で
    同じファイルを読めば同じ語彙になる)。"""
    path = Path(path) if path else _DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"card_vocab.jsonが無い: {path}\n"
            "  card_vocab.build_vocab_from_engine() + save_vocab() で一度生成してコミットする。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(x) for x in payload["card_ids"]]
    expected_hash = _compute_hash(ids)
    if expected_hash != payload["hash"]:
        raise ValueError(
            f"card_vocab.jsonのhashが内容と一致しない({path})。ファイルが壊れているか、"
            "手動編集された可能性がある。")
    return CardVocab(ids, int(payload["version"]), payload["hash"])


def extend_vocab(path: str | Path | None = None) -> CardVocab:
    """既存 ``card_vocab.json`` の並び順を一切変えず、現在のcgエンジンにあって
    まだ語彙に無いカードだけを**末尾に追加**した新しい ``CardVocab`` を返す
    (ファイルへの書き込みはしない。呼び出し側が ``save_vocab()`` で確定する)。

    既存ファイルが無ければ ``build_vocab_from_engine()`` と同じ(初回生成)。
    新規カードが無ければ既存と全く同じ ``CardVocab``(versionもhashも不変)を返す。
    """
    from cg.api import all_card_data
    engine_ids = {c.cardId for c in all_card_data()}

    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return build_vocab_from_engine()

    existing = load_vocab(p)
    new_ids = sorted(engine_ids - set(existing.card_ids))
    if not new_ids:
        return existing
    merged = existing.card_ids + new_ids  # 既存の並びはそのまま、新規は末尾に昇順で追加
    return CardVocab(merged, existing.version + 1, _compute_hash(merged))


if __name__ == "__main__":
    v = extend_vocab()
    save_vocab(v)
    print(f"card_vocab.json を生成/更新: {len(v.card_ids)}カード + PAD/UNK = size {v.size}, "
          f"version={v.version}, hash={v.hash[:16]}...")
