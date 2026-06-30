"""Deck discovery / loading for the viewer.

A "deck" is just a list of 60 card IDs. Decks are discovered from two places:

  * viewer/decks/*.csv   -- drop a CSV here to add a deck (auto-discovered)
  * sample_submission/deck.csv -- always registered as the default deck

CSV format: one card ID per line (commas also accepted). Blank lines and lines
starting with '#' are ignored, EXCEPT an optional first '# <label>' line which
sets the deck's Japanese display label.

This module performs no engine calls, so it is cheap to import.
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
DECKS_DIR = os.path.join(_HERE, "decks")
DEFAULT_DECK_PATH = os.path.join(_REPO, "sample_submission", "deck.csv")
DEFAULT_DECK_ID = "dragapult_default"
DEFAULT_DECK_LABEL = "Dragapult ex（既定）"


def _parse_csv(path: str) -> tuple[str | None, list[int]]:
    """Return (label_or_None, card_ids) parsed from a deck CSV."""
    label: str | None = None
    ids: list[int] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f.read().splitlines()):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # The first comment line acts as the display label.
                if label is None:
                    label = line.lstrip("#").strip() or None
                continue
            for token in line.replace(",", " ").split():
                try:
                    ids.append(int(token))
                except ValueError:
                    pass
    return label, ids


def _deck_entry(deck_id: str, path: str, fallback_label: str) -> dict:
    try:
        label, ids = _parse_csv(path)
    except OSError:
        return {"id": deck_id, "label": fallback_label, "count": 0, "valid": False,
                "path": path, "error": "読み込み失敗"}
    valid = len(ids) == 60
    err = None if valid else f"カード枚数が60ではありません（{len(ids)}枚）"
    return {
        "id": deck_id,
        "label": label or fallback_label,
        "count": len(ids),
        "valid": valid,
        "path": path,
        "error": err,
    }


def list_decks() -> list[dict]:
    """List all available decks (default + viewer/decks/*.csv).

    Each entry: {id, label, count, valid, path, error}. The default deck is
    always first; user decks follow in filename order.
    """
    decks: list[dict] = []
    if os.path.exists(DEFAULT_DECK_PATH):
        decks.append(_deck_entry(DEFAULT_DECK_ID, DEFAULT_DECK_PATH, DEFAULT_DECK_LABEL))

    if os.path.isdir(DECKS_DIR):
        for name in sorted(os.listdir(DECKS_DIR)):
            if not name.lower().endswith(".csv"):
                continue
            stem = os.path.splitext(name)[0]
            path = os.path.join(DECKS_DIR, name)
            decks.append(_deck_entry(stem, path, stem))
    return decks


def _path_for(deck_id: str) -> str | None:
    for d in list_decks():
        if d["id"] == deck_id:
            return d["path"]
    return None


def load_deck(deck_id: str) -> list[int]:
    """Return the 60 card IDs for a deck id. Raises ValueError if missing/invalid."""
    path = _path_for(deck_id)
    if path is None:
        raise ValueError(f"不明なデッキです: {deck_id}")
    _label, ids = _parse_csv(path)
    if len(ids) < 60:
        raise ValueError(f"デッキ {deck_id} のカードが60枚未満です（{len(ids)}枚）")
    return ids[:60]
