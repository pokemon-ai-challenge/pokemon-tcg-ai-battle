"""Resolve a card id to its card-face image, extracted from the JP card PDF.

Reuses cardlist_referenced/pdf_card_editor (catalog + pdfplumber extraction):
each card id maps to a `target_page` in data/Card_ID List_JP.pdf, and
get_card_asset_map() crops+caches that page's card image. We only resolve the
resulting file path (lazily extracting on first request). Returns None if the
PDF or tool is unavailable, so the viewer can fall back to a text card.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_PDF = _REPO / "data" / "Card_ID List_JP.pdf"
_EDITOR = _REPO / "cardlist_referenced" / "pdf_card_editor"

if str(_EDITOR) not in sys.path:
    sys.path.insert(0, str(_EDITOR))


@lru_cache(maxsize=1)
def _catalog_by_id() -> dict:
    from pdf_card_tool import load_pdf_catalog
    catalog = load_pdf_catalog(str(_PDF))
    return {int(c.card_id): c for c in catalog.cards}


def available() -> bool:
    return _PDF.exists()


def japanese_names() -> dict[int, str]:
    """Map card_id -> Japanese card name (from the PDF catalog). Empty if no PDF."""
    if not _PDF.exists():
        return {}
    try:
        return {cid: c.name for cid, c in _catalog_by_id().items()}
    except Exception:
        return {}


def card_image_path(card_id: int):
    """Path to the card image (extracting+caching on first call), or None."""
    if not _PDF.exists():
        return None
    try:
        card = _catalog_by_id().get(int(card_id))
        if card is None:
            return None
        from pdf_card_tool import get_card_asset_map
        assets = get_card_asset_map(str(_PDF), [card.target_page])
        asset = assets.get(card.target_page)
        return asset.image_path if asset else None
    except Exception:
        return None
