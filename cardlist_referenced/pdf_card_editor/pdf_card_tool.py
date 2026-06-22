from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Iterable, Mapping

import pdfplumber
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject
from reportlab.lib.colors import black
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

HEADER_TEXT = "カード ID カード名 エキスパンション コレクション番号 リンク"
LINK_TEXT = "券面画像"
BACK_LINK_TEXT = "[表に戻す]"
DEFAULT_LINK_LEFT = 78
DEFAULT_LINK_TOP = 716
DEFAULT_CARD_WIDTH_MM = 63.0
DEFAULT_CARD_HEIGHT_MM = 88.0
DEFAULT_CARD_GAP_MM = 2.0
CACHE_VERSION = 1
LABEL_CACHE_VERSION = 1
CARD_STATE_CACHE_VERSION = 4
CARD_ASSET_CACHE_VERSION = 1
PREVIEW_CACHE_VERSION = 2
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
PREVIEW_DIR = CACHE_DIR / "previews"
CARD_ASSET_DIR = CACHE_DIR / "card_assets"
LABELS_DIR = CACHE_DIR / "labels"


@dataclass(frozen=True)
class CardRow:
    card_id: int
    name: str
    expansion: str
    collection_no: str
    index_page: int
    target_page: int


@dataclass(frozen=True)
class DeleteRequest:
    card_ids: frozenset[int]
    page_numbers: frozenset[int]


@dataclass(frozen=True)
class PdfCatalog:
    pdf_path: Path
    total_pages: int
    page_width: float
    page_height: float
    index_page_count: int
    rows_per_index_page: int
    cards: tuple[CardRow, ...]


@dataclass(frozen=True)
class CardStateStore:
    labels: dict[int, list[str]]
    card_deck_ids: dict[int, list[str]]
    deck_catalog: dict[str, str]
    work_list_card_ids: frozenset[int]
    excluded_card_ids: frozenset[int]


@dataclass(frozen=True)
class BuildResult:
    output_path: Path
    total_input_pages: int
    total_output_pages: int
    kept_cards: int
    removed_cards: int
    regenerated_index_pages: int
    ignored_page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class CardAsset:
    card_id: int
    target_page: int
    image_bbox: tuple[float, float, float, float]
    image_path: Path
    image_width: int
    image_height: int


@dataclass(frozen=True)
class PrintItem:
    card_id: int
    quantity: int
    label: str = ""


@dataclass(frozen=True)
class PrintLayoutSettings:
    page_size: str = "A4"
    card_width_mm: float = DEFAULT_CARD_WIDTH_MM
    card_height_mm: float = DEFAULT_CARD_HEIGHT_MM
    gap_mm: float = DEFAULT_CARD_GAP_MM
    cards_per_sheet: str | int = "auto"


@dataclass(frozen=True)
class PrintBuildResult:
    output_path: Path
    total_output_pages: int
    total_cards_requested: int
    total_cards_placed: int
    cards_per_sheet_used: int
    skipped_card_ids: tuple[int, ...]
    skipped_pages: tuple[int, ...]


def normalize_label_list(raw_label: object) -> list[str]:
    if raw_label is None:
        return []
    if isinstance(raw_label, str):
        value = raw_label.strip()
        return [value] if value else []
    if isinstance(raw_label, Iterable):
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_label:
            value = str(item).strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized
    value = str(raw_label).strip()
    return [value] if value else []


def label_text(raw_label: object, separator: str = " / ") -> str:
    return separator.join(normalize_label_list(raw_label))


@dataclass(frozen=True)
class _GeneratedIndexLink:
    card_id: int
    generated_index_page: int
    rect: tuple[float, float, float, float]


@dataclass(frozen=True)
class _SheetLayout:
    items_per_page: int
    card_width_pt: float
    card_height_pt: float
    positions: tuple[tuple[float, float], ...]


def parse_page_number_text(raw_text: str) -> tuple[frozenset[int], tuple[str, ...]]:
    page_numbers: set[int] = set()
    invalid_tokens: list[str] = []
    separators_normalized = raw_text.replace("\n", ",").replace("、", ",")
    for token in (part.strip() for part in separators_normalized.split(",")):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = (part.strip() for part in token.split("-", 1))
            if not start_text.isdigit() or not end_text.isdigit():
                invalid_tokens.append(token)
                continue
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                invalid_tokens.append(token)
                continue
            page_numbers.update(range(start, end + 1))
            continue
        if token.isdigit() and int(token) > 0:
            page_numbers.add(int(token))
            continue
        invalid_tokens.append(token)
    return frozenset(sorted(page_numbers)), tuple(invalid_tokens)


def default_output_path(input_pdf_path: str | Path) -> Path:
    input_path = Path(input_pdf_path)
    return input_path.with_name(f"{input_path.stem}.worklist.pdf")


def default_print_output_path(input_pdf_path: str | Path) -> Path:
    input_path = Path(input_pdf_path)
    return input_path.with_name(f"{input_path.stem}.print.pdf")


def resolve_output_pdf_path(raw_output_path: str | Path | None, fallback_output_path: str | Path) -> Path:
    fallback_path = Path(fallback_output_path).expanduser().resolve()
    raw_text = "" if raw_output_path is None else str(raw_output_path).strip()

    if not raw_text:
        return fallback_path

    candidate = Path(raw_text).expanduser()
    raw_ends_with_separator = raw_text.endswith(("\\", "/"))
    candidate_name = candidate.name.strip()
    looks_like_directory = (
        raw_ends_with_separator
        or candidate_name in {"", ".", ".."}
        or (candidate.exists() and candidate.is_dir())
    )

    if looks_like_directory:
        candidate = candidate / fallback_path.name
    elif candidate.suffix.lower() != ".pdf":
        candidate = candidate.with_suffix(".pdf")

    resolved = candidate.resolve()
    if resolved.exists() and resolved.is_dir():
        resolved = (resolved / fallback_path.name).resolve()
    if resolved.name.strip() in {"", ".", ".."}:
        raise ValueError("保存先には PDF ファイル名を含めてください。")
    return resolved


def pdf_fingerprint(input_pdf_path: str | Path) -> str:
    pdf_path = Path(input_pdf_path).expanduser()
    stat = pdf_path.stat()
    payload = f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_pdf_catalog(input_pdf_path: str | Path) -> PdfCatalog:
    pdf_path = Path(input_pdf_path).expanduser()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cached_catalog = _load_catalog_cache(pdf_path)
    if cached_catalog is not None:
        return cached_catalog

    reader = PdfReader(str(pdf_path))
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)

    cards: list[CardRow] = []
    index_page_numbers: list[int] = []
    rows_per_page: list[int] = []

    with pdfplumber.open(str(pdf_path)) as document:
        for page_number, (plumber_page, reader_page) in enumerate(
            zip(document.pages, reader.pages),
            start=1,
        ):
            text = plumber_page.extract_text() or ""
            annotations = reader_page.get("/Annots") or []
            if HEADER_TEXT not in text or not annotations:
                if index_page_numbers:
                    break
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            row_lines = [line for line in lines if line != HEADER_TEXT]
            if len(row_lines) != len(annotations):
                raise ValueError(
                    f"Index page {page_number} row count does not match link count: "
                    f"{len(row_lines)} rows vs {len(annotations)} annotations."
                )

            index_page_numbers.append(page_number)
            rows_per_page.append(len(row_lines))
            for row_line, annotation_ref in zip(row_lines, annotations):
                parsed_row = _parse_card_line(row_line)
                target_page = _resolve_destination_page(reader, annotation_ref.get_object()) + 1
                cards.append(
                    CardRow(
                        card_id=parsed_row.card_id,
                        name=parsed_row.name,
                        expansion=parsed_row.expansion,
                        collection_no=parsed_row.collection_no,
                        index_page=page_number,
                        target_page=target_page,
                    )
                )

    if not cards or not index_page_numbers:
        raise ValueError(f"No index pages found in {pdf_path}")

    catalog = PdfCatalog(
        pdf_path=pdf_path,
        total_pages=len(reader.pages),
        page_width=page_width,
        page_height=page_height,
        index_page_count=len(index_page_numbers),
        rows_per_index_page=max(rows_per_page),
        cards=tuple(cards),
    )
    _store_catalog_cache(catalog)
    return catalog


def load_card_state_store(input_pdf_path: str | Path) -> CardStateStore:
    label_path = _label_store_path(Path(input_pdf_path).expanduser())
    if not label_path.exists():
        return CardStateStore(labels={}, card_deck_ids={}, deck_catalog={}, work_list_card_ids=frozenset(), excluded_card_ids=frozenset())

    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CardStateStore(labels={}, card_deck_ids={}, deck_catalog={}, work_list_card_ids=frozenset(), excluded_card_ids=frozenset())

    cache_version = payload.get("cache_version")
    if cache_version not in {LABEL_CACHE_VERSION, 3, CARD_STATE_CACHE_VERSION}:
        return CardStateStore(labels={}, card_deck_ids={}, deck_catalog={}, work_list_card_ids=frozenset(), excluded_card_ids=frozenset())

    labels = {
        int(card_id): normalize_label_list(label)
        for card_id, label in payload.get("labels", {}).items()
        if normalize_label_list(label)
    }
    card_deck_ids = {
        int(card_id): normalize_label_list(deck_ids)
        for card_id, deck_ids in payload.get("card_deck_ids", {}).items()
        if normalize_label_list(deck_ids)
    }
    deck_catalog = {
        str(deck_id).strip(): str(deck_name).strip()
        for deck_id, deck_name in payload.get("deck_catalog", {}).items()
        if str(deck_id).strip() and str(deck_name).strip()
    }
    work_list_card_ids = frozenset(
        sorted(
            int(card_id)
            for card_id in payload.get("work_list_card_ids", [])
            if str(card_id).strip()
        )
    )
    excluded_card_ids = frozenset(
        sorted(
            int(card_id)
            for card_id in payload.get("excluded_card_ids", [])
            if str(card_id).strip()
        )
    )
    return CardStateStore(
        labels=labels,
        card_deck_ids=card_deck_ids,
        deck_catalog=deck_catalog,
        work_list_card_ids=work_list_card_ids,
        excluded_card_ids=excluded_card_ids,
    )


def save_card_state_store(
    input_pdf_path: str | Path,
    labels: Mapping[int, object],
    work_list_card_ids: Iterable[int],
    card_deck_ids: Mapping[int, object] | None = None,
    deck_catalog: Mapping[str, str] | None = None,
) -> Path:
    pdf_path = Path(input_pdf_path).expanduser()
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    filtered_labels = {
        str(int(card_id)): normalize_label_list(label)
        for card_id, label in labels.items()
        if normalize_label_list(label)
    }
    normalized_work_list_card_ids = sorted({int(card_id) for card_id in work_list_card_ids})
    normalized_card_deck_ids = {
        str(int(card_id)): normalize_label_list(deck_ids)
        for card_id, deck_ids in (card_deck_ids or {}).items()
        if normalize_label_list(deck_ids)
    }
    normalized_deck_catalog = {
        str(deck_id).strip(): str(deck_name).strip()
        for deck_id, deck_name in (deck_catalog or {}).items()
        if str(deck_id).strip() and str(deck_name).strip()
    }
    payload = {
        "cache_version": CARD_STATE_CACHE_VERSION,
        "pdf_path": str(pdf_path.resolve()),
        "pdf_fingerprint": pdf_fingerprint(pdf_path),
        "labels": filtered_labels,
        "card_deck_ids": normalized_card_deck_ids,
        "deck_catalog": normalized_deck_catalog,
        "work_list_card_ids": normalized_work_list_card_ids,
    }
    label_path = _label_store_path(pdf_path)
    label_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return label_path


def load_label_store(input_pdf_path: str | Path) -> dict[int, list[str]]:
    return load_card_state_store(input_pdf_path).labels


def save_label_store(input_pdf_path: str | Path, labels: Mapping[int, object]) -> Path:
    existing_state = load_card_state_store(input_pdf_path)
    return save_card_state_store(
        input_pdf_path,
        labels,
        existing_state.work_list_card_ids,
        existing_state.card_deck_ids,
        existing_state.deck_catalog,
    )


def get_card_asset_map(
    input_pdf_path: str | Path,
    target_pages: Iterable[int],
) -> dict[int, CardAsset]:
    pdf_path = Path(input_pdf_path).expanduser()
    catalog = load_pdf_catalog(pdf_path)
    page_to_card = {card.target_page: card for card in catalog.cards}
    unique_pages = tuple(
        sorted({int(page) for page in target_pages if int(page) > 0 and int(page) in page_to_card})
    )
    if not unique_pages:
        return {}

    asset_dir = _card_asset_cache_dir(pdf_path)
    asset_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = asset_dir / "metadata.json"
    metadata = _load_json_file(metadata_path)
    stored_assets = metadata.get("assets", {}) if isinstance(metadata, dict) else {}

    missing_pages = [
        page
        for page in unique_pages
        if not _asset_entry_is_usable(asset_dir, stored_assets.get(str(page)))
    ]
    if missing_pages:
        with pdfplumber.open(str(pdf_path)) as document:
            for page_number in missing_pages:
                page = document.pages[page_number - 1]
                image_info = _pick_card_image(page)
                if image_info is None:
                    continue

                raw_bytes = image_info["stream"].get_data()
                with Image.open(BytesIO(raw_bytes)) as image:
                    image.load()
                    file_suffix = ".jpg" if (image.format or "").upper() == "JPEG" else ".png"
                    file_name = f"page_{page_number}{file_suffix}"
                    cache_path = asset_dir / file_name
                    if file_suffix == ".jpg":
                        cache_path.write_bytes(raw_bytes)
                    else:
                        image.save(cache_path, format="PNG")
                    stored_assets[str(page_number)] = {
                        "card_id": page_to_card[page_number].card_id,
                        "bbox": [
                            float(image_info["x0"]),
                            float(image_info["y0"]),
                            float(image_info["x1"]),
                            float(image_info["y1"]),
                        ],
                        "file_name": file_name,
                        "image_width": int(image.width),
                        "image_height": int(image.height),
                    }

        metadata = {
            "cache_version": CARD_ASSET_CACHE_VERSION,
            "pdf_fingerprint": pdf_fingerprint(pdf_path),
            "assets": stored_assets,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    assets: dict[int, CardAsset] = {}
    for page_number in unique_pages:
        entry = stored_assets.get(str(page_number))
        if not _asset_entry_is_usable(asset_dir, entry):
            continue
        image_path = asset_dir / str(entry["file_name"])
        assets[page_number] = CardAsset(
            card_id=int(entry["card_id"]),
            target_page=page_number,
            image_bbox=tuple(float(value) for value in entry["bbox"]),
            image_path=image_path,
            image_width=int(entry["image_width"]),
            image_height=int(entry["image_height"]),
        )

    return assets


def get_card_preview_data_map(
    input_pdf_path: str | Path,
    target_pages: Iterable[int],
    *,
    max_width: int = 240,
) -> dict[int, bytes]:
    pdf_path = Path(input_pdf_path).expanduser()
    unique_pages = tuple(sorted({int(page) for page in target_pages if int(page) > 0}))
    if not unique_pages:
        return {}

    assets = get_card_asset_map(pdf_path, unique_pages)
    preview_dir = _preview_cache_dir(pdf_path)
    preview_dir.mkdir(parents=True, exist_ok=True)

    output: dict[int, bytes] = {}
    for page_number, asset in assets.items():
        preview_path = _preview_cache_path(pdf_path, page_number, max_width)
        if not preview_path.exists():
            with Image.open(asset.image_path) as image:
                image.load()
                if image.width > max_width:
                    ratio = max_width / float(image.width)
                    resized = image.resize(
                        (max_width, max(1, int(image.height * ratio))),
                        Image.Resampling.LANCZOS,
                    )
                else:
                    resized = image.copy()
                try:
                    resized.save(preview_path, format="PNG")
                finally:
                    resized.close()
        output[page_number] = preview_path.read_bytes()

    return output


def build_filtered_pdf(
    input_pdf_path: str | Path,
    output_pdf_path: str | Path,
    delete_request: DeleteRequest,
) -> BuildResult:
    catalog = load_pdf_catalog(input_pdf_path)
    page_numbers_to_drop = set(delete_request.page_numbers)
    card_ids_to_drop = set(delete_request.card_ids)

    kept_cards = [
        card
        for card in catalog.cards
        if card.card_id not in card_ids_to_drop and card.target_page not in page_numbers_to_drop
    ]
    if not kept_cards:
        raise ValueError("The delete request would remove every card page.")

    ignored_page_numbers = tuple(
        sorted(
            page
            for page in page_numbers_to_drop
            if page < 1 or page > catalog.total_pages or page <= catalog.index_page_count
        )
    )

    index_pdf_bytes, index_links = _generate_index_pdf(
        kept_cards,
        page_width=catalog.page_width,
        page_height=catalog.page_height,
        rows_per_page=catalog.rows_per_index_page,
    )

    writer = PdfWriter()
    generated_index_reader = PdfReader(BytesIO(index_pdf_bytes))
    for index_page in generated_index_reader.pages:
        writer.add_page(index_page)

    source_reader = PdfReader(str(catalog.pdf_path))
    new_page_number_by_card_id: dict[int, int] = {}
    for card in kept_cards:
        source_page = source_reader.pages[card.target_page - 1]
        copied_page = writer.add_page(source_page)
        _retarget_back_links(copied_page, writer.pages[0])
        new_page_number_by_card_id[card.card_id] = len(writer.pages)

    for index_link in index_links:
        source_page = writer.pages[index_link.generated_index_page - 1]
        target_page = writer.pages[new_page_number_by_card_id[index_link.card_id] - 1]
        _append_link_annotation(writer, source_page, index_link.rect, target_page)

    output_path = resolve_output_pdf_path(output_pdf_path, default_output_path(input_pdf_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        writer.write(output_file)

    return BuildResult(
        output_path=output_path,
        total_input_pages=catalog.total_pages,
        total_output_pages=len(writer.pages),
        kept_cards=len(kept_cards),
        removed_cards=len(catalog.cards) - len(kept_cards),
        regenerated_index_pages=len(generated_index_reader.pages),
        ignored_page_numbers=ignored_page_numbers,
    )


def build_print_pdf(
    input_pdf_path: str | Path,
    output_pdf_path: str | Path,
    print_items: Iterable[PrintItem],
    layout_settings: PrintLayoutSettings,
) -> PrintBuildResult:
    catalog = load_pdf_catalog(input_pdf_path)
    cards_by_id = {card.card_id: card for card in catalog.cards}
    normalized_items = [item for item in print_items if int(item.quantity) > 0]
    if not normalized_items:
        raise ValueError("At least one print item with quantity > 0 is required.")

    requested_total = sum(int(item.quantity) for item in normalized_items)
    requested_pages = [
        cards_by_id[item.card_id].target_page
        for item in normalized_items
        if item.card_id in cards_by_id
    ]
    assets_by_page = get_card_asset_map(input_pdf_path, requested_pages)

    expanded_assets: list[CardAsset] = []
    skipped_card_ids: set[int] = set()
    skipped_pages: set[int] = set()
    for item in normalized_items:
        card = cards_by_id.get(item.card_id)
        if card is None:
            skipped_card_ids.add(item.card_id)
            continue
        asset = assets_by_page.get(card.target_page)
        if asset is None:
            skipped_card_ids.add(item.card_id)
            skipped_pages.add(card.target_page)
            continue
        expanded_assets.extend([asset] * int(item.quantity))

    if not expanded_assets:
        raise ValueError("No printable card images were found for the selected print items.")

    page_width, page_height = _resolve_page_size(layout_settings.page_size)
    sheet_layout = _build_sheet_layout(layout_settings, page_width, page_height)

    output_path = resolve_output_pdf_path(output_pdf_path, default_print_output_path(input_pdf_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_canvas = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))
    for asset_index, asset in enumerate(expanded_assets):
        if asset_index > 0 and asset_index % sheet_layout.items_per_page == 0:
            pdf_canvas.showPage()
        slot_x, slot_y = sheet_layout.positions[asset_index % sheet_layout.items_per_page]
        pdf_canvas.drawImage(
            str(asset.image_path),
            slot_x,
            slot_y,
            width=sheet_layout.card_width_pt,
            height=sheet_layout.card_height_pt,
            preserveAspectRatio=True,
            mask="auto",
        )

    pdf_canvas.save()

    return PrintBuildResult(
        output_path=output_path,
        total_output_pages=ceil(len(expanded_assets) / sheet_layout.items_per_page),
        total_cards_requested=requested_total,
        total_cards_placed=len(expanded_assets),
        cards_per_sheet_used=sheet_layout.items_per_page,
        skipped_card_ids=tuple(sorted(skipped_card_ids)),
        skipped_pages=tuple(sorted(skipped_pages)),
    )


def _parse_card_line(line: str) -> CardRow:
    stripped = line.strip()
    if not stripped.endswith(LINK_TEXT):
        raise ValueError(f"Unsupported index row format: {line}")

    body = stripped[: -len(LINK_TEXT)].strip()
    parts = body.split()
    if len(parts) < 3 or not parts[0].isdigit():
        raise ValueError(f"Unsupported index row format: {line}")

    card_id = int(parts[0])
    content_tokens = parts[1:]
    if len(content_tokens) < 2:
        raise ValueError(f"Unsupported index row format: {line}")

    if "/" in content_tokens[-1]:
        if len(content_tokens) < 3:
            raise ValueError(f"Unsupported index row format: {line}")
        collection_no = content_tokens[-1]
        expansion = content_tokens[-2]
        name_tokens = content_tokens[:-2]
    else:
        collection_no = ""
        expansion = content_tokens[-1]
        name_tokens = content_tokens[:-1]

    if not name_tokens:
        raise ValueError(f"Unsupported index row format: {line}")

    return CardRow(
        card_id=card_id,
        name=" ".join(name_tokens),
        expansion=expansion,
        collection_no=collection_no,
        index_page=0,
        target_page=0,
    )


def _resolve_destination_page(reader: PdfReader, annotation: DictionaryObject) -> int:
    destination = annotation.get("/Dest")
    if destination is None:
        action = annotation.get("/A")
        if action is not None:
            destination = action.get("/D")
    if destination is None:
        raise ValueError("Link annotation does not contain a destination.")
    if isinstance(destination, list) and destination:
        return reader.get_page_number(destination[0])
    resolved = reader._build_destination(destination)
    return reader.get_page_number(resolved.page)


def _generate_index_pdf(
    cards: Iterable[CardRow],
    *,
    page_width: float,
    page_height: float,
    rows_per_page: int,
) -> tuple[bytes, tuple[_GeneratedIndexLink, ...]]:
    font_name = _ensure_japanese_font()
    header_font_size = 10
    row_font_size = 10
    row_height = 18
    header_y = page_height - 54
    first_row_y = page_height - 126
    x_id = 48
    x_name = 90
    x_exp = 332
    x_collection = 420
    x_link = 510
    link_width = 42
    link_height = 12

    card_list = list(cards)
    if not card_list:
        raise ValueError("At least one card must remain in the filtered PDF.")

    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    generated_links: list[_GeneratedIndexLink] = []

    total_pages = ceil(len(card_list) / rows_per_page)
    for generated_page_number in range(1, total_pages + 1):
        start = (generated_page_number - 1) * rows_per_page
        end = start + rows_per_page
        page_cards = card_list[start:end]

        _draw_index_header(
            pdf_canvas,
            font_name=font_name,
            font_size=header_font_size,
            header_y=header_y,
            page_width=page_width,
        )

        for row_index, card in enumerate(page_cards):
            y = first_row_y - row_index * row_height
            _draw_index_row(
                pdf_canvas,
                card,
                font_name=font_name,
                font_size=row_font_size,
                y=y,
                x_id=x_id,
                x_name=x_name,
                x_exp=x_exp,
                x_collection=x_collection,
                x_link=x_link,
            )
            generated_links.append(
                _GeneratedIndexLink(
                    card_id=card.card_id,
                    generated_index_page=generated_page_number,
                    rect=(x_link - 2, y - 2, x_link + link_width, y - 2 + link_height),
                )
            )

        if generated_page_number != total_pages:
            pdf_canvas.showPage()

    pdf_canvas.save()
    return buffer.getvalue(), tuple(generated_links)


def _draw_index_header(
    pdf_canvas: canvas.Canvas,
    *,
    font_name: str,
    font_size: float,
    header_y: float,
    page_width: float,
) -> None:
    pdf_canvas.setFont(font_name, font_size)
    pdf_canvas.setFillColor(black)
    pdf_canvas.drawString(48, header_y, HEADER_TEXT)
    pdf_canvas.line(42, header_y - 8, page_width - 42, header_y - 8)


def _draw_index_row(
    pdf_canvas: canvas.Canvas,
    card: CardRow,
    *,
    font_name: str,
    font_size: float,
    y: float,
    x_id: float,
    x_name: float,
    x_exp: float,
    x_collection: float,
    x_link: float,
) -> None:
    pdf_canvas.setFont(font_name, font_size)
    pdf_canvas.drawString(x_id, y, str(card.card_id))
    pdf_canvas.drawString(x_name, y, _truncate_text(pdf_canvas, card.name, font_name, font_size, 228))
    pdf_canvas.drawString(x_exp, y, _truncate_text(pdf_canvas, card.expansion, font_name, font_size, 72))
    pdf_canvas.drawString(
        x_collection,
        y,
        _truncate_text(pdf_canvas, card.collection_no or "-", font_name, font_size, 78),
    )
    pdf_canvas.drawString(x_link, y, LINK_TEXT)


def _truncate_text(
    pdf_canvas: canvas.Canvas,
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
) -> str:
    if pdf_canvas.stringWidth(text, font_name, font_size) <= max_width:
        return text
    ellipsis = "..."
    current = text
    while current and pdf_canvas.stringWidth(current + ellipsis, font_name, font_size) > max_width:
        current = current[:-1]
    return (current + ellipsis) if current else ellipsis


def _retarget_back_links(page, first_index_page) -> None:
    annotations = page.get("/Annots") or []
    for annotation_ref in annotations:
        annotation = annotation_ref.get_object()
        if annotation.get("/Subtype") != "/Link":
            continue
        annotation[NameObject("/Dest")] = _destination_array(first_index_page)
        if "/A" in annotation:
            del annotation["/A"]


def _append_link_annotation(writer: PdfWriter, page, rect: tuple[float, float, float, float], destination_page) -> None:
    annotation = DictionaryObject()
    annotation.update(
        {
            NameObject("/Type"): NameObject("/Annot"),
            NameObject("/Subtype"): NameObject("/Link"),
            NameObject("/Border"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(0)]),
            NameObject("/Rect"): ArrayObject([FloatObject(value) for value in rect]),
            NameObject("/Dest"): _destination_array(destination_page),
        }
    )
    annotation_object = writer._add_object(annotation)
    if "/Annots" not in page:
        page[NameObject("/Annots")] = ArrayObject()
    page["/Annots"].append(annotation_object)


def _destination_array(destination_page) -> ArrayObject:
    return ArrayObject(
        [
            destination_page.indirect_reference,
            NameObject("/XYZ"),
            FloatObject(DEFAULT_LINK_LEFT),
            FloatObject(DEFAULT_LINK_TOP),
            NumberObject(0),
        ]
    )


def _resolve_page_size(page_size: str) -> tuple[float, float]:
    if page_size.upper() == "A4":
        return A4
    raise ValueError(f"Unsupported page size: {page_size}")


def _build_sheet_layout(
    settings: PrintLayoutSettings,
    page_width: float,
    page_height: float,
) -> _SheetLayout:
    card_width_pt = settings.card_width_mm * mm
    card_height_pt = settings.card_height_mm * mm
    gap_pt = settings.gap_mm * mm

    max_cols = max(1, int((page_width + gap_pt) // (card_width_pt + gap_pt)))
    max_rows = max(1, int((page_height + gap_pt) // (card_height_pt + gap_pt)))

    candidates: list[dict[str, float]] = []
    for rows in range(1, max_rows + 1):
        for cols in range(1, max_cols + 1):
            total_width = cols * card_width_pt + max(0, cols - 1) * gap_pt
            total_height = rows * card_height_pt + max(0, rows - 1) * gap_pt
            if total_width <= page_width + 0.01 and total_height <= page_height + 0.01:
                candidates.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "slots": rows * cols,
                        "total_width": total_width,
                        "total_height": total_height,
                    }
                )

    if not candidates:
        raise ValueError("The requested card size does not fit on the target page.")

    if str(settings.cards_per_sheet).lower() == "auto":
        chosen = max(
            candidates,
            key=lambda item: (
                int(item["slots"]),
                -abs((item["total_width"] / item["total_height"]) - (page_width / page_height)),
            ),
        )
        items_per_page = int(chosen["slots"])
    else:
        desired = int(settings.cards_per_sheet)
        feasible = [item for item in candidates if int(item["slots"]) >= desired]
        if not feasible:
            raise ValueError("The selected cards-per-sheet setting does not fit on the page.")
        chosen = min(
            feasible,
            key=lambda item: (
                int(item["slots"]) - desired,
                abs((item["total_width"] / item["total_height"]) - (page_width / page_height)),
            ),
        )
        items_per_page = desired

    cols = int(chosen["cols"])
    rows = int(chosen["rows"])
    total_width = float(chosen["total_width"])
    total_height = float(chosen["total_height"])

    start_x = (page_width - total_width) / 2.0
    top_y = page_height - (page_height - total_height) / 2.0

    positions: list[tuple[float, float]] = []
    for row_index in range(rows):
        for col_index in range(cols):
            x = start_x + col_index * (card_width_pt + gap_pt)
            y = top_y - card_height_pt - row_index * (card_height_pt + gap_pt)
            positions.append((x, y))

    return _SheetLayout(
        items_per_page=items_per_page,
        card_width_pt=card_width_pt,
        card_height_pt=card_height_pt,
        positions=tuple(positions[:items_per_page]),
    )


def _pick_card_image(page: pdfplumber.page.Page):
    if not page.images:
        return None
    return max(
        page.images,
        key=lambda image: float(image["width"]) * float(image["height"]),
    )


def _asset_entry_is_usable(asset_dir: Path, entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    file_name = entry.get("file_name")
    if not isinstance(file_name, str):
        return False
    image_path = asset_dir / file_name
    return image_path.exists()


def _ensure_japanese_font() -> str:
    font_name = "HeiseiKakuGo-W5"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def _load_catalog_cache(pdf_path: Path) -> PdfCatalog | None:
    cache_path = _catalog_cache_path(pdf_path)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    stat = pdf_path.stat()
    expected_source = {
        "path": str(pdf_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "cache_version": CACHE_VERSION,
    }
    if payload.get("source") != expected_source:
        return None

    cards = tuple(
        CardRow(
            card_id=int(card["card_id"]),
            name=str(card["name"]),
            expansion=str(card["expansion"]),
            collection_no=str(card["collection_no"]),
            index_page=int(card["index_page"]),
            target_page=int(card["target_page"]),
        )
        for card in payload["cards"]
    )

    return PdfCatalog(
        pdf_path=pdf_path,
        total_pages=int(payload["total_pages"]),
        page_width=float(payload["page_width"]),
        page_height=float(payload["page_height"]),
        index_page_count=int(payload["index_page_count"]),
        rows_per_index_page=int(payload["rows_per_index_page"]),
        cards=cards,
    )


def _store_catalog_cache(catalog: PdfCatalog) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stat = catalog.pdf_path.stat()
    payload = {
        "source": {
            "path": str(catalog.pdf_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "cache_version": CACHE_VERSION,
        },
        "total_pages": catalog.total_pages,
        "page_width": catalog.page_width,
        "page_height": catalog.page_height,
        "index_page_count": catalog.index_page_count,
        "rows_per_index_page": catalog.rows_per_index_page,
        "cards": [
            {
                "card_id": card.card_id,
                "name": card.name,
                "expansion": card.expansion,
                "collection_no": card.collection_no,
                "index_page": card.index_page,
                "target_page": card.target_page,
            }
            for card in catalog.cards
        ],
    }
    _catalog_cache_path(catalog.pdf_path).write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _catalog_cache_path(pdf_path: Path) -> Path:
    normalized = str(pdf_path.resolve()).replace(":", "").replace("\\", "_").replace("/", "_")
    return CACHE_DIR / f"{normalized}.catalog.json"


def _card_asset_cache_dir(pdf_path: Path) -> Path:
    return CARD_ASSET_DIR / pdf_fingerprint(pdf_path)


def _label_store_path(pdf_path: Path) -> Path:
    return LABELS_DIR / f"{pdf_fingerprint(pdf_path)}.labels.json"


def _preview_cache_dir(pdf_path: Path) -> Path:
    return PREVIEW_DIR / pdf_fingerprint(pdf_path)


def _preview_cache_path(pdf_path: Path, page_number: int, max_width: int) -> Path:
    return _preview_cache_dir(pdf_path) / f"page_{page_number}_w{max_width}.png"
