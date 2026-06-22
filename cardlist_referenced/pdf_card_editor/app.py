from __future__ import annotations

import importlib
import json
import mimetypes
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlencode, urlparse

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required to run this app. Install the packages in "
        "pdf_tool_requirements.txt first."
    ) from exc

import pandas as pd

import pdf_card_tool as _pdf_card_tool

_pdf_card_tool = importlib.reload(_pdf_card_tool)

BuildResult = _pdf_card_tool.BuildResult
DeleteRequest = _pdf_card_tool.DeleteRequest
PrintBuildResult = _pdf_card_tool.PrintBuildResult
PrintItem = _pdf_card_tool.PrintItem
PrintLayoutSettings = _pdf_card_tool.PrintLayoutSettings
build_filtered_pdf = _pdf_card_tool.build_filtered_pdf
build_print_pdf = _pdf_card_tool.build_print_pdf
default_output_path = _pdf_card_tool.default_output_path
default_print_output_path = _pdf_card_tool.default_print_output_path
get_card_preview_data_map = _pdf_card_tool.get_card_preview_data_map
load_card_state_store = _pdf_card_tool.load_card_state_store
load_pdf_catalog = _pdf_card_tool.load_pdf_catalog
parse_page_number_text = _pdf_card_tool.parse_page_number_text
pdf_fingerprint = _pdf_card_tool.pdf_fingerprint
resolve_output_pdf_path = _pdf_card_tool.resolve_output_pdf_path
save_card_state_store = _pdf_card_tool.save_card_state_store
normalize_label_list = _pdf_card_tool.normalize_label_list
label_text = _pdf_card_tool.label_text

DEFAULT_PDF_PATH = str(Path(__file__).resolve().parent.parent / "Card_ID List_JP_original.pdf")
CARD_TABLE_HEIGHT = 840
PREVIEW_MAX_WIDTH = 320
PRINT_SHEET_OPTIONS = ["auto", "1", "2", "4", "6", "8", "9"]
PROJECTS_DIR = Path(__file__).resolve().parent / "projects"
PROJECT_FILE_NAME = "project.json"
PROJECT_VERSION = 1


def _merge_labels_for_card(labels: dict[int, list[str]], card_id: int, raw_labels: object) -> bool:
    existing = normalize_label_list(labels.get(card_id))
    incoming = normalize_label_list(raw_labels)
    changed = False
    for value in incoming:
        if value not in existing:
            existing.append(value)
            changed = True
    if changed:
        labels[int(card_id)] = existing
    return changed


def _add_label_to_cards(labels: dict[int, list[str]], card_ids: Iterable[int], raw_label: object) -> int:
    changed = 0
    for card_id in card_ids:
        if _merge_labels_for_card(labels, int(card_id), raw_label):
            changed += 1
    return changed


def _merge_string_values_for_card(store: dict[int, list[str]], card_id: int, raw_values: object) -> bool:
    existing = normalize_label_list(store.get(card_id))
    incoming = normalize_label_list(raw_values)
    changed = False
    for value in incoming:
        if value not in existing:
            existing.append(value)
            changed = True
    if changed:
        store[int(card_id)] = existing
    return changed


def _remove_specific_labels_from_cards(
    labels: dict[int, list[str]],
    card_ids: Iterable[int],
    raw_labels_to_remove: object,
) -> int:
    labels_to_remove = set(normalize_label_list(raw_labels_to_remove))
    if not labels_to_remove:
        return 0

    changed = 0
    for card_id in card_ids:
        existing = normalize_label_list(labels.get(int(card_id)))
        if not existing:
            continue
        updated = [label for label in existing if label not in labels_to_remove]
        if updated != existing:
            changed += 1
            if updated:
                labels[int(card_id)] = updated
            else:
                labels.pop(int(card_id), None)
    return changed


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _projects_root() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def _project_file_path(project_dir: Path) -> Path:
    return project_dir / PROJECT_FILE_NAME


def _sanitize_project_name(raw_name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", str(raw_name)).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "card-project"


def _unique_project_dir(project_name: str) -> Path:
    base_name = _sanitize_project_name(project_name)
    project_dir = _projects_root() / base_name
    if not project_dir.exists():
        return project_dir
    suffix = 2
    while True:
        candidate = _projects_root() / f"{base_name}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _discover_project_candidates() -> list[str]:
    root = _projects_root()
    return sorted(
        (
            str(path.resolve())
            for path in root.rglob(PROJECT_FILE_NAME)
            if path.is_file()
        ),
        key=lambda item: (Path(item).parent.name.casefold(), item.casefold()),
    )


def _load_project_payload(project_file: str | Path) -> dict:
    path = Path(project_file).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("project_version", 0)) != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {payload.get('project_version')}")
    return payload


def _normalize_project_state(payload: dict) -> dict:
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
    work_list_card_ids = {
        int(card_id)
        for card_id in payload.get("work_list_card_ids", [])
        if str(card_id).strip()
    }
    print_quantities = {
        int(card_id): int(quantity)
        for card_id, quantity in payload.get("print_quantities", {}).items()
        if int(quantity) > 0
    }
    print_order = [
        int(card_id)
        for card_id in payload.get("print_order", [])
        if int(card_id) in print_quantities
    ]
    print_excluded_card_ids = {
        int(card_id)
        for card_id in payload.get("print_excluded_card_ids", [])
        if str(card_id).strip()
    }
    return {
        "labels": labels,
        "card_deck_ids": card_deck_ids,
        "deck_catalog": deck_catalog,
        "work_list_card_ids": work_list_card_ids,
        "print_quantities": print_quantities,
        "print_order": print_order,
        "print_excluded_card_ids": print_excluded_card_ids,
    }


def _normalize_deck_card_quantities(raw_store: object) -> dict[str, dict[int, int]]:
    normalized: dict[str, dict[int, int]] = {}
    if not isinstance(raw_store, dict):
        return normalized

    for raw_deck_id, raw_recipe in raw_store.items():
        deck_id = str(raw_deck_id).strip()
        if not deck_id or not isinstance(raw_recipe, dict):
            continue
        recipe: dict[int, int] = {}
        for raw_card_id, raw_quantity in raw_recipe.items():
            try:
                card_id = int(raw_card_id)
                quantity = int(raw_quantity)
            except (TypeError, ValueError):
                continue
            if quantity > 0:
                recipe[card_id] = quantity
        if recipe:
            normalized[deck_id] = recipe
    return normalized


def _project_source_pdf_path(project_file: str | Path, payload: dict) -> Path:
    project_path = Path(project_file).expanduser().resolve()
    source_rel = str(payload.get("source_pdf_path", "source.pdf")).strip() or "source.pdf"
    return (project_path.parent / source_rel).resolve()


def _save_project_payload(
    project_file: str | Path,
    *,
    project_name: str,
    source_pdf_path: str | Path,
    labels: dict[int, list[str]],
    card_deck_ids: dict[int, list[str]],
    deck_catalog: dict[str, str],
    work_list_card_ids: set[int],
    print_excluded_card_ids: set[int],
    print_quantities: dict[int, int],
    print_order: list[int],
    created_at: str | None = None,
) -> Path:
    project_path = Path(project_file).expanduser().resolve()
    project_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_version": PROJECT_VERSION,
        "project_name": project_name,
        "created_at": created_at or _iso_now(),
        "updated_at": _iso_now(),
        "source_pdf_path": str(Path(source_pdf_path).name),
        "labels": {
            str(int(card_id)): normalize_label_list(label)
            for card_id, label in labels.items()
            if normalize_label_list(label)
        },
        "card_deck_ids": {
            str(int(card_id)): normalize_label_list(deck_ids)
            for card_id, deck_ids in card_deck_ids.items()
            if normalize_label_list(deck_ids)
        },
        "deck_catalog": {
            str(deck_id).strip(): str(deck_name).strip()
            for deck_id, deck_name in deck_catalog.items()
            if str(deck_id).strip() and str(deck_name).strip()
        },
        "work_list_card_ids": sorted(int(card_id) for card_id in work_list_card_ids),
        "print_excluded_card_ids": sorted(int(card_id) for card_id in print_excluded_card_ids),
        "print_quantities": {
            str(int(card_id)): int(quantity)
            for card_id, quantity in print_quantities.items()
            if int(quantity) > 0
        },
        "print_order": [int(card_id) for card_id in print_order if int(card_id) in print_quantities],
    }
    project_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return project_path


def _create_project_from_pdf(
    source_pdf_path: str,
    project_name: str,
    *,
    labels: dict[int, list[str]],
    card_deck_ids: dict[int, list[str]],
    deck_catalog: dict[str, str],
    work_list_card_ids: set[int],
    print_excluded_card_ids: set[int],
    print_quantities: dict[int, int],
    print_order: list[int],
) -> Path:
    source_path = Path(source_pdf_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF not found: {source_path}")

    project_dir = _unique_project_dir(project_name)
    project_dir.mkdir(parents=True, exist_ok=False)
    copied_source_path = project_dir / source_path.name
    shutil.copy2(source_path, copied_source_path)
    project_file = _project_file_path(project_dir)
    _save_project_payload(
        project_file,
        project_name=_sanitize_project_name(project_name),
        source_pdf_path=copied_source_path,
        labels=labels,
        card_deck_ids=card_deck_ids,
        deck_catalog=deck_catalog,
        work_list_card_ids=work_list_card_ids,
        print_excluded_card_ids=print_excluded_card_ids,
        print_quantities=print_quantities,
        print_order=print_order,
    )
    return project_file


def _default_project_output_path(project_file: str | Path, suffix: str) -> Path:
    project_path = Path(project_file).expanduser().resolve()
    payload = _load_project_payload(project_path)
    project_name = _sanitize_project_name(str(payload.get("project_name", project_path.parent.name)))
    output_dir = project_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{project_name}.{suffix}.pdf"


@st.cache_data(show_spinner=False)
def cached_catalog(pdf_path: str):
    return load_pdf_catalog(pdf_path)


@st.cache_resource(show_spinner=False)
def ensure_local_asset_server() -> str:
    class AssetRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if parsed.path == "/preview":
                self._serve_preview(params)
                return
            if parsed.path == "/artifact":
                self._serve_artifact(params)
                return
            self.send_error(404, "Not Found")

        def _serve_preview(self, params: dict[str, list[str]]) -> None:
            pdf_path = params.get("pdf", [""])[0]
            page_text = params.get("page", [""])[0]
            width_text = params.get("width", [""])[0]
            if not pdf_path or not page_text.isdigit():
                self.send_error(400, "Missing or invalid preview parameters")
                return

            max_width = PREVIEW_MAX_WIDTH
            if width_text.isdigit():
                max_width = max(80, min(1200, int(width_text)))

            try:
                page_number = int(page_text)
                preview_map = get_card_preview_data_map(pdf_path, [page_number], max_width=max_width)
                preview_bytes = preview_map.get(page_number)
                if not preview_bytes:
                    self.send_error(404, "Preview not found")
                    return
            except FileNotFoundError:
                self.send_error(404, "PDF not found")
                return
            except Exception as exc:  # pragma: no cover
                self.send_error(500, f"Preview generation failed: {exc}")
                return

            self._send_binary(preview_bytes, "image/png")

        def _serve_artifact(self, params: dict[str, list[str]]) -> None:
            file_text = params.get("path", [""])[0]
            if not file_text:
                self.send_error(400, "Missing file path")
                return

            file_path = Path(file_text).expanduser()
            if not file_path.exists() or file_path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
                self.send_error(404, "Artifact not found")
                return

            try:
                data = file_path.read_bytes()
            except OSError as exc:  # pragma: no cover
                self.send_error(500, f"Could not read artifact: {exc}")
                return

            mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            self._send_binary(data, mime_type)

        def _send_binary(self, data: bytes, mime_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), AssetRequestHandler)
    thread = threading.Thread(target=server.serve_forever, name="card-asset-server", daemon=True)
    thread.start()
    return f"http://127.0.0.1:{server.server_port}"


def _pdf_version_token(pdf_path: str | Path) -> str:
    pdf_file = Path(pdf_path).expanduser()
    return str(pdf_file.stat().st_mtime_ns) if pdf_file.exists() else "0"


def build_preview_url(
    pdf_path: str | Path,
    target_page: int,
    *,
    max_width: int,
    version_token: str | None = None,
) -> str:
    pdf_file = Path(pdf_path).expanduser()
    if version_token is None:
        version_token = _pdf_version_token(pdf_file)
    query = urlencode(
        {
            "pdf": str(pdf_file),
            "page": int(target_page),
            "width": int(max_width),
            "v": version_token,
        }
    )
    return f"{ensure_local_asset_server()}/preview?{query}"


def build_artifact_url(file_path: str | Path) -> str:
    artifact = Path(file_path).expanduser()
    version_token = str(artifact.stat().st_mtime_ns) if artifact.exists() else "0"
    query = urlencode({"path": str(artifact), "v": version_token})
    return f"{ensure_local_asset_server()}/artifact?{query}"


def main() -> None:
    st.set_page_config(
        page_title="カード印刷ツール", page_icon="🃏", layout="wide", initial_sidebar_state="expanded"
    )
    _inject_app_styles()
    _ensure_app_state()

    candidate_paths = _discover_pdf_candidates(DEFAULT_PDF_PATH)
    active_project = _active_project_file()
    active_project_payload: dict | None = None
    if active_project:
        try:
            active_project_payload = _load_project_payload(active_project)
            project_source_pdf = _project_source_pdf_path(active_project, active_project_payload)
            pdf_path = str(project_source_pdf)
        except Exception as exc:
            st.error(f"プロジェクトの読み込みに失敗しました: {exc}")
            _clear_active_project()
            return
    else:
        pdf_path = str(st.session_state.get("pdf_path_input", DEFAULT_PDF_PATH) or "").strip()

    if not pdf_path.strip():
        with st.sidebar:
            _render_file_menu(pdf_path, candidate_paths, active_project, None, None)
        st.info("📂 左のサイドバーから PDF を選んでください。")
        return

    try:
        catalog = cached_catalog(pdf_path.strip())
    except Exception as exc:
        with st.sidebar:
            _render_file_menu(pdf_path, candidate_paths, active_project, None, None)
        st.error(f"PDF の読み込みに失敗しました: {exc}")
        return

    ensure_local_asset_server()
    pdf_path_value = str(catalog.pdf_path)
    fingerprint = pdf_fingerprint(pdf_path_value)
    pdf_state = _get_pdf_state(fingerprint)
    if not pdf_state.get("deck_data_loaded", False):
        pdf_state["card_deck_ids"] = {}
        pdf_state["deck_catalog"] = {}
        pdf_state["deck_archetype"] = {}
        pdf_state["deck_url"] = {}
        pdf_state["deck_card_quantities"] = {}

    if active_project_payload is not None:
        project_state = _normalize_project_state(active_project_payload)
        labels = project_state["labels"]
    else:
        stored_state = load_card_state_store(pdf_path_value)
        labels = stored_state.labels

    deck_catalog = {
        str(deck_id).strip(): str(deck_name).strip()
        for deck_id, deck_name in pdf_state.get("deck_catalog", {}).items()
        if str(deck_id).strip()
    }

    card_df = pd.DataFrame(
        [
            {
                "card_id": card.card_id,
                "name": card.name,
                "label_list": normalize_label_list(labels.get(card.card_id)),
                "label": label_text(labels.get(card.card_id)),
                "deck_list": normalize_label_list(pdf_state.get("card_deck_ids", {}).get(card.card_id)),
                "deck_text": ", ".join(
                    _format_deck_option(deck_id, deck_catalog)
                    for deck_id in normalize_label_list(pdf_state.get("card_deck_ids", {}).get(card.card_id))
                ),
                "expansion": card.expansion,
                "collection_no": card.collection_no or "-",
                "target_page": card.target_page,
            }
            for card in catalog.cards
        ]
    )
    # 検索用の連結キー列を一度だけ作り、検索を str.contains でベクトル化する
    # （従来は検索のたびに DataFrame.apply(axis=1) で全行を Python ループしていた）
    card_df["_haystack"] = (
        card_df["card_id"].astype(str)
        + " "
        + card_df["name"].astype(str)
        + " "
        + card_df["label"].astype(str)
        + " "
        + card_df["deck_text"].astype(str)
        + " "
        + card_df["expansion"].astype(str)
        + " "
        + card_df["collection_no"].astype(str)
        + " "
        + card_df["target_page"].astype(str)
    ).str.casefold()
    card_ids = [int(card_id) for card_id in card_df["card_id"]]
    valid_card_ids = set(card_ids)

    if active_project_payload is not None:
        pdf_state["card_deck_ids"] = {
            int(card_id): normalize_label_list(deck_ids)
            for card_id, deck_ids in pdf_state.get("card_deck_ids", {}).items()
            if int(card_id) in valid_card_ids and normalize_label_list(deck_ids)
        }
        pdf_state["deck_catalog"] = {
            str(deck_id).strip(): str(deck_name).strip()
            for deck_id, deck_name in pdf_state.get("deck_catalog", {}).items()
            if str(deck_id).strip() and str(deck_name).strip()
        }
        pdf_state["deck_card_quantities"] = {
            deck_id: {
                int(card_id): int(quantity)
                for card_id, quantity in recipe.items()
                if int(card_id) in valid_card_ids and int(quantity) > 0
            }
            for deck_id, recipe in _normalize_deck_card_quantities(pdf_state.get("deck_card_quantities", {})).items()
        }
        pdf_state["deck_card_quantities"] = {
            deck_id: recipe for deck_id, recipe in pdf_state["deck_card_quantities"].items() if recipe
        }
        pdf_state["print_excluded_card_ids"] = {
            int(card_id)
            for card_id in project_state["print_excluded_card_ids"]
            if int(card_id) in valid_card_ids
        }
        pdf_state["work_list_card_ids"] = {
            int(card_id) for card_id in project_state["work_list_card_ids"] if int(card_id) in valid_card_ids
        }
        pdf_state["print_quantities"] = {
            int(card_id): int(quantity)
            for card_id, quantity in project_state["print_quantities"].items()
            if int(card_id) in valid_card_ids and int(quantity) > 0
        }
        pdf_state["print_order"] = [
            int(card_id)
            for card_id in project_state["print_order"]
            if int(card_id) in pdf_state["print_quantities"]
        ]
    else:
        pdf_state["card_deck_ids"] = {
            int(card_id): normalize_label_list(deck_ids)
            for card_id, deck_ids in pdf_state.get("card_deck_ids", {}).items()
            if int(card_id) in valid_card_ids and normalize_label_list(deck_ids)
        }
        pdf_state["deck_catalog"] = {
            str(deck_id).strip(): str(deck_name).strip()
            for deck_id, deck_name in pdf_state.get("deck_catalog", {}).items()
            if str(deck_id).strip() and str(deck_name).strip()
        }
        pdf_state["deck_card_quantities"] = {
            deck_id: {
                int(card_id): int(quantity)
                for card_id, quantity in recipe.items()
                if int(card_id) in valid_card_ids and int(quantity) > 0
            }
            for deck_id, recipe in _normalize_deck_card_quantities(pdf_state.get("deck_card_quantities", {})).items()
        }
        pdf_state["deck_card_quantities"] = {
            deck_id: recipe for deck_id, recipe in pdf_state["deck_card_quantities"].items() if recipe
        }
        pdf_state["print_excluded_card_ids"] = {
            int(card_id) for card_id in pdf_state.get("print_excluded_card_ids", set()) if int(card_id) in valid_card_ids
        }
        stored_work_list_card_ids = {
            int(card_id)
            for card_id in getattr(stored_state, "work_list_card_ids", frozenset())
            if int(card_id) in valid_card_ids
        }
        if stored_work_list_card_ids:
            pdf_state["work_list_card_ids"] = stored_work_list_card_ids
        elif getattr(stored_state, "excluded_card_ids", frozenset()):
            legacy_excluded_card_ids = {
                int(card_id)
                for card_id in getattr(stored_state, "excluded_card_ids", frozenset())
                if int(card_id) in valid_card_ids
            }
            pdf_state["work_list_card_ids"] = valid_card_ids.difference(legacy_excluded_card_ids)
        else:
            pdf_state["work_list_card_ids"] = {
                int(card_id) for card_id in pdf_state["work_list_card_ids"] if int(card_id) in valid_card_ids
            }
    pdf_state["work_selected_card_ids"] = {
        int(card_id) for card_id in pdf_state["work_selected_card_ids"] if int(card_id) in valid_card_ids
    }
    _sync_print_queue_to_work_list(pdf_state)
    # 既定の出力先（空になっていたら必ず埋め直す。空のままだと '.' へ書こうとして失敗するため）
    filtered_output_key = f"filtered_output_path_{fingerprint}"
    if not str(st.session_state.get(filtered_output_key, "")).strip():
        if active_project:
            st.session_state[filtered_output_key] = str(_default_project_output_path(active_project, "worklist"))
        else:
            st.session_state[filtered_output_key] = str(default_output_path(pdf_path_value))
    print_output_key = f"print_output_path_{fingerprint}"
    if not str(st.session_state.get(print_output_key, "")).strip():
        if active_project:
            st.session_state[print_output_key] = str(_default_project_output_path(active_project, "print"))
        else:
            st.session_state[print_output_key] = str(default_print_output_path(pdf_path_value))

    _mount_unsaved_changes_guard(
        enabled=bool(active_project and _is_project_dirty(active_project)),
        message="未保存の変更があります。保存してから閉じてください。",
    )
    screen = _render_sidebar(
        pdf_path_value, candidate_paths, active_project, labels, pdf_state, len(catalog.cards)
    )

    if screen == SCREEN_WORKLIST:
        _render_worklist_screen(
            card_df, catalog, pdf_path_value, fingerprint, pdf_state, labels, filtered_output_key
        )
    elif screen == SCREEN_PRINT:
        _render_print_screen(card_df, pdf_path_value, fingerprint, pdf_state, labels, print_output_key)
    else:
        _render_card_screen(card_df, pdf_path_value, fingerprint, pdf_state, labels)


def _ensure_app_state() -> None:
    st.session_state.setdefault("pdf_states", {})
    st.session_state.setdefault("pdf_path_input", DEFAULT_PDF_PATH)
    st.session_state.setdefault("candidate_pdf_choice", "手入力のまま")
    st.session_state.setdefault("pdf_picker_mode", "compact")
    st.session_state.setdefault("active_project_file", "")
    st.session_state.setdefault("project_choice", "")
    st.session_state.setdefault("project_dirty_flags", {})
    st.session_state.setdefault("new_project_name", "")
    st.session_state.setdefault("active_screen", SCREEN_CARDS)
    st.session_state.setdefault("active_workflow", "印刷")
    current_tab = st.session_state.setdefault("active_tab", "選択")
    if current_tab == "カード一覧":
        st.session_state["active_tab"] = "選択"
    elif current_tab in {"削除", "除外"}:
        st.session_state["active_tab"] = "作業リスト"


def _active_project_file() -> str:
    return str(st.session_state.get("active_project_file", "") or "").strip()


def _project_dirty_flags() -> dict[str, bool]:
    return st.session_state.setdefault("project_dirty_flags", {})


def _is_project_dirty(project_file: str | Path | None = None) -> bool:
    target = str(Path(project_file).expanduser().resolve()) if project_file else _active_project_file()
    if not target:
        return False
    return bool(_project_dirty_flags().get(target, False))


def _mark_project_dirty(project_file: str | Path | None = None) -> None:
    target = str(Path(project_file).expanduser().resolve()) if project_file else _active_project_file()
    if not target:
        return
    _project_dirty_flags()[target] = True


def _clear_project_dirty(project_file: str | Path | None = None) -> None:
    target = str(Path(project_file).expanduser().resolve()) if project_file else _active_project_file()
    if not target:
        return
    _project_dirty_flags()[target] = False


def _set_active_project(project_file: str | Path) -> None:
    resolved = str(Path(project_file).expanduser().resolve())
    st.session_state["active_project_file"] = resolved
    _project_dirty_flags().setdefault(resolved, False)


def _clear_active_project() -> None:
    st.session_state["active_project_file"] = ""


def _mount_unsaved_changes_guard(*, enabled: bool, message: str) -> None:
    components.html(
        f"""
        <script>
        (() => {{
          const parentWin = window.parent;
          const enabled = {json.dumps(enabled)};
          const message = {json.dumps(message)};
          if (parentWin.__cardPdfUnsavedHandler) {{
            parentWin.removeEventListener("beforeunload", parentWin.__cardPdfUnsavedHandler);
            delete parentWin.__cardPdfUnsavedHandler;
          }}
          if (enabled) {{
            const handler = (event) => {{
              event.preventDefault();
              event.returnValue = message;
              return message;
            }};
            parentWin.__cardPdfUnsavedHandler = handler;
            parentWin.addEventListener("beforeunload", handler);
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _discover_pdf_candidates(default_pdf_path: str) -> list[str]:
    roots = {
        Path(default_pdf_path).expanduser().parent,
        Path(__file__).resolve().parent,
    }
    candidates: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.pdf"):
            candidates[str(path.resolve())] = path

    default_resolved = str(Path(default_pdf_path).expanduser().resolve())
    return sorted(
        candidates,
        key=lambda item: (
            0 if item == default_resolved else 1,
            Path(item).name.casefold(),
            item.casefold(),
        ),
    )


def _find_latest_output_candidate(candidate_paths: list[str]) -> str | None:
    if not candidate_paths:
        return None

    default_resolved = str(Path(DEFAULT_PDF_PATH).expanduser().resolve())
    non_default = [path for path in candidate_paths if str(Path(path).resolve()) != default_resolved]
    pool = non_default or candidate_paths
    return max(pool, key=lambda item: Path(item).stat().st_mtime_ns)


def _get_pdf_state(fingerprint: str) -> dict:
    pdf_states = st.session_state["pdf_states"]
    if fingerprint not in pdf_states:
        pdf_states[fingerprint] = {
            "work_selected_card_ids": set(),
            "work_list_card_ids": set(),
            "card_deck_ids": {},
            "deck_catalog": {},
            "deck_archetype": {},
            "deck_url": {},
            "deck_card_quantities": {},
            "deck_data_loaded": False,
            "print_excluded_card_ids": set(),
            "print_quantities": {},
            "print_order": [],
        }

    state = pdf_states[fingerprint]
    state["work_selected_card_ids"] = set(state.get("work_selected_card_ids", set()))
    state["work_list_card_ids"] = set(
        state.get("work_list_card_ids", state.get("excluded_card_ids", state.get("delete_card_ids", set())))
    )
    state["deck_data_loaded"] = bool(state.get("deck_data_loaded", False))
    state["print_excluded_card_ids"] = {
        int(card_id) for card_id in state.get("print_excluded_card_ids", set()) if str(card_id).strip()
    }
    state["card_deck_ids"] = {
        int(card_id): normalize_label_list(deck_ids)
        for card_id, deck_ids in state.get("card_deck_ids", {}).items()
        if normalize_label_list(deck_ids)
    }
    state["deck_catalog"] = {
        str(deck_id).strip(): str(deck_name).strip()
        for deck_id, deck_name in state.get("deck_catalog", {}).items()
        if str(deck_id).strip() and str(deck_name).strip()
    }
    state["deck_archetype"] = {
        str(deck_id).strip(): str(archetype).strip()
        for deck_id, archetype in state.get("deck_archetype", {}).items()
        if str(deck_id).strip() and str(archetype).strip()
    }
    state["deck_url"] = {
        str(deck_id).strip(): str(url).strip()
        for deck_id, url in state.get("deck_url", {}).items()
        if str(deck_id).strip() and str(url).strip()
    }
    state["deck_card_quantities"] = _normalize_deck_card_quantities(state.get("deck_card_quantities", {}))
    state["print_quantities"] = {
        int(card_id): int(quantity)
        for card_id, quantity in state.get("print_quantities", {}).items()
        if int(quantity) > 0
    }
    state["print_order"] = [int(card_id) for card_id in state.get("print_order", [])]
    state["print_order"] = [card_id for card_id in state["print_order"] if card_id in state["print_quantities"]]
    return state


def _persist_active_project_state(pdf_path: str, labels: dict[int, list[str]], pdf_state: dict) -> None:
    active_project = _active_project_file()
    if not active_project:
        return
    _mark_project_dirty(active_project)


def _save_card_state(pdf_path: str, labels: dict[int, list[str]], work_list_card_ids: set[int]) -> None:
    if _active_project_file():
        pdf_state = _get_pdf_state(pdf_fingerprint(pdf_path))
        pdf_state["work_list_card_ids"] = set(work_list_card_ids)
        _mark_project_dirty()
        return
    save_card_state_store(pdf_path, labels, work_list_card_ids)


def _push_exclude_feedback(level: str, message: str) -> None:
    feedback = st.session_state.setdefault("exclude_feedback_messages", [])
    feedback.append({"level": level, "message": message})


def _render_exclude_feedback() -> None:
    for entry in st.session_state.pop("exclude_feedback_messages", []):
        level = entry.get("level", "info")
        message = entry.get("message", "")
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)


def _sync_print_queue_to_work_list(pdf_state: dict) -> None:
    pdf_state["print_excluded_card_ids"].intersection_update(pdf_state["work_list_card_ids"])
    _remove_from_print_queue(
        pdf_state,
        set(pdf_state["print_order"]).difference(pdf_state["work_list_card_ids"]),
    )
    for card_id in sorted(pdf_state["work_list_card_ids"]):
        if card_id in pdf_state["print_excluded_card_ids"]:
            continue
        if card_id not in pdf_state["print_quantities"]:
            pdf_state["print_order"].append(card_id)
            pdf_state["print_quantities"][card_id] = 1


def _resolve_card_ids_from_page_numbers(catalog, page_numbers: frozenset[int]) -> tuple[set[int], list[int], list[int]]:
    page_to_card_id = {card.target_page: card.card_id for card in catalog.cards}
    resolved_card_ids: set[int] = set()
    resolved_pages: list[int] = []
    ignored_pages: list[int] = []
    for page_number in sorted(page_numbers):
        if page_number < 1 or page_number > catalog.total_pages:
            ignored_pages.append(page_number)
            continue
        if page_number <= catalog.index_page_count:
            ignored_pages.append(page_number)
            continue
        card_id = page_to_card_id.get(page_number)
        if card_id is None:
            ignored_pages.append(page_number)
            continue
        resolved_card_ids.add(card_id)
        resolved_pages.append(page_number)
    return resolved_card_ids, resolved_pages, ignored_pages


def _set_work_selection(
    all_card_ids: list[int],
    fingerprint: str,
    pdf_state: dict,
    target_card_ids: list[int],
    *,
    mode: str,
) -> None:
    target_set = {int(card_id) for card_id in target_card_ids}
    if mode == "replace":
        next_selected = set(target_set)
    elif mode == "remove":
        next_selected = set(pdf_state["work_selected_card_ids"]).difference(target_set)
    else:
        next_selected = set(pdf_state["work_selected_card_ids"]).union(target_set)

    pdf_state["work_selected_card_ids"] = next_selected
    for card_id in all_card_ids:
        key = _work_checkbox_key(fingerprint, card_id)
        st.session_state[key] = int(card_id) in next_selected


def _render_print_settings(fingerprint: str, *, disabled: bool = False) -> PrintLayoutSettings:
    cards_per_sheet = st.selectbox(
        "1ページあたりのカード数",
        options=PRINT_SHEET_OPTIONS,
        format_func=lambda value: "auto (A4最大)" if value == "auto" else value,
        key=f"cards_per_sheet_{fingerprint}",
        disabled=disabled,
    )
    settings_cols = st.columns(3, gap="small")
    card_width_mm = settings_cols[0].number_input(
        "カード幅(mm)",
        min_value=40.0,
        max_value=80.0,
        step=0.5,
        value=63.0,
        key=f"card_width_mm_{fingerprint}",
        disabled=disabled,
    )
    card_height_mm = settings_cols[1].number_input(
        "カード高さ(mm)",
        min_value=50.0,
        max_value=100.0,
        step=0.5,
        value=88.0,
        key=f"card_height_mm_{fingerprint}",
        disabled=disabled,
    )
    gap_mm = settings_cols[2].number_input(
        "カード間余白(mm)",
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        value=2.0,
        key=f"card_gap_mm_{fingerprint}",
        disabled=disabled,
    )
    return PrintLayoutSettings(
        page_size="A4",
        card_width_mm=float(card_width_mm),
        card_height_mm=float(card_height_mm),
        gap_mm=float(gap_mm),
        cards_per_sheet=cards_per_sheet,
    )


def _build_print_items(print_order: list[int], print_quantities: dict[int, int], labels: dict[int, list[str]]) -> list[PrintItem]:
    items: list[PrintItem] = []
    for card_id in print_order:
        quantity = int(print_quantities.get(card_id, 0))
        if quantity <= 0:
            continue
        items.append(PrintItem(card_id=card_id, quantity=quantity, label=label_text(labels.get(card_id))))
    return items


def _add_to_print_queue(pdf_state: dict, ordered_selected_ids: list[int]) -> None:
    for card_id in ordered_selected_ids:
        pdf_state["print_excluded_card_ids"].discard(card_id)
        if card_id not in pdf_state["print_quantities"]:
            pdf_state["print_order"].append(card_id)
            pdf_state["print_quantities"][card_id] = 1


def _deck_recipe_summary(pdf_state: dict, deck_id: str) -> tuple[int, int]:
    recipe = _normalize_deck_card_quantities(pdf_state.get("deck_card_quantities", {})).get(str(deck_id).strip(), {})
    if not recipe:
        return (0, 0)
    return (len(recipe), sum(int(quantity) for quantity in recipe.values() if int(quantity) > 0))


def _add_deck_to_print_queue(pdf_state: dict, deck_id: str) -> tuple[int, int]:
    recipe = _normalize_deck_card_quantities(pdf_state.get("deck_card_quantities", {})).get(str(deck_id).strip(), {})
    added_kinds = 0
    added_cards = 0
    for card_id, quantity in recipe.items():
        qty = int(quantity)
        if qty <= 0:
            continue
        pdf_state["work_list_card_ids"].add(int(card_id))
        pdf_state["print_excluded_card_ids"].discard(int(card_id))
        if int(card_id) not in pdf_state["print_quantities"]:
            pdf_state["print_order"].append(int(card_id))
            pdf_state["print_quantities"][int(card_id)] = 0
            added_kinds += 1
        pdf_state["print_quantities"][int(card_id)] += qty
        added_cards += qty
    return (added_kinds, added_cards)


def _remove_from_print_queue(pdf_state: dict, card_ids: set[int]) -> None:
    for card_id in list(card_ids):
        pdf_state["print_quantities"].pop(card_id, None)
    pdf_state["print_order"] = [card_id for card_id in pdf_state["print_order"] if card_id not in card_ids]


def _clear_work_selection_widgets(card_ids: list[int], fingerprint: str) -> None:
    for card_id in card_ids:
        key = _work_checkbox_key(fingerprint, card_id)
        if key in st.session_state:
            st.session_state[key] = False


def _work_checkbox_key(fingerprint: str, card_id: int) -> str:
    return f"work_select_{fingerprint}_{card_id}"


def _filter_cards(card_df: pd.DataFrame, search_text: str, label_filter: str, deck_filter: str = "すべて") -> pd.DataFrame:
    working = card_df
    if label_filter != "すべて":
        working = working.loc[working["label_list"].map(lambda items: label_filter in normalize_label_list(items))]
    if deck_filter != "すべて":
        working = working.loc[working["deck_list"].map(lambda items: deck_filter in normalize_label_list(items))]
    needle = search_text.strip().casefold()
    if not needle:
        return working.copy()

    if "_haystack" in working.columns:
        mask = working["_haystack"].str.contains(needle, regex=False, na=False)
    else:  # フォールバック（_haystack 列が無い場合）
        mask = working.apply(
            lambda row: needle
            in " ".join(
                [
                    str(row["card_id"]),
                    str(row["name"]),
                    str(row["label"]),
                    str(row.get("deck_text", "")),
                    str(row["expansion"]),
                    str(row["collection_no"]),
                    str(row["target_page"]),
                ]
            ).casefold(),
            axis=1,
        )
    return working.loc[mask].copy()


def _sort_cards(card_df: pd.DataFrame, sort_key: str, group_by_label: bool) -> pd.DataFrame:
    if card_df.empty:
        return card_df.copy()

    working = card_df.copy()
    working["label_sort"] = working["label"].map(lambda value: str(value).casefold() if str(value).strip() else "\uffff")
    secondary_key = sort_key if sort_key != "label" else "card_id"
    sort_columns = ["label_sort", secondary_key, "card_id"] if group_by_label else [sort_key, "card_id"]
    ascending = [True] * len(sort_columns)
    return working.sort_values(sort_columns, ascending=ascending, kind="stable").drop(columns=["label_sort"], errors="ignore")


def _format_deck_option(deck_id: str, deck_catalog: dict[str, str], *, strip_prefix: str = "") -> str:
    if deck_id == "すべて":
        return "すべてのデッキ"
    deck_name = str(deck_catalog.get(deck_id, "")).strip()
    if not deck_name:
        return deck_id
    # アーキタイプ(ラベル)で絞り込み済みのときは、先頭の重複するアーキタイプ名を省いて見やすくする。
    if strip_prefix and deck_name.startswith(strip_prefix):
        stripped = deck_name[len(strip_prefix):].strip()
        if stripped:
            deck_name = stripped
    # 表示は読みやすい名前を主に。重複しても区別できるよう短いIDを末尾に添える。
    return f"{deck_name}（{deck_id[:4]}）"


def _format_label_filter_option(label_value: str) -> str:
    return "すべてのラベル" if label_value == "すべて" else label_value


def _order_cards_for_display(card_df: pd.DataFrame, work_list_card_ids: set[int]) -> pd.DataFrame:
    if card_df.empty:
        return card_df.copy()

    working = card_df.copy()
    working["in_work_list"] = working["card_id"].isin(work_list_card_ids)
    normal_cards = working.loc[~working["in_work_list"]].copy()
    work_list_cards = working.loc[working["in_work_list"]].copy()
    return pd.concat([normal_cards, work_list_cards], ignore_index=True)


# ============================================================
# モダン Web アプリ UI（サイドバー + ワイド + カードグリッド）
# 既存のバックエンド処理・状態管理ヘルパーはそのまま再利用し、
# 見せ方（プレゼンテーション層）だけを作り直している。
# ============================================================

SCREEN_CARDS = "📇 カード"
SCREEN_WORKLIST = "📋 作業リスト"
SCREEN_PRINT = "🖨 印刷"
CARD_PAGE_SIZE_OPTIONS = [24, 48, 96]
THUMB_WIDTH = 180
TIER_RANKING_URL = "https://pokeka-win-decks.jp/tier-ranking"
TIER_SITE_BASE = "https://pokeka-win-decks.jp"
SORT_OPTIONS = [
    ("card_id", "カードID順"),
    ("name", "カード名順"),
    ("label", "ラベル順"),
    ("expansion", "エキスパンション順"),
    ("target_page", "ページ順"),
]


def _save_existing_project(active_project: str, labels: dict[int, list[str]], pdf_state: dict) -> None:
    try:
        payload = _load_project_payload(active_project)
        source_pdf_path = _project_source_pdf_path(active_project, payload)
        _save_project_payload(
            active_project,
            project_name=str(payload.get("project_name", Path(active_project).parent.name)),
            source_pdf_path=source_pdf_path,
            labels=labels,
            card_deck_ids={},
            deck_catalog={},
            work_list_card_ids=set(pdf_state["work_list_card_ids"]),
            print_excluded_card_ids=set(pdf_state.get("print_excluded_card_ids", set())),
            print_quantities=dict(pdf_state["print_quantities"]),
            print_order=list(pdf_state["print_order"]),
            created_at=str(payload.get("created_at", _iso_now())),
        )
    except Exception as exc:
        st.error(f"保存に失敗しました: {exc}")
    else:
        _clear_project_dirty(active_project)
        st.success("保存しました。")
        st.rerun()


def _create_project_action(pdf_path: str, name: str, labels: dict[int, list[str]], pdf_state: dict) -> None:
    try:
        created = _create_project_from_pdf(
            pdf_path,
            name,
            labels=labels,
            card_deck_ids={},
            deck_catalog={},
            work_list_card_ids=set(pdf_state["work_list_card_ids"]),
            print_excluded_card_ids=set(pdf_state.get("print_excluded_card_ids", set())),
            print_quantities=dict(pdf_state["print_quantities"]),
            print_order=list(pdf_state["print_order"]),
        )
    except Exception as exc:
        st.error(f"作成に失敗しました: {exc}")
    else:
        _set_active_project(created)
        _clear_project_dirty(created)
        st.success("プロジェクトを作成しました。")
        st.rerun()


def _render_file_menu(
    pdf_path: str,
    candidate_paths: list[str],
    active_project: str,
    labels: dict[int, list[str]] | None,
    pdf_state: dict | None,
) -> None:
    st.markdown("###### プロジェクト")
    project_candidates = _discover_project_candidates()
    if project_candidates:
        default_index = (
            project_candidates.index(active_project)
            if active_project in project_candidates
            else 0
        )
        choice = st.selectbox(
            "保存済みプロジェクト",
            project_candidates,
            index=default_index,
            format_func=lambda value: Path(value).parent.name,
            key="project_choice_menu",
        )
        open_cols = st.columns(2, gap="small")
        if open_cols[0].button("開く", use_container_width=True):
            if active_project and _is_project_dirty(active_project):
                st.warning("未保存の変更があります。先に保存してください。")
            else:
                _set_active_project(choice)
                st.rerun()
        if open_cols[1].button("閉じる", use_container_width=True, disabled=not active_project):
            if active_project and _is_project_dirty(active_project):
                st.warning("未保存の変更があります。先に保存してください。")
            else:
                _clear_active_project()
                st.rerun()
    else:
        st.caption("保存済みプロジェクトはまだありません。")

    if labels is not None and pdf_state is not None:
        st.markdown("###### 保存")
        default_name = Path(pdf_path).stem
        if not st.session_state.get("new_project_name"):
            st.session_state["new_project_name"] = default_name
        project_name = st.text_input("プロジェクト名", key="new_project_name")
        save_cols = st.columns(2, gap="small")
        if active_project:
            if save_cols[0].button("💾 上書き保存", type="primary", use_container_width=True):
                _save_existing_project(active_project, labels, pdf_state)
        else:
            save_cols[0].button("💾 上書き保存", use_container_width=True, disabled=True)
        if save_cols[1].button("➕ 新規保存", use_container_width=True):
            _create_project_action(pdf_path, project_name.strip() or default_name, labels, pdf_state)

    st.markdown("###### PDF")
    if active_project:
        st.caption("PDFを変えるには、いったんプロジェクトを閉じてください。")
    else:
        if st.button("元のPDFに戻す", use_container_width=True):
            st.session_state["pdf_path_input"] = DEFAULT_PDF_PATH
            st.rerun()
        latest = _find_latest_output_candidate(candidate_paths)
        if st.button("最新の出力PDFを開く", use_container_width=True, disabled=latest is None):
            if latest:
                st.session_state["pdf_path_input"] = latest
                st.rerun()
        if candidate_paths:
            current = st.session_state.get("pdf_path_input", DEFAULT_PDF_PATH)
            idx = candidate_paths.index(current) if current in candidate_paths else 0
            pick = st.selectbox(
                "候補から選ぶ",
                candidate_paths,
                index=idx,
                format_func=lambda value: Path(value).name,
                key="pdf_candidate_menu",
            )
            if st.button("このPDFを開く", use_container_width=True):
                st.session_state["pdf_path_input"] = pick
                st.rerun()
        st.text_input("パスを直接入力", key="pdf_path_input")


def _render_summary_sidebar(pdf_state: dict, total_cards: int) -> None:
    work_list = len(pdf_state["work_list_card_ids"])
    selected = len(pdf_state["work_selected_card_ids"])
    kinds = len(pdf_state["print_quantities"])
    sheets = sum(int(v) for v in pdf_state["print_quantities"].values())
    st.markdown(
        "<div class='side-stats'>"
        f"<div class='side-stat'><span class='ss-num'>{selected}</span><span class='ss-lbl'>選択中</span></div>"
        f"<div class='side-stat'><span class='ss-num'>{work_list}</span><span class='ss-lbl'>作業リスト</span></div>"
        f"<div class='side-stat'><span class='ss-num'>{sheets}</span><span class='ss-lbl'>印刷 {kinds}種</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _normalize_card_name(name: str) -> str:
    """カード名照合用の正規化。
    - 末尾の技/特性名などの括弧書き（…）(…) を除去
    - 【】「」『』 などの装飾括弧や記号・空白を除去（中身は残す）
    例: 「ドラパルトex(ジェットヘッド)」「基本【炎】エネルギー」→ 比較可能な形に揃える。
    """
    text = re.sub(r"[（(][^（）()]*[)）]", "", str(name))
    text = re.sub(r"[【】「」『』\[\]（）()・,，、。.\s]", "", text)
    return text.casefold()


def _resolve_tier_label_names(raw_labels: object, card_names: Iterable[str]) -> list[str]:
    card_name_by_norm: dict[str, str] = {}
    for card_name in card_names:
        normalized = _normalize_card_name(str(card_name))
        if normalized and normalized not in card_name_by_norm:
            card_name_by_norm[normalized] = str(card_name)

    resolved: list[str] = []
    seen: set[str] = set()
    for raw_label in normalize_label_list(raw_labels):
        normalized = _normalize_card_name(raw_label)
        if not normalized:
            continue
        matched = card_name_by_norm.get(normalized)
        if matched and matched not in seen:
            resolved.append(matched)
            seen.add(matched)
    return resolved


def _strip_legacy_deck_ids(pdf_state: dict) -> None:
    legacy_pattern = re.compile(r"^PWD-\d{3}$")
    pdf_state["deck_catalog"] = {
        deck_id: deck_name
        for deck_id, deck_name in pdf_state.get("deck_catalog", {}).items()
        if not legacy_pattern.fullmatch(str(deck_id).strip())
    }
    normalized: dict[int, list[str]] = {}
    for card_id, deck_ids in pdf_state.get("card_deck_ids", {}).items():
        kept = [deck_id for deck_id in normalize_label_list(deck_ids) if not legacy_pattern.fullmatch(str(deck_id).strip())]
        if kept:
            normalized[int(card_id)] = kept
    pdf_state["card_deck_ids"] = normalized


def _cleanup_existing_labels(
    labels: dict[int, list[str]],
    card_names: Iterable[str],
) -> tuple[int, int, int]:
    updated_cards = 0
    removed_cards = 0
    total_labels_after = 0

    for card_id in list(labels):
        current = normalize_label_list(labels.get(card_id))
        cleaned = _resolve_tier_label_names(current, card_names)
        if cleaned:
            total_labels_after += len(cleaned)
            if cleaned != current:
                labels[int(card_id)] = cleaned
                updated_cards += 1
        else:
            labels.pop(int(card_id), None)
            removed_cards += 1

    return updated_cards, removed_cards, total_labels_after


def _run_label_cleanup(pdf_path: str, labels: dict[int, list[str]], pdf_state: dict, card_names: Iterable[str]) -> None:
    updated_cards, removed_cards, total_labels_after = _cleanup_existing_labels(labels, card_names)
    _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
    st.session_state["label_cleanup_result"] = (
        "success",
        f"ラベル整理を実行しました。更新 {updated_cards}件 / 完全削除 {removed_cards}件 / 残りラベル {total_labels_after}件",
    )
    st.rerun()


TIER_FETCH_WORKERS = 8


def _tier_cache_path() -> Path:
    cache_dir = Path(__file__).resolve().parent / ".cache" / "tier_ranking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{date.today().isoformat()}.json"


def _load_tier_cache() -> dict | None:
    path = _tier_cache_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _store_tier_cache(data: dict) -> None:
    try:
        _tier_cache_path().write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _tier_http_get(url: str) -> str:
    import requests  # 遅延 import（未導入環境でもアプリ自体は起動できるように）

    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (card-print-tool)"}, timeout=25)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return resp.text


def _extract_deck_label(slug: str, html: str) -> str:
    title_match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.IGNORECASE)
    if not title_match:
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    if title_match:
        text = re.sub(r"\s+", " ", title_match.group(1)).strip()
        bracket_match = re.search(r"\[([^\[\]]+)\]", text)
        if not bracket_match:
            bracket_match = re.search(r"【([^【】]+)】", text)
        if bracket_match:
            bracket_text = re.sub(r"\s+", " ", bracket_match.group(1)).strip()
            if bracket_text:
                return bracket_text
        text = re.sub(r"\s*[|｜].*$", "", text).strip()
        text = re.sub(r"\s*[-–—].*$", "", text).strip()
        if text:
            return text
    fallback = unquote(slug).replace("-", " ").replace("_", " ").strip()
    return re.sub(r"\s+", " ", fallback) or slug


def _extract_detail_deck_path(html: str) -> str | None:
    match = re.search(r'/(season/[^"\']+/deck/[^"\']+)', html)
    if not match:
        return None
    return "/" + match.group(1).lstrip("/")


def _extract_detail_deck_paths(html: str) -> list[str]:
    paths: list[str] = []
    seen_paths: set[str] = set()
    for match in re.findall(r'/(season/[^"\']+/deck/[^"\']+)', html):
        path = "/" + str(match).lstrip("/")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        paths.append(path)
    return paths


def _extract_recipe_card_names(html: str) -> list[str]:
    table_match = re.search(
        r'<table[^>]+class=["\'][^"\']*card_list_table[^"\']*["\'][^>]*>(.*?)</table>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    target_html = table_match.group(1) if table_match else html
    names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in re.findall(r'href=["\'][^"\']*/card/([^"\']+?)(?:\?[^"\']*)?["\']', target_html, re.IGNORECASE):
        name = unquote(str(raw_name)).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        names.append(name)
    return names


def _extract_recipe_cards(html: str) -> list[tuple[str, int]]:
    table_match = re.search(
        r'<table[^>]+class=["\'][^"\']*card_list_table[^"\']*["\'][^>]*>(.*?)</table>',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    target_html = table_match.group(1) if table_match else html
    cards: list[tuple[str, int]] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", target_html, re.IGNORECASE | re.DOTALL):
        name_match = re.search(r'/card/([^"\'/?]+)', row_html, re.IGNORECASE)
        qty_match = re.search(r'<span[^>]+class=["\'][^"\']*number[^"\']*["\'][^>]*>\s*(\d+)\s*</span>', row_html, re.IGNORECASE)
        if not name_match or not qty_match:
            continue
        name = unquote(str(name_match.group(1))).strip()
        quantity = int(qty_match.group(1))
        if name and quantity > 0:
            cards.append((name, quantity))
    return cards


def _extract_recipe_meta_label(html: str, archetype_label: str) -> str:
    """レシピページの og:title 等から、大会・日付・順位を抜き出して見分けの付くデッキ名を作る。
    例: 「ドラパルトex 5月6日 〇〇市 優勝」。取れない場合はアーキタイプ名にフォールバック。
    """
    title_match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.IGNORECASE
    )
    if not title_match:
        title_match = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    date_match = re.search(r"(\d{1,2}月\d{1,2}日)", title)
    rank_match = re.search(r"(優勝|準優勝|ベスト\d+|トップ\d+|\d+位)", title)
    place_match = re.search(r"([^\s「」<>【】/]{1,8}?(?:県|都|府|道|市))", title)
    bits = [archetype_label]
    if date_match:
        bits.append(date_match.group(1))
    if place_match:
        bits.append(place_match.group(1))
    if rank_match:
        bits.append(rank_match.group(1))
    label = re.sub(r"\s+", " ", " ".join(bit for bit in bits if bit).strip())
    return label or archetype_label


def _fetch_tier_adopted_cards(progress=None, *, force: bool = False) -> dict:
    """Tierランキング上位デッキを巡回し、各デッキで採用されているカード名・枚数を集計する。
    - 当日のディスクキャッシュがあれば即時返す（force=Trueで無視）。
    - レシピページ取得は ThreadPoolExecutor で並行化。progress(done, total) で進捗通知。
    返り値:
    {
      "deck_count": int,
      "deck_catalog": {deck_id: 表示名(大会/日付つき)},
      "cards": {正規化名: {"name", "decks", "deck_labels":[...], "deck_ids":[...]}},
      "deck_recipes": {deck_id: {"label", "cards": {正規化名: {"name", "quantity"}}}}
    }
    """
    if not force:
        cached = _load_tier_cache()
        if cached:
            if progress:
                progress(1, 1)
            return cached

    try:
        import requests  # noqa: F401  （未導入なら明示エラー）
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("requests が必要です。`pip install requests` を実行してください。") from exc

    tier_html = _tier_http_get(TIER_RANKING_URL)
    slugs: list[str] = []
    seen: set[str] = set()
    for slug in re.findall(r"/all-deck/([^\"'#?]+)", tier_html):
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)

    # Phase A: 各アーキタイプページを並行取得し、ラベルとレシピURL一覧を得る
    archetypes: dict[str, tuple[str, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=TIER_FETCH_WORKERS) as executor:
        future_map = {
            executor.submit(_tier_http_get, f"{TIER_SITE_BASE}/all-deck/{slug}"): slug for slug in slugs
        }
        for future in as_completed(future_map):
            slug = future_map[future]
            try:
                html = future.result()
            except Exception:
                continue
            label = _extract_deck_label(slug, html)
            paths = _extract_detail_deck_paths(html)
            if not paths:
                single = _extract_detail_deck_path(html)
                paths = [single] if single else []
            archetypes[slug] = (label, paths)

    # レシピジョブを作成（deck_id でアーキタイプ横断の重複を排除）
    jobs: list[tuple[str, str, str]] = []  # (deck_id, path, archetype_label)
    seen_ids: set[str] = set()
    for slug in slugs:
        if slug not in archetypes:
            continue
        label, paths = archetypes[slug]
        for path in paths:
            deck_id = path.rsplit("/", 1)[-1]
            if not deck_id or deck_id in seen_ids:
                continue
            seen_ids.add(deck_id)
            jobs.append((deck_id, path, label))

    total = len(jobs)
    if progress:
        progress(0, total)

    def _fetch_recipe(job: tuple[str, str, str]):
        deck_id, path, label = job
        try:
            detail_html = _tier_http_get(f"{TIER_SITE_BASE}{path}")
        except Exception:
            return None
        recipe_cards = _extract_recipe_cards(detail_html)
        if not recipe_cards:
            recipe_cards = [(name, 1) for name in _extract_recipe_card_names(detail_html)]
        if not recipe_cards:
            return None
        meta_label = _extract_recipe_meta_label(detail_html, label)
        return (deck_id, label, meta_label, recipe_cards, f"{TIER_SITE_BASE}{path}")

    cards: dict[str, dict] = {}
    deck_catalog: dict[str, str] = {}
    deck_archetype: dict[str, str] = {}
    deck_url: dict[str, str] = {}
    deck_recipes: dict[str, dict] = {}
    deck_count = 0
    done = 0
    # Phase B: 全レシピページを並行取得。集計は as_completed のメインスレッドで行う（スレッド安全）。
    with ThreadPoolExecutor(max_workers=TIER_FETCH_WORKERS) as executor:
        futures = [executor.submit(_fetch_recipe, job) for job in jobs]
        for future in as_completed(futures):
            done += 1
            if progress and (done % 8 == 0 or done == total):
                progress(done, total)
            result = future.result()
            if not result:
                continue
            deck_id, label, meta_label, recipe_cards, deck_page_url = result
            deck_count += 1
            deck_catalog[deck_id] = meta_label
            deck_archetype[deck_id] = label  # アーキタイプ(ラベル) ↔ デッキ(レシピ) の対応
            deck_url[deck_id] = deck_page_url  # 元サイトのレシピページURL
            deck_recipe_entry = deck_recipes.setdefault(deck_id, {"label": meta_label, "cards": {}})
            seen_norm: set[str] = set()
            for raw, quantity in recipe_cards:
                norm = _normalize_card_name(raw)
                if not norm or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                entry = cards.setdefault(norm, {"name": raw, "decks": 0, "deck_labels": [], "deck_ids": []})
                entry["decks"] += 1
                if label not in entry["deck_labels"]:
                    entry["deck_labels"].append(label)
                if deck_id not in entry["deck_ids"]:
                    entry["deck_ids"].append(deck_id)
                deck_recipe_entry["cards"][norm] = {"name": raw, "quantity": int(quantity)}

    result_data = {
        "deck_count": deck_count,
        "deck_catalog": deck_catalog,
        "deck_archetype": deck_archetype,
        "deck_url": deck_url,
        "cards": cards,
        "deck_recipes": deck_recipes,
    }
    _store_tier_cache(result_data)
    return result_data


def _import_tier_cards(
    card_df: pd.DataFrame,
    pdf_path: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
    min_decks: int,
) -> None:
    progress_bar = st.progress(0.0, text="Tierランキングを取得中...")

    def _report(done: int, total: int) -> None:
        if total <= 0:
            progress_bar.progress(1.0, text="集計中...")
            return
        ratio = max(0.0, min(1.0, done / total))
        progress_bar.progress(ratio, text=f"デッキレシピを取得中... {done}/{total}")

    try:
        data = _fetch_tier_adopted_cards(progress=_report)
    except Exception as exc:
        progress_bar.empty()
        st.session_state["tier_import_result"] = ("error", f"取得に失敗しました: {exc}")
        st.rerun()
        return
    progress_bar.progress(1.0, text="カードと照合中...")

    catalog_norm: dict[str, list[int]] = {}
    for card_id, name in zip(card_df["card_id"], card_df["name"]):
        catalog_norm.setdefault(_normalize_card_name(str(name)), []).append(int(card_id))
    catalog_card_names = [str(name) for name in card_df["name"]]

    matched_ids: set[int] = set()
    matched_names: set[str] = set()
    unmatched: list[str] = []
    labels_added = 0
    deck_refs_added = 0
    deck_catalog = {
        str(deck_id).strip(): str(deck_name).strip()
        for deck_id, deck_name in data.get("deck_catalog", {}).items()
        if str(deck_id).strip() and str(deck_name).strip()
    }
    pdf_state["deck_data_loaded"] = True
    pdf_state["card_deck_ids"] = {}
    pdf_state["deck_catalog"] = dict(deck_catalog)
    pdf_state["deck_archetype"] = {
        str(deck_id).strip(): str(archetype).strip()
        for deck_id, archetype in data.get("deck_archetype", {}).items()
        if str(deck_id).strip() and str(archetype).strip()
    }
    pdf_state["deck_url"] = {
        str(deck_id).strip(): str(url).strip()
        for deck_id, url in data.get("deck_url", {}).items()
        if str(deck_id).strip() and str(url).strip()
    }
    pdf_state["deck_card_quantities"] = {}
    _strip_legacy_deck_ids(pdf_state)
    for deck_id, recipe in data.get("deck_recipes", {}).items():
        deck_id_text = str(deck_id).strip()
        if not deck_id_text:
            continue
        resolved_recipe: dict[int, int] = {}
        for norm_name, card_info in dict(recipe.get("cards", {})).items():
            ids = catalog_norm.get(str(norm_name), [])
            if not ids:
                continue
            try:
                quantity = int(dict(card_info).get("quantity", 0))
            except (TypeError, ValueError):
                quantity = 0
            if quantity <= 0:
                continue
            for card_id in ids:
                resolved_recipe[int(card_id)] = max(int(quantity), int(resolved_recipe.get(int(card_id), 0)))
        if resolved_recipe:
            pdf_state["deck_card_quantities"][deck_id_text] = resolved_recipe
    for norm, info in data["cards"].items():
        ids = catalog_norm.get(norm)
        if ids:
            deck_ids = normalize_label_list(info.get("deck_ids", []))
            for card_id in ids:
                if _merge_string_values_for_card(pdf_state["card_deck_ids"], int(card_id), deck_ids):
                    deck_refs_added += 1
            if int(info["decks"]) < int(min_decks):
                continue
            matched_ids.update(ids)
            matched_names.add(info["name"])
            deck_labels = _resolve_tier_label_names(info.get("deck_labels", []), catalog_card_names)
            for card_id in ids:
                if _merge_labels_for_card(labels, int(card_id), deck_labels):
                    labels_added += 1
        else:
            unmatched.append(info["name"])

    if not matched_ids:
        st.session_state["tier_import_result"] = (
            "warning",
            f"{data['deck_count']}デッキを解析しましたが、条件に合う一致カードがありませんでした。",
        )
        st.rerun()
        return

    before = len(pdf_state["work_list_card_ids"])
    pdf_state["work_list_card_ids"].update(matched_ids)
    added = len(pdf_state["work_list_card_ids"]) - before
    _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
    unmatched_note = f" / 未一致 {len(unmatched)}種" if unmatched else ""
    st.session_state["tier_import_result"] = (
        "success",
        f"{data['deck_count']}デッキを解析。{len(matched_names)}種が一致し、作業リストに {added}枚 追加、ラベルを {labels_added}枚、デッキ番号を {deck_refs_added}枚 更新しました。{unmatched_note}",
    )
    st.rerun()


def _render_sidebar(
    pdf_path: str,
    candidate_paths: list[str],
    active_project: str,
    labels: dict[int, list[str]],
    pdf_state: dict,
    total_cards: int,
) -> str:
    with st.sidebar:
        st.markdown(
            "<div class='brand'><div class='brand-logo'>🃏</div>"
            "<div class='brand-text'><div class='brand-title'>カード印刷ツール</div>"
            "<div class='brand-sub'>Card Print Studio</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='nav-label'>メニュー</div>", unsafe_allow_html=True)
        current = st.session_state.get("active_screen", SCREEN_CARDS)
        nav_items = [
            (SCREEN_CARDS, "カード", "📇"),
            (SCREEN_WORKLIST, "作業リスト", "📋"),
            (SCREEN_PRINT, "印刷", "🖨"),
        ]
        for value, label, icon in nav_items:
            is_active = current == value
            if st.button(
                f"{icon} {label}",
                key=f"nav_{value}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["active_screen"] = value
                st.rerun()

        st.markdown("<div class='nav-label'>状況</div>", unsafe_allow_html=True)
        _render_summary_sidebar(pdf_state, total_cards)

        st.markdown("<div class='nav-label'>表示</div>", unsafe_allow_html=True)
        st.toggle("カード画像を表示", value=st.session_state.get("show_thumbs", True), key="show_thumbs")

        with st.expander("📂 ファイル / 保存", expanded=False):
            _render_file_menu(pdf_path, candidate_paths, active_project, labels, pdf_state)

        if active_project:
            dirty = _is_project_dirty(active_project)
            status = "● 未保存の変更" if dirty else "✓ 保存済み"
            st.caption(f"{status} ・ {Path(active_project).parent.name}")
        else:
            st.caption(f"PDF: {Path(pdf_path).name}")
    return st.session_state.get("active_screen", SCREEN_CARDS)


def _status_chips_inline(card_id: int, work_list_card_ids: set[int], print_quantities: dict[int, int]) -> str:
    parts: list[str] = []
    if card_id in work_list_card_ids:
        parts.append("<span class='card-status-chip excluded'>作業リスト</span>")
    if card_id in print_quantities:
        parts.append(f"<span class='card-status-chip print'>印刷×{int(print_quantities[card_id])}</span>")
    return "".join(parts)


def _render_card_screen(
    card_df: pd.DataFrame,
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
) -> None:
    st.markdown("<div class='page-title'>カード一覧</div>", unsafe_allow_html=True)

    # 取り込み結果は rerun を跨いで一度だけトーストで知らせる
    for feedback_key in ("tier_import_result", "label_cleanup_result"):
        pending = st.session_state.pop(feedback_key, None)
        if pending:
            level, message = pending
            st.toast(message, icon={"success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️"))

    with st.expander("🏆 Tier上位デッキの採用カードを取り込む", expanded=False):
        st.caption(
            "pokeka-win-decks.jp のTierランキング上位デッキで採用されているカードを照合し、"
            "作業リストにまとめて追加します。取り込み時に、採用デッキ名ラベルも自動で付きます。"
        )
        ctrl = st.columns([2.0, 1.0, 1.0], gap="medium")
        min_decks = ctrl[0].slider(
            "何デッキ以上で採用されているカードを対象にするか",
            min_value=1,
            max_value=8,
            value=2,
            key=f"tier_min_decks_{fingerprint}",
        )
        ctrl[1].markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        if ctrl[1].button("取り込む", type="primary", use_container_width=True, key=f"tier_import_{fingerprint}"):
            _import_tier_cards(card_df, pdf_path, pdf_state, labels, min_decks)
        ctrl[2].markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        if ctrl[2].button("既存ラベルを整理", use_container_width=True, key=f"label_cleanup_{fingerprint}"):
            _run_label_cleanup(pdf_path, labels, pdf_state, card_df["name"])

    known_labels = sorted(
        {
            str(label).strip()
            for label_list in card_df["label_list"]
            for label in normalize_label_list(label_list)
            if str(label).strip()
        }
    )
    # --- 絞り込み状態を画面移動を跨いで保持する ---
    # Streamlit は「その実行で描画されなかったウィジェット」のキーをセッションから破棄するため、
    # 印刷/作業リストタブへ移動して戻ると絞り込みが初期化されてしまう。
    # 専用の保存領域に退避し、カード画面に戻ったとき（キーが消えていれば）復元する。
    _filter_specs = [
        (f"search_{fingerprint}", None),
        (f"labelfilter_{fingerprint}", ["すべて", *known_labels]),
        (f"deckfilter_{fingerprint}", None),
        (f"sort_{fingerprint}", SORT_OPTIONS),
        (f"pagesize_{fingerprint}", CARD_PAGE_SIZE_OPTIONS),
    ]
    _filter_saved = st.session_state.setdefault("card_filter_state", {})
    for _wk, _valid in _filter_specs:
        if _wk not in st.session_state and _wk in _filter_saved:
            _restored = _filter_saved[_wk]
            if _valid is None or _restored in _valid:
                st.session_state[_wk] = _restored

    with st.container(key="filterbar"):
        top = st.columns([2.4, 1.0, 1.45, 1.0, 0.75], gap="medium")
        search_text = top[0].text_input(
            "検索",
            placeholder="🔍 カード名・ラベル・番号・IDで検索",
            key=f"search_{fingerprint}",
        )
        label_filter = top[1].selectbox(
            "ラベル",
            ["すべて"] + known_labels,
            format_func=_format_label_filter_option,
            key=f"labelfilter_{fingerprint}",
        )
        # ラベル(アーキタイプ)→デッキ(レシピ) の階層で絞り込む。
        # アーキタイプ↔デッキの明示マップ(deck_archetype)があれば、それを使って
        # 「選んだラベルに属するデッキだけ」を候補にする（共有カード経由の混入を防ぐ）。
        deck_archetype = pdf_state.get("deck_archetype", {})
        deck_catalog_now = pdf_state.get("deck_catalog", {})
        if label_filter != "すべて" and deck_archetype:
            target_norm = _normalize_card_name(label_filter)
            known_deck_ids = sorted(
                deck_id
                for deck_id, archetype in deck_archetype.items()
                if str(deck_id).strip()
                and deck_id in deck_catalog_now
                and _normalize_card_name(str(archetype)) == target_norm
            )
        else:
            # フォールバック（全ラベル時 / 旧キャッシュで deck_archetype が無い場合）
            deck_scope_df = _filter_cards(card_df, "", label_filter, "すべて")
            known_deck_ids = sorted(
                {
                    str(deck_id).strip()
                    for deck_list in deck_scope_df["deck_list"]
                    for deck_id in normalize_label_list(deck_list)
                    if str(deck_id).strip()
                }
            )
        deck_filter_key = f"deckfilter_{fingerprint}"
        active_deck_filter = str(st.session_state.get(deck_filter_key, "すべて"))
        if active_deck_filter not in {"すべて", *known_deck_ids}:
            st.session_state[deck_filter_key] = "すべて"
        deck_filter = top[2].selectbox(
            "デッキ番号",
            ["すべて"] + known_deck_ids,
            format_func=lambda value: _format_deck_option(
                value,
                deck_catalog_now,
                strip_prefix=(deck_archetype.get(value, "") if label_filter != "すべて" else ""),
            ),
            key=deck_filter_key,
        )
        sort_choice = top[3].selectbox(
            "並び順", SORT_OPTIONS, format_func=lambda item: item[1], key=f"sort_{fingerprint}"
        )
        page_size = int(
            top[4].selectbox("表示数", CARD_PAGE_SIZE_OPTIONS, key=f"pagesize_{fingerprint}")
        )

        # 現在の絞り込み状態を退避（他タブへ移動して戻ったときに復元するため）
        for _wk, _ in _filter_specs:
            if _wk in st.session_state:
                _filter_saved[_wk] = st.session_state[_wk]

    if pdf_state.get("deck_data_loaded"):
        st.caption(
            "💡 ①「ラベル」でアーキタイプを選ぶと②「デッキ番号」がそのデッキだけに絞られます。"
            "デッキを選ぶと、そのレシピを採用枚数つきで印刷キューへ追加できます。"
        )
    deck_summary_parts: list[str] = []
    if pdf_state.get("deck_data_loaded"):
        deck_summary_parts.append(f"Tier取込デッキ {len(pdf_state.get('deck_catalog', {}))}件")
    if label_filter != "すべて":
        deck_summary_parts.append(f"ラベル内デッキ候補 {len(known_deck_ids)}件")
    if deck_summary_parts:
        st.caption(" / ".join(deck_summary_parts))

    if deck_filter != "すべて":
        deck_kinds, deck_total_cards = _deck_recipe_summary(pdf_state, deck_filter)
        deck_page_url = str(pdf_state.get("deck_url", {}).get(deck_filter, "")).strip()
        deck_cols = st.columns([1.2, 1.4, 1.4], gap="small")
        deck_cols[0].caption(
            f"選択中デッキ: {deck_kinds}種 / {deck_total_cards}枚"
            if deck_kinds
            else "選択中デッキ: 枚数情報なし"
        )
        if deck_page_url:
            deck_cols[1].link_button(
                "🔗 元サイトでこのデッキを開く",
                deck_page_url,
                use_container_width=True,
            )
        else:
            deck_cols[1].caption("（このデッキのURLは未取得。再取り込みで付きます）")
        if deck_cols[2].button(
            "🖨 このデッキを印刷に追加",
            use_container_width=True,
            key=f"add_deck_print_{fingerprint}_{deck_filter}",
            disabled=deck_kinds == 0,
        ):
            added_kinds, added_cards = _add_deck_to_print_queue(pdf_state, deck_filter)
            if added_cards <= 0:
                st.warning("このデッキから印刷に追加できるカードがありませんでした。")
            else:
                _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
                _persist_active_project_state(pdf_path, labels, pdf_state)
                st.session_state["tier_import_result"] = (
                    "success",
                    f"{_format_deck_option(deck_filter, pdf_state.get('deck_catalog', {}))} を印刷キューへ追加しました。"
                    f" {added_kinds}種 / {added_cards}枚 を反映しています。",
                )
                st.rerun()

    filtered_df = _sort_cards(_filter_cards(card_df, search_text, label_filter, deck_filter), sort_choice[0], False)
    visible_df = _order_cards_for_display(filtered_df, pdf_state["work_list_card_ids"])
    all_card_ids = [int(card_id) for card_id in card_df["card_id"]]
    visible_card_ids = [int(card_id) for card_id in filtered_df["card_id"]]
    total_visible = len(visible_df)

    total_pages = max(1, (total_visible + page_size - 1) // page_size)
    page_key = f"cardpage_{fingerprint}"
    st.session_state.setdefault(page_key, 1)
    st.session_state[page_key] = min(max(1, int(st.session_state[page_key])), total_pages)
    page_num = int(st.session_state[page_key])
    start = (page_num - 1) * page_size
    end = start + page_size
    page_df = visible_df.iloc[start:end]

    _render_selection_toolbar(
        pdf_path, fingerprint, pdf_state, labels, all_card_ids, visible_card_ids, known_labels
    )

    page_cols = st.columns([1.0, 3.0, 1.0], gap="small")
    if page_cols[0].button("‹ 前へ", use_container_width=True, disabled=page_num <= 1):
        st.session_state[page_key] = page_num - 1
        st.rerun()
    shown_from = start + 1 if total_visible else 0
    page_cols[1].markdown(
        f"<div class='page-indicator'>{shown_from}–{min(end, total_visible)} / {total_visible}件"
        f" （{page_num}/{total_pages}ページ）</div>",
        unsafe_allow_html=True,
    )
    if page_cols[2].button("次へ ›", use_container_width=True, disabled=page_num >= total_pages):
        st.session_state[page_key] = page_num + 1
        st.rerun()

    if page_df.empty:
        st.info("該当するカードがありません。検索条件を変えてみてください。")
        return

    show_thumbs = bool(st.session_state.get("show_thumbs", True))
    if show_thumbs:
        # 現ページ分のサムネイルを1回の PDF オープンでまとめて生成（=キャッシュ温め）し、
        # 個々の <img> リクエストはキャッシュヒットさせて高速・低負荷にする。
        try:
            with st.spinner("カード画像を準備中..."):
                get_card_preview_data_map(
                    pdf_path, [int(page) for page in page_df["target_page"]], max_width=THUMB_WIDTH
                )
        except Exception:
            pass

    _mount_hover_preview_layer()
    version_token = _pdf_version_token(pdf_path)
    rows = list(page_df.itertuples(index=False))
    # カードグリッドを独立したスクロール領域に閉じ込める。
    # これによりフィルターバー・ツールバー等はページ上部に常時表示される。
    # CSS（.st-key-cardscroll）で height を calc(100vh - Npx) に上書きする。
    with st.container(height=600, key="cardscroll", border=False):
        # 全タイルを1つの横ブロックに入れ、CSS（.st-key-cardgrid）で flex-wrap させる。
        with st.container(key="cardgrid"):
            grid_cols = st.columns(len(rows), gap="medium")
            for offset, row in enumerate(rows):
                _render_card_tile(
                    grid_cols[offset], row, pdf_path, fingerprint, pdf_state, version_token, show_thumbs
                )


def _render_card_tile(
    col,
    row,
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    version_token: str,
    show_thumbs: bool,
) -> None:
    card_id = int(row.card_id)
    in_work_list = card_id in pdf_state["work_list_card_ids"]
    selected = card_id in pdf_state["work_selected_card_ids"]
    with col:
        if show_thumbs:
            thumb_url = build_preview_url(
                pdf_path, int(row.target_page), max_width=THUMB_WIDTH, version_token=version_token
            )
            img_html = f"<div class='tile-img'><img src='{thumb_url}' loading='lazy' alt=''></div>"
        else:
            img_html = "<div class='tile-img tile-img-empty'>🃏</div>"
        anchor = _build_name_anchor_html(
            card_id=card_id,
            title=str(row.name),
            expansion=str(row.expansion),
            collection_no=str(row.collection_no),
            target_page=int(row.target_page),
            image_url=build_preview_url(
                pdf_path, int(row.target_page), max_width=PREVIEW_MAX_WIDTH, version_token=version_token
            ),
            is_excluded=in_work_list,
        )
        label_pill = (
            f"<span class='card-label-pill'>{escape(str(row.label))}</span>"
            if str(row.label).strip()
            else ""
        )
        # デッキ名の羅列は長くなりすぎるため、採用デッキ数だけをコンパクトに表示する
        # （どのアーキタイプかは上のラベル、具体レシピは「デッキ番号」絞り込みで辿れる）
        deck_count = len(normalize_label_list(getattr(row, "deck_list", [])))
        deck_meta = f"<span class='row-dim'>{deck_count}デッキ採用</span>" if deck_count else ""
        chips = _status_chips_inline(card_id, pdf_state["work_list_card_ids"], pdf_state["print_quantities"])
        tile_cls = "tile"
        if selected:
            tile_cls += " selected"
        if in_work_list:
            tile_cls += " wl"
        st.markdown(
            f"<div class='{tile_cls}'>{img_html}"
            f"<div class='tile-name'>{anchor}</div>"
            f"<div class='tile-sub'>{label_pill}{chips}</div>"
            f"<div class='tile-meta'>{escape(str(row.expansion))} · {escape(str(row.collection_no))} · p{int(row.target_page)}"
            f"{' · ' + deck_meta if deck_meta else ''}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button(
            "✓ 選択中" if selected else "＋ 選択",
            key=f"tsel_{fingerprint}_{card_id}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            if selected:
                pdf_state["work_selected_card_ids"].discard(card_id)
            else:
                pdf_state["work_selected_card_ids"].add(card_id)
            st.rerun()


def _render_selection_toolbar(
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
    all_card_ids: list[int],
    visible_card_ids: list[int],
    known_labels: list[str],
) -> None:
    selected = pdf_state["work_selected_card_ids"]
    count = len(selected)
    cols = st.columns([1.2, 1.1, 1.1, 1.0, 0.8], gap="small")
    cols[0].markdown(
        f"<div class='sel-bar-label'>選択中 <b>{count}</b> 件</div>", unsafe_allow_html=True
    )

    with cols[1].popover("🏷 ラベル", use_container_width=True, disabled=count == 0):
        st.caption(f"{count}件のカードにラベルを付けます。")
        existing = st.selectbox(
            "既存ラベル",
            [""] + known_labels,
            format_func=lambda value: value or "（未選択）",
            key=f"lbl_exist_{fingerprint}",
        )
        new_label = st.text_input("新しいラベル", key=f"lbl_new_{fingerprint}")
        if st.button("付ける", type="primary", use_container_width=True, key=f"lbl_apply_{fingerprint}"):
            value = new_label.strip() or existing.strip()
            if not value:
                st.warning("ラベルを選ぶか入力してください。")
            else:
                _add_label_to_cards(labels, selected, value)
                _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
                st.rerun()
        selected_labels = sorted(
            {
                label
                for card_id in selected
                for label in normalize_label_list(labels.get(int(card_id)))
            }
        )
        remove_labels = st.multiselect(
            "外したいラベル",
            selected_labels,
            key=f"lbl_remove_pick_{fingerprint}",
            placeholder="選択カードに付いているラベルだけ表示します",
        )
        if st.button("選んだラベルだけ外す", use_container_width=True, key=f"lbl_remove_some_{fingerprint}"):
            if not remove_labels:
                st.warning("外したいラベルを選択してください。")
            else:
                _remove_specific_labels_from_cards(labels, selected, remove_labels)
                _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
                st.rerun()
        if st.button("ラベルをすべて外す", use_container_width=True, key=f"lbl_clear_{fingerprint}"):
            for card_id in list(selected):
                labels.pop(card_id, None)
            _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
            st.rerun()

    with cols[2].popover("➕ 追加", use_container_width=True, disabled=count == 0):
        if st.button("📋 作業リストに追加", use_container_width=True, key=f"add_wl_{fingerprint}"):
            pdf_state["work_list_card_ids"].update(selected)
            _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
            st.rerun()
        if st.button("🖨 印刷に追加", use_container_width=True, key=f"add_pr_{fingerprint}"):
            pdf_state["work_list_card_ids"].update(selected)
            _add_to_print_queue(pdf_state, sorted(selected))
            _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
            _persist_active_project_state(pdf_path, labels, pdf_state)
            st.rerun()
        st.divider()
        if st.button("作業リストから外す", use_container_width=True, key=f"rm_wl_{fingerprint}"):
            pdf_state["work_list_card_ids"].difference_update(selected)
            _sync_print_queue_to_work_list(pdf_state)
            _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
            st.rerun()

    with cols[3].popover("☑ まとめて選択", use_container_width=True):
        if st.button(
            "表示中をすべて選択", use_container_width=True, key=f"selall_{fingerprint}", disabled=not visible_card_ids
        ):
            _set_work_selection(all_card_ids, fingerprint, pdf_state, visible_card_ids, mode="add")
            st.rerun()
        if st.button(
            "表示中の選択を解除", use_container_width=True, key=f"unselvis_{fingerprint}", disabled=not visible_card_ids
        ):
            _set_work_selection(all_card_ids, fingerprint, pdf_state, visible_card_ids, mode="remove")
            st.rerun()
        if st.button(
            "作業リストを選択",
            use_container_width=True,
            key=f"selwl_{fingerprint}",
            disabled=not pdf_state["work_list_card_ids"],
        ):
            _set_work_selection(
                all_card_ids, fingerprint, pdf_state, sorted(pdf_state["work_list_card_ids"]), mode="replace"
            )
            st.rerun()

    if cols[4].button("選択クリア", use_container_width=True, key=f"clearsel_{fingerprint}", disabled=count == 0):
        pdf_state["work_selected_card_ids"].clear()
        _clear_work_selection_widgets(all_card_ids, fingerprint)
        st.rerun()


def _render_worklist_tile(
    col,
    row,
    card_id: int,
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
    version_token: str,
) -> None:
    with col:
        thumb_url = build_preview_url(
            pdf_path, int(row["target_page"]), max_width=THUMB_WIDTH, version_token=version_token
        )
        img_html = f"<div class='tile-img'><img src='{thumb_url}' loading='lazy' alt=''></div>"
        anchor = _build_name_anchor_html(
            card_id=card_id,
            title=str(row["name"]),
            expansion=str(row["expansion"]),
            collection_no=str(row["collection_no"]),
            target_page=int(row["target_page"]),
            image_url=build_preview_url(
                pdf_path, int(row["target_page"]), max_width=PREVIEW_MAX_WIDTH, version_token=version_token
            ),
            is_excluded=False,
        )
        label_pill = (
            f"<span class='card-label-pill'>{escape(str(row['label']))}</span>"
            if str(row["label"]).strip()
            else ""
        )
        in_queue_html = (
            f"<span class='card-status-chip print'>印刷×{int(pdf_state['print_quantities'][card_id])}</span>"
            if card_id in pdf_state["print_quantities"]
            else ""
        )
        st.markdown(
            f"<div class='tile wl'>{img_html}"
            f"<div class='tile-name'>{anchor}</div>"
            f"<div class='tile-sub'>{label_pill}{in_queue_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("外す", key=f"wl_rm_{fingerprint}_{card_id}", use_container_width=True):
            pdf_state["work_list_card_ids"].discard(card_id)
            _sync_print_queue_to_work_list(pdf_state)
            _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
            st.rerun()


def _render_worklist_screen(
    card_df: pd.DataFrame,
    catalog,
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
    filtered_output_key: str,
) -> None:
    _render_exclude_feedback()
    pending = st.session_state.pop("label_cleanup_result", None)
    if pending:
        level, message = pending
        st.toast(message, icon={"success": "✅", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️"))
    st.markdown("<div class='page-title'>作業リスト</div>", unsafe_allow_html=True)
    work_list = sorted(pdf_state["work_list_card_ids"])
    left, right = st.columns([1.7, 1.0], gap="large")

    with left:
        st.markdown(
            f"<div class='section-title'>カード <span class='muted'>{len(work_list)}件</span></div>",
            unsafe_allow_html=True,
        )
        if not work_list:
            st.info("作業リストは空です。「カード」画面でカードを選び、🏷 や ➕ から追加できます。")
        else:
            queue_df = card_df.set_index("card_id")
            visible_wl = [cid for cid in work_list if cid in queue_df.index]
            try:
                wl_pages = [int(queue_df.loc[cid]["target_page"]) for cid in visible_wl]
                with st.spinner("カード画像を準備中..."):
                    get_card_preview_data_map(pdf_path, wl_pages, max_width=THUMB_WIDTH)
            except Exception:
                pass
            _mount_hover_preview_layer()
            version_token = _pdf_version_token(pdf_path)
            with st.container(height=600, key="wl_cardscroll", border=False):
                with st.container(key="wl_cardgrid"):
                    grid_cols = st.columns(max(len(visible_wl), 1), gap="medium")
                    for offset, card_id in enumerate(visible_wl):
                        _render_worklist_tile(
                            grid_cols[offset],
                            queue_df.loc[card_id],
                            card_id,
                            pdf_path,
                            fingerprint,
                            pdf_state,
                            labels,
                            version_token,
                        )
            if st.button("作業リストを空にする", use_container_width=True):
                pdf_state["work_list_card_ids"].clear()
                _sync_print_queue_to_work_list(pdf_state)
                _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("<div class='section-title'>ラベル整理</div>", unsafe_allow_html=True)
            st.caption("古い長い Tier ラベルを、カード一覧に存在するカード名だけへ整理します。")
            if st.button("既存ラベルを整理", use_container_width=True, key=f"label_cleanup_worklist_{fingerprint}"):
                _run_label_cleanup(pdf_path, labels, pdf_state, card_df["name"])

        with st.container(border=True):
            st.markdown("<div class='section-title'>PDFを書き出す</div>", unsafe_allow_html=True)
            st.caption(f"作業リスト {len(work_list)} 件 / 元 {catalog.total_pages} ページ")
            with st.expander("保存先を変更", expanded=False):
                st.text_input("保存先", key=filtered_output_key)
                st.caption("フォルダだけを入れた場合は、その場所に既定ファイル名で保存します。")
            out_path = resolve_output_pdf_path(
                st.session_state.get(filtered_output_key, ""),
                default_output_path(pdf_path),
            )
            st.caption(f"出力先: {out_path}")
            if st.button("📄 作業リストPDFを書き出す", type="primary", use_container_width=True):
                if not work_list:
                    st.error("作業リストが空のため、書き出せるカードがありません。")
                else:
                    request = DeleteRequest(
                        card_ids=frozenset(
                            int(card.card_id)
                            for card in catalog.cards
                            if int(card.card_id) not in pdf_state["work_list_card_ids"]
                        ),
                        page_numbers=frozenset(),
                    )
                    try:
                        with st.spinner("PDFを作成中..."):
                            result = build_filtered_pdf(pdf_path, out_path, request)
                    except Exception as exc:
                        st.error(f"書き出しに失敗しました: {exc}")
                    else:
                        _render_filtered_result(result)

        with st.container(border=True):
            st.markdown("<div class='section-title'>ページ番号で追加</div>", unsafe_allow_html=True)
            page_text = st.text_area(
                "ページ番号", key=f"page_add_{fingerprint}", placeholder="例: 40, 55-60, 1306", height=80
            )
            if st.button("ページ番号を追加", use_container_width=True):
                parsed_pages, invalid_tokens = parse_page_number_text(page_text)
                if invalid_tokens:
                    _push_exclude_feedback("warning", f"解釈できなかった入力: {', '.join(invalid_tokens)}")
                resolved_ids, resolved_pages, ignored_pages = _resolve_card_ids_from_page_numbers(catalog, parsed_pages)
                if resolved_ids:
                    pdf_state["work_list_card_ids"].update(resolved_ids)
                    _save_card_state(pdf_path, labels, pdf_state["work_list_card_ids"])
                    _push_exclude_feedback("success", "追加: " + ", ".join(str(page) for page in resolved_pages))
                if ignored_pages:
                    _push_exclude_feedback("info", "使えなかったページ: " + ", ".join(str(page) for page in ignored_pages))
                st.rerun()


def _render_print_screen(
    card_df: pd.DataFrame,
    pdf_path: str,
    fingerprint: str,
    pdf_state: dict,
    labels: dict[int, list[str]],
    print_output_key: str,
) -> None:
    _sync_print_queue_to_work_list(pdf_state)
    st.markdown("<div class='page-title'>印刷</div>", unsafe_allow_html=True)
    order = list(pdf_state["print_order"])
    sheets = sum(int(v) for v in pdf_state["print_quantities"].values())
    left, right = st.columns([1.7, 1.0], gap="large")

    with left:
        st.markdown(
            f"<div class='section-title'>印刷キュー <span class='muted'>{len(order)}種 / {sheets}枚</span></div>",
            unsafe_allow_html=True,
        )
        if not order:
            st.info("印刷キューは空です。「カード」画面で作業リストのカードを選び、➕ →「印刷に追加」できます。")
        else:
            tool_cols = st.columns(2, gap="small")
            if tool_cols[0].button("ラベル順に並べ替え", use_container_width=True):
                queue_info = card_df.set_index("card_id")[["name"]].to_dict("index")
                pdf_state["print_order"].sort(
                    key=lambda card_id: (
                        label_text(labels.get(card_id)).casefold() if labels.get(card_id) else "￿",
                        str(queue_info.get(card_id, {}).get("name", "")).casefold(),
                        card_id,
                    )
                )
                _persist_active_project_state(pdf_path, labels, pdf_state)
                st.rerun()
            if tool_cols[1].button("キューを空にする", use_container_width=True):
                pdf_state["print_excluded_card_ids"].update(pdf_state["work_list_card_ids"])
                pdf_state["print_quantities"].clear()
                pdf_state["print_order"].clear()
                _persist_active_project_state(pdf_path, labels, pdf_state)
                st.rerun()

            queue_df = card_df.set_index("card_id")
            dirty = False
            with st.container(height=520, border=False):
                for card_id in list(pdf_state["print_order"]):
                    if card_id not in pdf_state["print_quantities"] or card_id not in queue_df.index:
                        continue
                    row = queue_df.loc[card_id]
                    item_cols = st.columns([0.6, 0.27, 0.13], gap="small")
                    label_pill = (
                        f"<span class='card-label-pill'>{escape(str(row['label']))}</span>"
                        if str(row["label"]).strip()
                        else ""
                    )
                    item_cols[0].markdown(
                        f"<div class='row-card'><div class='row-name'>{escape(str(row['name']))}</div>"
                        f"<div class='row-sub'>{label_pill}<span class='row-dim'>ID {card_id}</span></div></div>",
                        unsafe_allow_html=True,
                    )
                    quantity = item_cols[1].number_input(
                        "枚数",
                        min_value=1,
                        step=1,
                        value=int(pdf_state["print_quantities"].get(card_id, 1)),
                        key=f"pq_{fingerprint}_{card_id}",
                        label_visibility="collapsed",
                    )
                    if int(quantity) != int(pdf_state["print_quantities"].get(card_id, 1)):
                        dirty = True
                    pdf_state["print_quantities"][card_id] = int(quantity)
                    if item_cols[2].button("✕", key=f"pqrm_{fingerprint}_{card_id}", use_container_width=True):
                        pdf_state["print_excluded_card_ids"].add(card_id)
                        _remove_from_print_queue(pdf_state, {card_id})
                        _persist_active_project_state(pdf_path, labels, pdf_state)
                        st.rerun()
            if dirty:
                _persist_active_project_state(pdf_path, labels, pdf_state)

    with right:
        with st.container(border=True):
            st.markdown("<div class='section-title'>印刷用PDFを書き出す</div>", unsafe_allow_html=True)
            with st.expander("詳細設定（サイズ・余白）", expanded=False):
                st.caption("既定は 63×88mm・余白2mm・A4最大配置です。")
                print_settings = _render_print_settings(fingerprint)
            with st.expander("保存先を変更", expanded=False):
                st.text_input("保存先", key=print_output_key)
                st.caption("フォルダだけを入れた場合は、その場所に既定ファイル名で保存します。")
            out_path = resolve_output_pdf_path(
                st.session_state.get(print_output_key, ""),
                default_print_output_path(pdf_path),
            )
            st.caption(f"出力先: {out_path}")
            if st.button("🖨 印刷用PDFを書き出す", type="primary", use_container_width=True):
                try:
                    print_items = _build_print_items(
                        pdf_state["print_order"], pdf_state["print_quantities"], labels
                    )
                    if not print_items:
                        raise ValueError("印刷キューにカードがありません。")
                    with st.spinner("印刷用PDFを作成中..."):
                        result = build_print_pdf(pdf_path, out_path, print_items, print_settings)
                except Exception as exc:
                    st.error(f"書き出しに失敗しました: {exc}")
                else:
                    _render_print_result(result)


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --accent: #6366f1;
          --accent-weak: #eef0ff;
          --ink: #1f2430;
          --ink-soft: #6b7280;
          --ink-faint: #9aa1ad;
          --surface: #ffffff;
          --bg: #f6f7fb;
          --line: #ececf3;
          --green: #2e9e6b;
          --green-weak: #e7f6ee;
          --amber: #b9791b;
        }
        html, body, [class*="css"], .stMarkdown, .stButton, input, textarea, select {
          font-family: -apple-system, "Segoe UI", Roboto, "Noto Sans JP", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;
        }
        .stApp { background: var(--bg); }
        .block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1500px; }

        /* ---- サイドバー ---- */
        [data-testid="stSidebar"] {
          background: var(--surface);
          border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
        .brand { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1.1rem; }
        .brand-logo {
          width: 42px; height: 42px; border-radius: 13px; display: flex; align-items: center;
          justify-content: center; font-size: 1.35rem;
          background: linear-gradient(140deg, #6366f1, #8b5cf6);
          box-shadow: 0 8px 18px rgba(99,102,241,0.35);
        }
        .brand-title { font-size: 1.05rem; font-weight: 800; color: var(--ink); line-height: 1.15; }
        .brand-sub { font-size: 0.72rem; color: var(--ink-faint); letter-spacing: 0.04em; }
        .nav-label {
          font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
          color: var(--ink-faint); margin: 1.1rem 0 0.4rem;
        }
        .side-stats { display: flex; flex-direction: column; gap: 0.45rem; }
        .side-stat {
          display: flex; align-items: baseline; justify-content: space-between;
          padding: 0.6rem 0.85rem; border-radius: 13px; background: var(--bg); border: 1px solid var(--line);
        }
        .side-stat .ss-num { font-size: 1.25rem; font-weight: 800; color: var(--accent); }
        .side-stat .ss-lbl { font-size: 0.8rem; color: var(--ink-soft); }

        /* ---- 見出し ---- */
        .page-title { font-size: 1.6rem; font-weight: 800; color: var(--ink); margin: 0 0 1rem; letter-spacing: 0.01em; }
        .section-title { font-size: 1.02rem; font-weight: 800; color: var(--ink); margin: 0.1rem 0 0.6rem; }
        .section-title .muted { font-size: 0.82rem; font-weight: 700; color: var(--ink-faint); margin-left: 0.4rem; }
        .sel-bar-label {
          display: flex; align-items: center; min-height: 40px;
          font-size: 0.92rem; color: var(--ink-soft);
        }
        .sel-bar-label b { color: var(--accent); font-size: 1.1rem; margin: 0 0.15rem; }
        .page-indicator {
          text-align: center; font-size: 0.85rem; color: var(--ink-soft);
          min-height: 40px; display: flex; align-items: center; justify-content: center;
        }

        /* ---- カードグリッドのタイル ---- */
        .tile {
          border: 1px solid var(--line); border-radius: 16px; background: var(--surface);
          padding: 10px 10px 8px; box-shadow: 0 2px 6px rgba(31,36,48,0.04);
          transition: box-shadow 160ms ease, border-color 160ms ease, transform 160ms ease;
        }
        .tile:hover { box-shadow: 0 12px 26px rgba(31,36,48,0.12); transform: translateY(-2px); }
        .tile.selected { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-weak); }
        .tile.wl { background: linear-gradient(180deg, #ffffff, #fbfbff); }
        .tile-img {
          width: 100%; aspect-ratio: 63 / 88; border-radius: 11px; overflow: hidden;
          background: #f1f2f7; display: flex; align-items: center; justify-content: center;
          margin-bottom: 8px;
        }
        .tile-img img { width: 100%; height: 100%; object-fit: contain; display: block; }
        .tile-img-empty { font-size: 2rem; color: #c7cad6; }
        .tile-name { font-size: 0.92rem; font-weight: 700; line-height: 1.3; min-height: 2.4em; }
        .tile-sub { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; min-height: 22px; }
        .tile-meta { font-size: 0.7rem; color: var(--ink-faint); margin-top: 4px; }
        .card-hover-anchor { color: var(--ink); font-weight: 700; cursor: pointer; border-radius: 6px; }
        .card-hover-anchor:hover, .card-hover-anchor:focus { color: var(--accent); outline: none; }
        .card-hover-anchor.excluded { color: var(--ink-soft); }

        /* ---- リスト行（作業リスト / 印刷キュー）---- */
        .row-card {
          padding: 0.55rem 0.75rem; border: 1px solid var(--line); border-radius: 13px;
          background: var(--surface); margin-bottom: 2px;
        }
        .row-name { font-size: 0.98rem; font-weight: 700; color: var(--ink); line-height: 1.3; }
        .row-sub { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 3px; }
        .row-dim { font-size: 0.74rem; color: var(--ink-faint); }

        /* ---- チップ類 ---- */
        .card-label-pill {
          display: inline-block; background: var(--accent-weak); color: var(--accent);
          padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
        }
        .card-status-chip { display: inline-block; border-radius: 999px; padding: 0.08rem 0.46rem; font-size: 0.7rem; font-weight: 700; }
        .card-status-chip.excluded { background: #eef1f6; color: #5b6472; }
        .card-status-chip.print { background: var(--green-weak); color: var(--green); }

        /* ---- 入力・ボタン・コンテナをモダンに ---- */
        .stButton button { border-radius: 11px; font-weight: 600; }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
          border-radius: 11px !important;
        }
        [data-testid="stExpander"] { border-radius: 13px; border: 1px solid var(--line); }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 16px;
        }
        [data-testid="stPopoverBody"] { border-radius: 14px; }

        /* ---- レスポンシブなカードグリッド ----
           タイルを入れた横ブロックを折り返し可能にし、各タイルに最小幅を与えることで
           ウィンドウ幅に応じて 1 行あたりの枚数が自動で変わる（広い画面ほど多く並ぶ）。
           wl_cardgrid（作業リスト）も同じルールを適用する。 */
        .st-key-cardgrid [data-testid="stHorizontalBlock"],
        .st-key-wl_cardgrid [data-testid="stHorizontalBlock"] {
          flex-wrap: wrap;
          row-gap: 1.1rem;
        }
        .st-key-cardgrid [data-testid="stColumn"],
        .st-key-wl_cardgrid [data-testid="stColumn"] {
          flex: 1 1 168px !important;
          min-width: 168px !important;
          max-width: 240px !important;
          width: auto !important;
        }
        @media (max-width: 640px) {
          .st-key-cardgrid [data-testid="stColumn"],
          .st-key-wl_cardgrid [data-testid="stColumn"] {
            flex-basis: 130px !important;
            min-width: 130px !important;
          }
        }

        /* ---- カードグリッド専用スクロール領域 ----
           カードグリッド以外のUIはページ内に収まり、カードエリアだけが
           画面余白を埋める可変高さのスクロールボックスになる。
           Streamlit が setする height:Npx の inline style を上書きする。 */
        .st-key-cardscroll > [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-wl_cardscroll > [data-testid="stVerticalBlockBorderWrapper"] {
          height: calc(100vh - 530px) !important;
          min-height: 280px !important;
          overflow-y: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_name_anchor_html(
    *,
    card_id: int,
    title: str,
    expansion: str,
    collection_no: str,
    target_page: int,
    image_url: str,
    is_excluded: bool,
) -> str:
    dataset = {
        "card-id": str(card_id),
        "title": title,
        "meta": f"{expansion} / {collection_no} / page {target_page}",
        "image": image_url,
    }
    attrs = " ".join(
        f'data-{key}="{escape(value, quote=True)}"'
        for key, value in dataset.items()
    )
    anchor_classes = "card-hover-anchor excluded" if is_excluded else "card-hover-anchor"
    return f'<span class="{anchor_classes}" tabindex="0" {attrs}>{escape(title)}</span>'


def _mount_hover_preview_layer() -> None:
    config = {
        "panelWidth": 360,
        "panelTop": 84,
        "panelRight": 28,
    }
    components.html(
        f"""
        <script>
        (() => {{
          const config = {json.dumps(config)};
          const parentDoc = window.parent.document;
          const parentWin = window.parent;

          function ensureOverlay() {{
            if (parentDoc.getElementById("card-hover-preview-root")) {{
              return;
            }}

            const style = parentDoc.createElement("style");
            style.id = "card-hover-preview-style";
            style.textContent = `
              #card-hover-preview-scrim {{
                position: fixed;
                inset: 0;
                background: rgba(34, 26, 17, 0.14);
                backdrop-filter: blur(3px) saturate(0.92);
                opacity: 0;
                pointer-events: none;
                transition: opacity 160ms ease;
                z-index: 9997;
              }}
              #card-hover-preview-panel {{
                position: fixed;
                top: ${{config.panelTop}}px;
                right: ${{config.panelRight}}px;
                width: ${{config.panelWidth}}px;
                max-width: min(38vw, ${{config.panelWidth}}px);
                border-radius: 24px;
                background:
                  linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,241,228,0.98));
                box-shadow: 0 28px 68px rgba(18, 12, 7, 0.28);
                border: 1px solid rgba(131, 108, 70, 0.18);
                opacity: 0;
                transform: translateY(-10px) scale(0.985);
                transition: opacity 180ms ease, transform 220ms ease;
                pointer-events: none;
                z-index: 9998;
                overflow: hidden;
              }}
              #card-hover-preview-panel.visible {{
                opacity: 1;
                transform: translateY(0) scale(1);
              }}
              #card-hover-preview-panel.pinned {{
                pointer-events: auto;
              }}
              #card-hover-preview-scrim.visible {{
                opacity: 1;
              }}
              .card-hover-preview-inner {{
                padding: 14px 14px 16px;
              }}
              .card-hover-preview-top {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                margin-bottom: 12px;
              }}
              .card-hover-preview-title {{
                font: 800 16px/1.3 "Yu Gothic UI","Hiragino Sans",sans-serif;
                color: #2e2418;
              }}
              .card-hover-preview-meta {{
                font: 500 12px/1.5 "Yu Gothic UI","Hiragino Sans",sans-serif;
                color: #6f624f;
                margin-top: 3px;
              }}
              .card-hover-preview-close {{
                appearance: none;
                border: 0;
                width: 34px;
                height: 34px;
                border-radius: 999px;
                background: rgba(233, 223, 202, 0.95);
                color: #5b4a33;
                font-size: 19px;
                font-weight: 700;
                cursor: pointer;
                display: none;
              }}
              #card-hover-preview-panel.pinned .card-hover-preview-close {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
              }}
              .card-hover-preview-frame {{
                background: #fff;
                border-radius: 18px;
                padding: 12px;
                box-shadow: inset 0 0 0 1px rgba(160, 141, 108, 0.18);
                min-height: 220px;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
              }}
              .card-hover-preview-frame.loading::before {{
                content: "読み込み中...";
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font: 600 13px/1 "Yu Gothic UI","Hiragino Sans",sans-serif;
                color: #6f624f;
                letter-spacing: 0.03em;
              }}
              .card-hover-preview-frame.error::before {{
                content: "プレビューを読み込めませんでした";
                position: absolute;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                padding: 24px;
                font: 600 13px/1.5 "Yu Gothic UI","Hiragino Sans",sans-serif;
                color: #8a4c35;
              }}
              .card-hover-preview-image {{
                width: 100%;
                display: block;
                border-radius: 12px;
                opacity: 1;
                transition: opacity 120ms ease;
              }}
              .card-hover-preview-frame.loading .card-hover-preview-image,
              .card-hover-preview-frame.error .card-hover-preview-image {{
                opacity: 0;
              }}
            `;
            parentDoc.head.appendChild(style);

            const scrim = parentDoc.createElement("div");
            scrim.id = "card-hover-preview-scrim";
            const panel = parentDoc.createElement("div");
            panel.id = "card-hover-preview-panel";
            panel.innerHTML = `
              <div class="card-hover-preview-inner">
                <div class="card-hover-preview-top">
                  <div>
                    <div id="card-hover-preview-title" class="card-hover-preview-title"></div>
                    <div id="card-hover-preview-meta" class="card-hover-preview-meta"></div>
                  </div>
                  <button id="card-hover-preview-close" type="button" class="card-hover-preview-close">×</button>
                </div>
                <div class="card-hover-preview-frame">
                  <img id="card-hover-preview-image" class="card-hover-preview-image" alt="card preview" />
                </div>
              </div>
            `;

            const root = parentDoc.createElement("div");
            root.id = "card-hover-preview-root";
            root.appendChild(scrim);
            root.appendChild(panel);
            parentDoc.body.appendChild(root);

            parentWin.__cardHoverPreviewState = {{
              scrim,
              panel,
              closeButton: panel.querySelector("#card-hover-preview-close"),
              titleEl: panel.querySelector("#card-hover-preview-title"),
              metaEl: panel.querySelector("#card-hover-preview-meta"),
              frameEl: panel.querySelector(".card-hover-preview-frame"),
              imageEl: panel.querySelector("#card-hover-preview-image"),
              pinned: false,
            }};

            scrim.addEventListener("click", () => closePreview(true));
            parentWin.__cardHoverPreviewState.closeButton.addEventListener("click", () => closePreview(true));
            parentDoc.addEventListener("keydown", (event) => {{
              if (event.key === "Escape") {{
                closePreview(true);
              }}
            }});
          }}

          function openPreview(anchor, pinned) {{
            const state = parentWin.__cardHoverPreviewState;
            if (!state) return;
            state.pinned = pinned;
            state.titleEl.textContent = anchor.dataset.title || "";
            state.metaEl.textContent = anchor.dataset.meta || "";
            state.frameEl.classList.remove("error");
            state.frameEl.classList.add("loading");
            state.imageEl.src = "";
            state.imageEl.onerror = () => {{
              state.frameEl.classList.remove("loading");
              state.frameEl.classList.add("error");
            }};
            state.imageEl.onload = () => {{
              state.frameEl.classList.remove("loading", "error");
            }};
            if (anchor.dataset.image) {{
              state.imageEl.src = anchor.dataset.image;
            }} else {{
              state.frameEl.classList.remove("loading");
              state.frameEl.classList.add("error");
            }}
            state.panel.classList.add("visible");
            state.panel.classList.toggle("pinned", pinned);
            if (pinned) {{
              state.scrim.classList.add("visible");
            }} else {{
              state.scrim.classList.remove("visible");
            }}
            state.scrim.style.pointerEvents = pinned ? "auto" : "none";
            state.panel.style.pointerEvents = pinned ? "auto" : "none";
          }}

          function closePreview(force) {{
            const state = parentWin.__cardHoverPreviewState;
            if (!state) return;
            if (state.pinned && !force) return;
            state.pinned = false;
            state.panel.classList.remove("visible", "pinned");
            state.scrim.classList.remove("visible");
            state.panel.style.pointerEvents = "none";
            state.scrim.style.pointerEvents = "none";
            state.frameEl.classList.remove("loading", "error");
            state.imageEl.onload = null;
            state.imageEl.onerror = null;
            state.imageEl.removeAttribute("src");
            state.titleEl.textContent = "";
            state.metaEl.textContent = "";
          }}

          function bindAnchors() {{
            parentDoc.querySelectorAll(".card-hover-anchor").forEach((anchor) => {{
              if (anchor.dataset.previewBound === "1") return;
              anchor.dataset.previewBound = "1";
              anchor.addEventListener("mouseenter", () => openPreview(anchor, false));
              anchor.addEventListener("focus", () => openPreview(anchor, false));
              anchor.addEventListener("mouseleave", () => {{
                const state = parentWin.__cardHoverPreviewState;
                if (state && !state.pinned) {{
                  closePreview(true);
                }}
              }});
              anchor.addEventListener("blur", () => {{
                const state = parentWin.__cardHoverPreviewState;
                if (state && !state.pinned) {{
                  closePreview(true);
                }}
              }});
              anchor.addEventListener("click", (event) => {{
                event.preventDefault();
                openPreview(anchor, true);
              }});
            }});
          }}

          ensureOverlay();
          bindAnchors();

          if (!parentWin.__cardHoverPreviewObserver) {{
            parentWin.__cardHoverPreviewObserver = new MutationObserver(() => bindAnchors());
            parentWin.__cardHoverPreviewObserver.observe(parentDoc.body, {{ childList: true, subtree: true }});
          }}
        }})();
        </script>
        """,
        height=0,
        width=0,
    )


def _render_filtered_result(result: BuildResult) -> None:
    st.success(
        f"作業リストPDFの書き出し完了: {result.output_path} "
        f"({result.total_input_pages} -> {result.total_output_pages} ページ)"
    )
    st.write(
        f"作業リスト外 {result.removed_cards} 件 / 書き出しカード {result.kept_cards} 件 / "
        f"再生成した一覧 {result.regenerated_index_pages} ページ"
    )
    if result.ignored_page_numbers:
        st.info(
            "一覧ページや範囲外ページは書き出し対象に含めませんでした: "
            + ", ".join(str(page) for page in result.ignored_page_numbers)
        )
    if Path(result.output_path).exists():
        with Path(result.output_path).open("rb") as output_file:
            st.download_button(
                "生成したPDFをダウンロード",
                data=output_file.read(),
                file_name=Path(result.output_path).name,
                mime="application/pdf",
            )
        st.link_button("生成したPDFを開く", build_artifact_url(result.output_path), use_container_width=True)


def _render_print_result(result: PrintBuildResult) -> None:
    st.success(
        f"印刷用PDFの書き出し完了: {result.output_path} "
        f"({result.total_cards_placed} 枚 / {result.total_output_pages} ページ)"
    )
    st.write(
        f"要求枚数 {result.total_cards_requested} 枚 / 配置 {result.total_cards_placed} 枚 / "
        f"1ページあたり {result.cards_per_sheet_used} 枚"
    )
    if result.skipped_card_ids:
        st.warning(
            "画像抽出できず印刷対象に入れられなかったカードID: "
            + ", ".join(str(card_id) for card_id in result.skipped_card_ids)
        )
    if result.skipped_pages:
        st.info("印刷対象に入れられなかったページ: " + ", ".join(str(page) for page in result.skipped_pages))
    if Path(result.output_path).exists():
        with Path(result.output_path).open("rb") as output_file:
            st.download_button(
                "印刷用PDFをダウンロード",
                data=output_file.read(),
                file_name=Path(result.output_path).name,
                mime="application/pdf",
            )
        st.link_button("印刷用PDFを開く / 印刷", build_artifact_url(result.output_path), use_container_width=True)


if __name__ == "__main__":
    main()
