from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from pypdf import PdfReader

from pdf_card_tool import (
    DeleteRequest,
    PrintItem,
    PrintLayoutSettings,
    build_filtered_pdf,
    build_print_pdf,
    get_card_asset_map,
    load_card_state_store,
    load_label_store,
    load_pdf_catalog,
    save_card_state_store,
    save_label_store,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PDF_CANDIDATES = (
    REPO_ROOT / "data" / "Card_ID List_JP.pdf",
    REPO_ROOT / "cardlist_referenced" / "Card_ID List_JP_original.pdf",
)
SAMPLE_PDF_PATH = next((path for path in SAMPLE_PDF_CANDIDATES if path.exists()), SAMPLE_PDF_CANDIDATES[0])
TEST_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test_outputs"


@unittest.skipUnless(SAMPLE_PDF_PATH.exists(), "sample PDF not available")
class PdfCardToolTests(unittest.TestCase):
    def tearDown(self) -> None:
        if TEST_OUTPUT_ROOT.exists():
            shutil.rmtree(TEST_OUTPUT_ROOT, ignore_errors=True)
        save_card_state_store(SAMPLE_PDF_PATH, {}, [])

    def test_catalog_parses_sample_pdf(self) -> None:
        catalog = load_pdf_catalog(SAMPLE_PDF_PATH)

        self.assertEqual(catalog.total_pages, 1306)
        self.assertEqual(catalog.index_page_count, 39)
        self.assertEqual(catalog.rows_per_index_page, 33)
        self.assertEqual(len(catalog.cards), 1267)
        self.assertEqual(catalog.cards[0].card_id, 1)
        self.assertEqual(catalog.cards[0].target_page, 40)
        self.assertEqual(catalog.cards[-1].card_id, 1267)
        self.assertEqual(catalog.cards[-1].target_page, 1306)

    def test_card_assets_extract_image_without_margins(self) -> None:
        assets = get_card_asset_map(SAMPLE_PDF_PATH, [40, 41, 101])

        self.assertEqual(sorted(assets), [40, 41, 101])
        first_asset = assets[40]
        self.assertEqual(first_asset.card_id, 1)
        self.assertEqual(first_asset.image_bbox, (162.0, 306.0, 450.0, 702.0))
        self.assertEqual((first_asset.image_width, first_asset.image_height), (868, 1212))
        self.assertTrue(first_asset.image_path.exists())

    def test_label_store_roundtrip(self) -> None:
        save_label_store(SAMPLE_PDF_PATH, {1: "草", 2: "炎", 3: ""})
        labels = load_label_store(SAMPLE_PDF_PATH)

        self.assertEqual(labels, {1: ["草"], 2: ["炎"]})

    def test_card_state_store_roundtrip(self) -> None:
        save_card_state_store(SAMPLE_PDF_PATH, {1: "alpha", 2: "", 3: "beta"}, [3, 1, 3])
        state = load_card_state_store(SAMPLE_PDF_PATH)

        self.assertEqual(state.labels, {1: ["alpha"], 3: ["beta"]})
        self.assertEqual(state.work_list_card_ids, frozenset({1, 3}))
        self.assertEqual(state.excluded_card_ids, frozenset())

    def test_build_filtered_pdf_rewrites_links(self) -> None:
        TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output_path = TEST_OUTPUT_ROOT / "filtered.pdf"

        result = build_filtered_pdf(
            SAMPLE_PDF_PATH,
            output_path,
            DeleteRequest(card_ids=frozenset({1}), page_numbers=frozenset({1306})),
        )

        self.assertEqual(result.total_input_pages, 1306)
        self.assertEqual(result.total_output_pages, 1304)
        self.assertEqual(result.kept_cards, 1265)
        self.assertEqual(result.removed_cards, 2)
        self.assertTrue(output_path.exists())

        with output_path.open("rb") as output_file:
            reader = PdfReader(output_file)
            self.assertEqual(len(reader.pages), 1304)

            first_index_annotations = reader.pages[0].get("/Annots") or []
            first_target_page = reader.get_page_number(first_index_annotations[0].get_object()["/Dest"][0]) + 1
            self.assertEqual(first_target_page, 40)

            first_card_annotations = reader.pages[39].get("/Annots") or []
            backlink_target = reader.get_page_number(first_card_annotations[0].get_object()["/Dest"][0]) + 1
            self.assertEqual(backlink_target, 1)

    def test_build_print_pdf_auto_and_fixed_layouts(self) -> None:
        TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        auto_output = TEST_OUTPUT_ROOT / "print-auto.pdf"
        fixed_output = TEST_OUTPUT_ROOT / "print-fixed.pdf"

        auto_result = build_print_pdf(
            SAMPLE_PDF_PATH,
            auto_output,
            [
                PrintItem(card_id=1, quantity=5, label="基本"),
                PrintItem(card_id=2, quantity=5, label="基本"),
            ],
            PrintLayoutSettings(cards_per_sheet="auto"),
        )
        fixed_result = build_print_pdf(
            SAMPLE_PDF_PATH,
            fixed_output,
            [PrintItem(card_id=1, quantity=5, label="基本")],
            PrintLayoutSettings(cards_per_sheet="4"),
        )

        self.assertEqual(auto_result.total_cards_requested, 10)
        self.assertEqual(auto_result.total_cards_placed, 10)
        self.assertEqual(auto_result.cards_per_sheet_used, 9)
        self.assertEqual(auto_result.total_output_pages, 2)
        self.assertEqual(auto_result.skipped_card_ids, ())

        self.assertEqual(fixed_result.total_cards_requested, 5)
        self.assertEqual(fixed_result.total_cards_placed, 5)
        self.assertEqual(fixed_result.cards_per_sheet_used, 4)
        self.assertEqual(fixed_result.total_output_pages, 2)
        self.assertTrue(auto_output.exists())
        self.assertTrue(fixed_output.exists())

    def test_filtered_pdf_can_be_reloaded_for_print(self) -> None:
        TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        filtered_output = TEST_OUTPUT_ROOT / "filtered.pdf"
        print_output = TEST_OUTPUT_ROOT / "filtered-print.pdf"

        build_filtered_pdf(
            SAMPLE_PDF_PATH,
            filtered_output,
            DeleteRequest(card_ids=frozenset({1, 2}), page_numbers=frozenset()),
        )
        filtered_catalog = load_pdf_catalog(filtered_output)
        print_result = build_print_pdf(
            filtered_output,
            print_output,
            [PrintItem(card_id=3, quantity=2, label="印刷")],
            PrintLayoutSettings(cards_per_sheet="1"),
        )

        self.assertEqual(filtered_catalog.cards[0].card_id, 3)
        self.assertEqual(filtered_catalog.cards[0].target_page, 40)
        self.assertEqual(print_result.total_cards_placed, 2)
        self.assertEqual(print_result.cards_per_sheet_used, 1)
        self.assertTrue(print_output.exists())


if __name__ == "__main__":
    unittest.main()
