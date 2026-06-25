from __future__ import annotations

import ast
import unittest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SOURCE = APP_PATH.read_text(encoding="utf-8")
APP_AST = ast.parse(APP_SOURCE, filename=str(APP_PATH))


def _load_export_helpers() -> dict:
    target_names = [
        "_normalize_deck_card_quantities",
        "_coerce_positive_int_list",
        "_normalize_deck_card_resolutions",
        "_normalize_deck_csv_ambiguities",
        "_resolve_deck_recipe",
        "_format_unresolved_deck_cards",
        "_resolve_deck_csv_output_path",
        "_expand_deck_card_ids",
        "_build_deck_csv_text",
        "_build_complete_deck_csv_text",
        "_add_cards_to_deck_recipe",
        "_set_deck_card_quantity",
    ]
    function_nodes = {
        node.name: node
        for node in APP_AST.body
        if isinstance(node, ast.FunctionDef) and node.name in target_names
    }
    namespace = {
        "Path": Path,
        "DEFAULT_DECK_CSV_OUTPUT_PATH": Path(r"C:\dev\pokemon-tcg-ai-battle\sample_submission\deck.csv"),
    }
    for name in target_names:
        module = ast.Module(body=[function_nodes[name]], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


HELPERS = _load_export_helpers()


class DeckCsvExportTests(unittest.TestCase):
    def test_build_deck_csv_text_expands_to_60_lines(self) -> None:
        pdf_state = {
            "deck_card_quantities": {
                "deck-1": {
                    101: 4,
                    202: 56,
                }
            }
        }

        deck_csv_text = HELPERS["_build_deck_csv_text"](pdf_state, "deck-1")

        self.assertEqual(deck_csv_text.count("\n"), 60)
        self.assertEqual(deck_csv_text.splitlines()[:4], ["101", "101", "101", "101"])
        self.assertEqual(deck_csv_text.splitlines()[-1], "202")

    def test_build_deck_csv_text_allows_non_60_card_recipe_for_manual_fixups(self) -> None:
        pdf_state = {"deck_card_quantities": {"deck-1": {101: 4, 202: 12}}}

        deck_csv_text = HELPERS["_build_deck_csv_text"](pdf_state, "deck-1")

        self.assertEqual(deck_csv_text.count("\n"), 16)
        self.assertEqual(deck_csv_text.splitlines().count("101"), 4)
        self.assertEqual(deck_csv_text.splitlines().count("202"), 12)

    def test_build_complete_deck_csv_text_rejects_non_60_card_recipe(self) -> None:
        pdf_state = {"deck_card_quantities": {"deck-1": {101: 4, 202: 12}}}

        with self.assertRaises(ValueError):
            HELPERS["_build_complete_deck_csv_text"](pdf_state, "deck-1")

    def test_build_deck_csv_text_requires_resolving_ambiguous_same_name_cards(self) -> None:
        pdf_state = {
            "deck_card_quantities": {"deck-1": {202: 58}},
            "deck_csv_ambiguities": {
                "deck-1": [
                    {"norm_name": "リオル", "name": "リオル", "quantity": 2, "candidate_ids": [333, 677]}
                ]
            },
        }

        with self.assertRaises(ValueError):
            HELPERS["_build_deck_csv_text"](pdf_state, "deck-1")

    def test_build_deck_csv_text_uses_selected_card_id_for_ambiguous_same_name_cards(self) -> None:
        pdf_state = {
            "deck_card_quantities": {"deck-1": {202: 58}},
            "deck_csv_ambiguities": {
                "deck-1": [
                    {"norm_name": "リオル", "name": "リオル", "quantity": 2, "candidate_ids": [333, 677]}
                ]
            },
            "deck_card_resolutions": {"deck-1": {"リオル": 677}},
        }

        deck_csv_text = HELPERS["_build_deck_csv_text"](pdf_state, "deck-1")

        self.assertEqual(deck_csv_text.count("\n"), 60)
        self.assertEqual(deck_csv_text.splitlines().count("677"), 2)
        self.assertEqual(deck_csv_text.splitlines().count("202"), 58)

    def test_resolve_deck_csv_output_path_adds_default_name_for_directory(self) -> None:
        resolved = HELPERS["_resolve_deck_csv_output_path"](Path.cwd() / "sample_submission")

        self.assertEqual(resolved.name, "deck.csv")

    def test_add_cards_to_deck_recipe_increments_existing_recipe(self) -> None:
        pdf_state = {"deck_card_quantities": {"deck-1": {101: 4}}}

        added = HELPERS["_add_cards_to_deck_recipe"](pdf_state, "deck-1", [101, 202], 2)

        self.assertEqual(added, 4)
        self.assertEqual(pdf_state["deck_card_quantities"]["deck-1"][101], 6)
        self.assertEqual(pdf_state["deck_card_quantities"]["deck-1"][202], 2)

    def test_set_deck_card_quantity_updates_and_removes_recipe_entries(self) -> None:
        pdf_state = {"deck_card_quantities": {"deck-1": {101: 4, 202: 2}}}

        updated = HELPERS["_set_deck_card_quantity"](pdf_state, "deck-1", 101, 3)
        removed = HELPERS["_set_deck_card_quantity"](pdf_state, "deck-1", 202, 0)

        self.assertEqual(updated, 3)
        self.assertEqual(removed, 0)
        self.assertEqual(pdf_state["deck_card_quantities"]["deck-1"], {101: 3})


if __name__ == "__main__":
    unittest.main()
