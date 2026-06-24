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
        "_resolve_deck_csv_output_path",
        "_expand_deck_card_ids",
        "_build_deck_csv_text",
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

    def test_build_deck_csv_text_rejects_non_60_card_recipe(self) -> None:
        pdf_state = {"deck_card_quantities": {"deck-1": {101: 4, 202: 12}}}

        with self.assertRaisesRegex(ValueError, "60 枚ちょうど"):
            HELPERS["_build_deck_csv_text"](pdf_state, "deck-1")

    def test_resolve_deck_csv_output_path_adds_default_name_for_directory(self) -> None:
        resolved = HELPERS["_resolve_deck_csv_output_path"](Path.cwd() / "sample_submission")

        self.assertEqual(resolved.name, "deck.csv")


if __name__ == "__main__":
    unittest.main()
