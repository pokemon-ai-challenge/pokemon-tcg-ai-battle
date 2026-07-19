"""Build web/attack_names_jp.json from the official EN/JP card CSV files.

The game engine exposes attack IDs with English names, while the card CSVs
contain localized move names but no attack IDs.  The English move name is unique
in the current data set, so we use it as the bridge and emit both lookup forms:

    byId:   attackId -> Japanese move name
    byName: English move name -> Japanese move name

Run:
    python battle_review_viewer/build_attack_names_jp.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
EN_CSV = ROOT_DIR / "data" / "EN_Card_Data.csv"
JP_CSV = ROOT_DIR / "data" / "JP_Card_Data.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "web" / "attack_names_jp.json"
MOVE_NAME_INDEX = 13

if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import all_attack  # noqa: E402


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))[1:]


def _build_name_map() -> dict[str, str]:
    english_rows = _read_rows(EN_CSV)
    japanese_rows = _read_rows(JP_CSV)
    candidates: dict[str, set[str]] = defaultdict(set)

    for english, japanese in zip(english_rows, japanese_rows):
        if len(english) <= MOVE_NAME_INDEX or len(japanese) <= MOVE_NAME_INDEX:
            continue
        english_name = english[MOVE_NAME_INDEX].strip()
        japanese_name = japanese[MOVE_NAME_INDEX].strip()
        if not english_name or not japanese_name or english_name == "n/a" or japanese_name == "n/a":
            continue
        candidates[english_name].add(japanese_name)

    ambiguous = {name: values for name, values in candidates.items() if len(values) > 1}
    if ambiguous:
        sample = ", ".join(sorted(ambiguous)[:5])
        raise RuntimeError(f"Ambiguous English attack names in card CSV: {sample}")

    return {name: next(iter(values)) for name, values in candidates.items()}


def build_attack_names_jp() -> dict[str, dict[str, str]]:
    by_name = _build_name_map()
    by_id: dict[str, str] = {}

    missing: list[str] = []
    for attack in all_attack():
        japanese_name = by_name.get(attack.name)
        if japanese_name:
            by_id[str(attack.attackId)] = japanese_name
        else:
            missing.append(f"{attack.attackId}:{attack.name}")

    if missing:
        sample = ", ".join(missing[:5])
        raise RuntimeError(f"Missing Japanese attack names: {sample}")

    return {"byId": by_id, "byName": by_name}


def main() -> None:
    attack_names = build_attack_names_jp()
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(
        json.dumps(attack_names, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[build_attack_names_jp] "
        f"{len(attack_names['byId'])} attack IDs, "
        f"{len(attack_names['byName'])} names -> {DEFAULT_OUT}"
    )


if __name__ == "__main__":
    main()
