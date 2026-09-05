from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT_DIR / "battle_review_viewer"
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
DEFAULT_OUTPUT_DIR = VIEWER_DIR / "replays"
PREDICTOR_CONFIG_PATH = SAMPLE_SUBMISSION_DIR / "ptcg_ai" / "opponent_modeling" / "rough_predictor.json"
TIER_CACHE_DIR = ROOT_DIR / "cardlist_referenced" / "pdf_card_editor" / ".cache" / "tier_ranking"
JP_CARD_DATA_PATH = ROOT_DIR / "data" / "JP_Card_Data.csv"

if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))
if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import Observation, to_observation_class  # noqa: E402
from export_replay import run_match, trace, working_directory  # noqa: E402
from main import agent, read_deck_csv  # noqa: E402


AgentFn = Callable[[dict], list[int]]
SPECIAL_LABEL_ALIASES = {
    "toxtricity": ["ストリンダー", "ストリンダーバレット"],
}


def _normalize_card_name(name: str) -> str:
    text = re.sub(r"[（(][^（）()]*[)）]", "", str(name))
    text = re.sub(r"[【】「」『』\[\]（）()・,，、。.\s]", "", text)
    return text.casefold()


def _read_deck_csv_file(path: Path) -> list[int]:
    card_ids: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            card_ids.append(int(line.split(",")[0].strip()))
    return card_ids


def _fixed_deck_agent(base_agent: AgentFn, fixed_deck: list[int]) -> AgentFn:
    fixed_copy = list(fixed_deck)

    def wrapped(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(fixed_copy)
        return base_agent(obs_dict)

    return wrapped


def _random_turn_agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return []
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def _latest_tier_cache_path() -> Path:
    candidates = sorted(TIER_CACHE_DIR.glob("*-v4.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted(TIER_CACHE_DIR.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No tier cache JSON found in {TIER_CACHE_DIR}")
    return candidates[-1]


def _load_predictor_archetypes() -> list[dict[str, str]]:
    payload = json.loads(PREDICTOR_CONFIG_PATH.read_text(encoding="utf-8"))
    return [
        {"deck_type": deck_type, "display_name": str(arch.get("display_name", deck_type)).strip()}
        for deck_type, arch in payload.get("archetypes", {}).items()
    ]


def _load_card_catalog() -> tuple[dict[str, list[int]], dict[str, list[int]], dict[int, str]]:
    normalized_name_to_ids: dict[str, list[int]] = {}
    exact_name_to_ids: dict[str, list[int]] = {}
    card_id_to_name: dict[int, str] = {}

    with JP_CARD_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_card_id = str(row.get("カード ID", "")).strip()
            raw_name = str(row.get("カード名", "")).strip()
            if not raw_card_id or not raw_name:
                continue
            card_id = int(raw_card_id)
            normalized = _normalize_card_name(raw_name)
            card_id_to_name[card_id] = raw_name
            normalized_name_to_ids.setdefault(normalized, []).append(card_id)
            exact_name_to_ids.setdefault(raw_name, []).append(card_id)

    for ids in normalized_name_to_ids.values():
        ids.sort()
    for ids in exact_name_to_ids.values():
        ids.sort()
    return normalized_name_to_ids, exact_name_to_ids, card_id_to_name


def _candidate_label_norms(deck_type: str, display_name: str) -> set[str]:
    aliases = [display_name, *SPECIAL_LABEL_ALIASES.get(deck_type, [])]
    return {_normalize_card_name(alias) for alias in aliases if str(alias).strip()}


def _resolve_recipe_cards(
    recipe_payload: dict[str, Any],
    normalized_name_to_ids: dict[str, list[int]],
    exact_name_to_ids: dict[str, list[int]],
) -> tuple[list[int], list[dict[str, Any]]]:
    resolved_card_ids: list[int] = []
    ambiguities: list[dict[str, Any]] = []

    cards = dict(recipe_payload.get("cards", {}))
    for raw_key, raw_info in cards.items():
        info = dict(raw_info)
        quantity = int(info.get("quantity", 0))
        if quantity <= 0:
            continue

        display_name = str(info.get("name", "")).strip() or str(raw_key).strip()
        normalized = _normalize_card_name(display_name or str(raw_key))
        candidate_ids = list(normalized_name_to_ids.get(normalized, []))
        if not candidate_ids:
            raise ValueError(f"Card not found in JP_Card_Data.csv: {display_name}")

        exact_ids = exact_name_to_ids.get(display_name, [])
        chosen_id = None
        if exact_ids:
            chosen_id = exact_ids[0]
        else:
            chosen_id = candidate_ids[0]

        if len(candidate_ids) > 1:
            ambiguities.append(
                {
                    "card_name": display_name,
                    "chosen_card_id": chosen_id,
                    "candidate_ids": candidate_ids,
                    "quantity": quantity,
                }
            )

        resolved_card_ids.extend([chosen_id] * quantity)

    if len(resolved_card_ids) != 60:
        label = str(recipe_payload.get("label", "")).strip() or "(unknown recipe)"
        raise ValueError(f"Resolved deck is {len(resolved_card_ids)} cards instead of 60: {label}")

    return resolved_card_ids, ambiguities


def _pick_representative_recipe(
    cache_payload: dict[str, Any],
    deck_type: str,
    display_name: str,
    normalized_name_to_ids: dict[str, list[int]],
    exact_name_to_ids: dict[str, list[int]],
) -> dict[str, Any]:
    deck_archetype = dict(cache_payload.get("deck_archetype", {}))
    deck_catalog = dict(cache_payload.get("deck_catalog", {}))
    deck_tier = dict(cache_payload.get("deck_tier", {}))
    deck_recipes = dict(cache_payload.get("deck_recipes", {}))

    target_norms = _candidate_label_norms(deck_type, display_name)
    candidates: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []

    for deck_id, archetype_name in deck_archetype.items():
        recipe_payload = deck_recipes.get(deck_id)
        if not isinstance(recipe_payload, dict):
            continue

        recipe_label = str(recipe_payload.get("label", "")).strip()
        label_norm = _normalize_card_name(str(archetype_name))
        record = {
            "deck_id": str(deck_id),
            "archetype_name": str(archetype_name),
            "deck_name": str(deck_catalog.get(deck_id, recipe_label)).strip() or recipe_label or str(deck_id),
            "tier": int(deck_tier.get(deck_id, 99) or 99),
            "recipe": recipe_payload,
        }
        if label_norm in target_norms:
            candidates.append(record)
        elif any(norm and (norm in label_norm or label_norm in norm) for norm in target_norms):
            fallbacks.append(record)

    pool = candidates or fallbacks
    if not pool:
        raise ValueError(f"No deck recipe found for {display_name}")

    pool.sort(key=lambda item: (item["tier"], item["deck_name"], item["deck_id"]))

    last_error: Exception | None = None
    for item in pool:
        try:
            deck_card_ids, ambiguities = _resolve_recipe_cards(
                item["recipe"],
                normalized_name_to_ids,
                exact_name_to_ids,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        return {
            **item,
            "deck_card_ids": deck_card_ids,
            "ambiguities": ambiguities,
        }

    raise ValueError(f"Deck recipes were found for {display_name}, but none resolved cleanly: {last_error}")


def _write_replay(
    output_path: Path,
    replay: dict[str, Any],
    *,
    opponent_seed: int,
    source_cache_path: Path,
    recipe_info: dict[str, Any],
) -> None:
    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "generator": "export_predictor_archetype_replays.py",
            "opponent": "random-fixed-deck",
            "seed": opponent_seed,
            "deckPath": "sample_submission/deck.csv",
            "sampleSubmissionPath": "sample_submission",
            "result": replay["result"],
            "steps": replay["steps"],
            "predictorDeckType": recipe_info["deck_type"],
            "predictorDisplayName": recipe_info["display_name"],
            "sourceDeckId": recipe_info["deck_id"],
            "sourceDeckName": recipe_info["deck_name"],
            "sourceDeckLabel": str(recipe_info["recipe"].get("label", "")).strip(),
            "sourceTier": recipe_info["tier"],
            "sourceTierCache": str(source_cache_path.relative_to(ROOT_DIR)),
            "ambiguityCount": len(recipe_info["ambiguities"]),
            "ambiguities": recipe_info["ambiguities"],
        },
        "frames": replay["frames"],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one replay per rough-predictor archetype.")
    parser.add_argument("--tier-cache", type=Path, default=None, help="Tier cache JSON to use.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Replay JSON output directory.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base seed for random opponent actions.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap for turns/actions.")
    parser.add_argument(
        "--archetype",
        action="append",
        default=[],
        help="Limit export to specific predictor deck_type values. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing replay files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_cache_path = args.tier_cache.resolve() if args.tier_cache else _latest_tier_cache_path().resolve()
    cache_payload = json.loads(source_cache_path.read_text(encoding="utf-8"))
    normalized_name_to_ids, exact_name_to_ids, _ = _load_card_catalog()
    archetypes = _load_predictor_archetypes()
    requested = {item.strip() for item in args.archetype if item.strip()}
    if requested:
        archetypes = [item for item in archetypes if item["deck_type"] in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if trace is not None:
        trace.enable_trace()
        trace.pop()

    with working_directory(SAMPLE_SUBMISSION_DIR):
        player_deck = read_deck_csv()
        player0_agent = _fixed_deck_agent(agent, player_deck)

        summary: list[dict[str, Any]] = []
        for index, archetype in enumerate(archetypes):
            # 1アーキタイプの構築・対戦に失敗しても（tier_ranking の表記ゆれでカード名が
            # 解決できない等）、残りのアーキタイプの生成は続行する。
            try:
                recipe_info = _pick_representative_recipe(
                    cache_payload,
                    archetype["deck_type"],
                    archetype["display_name"],
                    normalized_name_to_ids,
                    exact_name_to_ids,
                )
                recipe_info["deck_type"] = archetype["deck_type"]
                recipe_info["display_name"] = archetype["display_name"]

                opponent_seed = args.seed_base + index
                output_name = f"predictor-{archetype['deck_type']}-seed{opponent_seed}.json"
                output_path = args.output_dir / output_name
                if output_path.exists() and not args.overwrite:
                    print(f"[skip] {output_path} は既に存在します（--overwrite で上書き可能）")
                    continue

                random.seed(opponent_seed)
                player1_agent = _fixed_deck_agent(_random_turn_agent, recipe_info["deck_card_ids"])
                replay = run_match(
                    player0_agent,
                    player1_agent,
                    player_deck,
                    recipe_info["deck_card_ids"],
                    max_steps=args.max_steps,
                )
                _write_replay(
                    output_path,
                    replay,
                    opponent_seed=opponent_seed,
                    source_cache_path=source_cache_path,
                    recipe_info=recipe_info,
                )
            except Exception as exc:  # noqa: BLE001 -- 1件の失敗で残りの生成を止めない
                print(f"[skip] {archetype['deck_type']}: {exc}")
                continue
            summary.append(
                {
                    "deck_type": archetype["deck_type"],
                    "display_name": archetype["display_name"],
                    "output": str(output_path.relative_to(ROOT_DIR)),
                    "source_deck_id": recipe_info["deck_id"],
                    "source_deck_name": recipe_info["deck_name"],
                    "source_label": str(recipe_info["recipe"].get("label", "")).strip(),
                    "tier": recipe_info["tier"],
                    "ambiguity_count": len(recipe_info["ambiguities"]),
                    "steps": replay["steps"],
                    "result": replay["result"],
                }
            )
            print(f"[ok] {archetype['deck_type']} -> {output_name}")

    summary_path = args.output_dir / "predictor-archetype-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
