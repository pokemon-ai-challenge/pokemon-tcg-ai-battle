from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from cg.api import Card, EnergyType, Pokemon, State

from .opponent_knowledge import OpponentKnowledge


_CONFIG_PATH = Path(__file__).with_name("rough_predictor.json")
_PUBLIC_ZONE_NAMES = ("active", "bench", "discard", "energy", "tool", "pre_evolution", "stadium", "revealed")


def predict(state: State, opponent_knowledge: OpponentKnowledge | dict | None = None) -> dict:
    """Predict the opponent's rough deck archetype from public information."""
    if state is None:
        return _unknown_result()

    # 入口では「特徴量を集める → 全候補を採点する」だけに寄せる。
    config = _load_config()
    features = _extract_features(state, opponent_knowledge)
    return _score_prediction(features, config)


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _extract_features(state: State, opponent_knowledge: OpponentKnowledge | dict | None) -> dict[str, Any]:
    # 現在盤面だけでも動くようにし、履歴があれば後から合流する。
    state_features = _state_prediction_features(state)
    if opponent_knowledge is None:
        return state_features

    knowledge_features = (
        opponent_knowledge.get_prediction_features()
        if hasattr(opponent_knowledge, "get_prediction_features")
        else dict(opponent_knowledge)
    )
    return _merge_features(state_features, knowledge_features)


def _state_prediction_features(state: State) -> dict[str, Any]:
    # 公開盤面の読み出しは OpponentKnowledge 側の表現にそろえる。
    knowledge = OpponentKnowledge(opponent_index=1 - state.yourIndex)
    knowledge.update_from_state(state)
    return knowledge.get_prediction_features()


def _merge_features(state_features: dict[str, Any], knowledge_features: dict[str, Any]) -> dict[str, Any]:
    observed_card_ids = Counter(state_features.get("observed_card_ids", {}))
    observed_card_ids.update(knowledge_features.get("observed_card_ids", {}))

    # 現在盤面ですでに見えているカードは、履歴と二重加点しない。
    for card_id, count in state_features.get("observed_card_ids", {}).items():
        observed_card_ids[card_id] = max(count, observed_card_ids[card_id] - count)
        observed_card_ids[card_id] = max(observed_card_ids[card_id], count)

    zone_cards: dict[str, list[str]] = {
        zone: list(state_features.get("zone_cards", {}).get(zone, [])) for zone in _PUBLIC_ZONE_NAMES
    }
    for zone, names in knowledge_features.get("zone_cards", {}).items():
        bucket = zone_cards.setdefault(zone, [])
        for name in names:
            if name not in bucket:
                bucket.append(name)

    observed_cards = Counter()
    name_to_card_ids: dict[str, dict[int, int]] = {}
    for source in (state_features, knowledge_features):
        for name, count in source.get("observed_cards", {}).items():
            observed_cards[name] = max(observed_cards[name], count)
        for name, ids in source.get("name_to_card_ids", {}).items():
            merged = name_to_card_ids.setdefault(name, {})
            for card_id, count in ids.items():
                merged[card_id] = max(merged.get(card_id, 0), count)

    return {
        "observed_card_ids": dict(observed_card_ids),
        "observed_cards": dict(observed_cards),
        "name_to_card_ids": name_to_card_ids,
        "zone_cards": {zone: names for zone, names in zone_cards.items() if names},
        "observed_pokemon": _unique_list(state_features.get("observed_pokemon", []) + knowledge_features.get("observed_pokemon", [])),
        "observed_energies": _unique_list(state_features.get("observed_energies", []) + knowledge_features.get("observed_energies", [])),
        "observed_tools": _unique_list(state_features.get("observed_tools", []) + knowledge_features.get("observed_tools", [])),
        "energy_types": sorted(set(state_features.get("energy_types", [])) | set(knowledge_features.get("energy_types", []))),
    }


def _unique_list(values: list[Any]) -> list[Any]:
    seen: dict[Any, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen.keys())


def _score_prediction(features: dict[str, Any], config: dict[str, Any]) -> dict:
    generic_names = _extract_generic_names(config.get("generic_cards", []))
    candidates = []

    # 各アーキタイプを独立に採点し、最後に順位付けする。
    for deck_type, archetype in config.get("archetypes", {}).items():
        candidate = _score_archetype(deck_type, archetype, features, config, generic_names)
        if candidate["score"] > 0:
            candidates.append(_with_candidate_metrics(candidate))

    candidates.sort(key=lambda item: (-item["normalized_score"], -item["score"], item["deck_type"]))

    if not candidates:
        return _unknown_result()

    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    margin = _normalized_margin(top, second)
    status = _prediction_status(top, second, config)
    visible_candidates = candidates[:3]
    top_display_name = top["display_name"]
    top_deck_type = top["deck_type"]
    resolved_deck_type = top_deck_type if status == "confident" else "unknown"
    resolved_display_name = top_display_name if status == "confident" else "unknown"
    top_result = {
        "deck_type": resolved_deck_type,
        "display_name": resolved_display_name,
        "top_candidate": top_display_name,
        "top_candidate_deck_type": top_deck_type,
        "status": status,
        "score": round(top["score"], 2),
        "confident_score": round(top["confident_score"], 2),
        "normalized_score": round(top["normalized_score"], 4),
        "match_rate": round(top["match_rate"], 4),
        "margin": round(margin, 4) if margin is not None else None,
        "evidence_count": top["evidence_count"],
        "evidence": top["evidence"],
        # Deprecated compatibility field. Viewer must use match_rate instead.
        "confidence": round(top["match_rate"], 4),
        "candidates": [
            {
                "deck_type": item["deck_type"],
                "display_name": item["display_name"],
                "score": round(item["score"], 2),
                "confident_score": round(item["confident_score"], 2),
                "normalized_score": round(item["normalized_score"], 4),
                "match_rate": round(item["match_rate"], 4),
                "evidence_count": item["evidence_count"],
                "evidence": item["evidence"],
                # Deprecated compatibility field. Viewer must use match_rate instead.
                "confidence": round(item["match_rate"], 4),
            }
            for item in visible_candidates
        ],
    }
    return top_result


def _score_archetype(
    deck_type: str,
    archetype: dict[str, Any],
    features: dict[str, Any],
    config: dict[str, Any],
    generic_names: set[str],
) -> dict[str, Any]:
    role_weights = config.get("role_weights", {})
    ace_spec_bonus = float(config.get("ace_spec_bonus", 1.0))
    role_reason_templates = config.get("role_reason_templates", {})
    observed_card_ids = features.get("observed_card_ids", {})
    name_to_card_ids = features.get("name_to_card_ids", {})
    zone_cards = features.get("zone_cards", {})
    energy_types_seen = set(features.get("energy_types", []))

    score = 0.0
    evidence = []
    seen_keys: set[tuple[str, str, str]] = set()
    seen_line_keys: set[str] = set()

    # cards は主軸カード、role_cards は role ごとにまとめた補助カード群。
    for entry in archetype.get("cards", []):
        score += _match_entry(
            entry,
            zone_cards,
            observed_card_ids,
            name_to_card_ids,
            role_weights,
            role_reason_templates,
            ace_spec_bonus,
            generic_names,
            evidence,
            seen_keys,
            seen_line_keys,
        )

    for role, entries in archetype.get("role_cards", {}).items():
        for entry in entries:
            role_entry = dict(entry)
            role_entry.setdefault("role", role)
            score += _match_entry(
                role_entry,
                zone_cards,
                observed_card_ids,
                name_to_card_ids,
                role_weights,
                role_reason_templates,
                ace_spec_bonus,
                generic_names,
                evidence,
                seen_keys,
                seen_line_keys,
            )

    for energy_entry in archetype.get("energy_types", []):
        energy_name = energy_entry.get("type")
        if energy_name is None:
            continue
        try:
            energy_value = int(EnergyType[energy_name])
        except KeyError:
            continue
        if energy_value not in energy_types_seen:
            continue
        role = energy_entry.get("role", "energy")
        weight = float(role_weights.get(role, 0.0))
        if weight <= 0:
            continue
        score += weight
        evidence.append(
            {
                "card": energy_name,
                "zone": "energy",
                "role": role,
                "weight": round(weight, 2),
                "reason": energy_entry.get("reason") or role_reason_templates.get(role, "{name}").format(name=energy_name),
            }
        )

    # 複数カードが同時に見えているときだけ追加ボーナスを入れる。
    for combo_rule in _combo_rules_for_archetype(archetype, config):
        required_names = combo_rule.get("names", [])
        any_names = combo_rule.get("any_names", [])
        matched_any_name = _matched_any_name(any_names, name_to_card_ids)
        required_matched = all(name in name_to_card_ids for name in required_names)
        any_matched = not any_names or matched_any_name is not None
        if required_names and required_matched and any_matched:
            bonus = float(combo_rule.get("bonus", 0.0))
            if bonus > 0:
                evidence_names = list(required_names)
                if matched_any_name is not None:
                    evidence_names.append(matched_any_name)
                score += bonus
                evidence.append(
                    {
                        "card": " + ".join(evidence_names),
                        "zone": "combo",
                        "role": combo_rule.get("role", "combo"),
                        "weight": round(bonus, 2),
                        "reason": combo_rule.get("reason", "複数の特徴カードが同時に見えている"),
                    }
                )

    evidence.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "deck_type": deck_type,
        "display_name": archetype.get("display_name", deck_type),
        "score": score,
        "evidence": evidence[:8],
        "min_score": archetype.get("min_score", 0),
        "confident_score": archetype.get("confident_score", 0),
    }


def _matched_any_name(any_names: list[str], name_to_card_ids: dict[str, dict[int, int]]) -> str | None:
    for name in any_names:
        if name in name_to_card_ids:
            return name
    return None


def _combo_rules_for_archetype(archetype: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    combo_rules = list(archetype.get("combo_rules", []))
    generated_rule = _evolution_candy_combo_rule(archetype, config)
    if generated_rule is not None:
        combo_rules.append(generated_rule)
    return combo_rules


def _evolution_candy_combo_rule(archetype: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    combo_config = config.get("evolution_candy_combo", {})
    if not combo_config.get("enabled", False):
        return None

    candy_names = list(combo_config.get("candy_names", ["ふしぎなアメ"]))
    evolution_names = _evolution_line_names(archetype)
    if not candy_names or not evolution_names:
        return None
    if not _archetype_has_any_name(archetype, candy_names):
        return None

    candy_name = candy_names[0]
    reason_template = combo_config.get(
        "reason_template",
        "{candy} + {evolution}: 2進化anchorの進化元とふしぎなアメが同時に見えた強い根拠",
    )
    return {
        "names": [candy_name],
        "any_names": evolution_names,
        "bonus": float(combo_config.get("bonus", 0.0)),
        "role": combo_config.get("role", "combo"),
        "reason": reason_template.format(candy=candy_name, evolution=" / ".join(evolution_names)),
    }


def _evolution_line_names(archetype: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in archetype.get("role_cards", {}).get("evolution_line", []):
        names.extend(_entry_names(entry))
    return names


def _archetype_has_any_name(archetype: dict[str, Any], names: list[str]) -> bool:
    target_names = set(names)
    for entry in archetype.get("cards", []):
        if target_names.intersection(_entry_names(entry)):
            return True
    for entries in archetype.get("role_cards", {}).values():
        for entry in entries:
            if target_names.intersection(_entry_names(entry)):
                return True
    return False


def _match_entry(
    entry: dict[str, Any],
    zone_cards: dict[str, list[str]],
    observed_card_ids: dict[int, int],
    name_to_card_ids: dict[str, dict[int, int]],
    role_weights: dict[str, float],
    role_reason_templates: dict[str, str],
    ace_spec_bonus: float,
    generic_names: set[str],
    evidence: list[dict[str, Any]],
    seen_keys: set[tuple[str, str, str]],
    seen_line_keys: set[str],
) -> float:
    role = entry.get("role", "generic")
    if role == "generic":
        return 0.0

    # 汎用カードは見えていてもデッキ推定の根拠にしない。
    names = _entry_names(entry)
    if not names:
        return 0.0
    if any(name in generic_names for name in names):
        return 0.0

    matching_zones = _matching_zones(names, zone_cards)
    if not matching_zones and not _entry_matches_by_card_id(entry, observed_card_ids):
        return 0.0

    line_key = entry.get("line_key")
    if line_key:
        if line_key in seen_line_keys:
            return 0.0
        seen_line_keys.add(line_key)

    base_weight = float(role_weights.get(role, 0.0))
    if base_weight <= 0:
        return 0.0
    if entry.get("ace_spec"):
        base_weight *= ace_spec_bonus

    chosen_zone = matching_zones[0] if matching_zones else "observed"
    final_weight = round(base_weight, 2)
    primary_name = _primary_matched_name(names, chosen_zone, zone_cards, name_to_card_ids)
    evidence_key = (primary_name, chosen_zone, role)
    # 同じ根拠を evidence に重複登録しない。
    if evidence_key in seen_keys:
        return 0.0
    seen_keys.add(evidence_key)

    evidence.append(
        {
            "card": primary_name,
            "zone": chosen_zone,
            "role": role,
            "weight": final_weight,
            "reason": entry.get("reason") or role_reason_templates.get(role, "{name}").format(name=primary_name),
        }
    )
    return final_weight


def _matching_zones(names: list[str], zone_cards: dict[str, list[str]]) -> list[str]:
    zones = []
    for zone, cards in zone_cards.items():
        if any(name in cards for name in names):
            zones.append(zone)
    return zones


def _entry_matches_by_card_id(entry: dict[str, Any], observed_card_ids: dict[int, int]) -> bool:
    card_ids = entry.get("card_id") or entry.get("card_ids") or []
    return any(card_id in observed_card_ids for card_id in card_ids)


def _primary_matched_name(
    names: list[str],
    zone: str,
    zone_cards: dict[str, list[str]],
    name_to_card_ids: dict[str, dict[int, int]],
) -> str:
    zone_names = zone_cards.get(zone, [])
    for name in names:
        if name in zone_names:
            return name
    for name in names:
        if name in name_to_card_ids:
            return name
    return names[0]


def _entry_names(entry: dict[str, Any]) -> list[str]:
    if "name" in entry:
        return [entry["name"]]
    return list(entry.get("names", []))


def _extract_generic_names(generic_cards: list[dict[str, Any]]) -> set[str]:
    names = set()
    for entry in generic_cards:
        names.update(_entry_names(entry))
    return names


def _with_candidate_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    confident_score = _safe_confident_score(candidate)
    normalized_score = float(candidate["score"]) / confident_score if confident_score > 0 else 0.0
    enriched["confident_score"] = confident_score
    enriched["normalized_score"] = normalized_score
    enriched["match_rate"] = max(0.0, min(normalized_score, 1.0))
    enriched["evidence_count"] = len(candidate.get("evidence", []))
    return enriched


def _safe_confident_score(candidate: dict[str, Any]) -> float:
    try:
        confident_score = float(candidate.get("confident_score", 0.0))
    except (TypeError, ValueError):
        confident_score = 0.0
    if confident_score > 0:
        return confident_score
    return max(float(candidate.get("score", 0.0)), 1.0)


def _normalized_margin(top: dict[str, Any], second: dict[str, Any] | None) -> float | None:
    if second is None:
        return None
    return float(top["normalized_score"]) - float(second["normalized_score"])


def _prediction_status(top: dict[str, Any], second: dict[str, Any] | None, config: dict[str, Any]) -> str:
    decision = config.get("prediction_decision", {})
    min_evidence_count = int(decision.get("min_evidence_count", 2))
    min_top_normalized_score = float(decision.get("min_top_normalized_score", 1.0))
    min_normalized_margin = float(decision.get("min_normalized_margin", 0.2))

    if top["evidence_count"] < min_evidence_count:
        return "insufficient_evidence"
    if float(top["normalized_score"]) < min_top_normalized_score:
        return "insufficient_evidence"
    margin = _normalized_margin(top, second)
    if margin is not None and margin < min_normalized_margin:
        return "ambiguous"
    return "confident"


def _unknown_result(candidates: list[dict[str, Any]] | None = None) -> dict:
    candidate_items = candidates or []
    top = candidate_items[0] if candidate_items else None
    second = candidate_items[1] if len(candidate_items) > 1 else None
    margin = _normalized_margin(top, second) if top is not None else None
    status = "no_candidate" if top is None else "insufficient_evidence"
    return {
        "deck_type": "unknown",
        "display_name": "unknown",
        "top_candidate": top["display_name"] if top is not None else None,
        "top_candidate_deck_type": top["deck_type"] if top is not None else None,
        "status": status,
        "score": round(top["score"], 2) if top is not None else 0.0,
        "confident_score": round(top["confident_score"], 2) if top is not None else 0.0,
        "normalized_score": round(top["normalized_score"], 4) if top is not None else 0.0,
        "match_rate": round(top["match_rate"], 4) if top is not None else 0.0,
        "margin": round(margin, 4) if margin is not None else None,
        "evidence_count": top["evidence_count"] if top is not None else 0,
        "evidence": top.get("evidence", []) if top is not None else [],
        # Deprecated compatibility field. Viewer must use match_rate instead.
        "confidence": round(top["match_rate"], 4) if top is not None else 0.0,
        "candidates": [
            {
                "deck_type": item["deck_type"],
                "display_name": item["display_name"],
                "score": round(item["score"], 2),
                "confident_score": round(item["confident_score"], 2),
                "normalized_score": round(item["normalized_score"], 4),
                "match_rate": round(item["match_rate"], 4),
                "evidence_count": item["evidence_count"],
                "evidence": item["evidence"],
                # Deprecated compatibility field. Viewer must use match_rate instead.
                "confidence": round(item["match_rate"], 4),
            }
            for item in candidate_items
        ],
    }
