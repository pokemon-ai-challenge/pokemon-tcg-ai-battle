from __future__ import annotations

import json
from typing import Any

from cg.api import Observation, all_attack, to_observation_class
from cg.game import visualize_data


ATTACK_NAME_BY_ID = {attack.attackId: attack.name for attack in all_attack()}
AREA_NAMES = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
    6: "prize",
    7: "stadium",
    12: "looking",
}


def current_visual_frame() -> dict[str, Any]:
    visual_history = json.loads(visualize_data())
    return visual_history[-1]


def get_card_from_area(
    frame: dict[str, Any],
    area: Any,
    index: Any,
    player_index: Any,
) -> dict[str, Any] | None:
    if area is None or index is None or player_index is None:
        return None
    current = frame.get("current")
    if current is None:
        return None
    players = current.get("players") or []
    if not (0 <= player_index < len(players)):
        return None
    player = players[player_index]
    area_name = AREA_NAMES.get(area)
    if area_name is None:
        return None

    zone = player.get(area_name)
    if zone is None:
        return None

    if area_name == "stadium":
        return zone[0] if zone else None
    if not isinstance(zone, list):
        return None
    if not (0 <= index < len(zone)):
        return None
    card = zone[index]
    if area_name == "active" and isinstance(card, list):
        return card[0] if card else None
    return card


def format_card_label(card: dict[str, Any] | None) -> str:
    if not card:
        return "unknown card"
    name = card.get("name") or f"card #{card.get('id', '?')}"
    serial = card.get("serial")
    card_id = card.get("id")
    suffix = []
    if card_id is not None:
        suffix.append(f"id={card_id}")
    if serial is not None:
        suffix.append(f"serial={serial}")
    if suffix:
        return f"{name} ({', '.join(suffix)})"
    return str(name)


def describe_option(
    frame: dict[str, Any],
    option: dict[str, Any],
    acting_player: int | None,
) -> str:
    option_type = option.get("type")

    if option_type == "Yes":
        return "Yes"
    if option_type == "No":
        return "No"
    if option_type == "Number":
        return f"Number {option.get('number')}"
    if option_type == "Attack":
        attack_id = option.get("attackId")
        attack_name = ATTACK_NAME_BY_ID.get(attack_id, f"attack #{attack_id}")
        return f"Use attack: {attack_name}"
    if option_type == "Play":
        card = get_card_from_area(frame, 2, option.get("index"), acting_player)
        return f"Play from hand: {format_card_label(card)}"
    if option_type == "Card":
        card = get_card_from_area(
            frame,
            option.get("area"),
            option.get("index"),
            option.get("playerIndex"),
        )
        area_name = AREA_NAMES.get(option.get("area"), f"area {option.get('area')}")
        owner = option.get("playerIndex")
        return f"Select {format_card_label(card)} from P{owner} {area_name}"
    if option_type == "Ability":
        card = get_card_from_area(
            frame,
            option.get("area"),
            option.get("index"),
            option.get("playerIndex"),
        )
        return f"Use ability on {format_card_label(card)}"
    if option_type == "Attach":
        source = get_card_from_area(frame, option.get("area"), option.get("index"), acting_player)
        target = get_card_from_area(
            frame,
            option.get("inPlayArea"),
            option.get("inPlayIndex"),
            acting_player,
        )
        return f"Attach {format_card_label(source)} to {format_card_label(target)}"
    if option_type == "Evolve":
        evolved = get_card_from_area(frame, option.get("area"), option.get("index"), acting_player)
        base = get_card_from_area(
            frame,
            option.get("inPlayArea"),
            option.get("inPlayIndex"),
            acting_player,
        )
        return f"Evolve {format_card_label(base)} into {format_card_label(evolved)}"
    if option_type == "Energy":
        return f"Choose energy index {option.get('energyIndex')} on slot {option.get('index')}"
    if option_type == "EnergyCard":
        return f"Choose attached energy card index {option.get('energyIndex')}"
    if option_type == "ToolCard":
        return f"Choose attached tool card index {option.get('toolIndex')}"
    if option_type == "Retreat":
        return "Retreat"
    if option_type == "End":
        return "End turn"

    return json.dumps(option, ensure_ascii=False)


def summarize_option_list(frame: dict[str, Any], acting_player: int | None) -> list[dict[str, Any]]:
    select = frame.get("select")
    if not select:
        return []

    options = []
    for option_index, option in enumerate(select.get("option", [])):
        options.append(
            {
                "index": option_index,
                "label": describe_option(frame, option, acting_player),
                "raw": option,
            }
        )
    return options


def build_frame_snapshot(
    step_index: int,
    frame: dict[str, Any],
    action: list[int] | None,
    opponent_knowledge_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = frame.get("current")
    acting_player = None if current is None else current.get("yourIndex")
    options = summarize_option_list(frame, acting_player)
    action_labels = []
    if action is not None:
        for action_index in action:
            if 0 <= action_index < len(options):
                action_labels.append(options[action_index]["label"])
            else:
                action_labels.append(f"invalid option index {action_index}")

    return {
        "stepIndex": step_index,
        "turn": None if current is None else current.get("turn"),
        "actingPlayer": acting_player,
        "context": None if frame.get("select") is None else frame["select"].get("context"),
        "action": action,
        "actionLabels": action_labels,
        "options": options,
        "visual": frame,
        # デバッグ用: player0(提出エージェント)視点の OpponentKnowledge スナップショットと、
        # 神視点(visualize_data)との自動突き合わせ結果。player0 の手番以外は None。
        "opponentKnowledgeDebug": opponent_knowledge_debug,
    }


def validate_action(obs: Observation, action: list[int]) -> None:
    if obs.select is None:
        if len(action) != 60:
            raise ValueError(f"Deck selection must return 60 cards, got {len(action)}.")
        return

    if not isinstance(action, list) or not all(isinstance(i, int) for i in action):
        raise TypeError("agent() must return list[int].")
    if not (obs.select.minCount <= len(action) <= obs.select.maxCount):
        raise ValueError(
            f"Action length must be between {obs.select.minCount} and "
            f"{obs.select.maxCount}, got {len(action)}."
        )
    if len(action) != len(set(action)):
        raise ValueError(f"Action must not contain duplicates: {action}")

    option_count = len(obs.select.option)
    for index in action:
        if not 0 <= index < option_count:
            raise IndexError(f"Action index out of range: {index} (options={option_count})")


def snapshot_from_observation(obs_dict: dict, action: list[int] | None = None) -> dict[str, Any]:
    obs = to_observation_class(obs_dict)
    if obs.current is None:
        return build_frame_snapshot(0, current_visual_frame(), action)
    return build_frame_snapshot(0, current_visual_frame(), action)
