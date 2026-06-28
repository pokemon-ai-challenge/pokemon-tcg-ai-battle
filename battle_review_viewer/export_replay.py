import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "replays"

if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import Observation, all_attack, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from main import agent, read_deck_csv  # noqa: E402


AgentFn = Callable[[dict], list[int]]
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


@contextmanager
def working_directory(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def random_agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def normalize_name(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def current_visual_frame() -> dict[str, Any]:
    visual_history = json.loads(visualize_data())
    return visual_history[-1]


def get_card_from_area(frame: dict[str, Any], area: Any, index: Any, player_index: Any) -> dict[str, Any] | None:
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


def describe_option(frame: dict[str, Any], option: dict[str, Any], acting_player: int | None) -> str:
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
        card = get_card_from_area(frame, option.get("area"), option.get("index"), option.get("playerIndex"))
        area_name = AREA_NAMES.get(option.get("area"), f"area {option.get('area')}")
        owner = option.get("playerIndex")
        return f"Select {format_card_label(card)} from P{owner} {area_name}"
    if option_type == "Ability":
        card = get_card_from_area(frame, option.get("area"), option.get("index"), option.get("playerIndex"))
        return f"Use ability on {format_card_label(card)}"
    if option_type == "Attach":
        source = get_card_from_area(frame, option.get("area"), option.get("index"), acting_player)
        target = get_card_from_area(frame, option.get("inPlayArea"), option.get("inPlayIndex"), acting_player)
        return f"Attach {format_card_label(source)} to {format_card_label(target)}"
    if option_type == "Evolve":
        evolved = get_card_from_area(frame, option.get("area"), option.get("index"), acting_player)
        base = get_card_from_area(frame, option.get("inPlayArea"), option.get("inPlayIndex"), acting_player)
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


def build_frame_snapshot(step_index: int, frame: dict[str, Any], action: list[int] | None) -> dict[str, Any]:
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
    }


def run_match(player0: AgentFn, player1: AgentFn, deck0: list[int], deck1: list[int], max_steps: int) -> dict[str, Any]:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")

    frames: list[dict[str, Any]] = []
    result = None
    steps = 0

    try:
        while True:
            frame = current_visual_frame()
            obs = to_observation_class(obs_dict)

            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                frames.append(build_frame_snapshot(steps, frame, action=None))
                break

            acting_player = obs.current.yourIndex if obs.current is not None else 0
            acting_agent = player0 if acting_player == 0 else player1
            action = acting_agent(obs_dict)
            frames.append(build_frame_snapshot(steps, frame, action=action))
            obs_dict = battle_select(action)
            steps += 1

            if steps >= max_steps:
                raise RuntimeError(f"Reached max_steps={max_steps} before the match finished.")
    finally:
        battle_finish()

    return {
        "result": result,
        "steps": steps,
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a local replay for the battle review viewer.")
    parser.add_argument("--output", type=Path, default=None, help="Replay JSON output path.")
    parser.add_argument(
        "--opponent",
        choices=("self", "random"),
        default="self",
        help="Opponent policy. 'self' uses main.agent for both players.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for the random opponent.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap for turns/actions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"replay-{timestamp}-{args.opponent}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with working_directory(SAMPLE_SUBMISSION_DIR):
        deck0 = read_deck_csv()
        deck1 = read_deck_csv()
        player1 = agent if args.opponent == "self" else random_agent
        replay = run_match(agent, player1, deck0, deck1, max_steps=args.max_steps)

    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "opponent": args.opponent,
            "seed": args.seed,
            "deckPath": "sample_submission/deck.csv",
            "sampleSubmissionPath": "sample_submission",
            "result": replay["result"],
            "steps": replay["steps"],
        },
        "frames": replay["frames"],
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote replay: {output_path}")
    print(f"Result={payload['metadata']['result']} steps={payload['metadata']['steps']}")


if __name__ == "__main__":
    main()
