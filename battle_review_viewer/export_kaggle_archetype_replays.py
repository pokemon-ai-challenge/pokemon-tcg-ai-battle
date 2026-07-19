"""kaggle_replays/deck_predictor/ の学習パイプラインが実際にラベル付けした「本物の」デッキから、
アーキタイプごとに代表デッキを1つ選んで対戦リプレイを生成する。

export_predictor_archetype_replays.py（PDFツール経由で取得した tier_ranking のデッキが元）とは
デッキの出どころが異なる: こちらは Kaggle から取得した実際のリプレイの60枚デッキそのもの
（＝ ML予測器の学習データに実際に使われたデッキ）を使う。

使い方:
  python export_kaggle_archetype_replays.py
  python export_kaggle_archetype_replays.py --archetype alakazam --archetype mega_lucario_ex
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
VIEWER_DIR = ROOT_DIR / "battle_review_viewer"
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"
DEFAULT_OUTPUT_DIR = VIEWER_DIR / "replays"
DECK_PREDICTOR_OUTPUT_DIR = ROOT_DIR / "kaggle_replays" / "deck_predictor" / "output"
DECK_DB_PATH = DECK_PREDICTOR_OUTPUT_DIR / "deck_db.jsonl"
DECK_LABELS_PATH = DECK_PREDICTOR_OUTPUT_DIR / "deck_labels.jsonl"

if str(VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(VIEWER_DIR))
if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import Observation, to_observation_class  # noqa: E402
from export_replay import run_match, trace, working_directory  # noqa: E402
from main import agent, read_deck_csv  # noqa: E402


AgentFn = Callable[[dict], list[int]]


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


def _load_deck_db() -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    with DECK_DB_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[(str(row["episode_id"]), int(row["player_index"]))] = row
    return index


def _load_labels_by_archetype() -> dict[str, list[dict[str, Any]]]:
    by_archetype: dict[str, list[dict[str, Any]]] = {}
    with DECK_LABELS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_archetype.setdefault(row["archetype"], []).append(row)
    return by_archetype


def _pick_representative_label(
    labels: list[dict[str, Any]],
    deck_db: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """代表デッキを1つ選ぶ。rank_at_fetch が分かっている中で最も上位のものを優先し、
    同点は episode_id で決定的にソートする（再実行しても同じデッキが選ばれる）。

    rank_at_fetch は deck_labels.jsonl ではなく deck_db.jsonl 側のフィールドなので、
    ここで突き合わせて参照する（deck_labels.jsonl だけを見て選ぶと常に rank 不明になる）。
    """

    def sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
        deck_row = deck_db.get((str(row["episode_id"]), int(row["player_index"])))
        rank = deck_row.get("rank_at_fetch") if deck_row else None
        rank_key = rank if isinstance(rank, int) else 10**9
        return (rank_key, str(row["episode_id"]), int(row["player_index"]))

    return sorted(labels, key=sort_key)[0]


def _write_replay(
    output_path: Path,
    replay: dict[str, Any],
    *,
    opponent_seed: int,
    deck_type: str,
    label_row: dict[str, Any],
    deck_row: dict[str, Any],
) -> None:
    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "generator": "export_kaggle_archetype_replays.py",
            "opponent": "random-fixed-deck",
            "seed": opponent_seed,
            "deckPath": "sample_submission/deck.csv",
            "sampleSubmissionPath": "sample_submission",
            "result": replay["result"],
            "steps": replay["steps"],
            "predictorDeckType": deck_type,
            "sourceKind": "kaggle_replay",
            "sourceEpisodeId": label_row["episode_id"],
            "sourcePlayerIndex": label_row["player_index"],
            "sourceTeamName": deck_row.get("team_name"),
            "sourceRankAtFetch": deck_row.get("rank_at_fetch"),
            "labelScore": label_row.get("score"),
            "labelMatchedCards": label_row.get("matched_cards", []),
        },
        "frames": replay["frames"],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one replay per ML-predictor archetype, using real Kaggle-sourced decks."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Replay JSON output directory.")
    parser.add_argument("--seed-base", type=int, default=2000, help="Base seed for random opponent actions.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap for turns/actions.")
    parser.add_argument(
        "--archetype",
        action="append",
        default=[],
        help="Limit export to specific archetype labels (as in deck_labels.jsonl). Repeatable.",
    )
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Also export a replay for the 'other' (unclassified) bucket.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing replay files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DECK_DB_PATH.exists() or not DECK_LABELS_PATH.exists():
        raise FileNotFoundError(
            "kaggle_replays/deck_predictor/output/{deck_db,deck_labels}.jsonl が見つかりません。"
            " 先に extract_decks.py / label_decks.py を実行してください。"
        )

    deck_db = _load_deck_db()
    labels_by_archetype = _load_labels_by_archetype()

    archetypes = sorted(labels_by_archetype.keys())
    if not args.include_other:
        archetypes = [a for a in archetypes if a != "other"]
    requested = {item.strip() for item in args.archetype if item.strip()}
    if requested:
        archetypes = [a for a in archetypes if a in requested]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if trace is not None:
        trace.enable_trace()
        trace.pop()

    with working_directory(SAMPLE_SUBMISSION_DIR):
        player_deck = read_deck_csv()
        player0_agent = _fixed_deck_agent(agent, player_deck)

        summary: list[dict[str, Any]] = []
        for index, deck_type in enumerate(archetypes):
            label_row = _pick_representative_label(labels_by_archetype[deck_type], deck_db)
            deck_key = (str(label_row["episode_id"]), int(label_row["player_index"]))
            deck_row = deck_db.get(deck_key)
            if deck_row is None:
                print(f"[skip] {deck_type}: deck_db.jsonl に対応する行が無い ({deck_key})")
                continue
            deck_card_ids = list(deck_row["deck_card_ids"])
            if len(deck_card_ids) != 60:
                print(f"[skip] {deck_type}: デッキが60枚でない ({len(deck_card_ids)}枚)")
                continue

            opponent_seed = args.seed_base + index
            output_name = f"kaggle-{deck_type}-seed{opponent_seed}.json"
            output_path = args.output_dir / output_name
            if output_path.exists() and not args.overwrite:
                print(f"[skip] {output_path} は既に存在します（--overwrite で上書き可能）")
                continue

            random.seed(opponent_seed)
            player1_agent = _fixed_deck_agent(_random_turn_agent, deck_card_ids)
            replay = run_match(
                player0_agent,
                player1_agent,
                player_deck,
                deck_card_ids,
                max_steps=args.max_steps,
            )
            _write_replay(
                output_path,
                replay,
                opponent_seed=opponent_seed,
                deck_type=deck_type,
                label_row=label_row,
                deck_row=deck_row,
            )
            summary.append(
                {
                    "deck_type": deck_type,
                    "output": str(output_path.relative_to(ROOT_DIR)),
                    "source_episode_id": label_row["episode_id"],
                    "source_team_name": deck_row.get("team_name"),
                    "source_rank_at_fetch": deck_row.get("rank_at_fetch"),
                    "steps": replay["steps"],
                    "result": replay["result"],
                }
            )
            print(f"[ok] {deck_type} -> {output_name} (episode {label_row['episode_id']}, rank={deck_row.get('rank_at_fetch')})")

    summary_path = args.output_dir / "kaggle-archetype-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
