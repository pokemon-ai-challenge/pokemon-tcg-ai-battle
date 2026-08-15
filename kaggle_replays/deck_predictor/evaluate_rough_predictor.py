#!/usr/bin/env python3
"""本番の相手デッキ予測器(rough_predictor)の的中率を、実際のリプレイで測る。

label_decks.py が「60枚デッキが丸ごと見えている」前提のオフライン判定なのに対し、
こちらは**本番と同じ経路**——各意思決定時点で OpponentKnowledge に公開情報を積み上げ、
rough_predictor.predict(state, knowledge) を呼ぶ——で評価する。両者はロジックが違う
(実行時にはゲート判定が無く、代わりに confident_score による正規化としきい値がある)ため、
オフラインのラベル精度が良くても実行時の的中率は別物になりうる。

正解ラベルは deck_labels(そのプレイヤーの60枚デッキから判定したアーキタイプ)。

--rough-predictor-json で定義ファイルを差し替えられるので、本番定義と修正案を
同じリプレイ・同じ観測で A/B できる。

指標:
  confident率   : status == "confident" になった割合(予測器が言い切れた割合)
  確信時の正解率: confident のうち正解だった割合(**誤って言い切る**のが一番危ない)
  top1正解率    : status を無視して最有力候補が正解だった割合

使い方:
  python evaluate_rough_predictor.py --limit 300
  python evaluate_rough_predictor.py --limit 300 --rough-predictor-json ./rough_predictor_g2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO / "sample_submission"))

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.opponent_modeling import rough_predictor  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402


def load_labels(path: Path) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                labels[(row["episode_id"], row["player_index"])] = row["archetype"]
    return labels


def evaluate_replay(replay: dict, episode_id: str, labels: dict, checkpoints: list[int], stats: dict) -> None:
    """1リプレイを両プレイヤー視点で走査する。observer から見た「相手」を当てにいく。"""
    steps = replay["steps"]
    for observer in (0, 1):
        truth = labels.get((episode_id, 1 - observer))
        if truth is None:
            continue
        knowledge = OpponentKnowledge(opponent_index=1 - observer)
        pending = sorted(checkpoints)
        seen_turn = 0
        for step in steps:
            agent_step = step[observer]
            if agent_step.get("status") != "ACTIVE":
                continue
            obs_dict = agent_step.get("observation")
            if not obs_dict or obs_dict.get("current") is None:
                continue
            try:
                state = to_observation_class(obs_dict).current
            except Exception:  # noqa: BLE001 -- 壊れたステップは飛ばす
                continue
            knowledge.update_from_state(state)
            seen_turn = state.turn or seen_turn
            # そのチェックポイント以降に到達した最初の意思決定点で1回だけ測る
            while pending and seen_turn >= pending[0]:
                turn = pending.pop(0)
                try:
                    pred = rough_predictor.predict(state, knowledge)
                except Exception:  # noqa: BLE001
                    continue
                record(stats, turn, truth, pred)


def record(stats: dict, turn: int, truth: str, pred: dict) -> None:
    bucket = stats[turn]
    bucket["n"] += 1
    bucket["by_truth"][truth] += 1
    top = pred.get("top_candidate_deck_type")
    if top == truth:
        bucket["top1"] += 1
        bucket["top1_by_truth"][truth] += 1
    if pred.get("status") == "confident":
        bucket["confident"] += 1
        bucket["confident_by_truth"][truth] += 1
        if pred.get("deck_type") == truth:
            bucket["confident_correct"] += 1
            bucket["confident_correct_by_truth"][truth] += 1
        else:
            bucket["confusion"][(truth, pred.get("deck_type"))] += 1


def new_bucket() -> dict:
    return {
        "n": 0, "top1": 0, "confident": 0, "confident_correct": 0,
        "by_truth": Counter(), "top1_by_truth": Counter(),
        "confident_by_truth": Counter(), "confident_correct_by_truth": Counter(),
        "confusion": Counter(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_HERE.parent / "replays_g2"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels_g2.jsonl"))
    parser.add_argument("--rough-predictor-json", default=None, help="差し替える定義(既定=本番)")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--checkpoint-turns", default="4,8,12")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.rough_predictor_json:
        rough_predictor._CONFIG_PATH = Path(args.rough_predictor_json)
        rough_predictor._load_config.cache_clear()
    print(f"定義: {rough_predictor._CONFIG_PATH}")

    labels = load_labels(Path(args.deck_labels))
    checkpoints = [int(x) for x in args.checkpoint_turns.split(",")]
    stats: dict[int, dict] = defaultdict(new_bucket)

    replay_paths = sorted(Path(args.replays_dir).glob("episode-*-replay.json"))[: args.limit]
    for i, path in enumerate(replay_paths, 1):
        episode_id = path.stem.split("-")[1]
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        evaluate_replay(replay, episode_id, labels, checkpoints, stats)
        if i % 50 == 0:
            print(f"  {i}/{len(replay_paths)} replays", flush=True)

    lines = [f"# rough_predictor 実戦的中率 ({rough_predictor._CONFIG_PATH.name})", ""]
    for turn in sorted(stats):
        b = stats[turn]
        if not b["n"]:
            continue
        lines += [
            f"## ターン{turn}時点 (n={b['n']})", "",
            f"- confident率: {b['confident'] / b['n'] * 100:.1f}%",
            f"- 確信時の正解率: {(b['confident_correct'] / b['confident'] * 100) if b['confident'] else 0:.1f}%"
            f"  ({b['confident_correct']}/{b['confident']})",
            f"- top1正解率(status無視): {b['top1'] / b['n'] * 100:.1f}%", "",
            "| 正解アーキ | n | top1正解 | confident | 確信時正解 |", "|---|---:|---:|---:|---:|",
        ]
        for truth, n in b["by_truth"].most_common(12):
            lines.append(
                f"| {truth} | {n} | {b['top1_by_truth'][truth] / n * 100:.0f}% "
                f"| {b['confident_by_truth'][truth] / n * 100:.0f}% "
                f"| {b['confident_correct_by_truth'][truth]}/{b['confident_by_truth'][truth]} |"
            )
        lines += ["", "**誤って言い切った上位**: " + ", ".join(
            f"{t}→{p}×{c}" for (t, p), c in b["confusion"].most_common(6)) or "なし", ""]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
