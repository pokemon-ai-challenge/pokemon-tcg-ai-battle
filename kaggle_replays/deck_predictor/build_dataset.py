#!/usr/bin/env python3
"""各リプレイの各意思決定時点について「相手の見えている情報」を特徴量化し、dataset.jsonl を作る。

特徴量抽出は sample_submission の OpponentKnowledge をそのまま再利用する
(学習時とランタイムで同一の特徴量抽出コードを共有し、train/serve skew を防ぐため)。

各プレイヤー視点(player_index)ごとに OpponentKnowledge を1つ作り、そのプレイヤーが
ACTIVE だったステップを時系列順に流し込む:
    obs = to_observation_class(step_obs_dict)
    knowledge.update_from_logs(obs.logs)   # 呼び出し順固定: logs が先
    knowledge.update_from_state(obs.current)  # state が後
各時点で knowledge.get_prediction_features()["observed_cards"] とターン数をスナップショットし、
1サンプルとして書き出す。ラベルは deck_labels.jsonl の「相手プレイヤー」のアーキタイプ。

デッキ選択ステップ(obs.select が None のステップ、通常は steps[0])はサンプルに含めない。

使い方:
  python build_dataset.py
  python build_dataset.py --replays-dir ../replays --deck-labels ./output/deck_labels.jsonl --out ./output/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent
_SAMPLE_SUBMISSION_DIR = _REPO_ROOT / "sample_submission"

# sample_submission/ を sys.path に追加してから cg.api / ptcg_ai を import する。
# cg.api の import 時に cg.dll がロードされるが、to_observation_class は純粋なデータ変換なので
# ゲームエンジン自体は起動しない(問題ない)。
sys.path.insert(0, str(_SAMPLE_SUBMISSION_DIR))

from cg.api import to_observation_class  # noqa: E402
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402


def load_deck_labels(path: Path) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[(row["episode_id"], row["player_index"])] = row["archetype"]
    return labels


def build_samples_for_replay(replay: dict, episode_id: str, deck_labels: dict[tuple[str, int], str]):
    steps = replay["steps"]
    for player_index in (0, 1):
        opponent_index = 1 - player_index
        label = deck_labels.get((episode_id, opponent_index))
        if label is None:
            # 相手側のデッキがラベル付けされていない(deck_labels.jsonl に無い)場合はスキップ。
            continue

        knowledge = OpponentKnowledge(opponent_index=opponent_index)

        for i, step in enumerate(steps):
            agent_step = step[player_index]
            if agent_step["status"] != "ACTIVE":
                continue
            obs_dict = agent_step["observation"]
            if obs_dict.get("select") is None:
                # デッキ選択ステップ(通常 steps[0])。current も None なので更新も不要。
                continue

            obs = to_observation_class(obs_dict)
            knowledge.update_from_logs(obs.logs)
            knowledge.update_from_state(obs.current)

            if obs.current is None:
                continue

            features = knowledge.get_prediction_features()
            yield {
                "episode_id": episode_id,
                "player_index": player_index,
                "turn": obs.current.turn,
                "step": i,
                "observed_cards": features["observed_cards"],
                "label": label,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays-dir", default=str(_REPO_ROOT / "kaggle_replays" / "replays"))
    parser.add_argument("--deck-labels", default=str(_HERE / "output" / "deck_labels.jsonl"))
    parser.add_argument("--out", default=str(_HERE / "output" / "dataset.jsonl"))
    args = parser.parse_args()

    replays_dir = Path(args.replays_dir)
    deck_labels = load_deck_labels(Path(args.deck_labels))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    replay_paths = sorted(replays_dir.glob("episode-*-replay.json"))
    n_files_ok = 0
    n_samples = 0
    n_errors = 0
    errors: list[str] = []

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, replay_path in enumerate(replay_paths, start=1):
            episode_id = replay_path.stem.split("-")[1]
            try:
                with replay_path.open(encoding="utf-8") as rf:
                    replay = json.load(rf)
                n_before = n_samples
                for sample in build_samples_for_replay(replay, episode_id, deck_labels):
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n_samples += 1
                if n_samples > n_before:
                    n_files_ok += 1
            except Exception as exc:  # noqa: BLE001 - keep going on any per-file failure
                n_errors += 1
                errors.append(f"{replay_path.name}: {exc!r}")

            if i % 50 == 0 or i == len(replay_paths):
                print(
                    f"  {i}/{len(replay_paths)} replays processed, {n_samples} samples so far, "
                    f"{n_errors} errors...",
                    file=sys.stderr,
                )

    print(f"{n_files_ok}/{len(replay_paths)}件のリプレイから{n_samples}件のサンプルを {out_path} に書き出しました")
    if n_errors:
        print(f"エラー: {n_errors}件のリプレイでエラーが発生しました")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ...他 {len(errors) - 20} 件")


if __name__ == "__main__":
    main()
