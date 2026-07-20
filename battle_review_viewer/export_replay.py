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
DEFAULT_DECK_PATH = SAMPLE_SUBMISSION_DIR / "deck.csv"

if str(SAMPLE_SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start, visualize_data  # noqa: E402
from main import agent, read_deck_csv  # noqa: E402
try:
    from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402
except Exception:  # noqa: BLE001 -- keep the viewer usable without the predictor branch
    OpponentKnowledge = None
try:
    from ptcg_ai.opponent_modeling.rough_predictor import predict as predict_deck  # noqa: E402
except Exception:  # noqa: BLE001 -- 予測器が無い/壊れていてもリプレイ生成は続行する
    predict_deck = None
try:
    from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402

    _ml_predictor = HybridDeckPredictor()
except Exception:  # noqa: BLE001 -- ML予測器が無い/壊れていてもリプレイ生成は続行する
    _ml_predictor = None
try:
    from src.decision import trace  # noqa: E402
except ImportError:
    # sample_submission/src はこのブランチにはまだ無い（B層の意思決定トレースは別ブランチ由来の
    # 任意機能）。無ければトレース収集を単に無効化して、リプレイ生成自体は続行する。
    trace = None
try:
    from .viewer_state import build_frame_snapshot  # type: ignore[attr-defined]  # noqa: E402
except ImportError:
    from viewer_state import build_frame_snapshot  # noqa: E402
try:
    from .opponent_knowledge_diff import collect_ground_truth, diff_against_ground_truth  # noqa: E402
except ImportError:
    from opponent_knowledge_diff import collect_ground_truth, diff_against_ground_truth  # noqa: E402
try:
    from .ml_prediction_debug import build_ml_prediction_debug  # noqa: E402
except ImportError:
    from ml_prediction_debug import build_ml_prediction_debug  # noqa: E402
try:
    from .hidden_info_debug import build_hidden_info_debug  # noqa: E402
except ImportError:
    from hidden_info_debug import build_hidden_info_debug  # noqa: E402
try:
    from ptcg_ai.hidden_information.own_hidden_state import OwnHiddenState  # noqa: E402
    from ptcg_ai.hidden_information.opponent_hidden_state import OpponentHiddenState  # noqa: E402
except Exception:  # noqa: BLE001 -- 非公開情報推定レイヤーが無い/壊れていてもリプレイ生成は続行する
    OwnHiddenState = None
    OpponentHiddenState = None


AgentFn = Callable[[dict], list[int]]

# run_match は常に player0 = 提出エージェント(main.agent) として実行するため、
# 「自分のエージェントがどこまで相手(seat=1)を正しく観測できているか」を検証する対象は固定でよい。
OPPONENT_SEAT = 1


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


def read_deck_csv_file(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    deck: list[int] = []
    for raw_value in text.replace(",", "\n").splitlines():
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    if len(deck) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs, but found {len(deck)}.")
    return deck


def resolve_deck_path(value: str | Path | None) -> Path:
    if value is None or str(value).strip() == "":
        return DEFAULT_DECK_PATH
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def display_deck_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def with_initial_deck(base_agent: AgentFn, deck: list[int]) -> AgentFn:
    def wrapped(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        return base_agent(obs_dict)

    return wrapped


def random_agent_for_deck(deck: list[int]) -> AgentFn:
    def wrapped(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        return random.sample(range(len(obs.select.option)), obs.select.maxCount)

    return wrapped


def agent_for_policy(policy: str, deck: list[int]) -> AgentFn:
    if policy == "self":
        return with_initial_deck(agent, deck)
    if policy == "random":
        return random_agent_for_deck(deck)
    raise ValueError(f"Unknown CPU policy: {policy}")


def normalize_name(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def current_visual_frame() -> dict[str, Any]:
    visual_history = json.loads(visualize_data())
    return visual_history[-1]


def build_opponent_knowledge_debug(
    knowledge: Any,
    real_state: Any,
    visual_current: dict[str, Any] | None,
    own_state: Any = None,
    opponent_state: Any = None,
    select: Any = None,
) -> dict[str, Any]:
    """このフレーム時点の観測特徴量と、神視点との diff をまとめてビュアー向けに返す。

    ``real_state`` はエージェントに渡される本物の ``obs.current``。セットアップ中で
    まだ両者のバトル場が同時公開されていない間は、こちらでは相手のバトル場が
    伏せ（``None``）として表現される。神視点(``visual_current``)は伏せの中身まで
    見せてしまうため、伏せの間はバトル場を diff の比較対象から外す。
    """
    revealed_active = False
    if real_state is not None:
        opponent_player_state = real_state.players[OPPONENT_SEAT]
        revealed_active = bool(opponent_player_state.active and opponent_player_state.active[0] is not None)

    ground_truth = (
        {}
        if visual_current is None
        else collect_ground_truth(visual_current, OPPONENT_SEAT, revealed_active=revealed_active)
    )
    diff = diff_against_ground_truth(knowledge.get_observed_cards(), ground_truth)

    # 相手デッキ予測器を同じ観測（現盤面＋履歴）で走らせ、結果も一緒に埋め込む。
    prediction = None
    if predict_deck is not None and real_state is not None:
        try:
            prediction = predict_deck(real_state, knowledge)
        except Exception as exc:  # noqa: BLE001 -- 予測器が落ちてもリプレイ生成は止めない
            prediction = {"error": str(exc)}

    # ML版予測器（学習済みロジスティック回帰）も同じ観測で走らせ、rough_predictor と並べて見比べられるようにする。
    ml_prediction = build_ml_prediction_debug(_ml_predictor, knowledge, real_state)

    # 非公開情報推定レイヤー（自分の山札∪サイド、相手の山札/手札/サイド）の周辺確率もここで一緒に埋め込む。
    # own_state/opponent_state の update() 呼び出しはこの関数の内部（build_hidden_info_debug）が担う
    # （このフレームにつき build_opponent_knowledge_debug は1回しか呼ばれないため二重更新にならない）。
    hidden_info = build_hidden_info_debug(
        own_state, opponent_state, _ml_predictor, knowledge, real_state, select, visual_current=visual_current
    )

    return {
        "features": knowledge.get_prediction_features(),
        "diff": diff,
        "prediction": prediction,
        "ml_prediction": ml_prediction,
        "hidden_info": hidden_info,
    }


def run_match(player0: AgentFn, player1: AgentFn, deck0: list[int], deck1: list[int], max_steps: int) -> dict[str, Any]:
    obs_dict, start_data = battle_start(deck0, deck1)
    if start_data.errorType != 0:
        raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")

    # player0(提出エージェント)が実際に受け取れる情報だけから、相手(seat=1)の観測を組み立てる。
    knowledge = None if OpponentKnowledge is None else OpponentKnowledge(opponent_index=OPPONENT_SEAT)
    # 非公開情報推定レイヤーも同じく player0 視点で独立に構築する（match_context シングルトンは
    # player0/player1 が交互に同じインスタンスを踏み合ってしまうため、ビュアーのデバッグ表示には使わない）。
    own_state = None if OwnHiddenState is None else OwnHiddenState(deck0)
    opponent_state = None if OpponentHiddenState is None else OpponentHiddenState()

    frames: list[dict[str, Any]] = []
    result = None
    steps = 0

    try:
        while True:
            frame = current_visual_frame()
            obs = to_observation_class(obs_dict)
            current = frame.get("current")

            # obs_dict は「今まさに選択を求められているプレイヤー」視点の観測。player0 視点の時だけ
            # OpponentKnowledge を更新する（player1 視点の obs には player1 自身の非公開情報が
            # 含まれるため、それを player0 の観測として使ってしまうとカンニングになる）。
            # 呼び出し順が重要: logs は「この state に至るまでの出来事」なので先に処理し、
            # 盤面スキャン(update_from_state)を最後に当てて現在ゾーンを確定させる。
            debug_payload = None
            if knowledge is not None and obs.current is not None and obs.current.yourIndex == 0:
                knowledge.update_from_logs(obs.logs)
                knowledge.update_from_state(obs.current)
                debug_payload = build_opponent_knowledge_debug(
                    knowledge, obs.current, current, own_state, opponent_state, obs.select
                )

            if obs.current is not None and obs.current.result != -1:
                result = obs.current.result
                frames.append(
                    build_frame_snapshot(steps, frame, action=None, opponent_knowledge_debug=debug_payload)
                )
                break

            acting_player = obs.current.yourIndex if obs.current is not None else 0
            acting_agent = player0 if acting_player == 0 else player1
            action = acting_agent(obs_dict)
            snapshot = build_frame_snapshot(steps, frame, action=action, opponent_knowledge_debug=debug_payload)
            # このフレームの意思決定理由（B層）を回収して付ける（空なら付けない＝optional）。
            decision_trace = trace.pop() if trace is not None else None
            if decision_trace:
                snapshot["trace"] = decision_trace
            frames.append(snapshot)
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
    parser.add_argument(
        "--player-policy",
        choices=("self", "random"),
        default="self",
        help="Player0 policy. 'self' uses sample_submission/main.py; 'random' picks random legal actions.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for the random opponent.")
    parser.add_argument("--max-steps", type=int, default=400, help="Safety cap for turns/actions.")
    parser.add_argument(
        "--player-deck",
        type=Path,
        default=None,
        help="Deck CSV for player0. Defaults to sample_submission/deck.csv.",
    )
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        default=None,
        help="Deck CSV for player1. Defaults to sample_submission/deck.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"replay-{timestamp}-{args.player_policy}-vs-{args.opponent}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # レビュー用途なので意思決定理由（B層）を収集する。提出パスでは呼ばれないため影響なし。
    # sample_submission/src が無いブランチでは trace は None（機能自体を無効化する）。
    if trace is not None:
        trace.enable_trace()
        trace.pop()  # 念のため直前の残りをクリア。

    player_deck_path = resolve_deck_path(args.player_deck)
    opponent_deck_path = resolve_deck_path(args.opponent_deck)

    with working_directory(SAMPLE_SUBMISSION_DIR):
        deck0 = read_deck_csv_file(player_deck_path)
        deck1 = read_deck_csv_file(opponent_deck_path)
        player0 = agent_for_policy(args.player_policy, deck0)
        player1 = agent_for_policy(args.opponent, deck1)
        replay = run_match(player0, player1, deck0, deck1, max_steps=args.max_steps)

    payload = {
        "metadata": {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "playerPolicy": args.player_policy,
            "opponent": args.opponent,
            "seed": args.seed,
            "deckPath": display_deck_path(player_deck_path),
            "playerDeckPath": display_deck_path(player_deck_path),
            "opponentDeckPath": display_deck_path(opponent_deck_path),
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
