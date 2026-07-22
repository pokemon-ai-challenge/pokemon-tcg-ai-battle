from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_SUBMISSION_DIR = ROOT_DIR / "sample_submission"

for candidate in (str(ROOT_DIR), str(SAMPLE_SUBMISSION_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402
except Exception:  # noqa: BLE001 -- keep live mode usable without the predictor branch
    try:
        from sample_submission.ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge  # noqa: E402
    except Exception:  # noqa: BLE001
        OpponentKnowledge = None
try:
    from .opponent_knowledge_diff import (  # noqa: E402
        collect_ground_truth,
        diff_against_ground_truth,
    )
except ImportError:  # noqa: BLE001 -- スクリプト実行時のフォールバック
    from opponent_knowledge_diff import (  # noqa: E402
        collect_ground_truth,
        diff_against_ground_truth,
    )
try:
    from .ml_prediction_debug import build_ml_prediction_debug  # noqa: E402
except ImportError:  # noqa: BLE001 -- スクリプト実行時のフォールバック
    from ml_prediction_debug import build_ml_prediction_debug  # noqa: E402
try:
    from ptcg_ai.opponent_modeling.rough_predictor import predict as predict_deck  # noqa: E402
except Exception:  # noqa: BLE001 -- 予測器が無い/壊れていてもライブモードは続行する
    predict_deck = None
try:
    from ptcg_ai.opponent_modeling.hybrid_predictor import HybridDeckPredictor  # noqa: E402

    _ml_predictor = HybridDeckPredictor()
except Exception:  # noqa: BLE001 -- ML予測器が無い/壊れていてもライブモードは続行する
    _ml_predictor = None

from cg.api import Observation, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
try:
    from src.agent import choose_action_from_observation_dict  # noqa: E402
except ImportError:
    # sample_submission/src はこのブランチにはまだ無い（B層の意思決定ランタイムは
    # 別ブランチ由来の機能）。無ければ提出物そのものの main.agent にフォールバックする。
    choose_action_from_observation_dict = None

try:
    from .viewer_state import build_frame_snapshot, current_visual_frame, validate_action
except ImportError:
    from viewer_state import build_frame_snapshot, current_visual_frame, validate_action


AgentFn = Callable[[dict], list[int]]
SELF_INDEX = 0
OPPONENT_INDEX = 1


def resolve_deck_path(value: str | Path | None = None) -> Path:
    if value is None or str(value).strip() == "":
        return SAMPLE_SUBMISSION_DIR / "deck.csv"
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def read_deck_csv_file(path: str | Path | None = None) -> list[int]:
    text = resolve_deck_path(path).read_text(encoding="utf-8")
    deck: list[int] = []
    for raw_value in text.replace(",", "\n").splitlines():
        value = raw_value.strip()
        if not value or value.startswith("#"):
            continue
        deck.append(int(value))
    if len(deck) != 60:
        raise ValueError(f"Deck must contain exactly 60 card IDs, got {len(deck)}.")
    return deck


def make_cpu_self_agent(deck: list[int]) -> AgentFn:
    def wrapped(obs_dict: dict) -> list[int]:
        if choose_action_from_observation_dict is not None:
            return choose_action_from_observation_dict(obs_dict, lambda: list(deck))
        from main import agent  # noqa: PLC0415 -- sample_submission is on sys.path via SAMPLE_SUBMISSION_DIR
        return agent(obs_dict)

    return wrapped


def make_random_agent(deck: list[int]) -> AgentFn:
    def wrapped(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        return random.sample(range(len(obs.select.option)), obs.select.maxCount)

    return wrapped


@dataclass
class LiveMatchSession:
    lock: Lock = field(default_factory=Lock)
    active: bool = False
    obs_dict: dict[str, Any] | None = None
    cpu_policy: str = "self"
    cpu_agent: AgentFn | None = None
    seed: int = 7
    player_deck_path: str = "sample_submission/deck.csv"
    opponent_deck_path: str = "sample_submission/deck.csv"
    deck0: list[int] = field(default_factory=list)
    deck1: list[int] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    decision_frames: list[dict[str, Any]] = field(default_factory=list)
    opponent_knowledge: Any | None = None

    def start(
        self,
        cpu_policy: str = "self",
        seed: int = 7,
        player_deck: str | None = None,
        opponent_deck: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self._stop_locked()
            self.cpu_policy = cpu_policy if cpu_policy in {"self", "random"} else "self"
            self.seed = seed
            self.player_deck_path = player_deck or "sample_submission/deck.csv"
            self.opponent_deck_path = opponent_deck or "sample_submission/deck.csv"
            self.deck0 = read_deck_csv_file(self.player_deck_path)
            self.deck1 = read_deck_csv_file(self.opponent_deck_path)
            self.cpu_agent = make_cpu_self_agent(self.deck1) if self.cpu_policy == "self" else make_random_agent(self.deck1)
            self.action_history = []
            self.decision_frames = []
            self.opponent_knowledge = (
                None if OpponentKnowledge is None else OpponentKnowledge(opponent_index=OPPONENT_INDEX)
            )
            self._start_battle_locked()
            self._advance_cpu_locked()
            return self._snapshot_locked()

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def submit_action(self, action: list[int]) -> dict[str, Any]:
        with self.lock:
            if not self.active or self.obs_dict is None:
                raise RuntimeError("No active live match.")

            obs = to_observation_class(self.obs_dict)
            if obs.current is None or obs.current.result != -1:
                raise RuntimeError("The live match is already finished.")
            if obs.current.yourIndex != SELF_INDEX:
                raise RuntimeError("It is not the human player's turn.")
            if obs.select is None:
                raise RuntimeError("No selectable action is available.")

            validate_action(obs, action)
            self._apply_action_locked(action, SELF_INDEX)
            self._advance_cpu_locked()
            return self._snapshot_locked()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self._stop_locked()
            return self._inactive_payload()

    def undo_last_human_turn(self) -> dict[str, Any]:
        raise RuntimeError("Undo is disabled because the game engine cannot restore live battle state safely.")

    def _advance_cpu_locked(self) -> None:
        while self.active and self.obs_dict is not None:
            obs = to_observation_class(self.obs_dict)
            if obs.current is None or obs.current.result != -1:
                return
            if obs.current.yourIndex != OPPONENT_INDEX:
                return
            if self.cpu_agent is None:
                raise RuntimeError("CPU agent is not initialized.")
            action = self.cpu_agent(self.obs_dict)
            validate_action(obs, action)
            self._apply_action_locked(action, OPPONENT_INDEX)

    def _build_opponent_knowledge_debug_locked(self) -> dict[str, Any] | None:
        if self.obs_dict is None:
            return None
        if OpponentKnowledge is None:
            return None
        obs = to_observation_class(self.obs_dict)
        if obs.current is None or obs.current.yourIndex != SELF_INDEX:
            return None
        if self.opponent_knowledge is None:
            self.opponent_knowledge = OpponentKnowledge(opponent_index=OPPONENT_INDEX)
        self.opponent_knowledge.update_from_logs(obs.logs)
        self.opponent_knowledge.update_from_state(obs.current)

        visual_frame = current_visual_frame()
        current = visual_frame.get("current")
        revealed_active = False
        if obs.current is not None:
            opponent_state = obs.current.players[OPPONENT_INDEX]
            revealed_active = bool(opponent_state.active and opponent_state.active[0] is not None)
        ground_truth = (
            {}
            if current is None
            else collect_ground_truth(current, OPPONENT_INDEX, revealed_active=revealed_active)
        )
        diff = diff_against_ground_truth(self.opponent_knowledge.get_observed_cards(), ground_truth)

        prediction = None
        if predict_deck is not None and obs.current is not None:
            try:
                prediction = predict_deck(obs.current, self.opponent_knowledge)
            except Exception as exc:  # noqa: BLE001 -- 予測器が落ちてもライブモードは止めない
                prediction = {"error": str(exc)}

        ml_prediction = build_ml_prediction_debug(_ml_predictor, self.opponent_knowledge, obs.current)

        return {
            "features": self.opponent_knowledge.get_prediction_features(),
            "diff": diff,
            "prediction": prediction,
            "ml_prediction": ml_prediction,
        }

    def _apply_action_locked(self, action: list[int], player_index: int) -> None:
        debug_payload = self._build_opponent_knowledge_debug_locked()
        self.decision_frames.append(
            build_frame_snapshot(
                len(self.decision_frames),
                current_visual_frame(),
                list(action),
                opponent_knowledge_debug=debug_payload,
            )
        )
        self.obs_dict = battle_select(action)
        self.action_history.append(
            {
                "player": player_index,
                "action": list(action),
            }
        )

    def _start_battle_locked(self) -> None:
        random.seed(self.seed)
        obs_dict, start_data = battle_start(self.deck0, self.deck1)
        if start_data.errorType != 0:
            self.active = False
            self.obs_dict = None
            raise RuntimeError(f"battle_start failed with errorType={start_data.errorType}")
        self.active = True
        self.obs_dict = obs_dict

    def _restart_from_history_locked(self) -> None:
        if self.active:
            battle_finish()
        self._start_battle_locked()
        replay_history = list(self.action_history)
        self.action_history = []
        self.decision_frames = []
        for item in replay_history:
            self._apply_action_locked(list(item["action"]), int(item["player"]))

    def _snapshot_locked(self) -> dict[str, Any]:
        if not self.active or self.obs_dict is None:
            return self._inactive_payload()

        obs = to_observation_class(self.obs_dict)
        debug_payload = self._build_opponent_knowledge_debug_locked()
        current_frame = build_frame_snapshot(
            len(self.decision_frames),
            current_visual_frame(),
            action=[],
            opponent_knowledge_debug=debug_payload,
        )
        result = None
        acting_player = None
        min_count = 0
        max_count = 0
        option_count = 0
        if obs.current is not None:
            acting_player = obs.current.yourIndex
            if obs.current.result != -1:
                result = obs.current.result
        if obs.select is not None:
            min_count = obs.select.minCount
            max_count = obs.select.maxCount
            option_count = len(obs.select.option)

        human_turn = result is None and acting_player == SELF_INDEX and obs.select is not None
        status = "finished" if result is not None else ("waiting_human" if human_turn else "waiting_cpu")
        status_text = {
            "finished": "Match finished",
            "waiting_human": "Your turn",
            "waiting_cpu": "CPU is thinking",
        }[status]

        return {
            "active": True,
            "metadata": {
                "opponent": f"live-{self.cpu_policy}",
                "playerDeckPath": self.player_deck_path,
                "opponentDeckPath": self.opponent_deck_path,
                "result": result,
                "steps": len(self.action_history),
                "players": {
                    0: {"name": "Human"},
                    1: {"name": f"CPU ({self.cpu_policy})"},
                },
                "modeLabel": "live",
            },
            "frames": [*self.decision_frames, current_frame],
            "live": {
                "active": True,
                "status": status,
                "statusText": status_text,
                "humanTurn": human_turn,
                "actingPlayer": acting_player,
                "minCount": min_count,
                "maxCount": max_count,
                "optionCount": option_count,
                "cpuPolicy": self.cpu_policy,
                "playerDeck": self.player_deck_path,
                "opponentDeck": self.opponent_deck_path,
                "result": result,
                "latestFrameIndex": len(self.decision_frames),
                "canUndo": False,
            },
        }

    def _inactive_payload(self) -> dict[str, Any]:
        return {
            "active": False,
            "metadata": {
                "opponent": "live",
                "playerDeckPath": self.player_deck_path,
                "opponentDeckPath": self.opponent_deck_path,
                "result": None,
                "players": {
                    0: {"name": "Human"},
                    1: {"name": "CPU"},
                },
                "modeLabel": "live",
            },
            "frames": [],
            "live": {
                "active": False,
                "status": "inactive",
                "statusText": "No active live match",
                "humanTurn": False,
                "actingPlayer": None,
                "minCount": 0,
                "maxCount": 0,
                "optionCount": 0,
                "cpuPolicy": self.cpu_policy,
                "playerDeck": self.player_deck_path,
                "opponentDeck": self.opponent_deck_path,
                "result": None,
                "latestFrameIndex": 0,
                "canUndo": False,
            },
        }

    def _stop_locked(self) -> None:
        if self.active:
            battle_finish()
        self.active = False
        self.obs_dict = None
        self.action_history = []
        self.decision_frames = []
