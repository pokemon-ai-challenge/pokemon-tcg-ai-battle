"""Game execution for the viewer.

Wraps the native cg engine so the HTTP layer never touches it directly. Because
the engine is a single global instance (one Battle pointer, one search agent
pointer), ALL engine access is serialised through ``ENGINE_LOCK`` and only one
session is active at a time.

Two entry points:
  * run_match(...)       -- run a full AI vs AI game and return a god-view replay
  * HumanSession(...)    -- a live, interactive Human vs AI game
"""

from __future__ import annotations

import json
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SAMPLE = os.path.join(_REPO, "sample_submission")
if _SAMPLE not in sys.path:
    sys.path.insert(0, _SAMPLE)

from cg.api import to_observation_class  # noqa: E402
from cg.game import (  # noqa: E402
    battle_finish,
    battle_select,
    battle_start,
    visualize_data,
)

import agents as agents_mod  # noqa: E402
import decks as decks_mod  # noqa: E402
import describe as describe_mod  # noqa: E402

ENGINE_LOCK = threading.RLock()
_MAX_STEPS = 5000

# Track whether a battle is currently open so we can clean up before a new one.
_battle_open = False


def _finish_open_battle():
    global _battle_open
    if _battle_open:
        try:
            battle_finish()
        except Exception:
            pass
        _battle_open = False


def _agent_label(agent_id: str) -> str:
    for a in agents_mod.list_agents():
        if a["id"] == agent_id:
            return a["label"]
    return agent_id


def _acting_index(obs) -> int:
    return obs.current.yourIndex if obs.current is not None else 0


# --------------------------------------------------------------------------- #
# AI vs AI
# --------------------------------------------------------------------------- #
def run_match(ai0: str, deck0: str, ai1: str, deck1: str) -> dict:
    """Play a full AI vs AI game; return {labels, replay, result}."""
    global _battle_open
    d0 = decks_mod.load_deck(deck0)
    d1 = decks_mod.load_deck(deck1)
    a0 = agents_mod.build_agent(ai0)
    a1 = agents_mod.build_agent(ai1)
    labels = [_agent_label(ai0), _agent_label(ai1)]

    with ENGINE_LOCK:
        _finish_open_battle()
        obs_dict, start = battle_start(d0, d1)
        if start.errorType != 0:
            raise RuntimeError(f"battle_start に失敗しました (errorType={start.errorType})")
        _battle_open = True
        try:
            steps = 0
            while steps < _MAX_STEPS:
                obs = to_observation_class(obs_dict)
                if obs.current is not None and obs.current.result != -1:
                    break
                actor = _acting_index(obs)
                action = (a0 if actor == 0 else a1)(obs_dict)
                obs_dict = battle_select(action)
                steps += 1

            snapshots = json.loads(visualize_data())
        finally:
            _finish_open_battle()

    replay = [
        describe_mod.build_view(snap, labels, god_view=True, bottom_index=0)
        for snap in snapshots
    ]
    result = -1
    for view in reversed(replay):
        if view["result"] != -1:
            result = view["result"]
            break
    return {
        "labels": labels,
        "ai": [ai0, ai1],
        "deck": [deck0, deck1],
        "replay": replay,
        "result": result,
    }


# --------------------------------------------------------------------------- #
# Human vs AI (live)
# --------------------------------------------------------------------------- #
class HumanSession:
    """One live game where ``human_index`` is the human and the other side is AI."""

    def __init__(self, human_index: int, ai_id: str, human_deck: str, ai_deck: str):
        self.human_index = human_index
        self.ai_index = 1 - human_index
        self.ai_id = ai_id
        self.ai = agents_mod.build_agent(ai_id)
        self.labels = ["", ""]
        self.labels[human_index] = "あなた"
        self.labels[self.ai_index] = _agent_label(ai_id)

        decks_by_index = [None, None]
        decks_by_index[human_index] = decks_mod.load_deck(human_deck)
        decks_by_index[self.ai_index] = decks_mod.load_deck(ai_deck)

        self.obs_dict = None
        self.finished = False
        self.result = -1

        global _battle_open
        with ENGINE_LOCK:
            _finish_open_battle()
            obs_dict, start = battle_start(decks_by_index[0], decks_by_index[1])
            if start.errorType != 0:
                raise RuntimeError(f"battle_start に失敗しました (errorType={start.errorType})")
            _battle_open = True
            self.obs_dict = obs_dict
            self._ai_moves = self._advance_until_human()

    # -- internal -------------------------------------------------------- #
    def _advance_until_human(self) -> list:
        """Run AI selections until it is the human's turn or the game ends.

        Returns a list of the AI's decisions made along the way:
        [{context, options:[{idx,label,kind,selected}]}].
        """
        ai_moves = []
        steps = 0
        while steps < _MAX_STEPS:
            obs = to_observation_class(self.obs_dict)
            if obs.current is not None and obs.current.result != -1:
                self.finished = True
                self.result = obs.current.result
                return ai_moves
            actor = _acting_index(obs)
            if actor == self.human_index:
                return ai_moves  # human's turn
            # AI's turn: decide, record, apply.
            action = self.ai(self.obs_dict)
            ai_moves.append(self._record_ai_move(self.obs_dict, action))
            self.obs_dict = battle_select(action)
            steps += 1
        return ai_moves

    def _record_ai_move(self, obs_dict: dict, action: list) -> dict:
        current = obs_dict.get("current") or {}
        players = current.get("players") or []
        select = obs_dict.get("select") or {}
        actor = current.get("yourIndex", self.ai_index)
        options = describe_mod._options_block(select, players, actor, action)
        sel_block = describe_mod._select_block(select)
        return {
            "turn": current.get("turn", 0),
            "context": sel_block["contextLabel"] if sel_block else "",
            "options": options,
        }

    def _build_state(self, ai_moves: list) -> dict:
        snap = {
            "current": self.obs_dict.get("current"),
            "select": None if self.finished else self.obs_dict.get("select"),
            "logs": self.obs_dict.get("logs"),
            "selected": None,
        }
        view = describe_mod.build_view(
            snap, self.labels, god_view=False,
            bottom_index=self.human_index, human_index=self.human_index,
        )
        view["humanIndex"] = self.human_index
        view["aiLabel"] = self.labels[self.ai_index]
        view["aiMoves"] = ai_moves
        view["finished"] = self.finished
        view["yourTurn"] = (not self.finished and
                            view["actingIndex"] == self.human_index)
        return view

    # -- public ---------------------------------------------------------- #
    def state(self) -> dict:
        with ENGINE_LOCK:
            return self._build_state(self._ai_moves)

    def act(self, action: list) -> dict:
        with ENGINE_LOCK:
            if self.finished:
                return self._build_state([])
            self.obs_dict = battle_select(action)
            ai_moves = self._advance_until_human()
            return self._build_state(ai_moves)

    def close(self):
        with ENGINE_LOCK:
            _finish_open_battle()


# Single active human session (engine is single-instance).
_HUMAN_SESSION: HumanSession | None = None


def new_human_session(human_index: int, ai_id: str, human_deck: str, ai_deck: str) -> dict:
    global _HUMAN_SESSION
    with ENGINE_LOCK:
        if _HUMAN_SESSION is not None:
            _HUMAN_SESSION.close()
        _HUMAN_SESSION = HumanSession(human_index, ai_id, human_deck, ai_deck)
        return _HUMAN_SESSION.state()


def human_act(action: list) -> dict:
    if _HUMAN_SESSION is None:
        raise RuntimeError("対戦が開始されていません")
    return _HUMAN_SESSION.act(action)


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ai0", default="random")
    parser.add_argument("--ai1", default="random")
    args = parser.parse_args()

    if args.smoke:
        decks = decks_mod.list_decks()
        deck_id = decks[0]["id"]
        out = run_match(args.ai0, deck_id, args.ai1, deck_id)
        print(f"labels={out['labels']} replay_len={len(out['replay'])} result={out['result']}")
        last = out["replay"][-1]
        print(f"last turn={last['turn']} result={last['result']} "
              f"P0 prize={last['players'][0]['prizeTotal']} "
              f"P1 prize={last['players'][1]['prizeTotal']}")
