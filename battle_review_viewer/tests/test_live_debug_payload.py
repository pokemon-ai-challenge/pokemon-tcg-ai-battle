from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAMPLE_SUBMISSION = ROOT.parent / "sample_submission"
if str(SAMPLE_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION))

from cg.game import battle_finish, battle_start  # noqa: E402
from live_match import LiveMatchSession, read_deck_csv_file  # noqa: E402


def test_live_snapshot_includes_debug_payload_when_available() -> None:
    session = LiveMatchSession()
    session.player_deck_path = str(Path(__file__).resolve().parents[1] / ".." / "sample_submission" / "deck.csv")
    session.opponent_deck_path = str(Path(__file__).resolve().parents[1] / ".." / "sample_submission" / "deck.csv")
    session.deck0 = read_deck_csv_file(session.player_deck_path)
    session.deck1 = read_deck_csv_file(session.opponent_deck_path)
    session.opponent_knowledge = None
    obs_dict, start_data = battle_start(session.deck0, session.deck1)
    try:
        assert start_data.errorType == 0
        session.active = True
        session.obs_dict = obs_dict
        snapshot = session._snapshot_locked()
        assert snapshot["frames"]
        frame = snapshot["frames"][-1]
        assert "opponentKnowledgeDebug" in frame
        assert frame["opponentKnowledgeDebug"] is None or isinstance(frame["opponentKnowledgeDebug"], dict)
    finally:
        battle_finish()
