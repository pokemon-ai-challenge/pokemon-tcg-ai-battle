from pathlib import Path
import sys

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Card, Log, LogType, PlayerState, Pokemon, State
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge

# Real card IDs from data/*, picked so is_pokemon/is_energy/is_tool resolve meaningfully.
SCRAFTY_A = 21   # "Scrafty" (reprint 1)
SCRAFTY_B = 611  # "Scrafty" (reprint 2, same name, different card_id)
ENERGY_CARD = 1  # "Basic {G} Energy"
TOOL_CARD = 1154  # a Pokemon Tool card
ITEM_CARD = 2  # any non-Pokemon card id used as a stand-in for a played Item/Supporter


def make_pokemon(card_id, serial, energy_cards=None, tools=None, pre_evolution=None, energies=None):
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=60,
        maxHp=60,
        appearThisTurn=False,
        energies=energies or [],
        energyCards=energy_cards or [],
        tools=tools or [],
        preEvolution=pre_evolution or [],
    )


def make_player_state(active=None, bench=None, discard=None, hand=None):
    return PlayerState(
        active=active or [],
        bench=bench or [],
        benchMax=5,
        deckCount=50,
        discard=discard or [],
        prize=[None] * 6,
        handCount=5,
        hand=hand,
        poisoned=False,
        burned=False,
        asleep=False,
        paralyzed=False,
        confused=False,
    )


def make_state(your_player, opponent_player, your_index=0, turn=1, stadium=None, looking=None):
    players = [None, None]
    players[your_index] = your_player
    players[1 - your_index] = opponent_player
    return State(
        turn=turn,
        turnActionCount=0,
        yourIndex=your_index,
        firstPlayer=your_index,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        result=-1,
        stadium=stadium or [],
        looking=looking,
        players=players,
    )


def test_active_bench_discard_are_observed():
    opponent = make_player_state(
        active=[make_pokemon(SCRAFTY_A, serial=1)],
        bench=[make_pokemon(SCRAFTY_B, serial=2)],
        discard=[Card(id=ENERGY_CARD, serial=3, playerIndex=1)],
    )
    knowledge = OpponentKnowledge()
    knowledge.update_from_state(make_state(make_player_state(), opponent))

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"] == {SCRAFTY_A: 1, SCRAFTY_B: 1, ENERGY_CARD: 1}
    assert features["observed_cards"]["Scrafty"] == 2
    assert features["name_to_card_ids"]["Scrafty"] == {SCRAFTY_A: 1, SCRAFTY_B: 1}
    assert features["zone_cards"]["active"] == ["Scrafty"]
    assert features["zone_cards"]["bench"] == ["Scrafty"]
    assert "Basic {G} Energy" in features["zone_cards"]["discard"]


def test_same_card_id_different_serials_counts_twice_but_reobservation_does_not_inflate():
    opponent = make_player_state(bench=[
        make_pokemon(SCRAFTY_A, serial=1),
        make_pokemon(SCRAFTY_A, serial=2),
    ])
    knowledge = OpponentKnowledge()
    state = make_state(make_player_state(), opponent)

    knowledge.update_from_state(state)
    knowledge.update_from_state(state)  # re-observing the same snapshot must not double count

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"][SCRAFTY_A] == 2


def test_attached_energy_and_tool_are_classified():
    opponent = make_player_state(active=[
        make_pokemon(
            SCRAFTY_A,
            serial=1,
            energy_cards=[Card(id=ENERGY_CARD, serial=2, playerIndex=1)],
            tools=[Card(id=TOOL_CARD, serial=3, playerIndex=1)],
        )
    ])
    knowledge = OpponentKnowledge()
    knowledge.update_from_state(make_state(make_player_state(), opponent))

    features = knowledge.get_prediction_features()
    assert "Basic {G} Energy" in features["observed_energies"]
    assert features["zone_cards"]["energy"] == ["Basic {G} Energy"]
    assert features["zone_cards"]["tool"][0] != ""  # tool name resolved
    assert features["observed_tools"] == features["zone_cards"]["tool"]


def test_card_moving_zones_tracked_by_serial_without_double_counting():
    knowledge = OpponentKnowledge()

    turn1_opponent = make_player_state(active=[make_pokemon(SCRAFTY_A, serial=1)])
    knowledge.update_from_state(make_state(make_player_state(), turn1_opponent, turn=1))

    turn2_opponent = make_player_state(discard=[Card(id=SCRAFTY_A, serial=1, playerIndex=1)])
    knowledge.update_from_state(make_state(make_player_state(), turn2_opponent, turn=2))

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"][SCRAFTY_A] == 1
    assert "active" not in features["zone_cards"]
    assert features["zone_cards"]["discard"] == ["Scrafty"]


def test_logs_reveal_ephemeral_card_not_present_on_board():
    knowledge = OpponentKnowledge(opponent_index=1)
    log = Log(type=LogType.PLAY, playerIndex=1, cardId=ITEM_CARD, serial=42)
    knowledge.update_from_logs([log])

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"][ITEM_CARD] == 1


def test_facedown_and_unrevealed_events_are_not_recorded():
    opponent = make_player_state(active=[None])
    knowledge = OpponentKnowledge()
    knowledge.update_from_state(make_state(make_player_state(), opponent, looking=[None]))

    reverse_log = Log(type=LogType.MOVE_CARD_REVERSE, playerIndex=1, fromArea=None, toArea=None)
    knowledge.update_from_logs([reverse_log])

    assert knowledge.get_observed_cards() == []


def test_own_side_events_are_ignored():
    knowledge = OpponentKnowledge(opponent_index=1)
    own_log = Log(type=LogType.PLAY, playerIndex=0, cardId=ITEM_CARD, serial=99)
    knowledge.update_from_logs([own_log])

    assert knowledge.get_observed_cards() == []


def test_switch_log_maps_active_and_bench_to_their_destination_not_their_source():
    # cg engine field naming is the reverse of what it looks like: cardIdActive/serialActive
    # identify the card that WAS active and is now moving to the bench, while
    # cardIdBench/serialBench identify the card that WAS benched and is now moving to active
    # (verified against real engine replay data -- see opponent_knowledge.py's _handle_switch).
    knowledge = OpponentKnowledge(opponent_index=1)
    log = Log(
        type=LogType.SWITCH,
        playerIndex=1,
        cardIdActive=SCRAFTY_A,
        serialActive=1,
        cardIdBench=SCRAFTY_B,
        serialBench=2,
    )
    knowledge.update_from_logs([log])

    zone_cards = knowledge.get_prediction_features()["zone_cards"]
    assert zone_cards["bench"] == ["Scrafty"]  # the (formerly active) SCRAFTY_A/serial=1 card
    assert zone_cards["active"] == ["Scrafty"]  # the (formerly benched) SCRAFTY_B/serial=2 card
    records = {record.serial: record for record in knowledge.get_observed_cards()}
    assert records[1].current_zone == "bench"
    assert records[2].current_zone == "active"


def test_update_from_state_after_update_from_logs_wins_the_current_zone():
    # Required call order is update_from_logs() then update_from_state(): logs carry provisional
    # zone tags (e.g. "revealed" for an evolve/switch event), and the full board scan that follows
    # must be the one that settles where the card actually ended up. Calling them in the opposite
    # order would let the provisional log-derived zone silently overwrite the correct one.
    knowledge = OpponentKnowledge(opponent_index=1)
    evolve_log = Log(type=LogType.EVOLVE, playerIndex=1, cardId=SCRAFTY_A, serial=1)
    knowledge.update_from_logs([evolve_log])
    assert knowledge.get_observed_cards()[0].current_zone == "revealed"

    opponent = make_player_state(bench=[make_pokemon(SCRAFTY_A, serial=1)])
    knowledge.update_from_state(make_state(make_player_state(), opponent))

    assert knowledge.get_zone_cards() == {"bench": [knowledge.get_observed_cards()[0]]}


def test_logs_before_first_state_are_replayed_after_opponent_index_is_known():
    knowledge = OpponentKnowledge()
    log = Log(type=LogType.PLAY, playerIndex=1, cardId=ITEM_CARD, serial=42)

    knowledge.update_from_logs([log])
    assert knowledge.get_observed_cards() == []

    knowledge.update_from_state(make_state(make_player_state(), make_player_state()))

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"][ITEM_CARD] == 1


def test_no_serial_card_moving_zones_reuses_existing_record():
    knowledge = OpponentKnowledge(opponent_index=1)

    knowledge.observe_card(ITEM_CARD, "revealed", turn=1, serial=None)
    knowledge.update_from_state(make_state(make_player_state(), make_player_state(), turn=2))
    knowledge.observe_card(ITEM_CARD, "discard", turn=2, serial=None)

    records = knowledge.get_observed_cards()
    assert len(records) == 1
    assert records[0].zones_seen == {"revealed", "discard"}
    assert records[0].current_zone == "discard"

    features = knowledge.get_prediction_features()
    assert features["observed_card_ids"][ITEM_CARD] == 1
    assert features["zone_cards"]["discard"] == [records[0].name]
