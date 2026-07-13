from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Card, PlayerState, Pokemon, State
from ptcg_ai.opponent_modeling.opponent_knowledge import OpponentKnowledge
from ptcg_ai.opponent_modeling.rough_predictor import predict
from ptcg_ai.opponent_modeling import rough_predictor


MEGA_LUCARIO_EX = 678
RIOLU = 333
LUCARIO = 677
ALAKAZAM = 245
ABRA = 109
KADABRA = 742
DRAGAPULT_EX = 121
DREEPY = 119
DRAKLOAK = 120
ROCKET_HONCHKROW = 891
ROCKET_MURKROW = 463
HYDRAPPLE_EX = 150
APPLIN = 93
DIPPLIN = 149
GENERIC_ITEM = 1080


def make_pokemon(card_id, serial, energies=None, energy_cards=None, tools=None, pre_evolution=None):
    return Pokemon(
        id=card_id,
        serial=serial,
        hp=100,
        maxHp=100,
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


def make_state(opponent_player, your_player=None, your_index=0, turn=1):
    your_player = your_player or make_player_state()
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
        stadium=[],
        looking=None,
        players=players,
    )


def make_test_config():
    return {
        "role_weights": {
            "anchor": 10,
            "evolution_line": 5,
            "core": 4,
            "flex": 2,
            "energy": 2,
            "generic": 0,
        },
        "ace_spec_bonus": 1.3,
        "role_reason_templates": {},
        "generic_cards": [],
        "prediction_decision": {
            "min_evidence_count": 2,
            "min_top_normalized_score": 1.0,
            "min_normalized_margin": 0.2,
        },
        "archetypes": {
            "mega_lucario_ex": {
                "display_name": "Mega Lucario",
                "cards": [],
                "role_cards": {
                    "evolution_line": [{"name": "riolu", "card_id": [1]}],
                    "core": [
                        {"name": "solrock", "card_id": [2]},
                        {"name": "power_protein", "card_id": [3]},
                        {"name": "makuhita", "card_id": [4]},
                    ],
                    "flex": [
                        {"name": "fight_gong", "card_id": [5]},
                        {"name": "lucario", "card_id": [6]},
                    ],
                },
                "energy_types": [],
                "combo_rules": [],
                "variants": {},
                "confident_score": 20,
            },
            "shirona_garchomp_ex": {
                "display_name": "Shirona Garchomp",
                "cards": [],
                "role_cards": {
                    "core": [{"name": "power_protein", "card_id": [3]}],
                },
                "energy_types": [],
                "combo_rules": [],
                "variants": {},
                "confident_score": 8.5,
            },
            "ambiguous_rival": {
                "display_name": "Ambiguous Rival",
                "cards": [],
                "role_cards": {
                    "core": [
                        {"name": "solrock", "card_id": [2]},
                        {"name": "makuhita", "card_id": [4]},
                    ],
                    "flex": [{"name": "fight_gong", "card_id": [5]}],
                },
                "energy_types": [],
                "combo_rules": [],
                "variants": {},
                "confident_score": 10.5,
            },
            "single_evidence": {
                "display_name": "Single Evidence",
                "cards": [{"name": "solo_ace", "card_id": [7], "role": "anchor"}],
                "role_cards": {},
                "energy_types": [],
                "combo_rules": [],
                "variants": {},
                "confident_score": 8,
            },
        },
    }


def make_features(
    names: list[str],
    name_to_id: dict[str, int] | None = None,
    zone: str | None = "revealed",
) -> dict:
    name_to_id = name_to_id or {}
    observed_card_ids = {}
    name_to_card_ids = {}
    for name in names:
        card_id = name_to_id.get(name, len(name_to_id) + len(observed_card_ids) + 1)
        observed_card_ids[card_id] = observed_card_ids.get(card_id, 0) + 1
        name_to_card_ids.setdefault(name, {})
        name_to_card_ids[name][card_id] = observed_card_ids[card_id]
    return {
        "observed_card_ids": observed_card_ids,
        "observed_cards": {name: names.count(name) for name in set(names)},
        "name_to_card_ids": name_to_card_ids,
        "zone_cards": {zone: names} if zone is not None else {},
        "observed_pokemon": [],
        "observed_energies": [],
        "observed_tools": [],
        "energy_types": [],
    }


def test_predict_mega_lucario_ex_from_public_board():
    opponent = make_player_state(
        active=[make_pokemon(MEGA_LUCARIO_EX, serial=1)],
        bench=[make_pokemon(RIOLU, serial=2), make_pokemon(LUCARIO, serial=3)],
    )
    result = predict(make_state(opponent))

    assert result["deck_type"] == "mega_lucario_ex"
    assert result["display_name"] == "メガルカリオex"
    assert result["score"] > 0
    assert any(item["card"] == "メガルカリオex" for item in result["evidence"])
    assert result["status"] == "confident"
    assert result["match_rate"] == 1.0


def test_predict_alakazam_from_evolution_line():
    opponent = make_player_state(
        active=[make_pokemon(ALAKAZAM, serial=11)],
        bench=[make_pokemon(ABRA, serial=12), make_pokemon(KADABRA, serial=13)],
    )
    result = predict(make_state(opponent))

    assert result["deck_type"] == "alakazam"
    assert result["display_name"] == "フーディン"


def test_predict_dragapult_ex_from_public_board():
    opponent = make_player_state(
        active=[make_pokemon(DRAGAPULT_EX, serial=21)],
        bench=[make_pokemon(DREEPY, serial=22), make_pokemon(DRAKLOAK, serial=23)],
    )
    result = predict(make_state(opponent))

    assert result["deck_type"] == "dragapult_ex"
    assert result["display_name"] == "ドラパルトex"


def test_predict_rocket_honchkrow_from_public_board():
    opponent = make_player_state(
        active=[make_pokemon(ROCKET_HONCHKROW, serial=31)],
        bench=[make_pokemon(ROCKET_MURKROW, serial=32)],
    )
    result = predict(make_state(opponent))

    assert result["deck_type"] == "rocket_honchkrow"
    assert result["display_name"] == "ロケット団のドンカラス"


def test_predict_kamitsuorochi_from_public_board():
    opponent = make_player_state(
        active=[make_pokemon(HYDRAPPLE_EX, serial=41)],
        bench=[make_pokemon(APPLIN, serial=42), make_pokemon(DIPPLIN, serial=43)],
    )
    result = predict(make_state(opponent))

    assert result["deck_type"] == "kamitsuorochi_ex"
    assert result["display_name"] == "カミツオロチex"


def test_predict_returns_unknown_for_generic_information_only():
    opponent = make_player_state(discard=[Card(id=GENERIC_ITEM, serial=99, playerIndex=1)])
    result = predict(make_state(opponent))

    assert result["deck_type"] == "unknown"
    assert result["display_name"] == "unknown"
    assert result["status"] == "no_candidate"
    assert result["top_candidate"] is None
    assert result["match_rate"] == 0.0


def test_predict_uses_opponent_knowledge_for_seen_but_not_current_cards():
    knowledge = OpponentKnowledge(opponent_index=1)
    knowledge.observe_card(ROCKET_HONCHKROW, "revealed", turn=1, serial=500)
    knowledge.observe_card(ROCKET_MURKROW, "revealed", turn=1, serial=501)

    opponent = make_player_state(discard=[Card(id=GENERIC_ITEM, serial=600, playerIndex=1)])
    result = predict(make_state(opponent), opponent_knowledge=knowledge)

    assert result["deck_type"] == "rocket_honchkrow"
    assert result["candidates"][0]["deck_type"] == "rocket_honchkrow"


def test_score_prediction_returns_unknown_when_top_is_clear_but_below_threshold():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    features = make_features(
        ["riolu", "solrock", "power_protein", "makuhita", "fight_gong"],
        {"riolu": 1, "solrock": 2, "power_protein": 3, "makuhita": 4, "fight_gong": 5},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)

    assert result["top_candidate"] == "Mega Lucario"
    assert result["deck_type"] == "unknown"
    assert result["status"] == "insufficient_evidence"
    assert result["score"] == 19
    assert result["confident_score"] == 20
    assert result["normalized_score"] == pytest.approx(0.95)
    assert result["match_rate"] == pytest.approx(0.95)
    assert result["margin"] == pytest.approx(0.4794, abs=1e-4)
    assert result["evidence_count"] == 5


def test_score_prediction_becomes_confident_when_top_meets_threshold_and_margin():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    features = make_features(
        ["riolu", "solrock", "power_protein", "makuhita", "fight_gong", "lucario"],
        {"riolu": 1, "solrock": 2, "power_protein": 3, "makuhita": 4, "fight_gong": 5, "lucario": 6},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)

    assert result["deck_type"] == "mega_lucario_ex"
    assert result["status"] == "confident"
    assert result["score"] == 21
    assert result["normalized_score"] == pytest.approx(1.05)
    assert result["match_rate"] == 1.0


def test_score_prediction_does_not_scale_score_by_zone():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    config["prediction_decision"]["min_top_normalized_score"] = 0.0

    active_result = rough_predictor._score_prediction(
        make_features(["solrock"], {"solrock": 2}, zone="active"),
        config,
    )
    revealed_result = rough_predictor._score_prediction(
        make_features(["solrock"], {"solrock": 2}, zone="revealed"),
        config,
    )

    assert active_result["candidates"][0]["score"] == 4
    assert revealed_result["candidates"][0]["score"] == 4


def test_score_prediction_returns_ambiguous_when_top_and_second_are_close():
    config = make_test_config()
    config["archetypes"]["mega_lucario_ex"]["confident_score"] = 6.5
    config["archetypes"]["ambiguous_rival"]["confident_score"] = 6.0
    features = make_features(
        ["solrock", "makuhita", "fight_gong"],
        {"solrock": 2, "makuhita": 4, "fight_gong": 5},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)

    assert result["top_candidate"] == "Ambiguous Rival"
    assert result["deck_type"] == "unknown"
    assert result["status"] == "ambiguous"
    assert result["margin"] == pytest.approx(0.1282, abs=1e-4)


def test_score_prediction_supports_combo_any_names_without_double_counting():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    config["archetypes"]["mega_lucario_ex"]["combo_rules"] = [
        {
            "names": ["solrock"],
            "any_names": ["riolu", "lucario"],
            "bonus": 4,
            "role": "combo",
        }
    ]
    features = make_features(
        ["riolu", "lucario", "solrock"],
        {"riolu": 1, "lucario": 6, "solrock": 2},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)

    top = result["candidates"][0]
    assert top["deck_type"] == "mega_lucario_ex"
    assert top["score"] == 15
    assert [item for item in top["evidence"] if item["role"] == "combo"][0]["weight"] == 4


def test_score_prediction_generates_evolution_candy_combo():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    config["evolution_candy_combo"] = {
        "enabled": True,
        "candy_names": ["rare_candy"],
        "bonus": 4,
        "role": "combo",
    }
    config["archetypes"]["mega_lucario_ex"]["role_cards"]["core"].append(
        {"name": "rare_candy", "card_id": [8]}
    )
    features = make_features(
        ["riolu", "rare_candy"],
        {"riolu": 1, "rare_candy": 8},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)
    top = result["candidates"][0]

    assert top["deck_type"] == "mega_lucario_ex"
    assert top["score"] == 13
    assert [item for item in top["evidence"] if item["role"] == "combo"][0]["weight"] == 4


def test_score_prediction_counts_same_line_key_only_once():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    config["archetypes"]["mega_lucario_ex"]["role_cards"]["core"].extend(
        [
            {"name": "torchic", "card_id": [8], "line_key": "blaziken_line"},
            {"name": "combusken", "card_id": [9], "line_key": "blaziken_line"},
        ]
    )
    features = make_features(
        ["torchic", "combusken"],
        {"torchic": 8, "combusken": 9},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)
    top = result["candidates"][0]

    assert top["score"] == 4
    assert top["evidence_count"] == 1
    assert top["evidence"][0]["card"] == "torchic"


def test_score_prediction_returns_unknown_when_evidence_count_is_too_low():
    config = make_test_config()
    features = make_features(["solo_ace"], {"solo_ace": 7}, zone=None)

    result = rough_predictor._score_prediction(features, config)

    assert result["top_candidate"] == "Single Evidence"
    assert result["deck_type"] == "unknown"
    assert result["status"] == "insufficient_evidence"
    assert result["evidence_count"] == 1


def test_score_prediction_returns_no_candidate_for_empty_features():
    result = rough_predictor._score_prediction(make_features([]), make_test_config())

    assert result["deck_type"] == "unknown"
    assert result["status"] == "no_candidate"
    assert result["top_candidate"] is None


def test_score_prediction_keeps_normalized_score_over_one_for_sorting():
    config = make_test_config()
    config["archetypes"].pop("ambiguous_rival")
    config["prediction_decision"]["min_evidence_count"] = 1
    features = make_features(
        ["riolu", "solrock", "power_protein", "makuhita", "fight_gong", "lucario"],
        {"riolu": 1, "solrock": 2, "power_protein": 3, "makuhita": 4, "fight_gong": 5, "lucario": 6},
        zone=None,
    )

    result = rough_predictor._score_prediction(features, config)
    top = result["candidates"][0]

    assert top["normalized_score"] > 1.0
    assert top["match_rate"] == 1.0
    assert "probability" not in top
    assert "relative_probability" not in top
