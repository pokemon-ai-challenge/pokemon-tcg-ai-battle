from cg.api import AreaType, OptionType

from src.decision.main_turn_parts.buckets import MainOptionBuckets
from src.decision.main_turn_parts.priorities.ability import propose_ability_action
from src.decision.main_turn_parts.weights import MAIN_ACTION_BASE_WEIGHTS
from src.knowledge.card_cache import load_card_data
from src.tests.helpers_decision import ability_option, deck_card_ids


def test_deck_contains_real_ability_cards_for_priority_coverage():
    deck_ids = set(deck_card_ids())
    card_by_id = {card.cardId: card for card in load_card_data()}

    lunatone = card_by_id[675]
    hariyama = card_by_id[674]

    assert 675 in deck_ids
    assert 674 in deck_ids
    assert any("Lunar Cycle" in skill.name for skill in lunatone.skills)
    assert any("Heave-Ho Catcher" in skill.name for skill in hariyama.skills)


def test_offers_ability_when_real_deck_ability_option_exists():
    option = ability_option(area=AreaType.BENCH, index=0)

    proposal = propose_ability_action(
        obs=None,
        buckets=MainOptionBuckets(ability=[0]),
    )

    assert option.type == OptionType.ABILITY
    assert proposal is not None
    assert proposal.action == [0]
    assert proposal.score == MAIN_ACTION_BASE_WEIGHTS["ability"]


def test_returns_none_when_no_legal_ability_option_exists():
    proposal = propose_ability_action(
        obs=None,
        buckets=MainOptionBuckets(),
    )

    assert proposal is None
