"""クラスタ④ 検証（ベンチ/バトル場の展開選択と相手デッキ予測の連携）／担当B

card_move.bench_field.choose が、相手デッキ予測の対アーキタイプ加点
（MatchupPlan.card_priority_boost）を TO_BENCH / SETUP_BENCH_POKEMON にのみ適用し、
TO_FIELD / SETUP_ACTIVE_POKEMON（バトル場に出す場面）には適用しないことを検証する。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Option, OptionType, SelectContext, SelectData, SelectType
from ptcg_ai.rule_based.card_move import bench_field
from ptcg_ai.shared.profile_types import DeckPlan, MatchupPlan

CASEY_CARD_ID = 741  # opening_priority の先頭（本来最優先）
SHAYMIN_CARD_ID = 343  # opening_priority の末尾（本来最下位）


def _make_select(context: SelectContext) -> SelectData:
    options = [
        Option(type=OptionType.PLAY, cardId=CASEY_CARD_ID),
        Option(type=OptionType.PLAY, cardId=SHAYMIN_CARD_ID),
    ]
    return SelectData(
        type=SelectType.CARD,
        context=context,
        minCount=1,
        maxCount=1,
        remainDamageCounter=0,
        remainEnergyCost=0,
        option=options,
        deck=None,
        contextCard=None,
        effect=None,
    )


@pytest.fixture(autouse=True)
def _patched_deck_plan_and_matchup(monkeypatch):
    # opening_priority: ケーシィ(741)が最優先、シェイミ(343)が最下位という単純な2枚構成にする。
    fake_deck_plan = DeckPlan(opening_priority=[CASEY_CARD_ID, SHAYMIN_CARD_ID])
    monkeypatch.setattr(bench_field.profile_registry, "get_deck_plan", lambda: fake_deck_plan)
    # シェイミへの加点(+5.0)は opening_priority のランク差(1)より十分大きく、
    # 適用されれば必ずシェイミが選ばれるようにする。
    monkeypatch.setattr(
        bench_field.opponent_tracker,
        "current_matchup_plan",
        lambda: MatchupPlan(card_priority_boost={SHAYMIN_CARD_ID: 5.0}),
    )
    yield


def test_matchup_boost_applies_to_to_bench():
    select = _make_select(SelectContext.TO_BENCH)
    result = bench_field.choose(select, state=None)
    assert result == [1]  # シェイミ（加点により逆転）


def test_matchup_boost_applies_to_setup_bench_pokemon():
    select = _make_select(SelectContext.SETUP_BENCH_POKEMON)
    result = bench_field.choose(select, state=None)
    assert result == [1]  # シェイミ（加点により逆転）


def test_matchup_boost_does_not_apply_to_to_field():
    select = _make_select(SelectContext.TO_FIELD)
    result = bench_field.choose(select, state=None)
    assert result == [0]  # ケーシィ（加点は無視され、opening_priority通り）


def test_matchup_boost_does_not_apply_to_setup_active_pokemon():
    select = _make_select(SelectContext.SETUP_ACTIVE_POKEMON)
    result = bench_field.choose(select, state=None)
    assert result == [0]  # ケーシィ（加点は無視され、opening_priority通り）
