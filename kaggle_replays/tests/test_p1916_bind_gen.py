from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "sample_submission", ROOT / "kaggle_replays" / "measurement", ROOT / "kaggle_replays"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cg.api import AreaType, OptionType, SelectType  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("p1916_bind_gen", ROOT / "kaggle_replays" / "_p1916_bind_gen.py")
MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(MOD)


def card(card_id, serial):
    return SimpleNamespace(id=card_id, serial=serial)


def pokemon(card_id, serial, hp=100, energies=None, tools=None):
    return SimpleNamespace(
        id=card_id, serial=serial, hp=hp, maxHp=120, appearThisTurn=False,
        energies=list(energies or []), energyCards=[], tools=list(tools or []), preEvolution=[],
    )


def player(active=None, bench=None, hand=None):
    return SimpleNamespace(
        active=list(active or []), bench=list(bench or []), hand=list(hand or []),
        handCount=len(hand or []), deckCount=30, discard=[], prize=[],
    )


def state(me=0, mine=None, opponent=None, turn=4):
    return SimpleNamespace(
        yourIndex=me, players=[mine or player(), opponent or player()], turn=turn,
        supporterPlayed=False, energyAttached=False, retreated=False, stadium=[], result=-1,
    )


def option(**kwargs):
    defaults = dict(type=OptionType.CARD, number=None, area=None, index=None,
                    playerIndex=None, toolIndex=None, energyIndex=None, count=None,
                    inPlayArea=None, inPlayIndex=None, attackId=None, cardId=None, serial=None)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_source_and_target_bindings_cover_active_hand_and_attached_card():
    tool = card(301, 31)
    mine_active = pokemon(101, 11, tools=[tool])
    mine_hand = card(201, 21)
    mine_bench = pokemon(102, 12)
    current = state(mine=player(active=[mine_active], bench=[mine_bench], hand=[mine_hand]),
                    opponent=player())

    active = option(area=AreaType.ACTIVE, index=0, inPlayArea=AreaType.BENCH,
                    inPlayIndex=0, playerIndex=0)
    source, target, failures = MOD._option_bindings(active, current)
    assert source == {"area": int(AreaType.ACTIVE), "index": 0, "player_index": 0,
                      "serial": 11, "card_id": 101}
    assert target == {"area": int(AreaType.BENCH), "index": 0, "player_index": 0,
                      "serial": 12, "card_id": 102}
    assert failures == 0

    hand_source, _, failures = MOD._option_bindings(option(area=AreaType.HAND, index=0), current)
    assert hand_source["serial"] == 21 and hand_source["card_id"] == 201 and failures == 0

    tool_source, _, failures = MOD._option_bindings(
        option(area=AreaType.ACTIVE, index=0, toolIndex=0, playerIndex=0), current)
    assert tool_source["serial"] == 31 and tool_source["card_id"] == 301 and failures == 0


def test_delta_by_serial_tracks_hp_energy_and_zone_change():
    mine = pokemon(101, 11, hp=100, energies=[1])
    opponent = pokemon(102, 12, hp=80)
    before = MOD.snapshot_entities(state(mine=player(active=[mine]), opponent=player(bench=[opponent])), 0)
    mine.hp = 70
    mine.energies.append(2)
    after = MOD.snapshot_entities(state(mine=player(bench=[mine]), opponent=player(bench=[opponent])), 0)

    delta = MOD.delta_by_serial(before, after)
    assert delta["11"]["hp_delta"] == -30
    assert delta["11"]["energy_delta"] == 1
    assert delta["11"]["zone_change"] == {"from": "active", "to": "bench"}


def test_production_safety_rejects_opponent_hand_card_ids():
    safe = {"before_hand": [201], "before_global": {"hand_count_opp": 4}}
    MOD._assert_production_safe(safe)
    with pytest.raises(AssertionError):
        MOD._assert_production_safe({**safe, "opponent_hand": [999]})


def test_roll_candidate_stops_when_turn_returns_to_self(monkeypatch):
    mine = pokemon(101, 11)
    opponent = pokemon(102, 12)
    root_state = state(mine=player(active=[mine]), opponent=player(active=[opponent]))
    root_option = option(area=AreaType.ACTIVE, index=0)
    opp_option = option(area=AreaType.ACTIVE, index=0, playerIndex=1)
    root_obs = SimpleNamespace(select=SimpleNamespace(type=SelectType.MAIN, option=[root_option]), current=root_state)
    opponent_state = state(me=1, mine=player(active=[mine]), opponent=player(active=[opponent]))
    opponent_obs = SimpleNamespace(select=SimpleNamespace(type=SelectType.MAIN, option=[opp_option]), current=opponent_state)
    self_state = state(me=0, mine=player(active=[mine]), opponent=player(active=[opponent]))
    self_obs = SimpleNamespace(select=SimpleNamespace(type=SelectType.MAIN, option=[root_option]), current=self_state)
    root = SimpleNamespace(searchId=1, observation=root_obs)
    child = SimpleNamespace(searchId=2, observation=opponent_obs)
    returned = SimpleNamespace(searchId=3, observation=self_obs)

    class FakeCg:
        def search_step(self, search_id, selection):
            return child if search_id == 1 else returned

        def search_release(self, search_id):
            return None

        def search_end(self):
            return None

    monkeypatch.setattr(MOD, "_begin", lambda obs, me, seed: root)
    monkeypatch.setattr(MOD.P, "cg_api", FakeCg())
    monkeypatch.setattr(MOD.P, "_greedy_selection", lambda policy, obs: [0])
    monkeypatch.setitem(MOD.EV, "value", SimpleNamespace(evaluate=lambda current, me: 0.25 if current.yourIndex == 1 else 0.75))
    before = MOD.snapshot_entities(root_state, 0)

    result = MOD._roll_candidate(root_obs, 0, 0, 7, object(), before)
    assert [entry["actor"] for entry in result["trajectory"]] == [0, 1]
    assert result["teacher"] == {"C1b": 0.25, "C4": 0.75}
