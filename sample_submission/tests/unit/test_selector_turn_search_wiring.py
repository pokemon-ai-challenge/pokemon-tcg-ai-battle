"""``ptcg_ai.search.turn_beam`` の selector 配線を検証する。

``_use_ml_policy`` / ``test_selector_ml_policy_wiring.py`` と同じ設計の
「既定は無効」「環境変数が config より強い」「失敗してもターンを止めない(None を返すだけ)」
に加え、この探索は ``SelectContext.MAIN`` 限定であること・返す前に必ず
``is_valid_action()`` を通ることを検証する。turn_beam 自体の探索ロジック
(ビーム幅・時間予算・葉の評価)は ``test_turn_beam.py`` で検証済みなので、
ここでは selector.py 側の配線(フラグ解決・呼び出し・フォールバック)だけを見る。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation, Option, SelectContext, SelectData, SelectType, State
from ptcg_ai.action_selection import selector
from ptcg_ai.search import turn_beam


def _select(context: SelectContext, options: list[Option]) -> SelectData:
    return SelectData(
        type=SelectType.MAIN,
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


def _state(your_index: int = 0) -> State:
    return State(
        turn=3, turnActionCount=0, yourIndex=your_index, firstPlayer=0,
        supporterPlayed=False, stadiumPlayed=False, energyAttached=False,
        retreated=False, result=-1, stadium=[], looking=None,
        players=[
            {"active": [], "bench": [], "benchMax": 5, "deckCount": 40, "discard": [],
             "prize": [], "handCount": 0, "hand": [], "poisoned": False, "burned": False,
             "asleep": False, "paralyzed": False, "confused": False},
            {"active": [], "bench": [], "benchMax": 5, "deckCount": 40, "discard": [],
             "prize": [None] * 6, "handCount": 0, "hand": None, "poisoned": False,
             "burned": False, "asleep": False, "paralyzed": False, "confused": False},
        ],
    )


def _obs(context: SelectContext) -> Observation:
    options = [Option(type=14)]  # END のみ = 強制手だが議論を単純にするため
    return Observation(select=_select(context, options), logs=[], current=_state())


# --- フラグの解決 -----------------------------------------------------------


def test_flag_defaults_to_false(monkeypatch):
    monkeypatch.delenv("PTCG_TURN_SEARCH", raising=False)
    assert selector._use_turn_search({}) is False


def test_flag_reads_config(monkeypatch):
    monkeypatch.delenv("PTCG_TURN_SEARCH", raising=False)
    assert selector._use_turn_search({"turn_search": {"enabled": True}}) is True


def test_flag_defaults_to_false_when_turn_search_key_missing(monkeypatch):
    monkeypatch.delenv("PTCG_TURN_SEARCH", raising=False)
    assert selector._use_turn_search({"some_other_key": True}) is False


@pytest.mark.parametrize(
    ("env", "config_value", "expected"),
    [
        ("1", False, True),
        ("0", True, False),
    ],
)
def test_env_overrides_config(monkeypatch, env, config_value, expected):
    monkeypatch.setenv("PTCG_TURN_SEARCH", env)
    assert (
        selector._use_turn_search({"turn_search": {"enabled": config_value}}) is expected
    )


def test_unrecognised_env_value_is_ignored(monkeypatch):
    monkeypatch.setenv("PTCG_TURN_SEARCH", "yes")
    assert selector._use_turn_search({"turn_search": {"enabled": True}}) is True
    assert selector._use_turn_search({"turn_search": {"enabled": False}}) is False


# --- MAIN 限定 ---------------------------------------------------------------


def test_turn_search_action_skips_non_main_context(monkeypatch):
    called = []
    monkeypatch.setattr(turn_beam, "search", lambda *a, **k: called.append(1) or [0])
    obs = _obs(SelectContext.TO_HAND)
    assert selector._turn_search_action(obs, obs.select, {}, [1] * 60) is None
    assert called == []  # 呼ばれていない


def test_turn_search_action_returns_none_when_current_missing(monkeypatch):
    obs = Observation(select=_select(SelectContext.MAIN, [Option(type=14)]), logs=[], current=None)
    assert selector._turn_search_action(obs, obs.select, {}, [1] * 60) is None


# --- 例外時はターンを止めない -----------------------------------------------


def test_turn_search_action_returns_none_on_exception(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("turn_beam exploded")

    monkeypatch.setattr(turn_beam, "search", _boom)
    obs = _obs(SelectContext.MAIN)
    assert selector._turn_search_action(obs, obs.select, {}, [1] * 60) is None


def test_turn_search_action_returns_choice_when_search_selects(monkeypatch):
    monkeypatch.setattr(turn_beam, "search", lambda *a, **k: [0])
    obs = _obs(SelectContext.MAIN)
    assert selector._turn_search_action(obs, obs.select, {}, [1] * 60) == [0]


# --- select_action 全体: 無効な選択は次の候補へフォールバックする -----------


def test_select_action_falls_back_to_router_when_turn_search_choice_is_invalid(monkeypatch):
    """``is_valid_action`` を落ちる選択(不正な index)を返しても、既存の router.route に
    フォールバックしターンが止まらないこと。新しい合法性検証は書かず、既存の
    ``is_valid_action`` をそのまま使う設計を検証する。"""
    monkeypatch.setenv("PTCG_TURN_SEARCH", "1")
    monkeypatch.setattr(turn_beam, "search", lambda *a, **k: [999])  # 選択肢数を超える不正な index
    monkeypatch.setattr(selector.router, "route", lambda obs: [0])

    obs = _obs(SelectContext.MAIN)
    config = {"turn_search": {"enabled": True}, "lethal_search": {"enabled": False}, "ml_policy": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]


def test_select_action_uses_turn_search_choice_when_valid(monkeypatch):
    monkeypatch.setenv("PTCG_TURN_SEARCH", "1")
    monkeypatch.setattr(turn_beam, "search", lambda *a, **k: [0])

    obs = _obs(SelectContext.MAIN)  # option[0] = END, minCount=maxCount=1
    config = {"turn_search": {"enabled": True}, "lethal_search": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]


def test_select_action_skips_turn_search_when_disabled(monkeypatch):
    """既定(turn_search 無効)では、探索が呼ばれていても実行されないこと。"""
    monkeypatch.delenv("PTCG_TURN_SEARCH", raising=False)

    def _boom(*a, **k):
        raise AssertionError("turn_beam.search must not be called when disabled")

    monkeypatch.setattr(turn_beam, "search", _boom)
    monkeypatch.setattr(selector.router, "route", lambda obs: [0])

    obs = _obs(SelectContext.MAIN)
    config = {"lethal_search": {"enabled": False}, "ml_policy": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]


def test_select_action_tries_turn_search_before_ml_policy(monkeypatch):
    """turn_search が有効な選択を返したら、ml_policy には進まないこと。"""
    monkeypatch.setenv("PTCG_TURN_SEARCH", "1")
    monkeypatch.setattr(turn_beam, "search", lambda *a, **k: [0])

    def _ml_boom(obs, select):
        raise AssertionError("ml_policy must not run when turn_search already returned")

    monkeypatch.setattr(selector, "_ml_policy_action", _ml_boom)

    obs = _obs(SelectContext.MAIN)
    config = {
        "turn_search": {"enabled": True},
        "ml_policy": {"enabled": True},
        "lethal_search": {"enabled": False},
    }
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]
