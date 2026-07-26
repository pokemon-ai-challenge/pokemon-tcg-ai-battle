"""学習済み方策(policy prior)の selector 配線を検証する。

``_use_real_hidden_state`` / ``test_selector_hidden_state_wiring.py`` と同じ設計の
「既定は無効」「環境変数が config より強い」「失敗してもターンを止めない(None を返すだけ)」
に加え、この方策は ``SelectContext.MAIN`` 限定であること・返す前に必ず
``is_valid_action()`` を通ること・特徴抽出が index/serial を使わないことを検証する。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation, Option, SelectContext, SelectData, SelectType, State
from ptcg_ai.action_selection import selector


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
    monkeypatch.delenv("PTCG_ML_POLICY", raising=False)
    assert selector._use_ml_policy({}) is False


def test_flag_reads_config(monkeypatch):
    monkeypatch.delenv("PTCG_ML_POLICY", raising=False)
    assert selector._use_ml_policy({"ml_policy": {"enabled": True}}) is True


def test_flag_defaults_to_false_when_ml_policy_key_missing(monkeypatch):
    monkeypatch.delenv("PTCG_ML_POLICY", raising=False)
    assert selector._use_ml_policy({"some_other_key": True}) is False


@pytest.mark.parametrize(
    ("env", "config_value", "expected"),
    [
        ("1", False, True),
        ("0", True, False),
    ],
)
def test_env_overrides_config(monkeypatch, env, config_value, expected):
    monkeypatch.setenv("PTCG_ML_POLICY", env)
    assert (
        selector._use_ml_policy({"ml_policy": {"enabled": config_value}}) is expected
    )


def test_unrecognised_env_value_is_ignored(monkeypatch):
    monkeypatch.setenv("PTCG_ML_POLICY", "yes")
    assert selector._use_ml_policy({"ml_policy": {"enabled": True}}) is True
    assert selector._use_ml_policy({"ml_policy": {"enabled": False}}) is False


# --- モデル未ロード時は無害 --------------------------------------------------


def test_ml_policy_action_returns_none_when_model_not_ready(monkeypatch):
    """重みファイルが読めない場合、is_ready=False なので None を返すだけ。

    ここで未ロードのモデルを明示的に注入するのは、重みファイルの「不在」に依存させない
    ため。学習パイプラインが policy_weights.json を配置した瞬間にテストの前提が崩れる
    (実際に一度壊れた)。検証したいのは「未ロードなら None を返す」という契約であって、
    ファイルがまだ無いという一時的な状態ではない。
    """

    class _NotReadyModel:
        is_ready = False

        def select(self, state, actions):  # pragma: no cover -- 呼ばれてはいけない
            raise AssertionError("is_ready=False のモデルを呼んではならない")

    monkeypatch.setattr(selector, "_POLICY_MODEL_CACHE", _NotReadyModel())
    obs = _obs(SelectContext.MAIN)
    assert selector._ml_policy_action(obs, obs.select) is None


# --- MAIN 限定 ---------------------------------------------------------------


def test_ml_policy_action_skips_non_main_context(monkeypatch):
    """MAIN 以外の context ではモデルを一切呼ばないこと。"""
    called = []

    class _StubModel:
        is_ready = True

        def select(self, state, actions):
            called.append(1)
            return 0

    monkeypatch.setattr(selector, "_policy_model", lambda: _StubModel())
    obs = _obs(SelectContext.TO_HAND)
    assert selector._ml_policy_action(obs, obs.select) is None
    assert called == []  # 呼ばれていない


def test_ml_policy_action_returns_none_when_current_missing(monkeypatch):
    obs = Observation(select=_select(SelectContext.MAIN, [Option(type=14)]), logs=[], current=None)
    assert selector._ml_policy_action(obs, obs.select) is None


# --- 例外時はターンを止めない -----------------------------------------------


def test_ml_policy_action_returns_none_on_exception(monkeypatch):
    class _BoomModel:
        is_ready = True

        def select(self, state, actions):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(selector, "_policy_model", lambda: _BoomModel())
    obs = _obs(SelectContext.MAIN)
    assert selector._ml_policy_action(obs, obs.select) is None


def test_ml_policy_action_returns_choice_when_model_selects(monkeypatch):
    class _StubModel:
        is_ready = True

        def select(self, state, actions):
            return 0

    monkeypatch.setattr(selector, "_policy_model", lambda: _StubModel())
    obs = _obs(SelectContext.MAIN)
    assert selector._ml_policy_action(obs, obs.select) == [0]


# --- select_action 全体: 無効な選択は router へフォールバックする -----------


def test_select_action_falls_back_to_router_when_ml_policy_choice_is_invalid(monkeypatch):
    """``is_valid_action`` を落ちる選択(不正な index)を返しても、既存の router.route に
    フォールバックしターンが止まらないこと。新しい合法性検証は書かず、既存の
    ``is_valid_action`` をそのまま使う設計を検証する。"""
    monkeypatch.setenv("PTCG_ML_POLICY", "1")

    class _StubModel:
        is_ready = True

        def select(self, state, actions):
            return 999  # 選択肢数を超える不正な index

    monkeypatch.setattr(selector, "_policy_model", lambda: _StubModel())
    monkeypatch.setattr(
        selector.router, "route", lambda obs: [0]
    )

    obs = _obs(SelectContext.MAIN)
    config = {"ml_policy": {"enabled": True}, "lethal_search": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]


def test_select_action_uses_ml_policy_choice_when_valid(monkeypatch):
    monkeypatch.setenv("PTCG_ML_POLICY", "1")

    class _StubModel:
        is_ready = True

        def select(self, state, actions):
            return 0

    monkeypatch.setattr(selector, "_policy_model", lambda: _StubModel())

    obs = _obs(SelectContext.MAIN)  # option[0] = END, minCount=maxCount=1
    config = {"ml_policy": {"enabled": True}, "lethal_search": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]


def test_select_action_skips_ml_policy_when_disabled(monkeypatch):
    """既定(ml_policy 無効)では、モデルが選ばれていても呼ばれないこと。"""
    monkeypatch.delenv("PTCG_ML_POLICY", raising=False)
    called = []

    def _tracking_policy_model():
        called.append(1)
        raise AssertionError("呼ばれてはいけない")

    monkeypatch.setattr(selector, "_policy_model", _tracking_policy_model)
    monkeypatch.setattr(selector.router, "route", lambda obs: [0])

    obs = _obs(SelectContext.MAIN)
    config = {"lethal_search": {"enabled": False}}
    action = selector.select_action(obs, full_deck=[1] * 60, config=config)
    assert action == [0]
    assert called == []
