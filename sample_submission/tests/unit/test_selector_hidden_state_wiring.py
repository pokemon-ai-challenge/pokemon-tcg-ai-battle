"""FR-WIRE: selector が lethal search へ渡す隠れ情報の供給元を検証する。

`use_real_hidden_state` フラグにより、探索へ渡す `hidden_state_factory` が
ダミー(`hidden_information.search_state_stub`)と実推定(`match_context` +
`search_adapter`)のどちらを使うかが切り替わることを保証する。

このフラグは A/B の対照群/実験群そのものなので、
「既定はダミー(従来動作)」「環境変数が config より強い」「実推定が失敗しても
ターンを止めない(None を返すだけ)」の3点が壊れると実験の意味が失われる。
"""

from pathlib import Path
import sys

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

from cg.api import Observation
from ptcg_ai.action_selection import selector


_SENTINEL_DUMMY = {"source": "dummy"}
_SENTINEL_REAL = {"source": "real"}


def _obs() -> Observation:
    # factory の分岐だけを見るテストなので、obs の中身は使われない
    # (差し替えた供給元がどちらも obs を無視するため)。
    return Observation(select=None, logs=[], current=None)


@pytest.fixture
def stub_sources(monkeypatch):
    """ダミー側・実推定側の供給元を、区別可能な番兵に差し替える。"""
    monkeypatch.setattr(
        selector, "build_dummy_search_state", lambda obs, full_deck: _SENTINEL_DUMMY
    )
    monkeypatch.setattr(
        selector,
        "to_search_begin_kwargs",
        lambda own, opponent, obs: _SENTINEL_REAL,
    )
    monkeypatch.setattr(selector.match_context, "get_own_state", lambda: object())
    monkeypatch.setattr(selector.match_context, "get_opponent_state", lambda: object())


# --- フラグの解決 ---------------------------------------------------------


def test_flag_defaults_to_false(monkeypatch):
    """config に何も書かれていなければダミー(従来動作)のまま。"""
    monkeypatch.delenv("PTCG_REAL_HIDDEN_STATE", raising=False)
    assert selector._use_real_hidden_state({}) is False


def test_flag_reads_config(monkeypatch):
    monkeypatch.delenv("PTCG_REAL_HIDDEN_STATE", raising=False)
    assert selector._use_real_hidden_state({"use_real_hidden_state": True}) is True


@pytest.mark.parametrize(
    ("env", "config_value", "expected"),
    [
        ("1", False, True),   # 環境変数で有効化(config は false)
        ("0", True, False),   # 環境変数で無効化(config は true)
    ],
)
def test_env_overrides_config(monkeypatch, env, config_value, expected):
    monkeypatch.setenv("PTCG_REAL_HIDDEN_STATE", env)
    assert (
        selector._use_real_hidden_state({"use_real_hidden_state": config_value})
        is expected
    )


def test_unrecognised_env_value_is_ignored(monkeypatch):
    """想定外の値で暗黙に挙動が反転しないこと。"""
    monkeypatch.setenv("PTCG_REAL_HIDDEN_STATE", "yes")
    assert selector._use_real_hidden_state({"use_real_hidden_state": True}) is True
    assert selector._use_real_hidden_state({"use_real_hidden_state": False}) is False


# --- 供給元の切り替え -----------------------------------------------------


def test_factory_uses_dummy_when_disabled(stub_sources):
    factory = selector._hidden_state_factory(_obs(), [1, 2, 3], use_real=False)
    assert factory() is _SENTINEL_DUMMY


def test_factory_uses_real_estimation_when_enabled(stub_sources):
    factory = selector._hidden_state_factory(_obs(), [1, 2, 3], use_real=True)
    assert factory() is _SENTINEL_REAL


def test_real_factory_returns_none_on_failure(monkeypatch):
    """実推定が例外を投げても、探索の契約どおり None を返すだけに留める。

    (None はそのVerification replayをスキップする合図であり、ターンは止まらない)
    """

    def boom(*_args, **_kwargs):
        raise RuntimeError("estimation exploded")

    monkeypatch.setattr(selector, "to_search_begin_kwargs", boom)
    monkeypatch.setattr(selector.match_context, "get_own_state", lambda: object())
    monkeypatch.setattr(selector.match_context, "get_opponent_state", lambda: object())

    factory = selector._hidden_state_factory(_obs(), [1, 2, 3], use_real=True)
    assert factory() is None


def test_real_factory_resamples_each_call(monkeypatch):
    """呼ぶたびに再サンプルすること(シャッフル依存ラインの棄却がこれに依存する)。"""
    calls = []

    def counting(own, opponent, obs):
        calls.append(1)
        return {"sample": len(calls)}

    monkeypatch.setattr(selector, "to_search_begin_kwargs", counting)
    monkeypatch.setattr(selector.match_context, "get_own_state", lambda: object())
    monkeypatch.setattr(selector.match_context, "get_opponent_state", lambda: object())

    factory = selector._hidden_state_factory(_obs(), [1, 2, 3], use_real=True)
    assert factory() == {"sample": 1}
    assert factory() == {"sample": 2}
