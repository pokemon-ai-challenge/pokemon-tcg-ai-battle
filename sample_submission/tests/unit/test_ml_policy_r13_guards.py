"""r13(``use_closed_form_ko``)のユニットテスト: KO判定の3段フォールバック。

背景(実測): `ability_draw_brake`(r11 Fix-I)と `low_deck_draw_brake`(r9 Fix-D)は
スナップショットの単純局面では正しく発火するのに、**実対戦120試合で発火0回**だった。
KO可否をエンジン探索(`ko_search.can_ko_this_turn` /
`ability_draw_eval.already_ko_without_more_energy`)で判定しており、みどりのまいを持つ
オーガポンが複数体並ぶ盤面では 300ms 予算内に完走せず、すべて「判定不能」→安全側で
介入見送りになっていたため。

r13 は判定を **閉形式ファストパス(`search/closed_form_ko.py`)→ 従来の探索 →
それも判定不能なら不介入** の3段にする。ここで固定するのは:

  1. ``use_closed_form_ko`` が無い config では閉形式が**一度も呼ばれない**(=r12以前と同一)
  2. 閉形式が結論を出したら**探索を一切呼ばない**(=これがファストパスの目的)
  3. 閉形式が判定不能なら従来の探索に落ち、それも判定不能なら介入しない(安全側)
  4. 絶対例外(ATTACK選択肢が無いなら絶対にvetoしない)は閉形式より優先される

既存の `test_ml_policy_r11_guards.py` / `test_ml_policy_r12_guards.py` は他担当が同時に
編集しているため、競合を避けてこの新ファイルに追加している(検証対象は同じ
`ml_policy_agent`。`_snapshot_gate_r1x.py` を増設していくのと同じ流儀)。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import copy
import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"

_OGERPON = 96        # オーガポン みどりのめん ex(特性=みどりのまい / ワザ=まんようしぐれ)
_GRASS_ENERGY = 1    # 基本【草】エネルギー
_LILLIE = 1227       # リーリエの決心(手札を山に戻して6枚引く)

# r13 = r11/r9 のブレーキ + 閉形式ファストパス
_ABILITY_R13 = {
    "lethal_search": {"enabled": False},
    "ability_draw_brake": {
        "enabled": True, "deck_threshold": 8, "time_limit_ms": 300,
        "use_closed_form_ko": True,
    },
}
# r12以前(閉形式キー無し)
_ABILITY_R12 = {
    "lethal_search": {"enabled": False},
    "ability_draw_brake": {"enabled": True, "deck_threshold": 8, "time_limit_ms": 300},
}
_LOWDECK_R13 = {
    "lethal_search": {"enabled": False},
    "low_deck_draw_brake": {
        "enabled": True, "deck_threshold": 6, "use_closed_form_ko": True,
    },
}
_LOWDECK_R12 = {
    "lethal_search": {"enabled": False},
    "low_deck_draw_brake": {"enabled": True, "deck_threshold": 6},
}


@pytest.fixture(scope="module")
def ml_policy_agent():
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mod
    except Exception as exc:  # noqa: BLE001 - cg エンジン未ロード環境ではスキップ
        pytest.skip(f"ml_policy_agent / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


class _StubScoreModel:
    def __init__(self, scores):
        self._scores = list(scores)

    def score_options(self, obs, hidden_state_factory=None, deadline=None):
        return list(self._scores)

    def select_option(self, obs, hidden_state_factory=None, deadline=None):
        return max(range(len(self._scores)), key=lambda i: self._scores[i])


def _card(card_id: int, serial: int, player_index: int = 0) -> dict:
    return {"id": card_id, "playerIndex": player_index, "serial": serial}


def _build_obs(encoder_observations, options, *, hand_ids=None, active_id=None,
               bench_ids=None, deck_count=5, select_type=0):
    """mid_game を土台にした合成 obs(`test_ml_policy_r11_guards.py` と同じ流儀)。"""
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    state = base["current"]
    me = state["players"][0]
    if hand_ids is not None:
        me["hand"] = [_card(cid, 900 + i) for i, cid in enumerate(hand_ids)]
        me["handCount"] = len(hand_ids)
    if active_id is not None:
        me["active"][0]["id"] = active_id
    if bench_ids is not None:
        proto = copy.deepcopy(me["bench"][0])
        bench = []
        for i, cid in enumerate(bench_ids):
            slot = copy.deepcopy(proto)
            slot["id"] = cid
            slot["serial"] = 700 + i
            bench.append(slot)
        me["bench"] = bench
    me["deckCount"] = deck_count
    state["stadium"] = []
    state["looking"] = None
    base["select"] = {
        "context": 0, "contextCard": None, "deck": None, "effect": None,
        "maxCount": 1, "minCount": 1, "option": list(options),
        "remainDamageCounter": 0, "remainEnergyCost": 0, "type": select_type,
    }
    base["logs"] = []
    return to_observation_class(base)


def _attach(hand_index: int, *, to_active: bool) -> dict:
    return {"type": 8, "area": 2, "index": hand_index,
            "inPlayArea": 4 if to_active else 5, "inPlayIndex": 0}


def _ability(*, on_active: bool, index: int = 0) -> dict:
    return {"type": 10, "area": 4 if on_active else 5, "index": index}


def _play(hand_index: int) -> dict:
    return {"type": 7, "area": 2, "index": hand_index}


def _attack(attack_id: int = 120) -> dict:
    return {"type": 13, "attackId": attack_id}


def _end() -> dict:
    return {"type": 14}


def _ability_obs(encoder_observations, *, deck_count=5, with_attack=True):
    """みどりのまい(ベンチ個体)を選ぼうとしている局面。index 3 が対象。"""
    options = [
        _attach(0, to_active=False),
        _attach(0, to_active=True),
        _ability(on_active=True),
        _ability(on_active=False),
    ]
    if with_attack:
        options.append(_attack())
    options.append(_end())
    return _build_obs(encoder_observations, options, hand_ids=[_GRASS_ENERGY],
                      active_id=_OGERPON, bench_ids=[_OGERPON], deck_count=deck_count)


def _lowdeck_obs(encoder_observations, *, deck_count=5, with_attack=True):
    """リーリエ(index 0)を選ぼうとしている山札僅少局面。"""
    options = [_play(0), _attach(1, to_active=True)]
    if with_attack:
        options.append(_attack())
    options.append(_end())
    return _build_obs(
        encoder_observations, options,
        hand_ids=[_LILLIE, _GRASS_ENERGY, _GRASS_ENERGY, _GRASS_ENERGY,
                  _GRASS_ENERGY, _GRASS_ENERGY, _GRASS_ENERGY, _GRASS_ENERGY],
        active_id=_OGERPON, bench_ids=[_OGERPON], deck_count=deck_count)


def _explode(*_a, **_k):
    raise AssertionError("この経路は呼ばれてはならない")


def _closed_form_stub(now=None, impossible=None):
    """`closed_form_ko` の2関数を固定値に差し替えるための (fn_now, fn_impossible)。"""

    def _now(state, me, config=None, report=None):
        return now

    def _impossible(state, me, config=None, report=None):
        if report is not None:
            report["reason"] = "damage_modifier_in_play"
        return impossible

    return _now, _impossible


def _ko_stub(result, *, searched=True, aborted=False):
    def _f(obs, factory, cfg=None, deadline=None, report=None):
        if report is not None:
            report["searched"] = searched
            report["aborted"] = aborted
        return result

    return _f


def _already_ko_stub(already_ko, aborted=False):
    def _f(obs, factory, cfg=None, deadline=None):
        return already_ko, aborted

    return _f


# ===========================================================================
# 1. キーが無ければ閉形式は一度も呼ばれない(既定OFF=本番不変)
# ===========================================================================

def test_closed_form_not_called_without_key_ability(
    ml_policy_agent, encoder_observations, monkeypatch
):
    from ptcg_ai.search import ability_draw_eval, closed_form_ko

    monkeypatch.setattr(closed_form_ko, "can_ko_now", _explode)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", _explode)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))

    obs = _ability_obs(encoder_observations)
    # r12 config(閉形式キー無し)でも従来どおり発火する = 挙動不変
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R12) is not None


def test_closed_form_not_called_without_key_lowdeck(
    ml_policy_agent, encoder_observations, monkeypatch
):
    from ptcg_ai.search import closed_form_ko, ko_search

    monkeypatch.setattr(closed_form_ko, "can_ko_now", _explode)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", _explode)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7]))

    obs = _lowdeck_obs(encoder_observations)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R12) is not None


def test_absent_brake_key_is_still_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """ブレーキのキー自体が無い本番 config では、閉形式にも探索にも到達しない。"""
    from ptcg_ai.search import ability_draw_eval, closed_form_ko, ko_search

    for mod, name in ((closed_form_ko, "can_ko_now"),
                      (closed_form_ko, "is_ko_impossible_this_turn"),
                      (ko_search, "can_ko_this_turn"),
                      (ability_draw_eval, "already_ko_without_more_energy")):
        monkeypatch.setattr(mod, name, _explode)
    obs = _ability_obs(encoder_observations)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config={}) is None
    assert ml_policy_agent._try_low_deck_draw_brake(
        _lowdeck_obs(encoder_observations), [0], config={}) is None


# ===========================================================================
# 2. 閉形式が結論を出したら探索は呼ばない(=ファストパスとして機能している)
# ===========================================================================

def test_ability_fastpath_true_fires_without_search(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """閉形式が「今すぐKOできる」= (a) already_lethal で発火。探索は一切呼ばない。"""
    from ptcg_ai.search import ability_draw_eval, closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=True)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy", _explode)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _explode)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))

    ml_policy_agent.reset_guard_stats()
    obs = _ability_obs(encoder_observations)
    action = ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13)
    stats = ml_policy_agent.get_guard_stats()
    assert action == [0]  # 同型ABILITYを除いた中で最大スコアの手
    assert stats["ability_draw_brake_fired"] == 1
    assert stats["ability_draw_brake_fired_already_lethal"] == 1
    assert stats["closed_form_ko_true"] == 1
    assert stats["ability_draw_brake_closed_form_ko"] == 1
    assert stats["ability_draw_brake_inconclusive"] == 0


def test_ability_fastpath_impossible_fires_as_futile(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """閉形式が「攻撃でKOする手は存在しない」= (b) futile で発火。探索は呼ばない。"""
    from ptcg_ai.search import ability_draw_eval, closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=False, impossible=True)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy", _explode)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _explode)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))

    ml_policy_agent.reset_guard_stats()
    obs = _ability_obs(encoder_observations)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13) == [0]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_fired_futile"] == 1
    assert stats["closed_form_ko_false"] == 1
    assert stats["ability_draw_brake_closed_form_no_ko"] == 1


def test_lowdeck_fastpath_true_skips_brake_without_search(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """閉形式が「KOできる」なら引いて勝ちに行かせる(ブレーキを掛けない)。探索は呼ばない。"""
    from ptcg_ai.search import closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=True)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _explode)

    ml_policy_agent.reset_guard_stats()
    obs = _lowdeck_obs(encoder_observations)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R13) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["low_deck_draw_brake_fired"] == 0
    assert stats["low_deck_draw_brake_skipped_ko"] == 1
    assert stats["low_deck_draw_brake_closed_form_ko"] == 1


def test_lowdeck_fastpath_impossible_fires_without_search(
    ml_policy_agent, encoder_observations, monkeypatch
):
    from ptcg_ai.search import closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=False, impossible=True)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _explode)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7]))

    ml_policy_agent.reset_guard_stats()
    obs = _lowdeck_obs(encoder_observations)
    action = ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R13)
    stats = ml_policy_agent.get_guard_stats()
    assert action is not None and action != [0]
    assert stats["low_deck_draw_brake_fired"] == 1
    assert stats["low_deck_draw_brake_closed_form_no_ko"] == 1


# ===========================================================================
# 3. 閉形式が判定不能なら従来の探索へ、それも不能なら不介入(3段目)
# ===========================================================================

def test_ability_falls_back_to_search_when_closed_form_unresolved(
    ml_policy_agent, encoder_observations, monkeypatch
):
    from ptcg_ai.search import ability_draw_eval, closed_form_ko

    now, impossible = _closed_form_stub(now=None, impossible=None)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))

    ml_policy_agent.reset_guard_stats()
    obs = _ability_obs(encoder_observations)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13) == [0]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_fired_already_lethal"] == 1
    assert stats["closed_form_ko_unresolved"] == 1
    assert stats["ability_draw_brake_closed_form_unresolved"] == 1
    # 判定不能の理由が内訳カウンタに記録される(発火0のとき原因を追えるようにするため)
    assert stats["closed_form_ko_reason_damage_modifier_in_play"] == 1


def test_ability_no_intervention_when_both_stages_unresolved(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """閉形式も探索も判定不能なら介入しない(安全側)。r11 の inconclusive 挙動を保つ。"""
    from ptcg_ai.search import ability_draw_eval, closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=None, impossible=None)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(False, aborted=True))
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))

    ml_policy_agent.reset_guard_stats()
    obs = _ability_obs(encoder_observations)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_fired"] == 0
    assert stats["ability_draw_brake_inconclusive"] == 1


def test_middle_band_is_labelled_ko_possible_with_prep(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """「今すぐではKO不可・しかし不可能とも言い切れない」中間帯は専用ラベルで記録する。

    閉形式が両方の問いに**確定**で False を返したケース。閉形式の欠陥ではなく、この帯だけは
    探索でしか詰められないことを示す(実測60試合では unresolved 18件中12件がこれだった)。
    """
    from ptcg_ai.search import closed_form_ko, ko_search

    monkeypatch.setattr(closed_form_ko, "can_ko_now",
                        lambda state, me, config=None, report=None: False)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn",
                        lambda state, me, config=None, report=None: False)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))

    ml_policy_agent.reset_guard_stats()
    obs = _lowdeck_obs(encoder_observations)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R13) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["closed_form_ko_reason_ko_possible_with_prep"] == 1
    assert stats["closed_form_ko_reason_unknown"] == 0


def test_lowdeck_falls_back_to_search_when_closed_form_unresolved(
    ml_policy_agent, encoder_observations, monkeypatch
):
    from ptcg_ai.search import closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=None, impossible=None)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))

    ml_policy_agent.reset_guard_stats()
    obs = _lowdeck_obs(encoder_observations)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R13) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["low_deck_draw_brake_fired"] == 0
    assert stats["low_deck_draw_brake_inconclusive"] == 1


# ===========================================================================
# 4. 絶対例外・誤爆番人は閉形式より優先される
# ===========================================================================

def test_absolute_exception_beats_fastpath(ml_policy_agent, encoder_observations, monkeypatch):
    """ATTACK選択肢が無いなら、閉形式が True でも絶対にvetoしない(閉形式すら呼ばない)。"""
    from ptcg_ai.search import closed_form_ko

    monkeypatch.setattr(closed_form_ko, "can_ko_now", _explode)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", _explode)

    ml_policy_agent.reset_guard_stats()
    obs = _ability_obs(encoder_observations, deck_count=0, with_attack=False)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13) is None
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_fired"] == 0


def test_fastpath_true_requires_engine_attack_option(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """エンジンが ATTACK 選択肢を出していないなら閉形式の True を採用しない。

    盤面のスカラーには出ない一時効果(カブルモ506「スノットアップ」= 次の相手の番ワザが
    使えない。実例 ep93517227 row158)で攻撃自体が封じられていることがあるため。
    `low_deck_draw_brake` は ATTACK選択肢の有無を前提にしないので、この経路が唯一の防波堤。
    """
    from ptcg_ai.search import closed_form_ko, ko_search

    now, impossible = _closed_form_stub(now=True, impossible=None)
    monkeypatch.setattr(closed_form_ko, "can_ko_now", now)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", impossible)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))

    ml_policy_agent.reset_guard_stats()
    obs = _lowdeck_obs(encoder_observations, with_attack=False)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_LOWDECK_R13) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["low_deck_draw_brake_closed_form_ko"] == 0
    assert stats["closed_form_ko_reason_no_attack_option"] == 1
    assert stats["low_deck_draw_brake_inconclusive"] == 1  # 探索へ落ちて判定不能


def test_deck_threshold_still_gates_before_fastpath(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """山札が閾値超なら閉形式にも到達しない(発火条件の順序が変わっていないこと)。"""
    from ptcg_ai.search import closed_form_ko

    monkeypatch.setattr(closed_form_ko, "can_ko_now", _explode)
    monkeypatch.setattr(closed_form_ko, "is_ko_impossible_this_turn", _explode)
    obs = _ability_obs(encoder_observations, deck_count=9)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_R13) is None


# ===========================================================================
# 5. 計測カウンタの初期化(reset_guard_stats が r13 のキーも0に戻すこと)
# ===========================================================================

def test_reset_guard_stats_clears_r13_counters(ml_policy_agent):
    stats = ml_policy_agent.get_guard_stats()
    for key in ("closed_form_ko_true", "closed_form_ko_false", "closed_form_ko_unresolved",
                "ability_draw_brake_closed_form_ko", "low_deck_draw_brake_closed_form_ko",
                "closed_form_ko_reason_damage_modifier_in_play",
                "closed_form_ko_reason_no_attack_option"):
        assert key in stats
    ml_policy_agent._GUARD_STATS["closed_form_ko_true"] = 99
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent.get_guard_stats()["closed_form_ko_true"] == 0
