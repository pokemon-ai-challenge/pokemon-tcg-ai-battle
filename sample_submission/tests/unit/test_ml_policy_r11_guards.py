"""r11 Fix-I (``ability_draw_brake``) のユニットテスト。

実ラダーのリプレイ分析で確定した「特性ドロー(みどりのまい型)の垂れ流し」抑制:
オーガポン みどりのめん ex(96)の特性「みどりのまい」(自分自身にエネ装着+1ドロー、
各個体1回/ターン)を、盤面に複数体並べて毎ターン連打すると、それがそのまま山札消費エンジン
になる — 実例 episode 93517227 T9 row122 / T11 row146(壁無し・サイド2-6でリードし毎ターン
1サイド獲得中なのに、既に攻撃で足りる打点を差し置いて別個体のエネ加速を連打し、
山札 5→2→0 で敗北)。

既存の `test_ml_policy_agent.py` / `test_ml_policy_r10_guards.py` は別担当が同時に編集して
いるため、競合を避けてこの新ファイルに追加している(検証対象は同じ `ml_policy_agent`。
`_snapshot_gate_r10.py` が `_snapshot_gate_r8.py` を書き換えずに増設したのと同じ流儀)。

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

_OGERPON = 96          # オーガポン みどりのめん ex(たね。特性=みどりのまい)
_GRASS_ENERGY = 1      # 基本【草】エネルギー


_ABILITY_BRAKE_ON = {
    "lethal_search": {"enabled": False},
    "ability_draw_brake": {"enabled": True, "deck_threshold": 8, "time_limit_ms": 300},
}
_OFF = {"lethal_search": {"enabled": False}}


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
    """`score_options` が固定スコアを返すだけのモデル(差し替え先の決定を検証可能にする)。"""

    def __init__(self, scores):
        self._scores = list(scores)

    def score_options(self, obs, hidden_state_factory=None, deadline=None):
        return list(self._scores)

    def select_option(self, obs, hidden_state_factory=None, deadline=None):
        return max(range(len(self._scores)), key=lambda i: self._scores[i])


def _card(card_id: int, serial: int, player_index: int = 0) -> dict:
    return {"id": card_id, "playerIndex": player_index, "serial": serial}


def _build_obs(
    encoder_observations,
    options,
    *,
    hand_ids=None,
    active_id=None,
    bench_ids=None,
    bench_max=5,
    deck_count=5,
    select_type=0,
    min_count=1,
    max_count=1,
):
    """mid_game を土台に、各ガードの判定材料だけを明示した合成 obs を作る
    (`test_ml_policy_r10_guards.py` の同名ヘルパと同じ流儀。デッキ残枚数を追加)。
    """
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
    me["benchMax"] = bench_max
    me["deckCount"] = deck_count
    state["stadium"] = []
    state["looking"] = None
    base["select"] = {
        "context": 0,
        "contextCard": None,
        "deck": None,
        "effect": None,
        "maxCount": max_count,
        "minCount": min_count,
        "option": list(options),
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "type": select_type,
    }
    base["logs"] = []
    return to_observation_class(base)


# ---------------------------------------------------------------------------
# option ビルダ(`test_ml_policy_r10_guards.py` と同じフィールド構成)
# ---------------------------------------------------------------------------

def _attach(hand_index: int, *, to_active: bool, in_play_index: int = 0) -> dict:
    return {
        "type": 8, "area": 2, "index": hand_index,
        "inPlayArea": 4 if to_active else 5, "inPlayIndex": in_play_index,
    }


def _ability(*, on_active: bool, index: int = 0) -> dict:
    return {"type": 10, "area": 4 if on_active else 5, "index": index}


def _attack(attack_id: int = 120) -> dict:
    return {"type": 13, "attackId": attack_id}


def _end() -> dict:
    return {"type": 14}


def _ko_stub(result, *, searched=True, aborted=False):
    """`ko_search.can_ko_this_turn` のスタブ(report も実物と同じ内容を書く)。"""

    def _f(obs, factory, cfg=None, deadline=None, report=None):
        if report is not None:
            report["searched"] = searched
            report["aborted"] = aborted
        return result

    return _f


def _already_ko_stub(already_ko, aborted=False):
    """`ability_draw_eval.already_ko_without_more_energy` のスタブ。"""

    def _f(obs, factory, cfg=None, deadline=None):
        return already_ko, aborted

    return _f


def _default_options(*, with_attack=True):
    """アクティブ/ベンチともオーガポン(96)、みどりのまいをベンチ個体に使おうとしている局面。"""
    options = [
        _attach(0, to_active=False),  # 0: 手貼りをベンチへ
        _attach(0, to_active=True),   # 1: 手貼りをアクティブへ
        _ability(on_active=True),     # 2: みどりのまい(アクティブ)
        _ability(on_active=False),    # 3: みどりのまい(ベンチ)
    ]
    if with_attack:
        options.append(_attack())     # 4: ATTACK
    options.append(_end())            # 5 (or 4)
    return options


def _brake_obs(encoder_observations, *, deck_count=5, with_attack=True, options=None):
    return _build_obs(
        encoder_observations,
        options if options is not None else _default_options(with_attack=with_attack),
        hand_ids=[_GRASS_ENERGY],
        active_id=_OGERPON, bench_ids=[_OGERPON],
        deck_count=deck_count,
    )


# ===========================================================================
# キー無し = 常に不介入(本番不変)
# ===========================================================================

def test_ability_draw_brake_absent_key_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し=本番configでは常に None(KO探索にすら到達しない)。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations)

    def _must_not_be_called(*a, **k):
        raise AssertionError("ability_draw_brake disabled なのにKO探索を実行した")

    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy", _must_not_be_called)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_OFF) is None
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config={}) is None


# ===========================================================================
# 絶対例外: 攻撃コスト未充足なら何があっても veto しない
# ===========================================================================

def test_ability_draw_brake_never_fires_when_attack_unavailable(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ATTACK選択肢が無い(=攻撃コスト未充足)なら、山札僅少・特性選択でもKO探索にすら
    到達せず絶対にvetoしない(実例: episode 93517227 T13 row158、山0でも不介入)。
    """
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=0, with_attack=False)

    def _must_not_be_called(*a, **k):
        raise AssertionError("ATTACK選択肢が無いのにKO探索を実行した(絶対例外違反)")

    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy", _must_not_be_called)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON) is None
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_fired"] == 0


# ===========================================================================
# 発火条件: 山札閾値・対象特性の判定
# ===========================================================================

def test_ability_draw_brake_threshold_boundary(ml_policy_agent, encoder_observations, monkeypatch):
    """deck_threshold は「以下」で発火。8=発火 / 9=不発火。"""
    from ptcg_ai.search import ability_draw_eval

    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))

    fires = _brake_obs(encoder_observations, deck_count=8)
    assert ml_policy_agent._try_ability_draw_brake(fires, [3], config=_ABILITY_BRAKE_ON) is not None
    keeps = _brake_obs(encoder_observations, deck_count=9)
    assert ml_policy_agent._try_ability_draw_brake(keeps, [3], config=_ABILITY_BRAKE_ON) is None


def test_ability_draw_brake_ignores_non_target_ability(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """`card_ids` に無いポケモンの特性はみどりのまい型とみなさない(誤爆防止)。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations)

    def _must_not_be_called(*a, **k):
        raise AssertionError("対象外の特性なのにKO探索を実行した")

    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy", _must_not_be_called)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    cfg = {"lethal_search": {"enabled": False},
           "ability_draw_brake": {"enabled": True, "deck_threshold": 8, "card_ids": [12345]}}
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=cfg) is None


def test_ability_draw_brake_ignores_non_ability_choices(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """選んだ手が特性でなければ(手貼り/攻撃/終了)干渉しない。"""
    obs = _brake_obs(encoder_observations)
    for chosen in ([0], [1], [4], [5]):
        assert ml_policy_agent._try_ability_draw_brake(
            obs, chosen, config=_ABILITY_BRAKE_ON) is None


# ===========================================================================
# KO成否判定: (a) 既に現在の打点でKO可能 / (b) 最大まで足してもKO不可
# ===========================================================================

def test_ability_draw_brake_fires_when_already_lethal(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """(a) 攻撃を今すぐ実行すれば足さなくてもKOできる=足す必要が無い→veto。

    実例: episode 93517227 T9 row122(アクティブのオーガポン8エネで相手アクティブ
    HP70へ攻撃すれば確実にKO。まんようしぐれ=30+30×(自エネ+相手エネ)ダメージ)。
    (b) の `can_ko_this_turn` は短絡して呼ばれないこと(必要ない探索はしない)。
    """
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))

    def _must_not_be_called(*a, **k):
        raise AssertionError("(a)で既にKO可能と分かったのに(b)の全体探索を実行した")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
    ml_policy_agent.reset_guard_stats()

    # 対象の特性(idx2, idx3)は両方除外されるので、残り{0,1,4,5}でスコア最大のidx0(手貼り
    # をベンチへ)へ差し替わる。
    out = ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON)
    assert out == [0]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_fired"] == 1
    assert stats["ability_draw_brake_fired_already_lethal"] == 1
    assert stats["ability_draw_brake_fired_futile"] == 0
    assert stats["ability_draw_brake_no_alternative"] == 0


def test_ability_draw_brake_fires_when_max_still_futile(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """(b) 攻撃は今すぐKOできないが、理論上最善(特性連打込み)でもKOに届かない=足しても無駄→veto。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(False))
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
    ml_policy_agent.reset_guard_stats()

    out = ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON)
    assert out == [0]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_fired"] == 1
    assert stats["ability_draw_brake_fired_futile"] == 1
    assert stats["ability_draw_brake_fired_already_lethal"] == 0


def test_ability_draw_brake_does_not_fire_when_pivotal(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """今すぐの攻撃ではKOできないが、理論上の最善(特性込み)ならKOに届く=このエネ加速が
    KOに寄与しうる→veto しない(温存せず使わせる)。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(False))
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(True))
    ml_policy_agent.reset_guard_stats()

    def _must_not_be_called(config=None):
        raise AssertionError("veto しない結論なのにモデルを取得した")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON) is None
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_skipped_pivotal"] == 1
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_fired"] == 0


def test_ability_draw_brake_inconclusive_search_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """(a)(b) とも判定不能なら介入しない(過度な抑制を避ける、既存ガードと同じ安全側)。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(False, aborted=True))
    ml_policy_agent.reset_guard_stats()

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON) is None
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, searched=False))
    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON) is None
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_inconclusive"] == 2
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_fired"] == 0


def test_ability_draw_brake_b_fires_even_if_a_is_inconclusive(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """(a) が判定不能でも、(b) が完走して「最大まで足してもKO不可」と分かれば単独で発火してよい
    (OR条件: 片方が完走して条件を満たせば十分)。"""
    from ptcg_ai.search import ability_draw_eval, ko_search

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(False, aborted=True))
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
    ml_policy_agent.reset_guard_stats()

    out = ml_policy_agent._try_ability_draw_brake(obs, [3], config=_ABILITY_BRAKE_ON)
    assert out == [0]
    assert ml_policy_agent.get_guard_stats()["ability_draw_brake_fired_futile"] == 1


def test_ability_draw_brake_attack_is_always_a_valid_alternative(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """選択肢が特性(みどりのまい型)とATTACKだけの局面でも、ATTACK自体は除外対象では
    ないので代替として選ばれ、``no_alternative`` は発火しない(絶対例外がATTACK存在を
    前提にするため、この veto が発火できる局面では実質的に必ず ``no_alternative`` は
    到達しない=防御用のカウンタであることを固定する)。
    """
    from ptcg_ai.search import ability_draw_eval

    obs = _brake_obs(encoder_observations, deck_count=5, options=[
        _ability(on_active=True), _ability(on_active=False), _attack(),
    ])
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.5]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_ability_draw_brake(obs, [1], config=_ABILITY_BRAKE_ON) == [2]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["ability_draw_brake_no_alternative"] == 0
    assert stats["ability_draw_brake_fired"] == 1


# ===========================================================================
# veto連鎖 / 本番不変
# ===========================================================================

def test_ability_draw_brake_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """veto連鎖(実戦の呼ばれ方)経由でも差し替わること/キー無しでは不変であること。"""
    from ptcg_ai.search import ability_draw_eval

    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ability_draw_eval, "already_ko_without_more_energy",
                        _already_ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
    assert ml_policy_agent._apply_action_vetoes(obs, [3], config=_ABILITY_BRAKE_ON) == [0]
    assert ml_policy_agent._apply_action_vetoes(obs, [3], config=_OFF) == [3]


def test_ability_draw_brake_is_inert_without_key_end_to_end(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """abl_5_full 相当(新キー無し)では常に None を返し、統計も動かない。"""
    obs = _brake_obs(encoder_observations, deck_count=5)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))
    ml_policy_agent.reset_guard_stats()
    before = ml_policy_agent.get_guard_stats()

    assert ml_policy_agent._try_ability_draw_brake(obs, [3], config=_OFF) is None
    assert ml_policy_agent._apply_action_vetoes(obs, [3], config=_OFF) == [3]
    assert ml_policy_agent.get_guard_stats() == before
