"""r12「テラスタル退避」ゲート(``terastal_rotation``)のユニットテスト。

オーガポンex系の特性「テラスタル」=「このポケモンは、ベンチにいるかぎり、ワザのダメージを
受けない」(``CardData.tera``)。傷ついたアクティブを、より健康な同族のベンチ個体と入れ替える
のは、にげるコストが軽い限り原則ノーリスク。実ラダーの実例:

  - episode 93408551 T10(決定的): アクティブ HP10/エネ4、ベンチに HP150/エネ4 の健康な
    個体。ブライア+KOで3サイド取ったのは正しいが、「にげる(コスト1)→HP150の個体で攻撃」
    でも同じKO・同じ3サイドが取れた上に、HP10の個体をベンチへ退避できた。
  - episode 93473767 T11: アクティブHP60/260、ベンチHP240/240・エネ3。KOは元々不可能
    なので、傷んだ個体をベンチに逃がすのが明確に上。
  - 反例(発火してはいけない): 相手がドラパルトex(121)のとき、ファントムダイブは
    ダメージ**カウンタを置く**効果でありテラスタルを貫通する。

既存の `test_ml_policy_agent.py` / `test_ml_policy_r10_guards.py` / `test_ml_policy_r11_guards.py`
は別担当が同時に編集しているため、競合を避けてこの新ファイルに追加している(検証対象は同じ
`ml_policy_agent`。r11が同じ理由でr10を書き換えずに増設したのと同じ流儀)。

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

# 実カードID(data/JP_Card_Data.csv / all_card_data() で分類を確認済み)。
_OGERPON = 96       # オーガポン みどりのめんex(たね、テラスタル、まんようしぐれ=静的damage 30)
_SHAYMIN = 343       # mid_game 既定の自分アクティブ(非テラスタル、対照用)
_HARIYAMA = 674      # 非テラスタル・ワイルドプレス静的damage 210(対照用の高打点相手)
_DRAGAPULT = 121     # ドラパルトex(テラスタル。ファントムダイブがベンチダメカンでテラスタルを貫通)

_ROTATION_ON = {
    "lethal_search": {"enabled": False},
    "terastal_rotation": {"enabled": True, "damage_margin": 0,
                           "skip_vs_bench_damage_archetypes": [_DRAGAPULT]},
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


def _build_obs(
    encoder_observations,
    options,
    *,
    active_id=None,
    active_hp=None,
    bench=None,           # list[(card_id, hp)] : 自分のベンチ
    opp_active_id=None,
    opp_active_hp=None,
    opp_bench_ids=None,   # list[card_id] : 相手のベンチ
    bench_max=5,
    select_type=0,
    min_count=1,
    max_count=1,
):
    """mid_game を土台に、``terastal_rotation`` の判定材料だけを明示した合成 obs を作る
    (`test_ml_policy_r10_guards.py`/`test_ml_policy_r11_guards.py` と同じ流儀)。
    """
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    state = base["current"]
    me = state["players"][0]
    opp = state["players"][1]

    if active_id is not None:
        me["active"][0]["id"] = active_id
    if active_hp is not None:
        me["active"][0]["hp"] = active_hp
        me["active"][0]["maxHp"] = max(active_hp, me["active"][0].get("maxHp") or active_hp)

    if bench is not None:
        proto = copy.deepcopy(me["bench"][0])
        new_bench = []
        for i, (cid, hp) in enumerate(bench):
            slot = copy.deepcopy(proto)
            slot["id"] = cid
            slot["hp"] = hp
            slot["maxHp"] = max(hp, slot.get("maxHp") or hp)
            slot["serial"] = 700 + i
            new_bench.append(slot)
        me["bench"] = new_bench
    me["benchMax"] = bench_max

    if opp_active_id is not None:
        opp["active"][0]["id"] = opp_active_id
    if opp_active_hp is not None:
        opp["active"][0]["hp"] = opp_active_hp

    if opp_bench_ids is not None:
        oproto = copy.deepcopy(opp["bench"][0] if opp["bench"] else opp["active"][0])
        new_obench = []
        for i, cid in enumerate(opp_bench_ids):
            slot = copy.deepcopy(oproto)
            slot["id"] = cid
            slot["serial"] = 800 + i
            new_obench.append(slot)
        opp["bench"] = new_obench

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
# option ビルダ
# ---------------------------------------------------------------------------

def _attack(attack_id: int = 120) -> dict:
    return {"type": 13, "attackId": attack_id}


def _retreat() -> dict:
    return {"type": 12}


def _end() -> dict:
    return {"type": 14}


def _play(hand_index: int = 0) -> dict:
    return {"type": 7, "index": hand_index}


def _default_options() -> list[dict]:
    return [_attack(), _retreat(), _end()]


def _rotation_obs(encoder_observations, **kwargs):
    """episode 93408551 T10 相当(アクティブHP10/健康なベンチHP150)を既定にした合成局面。"""
    kwargs.setdefault("active_id", _OGERPON)
    kwargs.setdefault("active_hp", 10)
    kwargs.setdefault("bench", [(_OGERPON, 150)])
    kwargs.setdefault("opp_active_id", _HARIYAMA)  # 静的damage 210(非スキップ対象)
    kwargs.setdefault("opp_active_hp", 140)
    options = kwargs.pop("options", _default_options())
    return _build_obs(encoder_observations, options, **kwargs)


def _parity_stub(parity: bool, *, current_ko: bool = True):
    """`retreat_safety_eval.evaluate` のスタブ。"""

    def _f(obs, factory, retreat_index, target_bench_index, config=None, deadline=None):
        return {"current_ko": current_ko, "after_retreat_ko": parity if current_ko else None,
                "parity": parity}

    return _f


def _none_stub():
    def _f(obs, factory, retreat_index, target_bench_index, config=None, deadline=None):
        return None
    return _f


# ===========================================================================
# キー無し = 常に不介入(本番不変)
# ===========================================================================

def test_terastal_rotation_absent_key_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し=本番configでは常にNone(KO整合性探索にすら到達しない)。"""
    from ptcg_ai.search import retreat_safety_eval

    obs = _rotation_obs(encoder_observations)

    def _must_not_be_called(*a, **k):
        raise AssertionError("terastal_rotation disabled なのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_OFF) is None
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config={}) is None


def test_terastal_rotation_is_inert_without_key_end_to_end(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """abl_5_full 相当(新キー無し)では常に None を返し、統計も動かない。"""
    obs = _rotation_obs(encoder_observations)
    ml_policy_agent.reset_guard_stats()
    before = ml_policy_agent.get_guard_stats()

    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_OFF) is None
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_OFF) == [0]
    assert ml_policy_agent.get_guard_stats() == before


# ===========================================================================
# 選んだ手の種類 / テラスタル判定 / にげる合法性
# ===========================================================================

def test_terastal_rotation_ignores_non_attack_end_choices(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """選んだ手が ATTACK/END 以外(PLAY等)なら干渉しない。"""
    from ptcg_ai.search import retreat_safety_eval

    obs = _rotation_obs(encoder_observations, options=[_play(), _attack(), _retreat(), _end()])

    def _must_not_be_called(*a, **k):
        raise AssertionError("ATTACK/END以外の手なのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None


def test_terastal_rotation_requires_active_to_be_tera(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """アクティブがテラスタルでなければ、傷んでいて健康な退避先があっても発火しない。"""
    from ptcg_ai.search import retreat_safety_eval

    obs = _rotation_obs(
        encoder_observations, active_id=_SHAYMIN, active_hp=1, bench=[(_SHAYMIN, 80)],
    )

    def _must_not_be_called(*a, **k):
        raise AssertionError("非テラスタルのアクティブなのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None


def test_terastal_rotation_requires_legal_retreat_option(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """RETREAT が合法選択肢に無い(にげるコストを払えない等)なら発火しない。"""
    from ptcg_ai.search import retreat_safety_eval

    obs = _rotation_obs(encoder_observations, options=[_attack(), _end()])

    def _must_not_be_called(*a, **k):
        raise AssertionError("RETREAT選択肢が無いのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None


# ===========================================================================
# 条件2: 健康な退避先の有無
# ===========================================================================

def test_terastal_rotation_no_fire_without_healthy_bench_target(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ベンチが空/より健康でない/テラスタルでない、のいずれでも発火しない。"""
    from ptcg_ai.search import retreat_safety_eval

    def _must_not_be_called(*a, **k):
        raise AssertionError("健康な退避先が無いのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)

    empty_bench = _rotation_obs(encoder_observations, bench=[])
    assert ml_policy_agent._try_terastal_rotation(empty_bench, [0], config=_ROTATION_ON) is None

    not_healthier = _rotation_obs(encoder_observations, bench=[(_OGERPON, 10)])
    assert ml_policy_agent._try_terastal_rotation(not_healthier, [0], config=_ROTATION_ON) is None

    not_tera = _rotation_obs(encoder_observations, bench=[(_SHAYMIN, 150)])
    assert ml_policy_agent._try_terastal_rotation(not_tera, [0], config=_ROTATION_ON) is None


def test_terastal_rotation_picks_healthiest_tera_candidate(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """複数の健康な退避先候補がいれば、最もHPが高い個体を ``target_bench_index`` に使う。"""
    from ptcg_ai.search import retreat_safety_eval

    captured = {}

    def _capture(obs, factory, retreat_index, target_bench_index, config=None, deadline=None):
        captured["target_bench_index"] = target_bench_index
        return {"current_ko": False, "after_retreat_ko": None, "parity": True}

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _capture)

    # bench[0]=非テラスタル(候補外), bench[1]=健康度中位, bench[2]=最大HP(=選ばれるべき)
    obs = _rotation_obs(
        encoder_observations,
        bench=[(_SHAYMIN, 200), (_OGERPON, 100), (_OGERPON, 180)],
    )
    out = ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON)
    assert out == [1]  # RETREATのインデックス(options=[attack, retreat, end])
    assert captured["target_bench_index"] == 2


# ===========================================================================
# 条件1: 相手の想定1発打点とのHP比較
# ===========================================================================

def test_terastal_rotation_hp_threshold_boundary(ml_policy_agent, encoder_observations, monkeypatch):
    """相手アクティブの静的最大打点(ハリテヤマ=210)以下なら発火、超えるなら不発火。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True))

    # ベンチは常に十分健康(999)にして、条件2(健康な退避先)を固定した上でHP閾値だけを動かす。
    at_threshold = _rotation_obs(encoder_observations, active_hp=210, bench=[(_OGERPON, 999)])
    assert ml_policy_agent._try_terastal_rotation(at_threshold, [0], config=_ROTATION_ON) == [1]

    above_threshold = _rotation_obs(encoder_observations, active_hp=211, bench=[(_OGERPON, 999)])
    assert ml_policy_agent._try_terastal_rotation(above_threshold, [0], config=_ROTATION_ON) is None


def test_terastal_rotation_damage_margin_shifts_threshold(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """``damage_margin`` を足すと閾値がその分だけ緩む。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True))
    cfg = {
        "lethal_search": {"enabled": False},
        "terastal_rotation": {"enabled": True, "damage_margin": 20,
                               "skip_vs_bench_damage_archetypes": [_DRAGAPULT]},
    }
    obs = _rotation_obs(encoder_observations, active_hp=225, bench=[(_OGERPON, 999)])  # 210+20 以下
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=cfg) == [1]


def test_terastal_rotation_unknown_opponent_active_does_not_fire(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """相手アクティブが伏せ中/不明(None)なら条件1判定不能=安全側で不介入。"""
    from ptcg_ai.search import retreat_safety_eval

    def _must_not_be_called(*a, **k):
        raise AssertionError("相手アクティブ不明なのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)

    obs = _rotation_obs(encoder_observations)
    obs.current.players[1].active = [None]
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None


# ===========================================================================
# 条件4: ドラパルトex等(ベンチダメ貫通アーキタイプ)対面では発火しない
# ===========================================================================

def test_terastal_rotation_skips_when_opponent_active_is_dragapult(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """相手アクティブがドラパルトexなら、他条件が揃っていても発火しない。"""
    from ptcg_ai.search import retreat_safety_eval

    def _must_not_be_called(*a, **k):
        raise AssertionError("ドラパルトex対面なのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)
    ml_policy_agent.reset_guard_stats()

    obs = _rotation_obs(encoder_observations, opp_active_id=_DRAGAPULT, opp_active_hp=140)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None
    assert ml_policy_agent.get_guard_stats()["terastal_rotation_skipped_bench_damage_archetype"] == 1


def test_terastal_rotation_skips_when_opponent_bench_has_dragapult(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ドラパルトexがベンチに居るだけでも発火しない(アクティブでなくても対面リスクは残る)。"""
    from ptcg_ai.search import retreat_safety_eval

    def _must_not_be_called(*a, **k):
        raise AssertionError("相手ベンチにドラパルトexが居るのに探索を実行した")

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _must_not_be_called)

    obs = _rotation_obs(encoder_observations, opp_bench_ids=[_DRAGAPULT])
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None


# ===========================================================================
# 条件3: にげてもKO成否が変わらない
# ===========================================================================

def test_terastal_rotation_fires_when_ko_parity_holds(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """episode 93408551 T10 相当: 現アクティブでKO可能で、退避先でも同じKOが取れる→発火。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True))
    ml_policy_agent.reset_guard_stats()

    obs = _rotation_obs(encoder_observations)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) == [1]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["terastal_rotation_fired"] == 1
    assert stats["terastal_rotation_skipped_ko_mismatch"] == 0
    assert stats["terastal_rotation_inconclusive"] == 0


def test_terastal_rotation_fires_when_no_ko_exists_originally(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """episode 93473767 T11 相当: 現アクティブでもKOが元々無い→条件3は自動的に満たし発火。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True, current_ko=False))
    ml_policy_agent.reset_guard_stats()

    obs = _rotation_obs(encoder_observations, active_hp=60, bench=[(_OGERPON, 240)])
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) == [1]


def test_terastal_rotation_does_not_fire_when_ko_would_be_lost(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """にげると今ターンのKOが取れなくなる(parity=False)なら発火しない。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(False))
    ml_policy_agent.reset_guard_stats()

    obs = _rotation_obs(encoder_observations)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["terastal_rotation_skipped_ko_mismatch"] == 1
    assert stats["terastal_rotation_fired"] == 0


def test_terastal_rotation_inconclusive_search_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """KO整合性探索が完走しない(None)なら、既存ガードと同じ安全側で不介入。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _none_stub())
    ml_policy_agent.reset_guard_stats()

    obs = _rotation_obs(encoder_observations)
    assert ml_policy_agent._try_terastal_rotation(obs, [0], config=_ROTATION_ON) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["terastal_rotation_inconclusive"] == 1
    assert stats["terastal_rotation_fired"] == 0


def test_terastal_rotation_fires_when_chosen_action_is_end(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """選んだ手が END(攻撃できず/攻撃を選ばずターン終了しようとしている)でも発火する。"""
    from ptcg_ai.search import retreat_safety_eval

    from cg.api import OptionType

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True, current_ko=False))
    obs = _rotation_obs(encoder_observations)
    end_idx = next(i for i, o in enumerate(obs.select.option) if o.type == OptionType.END)
    assert ml_policy_agent._try_terastal_rotation(obs, [end_idx], config=_ROTATION_ON) == [1]


# ===========================================================================
# veto連鎖
# ===========================================================================

def test_terastal_rotation_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """veto連鎖(実戦の呼ばれ方)経由でも差し替わること/キー無しでは不変であること。"""
    from ptcg_ai.search import retreat_safety_eval

    monkeypatch.setattr(retreat_safety_eval, "evaluate", _parity_stub(True))
    obs = _rotation_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_ROTATION_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_OFF) == [0]
