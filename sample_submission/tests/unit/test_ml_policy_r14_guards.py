"""r14「壁デッキ検知 → 非exアタッカー起用」ルート(``wall_attacker_route``)のユニットテスト。

因果(実測):
  crustle 系の壁 イワパレス(345、"Prevent all damage done to this Pokémon by attacks from
  your opponent's Pokémon {ex}.")と オーガポン いしずえのめん ex(117、"... by your
  opponent's Pokémon that have an Ability.")に対し、og_v032 唯一のアタッカー
  オーガポン みどりのめん ex(96、**ex かつ 特性持ち**)は**数学的に0ダメージ**。
  1枚を カプ・ブルル(920、非ex・特性なし・ウッドハンマー attackId 1326 = 固定220/自傷30)に
  替えると crustle 17.75%→22.75%(n=400)だが、稼働診断60試合では
  4エネ到達率 6.7% / ウッドハンマー 0.0回/試合 / エネ0のまま終了 72% =
  **ブルルは置かれているだけで一度も攻撃していなかった**。

既存の `test_ml_policy_agent.py` / `test_ml_policy_r1x_guards.py` は別担当が同時に編集して
いるため、競合を避けてこの新ファイルに追加している(検証対象は同じ `ml_policy_agent`。
r11/r12/r13 が同じ理由で増設したのと同じ流儀)。

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

# 実カードID(all_card_data() で分類・テキストを確認済み)。
_OGERPON = 96        # オーガポン みどりのめん ex(ex=True / 特性=みどりのまい)
_BULU = 920          # カプ・ブルル(ex=False / 特性なし / HP140 / にげ3 / attacks=[1326])
_CRUSTLE = 345       # イワパレス(相手exのワザダメージを完全無効)
_CORNERSTONE = 117   # オーガポン いしずえのめん ex(特性持ちのワザダメージを完全無効)
_KANGASKHAN = 678    # mid_game 既定の相手アクティブ(壁ではない対照)
_PLAIN_BASIC = 22    # ヒポポタス(たね/非ex/特性なし=壁に通る側の対照)
_GRASS = 1           # 基本【草】エネルギー
_NS_PLAN = 1221      # Nの筋書き(ベンチのエネを最大2個バトル場へ移す)
_JUDGE = 1213        # ジャッジマン(Nの筋書きでないサポートの対照)
_BOSS = 1182         # ボスの指令((b1)が唯一潰さない PLAY)
_WOOD_HAMMER = 1326  # ウッドハンマー(固定220 / 自傷30)
_MYRIAD = 120        # まんようしぐれ

_ROUTE_ON = {
    "lethal_search": {"enabled": False},
    "wall_attacker_route": {
        "enabled": True,
        "wall_card_ids": [_CRUSTLE, _CORNERSTONE],
        "attacker_card_ids": [_BULU],
        "attacker_attack_cost": 4,
        "require_zero_damage": True,
        "energy_move_card_ids": [_NS_PLAN],
        "energy_move_max": 2,
    },
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
def wall_route():
    try:
        from ptcg_ai.search import wall_attacker_route as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"wall_attacker_route / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def cfk():
    try:
        from ptcg_ai.search import closed_form_ko as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"closed_form_ko / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def encoder_observations() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _card(card_id: int, serial: int, player_index: int = 0) -> dict:
    return {"id": card_id, "playerIndex": player_index, "serial": serial}


def _mon(proto: dict, card_id: int, hp: int, energies: int, serial: int,
         player_index: int = 0) -> dict:
    slot = copy.deepcopy(proto)
    slot["id"] = card_id
    slot["hp"] = hp
    slot["maxHp"] = max(hp, 1)
    slot["serial"] = serial
    slot["playerIndex"] = player_index
    slot["energies"] = [_GRASS] * energies
    slot["energyCards"] = [
        _card(_GRASS, 500 + serial * 10 + i, player_index) for i in range(energies)
    ]
    slot["tools"] = []
    slot["preEvolution"] = []
    return slot


def _build_obs(
    encoder_observations,
    options,
    *,
    active=(_OGERPON, 210, 2),          # (card_id, hp, energies)
    bench=((_BULU, 140, 0), (_OGERPON, 210, 2)),
    opp_active=(_CRUSTLE, 150),
    opp_bench_ids=(),
    hand_ids=(_GRASS,),
    supporter_played=False,
    select_type=0,
    context=0,
    min_count=1,
    max_count=1,
):
    """mid_game を土台に、壁ルートの判定材料だけを明示した合成 obs を作る。"""
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    state = base["current"]
    me = state["players"][0]
    opp = state["players"][1]
    my_proto = copy.deepcopy(me["active"][0])
    opp_proto = copy.deepcopy(opp["active"][0])

    me["active"] = [] if active is None else [_mon(my_proto, *active, serial=50)]
    me["bench"] = [_mon(my_proto, cid, hp, en, serial=70 + i)
                   for i, (cid, hp, en) in enumerate(bench)]
    me["benchMax"] = 5
    me["hand"] = [_card(cid, 900 + i) for i, cid in enumerate(hand_ids)]
    me["handCount"] = len(hand_ids)

    opp["active"] = [_mon(opp_proto, opp_active[0], opp_active[1], 0, serial=60, player_index=1)]
    opp["bench"] = [_mon(opp_proto, cid, 150, 0, serial=80 + i, player_index=1)
                    for i, cid in enumerate(opp_bench_ids)]

    state["stadium"] = []
    state["looking"] = None
    state["supporterPlayed"] = supporter_played
    base["select"] = {
        "context": context,
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


# --- option ビルダ(cg/api.py の OptionType コメントのフィールド構成に合わせる) ---------

def _attach(hand_index: int, *, area: int, in_play_index: int) -> dict:
    """手札のエネを場のポケモンに付ける ATTACH(area=4:ACTIVE / 5:BENCH)。"""
    return {"type": 8, "area": 2, "index": hand_index,
            "inPlayArea": area, "inPlayIndex": in_play_index}


def _ability(*, on_active: bool, index: int = 0) -> dict:
    return {"type": 10, "area": 4 if on_active else 5, "index": index}


def _play(hand_index: int) -> dict:
    return {"type": 7, "index": hand_index}


def _attack(attack_id: int = _WOOD_HAMMER) -> dict:
    return {"type": 13, "attackId": attack_id}


def _retreat() -> dict:
    return {"type": 12}


def _end() -> dict:
    return {"type": 14}


def _switch_option(bench_index: int, player_index: int = 0) -> dict:
    """交代先(自分のベンチ)を指す CARD/SWITCH の選択肢。"""
    return {"type": 3, "area": 5, "index": bench_index, "playerIndex": player_index}


# ===========================================================================
# 0. 壁の無効化判定(テキスト駆動)
# ===========================================================================

def test_walls_nullify_ogerpon_but_not_bulu(wall_route):
    """壁2種は「ex」「特性持ち」のオーガポンを無効化し、非ex・特性なしのブルルは無効化しない。"""
    assert wall_route.nullifies_damage_from(_CRUSTLE, _OGERPON) is True
    assert wall_route.nullifies_damage_from(_CORNERSTONE, _OGERPON) is True
    assert wall_route.nullifies_damage_from(_CRUSTLE, _BULU) is False
    assert wall_route.nullifies_damage_from(_CORNERSTONE, _BULU) is False


def test_unknown_card_is_inconclusive(wall_route):
    """カードデータを引けない場合は None(判定不能=発火しない)。"""
    assert wall_route.nullifies_damage_from(10**9, _OGERPON) is None
    assert wall_route.nullifies_damage_from(_CRUSTLE, 10**9) is None


# ===========================================================================
# 1. closed_form_ko のウッドハンマー拡張
# ===========================================================================

def test_wood_hammer_registered_and_consistent_with_card_data(cfk):
    """ウッドハンマー(1326)が既知パターンで、定数が実カードデータと一致する。"""
    assert cfk.is_known_attack(_WOOD_HAMMER) is True
    assert cfk.self_damage_of(_WOOD_HAMMER) == 30
    assert cfk.self_damage_of(_MYRIAD) == 0
    assert cfk.verify_known_attacks() == []


def test_wood_hammer_damage_is_constant(cfk, encoder_observations):
    """固定打点型なのでエネルギーを足しても打点は 220 のまま。"""
    obs = _build_obs(encoder_observations, [_end()], active=(_BULU, 140, 4))
    state = obs.current
    assert cfk.estimate_attack_damage(state, 0, _WOOD_HAMMER) == 220
    assert cfk.estimate_attack_damage(state, 0, _WOOD_HAMMER, extra_energy=3) == 220


def test_bulu_in_play_no_longer_blinds_closed_form(cfk, encoder_observations):
    """ブルルが場に居るだけで `is_ko_impossible_this_turn` が unknown_attack にならない。

    拡張前はブルルのワザが未知パターンだったため「自分の場に1体でもブルルが居ると常に
    判定不能」=デッキ全体で r13 の閉形式ファストパスが死んでいた。
    """
    obs = _build_obs(
        encoder_observations, [_end()],
        active=(_OGERPON, 210, 1), bench=((_BULU, 140, 0),),
        opp_active=(_KANGASKHAN, 340),   # 壁でない相手(無効化テキストを踏まない)
    )
    report: dict = {}
    cfk.is_ko_impossible_this_turn(obs.current, 0, {}, report)
    assert report.get("reason") != "unknown_attack"


# ===========================================================================
# 2. キー無し = 完全不変(本番config)
# ===========================================================================

def _all_route_fns(mod):
    return (mod._try_wall_attacker_energy, mod._try_wall_attacker_retreat,
            mod._try_wall_attacker_switch, mod._try_wall_attacker_attack)


def test_absent_key_is_inert(ml_policy_agent, encoder_observations):
    """キー無し(abl_5_full 相当)では4関数すべて None、veto連鎖も統計も不変。"""
    obs = _build_obs(encoder_observations,
                     [_attach(0, area=5, in_play_index=1), _attach(0, area=5, in_play_index=0),
                      _end()])
    ml_policy_agent.reset_guard_stats()
    before = ml_policy_agent.get_guard_stats()
    for fn in _all_route_fns(ml_policy_agent):
        assert fn(obs, [0], config=_OFF) is None
        assert fn(obs, [0], config={}) is None
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_OFF) == [0]
    assert ml_policy_agent.get_guard_stats() == before


def test_no_fire_without_wall_on_field(ml_policy_agent, encoder_observations):
    """相手の場に壁が1体も見えなければ、他が揃っていても発火しない。"""
    obs = _build_obs(
        encoder_observations,
        [_attach(0, area=5, in_play_index=1), _attach(0, area=5, in_play_index=0), _end()],
        opp_active=(_KANGASKHAN, 340),
    )
    for fn in _all_route_fns(ml_policy_agent):
        assert fn(obs, [0], config=_ROUTE_ON) is None


def test_no_fire_without_attacker_in_play(ml_policy_agent, encoder_observations):
    """自分の場に指定アタッカー(ブルル)が居なければ発火しない。"""
    obs = _build_obs(
        encoder_observations,
        [_attach(0, area=5, in_play_index=0), _attach(0, area=4, in_play_index=0), _end()],
        bench=((_OGERPON, 210, 2),),
    )
    for fn in _all_route_fns(ml_policy_agent):
        assert fn(obs, [0], config=_ROUTE_ON) is None


def test_no_fire_when_current_attacker_can_damage_the_wall(ml_policy_agent, encoder_observations):
    """``require_zero_damage``: 現アクティブが壁に通る(非ex・特性なし)なら発火しない。

    ヒポポタス(22、たね/非ex/特性なし)を現アクティブにすると、壁の無効化条件
    (ex / 特性持ち)にどちらも当たらない=打点0が確定しないので、ルート全体が黙る。
    """
    obs = _build_obs(
        encoder_observations,
        [_attach(0, area=5, in_play_index=1), _attach(0, area=5, in_play_index=0),
         _attack(_MYRIAD), _retreat(), _end()],
        active=(_PLAIN_BASIC, 90, 2),
        bench=((_BULU, 140, 4), (_OGERPON, 210, 2)),
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [2], config=_ROUTE_ON) is None


def test_require_zero_damage_false_lets_the_route_fire(ml_policy_agent, encoder_observations):
    """``require_zero_damage=false`` にすると、打点0が確定しなくても壁の存在だけで発火する
    (ablation 用の逃がし口が効いていることの確認)。"""
    cfg = {
        "lethal_search": {"enabled": False},
        "wall_attacker_route": {**_ROUTE_ON["wall_attacker_route"],
                                "require_zero_damage": False},
    }
    obs = _build_obs(
        encoder_observations,
        [_attach(0, area=5, in_play_index=1), _attach(0, area=5, in_play_index=0), _end()],
        active=(_PLAIN_BASIC, 90, 2),
        bench=((_BULU, 140, 0), (_OGERPON, 210, 2)),
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=cfg) == [1]


# ===========================================================================
# 3. (a1) エネルギーの行き先をブルルへ
# ===========================================================================

def _energy_obs(encoder_observations, **kwargs):
    """options = [0: ベンチのオーガポンへ手貼り, 1: ブルルへ手貼り, 2: アクティブへ手貼り, 3: END]。

    bench = [0]=ブルル / [1]=オーガポン。
    """
    kwargs.setdefault("options", [
        _attach(0, area=5, in_play_index=1),
        _attach(0, area=5, in_play_index=0),
        _attach(0, area=4, in_play_index=0),
        _end(),
    ])
    options = kwargs.pop("options")
    return _build_obs(encoder_observations, options, **kwargs)


def test_energy_redirect_to_attacker(ml_policy_agent, encoder_observations):
    """ベンチのオーガポンへの手貼りを、ブルルへの手貼りに差し替える。"""
    ml_policy_agent.reset_guard_stats()
    obs = _energy_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["wall_attacker_energy_fired"] == 1


def test_energy_redirect_also_from_active_target(ml_policy_agent, encoder_observations):
    """アクティブ(打点0のオーガポン)への手貼りも、ブルルへ振り替える。"""
    obs = _energy_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_energy(obs, [2], config=_ROUTE_ON) == [1]


def test_energy_redirect_no_fire_when_already_targeting_attacker(
    ml_policy_agent, encoder_observations
):
    """既にブルルへ付ける手を選んでいれば介入しない。"""
    obs = _energy_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_energy(obs, [1], config=_ROUTE_ON) is None


def test_energy_redirect_no_fire_when_attacker_is_full(ml_policy_agent, encoder_observations):
    """ブルルが既に必要数(4)に到達していれば、これ以上寄せない。"""
    obs = _energy_obs(encoder_observations, bench=((_BULU, 140, 4), (_OGERPON, 210, 2)))
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None


def test_energy_redirect_no_fire_for_non_energy_choice(ml_policy_agent, encoder_observations):
    """エネ装着でない手(サポートのPLAY等)には (a1) は干渉しない。"""
    obs = _energy_obs(
        encoder_observations,
        options=[_play(0), _attach(1, area=5, in_play_index=0), _end()],
        hand_ids=(_JUDGE, _GRASS),
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None


def test_energy_redirect_counts_missing_option(ml_policy_agent, encoder_observations):
    """ブルルへ付けられる選択肢が1つも無ければ介入せず、理由をカウントする。"""
    ml_policy_agent.reset_guard_stats()
    obs = _energy_obs(encoder_observations, options=[
        _attach(0, area=5, in_play_index=1), _attach(0, area=4, in_play_index=0), _end(),
    ])
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None
    assert ml_policy_agent.get_guard_stats()["wall_attacker_energy_no_option"] == 1


def test_energy_redirect_yields_to_confirmed_ko(ml_policy_agent, encoder_observations, monkeypatch):
    """アクティブへの手貼りで「+1エネでKO確定」なら譲る(閉形式が True のときだけ)。"""
    from ptcg_ai.search import closed_form_ko

    monkeypatch.setattr(closed_form_ko, "can_ko_with_more_energy",
                        lambda *a, **k: True)
    ml_policy_agent.reset_guard_stats()
    obs = _energy_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_energy(obs, [2], config=_ROUTE_ON) is None
    assert ml_policy_agent.get_guard_stats()["wall_attacker_energy_skipped_ko"] == 1


# ===========================================================================
# 4. (a2) Nの筋書き優先
# ===========================================================================

def test_ns_plan_prioritized_when_attacker_is_active_and_short(
    ml_policy_agent, encoder_observations
):
    """ブルルがバトル場でエネ2、ベンチのオーガポンにエネ2、Nの筋書きが手札 → PLAY に差し替え。"""
    ml_policy_agent.reset_guard_stats()
    obs = _build_obs(
        encoder_observations, [_end(), _play(0)],
        active=(_BULU, 140, 2), bench=((_OGERPON, 210, 2),),
        hand_ids=(_NS_PLAN,),
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["wall_attacker_energy_move_fired"] == 1


def test_ns_plan_not_used_when_supporter_already_played(ml_policy_agent, encoder_observations):
    """このターン既にサポートを使っていれば「届く見込み」に数えない。"""
    obs = _build_obs(
        encoder_observations, [_end(), _play(0)],
        active=(_BULU, 140, 2), bench=((_OGERPON, 210, 2),),
        hand_ids=(_NS_PLAN,), supporter_played=True,
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None


def test_ns_plan_not_used_when_bench_energy_insufficient(ml_policy_agent, encoder_observations):
    """移せるエネ(最大2)を足しても必要数に届かないなら差し替えない。"""
    obs = _build_obs(
        encoder_observations, [_end(), _play(0)],
        active=(_BULU, 140, 1), bench=((_OGERPON, 210, 2),),
        hand_ids=(_NS_PLAN,),
    )
    assert ml_policy_agent._try_wall_attacker_energy(obs, [0], config=_ROUTE_ON) is None


# ===========================================================================
# 5. (b1) にげてブルルを前に出す
# ===========================================================================

def _retreat_obs(encoder_observations, **kwargs):
    kwargs.setdefault("options", [_attack(_MYRIAD), _retreat(), _end()])
    kwargs.setdefault("bench", ((_BULU, 140, 4), (_OGERPON, 210, 2)))
    options = kwargs.pop("options")
    return _build_obs(encoder_observations, options, **kwargs)


def test_retreat_fires_when_attacker_is_ready(ml_policy_agent, encoder_observations):
    """壁がバトル場・現アクティブは打点0・ブルルは4エネ → にげるへ差し替え。"""
    ml_policy_agent.reset_guard_stats()
    obs = _retreat_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["wall_attacker_retreat_fired"] == 1


def test_retreat_fires_against_cornerstone_wall_too(ml_policy_agent, encoder_observations):
    """壁が いしずえのめん ex(特性持ちを無効化)でも同じく発火する。"""
    obs = _retreat_obs(encoder_observations, opp_active=(_CORNERSTONE, 210))
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) == [1]


def test_retreat_fires_from_end_choice(ml_policy_agent, encoder_observations):
    """選んだ手が END でも発火する。"""
    obs = _retreat_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [2], config=_ROUTE_ON) == [1]


def test_retreat_no_fire_when_attacker_not_ready(ml_policy_agent, encoder_observations):
    """ブルルのエネが足りず、Nの筋書きも無いなら出さない(的になるだけ)。"""
    obs = _retreat_obs(encoder_observations, bench=((_BULU, 140, 1), (_OGERPON, 210, 2)))
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) is None


def test_retreat_fires_when_ns_plan_can_top_up(ml_policy_agent, encoder_observations):
    """ブルル2エネ + ベンチのオーガポン2エネ + Nの筋書き = 出してから届く見込み → 発火。"""
    obs = _retreat_obs(
        encoder_observations,
        bench=((_BULU, 140, 2), (_OGERPON, 210, 2)), hand_ids=(_NS_PLAN,),
    )
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) == [1]


def test_retreat_no_fire_without_retreat_option(ml_policy_agent, encoder_observations):
    """RETREAT が合法選択肢に無ければ発火しない。"""
    obs = _retreat_obs(encoder_observations, options=[_attack(_MYRIAD), _end()])
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) is None


def test_retreat_no_fire_when_wall_is_only_on_bench(ml_policy_agent, encoder_observations):
    """壁がベンチに控えているだけ(バトル場は殴れる相手)なら、にげない。"""
    obs = _retreat_obs(
        encoder_observations, opp_active=(_KANGASKHAN, 340), opp_bench_ids=(_CRUSTLE,),
    )
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) is None


def test_retreat_overrides_development_moves(ml_policy_agent, encoder_observations):
    """RETREAT はターンを終わらせない(次の decision で同じ手を選び直せる)ので、
    展開手(PLAY/ABILITY)も差し替え対象にする。"""
    play = _retreat_obs(encoder_observations,
                        options=[_play(0), _retreat(), _end()], hand_ids=(_JUDGE,))
    assert ml_policy_agent._try_wall_attacker_retreat(play, [0], config=_ROUTE_ON) == [1]

    ability = _retreat_obs(encoder_observations,
                           options=[_ability(on_active=True), _retreat(), _end()])
    assert ml_policy_agent._try_wall_attacker_retreat(ability, [0], config=_ROUTE_ON) == [1]


def test_retreat_never_overrides_boss_orders(ml_policy_agent, encoder_observations):
    """ボスの指令(1182、そのターンのKO計画そのもの)の PLAY だけは潰さない。"""
    obs = _retreat_obs(encoder_observations,
                       options=[_play(0), _retreat(), _end()], hand_ids=(_BOSS,))
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) is None


def test_retreat_override_types_can_restrict_back_to_attack_end(
    ml_policy_agent, encoder_observations
):
    """``retreat_override_types`` で初版(ATTACK/END 限定)に戻せる(ablation 用の逃がし口)。"""
    cfg = {
        "lethal_search": {"enabled": False},
        "wall_attacker_route": {**_ROUTE_ON["wall_attacker_route"],
                                "retreat_override_types": ["ATTACK", "END"]},
    }
    obs = _retreat_obs(encoder_observations,
                       options=[_play(0), _retreat(), _end()], hand_ids=(_JUDGE,))
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=cfg) is None
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [2], config=cfg) == [1]


def test_retreat_yields_when_ko_is_available(ml_policy_agent, encoder_observations, monkeypatch):
    """番人: 閉形式が「今すぐKOできる」と確定したら譲る。"""
    from ptcg_ai.search import closed_form_ko

    monkeypatch.setattr(closed_form_ko, "can_ko_now", lambda *a, **k: True)
    ml_policy_agent.reset_guard_stats()
    obs = _retreat_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_retreat(obs, [0], config=_ROUTE_ON) is None
    assert ml_policy_agent.get_guard_stats()["wall_attacker_retreat_skipped_ko"] == 1


# ===========================================================================
# 6. (b2) 交代先の選択でブルルを選ぶ
# ===========================================================================

def _switch_obs(encoder_observations, **kwargs):
    """CARD / SWITCH(交代先=自分のベンチ)。options = [0: オーガポン(bench1), 1: ブルル(bench0)]。"""
    kwargs.setdefault("options", [_switch_option(1), _switch_option(0)])
    kwargs.setdefault("bench", ((_BULU, 140, 4), (_OGERPON, 210, 2)))
    kwargs.setdefault("select_type", 1)   # SelectType.CARD
    kwargs.setdefault("context", 3)       # SelectContext.SWITCH
    options = kwargs.pop("options")
    return _build_obs(encoder_observations, options, **kwargs)


def test_switch_picks_attacker(ml_policy_agent, encoder_observations):
    """壁がバトル場でブルルが撃てるなら、交代先をブルルに差し替える。"""
    ml_policy_agent.reset_guard_stats()
    obs = _switch_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_switch(obs, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["wall_attacker_switch_fired"] == 1


def test_switch_no_fire_when_already_picking_attacker(ml_policy_agent, encoder_observations):
    """既にブルルを選んでいれば介入しない。"""
    obs = _switch_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_switch(obs, [1], config=_ROUTE_ON) is None


def test_switch_no_fire_when_attacker_not_ready(ml_policy_agent, encoder_observations):
    """エネが足りないブルルを前に出す交代はしない。"""
    obs = _switch_obs(encoder_observations, bench=((_BULU, 140, 0), (_OGERPON, 210, 2)))
    assert ml_policy_agent._try_wall_attacker_switch(obs, [0], config=_ROUTE_ON) is None


def test_switch_ignores_opponent_bench_options(ml_policy_agent, encoder_observations):
    """ボスの指令の対象選択(相手のベンチ=playerIndex が違う)は巻き込まない。"""
    obs = _switch_obs(
        encoder_observations,
        options=[_switch_option(0, player_index=1), _switch_option(1, player_index=1)],
    )
    assert ml_policy_agent._try_wall_attacker_switch(obs, [0], config=_ROUTE_ON) is None


def test_switch_works_when_active_slot_is_empty(ml_policy_agent, encoder_observations):
    """バトル場がKOされて空(新しいバトルポケモンを選ぶ)局面でもブルルを選べる。"""
    obs = _switch_obs(encoder_observations, active=None)
    assert ml_policy_agent._try_wall_attacker_switch(obs, [0], config=_ROUTE_ON) == [1]


def test_switch_no_fire_when_wall_is_not_active(ml_policy_agent, encoder_observations):
    """壁がバトル場に居ないなら交代先を強制しない。"""
    obs = _switch_obs(encoder_observations,
                      opp_active=(_KANGASKHAN, 340), opp_bench_ids=(_CRUSTLE,))
    assert ml_policy_agent._try_wall_attacker_switch(obs, [0], config=_ROUTE_ON) is None


# ===========================================================================
# 7. (c) ブルルで殴る
# ===========================================================================

def _attack_obs(encoder_observations, **kwargs):
    kwargs.setdefault("options", [_end(), _attack(_WOOD_HAMMER)])
    kwargs.setdefault("active", (_BULU, 140, 4))
    kwargs.setdefault("bench", ((_OGERPON, 210, 2),))
    options = kwargs.pop("options")
    return _build_obs(encoder_observations, options, **kwargs)


def test_attack_replaces_end(ml_policy_agent, encoder_observations):
    """ブルルがバトル場で撃てるのに END を選んだら ATTACK に戻す。"""
    ml_policy_agent.reset_guard_stats()
    obs = _attack_obs(encoder_observations)
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["wall_attacker_attack_fired"] == 1


def test_attack_replaces_retreat(ml_policy_agent, encoder_observations):
    """せっかく前に出したブルルを下げる RETREAT も ATTACK に戻す。"""
    obs = _attack_obs(encoder_observations, options=[_retreat(), _attack(_WOOD_HAMMER)])
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) == [1]


def test_attack_does_not_replace_development_moves(ml_policy_agent, encoder_observations):
    """既定では PLAY/ATTACH 等の展開手は潰さない(ATTACKを選ぶとターンが終わるため)。"""
    obs = _attack_obs(
        encoder_observations,
        options=[_play(0), _attack(_WOOD_HAMMER), _end()], hand_ids=(_JUDGE,),
    )
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) is None


def test_attack_override_types_can_be_widened_by_config(ml_policy_agent, encoder_observations):
    """``attack_override_types`` で仕様どおり全種類に広げられる。"""
    cfg = {
        "lethal_search": {"enabled": False},
        "wall_attacker_route": {**_ROUTE_ON["wall_attacker_route"],
                                "attack_override_types": ["END", "RETREAT", "PLAY"]},
    }
    obs = _attack_obs(
        encoder_observations,
        options=[_play(0), _attack(_WOOD_HAMMER), _end()], hand_ids=(_JUDGE,),
    )
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=cfg) == [1]


def test_attack_no_fire_without_attack_option(ml_policy_agent, encoder_observations):
    """エネ不足等で ATTACK 選択肢が出ていなければ発火しない。"""
    obs = _attack_obs(encoder_observations, options=[_end(), _retreat()],
                      active=(_BULU, 140, 1))
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) is None


def test_attack_skips_pure_self_ko(ml_policy_agent, encoder_observations):
    """自傷30で自分だけ落ちる(相手はKOできない)なら撃たない。"""
    ml_policy_agent.reset_guard_stats()
    obs = _attack_obs(encoder_observations, active=(_BULU, 20, 4),
                      opp_active=(_CORNERSTONE, 300))
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) is None
    assert ml_policy_agent.get_guard_stats()["wall_attacker_attack_skipped_self_ko"] == 1


def test_attack_fires_when_self_ko_trades_for_a_ko(ml_policy_agent, encoder_observations):
    """自傷で落ちても相手をKOできる見込みなら撃つ(交換は成立している)。"""
    obs = _attack_obs(encoder_observations, active=(_BULU, 20, 4),
                      opp_active=(_CRUSTLE, 150))
    assert ml_policy_agent._try_wall_attacker_attack(obs, [0], config=_ROUTE_ON) == [1]


# ===========================================================================
# 8. veto連鎖(実戦の呼ばれ方)
# ===========================================================================

def test_route_wired_into_apply_action_vetoes(ml_policy_agent, encoder_observations):
    """4種すべてが veto 連鎖経由で効き、キー無しでは不変であること。"""
    energy = _energy_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(energy, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(energy, [0], config=_OFF) == [0]

    retreat = _retreat_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(retreat, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(retreat, [0], config=_OFF) == [0]

    switch = _switch_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(switch, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(switch, [0], config=_OFF) == [0]

    attack = _attack_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(attack, [0], config=_ROUTE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(attack, [0], config=_OFF) == [0]


def test_production_config_has_no_wall_route_key(ml_policy_agent):
    """本番 config(abl_5_full)に新キーが無いこと=挙動完全不変の担保。"""
    from ptcg_ai.core.config import load_config

    assert "wall_attacker_route" not in (load_config("abl_5_full") or {})
    assert "wall_attacker_route" in (load_config("abl_5_full_og_r14") or {})
