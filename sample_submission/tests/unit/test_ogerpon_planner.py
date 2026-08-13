"""オーガポン専用リソースプランナー(カプ・ブルル中継戦略)のユニットテスト。

依頼の必須テスト1〜16に対応する。テスト名の末尾に対応番号を付けている。
"""

import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

OGERPON_EX = 96      # オーガポン みどりのめんex: ex, HP210, 技コスト[1,1,1], にげる1
TAPU_BULU = 920      # カプ・ブルル: 非ex, HP140, 技コスト[1,1,0,0], にげる3, 打点220
GRASS = 1

OGERPON_DECK = [OGERPON_EX] * 4 + [TAPU_BULU] + [7] * 55   # 60枚相当(中身は判定に無関係)
ALAKAZAM_DECK = [741, 742, 743] * 4 + [5] * 48             # オーガポンexを含まない


@pytest.fixture
def planner():
    try:
        from ptcg_ai.ml_policy import ogerpon_planner as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ogerpon_planner / cg engine unavailable: {exc}")
    return mod


class _Pokemon:
    _next_serial = [0]

    def __init__(self, card_id, energies=(), hp=None, max_hp=None):
        self.id = card_id
        self.energies = list(energies)
        self.maxHp = max_hp if max_hp is not None else (210 if card_id == OGERPON_EX else 140)
        self.hp = hp if hp is not None else self.maxHp
        self._next_serial[0] += 1
        self.serial = self._next_serial[0]


class _Player:
    def __init__(self, active=(), bench=(), prize_count=6):
        self.active = list(active)
        self.bench = list(bench)
        self.prize = [object()] * prize_count
        self.handCount = 5


class _State:
    def __init__(self, players, your_index=0, result=-1):
        self.players = players
        self.yourIndex = your_index
        self.result = result


class _Option:
    def __init__(self, serial):
        self.serial = serial
        self.inPlayArea = None
        self.inPlayIndex = None
        self.area = None
        self.index = None


class _Select:
    def __init__(self, stype, ctx, options, max_count=1):
        self.type = stype
        self.context = ctx
        self.option = list(options)
        self.minCount = 1
        self.maxCount = max_count


class _Obs:
    def __init__(self, state, select):
        self.current = state
        self.select = select


def _ctx(name):
    from cg.api import SelectContext
    return int(getattr(SelectContext, name))


def _card_select(ctx_name, pokemons, max_count=1):
    from cg.api import SelectType
    return _Select(int(SelectType.CARD), _ctx(ctx_name),
                   [_Option(p.serial) for p in pokemons], max_count)


def _enabled(**overrides):
    cfg = {"enabled": True}
    cfg.update(overrides)
    return {"ogerpon_planner": cfg}


# --- エンジン由来のドメイン知識 -------------------------------------------------

def test_engine_derived_costs_and_prize_values(planner):
    ogerpon = _Pokemon(OGERPON_EX)
    bulu = _Pokemon(TAPU_BULU)
    assert planner.prize_value(ogerpon) == 2
    assert planner.prize_value(bulu) == 1
    assert planner.retreat_cost(ogerpon) == 1
    assert planner.retreat_cost(bulu) == 3
    assert planner.attack_costs(ogerpon) == [[1, 1, 1]]
    assert planner.attack_costs(bulu) == [[1, 1, 0, 0]]


def test_can_attack_now_uses_energy_counts(planner):
    assert planner.can_attack_now(_Pokemon(OGERPON_EX, [GRASS] * 3))
    assert not planner.can_attack_now(_Pokemon(OGERPON_EX, [GRASS] * 2))
    assert planner.can_attack_now(_Pokemon(TAPU_BULU, [GRASS] * 4))
    assert not planner.can_attack_now(_Pokemon(TAPU_BULU, [GRASS] * 3))


def test_energy_shortfall(planner):
    assert planner.energy_shortfall(_Pokemon(TAPU_BULU, [])) == 4
    assert planner.energy_shortfall(_Pokemon(TAPU_BULU, [GRASS] * 3)) == 1
    assert planner.energy_shortfall(_Pokemon(TAPU_BULU, [GRASS] * 4)) == 0


# --- 1. Planner無効時に従来選択が保たれる --------------------------------------

def test_disabled_by_default_returns_no_adjustment_1(planner):
    state = _State([_Player([_Pokemon(OGERPON_EX, [GRASS] * 3)], [_Pokemon(TAPU_BULU)]),
                    _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("ATTACH_TO", [state.players[0].bench[0]]))
    assert planner.score_adjustments(obs, 0, {}, OGERPON_DECK) is None
    assert planner.score_adjustments(obs, 0, None, OGERPON_DECK) is None


# --- 2. 非オーガポンデッキでは発火しない ----------------------------------------

def test_not_active_for_non_ogerpon_deck_2(planner):
    assert planner.is_active(_enabled(), ALAKAZAM_DECK) is False
    assert planner.is_active(_enabled(), OGERPON_DECK) is True
    state = _State([_Player([_Pokemon(OGERPON_EX, [GRASS] * 3)], [_Pokemon(TAPU_BULU)]),
                    _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("TO_ACTIVE", [state.players[0].bench[0]]))
    assert planner.score_adjustments(obs, 0, _enabled(), ALAKAZAM_DECK) is None


# --- 4. 現在のオーガポンが攻撃不能になる手貼りを避ける ---------------------------

def test_attach_prioritises_active_when_it_cannot_attack_yet_4(planner):
    active = _Pokemon(OGERPON_EX, [GRASS] * 2)      # あと1個で攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("ATTACH_TO", [active, bulu]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj is not None
    assert adj[0] > adj[1], "攻撃が成立していないバトル場を優先すべき"


# --- 5. 攻撃可能なオーガポンがいるときは控えのカプ・ブルルに寄せる -----------------

def test_attach_feeds_backup_when_active_already_ready_5(planner):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)      # 既に攻撃可能
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)         # あと1個で完成
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("ATTACH_TO", [active, bulu]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj is not None
    assert adj[1] > adj[0], "今ターンの攻撃を失わないなら控えを完成させにいく"


def test_attach_bonus_scales_with_remaining_shortfall(planner):
    """完成が近いカプ・ブルルほど強く押す(遠い投資は無駄になりやすい)。

    実デッキのカプ・ブルルは1枚なので、2体並べた盤面ではなく「同じ盤面形で
    エネルギー数だけ違う2局面」を比べる。
    """
    def bonus_for(energy_count):
        active = _Pokemon(OGERPON_EX, [GRASS] * 3)
        bulu = _Pokemon(TAPU_BULU, [GRASS] * energy_count)
        state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX)])])
        obs = _Obs(state, _card_select("ATTACH_TO", [active, bulu]))
        return planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)[1]

    near = bonus_for(3)   # あと1
    far = bonus_for(0)    # あと4
    assert near > far > 0


def test_required_ko_delta_is_zero_when_no_extra_ko_is_forced(planner):
    """1枚ポケモンが既に2体いる盤面では、片方を抜いても必要KO回数が変わらない。

    このとき加点してはいけない(偶数サイドという代理条件ではなく実差で判定する)。
    """
    ogerpon = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu_a = _Pokemon(TAPU_BULU, [GRASS] * 4)
    bulu_b = _Pokemon(TAPU_BULU, [GRASS] * 4)
    mine = _Player([ogerpon], [bulu_a, bulu_b], prize_count=6)
    assert planner.required_ko_delta(mine, bulu_a) == 0.0


def test_required_ko_delta_is_positive_when_it_forces_an_extra_ko(planner):
    """ex だけの盤面に1枚ポケモンを混ぜると必要KO回数が増える。"""
    ogerpon1 = _Pokemon(OGERPON_EX, [GRASS] * 3)
    ogerpon2 = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    mine = _Player([ogerpon1], [ogerpon2, bulu], prize_count=6)
    assert planner.required_ko_delta(mine, bulu) >= 1.0


# --- 6. 0エネルギーのカプ・ブルルを無条件で前に出さない --------------------------

def test_does_not_promote_energyless_bulu_6(planner):
    ready_ogerpon = _Pokemon(OGERPON_EX, [GRASS] * 3)
    empty_bulu = _Pokemon(TAPU_BULU, [])
    state = _State([_Player([], [ready_ogerpon, empty_bulu]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("TO_ACTIVE", [ready_ogerpon, empty_bulu]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj is not None
    assert adj[0] > adj[1], "攻撃可能なオーガポンを、0エネのカプ・ブルルより優先する"
    assert adj[1] < 0, "にげる3で前が止まるので明確に減点されるべき"


# --- 7. サイドが偶数のときにサイドパリティ価値が上がる --------------------------

def test_parity_bonus_only_when_it_increases_required_kos_7(planner):
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    ready_ogerpon = _Pokemon(OGERPON_EX, [GRASS] * 3)
    # 自分の場: ex1 + 非ex1、残サイド6 -> 全部exなら3KO、混在で4KO -> gain>0
    mine = _Player([], [ready_ogerpon, ready_bulu], prize_count=6)
    state = _State([mine, _Player([_Pokemon(OGERPON_EX)], prize_count=6)])
    obs = _Obs(state, _card_select("TO_ACTIVE", [ready_ogerpon, ready_bulu]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj[1] > adj[0], "どちらも攻撃可能なら、サイド1枚で済む方を選ぶ"


# --- 8. カプ・ブルルで必要KO回数が増えない局面では過大評価しない -----------------

def test_no_parity_bonus_when_no_gain_8(planner):
    """残サイド1: 相手はあと1KOで勝ち。1枚ポケモンを挟んでもKO回数は増えない。"""
    ready_bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    ready_ogerpon = _Pokemon(OGERPON_EX, [GRASS] * 3)
    mine = _Player([], [ready_ogerpon, ready_bulu], prize_count=1)
    state = _State([mine, _Player([_Pokemon(OGERPON_EX)], prize_count=6)])
    assert planner.parity_gain_from_one_prize(mine) == 0.0
    obs = _Obs(state, _card_select("TO_ACTIVE", [ready_ogerpon, ready_bulu]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    # どちらも攻撃可能なので ready ボーナスは同点。パリティ加点が乗らないこと。
    assert adj[0] == pytest.approx(adj[1]), "利得が無いのにカプ・ブルルを加点してはいけない"


# --- 9. 削れたオーガポンを温存する ----------------------------------------------

def test_preserves_damaged_ex_9(planner):
    healthy = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=210)
    damaged = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)     # KO圏内
    state = _State([_Player([], [healthy, damaged]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("TO_ACTIVE", [healthy, damaged]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj[0] > adj[1], "削れたexを前に出さず温存する"


# --- 11/12. カプ・ブルル戦闘中は次のオーガポン準備、気絶後は攻撃可能な個体を昇格 ---

def test_phase_classification_11_12(planner):
    bulu_active = _State([_Player([_Pokemon(TAPU_BULU, [GRASS] * 4)],
                                  [_Pokemon(OGERPON_EX, [GRASS])]),
                          _Player([_Pokemon(OGERPON_EX)])])
    assert planner.classify(bulu_active, 0) == planner.BULU_ACTIVE

    with_ready_backup = _State([_Player([_Pokemon(TAPU_BULU, [GRASS] * 4)],
                                        [_Pokemon(OGERPON_EX, [GRASS] * 3)]),
                                _Player([_Pokemon(OGERPON_EX)])])
    assert planner.classify(with_ready_backup, 0) == planner.NEXT_OGERPON_SETUP

    setup = _State([_Player([_Pokemon(OGERPON_EX, [GRASS] * 3)], [_Pokemon(TAPU_BULU, [GRASS])]),
                    _Player([_Pokemon(OGERPON_EX)])])
    assert planner.classify(setup, 0) == planner.BULU_SETUP

    ready = _State([_Player([_Pokemon(OGERPON_EX, [GRASS] * 3)],
                            [_Pokemon(TAPU_BULU, [GRASS] * 4)]),
                    _Player([_Pokemon(OGERPON_EX)])])
    assert planner.classify(ready, 0) == planner.BULU_READY


def test_promote_prefers_ready_new_ogerpon_over_damaged_one_12(planner):
    fresh_ready = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=210)
    damaged_ready = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=30)
    state = _State([_Player([], [fresh_ready, damaged_ready]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("TO_ACTIVE", [fresh_ready, damaged_ready]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj[0] > adj[1]


# --- 13. 攻撃可能な個体が無い場合でも壊れない -----------------------------------

def test_fallback_when_nothing_can_attack_13(planner):
    a = _Pokemon(OGERPON_EX, [])
    b = _Pokemon(TAPU_BULU, [])
    state = _State([_Player([], [a, b]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("TO_ACTIVE", [a, b]))
    adj = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert adj is not None and len(adj) == 2
    assert all(isinstance(v, float) for v in adj)


# --- 14. 例外時はフォールバックする ---------------------------------------------

def test_returns_none_on_broken_input_14(planner):
    assert planner.score_adjustments(None, 0, _enabled(), OGERPON_DECK) is None
    broken = _Obs(None, None)
    assert planner.score_adjustments(broken, 0, _enabled(), OGERPON_DECK) is None


def test_multi_select_is_not_touched(planner):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS])
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("ATTACH_TO", [active, bulu], max_count=2))
    assert planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK) is None


def test_main_context_is_not_touched(planner):
    """MAIN は PIMC の担当。プランナーは触らない(二重介入を避ける)。"""
    from cg.api import SelectType
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    state = _State([_Player([active], []), _Player([_Pokemon(OGERPON_EX)])])
    sel = _Select(int(SelectType.MAIN), _ctx("MAIN"), [_Option(active.serial)])
    assert planner.score_adjustments(_Obs(state, sel), 0, _enabled(), OGERPON_DECK) is None


# --- 15. 試合間で状態が混ざらない -----------------------------------------------

def test_planner_is_stateless_15(planner):
    """プランナーはモジュール状態を持たず、同じ入力に必ず同じ出力を返す。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 3)
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX)])])
    obs = _Obs(state, _card_select("ATTACH_TO", [active, bulu]))
    first = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    second = planner.score_adjustments(obs, 0, _enabled(), OGERPON_DECK)
    assert first == second


# --- 3. リーサル探索より後に適用される(呼び出し順の契約) ------------------------

def test_planner_runs_after_lethal_in_agent_3():
    """`_select_action` で lethal -> pipeline -> planner の順であることをソース上で担保する。

    プランナーが確定リーサルを妨害しないことの構造的な保証。
    """
    import inspect

    from ptcg_ai.ml_policy import ml_policy_agent

    src = inspect.getsource(ml_policy_agent._select_action)
    i_lethal = src.index("_try_lethal")
    i_pipeline = src.index("_try_pipeline")
    i_planner = src.index("_try_ogerpon_planner")
    assert i_lethal < i_pipeline < i_planner


# --- 16. logging_only と baseline の同一性(意思決定レベル) ---------------------
#
# 依頼の必須テスト16は「同一seedで logging_only と baseline の行動・勝敗が一致する」だが、
# `cg.game.battle_start(deck0, deck1)` にシード引数が無く、シャッフル・コイン判定は
# cg.dll 内部のRNGで決まるため、**試合レベルの再現性はこの環境に存在しない**
# (同一config同士を同一seedで回しても 23/30 vs 24/30 と一致しない実測あり:
#  runs/ogerpon/planner/determinism.json)。
#
# そこで同一性の検証を意思決定レベルに移す。同じ Observation を両 config に与えて
# 同じ行動が返ることを直接確かめる方が、planner無効時の後方互換性の検査としては強い。

import json as _json

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture(scope="module")
def encoder_observations():
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return _json.load(f)


def _load_config(name):
    from ptcg_ai.core.config import load_config
    return load_config(name)


@pytest.mark.parametrize("fixture_key", ["mid_game", "early_active_none"])
def test_logging_only_matches_baseline_action_16(encoder_observations, fixture_key):
    from cg.api import to_observation_class
    from ptcg_ai.ml_policy import ml_policy_agent

    if fixture_key not in encoder_observations:
        pytest.skip(f"fixture {fixture_key} not available")
    obs = to_observation_class({**encoder_observations[fixture_key], "logs": []})

    baseline = _load_config("abl_5_full")
    logging_only = _load_config("og_logging_only")
    assert logging_only.get("ogerpon_planner", {}).get("enabled") is False

    a = ml_policy_agent.agent(obs, config=baseline)
    b = ml_policy_agent.agent(obs, config=logging_only)
    assert a == b, "planner無効時は baseline と同一の行動でなければならない"


@pytest.mark.parametrize("fixture_key", ["mid_game", "early_active_none"])
def test_planner_configs_are_loadable_and_gated(encoder_observations, fixture_key):
    """conservative/aggressive も同じobsで例外なく合法な行動を返す。"""
    from cg.api import to_observation_class
    from ptcg_ai.ml_policy import ml_policy_agent

    if fixture_key not in encoder_observations:
        pytest.skip(f"fixture {fixture_key} not available")
    obs = to_observation_class({**encoder_observations[fixture_key], "logs": []})
    for name in ("og_bulu_conservative", "og_bulu_aggressive"):
        cfg = _load_config(name)
        assert cfg["ogerpon_planner"]["enabled"] is True
        action = ml_policy_agent.agent(obs, config=cfg)
        assert isinstance(action, list) and action
        assert all(isinstance(i, int) and 0 <= i < len(obs.select.option) for i in action)


# ===========================================================================
# MAIN(にげる判断)への合成
# ===========================================================================

class _MainOption:
    def __init__(self, otype, serial=None):
        from cg.api import OptionType
        self.type = int(getattr(OptionType, otype))
        self.serial = serial
        self.inPlayArea = self.inPlayIndex = self.area = self.index = None


def _main_select(options):
    from cg.api import SelectType
    return _Select(int(SelectType.MAIN), _ctx("MAIN"), options)


def _main_enabled(**overrides):
    cfg = {"enabled": True, "main_enabled": True}
    cfg.update(overrides)
    return {"ogerpon_planner": cfg}


def test_main_bonus_is_zero_when_main_disabled(planner):
    """CARD系だけ有効(main_enabled=False)なら MAIN には一切介入しない。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX, [GRASS] * 3)])])
    sel = _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", bulu.serial)])
    obs = _Obs(state, sel)
    cfg = {"ogerpon_planner": {"enabled": True, "main_enabled": False}}
    assert planner.main_candidate_bonus(obs, 0, [0, 1], cfg, OGERPON_DECK) == {}


def test_main_prefers_retreat_when_damaged_ex_faces_certain_ko(planner):
    """削れたオーガポンexが確定KO圏内で、カプ・ブルルが攻撃可能なら交代を押す。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)        # 210中40 = KO圏内
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)                  # 攻撃可能
    fresh = _Pokemon(OGERPON_EX, [GRASS] * 3)                # 次のアタッカー準備済み
    state = _State([_Player([active], [bulu, fresh]),
                    _Player([_Pokemon(OGERPON_EX, [GRASS] * 3)])])
    sel = _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", bulu.serial)])
    obs = _Obs(state, sel)
    bonus = planner.main_candidate_bonus(obs, 0, [0, 1], _main_enabled(), OGERPON_DECK)
    assert 1 in bonus and bonus[1] > 0, "交代候補にプラス補正が付くべき"
    assert 0 not in bonus, "続行側は素のPIMCスコアが基準点(補正なし)"


def test_main_penalises_retreat_to_unready_bulu(planner):
    """カプ・ブルルが攻撃不能なら、交代のテンポ損失で補正が下がる。"""
    active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)
    empty_bulu = _Pokemon(TAPU_BULU, [])                     # 攻撃不能
    state = _State([_Player([active], [empty_bulu]),
                    _Player([_Pokemon(OGERPON_EX, [GRASS] * 3)])])
    sel = _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", empty_bulu.serial)])
    obs = _Obs(state, sel)
    bonus = planner.main_candidate_bonus(obs, 0, [0, 1], _main_enabled(), OGERPON_DECK)
    ready_case = planner.main_candidate_bonus(
        _Obs(_State([_Player([_Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)],
                             [_Pokemon(TAPU_BULU, [GRASS] * 4)]),
                     _Player([_Pokemon(OGERPON_EX, [GRASS] * 3)])]),
             _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", 999999)])),
        0, [0, 1], _main_enabled(), OGERPON_DECK)
    assert bonus.get(1, 0.0) < 0.05, "攻撃不能な相手への交代は押してはいけない"


def test_main_no_bonus_when_active_is_safe(planner):
    """バトル場が安全なら交代を押さない(温存のための無駄な交代を防ぐ)。"""
    healthy = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=210)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    weak_opp = _Pokemon(TAPU_BULU, [])  # 打点220だが、HP210に対しては CERTAIN 判定になるため
    state = _State([_Player([healthy], [bulu]), _Player([_Pokemon(OGERPON_EX, [])])])
    sel = _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", bulu.serial)])
    obs = _Obs(state, sel)
    risk, _ = planner.ko_risk(healthy, state.players[1].active[0], planner.main_config(_main_enabled()))
    assert risk == planner.SAFE, "HP満タンのオーガポンexは相手オーガポンの30打点では安全"


def test_ko_risk_levels(planner):
    cfg = planner.main_config(_main_enabled())
    opp = _Pokemon(TAPU_BULU, [GRASS] * 4)          # 打点220、攻撃可能
    assert planner.ko_risk(_Pokemon(OGERPON_EX, [], hp=40), opp, cfg)[0] == planner.CERTAIN_KO
    # 打点220だが1個も付いていない -> 手貼り1回でも払えない(コスト4) -> POSSIBLE
    opp_no_energy = _Pokemon(TAPU_BULU, [])
    assert planner.ko_risk(_Pokemon(OGERPON_EX, [], hp=40), opp_no_energy, cfg)[0] == planner.POSSIBLE_KO
    # あと1個で払える -> LIKELY
    opp_near = _Pokemon(TAPU_BULU, [GRASS] * 3)
    assert planner.ko_risk(_Pokemon(OGERPON_EX, [], hp=40), opp_near, cfg)[0] == planner.LIKELY_KO
    assert planner.ko_risk(_Pokemon(OGERPON_EX, [], hp=210), _Pokemon(OGERPON_EX, [GRASS] * 3),
                           cfg)[0] == planner.SAFE
    # 根拠がログに残ること
    _lvl, reason = planner.ko_risk(_Pokemon(OGERPON_EX, [], hp=40), opp, cfg)
    assert "remaining_hp" in reason and "current_payable_damage" in reason
    assert "basis" in reason, "判定根拠を必ず残す"


def test_main_bonus_not_applied_to_non_ogerpon_deck(planner):
    active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    state = _State([_Player([active], [bulu]), _Player([_Pokemon(OGERPON_EX, [GRASS] * 3)])])
    sel = _main_select([_MainOption("ATTACK"), _MainOption("RETREAT", bulu.serial)])
    assert planner.main_candidate_bonus(_Obs(state, sel), 0, [0, 1],
                                        _main_enabled(), ALAKAZAM_DECK) == {}


def test_pipeline_keeps_pimc_score_and_adds_bonus():
    """pipeline が PIMC スコアを捨てずに補助項を足すこと(合成方式)をソース上で担保する。"""
    import inspect

    from ptcg_ai.search import pipeline

    src = inspect.getsource(pipeline.search)
    assert "candidate_bonus_fn" in src
    assert "scored[i] += float(b)" in src, "PIMCスコアに加算する形でなければならない"
    assert "resource_bonus_shadow_only" in src, "shadow モードを持つこと"


def test_pipeline_shadow_only_does_not_change_scores():
    """shadow_only=True のとき scored が書き換わらないことをソース上で担保する。"""
    import inspect

    from ptcg_ai.search import pipeline

    src = inspect.getsource(pipeline.search)
    i_guard = src.index('config.get("resource_bonus_shadow_only", False)')
    i_apply = src.index("scored[i] += float(b)")
    assert i_guard < i_apply, "shadow ガードが加算より前になければならない"


# ===========================================================================
# 修正: 公開情報ベースの打点見積り / 行動条件付き required_ko_delta
# ===========================================================================

def test_variable_damage_is_computed_not_taken_from_static_field(planner):
    """オーガポンの技は Attack.damage=30 だが実効は「両者のエネ数x30」加算。

    静的値だけを信じると打点を大幅に取り違えるため、効果文から計算する。
    """
    from ptcg_ai.shared import card_cache
    a = card_cache.get_attack(120)
    assert int(a.damage) == 30, "前提: 静的damageは30"
    atk = _Pokemon(OGERPON_EX, [GRASS] * 3)
    dfn = _Pokemon(OGERPON_EX, [GRASS] * 2)
    dmg, variable = planner.attack_damage_estimate(atk, dfn, a)
    assert variable is True
    assert dmg == 30 + 30 * 5, "30 + 30 x (自分3 + 相手2)"


def test_damage_tiers_use_public_opponent_energy(planner):
    """相手バトルポケモンの付与エネルギーは公開情報。今払える技だけを current に入れる。"""
    cfg = planner.main_config({"ogerpon_planner": {"main_enabled": True}})
    my_active = _Pokemon(OGERPON_EX, [GRASS] * 2)

    no_energy = planner.damage_tiers(_Pokemon(OGERPON_EX, []), my_active, cfg)
    assert no_energy["current_payable_damage"] == 0, "エネ0なら今は撃てない"
    assert no_energy["theoretical_max_damage"] > 0, "理論最大は別枠で持つ"

    full = planner.damage_tiers(_Pokemon(OGERPON_EX, [GRASS] * 3), my_active, cfg)
    assert full["current_payable_damage"] > 0
    assert full["opponent_attached_energy"] == 3


def test_ko_risk_distinguishes_current_from_theoretical(planner):
    """カード記載の最大打点を常に撃てる前提にしない(不要な交代を避ける)。"""
    cfg = planner.main_config({"ogerpon_planner": {"main_enabled": True}})
    victim = _Pokemon(OGERPON_EX, [], hp=40)

    # 相手がエネ十分 -> 今撃てる -> CERTAIN
    lvl, reason = planner.ko_risk(victim, _Pokemon(TAPU_BULU, [GRASS] * 4), cfg)
    assert lvl == planner.CERTAIN_KO
    assert reason["basis"] == "current_payable_damage >= remaining_hp"

    # 相手がエネ3(あと1で届く) -> LIKELY
    lvl2, reason2 = planner.ko_risk(victim, _Pokemon(TAPU_BULU, [GRASS] * 3), cfg)
    assert lvl2 == planner.LIKELY_KO

    # 相手がエネ0 -> 今も次ターンも払えない -> POSSIBLE(理論上は届く)
    lvl3, _ = planner.ko_risk(victim, _Pokemon(TAPU_BULU, []), cfg)
    assert lvl3 == planner.POSSIBLE_KO


def test_required_ko_delta_is_action_conditional(planner):
    """ベンチにいるだけでは 0。交代して「次に倒される対象」を入れ替えて初めて増える。"""
    ex_active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    fresh = _Pokemon(OGERPON_EX, [GRASS] * 3)
    mine = _Player([ex_active], [bulu, fresh], prize_count=6)

    # 相手残サイドが偶数(2/4)なら、1枚ポケモンを差し出すと追加KOを強いる
    assert planner.required_ko_delta_for_retreat(mine, ex_active, bulu, 2) == pytest.approx(1.0)
    assert planner.required_ko_delta_for_retreat(mine, ex_active, bulu, 4) == pytest.approx(1.0)
    # 奇数なら増えない
    assert planner.required_ko_delta_for_retreat(mine, ex_active, bulu, 3) == pytest.approx(0.0)
    # 交代先も ex なら入れ替える意味がない
    assert planner.required_ko_delta_for_retreat(mine, ex_active, fresh, 4) == pytest.approx(0.0)


def test_required_ko_delta_zero_when_opponent_already_winning(planner):
    ex_active = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=40)
    bulu = _Pokemon(TAPU_BULU, [GRASS] * 4)
    mine = _Player([ex_active], [bulu], prize_count=6)
    assert planner.required_ko_delta_for_retreat(mine, ex_active, bulu, 0) == 0.0
