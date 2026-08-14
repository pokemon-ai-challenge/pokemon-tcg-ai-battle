"""外部レビュー(feature/climb-per-archetype-prep HEAD 5828b5d)指摘の回帰テスト。

collect_ogerpon_counterfactuals.py / eval_bulu_loop.py / ogerpon_strategy.py /
ogerpon_strategy_model.py の修正に対応する。既存の
test_collect_ogerpon_counterfactuals.py とは分離し、レビュー対応であることを明示する。
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _RL_DIR.parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _p in (str(_RL_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OGERPON_EX = 96
TAPU_BULU = 920
GRASS = 1


@pytest.fixture
def mod():
    try:
        import collect_ogerpon_counterfactuals as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"collect_ogerpon_counterfactuals / cg engine unavailable: {exc}")
    return m


@pytest.fixture
def OS():
    from ptcg_ai.ml_policy import ogerpon_option_state as m
    return m


@pytest.fixture
def P():
    from ptcg_ai.ml_policy import ogerpon_planner as m
    return m


@pytest.fixture
def populated_W(mod, P, OS):
    """_rollout_optionが参照する _W["P"]/_W["OS"] を実モジュールで埋める
    (search_begin/search_step等の外部engine呼び出しだけをfakeに差し替えるテスト用)。
    """
    mod._W["P"] = P
    mod._W["OS"] = OS
    yield mod._W


# ===========================================================================
# #3: dummyのhidden state stubが長期rolloutに使われていない(source inspection)
# ===========================================================================

def test_dummy_hidden_state_stub_not_imported_or_used(mod):
    assert not hasattr(mod, "build_dummy_search_state"), (
        "確定リーサル探索用のダミーstubは長期counterfactual rolloutに使ってはいけない"
        "(相手の非公開領域が汎用エネルギーで埋まり、数百手のrolloutでは勝敗ラベルが信頼できない)")
    src = inspect.getsource(mod)
    assert "from ptcg_ai.hidden_information.search_state_stub import build_dummy_search_state" not in src
    assert "search_adapter.to_search_begin_kwargs" in src
    assert "match_context" in src


# ===========================================================================
# #1: Strategy Window検出はOption Stateの強制開始に妨害されない(source inspection)
# ===========================================================================

def test_detect_trigger_called_with_idle_not_advancing_state(mod, OS):
    """_process_task内でdetect_triggerに渡すmodeが、force_start=Trueで進めた
    option_stateではなく固定のOS.IDLEであることをソース上で担保する
    (外部レビュー指摘: ブルルが場に出た瞬間からTRIGGER_CONTINUEに固定され、
    main_attach/promote/retreatが収集できなくなるバグの再発防止)。
    """
    src = inspect.getsource(mod._process_task)
    assert "force_start=True" not in src, (
        "_process_task(外側の通常self-play)はOption Stateを強制開始してはいけない"
        "(まだSINGLE_PRIZE_ROTATIONを選ぶ意思決定は存在しないため)")
    assert "STRAT.detect_trigger(obs, learner_index, deck_l, OS.IDLE)" in src


# ===========================================================================
# #11: determinization seedがプロセス非依存(hash()を使っていない)
# ===========================================================================

def test_determinization_seed_matches_hardcoded_sha256_derivation(mod):
    """PYTHONHASHSEEDに依存しないことを、標準ライブラリのsha256で独立に再計算した
    期待値と突き合わせて確認する(hash()を使っていれば実行のたびに値が変わるので
    この等値は原理的に保証できない)。
    """
    import hashlib

    got = mod.determinization_seed(match_seed=42, turn=7, state_id="abc123", determinization_id=2)
    payload = b"42:7:abc123:2"
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert got == expected


def test_determinization_seed_is_deterministic_and_varies_by_input(mod):
    a = mod.determinization_seed(1, 1, "s", 0)
    b = mod.determinization_seed(1, 1, "s", 0)
    c = mod.determinization_seed(1, 1, "s", 1)
    assert a == b
    assert a != c


# ===========================================================================
# #9: loop_completeはbulu_attack_count>0かつ次アタッカー接続まで要求する
# ===========================================================================

def test_loop_complete_false_without_attack(mod, OS):
    assert mod.compute_loop_complete(OS.COMPLETE, OS, 0, True) is False


def test_loop_complete_false_without_next_attacker(mod, OS):
    assert mod.compute_loop_complete(OS.COMPLETE, OS, 1, False) is False


def test_loop_complete_false_when_not_complete(mod, OS):
    assert mod.compute_loop_complete(OS.ACTIVE, OS, 1, True) is False


def test_loop_complete_true_when_all_conditions_met(mod, OS):
    assert mod.compute_loop_complete(OS.COMPLETE, OS, 1, True) is True


# ===========================================================================
# #4, #5, #6: Option Controllerが実際にSINGLE_PRIZE_ROTATIONのrollout行動を拘束する
# ===========================================================================

class _Pokemon:
    _n = [50000]

    def __init__(self, card_id, energies=(), hp=None, max_hp=None):
        self.id = card_id
        self.energies = list(energies)
        self.maxHp = max_hp if max_hp is not None else (210 if card_id == OGERPON_EX else 140)
        self.hp = hp if hp is not None else self.maxHp
        _Pokemon._n[0] += 1
        self.serial = _Pokemon._n[0]


class _Player:
    def __init__(self, active=(), bench=(), hand=(), prize_count=6):
        self.active = list(active)
        self.bench = list(bench)
        self.hand = list(hand)
        self.prize = [object()] * prize_count
        self.handCount = len(hand)


class _State:
    def __init__(self, players, your_index=0, result=-1, turn=3):
        self.players = players
        self.yourIndex = your_index
        self.result = result
        self.turn = turn


class _Option:
    def __init__(self, otype):
        self.type = int(otype)
        self.area = self.index = self.inPlayArea = self.inPlayIndex = None
        self.serial = None
        self.cardId = None


class _Select:
    def __init__(self, options, max_count=1, stype=None):
        from cg.api import SelectType
        self.option = list(options)
        self.maxCount = max_count
        self.minCount = 1
        self.type = stype if stype is not None else int(SelectType.MAIN)


class _Obs:
    def __init__(self, state, select):
        self.current = state
        self.select = select


# #4/#5/#6(低位Controllerの各phase挙動)のテストは、その実体
# (ptcg_ai.ml_policy.ogerpon_strategy.single_prize_low_level_action)を
# sample_submission/tests/unit/test_ogerpon_strategy.py へ移設済み(本番agentと収集
# rolloutで低位方策の定義を一本化したため、そちらが正本)。

# ===========================================================================
# #2: 同一pairの2 Optionが同じdeterminization(hidden state)を使う
# ===========================================================================

def test_same_determinization_shared_across_both_options(mod, monkeypatch):
    calls = {"n": 0}

    def fake_hidden_state(obs, learner_index, rng):
        calls["n"] += 1
        return {"tag": calls["n"]}

    seen = []

    def fake_rollout(obs, hidden_state, candidate, learner_index, opp_pm, deck_ids, option_cfg):
        seen.append((candidate.option_name, hidden_state["tag"]))
        return {"win": 1, "terminal_turns": 1, "opponent_ko_count": 0,
               "bulu_attack_count": 0, "bulu_prizes_taken": 0, "loop_complete": False,
               "abort_reason": None, "error": None}

    class _FakeEnc:
        @staticmethod
        def encode_strategy_pair(obs, me, pair):
            return {"continuous_features": [], "slot_card_ids": [],
                   "option_features": {"EX_TEMPO": [], "SINGLE_PRIZE_ROTATION": []}}

    monkeypatch.setattr(mod, "build_estimated_hidden_state", fake_hidden_state)
    monkeypatch.setattr(mod, "_rollout_option", fake_rollout)
    monkeypatch.setattr(mod, "state_id_of", lambda state: "fixed_state_id")
    mod._W["ENC"] = _FakeEnc()
    mod._W["config"] = {}
    mod._W["git_commit"] = "test"

    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None,
                           target_card_id=None, turns_until_ready=0, required_ko_gain=0.0,
                           safety_flags={})
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=[1],
                               target_serial=None, target_card_id=None, turns_until_ready=0,
                               required_ko_gain=0.0, safety_flags={})
    obs = SimpleNamespace(current=SimpleNamespace())

    records = mod._process_triggered_state(
        obs, (ex_c, single_c), "main_attach", learner_index=0, match_seed=123, turn=5,
        arch="alakazam", opp_pm=None, deck_ids=[], determinizations=3,
        state_id="fixed_state_id", sample_id="fixed_sample_id")

    assert calls["n"] == 3, "hidden stateは決定化ごとに1回だけ生成し、Option間で共有するべき"
    assert len(records) == 6  # 3決定化 x 2Option

    by_tag = {}
    for name, tag in seen:
        by_tag.setdefault(tag, set()).add(name)
    assert len(by_tag) == 3
    for opts in by_tag.values():
        assert opts == {"EX_TEMPO", "SINGLE_PRIZE_ROTATION"}, "同じ決定化は両Optionで共有されるべき"


# ===========================================================================
# #7, #8, #10: _rollout_optionの統合的な挙動(fake cg.api)
# ===========================================================================

class _FakeSearchState:
    def __init__(self, search_id, observation):
        self.searchId = search_id
        self.observation = observation


class _ScriptedCgApi:
    """search_begin/search_step/search_release/search_endの最小フェイク。

    実際の合法性判定はしない(actionを無視して、あらかじめ渡された観測列を順番に
    返す)。生成したsearchId・解放されたsearchIdを記録し、#10(全経路での解放)を
    検証できるようにする。
    """

    def __init__(self, observations):
        self._observations = observations
        self._next_id = 0
        self.released_ids = []
        self.generated_ids = []
        self.search_end_called = 0

    def search_begin(self, root_obs, *hidden_state_args):
        return self._new_state(self._observations[0])

    def search_step(self, search_id, action):
        self._step_index = getattr(self, "_step_index", 0) + 1
        obs = self._observations[self._step_index]
        return self._new_state(obs)

    def _new_state(self, obs):
        self._next_id += 1
        sid = f"n{self._next_id}"
        self.generated_ids.append(sid)
        return _FakeSearchState(sid, obs)

    def search_release(self, search_id):
        self.released_ids.append(search_id)

    def search_end(self):
        self.search_end_called += 1


def _make_learner_turn_obs(active, bench, opp_active, opp_bench, turn, options, learner_index=0):
    from cg.api import SelectType

    mine = _Player(active=[active] if active else [], bench=list(bench))
    opp = _Player(active=[opp_active] if opp_active else [], bench=list(opp_bench))
    players = [mine, opp] if learner_index == 0 else [opp, mine]
    state = _State(players, your_index=learner_index, turn=turn)
    return _Obs(state, _Select(options, stype=int(SelectType.MAIN)))


def test_rollout_option_counts_ko_event_not_prize_delta(mod, monkeypatch, OS, populated_W):
    """#7: exを1体KOしても実イベント数(1)を数える(サイド差分の2ではない)。"""
    from cg.api import OptionType

    root_active = _Pokemon(TAPU_BULU, [GRASS] * 4)
    opp_ex = _Pokemon(OGERPON_EX, [GRASS] * 3, hp=1)  # すぐ倒れる想定のex(サイド2枚分)
    root_obs = _make_learner_turn_obs(
        root_active, [], opp_ex, [], turn=5,
        options=[_Option(OptionType.ATTACK)])

    # opponent turn: 攻撃を選び、直後にmineのactiveが消える(被弾KO)。
    my_active_before = _Pokemon(TAPU_BULU, [GRASS] * 4)
    opp_turn_obs = SimpleNamespace(
        current=_State(
            [_Player(active=[my_active_before]), _Player(active=[_Pokemon(OGERPON_EX, [GRASS] * 3)])],
            your_index=1, turn=6),
        select=_Select([_Option(OptionType.ATTACK)]))
    # 被弾後: mineのactiveが消えている(気絶)。
    after_opp_attack_obs = SimpleNamespace(
        current=_State([_Player(active=[]), _Player(active=[_Pokemon(OGERPON_EX, [GRASS] * 3)])],
                      your_index=0, turn=7, result=-1),
        select=None)
    # 終局。
    end_obs = SimpleNamespace(
        current=_State([_Player(active=[]), _Player(active=[])], your_index=0, turn=8, result=0),
        select=None)

    fake = _ScriptedCgApi([root_obs, opp_turn_obs, after_opp_attack_obs, end_obs])
    monkeypatch.setattr("cg.api.search_begin", fake.search_begin)
    monkeypatch.setattr("cg.api.search_step", fake.search_step)
    monkeypatch.setattr("cg.api.search_release", fake.search_release)
    monkeypatch.setattr("cg.api.search_end", fake.search_end)

    candidate = SimpleNamespace(option_name="EX_TEMPO", first_action=[0])

    def fake_opp_action(obs, opp_pm):
        return [0]

    monkeypatch.setattr(mod, "_opponent_rollout_action", fake_opp_action)
    monkeypatch.setattr(mod, "_learner_rollout_action", lambda *a, **k: [0])

    outcome = mod._rollout_option(root_obs, {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                             "opponent_prize": [], "opponent_hand": [],
                                             "opponent_active": []},
                                  candidate, learner_index=0, opp_pm=None, deck_ids=[], option_cfg={})
    assert outcome["opponent_ko_count"] == 1
    assert outcome["win"] == 1  # result=0 == learner_index(0) なので学習側の勝ち


def test_rollout_option_terminal_turns_relative_to_root(mod, monkeypatch, populated_W):
    """#8: terminal_turnsはrootのturnからの相対値。"""
    from cg.api import OptionType

    root_active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    root_obs = _make_learner_turn_obs(root_active, [], _Pokemon(OGERPON_EX, []), [], turn=10,
                                      options=[_Option(OptionType.ATTACK)])
    end_obs = SimpleNamespace(
        current=_State([_Player(active=[]), _Player(active=[])], your_index=0, turn=17, result=0),
        select=None)

    fake = _ScriptedCgApi([root_obs, end_obs])
    monkeypatch.setattr("cg.api.search_begin", fake.search_begin)
    monkeypatch.setattr("cg.api.search_step", fake.search_step)
    monkeypatch.setattr("cg.api.search_release", fake.search_release)
    monkeypatch.setattr("cg.api.search_end", fake.search_end)
    monkeypatch.setattr(mod, "_learner_rollout_action", lambda *a, **k: [0])
    monkeypatch.setattr(mod, "_opponent_rollout_action", lambda *a, **k: [0])

    candidate = SimpleNamespace(option_name="EX_TEMPO", first_action=[0])
    outcome = mod._rollout_option(root_obs, {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                             "opponent_prize": [], "opponent_hand": [],
                                             "opponent_active": []},
                                  candidate, learner_index=0, opp_pm=None, deck_ids=[], option_cfg={})
    assert outcome["terminal_turns"] == 7  # 17 - 10


def test_rollout_option_releases_all_intermediate_nodes(mod, monkeypatch, populated_W):
    """#10: 経路上のsearch nodeが全てreleaseされる(rootを含む)。"""
    from cg.api import OptionType

    root_active = _Pokemon(OGERPON_EX, [GRASS] * 3)
    root_obs = _make_learner_turn_obs(root_active, [], _Pokemon(OGERPON_EX, []), [], turn=1,
                                      options=[_Option(OptionType.ATTACK)])
    mid_obs = _make_learner_turn_obs(root_active, [], _Pokemon(OGERPON_EX, []), [], turn=2,
                                     options=[_Option(OptionType.ATTACK)])
    end_obs = SimpleNamespace(
        current=_State([_Player(active=[]), _Player(active=[])], your_index=0, turn=3, result=0),
        select=None)

    fake = _ScriptedCgApi([root_obs, mid_obs, end_obs])
    monkeypatch.setattr("cg.api.search_begin", fake.search_begin)
    monkeypatch.setattr("cg.api.search_step", fake.search_step)
    monkeypatch.setattr("cg.api.search_release", fake.search_release)
    monkeypatch.setattr("cg.api.search_end", fake.search_end)
    monkeypatch.setattr(mod, "_learner_rollout_action", lambda *a, **k: [0])
    monkeypatch.setattr(mod, "_opponent_rollout_action", lambda *a, **k: [0])

    candidate = SimpleNamespace(option_name="EX_TEMPO", first_action=[0])
    mod._rollout_option(root_obs, {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                   "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
                        candidate, learner_index=0, opp_pm=None, deck_ids=[], option_cfg={})

    # search_beginで生成されたroot + 各search_stepの中間node + 最終node、すべてreleaseされている。
    assert set(fake.generated_ids) <= set(fake.released_ids) | {None}
    assert len(fake.released_ids) == len(set(fake.released_ids)), "二重解放が無いこと"
