"""ptcg_ai.ml_policy.ml_policy_agent のユニットテスト。

`action_selection`(担当Bのルーター)を一切呼ばずに選択肢インデックスを返せることを確認する
(step2-design.md §1 の接続点方針)。ただし ATTACK専用ハイブリッド(`_try_attack_hybrid`、
attack-rulebased-hybrid-implementation-plan.md)は例外として
`rule_based.main_turn_parts.proposals.collect_proposals` を読み取り専用で呼ぶため、
該当テストではこれをスタブに差し替えて検証する(実際の rule_based ロジックには依存しない)。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import sys
from pathlib import Path

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"

# 探索モジュールを差し替えたテスト用の最小 lethal_search 設定。
_LETHAL_ON = {"lethal_search": {"enabled": True, "module": "lethal_simple"}}

# ATTACK専用ハイブリッド用の最小config。lethal_searchは無効にして
# (mid_gameにATTACK型選択肢を足すだけの合成obsではlethal_simpleの前提が崩れるため)
# _try_attack_hybrid の挙動だけを切り出して確認する。
_ATTACK_HYBRID_ON = {"lethal_search": {"enabled": False}, "attack_hybrid": {"enabled": True}}


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


def _obs(encoder_observations: dict, key: str, select_override=None):
    from cg.api import to_observation_class

    obs_dict = {**encoder_observations[key], "logs": []}
    if select_override is not None:
        obs_dict["select"] = select_override
    return to_observation_class(obs_dict)


def _obs_with_attack_option(encoder_observations: dict):
    """mid_game(PLAY,PLAY,RETREAT,END の4択)にATTACK型選択肢を1件追加した合成obs。

    `_try_attack_hybrid` は select.option に ATTACK 型が無ければ
    `rb_proposals.collect_proposals` を呼ばずに None を返すため、その先の分岐
    (rule_base スタブの category による分岐)を検証するにはATTACK型選択肢が要る。
    """
    base = encoder_observations["mid_game"]
    select = dict(base["select"])
    select["option"] = [*select["option"], {"type": 13, "attackId": 0}]
    return _obs(encoder_observations, "mid_game", select_override=select)


def test_deck_selection_returns_60_card_ids(ml_policy_agent):
    """select が None(初回)なら deck.csv 由来の60枚のカードIDリストを返す。"""
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    deck = ml_policy_agent.agent(obs)
    assert len(deck) == 60
    assert all(isinstance(card_id, int) for card_id in deck)


def test_single_choice_returns_one_valid_index(ml_policy_agent, encoder_observations):
    """maxCount==1 の局面(mid_game / early_active_none)で範囲内の1要素リストを返す。"""
    for key in ("mid_game", "early_active_none"):
        obs = _obs(encoder_observations, key)
        result = ml_policy_agent.agent(obs)
        assert len(result) == 1
        assert 0 <= result[0] < len(obs.select.option)


def test_try_lethal_returns_search_result_when_lethal_found(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """探索が有効な行動を返したら _try_lethal はそれをそのまま採用する。"""
    obs = _obs(encoder_observations, "mid_game")

    class _StubSearch:
        @staticmethod
        def search(state, options, context):
            # context がリーサル探索の契約を満たしていることも併せて確認する。
            assert context["observation"] is obs
            assert callable(context["hidden_state_factory"])
            return [2]

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _StubSearch)
    assert ml_policy_agent._try_lethal(obs, config=_LETHAL_ON) == [2]
    # agent() 経由でもリーサルが優先される(model のスコアに上書きされない)。
    assert ml_policy_agent.agent(obs) == [2]


def test_try_lethal_returns_none_when_no_lethal(ml_policy_agent, encoder_observations, monkeypatch):
    """リーサルが無ければ None を返し、通常のスコアリングにフォールバックする。"""
    obs = _obs(encoder_observations, "mid_game")

    class _NoLethal:
        @staticmethod
        def search(state, options, context):
            return None

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _NoLethal)
    assert ml_policy_agent._try_lethal(obs, config=_LETHAL_ON) is None

    result = ml_policy_agent.agent(obs)
    assert len(result) == 1
    assert 0 <= result[0] < len(obs.select.option)


def test_try_lethal_disabled_by_config(ml_policy_agent, encoder_observations, monkeypatch):
    """lethal_search.enabled=false のときは探索を呼ばずに常に None を返す。"""
    obs = _obs(encoder_observations, "mid_game")

    class _MustNotBeCalled:
        @staticmethod
        def search(state, options, context):
            raise AssertionError("search must not be called when disabled")

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _MustNotBeCalled)
    config = {"lethal_search": {"enabled": False, "module": "lethal_simple"}}
    assert ml_policy_agent._try_lethal(obs, config=config) is None


def test_try_lethal_swallows_exceptions(ml_policy_agent, encoder_observations, monkeypatch):
    """探索が例外を投げてもクラッシュせず None を返し、合法手を返し続ける。"""
    obs = _obs(encoder_observations, "mid_game")

    class _Boom:
        @staticmethod
        def search(state, options, context):
            raise RuntimeError("boom")

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _Boom)
    assert ml_policy_agent._try_lethal(obs, config=_LETHAL_ON) is None

    result = ml_policy_agent.agent(obs)
    assert len(result) == 1
    assert 0 <= result[0] < len(obs.select.option)


def test_try_lethal_rejects_illegal_search_result(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """探索が contract 違反(範囲外インデックス)を返したら採用しない。"""
    obs = _obs(encoder_observations, "mid_game")

    class _Illegal:
        @staticmethod
        def search(state, options, context):
            return [len(options) + 5]

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _Illegal)
    assert ml_policy_agent._try_lethal(obs, config=_LETHAL_ON) is None


def test_explicit_config_is_independent_per_call(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """同一プロセス内で `agent(obs, config=A)` と `agent(obs, config=B)` が独立に別config
    を使うことの回帰確認(config衝突の再発防止。§1 の head-to-head 前提)。

    `_get_config()`(モジュールグローバルの `_config_cache`)を経由せず、呼び出しごとに
    渡した config の `lethal_search.module` がそのまま効くことを、2つの探索モジュールを
    別々の結果を返すスタブに差し替えて確認する。
    """
    obs = _obs(encoder_observations, "mid_game")

    class _SearchA:
        @staticmethod
        def search(state, options, context):
            return [0]

    class _SearchB:
        @staticmethod
        def search(state, options, context):
            return [1]

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _SearchA)
    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "pimc", _SearchB)

    config_a = {"lethal_search": {"enabled": True, "module": "lethal_simple"}}
    config_b = {"lethal_search": {"enabled": True, "module": "pimc"}}

    # グローバル config_cache を汚染しても結果が変わらないことを示すため、あえて先に
    # _get_config() を触ってキャッシュを確定させておく。
    _ = ml_policy_agent._get_config()

    # 同一プロセス・同一obsでも、渡した config ごとに別モジュールが選ばれる。
    assert ml_policy_agent.agent(obs, config=config_a) == [0]
    assert ml_policy_agent.agent(obs, config=config_b) == [1]
    # 交互に呼んでも混線しない(キャッシュではなく引数が支配する)。
    assert ml_policy_agent.agent(obs, config=config_a) == [0]
    assert ml_policy_agent.agent(obs, config=config_b) == [1]


def test_policy_weights_path_config_is_independent_per_call(
    ml_policy_agent, encoder_observations, monkeypatch, tmp_path
):
    """`config["policy_weights_path"]` を指定すると既定モデルではなくそのパスの
    `PolicyModel` を使い、パスごとにキャッシュされて他の呼び出しと混線しないことを確認する
    (policymodel-skill-concentration-implementation-plan.md Step4 の head-to-head 前提)。
    """
    obs = _obs(encoder_observations, "mid_game")

    calls = []

    class _StubModel:
        def __init__(self, weights_path=None):
            calls.append(weights_path)
            self.weights_path = weights_path

        def select_option(self, obs, hidden_state_factory=None, deadline=None):
            return 0

    monkeypatch.setattr(ml_policy_agent, "PolicyModel", _StubModel)
    monkeypatch.setattr(ml_policy_agent, "_model", None)
    monkeypatch.setattr(ml_policy_agent, "_model_cache_by_weights_path", {})

    default_config = {"lethal_search": {"enabled": False}}
    custom_config = {"lethal_search": {"enabled": False}, "policy_weights_path": str(tmp_path / "cand.json")}

    ml_policy_agent.agent(obs, config=default_config)
    ml_policy_agent.agent(obs, config=custom_config)
    ml_policy_agent.agent(obs, config=custom_config)  # 2回目はキャッシュ済みモデルを再利用するはず

    assert calls == [None, str(tmp_path / "cand.json")]


def test_multi_select_greedy_fallback_respects_min_max_count(ml_policy_agent, encoder_observations):
    """maxCount>1(Step2学習スコープ外)は minCount〜maxCount 件・重複無しで返す。"""
    base = encoder_observations["mid_game"]
    select = dict(base["select"])
    # mid_game は4択(PLAY,PLAY,RETREAT,END)。minCount=2,maxCount=3 の複数選択に仕立てる。
    select["minCount"] = 2
    select["maxCount"] = 3

    obs = _obs(encoder_observations, "mid_game", select_override=select)
    result = ml_policy_agent.agent(obs)

    assert 2 <= len(result) <= 3
    assert len(set(result)) == len(result)
    assert all(0 <= i < len(obs.select.option) for i in result)


# ---------------------------------------------------------------------------
# _try_pipeline (decision-pipeline/design-and-implementation-plan.md)
# ---------------------------------------------------------------------------


def test_try_pipeline_returns_none_when_disabled(ml_policy_agent, encoder_observations, monkeypatch):
    """config未指定/`pipeline`キー無しでは常にNone(pipeline.searchを呼ばない)。本番挙動不変の回帰。"""
    obs = _obs(encoder_observations, "mid_game")

    def _must_not_be_called(state, options, context):
        raise AssertionError("pipeline.search must not be called when disabled")

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _must_not_be_called)

    assert ml_policy_agent._try_pipeline(obs, config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_pipeline(obs, config={}) is None


def test_try_pipeline_invokes_search_when_enabled(ml_policy_agent, encoder_observations, monkeypatch):
    """`pipeline.enabled=true` のとき pipeline.search を呼び、契約(policy_model 等)を満たす。"""
    obs = _obs(encoder_observations, "mid_game")

    def _stub_search(state, options, context):
        assert context["observation"] is obs
        assert context["policy_model"] is not None
        assert callable(context["hidden_state_factory"])
        return [1]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _stub_search)
    config = {"lethal_search": {"enabled": False}, "pipeline": {"enabled": True, "hidden_state_source": "dummy"}}
    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    # agent() 経由でも lethal 無効時は pipeline の結果が採用される。
    assert ml_policy_agent.agent(obs, config=config) == [1]


def test_try_pipeline_swallows_exceptions_and_rejects_illegal(ml_policy_agent, encoder_observations, monkeypatch):
    """pipeline.search が例外/contract違反を返してもクラッシュせず None(top1 へフォールバック)。"""
    obs = _obs(encoder_observations, "mid_game")
    config = {"lethal_search": {"enabled": False}, "pipeline": {"enabled": True}}

    def _boom(state, options, context):
        raise RuntimeError("boom")

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _boom)
    assert ml_policy_agent._try_pipeline(obs, config=config) is None

    def _illegal(state, options, context):
        return [len(options) + 5]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _illegal)
    assert ml_policy_agent._try_pipeline(obs, config=config) is None
    # フォールバックで合法手を返し続ける。
    result = ml_policy_agent.agent(obs, config=config)
    assert len(result) == 1 and 0 <= result[0] < len(obs.select.option)


# ---------------------------------------------------------------------------
# _try_attack_hybrid (attack-rulebased-hybrid-implementation-plan.md Step1/Step2)
# ---------------------------------------------------------------------------


def _make_proposal(category: str, select: list[int], score: float):
    from ptcg_ai.rule_based.main_turn_parts.proposals import ActionProposal

    return ActionProposal(category=category, select=select, score=score, reason="test-stub")


def test_try_attack_hybrid_returns_none_when_disabled(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """config未指定/`attack_hybrid`キー無しでは常にNone(rb_proposalsを呼ばない)。"""
    obs = _obs_with_attack_option(encoder_observations)

    def _must_not_be_called(_obs):
        raise AssertionError("collect_proposals must not be called when disabled")

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _must_not_be_called)

    assert ml_policy_agent._try_attack_hybrid(obs, config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_attack_hybrid(obs, config={}) is None


def test_try_attack_hybrid_returns_none_when_no_attack_option(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ATTACK型選択肢が無ければ(mid_gameそのまま)rb_proposalsを呼ばずにNone。"""
    obs = _obs(encoder_observations, "mid_game")

    def _must_not_be_called(_obs):
        raise AssertionError("collect_proposals must not be called without an ATTACK option")

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _must_not_be_called)

    assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) is None


def test_try_attack_hybrid_returns_action_when_rule_base_picks_attack(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """draw/board/ability/energyの提案が無く、最高スコアが"attack"ならその選択を採用する。"""
    obs = _obs_with_attack_option(encoder_observations)
    attack_option_index = len(obs.select.option) - 1

    def _stub_collect_proposals(_obs):
        return [
            _make_proposal("attack", [attack_option_index], 100.0),
            _make_proposal("retreat", [2], 1.0),
            _make_proposal("end", [3], -1000.0),
        ]

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _stub_collect_proposals)

    assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) == [attack_option_index]
    # agent() 経由でも採用される(lethal無効・attack_hybrid有効の場合)。
    assert ml_policy_agent.agent(obs, config=_ATTACK_HYBRID_ON) == [attack_option_index]


def test_try_attack_hybrid_returns_none_when_rule_base_picks_other_category(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """rule_baseの最高スコアが"attack"以外ならNone(PolicyModelに委ねる)。"""
    obs = _obs_with_attack_option(encoder_observations)
    attack_option_index = len(obs.select.option) - 1

    def _stub_collect_proposals(_obs):
        return [
            _make_proposal("attack", [attack_option_index], 5.0),
            _make_proposal("retreat", [2], 100.0),
        ]

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _stub_collect_proposals)

    assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) is None


def test_try_attack_hybrid_returns_none_when_blocking_category_present(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """draw/board/ability/energyのいずれかが提案されていれば、attackが最高スコアでもNone。

    Step0のオフライン実測(results/2026-07-21_attack_hybrid_gate_offline.md)で、この絞り込み
    無しに一致率が23.2%まで劣化することを確認済み(進化等の途中で即攻撃してしまう欠陥)。
    """
    obs = _obs_with_attack_option(encoder_observations)
    attack_option_index = len(obs.select.option) - 1

    for blocking_category in ("draw", "board", "ability", "energy"):
        def _stub_collect_proposals(_obs, _cat=blocking_category):
            return [
                _make_proposal("attack", [attack_option_index], 1000.0),
                _make_proposal(_cat, [0], 20.0),
            ]

        monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _stub_collect_proposals)
        assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) is None, blocking_category


def test_try_attack_hybrid_swallows_exceptions(ml_policy_agent, encoder_observations, monkeypatch):
    """rb_proposals.collect_proposalsが例外を投げてもクラッシュせずNoneを返す。"""
    obs = _obs_with_attack_option(encoder_observations)

    def _boom(_obs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _boom)

    assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) is None


def test_try_attack_hybrid_rejects_illegal_action(ml_policy_agent, encoder_observations, monkeypatch):
    """rule_base提案のインデックスがcontract違反(範囲外)ならNone。"""
    obs = _obs_with_attack_option(encoder_observations)

    def _stub_collect_proposals(_obs):
        return [_make_proposal("attack", [len(obs.select.option) + 5], 100.0)]

    monkeypatch.setattr(ml_policy_agent.rb_proposals, "collect_proposals", _stub_collect_proposals)

    assert ml_policy_agent._try_attack_hybrid(obs, config=_ATTACK_HYBRID_ON) is None


# ===========================================================================
# _ogerpon_q_shadow: design.md Phase3 item4のshadow-only Q-critic評価
# ===========================================================================

def test_ogerpon_q_shadow_noop_when_log_disabled(ml_policy_agent, encoder_observations, monkeypatch):
    """既定(OGERPON_Q_SHADOW_LOG=None)では即returnし、_get_deck等には一切触れない。"""
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", None)

    def _boom():
        raise AssertionError("OGERPON_Q_SHADOW_LOG=Noneのときは_get_deckすら呼ばれてはいけない")
    monkeypatch.setattr(ml_policy_agent, "_get_deck", _boom)

    obs = _obs(encoder_observations, "mid_game")
    assert ml_policy_agent._ogerpon_q_shadow(obs, [0], {}) is None


def test_ogerpon_q_shadow_skips_non_ogerpon_deck(ml_policy_agent, encoder_observations, monkeypatch):
    """実際のdeck.csv(フーディン)はオーガポンデッキではないため、ログは空のまま。"""
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    obs = _obs(encoder_observations, "mid_game")
    ml_policy_agent._ogerpon_q_shadow(obs, [0], {})
    assert log == []


def test_ogerpon_q_shadow_logs_unsupported_trigger(ml_policy_agent, encoder_observations, monkeypatch):
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "promote")
    obs = _obs(encoder_observations, "mid_game")
    ml_policy_agent._ogerpon_q_shadow(obs, [0], {})
    assert len(log) == 1
    assert log[0]["outcome"] == "unsupported_trigger"
    assert log[0]["trigger_kind"] == "promote"


def test_ogerpon_q_shadow_logs_candidate_build_failed(ml_policy_agent, encoder_observations, monkeypatch):
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: None)
    obs = _obs(encoder_observations, "mid_game")
    ml_policy_agent._ogerpon_q_shadow(obs, [0], {})
    assert log[0]["outcome"] == "candidate_build_failed"


def test_ogerpon_q_shadow_logs_weights_not_ready(ml_policy_agent, encoder_observations, monkeypatch, tmp_path):
    from types import SimpleNamespace

    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_model", None)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None, target_card_id=None)
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=[1],
                               target_serial=1, target_card_id=920, required_ko_gain=0.0)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: (ex_c, single_c))
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel",
                        lambda *a, **k: SimpleNamespace(is_ready=False))
    obs = _obs(encoder_observations, "mid_game")
    ml_policy_agent._ogerpon_q_shadow(obs, [0], {})
    assert log[0]["outcome"] == "weights_not_ready"


def test_ogerpon_q_shadow_swallows_exceptions(ml_policy_agent, encoder_observations, monkeypatch):
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)

    def _raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger", _raise)
    obs = _obs(encoder_observations, "mid_game")
    assert ml_policy_agent._ogerpon_q_shadow(obs, [0], {}) is None
    assert log == []


def test_ogerpon_q_shadow_never_mutates_final_action(ml_policy_agent, encoder_observations, monkeypatch):
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates", lambda *a, **k: None)
    obs = _obs(encoder_observations, "mid_game")
    action = [2]
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})
    assert action == [2], "final_actionはshadow評価によって一切変更されてはいけない"


def test_agent_return_value_identical_with_shadow_logging_on_or_off(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """design.md Phase3完了条件の核心: shadow logging有効/無効でagent()の戻り値が
    完全に一致すること(154ゲーム回帰の単体テスト版)。実dekcはオーガポンではないので
    _ogerpon_q_shadowは早期returnするだけだが、呼び出し自体が経路に混入しても
    挙動が変わらないことを確認する。"""
    for key in ("mid_game", "early_active_none"):
        obs_off = _obs(encoder_observations, key)
        monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", None)
        result_off = ml_policy_agent.agent(obs_off)

        obs_on = _obs(encoder_observations, key)
        monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", [])
        result_on = ml_policy_agent.agent(obs_on)

        assert result_off == result_on


# ===========================================================================
# _ogerpon_q_shadow: 整合性安全網(final_action/RNG状態の変化検知、ユーザー指摘の
# 「同一意思決定内での厳密な非干渉判定」への対応)
# ===========================================================================

def test_ogerpon_q_shadow_detects_and_reverts_action_mutation(ml_policy_agent, encoder_observations, monkeypatch):
    """_ogerpon_q_shadow_implがfinal_actionを書き換えてしまった場合、ラッパーが検知し、
    元の値へ戻し、OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONSへ記録する(安全網自体の回帰テスト)。
    """
    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)

    def _broken_impl(obs, final_action, effective_config, q_config, shadow_only, is_lethal=False):
        final_action.append(999)  # 意図的にfinal_actionを壊す

    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_shadow_impl", _broken_impl)
    obs = _obs(encoder_observations, "mid_game")
    action = [2]
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})
    assert action == [2], "検知後は元の値へ戻すべき"
    assert len(violations) == 1
    assert violations[0]["kind"] == "final_action_mutated"


def test_ogerpon_q_shadow_detects_and_reverts_rng_state_change(ml_policy_agent, encoder_observations, monkeypatch):
    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)

    def _broken_impl(obs, final_action, effective_config, q_config, shadow_only, is_lethal=False):
        import random as _random
        _random.random()  # 意図的にRNG状態を消費する

    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_shadow_impl", _broken_impl)
    obs = _obs(encoder_observations, "mid_game")
    import random
    state_before = random.getstate()
    ml_policy_agent._ogerpon_q_shadow(obs, [0], {})
    assert random.getstate() == state_before, "検知後は元のRNG状態へ戻すべき"
    assert len(violations) == 1
    assert violations[0]["kind"] == "rng_state_changed"


def test_ogerpon_q_shadow_no_integrity_violations_on_real_evaluated_path(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """実際に(モックした)候補構築・モデル評価まで到達するevaluated経路でも、
    final_action/RNG状態のいずれも変化しないこと。"""
    from types import SimpleNamespace

    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None, target_card_id=None)
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=[1],
                               target_serial=1, target_card_id=920, required_ko_gain=1.0)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: (ex_c, single_c))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy_encoder, "encode_strategy_pair",
                        lambda *a, **k: {"continuous_features": [], "slot_card_ids": [],
                                        "option_features": {"EX_TEMPO": [], "SINGLE_PRIZE_ROTATION": []}})
    # shadowモデルがSINGLEを強く推す(高いdelta)結果を返しても、baseline_actionは不変。
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel",
                        lambda *a, **k: SimpleNamespace(is_ready=True))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "compare_options",
                        lambda *a, **k: {"p_ex": 0.2, "p_single": 0.9, "delta": 0.7, "delta_std": 0.0,
                                         "lcb_delta": 0.7, "p_loop_complete": 0.95,
                                         "required_ko_gain": 1.0, "chosen_option": "SINGLE_PRIZE_ROTATION",
                                         "reason": "strict_override", "n_ensemble_members": 3,
                                         "thresholds": ml_policy_agent.ogerpon_strategy.DEFAULT_DECISION_THRESHOLDS})

    obs = _obs(encoder_observations, "mid_game")
    action = [7]  # baselineが実際に返した行動(ATTACHではない、任意の値)
    action_before = list(action)
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})

    assert log[0]["outcome"] == "evaluated"
    assert log[0]["would_override"] is True, "shadowはSINGLEを推すログを残す"
    assert action == action_before, "shadowがSINGLEを推しても、最終行動はbaselineのまま変わらない"
    assert violations == []


# ===========================================================================
# _ogerpon_q_shadow_impl: ソース検査による静的保証(探索エンジンの再実行・Option State
# の実開始・match_contextへの関与が一切無いこと)
# ===========================================================================

def test_ogerpon_q_shadow_impl_never_touches_search_engine_or_match_context(ml_policy_agent):
    """探索エンジン・match_contextには一切触れないこと(静的検査で常に保証できる不変条件)。
    永続Option Controllerの開始(``ogerpon_option_state_mod.advance``)はPhase4の能動ゲート
    でのみ意図的に呼ばれるため、ここでは対象外(別テストで実行時に検証する)。
    """
    import inspect

    src = inspect.getsource(ml_policy_agent._ogerpon_q_shadow_impl)
    forbidden = ["search_step", "search_begin", "match_context"]
    for token in forbidden:
        assert token not in src, f"_ogerpon_q_shadow_implは{token!r}を使ってはいけない"


def test_ogerpon_q_shadow_never_advances_option_state_when_shadow_only(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """shadow_only=True(既定)では、strict override条件を満たしていても永続Option State
    (``ogerpon_option_state_mod.advance``)を一切呼ばない(design.md Phase3 item4の
    「shadow_onlyでは状態を開始・更新しない」要件)。"""
    OS = ml_policy_agent.ogerpon_option_state_mod
    calls = []
    monkeypatch.setattr(OS, "advance", lambda *a, **k: calls.append(1) or OS.OgerponOptionState())
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    # _active_gate_setup自体がadvanceを差し替えるので、監視用に再度上書きする。
    monkeypatch.setattr(OS, "advance", lambda *a, **k: calls.append(1) or OS.OgerponOptionState())
    config = {"ogerpon_q_critic": {"active_triggers": ["main_attach"]}}  # shadow_only省略=True
    obs = _obs(encoder_observations, "mid_game")
    ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert calls == [], "shadow_onlyではOption Stateのadvance()を呼んではいけない"


def test_ogerpon_q_shadow_action_unchanged_when_weights_not_ready(ml_policy_agent, encoder_observations, monkeypatch):
    from types import SimpleNamespace

    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)
    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_model", None)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None, target_card_id=None)
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=[1],
                               target_serial=1, target_card_id=920, required_ko_gain=0.0)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: (ex_c, single_c))
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel",
                        lambda *a, **k: SimpleNamespace(is_ready=False))
    obs = _obs(encoder_observations, "mid_game")
    action = [4]
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})
    assert action == [4]
    assert violations == []


def test_ogerpon_q_shadow_action_unchanged_on_exception_mid_evaluation(
    ml_policy_agent, encoder_observations, monkeypatch
):
    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")

    exceptions = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_EXCEPTIONS", exceptions)

    def _raise(*a, **k):
        raise RuntimeError("mid-evaluation failure")
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates", _raise)
    obs = _obs(encoder_observations, "mid_game")
    action = [5]
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})
    assert action == [5]
    assert violations == []
    assert log == []
    assert len(exceptions) == 1
    assert "mid-evaluation failure" in exceptions[0]


def test_ogerpon_q_shadow_action_unchanged_when_shadow_recommends_ex_tempo(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """shadowが(既定通り)EX_TEMPOを推す場合も、最終行動はbaselineのまま(自明だが明示確認)。"""
    from types import SimpleNamespace

    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: "main_attach")
    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None, target_card_id=None)
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=[1],
                               target_serial=1, target_card_id=920, required_ko_gain=0.0)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: (ex_c, single_c))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy_encoder, "encode_strategy_pair",
                        lambda *a, **k: {"continuous_features": [], "slot_card_ids": [],
                                        "option_features": {"EX_TEMPO": [], "SINGLE_PRIZE_ROTATION": []}})
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel",
                        lambda *a, **k: SimpleNamespace(is_ready=True))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "compare_options",
                        lambda *a, **k: {"p_ex": 0.6, "p_single": 0.55, "delta": -0.05, "delta_std": 0.01,
                                         "lcb_delta": -0.06, "p_loop_complete": 0.3,
                                         "required_ko_gain": 0.0, "chosen_option": "EX_TEMPO",
                                         "reason": "default", "n_ensemble_members": 3,
                                         "thresholds": ml_policy_agent.ogerpon_strategy.DEFAULT_DECISION_THRESHOLDS})
    obs = _obs(encoder_observations, "mid_game")
    action = [3]
    action_before = list(action)
    ml_policy_agent._ogerpon_q_shadow(obs, action, {})
    assert log[0]["would_override"] is False
    assert action == action_before
    assert violations == []


# ===========================================================================
# Phase4 能動ゲート(shadow_only=False): design.md §11.2の厳密しきい値を満たし、
# かつtriggerがactive_triggersに含まれ、かつ旧Plannerが非shadowで同時稼働していない
# 場合だけ、SINGLE_PRIZE_ROTATION側の行動へ実際に置き換える。
# ===========================================================================

def _active_gate_setup(monkeypatch, ml_policy_agent, chosen_option="SINGLE_PRIZE_ROTATION",
                       trigger="main_attach", single_first_action=None):
    from types import SimpleNamespace

    log = []
    violations = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_INTEGRITY_VIOLATIONS", violations)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger",
                        lambda obs, me, deck_ids, mode: trigger)
    ex_c = SimpleNamespace(option_name="EX_TEMPO", first_action=[0], target_serial=None, target_card_id=None)
    single_action = single_first_action if single_first_action is not None else [1]
    single_c = SimpleNamespace(option_name="SINGLE_PRIZE_ROTATION", first_action=single_action,
                               target_serial=1, target_card_id=920, required_ko_gain=1.0)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates",
                        lambda *a, **k: (ex_c, single_c))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy_encoder, "encode_strategy_pair",
                        lambda *a, **k: {"continuous_features": [], "slot_card_ids": [],
                                        "option_features": {"EX_TEMPO": [], "SINGLE_PRIZE_ROTATION": []}})
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel",
                        lambda *a, **k: SimpleNamespace(is_ready=True))
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "compare_options",
                        lambda *a, **k: {"p_ex": 0.2, "p_single": 0.9, "delta": 0.7, "delta_std": 0.0,
                                         "lcb_delta": 0.7, "p_loop_complete": 0.95,
                                         "required_ko_gain": 1.0, "chosen_option": chosen_option,
                                         "reason": "strict_override" if chosen_option == "SINGLE_PRIZE_ROTATION"
                                         else "default",
                                         "n_ensemble_members": 3,
                                         "thresholds": ml_policy_agent.ogerpon_strategy.DEFAULT_DECISION_THRESHOLDS})
    monkeypatch.setattr(ml_policy_agent, "_legacy_ogerpon_planner_is_live", lambda cfg, deck_ids: False)
    # 永続Option Controllerを実際にBUILDへ開始できたことにする(テスト用obsには実際の
    # カプ・ブルルが存在しないため、本物の advance_state() は IDLE のまま返してしまう)。
    OS = ml_policy_agent.ogerpon_option_state_mod
    _set_advance_mock(OS, monkeypatch, OS.BUILD, target_serial=1, target_card_id=920)
    return log, violations


def _set_advance_mock(OS, monkeypatch, mode, **kwargs):
    """``ogerpon_option_state_mod.advance`` を差し替える。本物の``advance``と同じく
    モジュールグローバル ``_STATE`` を更新する副作用を持たせる(戻り値だけ差し替えると
    次の呼び出しで永続状態が反映されない)。
    """
    state = OS.OgerponOptionState(mode=mode, phase=mode, **kwargs)

    def _advance(obs, me, cfg, force_start=False):
        OS._STATE = state
        return state

    monkeypatch.setattr(OS, "advance", _advance)


def test_phase4_active_override_applies_when_conditions_met(ml_policy_agent, encoder_observations, monkeypatch):
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")  # 4択(index 0-3)。single_first_action=[1]は合法。
    action = [0]
    override = ml_policy_agent._ogerpon_q_shadow(obs, action, config)
    assert override == [1]
    assert action == [0], "オーバーライドはfinal_actionを書き換えず、別の戻り値で伝える"


def test_phase4_no_override_when_shadow_only_default(ml_policy_agent, encoder_observations, monkeypatch):
    """shadow_only未指定(既定True)なら、active_triggersを設定していても絶対に置き換わらない。"""
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    config = {"ogerpon_q_critic": {"active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_no_override_when_trigger_not_in_active_triggers(ml_policy_agent, encoder_observations, monkeypatch):
    _active_gate_setup(monkeypatch, ml_policy_agent, trigger="retreat", single_first_action=[1])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_no_override_when_active_triggers_empty_by_default(ml_policy_agent, encoder_observations, monkeypatch):
    """active_triggers省略時は既定で空集合(=どのtriggerも能動化しない、安全側デフォルト)。"""
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    config = {"ogerpon_q_critic": {"shadow_only": False}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_no_override_when_chosen_option_is_ex_tempo(ml_policy_agent, encoder_observations, monkeypatch):
    _active_gate_setup(monkeypatch, ml_policy_agent, chosen_option="EX_TEMPO", single_first_action=[1])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_no_override_when_legacy_planner_live(ml_policy_agent, encoder_observations, monkeypatch):
    """旧Plannerが非shadowで同時稼働している場合、条件を満たしていても能動介入しない
    (design.md Phase3 item4の排他要件)。"""
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    monkeypatch.setattr(ml_policy_agent, "_legacy_ogerpon_planner_is_live", lambda cfg, deck_ids: True)
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_no_override_when_action_illegal_for_current_select(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """single_first_actionが現在のselectに対して不正(範囲外等)なら、安全側でNoneに倒す。"""
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[999])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert override is None


def test_phase4_agent_returns_override_action_end_to_end(ml_policy_agent, encoder_observations, monkeypatch):
    """agent()自体がoverrideを実際に採用して返すことのend-to-end確認。"""
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[2])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    result = ml_policy_agent.agent(obs, config)
    assert result == [2]


def test_legacy_ogerpon_planner_is_live_detects_non_shadow_main_bonus(ml_policy_agent):
    config = {"ogerpon_planner": {"enabled": True, "main_enabled": True, "main_shadow_only": False}}
    deck_ids = [96, 96, 96, 96, 920] + [7] * 55
    assert ml_policy_agent._legacy_ogerpon_planner_is_live(config, deck_ids) is True


def test_legacy_ogerpon_planner_is_live_false_when_shadow_only(ml_policy_agent):
    config = {"ogerpon_planner": {"enabled": True, "main_enabled": True, "main_shadow_only": True}}
    deck_ids = [96, 96, 96, 96, 920] + [7] * 55
    assert ml_policy_agent._legacy_ogerpon_planner_is_live(config, deck_ids) is False


def test_legacy_ogerpon_planner_is_live_false_when_disabled(ml_policy_agent):
    deck_ids = [96, 96, 96, 96, 920] + [7] * 55
    assert ml_policy_agent._legacy_ogerpon_planner_is_live({}, deck_ids) is False


def test_phase4_active_override_never_replaces_lethal_action(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """回帰テスト: Stage7の安全レビューで発見した実バグ。確定リーサル(`_try_lethal`)が
    見つかった回に、Q-criticの能動ゲート条件がたまたま同時に成立しても、lethalの行動を
    上書きしてはいけない(lethal最優先はこのプロジェクト全体の絶対要件)。
    """
    obs = _obs(encoder_observations, "mid_game")
    lethal_action = [2]

    class _StubSearch:
        @staticmethod
        def search(state, options, context):
            return lethal_action

    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, "lethal_simple", _StubSearch)
    # Q-criticが強くSINGLE(=lethalとは異なる行動[1])を推す状況を作る。
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    config = {
        "lethal_search": {"enabled": True, "module": "lethal_simple"},
        "ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]},
    }
    result = ml_policy_agent.agent(obs, config)
    assert result == lethal_action, "lethalが見つかった回はQ-criticの能動ゲートで上書きしてはいけない"


def test_ogerpon_q_shadow_is_lethal_suppresses_override_and_records_it(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """`_ogerpon_q_shadow`単体でも、`is_lethal=True`を渡すとoverride条件を満たしていても
    `active_override_applied`がFalseになり、ログにも`is_lethal_baseline=True`として
    正しく記録されること(一次防御そのものの単体確認。agent()レベルの二次防御とは別)。
    """
    log, violations = _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    obs = _obs(encoder_observations, "mid_game")
    override = ml_policy_agent._ogerpon_q_shadow(obs, [0], config, is_lethal=True)
    assert override is None
    assert log[0]["is_lethal_baseline"] is True
    assert log[0]["active_override_applied"] is False
    assert violations == []


# ===========================================================================
# 永続Option Controller(design.md Phase3 item4、ユーザー指摘対応)の統合テスト。
#
# 学習時のSINGLE_PRIZE_ROTATION rolloutはBUILD→READY→ACTIVE→COMPLETE/ABORTの固定
# Controllerが最後まで低位行動を拘束することを前提にしている。実戦で最初の1手だけ
# 差し替えて次ターンから通常Policyへ戻ってしまうと、学習した
# ``Q(s, SINGLE_PRIZE_ROTATION)`` の値と実際に実行される戦略が食い違う。
# この一連のテストは、``agent()`` が複数ターンにわたって永続状態
# (``ogerpon_option_state_mod`` のモジュールグローバル)を保持し、低位Controllerが
# 一貫して行動を拘束し続けることを確認する。
# ===========================================================================

def test_persistent_option_controller_full_rotation_lifecycle(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """IDLE → strict override → BUILD → READY → ACTIVE → ブルル攻撃 →
    COMPLETE → 通常Policyへ復帰、を複数ターンのagent()呼び出しで追う。"""
    OS = ml_policy_agent.ogerpon_option_state_mod
    STRAT = ml_policy_agent.ogerpon_strategy
    OS.reset()
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: ([0], False))
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}

    # --- ターン1: IDLE。main_attachでstrict overrideが発火し、BUILDへ開始する。 ---
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])
    assert OS.get_state().mode == OS.IDLE
    result1 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result1 == [1], "strict override条件を満たしたので、SINGLE側の最初の一手を返す"
    assert OS.get_state().mode == OS.BUILD
    assert OS.get_state().target_card_id == 920

    # --- ターン2: BUILD継続。低位Controllerが行動を拘束し、IDLEへは戻らない。 ---
    _set_advance_mock(OS, monkeypatch, OS.BUILD, target_serial=1, target_card_id=920)
    monkeypatch.setattr(STRAT, "single_prize_low_level_action", lambda obs, cfg, deck_ids, mode: [3])
    result2 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result2 == [3], "BUILD中は低位Controllerの手貼り行動を拘束する"
    assert OS.get_state().mode == OS.BUILD, "1手だけで元のIDLEへ戻ってはいけない"

    # --- ターン3: READYへ遷移。 ---
    _set_advance_mock(OS, monkeypatch, OS.READY, target_serial=1, target_card_id=920)
    monkeypatch.setattr(STRAT, "single_prize_low_level_action", lambda obs, cfg, deck_ids, mode: [2])
    result3 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result3 == [2]
    assert OS.get_state().mode == OS.READY

    # --- ターン4: ACTIVEへ遷移。ブルルが攻撃する。 ---
    _set_advance_mock(OS, monkeypatch, OS.ACTIVE, target_serial=1, target_card_id=920, bulu_attack_count=1)
    monkeypatch.setattr(STRAT, "single_prize_low_level_action", lambda obs, cfg, deck_ids, mode: [1])
    result4 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result4 == [1], "ACTIVE中は原則ATTACKを拘束する"
    assert OS.get_state().mode == OS.ACTIVE
    assert OS.get_state().bulu_attack_count == 1

    # --- ターン5: 気絶してCOMPLETEへ。通常Policy(final_action)へ復帰する。 ---
    _set_advance_mock(OS, monkeypatch, OS.COMPLETE, target_serial=1, target_card_id=920, bulu_attack_count=1)
    result5 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result5 == [0], "COMPLETE後は低位Controllerが介入せず、通常Policyの行動を返す"
    assert OS.get_state().mode == OS.COMPLETE

    # --- ターン6: COMPLETE後、再評価されても新しい中継として再開しない(介入0)。 ---
    result6 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result6 == [0]
    assert OS.get_state().mode == OS.COMPLETE, "COMPLETEから勝手にBUILDへ再開してはいけない"

    OS.reset()


def test_persistent_option_controller_abort_returns_to_normal_policy(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """ABORT(対象消失等)後も、COMPLETEと同様に通常Policyへ戻り、再介入しない。"""
    OS = ml_policy_agent.ogerpon_option_state_mod
    OS.reset()
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: ([9], False))
    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    _active_gate_setup(monkeypatch, ml_policy_agent, single_first_action=[1])

    result1 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result1 == [1]
    assert OS.get_state().mode == OS.BUILD

    _set_advance_mock(OS, monkeypatch, OS.ABORT, target_serial=1, target_card_id=920,
                     abort_reason="target_lost_or_replaced")
    result2 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result2 == [9], "ABORT後は通常Policyへ戻る"
    assert OS.get_state().mode == OS.ABORT

    result3 = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result3 == [9]
    assert OS.get_state().mode == OS.ABORT, "ABORTから勝手に再開してはいけない"
    OS.reset()


def test_persistent_option_controller_lethal_interrupts_mid_rotation(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """中継進行中(ACTIVE)でも、lethalが見つかれば常にlethalが優先される。"""
    OS = ml_policy_agent.ogerpon_option_state_mod
    STRAT = ml_policy_agent.ogerpon_strategy
    OS.reset()
    OS._STATE = OS.OgerponOptionState(mode=OS.ACTIVE, phase=OS.ACTIVE,
                                      target_serial=1, target_card_id=920, bulu_attack_count=1)
    lethal_action = [7]
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: (lethal_action, True))

    def _must_not_be_called(*a, **k):
        raise AssertionError("lethal中はOption Controllerの継続判定を呼んではいけない")
    monkeypatch.setattr(STRAT, "single_prize_low_level_action", _must_not_be_called)
    monkeypatch.setattr(OS, "advance", _must_not_be_called)

    config = {"ogerpon_q_critic": {"shadow_only": False, "active_triggers": ["main_attach"]}}
    result = ml_policy_agent.agent(_obs(encoder_observations, "mid_game"), config)
    assert result == lethal_action
    assert OS.get_state().mode == OS.ACTIVE, "lethalの回はstateを変更しない(次ターンで再評価される)"
    OS.reset()


def test_persistent_option_controller_resets_on_new_match(ml_policy_agent):
    """新しい試合(obs.select is None)では、shadow_onlyに関わらず必ず状態がresetされる。"""
    from cg.api import to_observation_class

    OS = ml_policy_agent.ogerpon_option_state_mod
    OS._STATE = OS.OgerponOptionState(mode=OS.ACTIVE, phase=OS.ACTIVE,
                                      target_serial=1, target_card_id=920)
    obs = to_observation_class({"current": None, "logs": [], "select": None})
    ml_policy_agent.agent(obs)
    assert OS.get_state().mode == OS.IDLE, "新しい試合開始で必ずIDLEへ戻ること"


def test_ogerpon_option_continue_returns_none_when_shadow_only_even_if_state_leaked(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """万一(バグ等で)永続状態がBUILD等に残っていても、shadow_only=Trueの回は
    低位Controllerが一切介入しない(shadow_onlyでは状態を読み取って行動へ反映しない)。"""
    OS = ml_policy_agent.ogerpon_option_state_mod
    original_state = OS.get_state()
    try:
        OS._STATE = OS.OgerponOptionState(mode=OS.ACTIVE, phase=OS.ACTIVE,
                                          target_serial=1, target_card_id=920)
        obs = _obs(encoder_observations, "mid_game")
        result = ml_policy_agent._ogerpon_option_continue(obs, {"ogerpon_q_critic": {}})
        assert result is None
    finally:
        OS._STATE = original_state


# ===========================================================================
# Bulu Q-critic 開発終了(2026-08-15、docs/experiments/ogerpon_bulu_qcritic_final_report.md)
# 本番config(configs/abl_5_full.json, ogerpon_q_critic.enabled=false)が
# 推論・OptionState・行動のいずれにも一切触れず完全無効化されていることを検証する。
# ここでは合成configではなく実ファイル(load_config("abl_5_full"))を読み込んで検証する。
# ===========================================================================

def _load_real_production_config(ml_policy_agent):
    return ml_policy_agent.load_config("abl_5_full")


def test_production_config_has_ogerpon_q_critic_disabled(ml_policy_agent):
    """本番config実ファイルがenabled=false, shadow_only=trueであることの直接確認。"""
    config = _load_real_production_config(ml_policy_agent)
    q_config = config.get("ogerpon_q_critic")
    assert q_config is not None
    assert q_config.get("enabled") is False
    assert q_config.get("shadow_only") is True


def test_production_config_ogerpon_q_shadow_never_loads_model_or_detects_trigger(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """本番config実ファイルでは _ogerpon_q_shadow が deck判定・trigger検出・モデル生成の
    いずれにも到達しない(推論回数0)。"""
    config = _load_real_production_config(ml_policy_agent)
    log = []
    monkeypatch.setattr(ml_policy_agent, "OGERPON_Q_SHADOW_LOG", log)

    def _boom(*a, **k):
        raise AssertionError("enabled=falseなら deck_is_ogerpon すら呼ばれてはいけない")
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", _boom)
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "detect_trigger", _boom)
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel", _boom)

    obs = _obs(encoder_observations, "mid_game")
    result = ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
    assert result is None
    assert log == [], "ログ記録も一切行われない"


def test_production_config_ogerpon_option_continue_never_starts_option_state(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """本番config実ファイルでは _ogerpon_option_continue が OptionState.advance に到達
    しない(Option State開始0)。永続状態がBUILD等に残っていても読み取りすらしない。"""
    config = _load_real_production_config(ml_policy_agent)
    OS = ml_policy_agent.ogerpon_option_state_mod
    original_state = OS.get_state()

    def _boom(*a, **k):
        raise AssertionError("enabled=falseなら advance は一切呼ばれてはいけない")
    monkeypatch.setattr(OS, "advance", _boom)
    try:
        OS._STATE = OS.OgerponOptionState(mode=OS.BUILD, phase=OS.BUILD,
                                          target_serial=1, target_card_id=920)
        obs = _obs(encoder_observations, "mid_game")
        result = ml_policy_agent._ogerpon_option_continue(obs, config)
        assert result is None
        assert OS.get_state().mode == OS.BUILD, "状態自体には触れず、単に読み取らないだけ"
    finally:
        OS._STATE = original_state


def test_production_config_agent_output_matches_baseline_even_if_q_critic_would_override(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """本番config実ファイルでagent()を通しても、Q-criticが(もし動いていれば)介入したはず
    の状況を人工的に作っても、最終行動はbaseline Policyの結果と完全に一致する(行動差0)。"""
    config = _load_real_production_config(ml_policy_agent)
    baseline_action = [2]
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: (baseline_action, False))

    def _boom(*a, **k):
        raise AssertionError("enabled=falseなら候補構築(build_candidates)にも到達しない")
    monkeypatch.setattr(ml_policy_agent.ogerpon_strategy, "build_candidates", _boom)
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: True)

    obs = _obs(encoder_observations, "mid_game")
    result = ml_policy_agent.agent(obs, config)
    assert result == baseline_action


def test_production_config_non_ogerpon_deck_unaffected(ml_policy_agent, encoder_observations, monkeypatch):
    """非オーガポンデッキ(実際のsample_submission/deck.csv、フーディン)でも本番configでは
    Q-critic関連コードに一切触れない(deck判定コードすら呼ばれない=影響0)。"""
    config = _load_real_production_config(ml_policy_agent)

    def _boom(*a, **k):
        raise AssertionError("非オーガポンデッキか否かの判定自体、enabled=falseなら不要")
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", _boom)
    obs = _obs(encoder_observations, "mid_game")
    assert ml_policy_agent._ogerpon_q_shadow(obs, [0], config) is None


def test_production_config_lethal_path_unaffected_by_q_critic(ml_policy_agent, encoder_observations, monkeypatch):
    """本番config実ファイルで、lethal行動が見つかった回はQ-critic関連コードへ一切分岐しない
    (lethal経路への影響0)。agent()のトップレベルから検証する。"""
    config = _load_real_production_config(ml_policy_agent)
    lethal_action = [3]
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: (lethal_action, True))

    def _boom(*a, **k):
        raise AssertionError("lethalの回はQ-critic関連コードへ到達してはいけない")
    monkeypatch.setattr(ml_policy_agent, "_ogerpon_option_continue", _boom)
    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_shadow", _boom)

    obs = _obs(encoder_observations, "mid_game")
    result = ml_policy_agent.agent(obs, config)
    assert result == lethal_action


def test_production_config_works_without_weights_file(ml_policy_agent, encoder_observations, monkeypatch):
    """weights(ogerpon_strategy_weights.json)が存在しない/壊れていても、本番configでは
    そもそも _get_ogerpon_q_model が一度も呼ばれないため既存Policyは問題なく動作する。"""
    config = _load_real_production_config(ml_policy_agent)
    monkeypatch.setattr(ml_policy_agent, "_ogerpon_q_model", None)

    def _boom(*a, **k):
        raise AssertionError("enabled=falseならモデルを一切ロードしてはいけない(weights未配置でも動作する保証)")
    monkeypatch.setattr(ml_policy_agent, "_get_ogerpon_q_model", _boom)
    monkeypatch.setattr(ml_policy_agent, "OgerponStrategyModel", _boom)

    obs = _obs(encoder_observations, "mid_game")
    assert ml_policy_agent._ogerpon_q_shadow(obs, [0], config) is None
    assert ml_policy_agent._ogerpon_q_model is None, "遅延ロードされる _ogerpon_q_model が一切生成されない"


def test_research_shadow_config_agent_output_matches_baseline(ml_policy_agent, encoder_observations, monkeypatch):
    """configs/ogerpon_q_shadow_research.json(enabled=true, shadow_only=true)は推論・ログ
    記録こそ行うが、shadow_only=trueである限りagent()の最終行動はbaselineと完全に一致する
    (研究用configでも行動差0)。"""
    config = ml_policy_agent.load_config("ogerpon_q_shadow_research")
    assert config["ogerpon_q_critic"]["enabled"] is True
    assert config["ogerpon_q_critic"]["shadow_only"] is True

    baseline_action = [2]
    monkeypatch.setattr(ml_policy_agent, "_select_action", lambda obs, cfg: (baseline_action, False))
    # 実オーガポンデッキではないため detect_trigger 到達前に deck_is_ogerpon で弾かれるが、
    # 念のため明示的にFalseを返させ、推論経路には入っても行動には反映されないことを
    # _ogerpon_q_shadow の戻り値(None)と agent() の最終出力の両方で確認する。
    monkeypatch.setattr(ml_policy_agent.ogerpon_planner, "deck_is_ogerpon", lambda deck_ids: False)

    obs = _obs(encoder_observations, "mid_game")
    result = ml_policy_agent.agent(obs, config)
    assert result == baseline_action


def test_production_config_overhead_is_negligible(ml_policy_agent, encoder_observations):
    """本番config実ファイルでの _ogerpon_q_shadow / _ogerpon_option_continue の
    オーバーヘッドは辞書lookup数回のみ(モデル推論・エンコード・OptionState更新は一切
    発生しない)ことを、実測時間の粗いスモークガードで確認する
    (Q-critic導入前とのバイナリ比較はできないため、定数時間であることの代替確認)。"""
    import time

    config = _load_real_production_config(ml_policy_agent)
    obs = _obs(encoder_observations, "mid_game")
    t0 = time.perf_counter()
    for _ in range(2000):
        ml_policy_agent._ogerpon_q_shadow(obs, [0], config)
        ml_policy_agent._ogerpon_option_continue(obs, config)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"2000回の呼び出しで{elapsed:.3f}s(モデルロード等が紛れ込んでいる可能性)"
