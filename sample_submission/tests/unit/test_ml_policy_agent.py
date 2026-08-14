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
    # 既定 config が選ぶモジュール名は提出構成によって変わる(現在は lethal_phase1)ので、
    # ハードコードせず config から引いて同じスタブを差し込む。
    default_module = ml_policy_agent._get_config()["lethal_search"].get("module", "lethal_simple")
    monkeypatch.setitem(ml_policy_agent._SEARCH_MODULES, default_module, _StubSearch)
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
