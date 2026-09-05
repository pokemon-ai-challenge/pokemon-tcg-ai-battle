"""ptcg_ai.ml_policy.ml_policy_agent のユニットテスト。

`action_selection`(担当Bのルーター)を一切呼ばずに選択肢インデックスを返せることを確認する
(step2-design.md §1 の接続点方針)。ただし ATTACK専用ハイブリッド(`_try_attack_hybrid`、
attack-rulebased-hybrid-implementation-plan.md)は例外として
`rule_based.main_turn_parts.proposals.collect_proposals` を読み取り専用で呼ぶため、
該当テストではこれをスタブに差し替えて検証する(実際の rule_based ロジックには依存しない)。

実 cg エンジン(カードデータ)を使う。DLL がロードできない環境ではスキップされる。
"""

import json
import random
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


# ---------------------------------------------------------------------------
# 専用RNG (config-gated `lethal_search.isolated_rng`、乱数消費の交絡除去)
#
# 問題: リーサル探索の隠れ状態スタブ(`build_dummy_search_state`)は既定でグローバル
# `random` モジュールを消費する。探索の実行量(発火有無・verify_shuffles 回数)が変わる
# だけで、探索と無関係な後続の乱数列(pipeline の決定化サンプリング等)までずれてしまい、
# paired seed A/B のペア性が壊れる(交絡)。`isolated_rng=true` は `_try_lethal` が
# `lethal_simple` 専用の `random.Random` インスタンスをサンプリングに渡すことで、この
# 消費をグローバル `random` から完全に切り離す。
# ---------------------------------------------------------------------------

def _shuffling_build_stub(obs_arg, full_deck, rng=None):
    """`build_dummy_search_state` の代わりに使う、実際に乱数を1回消費するだけのスタブ。

    `rng` が None ならグローバル `random` を、そうでなければ渡された専用インスタンスを
    消費する(本物の `build_dummy_search_state` と同じ委譲則)。中身(隠れ状態)自体は
    このテスト群の関心事ではないので常に None を返す。
    """
    pool = [1, 2, 3, 4, 5]
    (rng if rng is not None else random).shuffle(pool)
    return None


def _search_stub_calling_factory_n_times(sample_calls: int, result):
    """`hidden_state_factory` を `sample_calls` 回呼ぶだけの `_SEARCH_MODULES` 差し替え。

    「初回サンプル + verify_shuffles 回の再サンプリング」を模す。`result` を返すことで
    「リーサルが見つかる/見つからない」の両方をテストできる。
    """
    class _Stub:
        @staticmethod
        def search(state, options, context):
            factory = context["hidden_state_factory"]
            for _ in range(sample_calls):
                factory()
            return result
    return _Stub


def test_try_lethal_isolated_rng_passes_dedicated_rng_to_factory(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """isolated_rng=true のとき、hidden_state_factory は build_dummy_search_state に
    `lethal_simple.get_rng()` の専用インスタンスを渡す。キー無し(既定)は rng=None の
    ままで、build_dummy_search_state 側がグローバル random にフォールバックする
    (既存挙動とバイト単位で同一)。"""
    obs = _obs(encoder_observations, "mid_game")
    captured_rng = []

    def _capture(obs_arg, full_deck, rng=None):
        captured_rng.append(rng)
        return None

    monkeypatch.setattr(ml_policy_agent, "build_dummy_search_state", _capture)
    monkeypatch.setitem(
        ml_policy_agent._SEARCH_MODULES, "lethal_simple",
        _search_stub_calling_factory_n_times(1, None),
    )

    ml_policy_agent.lethal_simple.reset_rng(1)
    dedicated = ml_policy_agent.lethal_simple.get_rng()

    isolated_config = {
        "lethal_search": {"enabled": True, "module": "lethal_simple", "isolated_rng": True}
    }
    ml_policy_agent._try_lethal(obs, config=isolated_config)
    assert captured_rng == [dedicated]

    captured_rng.clear()
    ml_policy_agent._try_lethal(obs, config=_LETHAL_ON)  # isolated_rng キー無し
    assert captured_rng == [None]


def test_try_lethal_isolated_rng_true_leaves_global_random_untouched(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """isolated_rng=true: リーサル探索が何回サンプリングしても(見つかる/見つからない
    いずれでも)グローバル random の状態は `_try_lethal` 呼び出し前後で完全に不変。"""
    obs = _obs(encoder_observations, "mid_game")
    config = {
        "lethal_search": {"enabled": True, "module": "lethal_simple", "isolated_rng": True}
    }
    monkeypatch.setattr(ml_policy_agent, "build_dummy_search_state", _shuffling_build_stub)

    for sample_calls, result in ((1, None), (4, [0])):
        monkeypatch.setitem(
            ml_policy_agent._SEARCH_MODULES, "lethal_simple",
            _search_stub_calling_factory_n_times(sample_calls, result),
        )
        ml_policy_agent.lethal_simple.reset_rng(2026)
        before = random.getstate()
        ml_policy_agent._try_lethal(obs, config=config)
        after = random.getstate()
        assert before == after, f"sample_calls={sample_calls} で global random が消費された"


def test_try_lethal_without_isolated_rng_consumption_varies_with_sample_count(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """対照: isolated_rng が無ければ(現行どおり)グローバル random を消費し、サンプリング
    回数(探索の実行量)が変わると到達する状態も変わる(=交絡の再現)。"""
    obs = _obs(encoder_observations, "mid_game")
    config = {"lethal_search": {"enabled": True, "module": "lethal_simple"}}  # isolated_rng キー無し
    monkeypatch.setattr(ml_policy_agent, "build_dummy_search_state", _shuffling_build_stub)

    random.seed(555)
    state_before = random.getstate()

    random.setstate(state_before)
    monkeypatch.setitem(
        ml_policy_agent._SEARCH_MODULES, "lethal_simple",
        _search_stub_calling_factory_n_times(1, None),
    )
    ml_policy_agent._try_lethal(obs, config=config)
    state_after_1 = random.getstate()
    assert state_after_1 != state_before  # 何かしら消費している(対照)

    random.setstate(state_before)
    monkeypatch.setitem(
        ml_policy_agent._SEARCH_MODULES, "lethal_simple",
        _search_stub_calling_factory_n_times(4, [0]),
    )
    ml_policy_agent._try_lethal(obs, config=config)
    state_after_4 = random.getstate()

    assert state_after_1 != state_after_4  # 実行量が変われば消費量(到達状態)も変わる


def test_try_lethal_isolated_rng_same_seed_is_reproducible(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """専用RNGを同一seedでリセットして2回 `_try_lethal` を実行すると、
    hidden_state_factory に渡る乱数列(=探索結果を左右する入力)が一致する。"""
    obs = _obs(encoder_observations, "mid_game")
    config = {
        "lethal_search": {"enabled": True, "module": "lethal_simple", "isolated_rng": True}
    }
    captured: list[list[int]] = []

    def _record(obs_arg, full_deck, rng=None):
        pool = [1, 2, 3, 4, 5]
        (rng if rng is not None else random).shuffle(pool)
        captured.append(list(pool))
        return None

    monkeypatch.setattr(ml_policy_agent, "build_dummy_search_state", _record)
    monkeypatch.setitem(
        ml_policy_agent._SEARCH_MODULES, "lethal_simple",
        _search_stub_calling_factory_n_times(1, None),
    )

    ml_policy_agent.lethal_simple.reset_rng(777)
    ml_policy_agent._try_lethal(obs, config=config)
    first = captured[-1]

    ml_policy_agent.lethal_simple.reset_rng(777)
    ml_policy_agent._try_lethal(obs, config=config)
    second = captured[-1]

    assert first == second


def test_agent_match_start_resets_lethal_rng_only_when_isolated(ml_policy_agent, monkeypatch):
    """試合開始(select=None)で isolated_rng=true のときだけ `lethal_simple.reset_rng` が
    呼ばれる。既定/キー無しではグローバル random に一切触れない(消費0)。
    seed 省略時はグローバル random から1回だけ引く(=試合単位で再現的、グローバル列は
    その1回だけ進む)。"""
    from cg.api import to_observation_class

    obs = to_observation_class({"current": None, "logs": [], "select": None})
    calls: list = []
    monkeypatch.setattr(
        ml_policy_agent.lethal_simple, "reset_rng", lambda seed=None: calls.append(seed)
    )

    # isolated_rng 無し(既定) -> reset_rng は呼ばれず、グローバル random も消費しない。
    before0 = random.getstate()
    ml_policy_agent.agent(obs, config={"lethal_search": {"enabled": True}})
    after0 = random.getstate()
    assert calls == []
    assert before0 == after0

    # isolated_rng=true かつ明示seed -> そのseedでreset、グローバルrandomは触らない。
    before1 = random.getstate()
    ml_policy_agent.agent(
        obs,
        config={
            "lethal_search": {"enabled": True, "isolated_rng": True, "isolated_rng_seed": 99}
        },
    )
    after1 = random.getstate()
    assert calls == [99]
    assert before1 == after1

    # isolated_rng=true・seed省略 -> グローバル random から1回だけ引いてシードする。
    calls.clear()
    before2 = random.getstate()
    ml_policy_agent.agent(obs, config={"lethal_search": {"enabled": True, "isolated_rng": True}})
    after2 = random.getstate()
    assert len(calls) == 1 and calls[0] is not None
    assert before2 != after2


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
        def __init__(self, weights_path=None, deck_card_ids=None):
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


# ---------------------------------------------------------------------------
# _try_deck_sustain (vs-crustle 縦串P0 / 山札切れ自滅対策)
# ---------------------------------------------------------------------------

# lethal 無効・deck_sustain 有効の最小 config(発火条件はユーザー確認済み Option A)。
_DECK_SUSTAIN_ON = {
    "lethal_search": {"enabled": False},
    "deck_sustain": {"enabled": True, "deck_count_threshold": 8, "max_rank": 4},
}


class _StubScoreModel:
    """`score_options` が固定スコアを返すだけのモデルスタブ(deck_sustain の順位判定用)。"""

    def __init__(self, scores):
        self._scores = scores

    def score_options(self, obs, hidden_state_factory=None, deadline=None):
        return list(self._scores)

    def select_option(self, obs, hidden_state_factory=None, deadline=None):
        return max(range(len(self._scores)), key=lambda i: self._scores[i])


def _deck_sustain_obs(encoder_observations, deck_count, recovery_card_id=1129):
    """mid_game を土台に「自分(player0)の deckCount を deck_count に、option[1] を回復札
    (PLAY, cardId=recovery_card_id)にした」合成 obs を作る。option は 4件(PLAY,PLAY,RETREAT,END)。
    """
    import copy

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["current"]["players"][0]["deckCount"] = deck_count
    # option[1] を回復札の PLAY にする(type 7 = PLAY)。他は据え置き。
    base["select"]["option"][1] = {"index": 5, "type": 7, "cardId": recovery_card_id}
    base["logs"] = []
    from cg.api import to_observation_class

    return to_observation_class(base)


def test_deck_sustain_returns_none_when_disabled(ml_policy_agent, encoder_observations, monkeypatch):
    """config未指定/`deck_sustain`キー無しでは常にNone(モデルも呼ばない)。本番挙動不変の回帰。"""
    obs = _deck_sustain_obs(encoder_observations, deck_count=5)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when deck_sustain disabled")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    assert ml_policy_agent._try_deck_sustain(obs, config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_deck_sustain(obs, config={}) is None


def test_deck_sustain_returns_none_when_deck_not_critical(ml_policy_agent, encoder_observations, monkeypatch):
    """残り山札が閾値超なら発火せず、スコアリング(モデル)にも到達しない。"""
    obs = _deck_sustain_obs(encoder_observations, deck_count=33)  # > 8

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when deck is not critical")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_ON) is None


def test_deck_sustain_returns_none_when_no_recovery_option(ml_policy_agent, encoder_observations, monkeypatch):
    """山札逼迫でも回復札の選択肢が無ければ、スコアリング前にNone(モデルを呼ばない)。"""
    obs = _obs(encoder_observations, "mid_game")  # deckCount 33 だが option に回復札なし
    # deckCount を逼迫させるため players を直接いじった obs を作り直す(回復札は入れない)。
    import copy
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["current"]["players"][0]["deckCount"] = 5
    base["logs"] = []
    obs = to_observation_class(base)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when no recovery option exists")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_ON) is None


def test_deck_sustain_picks_recovery_when_policy_plausible(ml_policy_agent, encoder_observations, monkeypatch):
    """山札逼迫+回復札あり+回復札が生スコア上位max_rank以内なら、その回復札を選ぶ。"""
    obs = _deck_sustain_obs(encoder_observations, deck_count=5)
    # 回復札は option[1]。スコアは idx1(回復札)を rank2(上位4以内)に置く。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_ON) == [1]
    # agent() 経由でも lethal 無効・deck_sustain 有効なら回復札が採用され、pipeline より先に返る。
    assert ml_policy_agent.agent(obs, config=_DECK_SUSTAIN_ON) == [1]


def test_deck_sustain_soft_gate_skips_when_policy_ranks_recovery_low(ml_policy_agent, encoder_observations, monkeypatch):
    """回復札の生スコア順位が max_rank 位より下なら触らない(安全側の低recall)。"""
    obs = _deck_sustain_obs(encoder_observations, deck_count=5)
    config = {"lethal_search": {"enabled": False},
              "deck_sustain": {"enabled": True, "deck_count_threshold": 8, "max_rank": 1}}
    # max_rank=1 なら回復札(idx1)は rank>=1 で不採用。idx1 のスコアを rank1(=順位1)にする。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))

    assert ml_policy_agent._try_deck_sustain(obs, config=config) is None


def test_deck_sustain_swallows_model_exceptions(ml_policy_agent, encoder_observations, monkeypatch):
    """score_options が例外を投げてもクラッシュせず None(既存フローに委ねる)。"""
    obs = _deck_sustain_obs(encoder_observations, deck_count=5)

    class _Boom:
        def score_options(self, obs, hidden_state_factory=None, deadline=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _Boom())

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_ON) is None


# --- ループ抑制(戦略依存): Dudunsparce「Run Away Draw」(ABILITY, cardId 66) ---

# suppress を有効にした config(deckburn_ability_ids は既定 66 を使う)。
_DECK_SUSTAIN_SUPPRESS_ON = {
    "lethal_search": {"enabled": False},
    "deck_sustain": {"enabled": True, "deck_count_threshold": 8, "max_rank": 4,
                     "suppress_deckburn_when_ahead": True},
}


def _deck_sustain_burn_obs(encoder_observations, deck_count, my_prize, opp_prize, ability_card_id=66):
    """mid_game を土台に、自分の deckCount / サイド残(my/opp) を設定し、option[1] を
    Dudunsparce「Run Away Draw」(ABILITY=10, cardId=ability_card_id)にした合成 obs。回復札は無し。
    """
    import copy

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["current"]["players"][0]["deckCount"] = deck_count
    base["current"]["players"][0]["prize"] = [None] * my_prize
    base["current"]["players"][1]["prize"] = [None] * opp_prize
    base["select"]["option"][1] = {"index": 5, "type": 10, "cardId": ability_card_id}  # 10=ABILITY
    base["logs"] = []
    from cg.api import to_observation_class

    return to_observation_class(base)


def test_deck_sustain_suppresses_deckburn_when_ahead(ml_policy_agent, encoder_observations, monkeypatch):
    """山札逼迫+サイド先行(my<=opp)で、ポリシー top1 が Run Away Draw なら最善の非burn手に振る。"""
    obs = _deck_sustain_burn_obs(encoder_observations, deck_count=5, my_prize=2, opp_prize=4)
    # idx1(=Run Away Draw)を最高スコアにする。非burnの最善は idx0(0.5)。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_SUPPRESS_ON) == [0]
    assert ml_policy_agent.agent(obs, config=_DECK_SUSTAIN_SUPPRESS_ON) == [0]


def test_deck_sustain_does_not_suppress_when_behind(ml_policy_agent, encoder_observations, monkeypatch):
    """サイド劣勢(my>opp)なら抑制せず(スコアリングにも到達しない)。札と打点が要るため。"""
    obs = _deck_sustain_burn_obs(encoder_observations, deck_count=5, my_prize=4, opp_prize=2)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when behind (no suppression, no recovery)")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_SUPPRESS_ON) is None


def test_deck_sustain_no_suppress_when_flag_absent(ml_policy_agent, encoder_observations, monkeypatch):
    """suppress_deckburn_when_ahead 未指定なら、先行でも Run Away Draw を止めない(不介入)。"""
    obs = _deck_sustain_burn_obs(encoder_observations, deck_count=5, my_prize=2, opp_prize=4)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when suppression flag is absent and no recovery")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    # _DECK_SUSTAIN_ON は suppress キーを持たない。
    assert ml_policy_agent._try_deck_sustain(obs, config=_DECK_SUSTAIN_ON) is None


# ---------------------------------------------------------------------------
# _try_hammer_veto: card_ids 一般化(改造ハンマー1081→クラッシュハンマー1120にも拡張)
# ---------------------------------------------------------------------------

_HAMMER_VETO_ON = {"lethal_search": {"enabled": False}, "hammer_veto": {"enabled": True}}
_HAMMER_VETO_ON_CRUSHING = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120]},
}


def _hammer_veto_obs(encoder_observations, hammer_card_id, *, clear_opp_bench_energy=False):
    """mid_game を土台に option[1] を指定ハンマーIDの PLAY(type=7)にした合成 obs。

    mid_game のデフォルトは相手(player1)ベンチに Basic {F} Energy(id=6、非特殊)x3 が
    付いている(guard の「特殊エネ vs 任意エネ」切替を確認するための素材)。
    clear_opp_bench_energy=True ならそれを空にし、guard を完全に外した状態を作る。
    """
    import copy
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["select"]["option"][1] = {"index": 5, "type": 7, "cardId": hammer_card_id}
    base["logs"] = []
    if clear_opp_bench_energy:
        for pk in base["current"]["players"][1]["bench"]:
            pk["energies"] = []
            pk["energyCards"] = []
    return to_observation_class(base)


def test_hammer_veto_default_card_ids_ignore_crushing_hammer(ml_policy_agent, encoder_observations, monkeypatch):
    """card_ids 未指定(既定 [1081])では、選んだ手がクラッシュハンマー(1120)でも無関係=None。

    KO探索にすら到達しないことを「呼ばれたら失敗」スタブで確認する
    (=現行挙動(改造ハンマー限定)がバイト不変であることの回帰)。
    """
    from ptcg_ai.search import ko_search

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when chosen card is not in default card_ids")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)

    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON) is None


def test_hammer_veto_card_ids_config_enables_crushing_hammer(ml_policy_agent, encoder_observations, monkeypatch):
    """card_ids=[1120] を指定すると、KO可能かつベンチに正当な対象が無ければクラッシュハンマーも
    差し替わる(改造ハンマー専用だった現行挙動の一般化)。
    """
    from ptcg_ai.search import ko_search

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    # idx1(=クラッシュハンマー)を最高スコアにする。除外後の最善は idx0(0.5)。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) == [0]


def test_hammer_veto_guard_switches_special_vs_any_energy(ml_policy_agent, encoder_observations, monkeypatch):
    """同一盤面(相手ベンチに非特殊エネのみ)で、改造ハンマーは guard に引っかからず veto が
    発火するが、クラッシュハンマーは(任意のエネで判定するため)guard が発火して veto しない。
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    enhanced_obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1081)
    assert ml_policy_agent._try_hammer_veto(enhanced_obs, [1], config=_HAMMER_VETO_ON) == [0]

    crushing_obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120)
    assert ml_policy_agent._try_hammer_veto(crushing_obs, [1], config=_HAMMER_VETO_ON_CRUSHING) is None


# ---------------------------------------------------------------------------
# hammer_veto.ignore_bench_guard: guard を無視して常に温存する ablation arm
# ---------------------------------------------------------------------------

_HAMMER_VETO_ON_CRUSHING_NOGUARD = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "ignore_bench_guard": True},
}


def test_hammer_veto_ignore_bench_guard_switches_guard_off(ml_policy_agent, encoder_observations, monkeypatch):
    """同一盤面(相手ベンチに任意エネあり)で、guard あり=vetoしない / ignore_bench_guard=vetoする。

    実ラダー監査では「ベンチにエネがある」局面でも実際の狙いはアクティブで、guard が浪費を
    素通ししていた。その素通しを潰す arm(温存主線)の on/off 差を固定する。
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120)
    # guard あり(現行): ベンチに正当な対象があるので veto しない。
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) is None
    # ignore_bench_guard: KO可能なら常に veto=ハンマー以外の最善手(idx0)へ。
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_NOGUARD) == [0]


def test_hammer_veto_records_ko_cache_for_redirect(ml_policy_agent, encoder_observations, monkeypatch):
    """guard で veto を見送った場合でも、KO判定は次の decision 用にキャッシュされる。

    (redirect は「ハンマーは打つ / ただしターゲットを変える」なので、guard で見送った
    ケースこそキャッシュが必要になる。)
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", None)

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) is None

    cache = ml_policy_agent._hammer_ko_cache
    assert cache is not None
    assert cache["card_id"] == 1120
    assert cache["can_ko"] is True
    assert cache["select_seq"] == ml_policy_agent._selects_seen


def test_hammer_veto_clears_ko_cache_when_play_is_replaced(ml_policy_agent, encoder_observations, monkeypatch):
    """veto が実際に差し替えた=ハンマーは打たれない → 直後にターゲット選択は来ないのでキャッシュを捨てる。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", {"stale": True})

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) == [0]
    assert ml_policy_agent._hammer_ko_cache is None


# ---------------------------------------------------------------------------
# _try_hammer_redirect: ハンマーのターゲットをアクティブ→ベンチへ振り替える
# (config-gated hammer_veto.redirect_target。実ラダー監査で浪費16件が全てアクティブ狙いだった)
# ---------------------------------------------------------------------------

_HAMMER_REDIRECT_ON = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "redirect_target": True},
}

# 実リプレイで確認した値(cg/api.py の Enum と一致): SelectType.ENERGY=4 /
# SelectContext.DISCARD_ENERGY=30 / OptionType.ENERGY=6 / AreaType.ACTIVE=4, BENCH=5。
_SEL_ENERGY = 4
_CTX_DISCARD_ENERGY = 30
_OPT_ENERGY = 6
_AREA_ACTIVE = 4
_AREA_BENCH = 5


def _hammer_target_obs(
    encoder_observations,
    *,
    effect_card_id=1120,
    bench_options=2,
    select_type=_SEL_ENERGY,
    context=_CTX_DISCARD_ENERGY,
    opp_index=1,
):
    """クラッシュハンマーの「剥がすエネを選ぶ select」を模した合成 obs。

    スキーマは Kaggle 実ラダーのリプレイ(episode-92385834 ほか、1120 の PLAY ログが載る行の
    observation.select)を実測したもの: type=4(ENERGY) / context=30(DISCARD_ENERGY) /
    effect=Card(id=1120) / option=[{type:6, area:4|5, index, playerIndex:相手, energyIndex, count}]。
    option[0] が相手アクティブ、以降が相手ベンチ。
    """
    import copy

    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["logs"] = []
    options = [
        {"type": _OPT_ENERGY, "area": _AREA_ACTIVE, "index": 0,
         "playerIndex": opp_index, "energyIndex": 0, "count": 1},
    ]
    for bench_index in range(bench_options):
        options.append({
            "type": _OPT_ENERGY, "area": _AREA_BENCH, "index": bench_index,
            "playerIndex": opp_index, "energyIndex": 0, "count": 1,
        })
    base["select"] = {
        "type": select_type,
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 1,
        "option": options,
        "deck": None,
        "contextCard": None,
        "effect": None if effect_card_id is None else {
            "id": effect_card_id, "playerIndex": 0, "serial": 35,
        },
    }
    return to_observation_class(base)


def _fresh_ko_cache(ml_policy_agent, can_ko=True, card_id=1120, turn=9, your_index=0):
    """「1つ前の decision で作られた」扱いになる KO 判定キャッシュを作る。"""
    return {
        "card_id": card_id, "can_ko": can_ko, "turn": turn,
        "your_index": your_index, "select_seq": ml_policy_agent._selects_seen - 1,
    }


def test_hammer_redirect_moves_target_from_active_to_bench(ml_policy_agent, encoder_observations, monkeypatch):
    """(a) KO可能 + ベンチ対象あり + 現在の選択がアクティブ → 方策スコア最上位のベンチへ振替。"""
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))
    # option = [active, bench0, bench1]。ベンチ候補のうちスコア最大は idx2。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_REDIRECT_ON) == [2]


def test_hammer_redirect_no_intervention_when_cannot_ko(ml_policy_agent, encoder_observations, monkeypatch):
    """(b) このターンKOできない=アクティブから剥がすのは無駄でない → 介入しない。"""
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent, can_ko=False))
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_REDIRECT_ON) is None


def test_hammer_redirect_no_intervention_without_bench_target(ml_policy_agent, encoder_observations, monkeypatch):
    """(c) 振替先(ベンチ対象)が無ければ介入しない。KO探索にも到達しない。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when there is no bench target to redirect to")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", None)

    obs = _hammer_target_obs(encoder_observations, bench_options=0)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_REDIRECT_ON) is None


def test_hammer_redirect_off_without_config_key(ml_policy_agent, encoder_observations, monkeypatch):
    """(d) redirect_target キーが無い config(=本番/現行)では何もしない(KO探索にも到達しない)。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when redirect_target is absent")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_VETO_ON_CRUSHING) is None
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_LETHAL_ON) is None


def test_hammer_redirect_state_resets_on_new_match(ml_policy_agent, monkeypatch):
    """(e) 試合開始(select=None)でKO判定キャッシュがクリアされる(worker再利用でも漏れない)。"""
    from cg.api import to_observation_class

    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))
    deck = ml_policy_agent.agent(to_observation_class({"current": None, "logs": [], "select": None}))

    assert len(deck) == 60
    assert ml_policy_agent._hammer_ko_cache is None


def test_hammer_redirect_uses_cache_without_rerunning_ko_search(ml_policy_agent, encoder_observations, monkeypatch):
    """直前の decision のキャッシュがあれば ko_search を張り直さない(1手あたりのコストを増やさない)。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not re-run when a fresh cache is available")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_REDIRECT_ON) == [2]


def test_hammer_redirect_recomputes_ko_when_cache_is_stale(ml_policy_agent, encoder_observations, monkeypatch):
    """キャッシュが古い(別ターン/別decision)なら ko_search を張り直して判断する。"""
    from ptcg_ai.search import ko_search

    calls = []

    def _fake_ko(*args, **kwargs):
        calls.append(1)
        return True

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _fake_ko)
    stale = _fresh_ko_cache(ml_policy_agent, can_ko=False)
    stale["select_seq"] -= 5  # 5 decision 前=古い
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", stale)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [0], config=_HAMMER_REDIRECT_ON) == [2]
    assert len(calls) == 1


def test_hammer_redirect_skips_when_already_targeting_bench(ml_policy_agent, encoder_observations, monkeypatch):
    """既にベンチを狙っている選択には介入しない(方策の判断を尊重)。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when the current choice is not the active target")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(obs, [1], config=_HAMMER_REDIRECT_ON) is None


def test_hammer_redirect_ignores_other_effect_selects(ml_policy_agent, encoder_observations, monkeypatch):
    """対象ハンマー以外の効果によるエネ選択(effect違い/effect無し/type違い)には触らない。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run for non-hammer target selects")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))

    other_effect = _hammer_target_obs(encoder_observations, effect_card_id=1081)
    assert ml_policy_agent._try_hammer_redirect(other_effect, [0], config=_HAMMER_REDIRECT_ON) is None

    no_effect = _hammer_target_obs(encoder_observations, effect_card_id=None)
    assert ml_policy_agent._try_hammer_redirect(no_effect, [0], config=_HAMMER_REDIRECT_ON) is None

    other_context = _hammer_target_obs(encoder_observations, context=8)
    assert ml_policy_agent._try_hammer_redirect(other_context, [0], config=_HAMMER_REDIRECT_ON) is None


def test_hammer_redirect_wired_into_apply_action_vetoes(ml_policy_agent, encoder_observations, monkeypatch):
    """`_apply_action_vetoes` 経由でも振替が効く(= _select_action の最終段に載っている)。"""
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", _fresh_ko_cache(ml_policy_agent))
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_HAMMER_REDIRECT_ON) == [2]
    # config にキーが無ければ素通し(本番不変)。
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_LETHAL_ON) == [0]


# ---------------------------------------------------------------------------
# hammer_veto.veto_mode: "shadow"(KO判定+キャッシュ構築のみ、手は差し替えない) vs
# 既定"swap"(現行挙動)。redirect単独A/B用のarm(1500試合A/Bでveto+redirect合成は
# 勝率-0.87pt(ns)、veto発火0.29→1.01回/試合は実ラダー無駄撃ちの3.5倍=ko_searchの
# 楽観誤りで正当なハンマーまで抑止している疑いがある一方、redirectは的が絞れている
# ため単独で切り分けたい)。
# ---------------------------------------------------------------------------

_HAMMER_VETO_ON_CRUSHING_SHADOW = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "veto_mode": "shadow"},
}
_HAMMER_VETO_ON_CRUSHING_SWAP = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "veto_mode": "swap"},
}
_HAMMER_REDIRECT_SHADOW_ON = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {
        "enabled": True, "card_ids": [1120], "veto_mode": "shadow", "redirect_target": True,
    },
}


def test_hammer_veto_shadow_mode_does_not_replace_action(ml_policy_agent, encoder_observations, monkeypatch):
    """veto_mode="shadow": KO可能・guardも掛からない盤面でも手は差し替えない(常にNoneを返す)。

    差し替え用のスコアリング(_get_model)まで到達しないことを「呼ばれたら失敗」スタブで
    確認する(shadowは播き専用で、代替手選びのロジックには入らない)。
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)

    def _must_not_be_called(config=None):
        raise AssertionError("shadow mode must not fetch the model to score alternatives")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_SHADOW) is None


def test_hammer_veto_shadow_mode_builds_cache_and_redirect_still_works(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """veto_mode="shadow"でもKO判定キャッシュは構築され、破棄されない(直後のredirectが読める)。

    「PLAY決定(shadow veto、手は不変)→次decisionのターゲット選択(redirect)」の2手を
    実際に連続実行し、redirectが機能することまで確認する(キャッシュが shadow で捨てられて
    いたら redirect は動かないはずなので、これが本番の意図を保証する回帰)。
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", None)
    # decision seq を固定して「次のdecision」を明示的に作る(_select_action を経由せず
    # 直接関数を呼ぶユニットテストのため、他テストの実行順に依存しないよう固定する)。
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 10)

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_SHADOW) is None

    cache = ml_policy_agent._hammer_ko_cache
    assert cache is not None
    assert cache["card_id"] == 1120
    assert cache["can_ko"] is True
    assert cache["select_seq"] == 10

    # 次のdecision(ターゲット選択)へ進む。
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 11)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    target_obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(target_obs, [0], config=_HAMMER_REDIRECT_SHADOW_ON) == [2]


def test_hammer_veto_swap_mode_matches_default_behavior(ml_policy_agent, encoder_observations, monkeypatch):
    """veto_mode="swap"(明示)は veto_mode キー省略時と同じ結果=既存挙動がバイト不変であることの回帰。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_SWAP) == [0]
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) == [0]


# ---------------------------------------------------------------------------
# hammer_veto.apply_veto: veto_mode="shadow" の別名(False で PLAY を差し替えない)。
# R7 の `configs/abl_5_full_og_r7.json` はこの arm を使う(redirect 用の KO キャッシュだけ
# 供給し、veto 自体は掛けない)。キー省略時は True=現行挙動。
# ---------------------------------------------------------------------------

_HAMMER_VETO_ON_CRUSHING_NOAPPLY = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "apply_veto": False},
}
_HAMMER_VETO_ON_CRUSHING_APPLY = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {"enabled": True, "card_ids": [1120], "apply_veto": True},
}
_HAMMER_REDIRECT_NOAPPLY_ON = {
    "lethal_search": {"enabled": False},
    "hammer_veto": {
        "enabled": True, "card_ids": [1120], "apply_veto": False, "redirect_target": True,
    },
}


def test_hammer_veto_apply_false_does_not_replace_action(ml_policy_agent, encoder_observations, monkeypatch):
    """apply_veto=false: KO可能・guard も掛からない盤面でも PLAY を差し替えない(常に None)。

    差し替え用のスコアリング(_get_model)へ到達しないことを「呼ばれたら失敗」スタブで確認する
    (= shadow と同じ経路に乗っていること)。
    """
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)

    def _must_not_be_called(config=None):
        raise AssertionError("apply_veto=false must not fetch the model to score alternatives")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_NOAPPLY) is None


def test_hammer_veto_apply_false_builds_cache_and_redirect_still_works(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """apply_veto=false でも ko_search は走り KO キャッシュは構築・保持され、redirect が機能する。

    「PLAY決定(差し替えなし)→次decisionのターゲット選択(redirect)」の2手を連続実行して、
    R7 config(veto は掛けない / redirect は効かせる)の意図どおりに動くことを固定する。
    """
    from ptcg_ai.search import ko_search

    ko_calls = []

    def _ko(*args, **kwargs):
        ko_calls.append(1)
        return True

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko)
    monkeypatch.setattr(ml_policy_agent, "_hammer_ko_cache", None)
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 20)

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_NOAPPLY) is None
    assert ko_calls, "apply_veto=false でも ko_search は実行されなければならない"

    cache = ml_policy_agent._hammer_ko_cache
    assert cache is not None
    assert cache["card_id"] == 1120
    assert cache["can_ko"] is True
    assert cache["select_seq"] == 20

    # 次のdecision(ターゲット選択)= redirect が キャッシュを読んでベンチへ振り替える。
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 21)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.9, 0.1, 0.5]))

    target_obs = _hammer_target_obs(encoder_observations)
    assert ml_policy_agent._try_hammer_redirect(target_obs, [0], config=_HAMMER_REDIRECT_NOAPPLY_ON) == [2]


def test_hammer_veto_apply_true_matches_default_behavior(ml_policy_agent, encoder_observations, monkeypatch):
    """apply_veto=true(明示)は キー省略時と同じ結果=既存 config の挙動が不変であることの回帰。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", lambda *a, **k: True)
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([0.5, 1.0, 0.4, 0.1]))

    obs = _hammer_veto_obs(encoder_observations, hammer_card_id=1120, clear_opp_bench_energy=True)
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING_APPLY) == [0]
    assert ml_policy_agent._try_hammer_veto(obs, [1], config=_HAMMER_VETO_ON_CRUSHING) == [0]


# ---------------------------------------------------------------------------
# _try_strategy_residual (ii) 戦略残差NN (docs/plans/strategy-residual-nn-experiment.md)
# ---------------------------------------------------------------------------

_STRAT_ON = {"lethal_search": {"enabled": False}, "strategy_residual": {"enabled": True, "alpha": 1.0}}


class _StubResidual:
    """bias_options が固定 bias を返す残差スタブ(配線検証用)。"""

    def __init__(self, bias):
        self._bias = bias

    def bias_options(self, obs, select):
        return list(self._bias)


def test_strategy_residual_disabled_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """config未指定/`strategy_residual`キー無しでは常にNone(モデルも取得しない)。本番挙動不変の回帰。"""
    obs = _obs(encoder_observations, "mid_game")

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when strategy_residual disabled")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)

    assert ml_policy_agent._try_strategy_residual(obs, config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_strategy_residual(obs, config={}) is None


def test_strategy_residual_flips_choice_when_bias_overcomes(ml_policy_agent, encoder_observations, monkeypatch):
    """残差 bias が模倣 top1 を覆すと、その手を返す。argmax(score + alpha*bias)。"""
    obs = _obs(encoder_observations, "mid_game")  # 4択
    # 模倣 top1 = idx0(1.0)。bias で idx1 を 0.9->1.4 に押し上げて逆転させる。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    monkeypatch.setattr(ml_policy_agent, "_get_strategy_residual", lambda config: _StubResidual([0.0, 0.5, 0.0, 0.0]))

    assert ml_policy_agent._try_strategy_residual(obs, config=_STRAT_ON) == [1]
    assert ml_policy_agent.agent(obs, config=_STRAT_ON) == [1]


def test_strategy_residual_no_change_returns_none(ml_policy_agent, encoder_observations, monkeypatch):
    """残差 bias が模倣の結論(top1)を変えないなら None(既存フローに委ねる=挙動不変)。"""
    obs = _obs(encoder_observations, "mid_game")
    # bias が小さく idx0 のまま。
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    monkeypatch.setattr(ml_policy_agent, "_get_strategy_residual", lambda config: _StubResidual([0.0, 0.05, 0.0, 0.0]))

    assert ml_policy_agent._try_strategy_residual(obs, config=_STRAT_ON) is None


def test_strategy_residual_default_weights_are_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """既定(重み未ロード)の StrategyResidual は bias 全0 = 結論を変えない = None(スケルトンの無効性)。"""
    obs = _obs(encoder_observations, "mid_game")
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    # _get_strategy_residual はモンキーパッチせず、本物の既定残差(重み無し=zeros)を使う。

    assert ml_policy_agent._try_strategy_residual(obs, config=_STRAT_ON) is None


# ---------------------------------------------------------------------------
# _try_hand_damage_guard (Fix-B): 手札枚数に比例する相手ワザへの生存ガード
#
# 実ラダー og_r7 の負け分析:
#   - 93335550 row168 / 93324616 T9: リーリエの決心で手札6枚 → メガユキメノコex(861)の
#     うらみぶし(相手の手札の枚数×50)= 300 ダメージで 210〜240HP のオーガポンが即死。
#   - 93309236 T9: 相手フーディン(743)の ハンドパワー(手札×2ダメカン=×20)、相手手札15枚で
#     300 ダメージ。ジャッジマンで手札4枚に落とせば 80 ダメージまで下がって生存できた。
# ---------------------------------------------------------------------------

_JUDGE = 1213
_LILLIE = 1227
_HARLEQUIN = 1223
_BRIAR = 1201
_BOSS = 1182
_FROSLASS = 861   # メガユキメノコex: うらみぶし = 相手(=自分)の手札 × 50
_ALAKAZAM = 743   # フーディン: ハンドパワー = 自分(=相手)の手札 × 20

_HAND_GUARD_ON = {
    "lethal_search": {"enabled": False},
    "hand_damage_guard": {"enabled": True, "sources": [
        {"card_id": _FROSLASS, "per_card": 50, "count": "own_hand"},
        {"card_id": _ALAKAZAM, "per_card": 20, "count": "opp_hand"},
    ]},
}


def _hand_guard_obs(
    encoder_observations,
    option_card_ids,
    *,
    active_hp=240,
    my_prizes=1,
    opp_hand=7,
    opp_field_id=_FROSLASS,
    opp_field_active=False,
    opp_field_energies=None,
):
    """mid_game を土台に、手札連動打点の判定材料をすべて明示した合成 obs。

    `option_card_ids` は選択肢の並び。int なら PLAY(type=7, cardId=その値)、
    None なら END(type=14)にする。

    `opp_field_active`(I2): True なら `card_id` を相手アクティブに置く(既定はベンチ)。
    `opp_field_energies`(I2): 指定すると `card_id` の個体のエネルギー配列を明示的に上書きする
    (未指定なら fixture の既定値のまま=ベンチは非空・アクティブは空)。
    """
    import copy
    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    me, opp = base["current"]["players"][0], base["current"]["players"][1]
    me["active"][0]["hp"] = active_hp
    me["active"][0]["maxHp"] = active_hp
    me["prize"] = [None] * my_prizes
    opp["handCount"] = opp_hand
    if opp_field_id is not None:
        slot = opp["active"][0] if opp_field_active else opp["bench"][0]
        slot["id"] = opp_field_id
        if opp_field_energies is not None:
            slot["energies"] = list(opp_field_energies)
    else:
        opp["bench"] = []
        opp["active"][0]["id"] = 678  # 手札連動打点を持たないポケモン
    base["select"]["option"] = [
        {"type": 14} if cid is None else {"index": i, "type": 7, "cardId": cid}
        for i, cid in enumerate(option_card_ids)
    ]
    base["logs"] = []
    return to_observation_class(base)


def test_hand_damage_guard_disabled_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し/未指定では常に None(モデルも取得しない)= 本番挙動不変の回帰。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, None])

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when hand_damage_guard is disabled")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(
        obs, [0], config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config={}) is None


def test_hand_damage_guard_own_hand_swaps_lillie_for_judge(ml_policy_agent, encoder_observations, monkeypatch):
    """own_hand型(861): リーリエ後の手札6×50=300 >= 240 → ジャッジマン(手札4×50=200)へ差し替え。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, _BOSS, None], active_hp=240)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [1]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["hand_damage_guard_fired"] == 1
    assert stats["hand_damage_guard_to_judge"] == 1


def test_hand_damage_guard_own_hand_avoids_refill_when_no_judge(ml_policy_agent, encoder_observations, monkeypatch):
    """ジャッジ不在なら「手札を増やす手」だけを避けて次善手へ(93324616 T9 と同型)。"""
    # 選択肢: 0=リーリエ(選択中), 1=リーリエ, 2=クラウン, 3=ボスの指令, 4=END。
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _LILLIE, _HARLEQUIN, _BOSS, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.4, 0.1]))
    ml_policy_agent.reset_guard_stats()

    # リーリエ(0,1)もクラウン(2)も手札を増やすので除外され、ボスの指令(3)が残る。
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [3]
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_to_other"] == 1


def test_hand_damage_guard_own_hand_may_end_turn_without_supporter(ml_policy_agent, encoder_observations, monkeypatch):
    """代替が END しか無ければサポート未使用(END)でよい。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _HARLEQUIN, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [2]


def test_hand_damage_guard_does_not_fire_when_damage_is_survivable(ml_policy_agent, encoder_observations, monkeypatch):
    """予測手札×per_card < 実効HP なら発火せず、スコアリングにも到達しない。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, None], active_hp=310)  # 6*50=300 < 310

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when the hit is survivable")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) is None


def test_hand_damage_guard_does_not_fire_when_source_absent(ml_policy_agent, encoder_observations, monkeypatch):
    """相手の場に該当ポケモンが見えていなければ発火しない。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, None], opp_field_id=None)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when no hand-damage source is in play")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) is None


def test_hand_damage_guard_lillie_draws_eight_at_six_prizes(ml_policy_agent, encoder_observations, monkeypatch):
    """リーリエは自サイド6枚なら8枚ドロー。8*50=400 は 310HP でも即死=発火する。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, None], active_hp=310, my_prizes=6)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [1]
    # サイド5枚なら6枚ドロー=300 < 310 で発火しない(境界)。
    obs5 = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, None], active_hp=310, my_prizes=5)
    assert ml_policy_agent._try_hand_damage_guard(obs5, [0], config=_HAND_GUARD_ON) is None


def test_hand_damage_guard_opp_hand_prefers_judge(ml_policy_agent, encoder_observations, monkeypatch):
    """opp_hand型(743): 相手手札15×20=300 >= 240 → 選んだサポート(ブライア)をジャッジマンへ。"""
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _JUDGE, _LILLIE, None],
        active_hp=240, opp_hand=15, opp_field_id=_ALAKAZAM,
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.9, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [1]


def test_hand_damage_guard_opp_hand_no_judge_no_intervention(ml_policy_agent, encoder_observations, monkeypatch):
    """opp_hand型だけが発火してジャッジが無ければ、代替の当てが無いので介入しない。"""
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _BOSS, None],
        active_hp=240, opp_hand=15, opp_field_id=_ALAKAZAM,
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) is None


def test_hand_damage_guard_skips_judge_when_judge_does_not_save(ml_policy_agent, encoder_observations, monkeypatch):
    """ジャッジ後(手札4枚)でも死ぬなら、ジャッジ狙いでなく「手札を増やさない手」の最善へ。"""
    # per_card=100 → ジャッジ後でも 4*100=400 >= 240。
    config = {
        "lethal_search": {"enabled": False},
        "hand_damage_guard": {"enabled": True, "sources": [
            {"card_id": _FROSLASS, "per_card": 100, "count": "own_hand"}]},
    }
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, _BOSS, None], active_hp=240)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=config) == [1]
    # ジャッジ経由ではなく「増やさない手」経由で選ばれたことをカウンタで固定する。
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_to_judge"] == 0
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_to_other"] == 1


def test_hand_damage_guard_ignores_non_supporter_and_judge_itself(ml_policy_agent, encoder_observations, monkeypatch):
    """選んだ手がサポートでない(グッズ/END)、またはジャッジマン自身なら干渉しない。"""
    # 1122 = ポケギア3.0(グッズ)。
    obs = _hand_guard_obs(encoder_observations, [1122, _JUDGE, _LILLIE, None], active_hp=240)

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched for non-supporter / judge choices")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) is None  # グッズ
    assert ml_policy_agent._try_hand_damage_guard(obs, [1], config=_HAND_GUARD_ON) is None  # ジャッジ自身
    assert ml_policy_agent._try_hand_damage_guard(obs, [3], config=_HAND_GUARD_ON) is None  # END


def test_hand_damage_guard_wired_into_apply_action_vetoes(ml_policy_agent, encoder_observations, monkeypatch):
    """`_apply_action_vetoes` 経由でも差し替わる(キー無しでは素通り)。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, _BOSS, None], active_hp=240)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.5, 0.1]))

    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_HAND_GUARD_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(
        obs, [0], config={"lethal_search": {"enabled": False}}) == [0]


# ---------------------------------------------------------------------------
# _try_hand_damage_guard の imminent 絞り込み(Iteration 2)
#
# I1(presence-only)は og_r7 実ラダーの生存ガードとしては機能したが、mega_froslass_ex 対面の
# フィールド実測で -6.2pt(n=400、境界有意)の退行を出した(相手の場に861が見えるだけで
# リーリエ/クラウンを封じ、ドローエンジンを止めるコストが被弾回避の利得を上回っていた)。
# `imminent: "active_or_energized"` を source に付けると、その card_id の個体が
# **アクティブに居る、またはエネルギーが1個以上付いている**場合だけに発火を絞れる。
# ---------------------------------------------------------------------------

_HAND_GUARD_ON_IMMINENT = {
    "lethal_search": {"enabled": False},
    "hand_damage_guard": {"enabled": True, "sources": [
        {"card_id": _FROSLASS, "per_card": 50, "count": "own_hand",
         "imminent": "active_or_energized"},
        {"card_id": _ALAKAZAM, "per_card": 20, "count": "opp_hand",
         "imminent": "active_or_energized"},
    ]},
}


def test_hand_damage_guard_imminent_absent_key_keeps_i1_presence_only(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """`imminent` キーが無い source(I1 config そのまま)は、ベンチ・エネ0でも発火する
    (=I2 導入後も既定挙動は presence-only のまま=本番不変の回帰)。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, None], active_hp=240,
        opp_field_active=False, opp_field_energies=[],
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [1]


def test_hand_damage_guard_imminent_bench_no_energy_does_not_fire_own_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """own_hand型(861)に imminent 指定: ベンチ・エネ0の個体では発火しない(I2 の絞り込み対象)。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, None], active_hp=240,
        opp_field_active=False, opp_field_energies=[],
    )

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when the source is not imminent")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) is None


def test_hand_damage_guard_imminent_bench_with_energy_fires_own_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """own_hand型(861)に imminent 指定: ベンチでもエネルギーが付いていれば発火する。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, None], active_hp=240,
        opp_field_active=False, opp_field_energies=[3],  # WATER 1個
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) == [1]
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_fired"] == 1


def test_hand_damage_guard_imminent_active_no_energy_fires_own_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """own_hand型(861)に imminent 指定: アクティブならエネ0でも発火する
    (手貼り+攻撃が同ターンで揃うため=切迫している)。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, None], active_hp=240,
        opp_field_active=True, opp_field_energies=[],
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) == [1]


def test_hand_damage_guard_imminent_bench_no_energy_does_not_fire_opp_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """opp_hand型(743/ハンドパワー、超1)に imminent 指定: ベンチ・エネ0では発火しない。

    743 は特性ではなくワザ(Move Name: Powerful Hand, Cost: {P}=超1、
    data/EN_Card_Data.csv で確認)なので、861 と同じ imminent 判定を適用してよい。
    """
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _JUDGE, None], active_hp=240,
        opp_hand=15, opp_field_id=_ALAKAZAM, opp_field_active=False, opp_field_energies=[],
    )

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when the source is not imminent")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) is None


def test_hand_damage_guard_imminent_bench_with_energy_fires_opp_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """opp_hand型(743)に imminent 指定: ベンチでもエネルギーが付いていれば発火する。"""
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _JUDGE, None], active_hp=240,
        opp_hand=15, opp_field_id=_ALAKAZAM, opp_field_active=False, opp_field_energies=[5],
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) == [1]


def test_hand_damage_guard_imminent_active_no_energy_fires_opp_hand(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """opp_hand型(743)に imminent 指定: アクティブならエネ0でも発火する。"""
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _JUDGE, None], active_hp=240,
        opp_hand=15, opp_field_id=_ALAKAZAM, opp_field_active=True, opp_field_energies=[],
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.1]))
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) == [1]


def test_hand_damage_guard_imminent_mixed_sources_only_imminent_one_fires(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """両 source が imminent 指定でも、場に居るのが片方だけなら片方の判定だけで決まる
    (861 がベンチ・エネ0=非発火、743 は場に居ない=対象外)。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, None], active_hp=240,
        opp_field_active=False, opp_field_energies=[],  # 861 のみ場に居る、エネ0
    )

    def _must_not_be_called(config=None):
        raise AssertionError("model must not be fetched when nothing is imminent")

    monkeypatch.setattr(ml_policy_agent, "_get_model", _must_not_be_called)
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON_IMMINENT) is None


# ---------------------------------------------------------------------------
# _try_hand_damage_guard の substitute_only 絞り込み(Iteration 3)
#
# I1(presence-only)はフィールド計測で mega_froslass 対面 -6.2pt の退行を出した。実敗着局面
# (93335550 row168 / 93324616 row132 / 93309236 row130)を再監査すると、いずれも代替
# (ジャッジマン1213 or ボスの指令1182)が手札にあり差し替えで解決していた=退行の有害枝は
# 「代替が無いのにリーリエ/クラウンを封じるだけ封じてドローエンジンを止める」純損だった。
# `substitute_only: true` はこの純損ケースだけを介入対象から外す。
# ---------------------------------------------------------------------------

_HAND_GUARD_ON_SUBSTITUTE_ONLY = {
    "lethal_search": {"enabled": False},
    "hand_damage_guard": {"enabled": True, "substitute_only": True, "sources": [
        {"card_id": _FROSLASS, "per_card": 50, "count": "own_hand"},
        {"card_id": _ALAKAZAM, "per_card": 20, "count": "opp_hand"},
    ]},
}


def test_hand_damage_guard_substitute_only_absent_key_keeps_i1_behavior(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """`substitute_only` キーが無ければ、代替が無くても従来どおり介入する(I1 の回帰)。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _HARLEQUIN, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=_HAND_GUARD_ON) == [2]
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_no_substitute"] == 0


def test_hand_damage_guard_substitute_only_no_substitute_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """代替(ジャッジ/ボス)が選択肢に無ければ介入しない=リーリエをそのまま許す(93335550等の
    実敗着局面はすべて代替ありだったので、この経路には入らないはず=退行の有害枝そのもの)。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _HARLEQUIN, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(
        obs, [0], config=_HAND_GUARD_ON_SUBSTITUTE_ONLY) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["hand_damage_guard_fired"] == 1
    assert stats["hand_damage_guard_no_substitute"] == 1
    assert stats["hand_damage_guard_to_other"] == 0


def test_hand_damage_guard_substitute_only_boss_present_swaps_to_boss(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """代替がボスの指令だけでも(ジャッジ不在)差し替える=従来どおりの挙動を維持する。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _LILLIE, _HARLEQUIN, _BOSS, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.8, 0.4, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(
        obs, [0], config=_HAND_GUARD_ON_SUBSTITUTE_ONLY) == [3]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["hand_damage_guard_to_other"] == 1
    assert stats["hand_damage_guard_to_boss"] == 1
    assert stats["hand_damage_guard_no_substitute"] == 0


def test_hand_damage_guard_substitute_only_judge_present_still_prefers_judge(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """ジャッジが救える(93335550 row168 等の型): substitute_only でも従来どおりジャッジ優先。"""
    obs = _hand_guard_obs(
        encoder_observations, [_LILLIE, _JUDGE, _BOSS, None], active_hp=240)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.2, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(
        obs, [0], config=_HAND_GUARD_ON_SUBSTITUTE_ONLY) == [1]
    assert ml_policy_agent.get_guard_stats()["hand_damage_guard_to_judge"] == 1


def test_hand_damage_guard_substitute_only_judge_does_not_save_falls_back_to_judge_via_step2(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """ジャッジ後(手札4枚)でも死ぬ場合: substitute_only の候補(ジャッジ/ボス)のうち
    方策スコア最大の方(ここではジャッジ)を選ぶ。END/攻撃等の非代替は候補に入らない。"""
    config = {
        "lethal_search": {"enabled": False},
        "hand_damage_guard": {"enabled": True, "substitute_only": True, "sources": [
            {"card_id": _FROSLASS, "per_card": 100, "count": "own_hand"}]},
    }
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _JUDGE, _BOSS, None], active_hp=240)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(obs, [0], config=config) == [1]
    stats = ml_policy_agent.get_guard_stats()
    assert stats["hand_damage_guard_to_judge"] == 0
    assert stats["hand_damage_guard_to_other"] == 1
    assert stats["hand_damage_guard_to_boss"] == 0


def test_hand_damage_guard_substitute_only_opp_hand_only_still_no_intervention(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """opp_hand型だけが発火するときは、ボスの指令が手札にあっても介入しない(ボスは相手の
    手札を減らさないので脅威を解決しない=既存の G3-b 挙動を維持)。"""
    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _BOSS, None],
        active_hp=240, opp_hand=15, opp_field_id=_ALAKAZAM,
    )
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_hand_damage_guard(
        obs, [0], config=_HAND_GUARD_ON_SUBSTITUTE_ONLY) is None
    stats = ml_policy_agent.get_guard_stats()
    assert stats["hand_damage_guard_no_substitute"] == 0  # own_hand_threat 不成立で手前で return


def test_hand_damage_guard_substitute_only_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch,
):
    """`_apply_action_vetoes` 経由でも substitute_only が効く(代替なしなら不介入)。"""
    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _HARLEQUIN, None], active_hp=210)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    assert ml_policy_agent._apply_action_vetoes(
        obs, [0], config=_HAND_GUARD_ON_SUBSTITUTE_ONLY) == [0]


# ---------------------------------------------------------------------------
# _try_briar_gate (Fix-C): KOできないターンにブライア(1201)を切らない
# ---------------------------------------------------------------------------

_BRIAR_GATE_ON = {"lethal_search": {"enabled": False}, "briar_gate": True}


def _ko_stub(result, *, searched=True, aborted=False):
    """`ko_search.can_ko_this_turn` のスタブ。`report` にも実物と同じ内容を書く。

    briar_gate は「探索が完走して KO 不能と分かった」ときだけ介入するので、
    report を埋めないスタブでは介入しない(=このヘルパを通すことが必須)。
    """

    def _f(obs, factory, cfg=None, deadline=None, report=None):
        if report is not None:
            report["searched"] = searched
            report["aborted"] = aborted
        return result

    return _f


def test_briar_gate_disabled_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し/false では KO探索にすら到達しない(本番挙動不変の回帰)。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _BOSS, None])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when briar_gate is disabled")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    assert ml_policy_agent._try_briar_gate(
        obs, [0], config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_briar_gate(obs, [0], config={"briar_gate": False}) is None


def test_briar_gate_replaces_briar_when_no_ko_available(ml_policy_agent, encoder_observations, monkeypatch):
    """KOできないターンのブライアは次善手へ差し替える。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _JUDGE, _BOSS, None])
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["briar_gate_fired"] == 1


def test_briar_gate_keeps_briar_when_ko_is_available(ml_policy_agent, encoder_observations, monkeypatch):
    """KOできるターンなら追加サイドが実現する=正当な使用として干渉しない。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _JUDGE, None])
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) is None


def test_briar_gate_ignores_other_cards(ml_policy_agent, encoder_observations, monkeypatch):
    """ブライア以外を選んでいるときは KO探索を走らせない(コストゼロ)。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_LILLIE, _BRIAR, None])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when the chosen card is not Briar")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) is None


def test_briar_gate_dict_form_allows_budget_override(ml_policy_agent, encoder_observations, monkeypatch):
    """dict 形式 {"enabled": true, "ko_search": {...}} でも有効化でき、予算が渡る。"""
    from ptcg_ai.search import ko_search

    seen = {}

    def _spy(obs, factory, cfg=None, deadline=None, report=None):
        seen["cfg"] = cfg
        if report is not None:
            report["searched"] = True
            report["aborted"] = False
        return False

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _BOSS, None])
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _spy)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))
    config = {"lethal_search": {"enabled": False},
              "briar_gate": {"enabled": True, "ko_search": {"max_depth": 3}}}

    assert ml_policy_agent._try_briar_gate(obs, [0], config=config) == [1]
    assert seen["cfg"] == {"max_depth": 3}
    # enabled: false の dict は無効。
    assert ml_policy_agent._try_briar_gate(
        obs, [0], config={"briar_gate": {"enabled": False}}) is None


def test_briar_gate_then_hand_damage_guard_chain(ml_policy_agent, encoder_observations, monkeypatch):
    """93309236 T9 と同型: ブライアを弾いた次善手がリーリエでも、生存ガードがジャッジへ回す。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(
        encoder_observations, [_BRIAR, _LILLIE, _JUDGE, None],
        active_hp=240, opp_hand=15, opp_field_id=_ALAKAZAM,
    )
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.2, 0.1]))
    config = {**_HAND_GUARD_ON, "briar_gate": True}

    # briar_gate: [0](ブライア) -> [1](リーリエ)、hand_damage_guard: [1] -> [2](ジャッジ)。
    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=config) == [2]


def test_briar_gate_does_not_intervene_when_ko_search_is_inconclusive(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """KO探索が予算切れ/隠れ状態不能で完走しなかったときは介入しない(安全側)。

    `can_ko_this_turn` の False は「KOできない」と「判定不能」を区別しない。ブライアでは
    判定不能を KO不能と誤読すると**正当なブライアまで潰す**(実測: 93326434 T8 は 300ms で
    can_ko=True だが 100ms では時間切れで False)。
    """
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _JUDGE, None])
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.1]))

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))
    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) is None

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, searched=False))
    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) is None


def test_briar_gate_default_ko_budget_is_300ms(ml_policy_agent, encoder_observations, monkeypatch):
    """bool 形式の既定 KO探索予算は 300ms(ハンマーvetoの100msより厚い)。"""
    import time

    from ptcg_ai.search import ko_search

    seen = {}

    def _spy(obs, factory, cfg=None, deadline=None, report=None):
        seen["budget_ms"] = (deadline - time.perf_counter()) * 1000.0
        if report is not None:
            report["searched"] = True
            report["aborted"] = False
        return True

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _JUDGE, None])
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _spy)
    assert ml_policy_agent._try_briar_gate(obs, [0], config=_BRIAR_GATE_ON) is None
    assert 250 < seen["budget_ms"] <= 300


def test_ko_search_report_is_opt_in(ml_policy_agent, encoder_observations):
    """`report` を渡さない既存呼び出しの挙動は不変(引数追加の後方互換)。"""
    from ptcg_ai.search import ko_search

    obs = _hand_guard_obs(encoder_observations, [_BRIAR, _JUDGE, None])
    # hidden_state_factory=None → 探索せず False。report 無しでも例外にならない。
    assert ko_search.can_ko_this_turn(obs, None, {}, None) is False
    report: dict = {}
    assert ko_search.can_ko_this_turn(obs, None, {}, None, report=report) is False
    assert report == {"searched": False, "aborted": False}


# ---------------------------------------------------------------------------
# r9 Fix-D: _try_low_deck_draw_brake(山札僅少ブレーキ)
#
# 実ラダー68試合・32敗の全数調査: 5敗(15.6%)が自分の山札0による敗北。
# 93526443 T21(山5・手札3)でリーリエの決心 → 山2 → T23 山0 で、サイド先取の優勢形から自滅。
# 一方「速く回すこと自体」は勝ちと正相関なので、抑制は本当に山を減らす1手だけに絞る。
# ---------------------------------------------------------------------------

_BRAKE_ON = {"lethal_search": {"enabled": False},
             "low_deck_draw_brake": {"enabled": True, "deck_threshold": 6}}


def _brake_obs(
    encoder_observations,
    option_card_ids,
    *,
    deck_count=5,
    hand_size=3,
    my_prizes=4,
):
    """mid_game を土台に、山札枚数・手札枚数・サイド枚数を明示した MAIN 局面。

    `option_card_ids` は `_hand_guard_obs` と同じ(int=PLAY / None=END)。
    手札枚数はドローサポートの山札増減(= (手札-1) - 引く枚数)の入力になるので、
    fixture の手札配列を複製/切り詰めて指定枚数に合わせる。
    """
    import copy

    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    me = base["current"]["players"][0]
    me["deckCount"] = deck_count
    me["prize"] = [None] * my_prizes
    hand = list(me["hand"] or [])
    while len(hand) < hand_size:
        hand.append(copy.deepcopy(hand[-1]))
    me["hand"] = hand[:hand_size]
    me["handCount"] = hand_size
    base["select"]["option"] = [
        {"type": 14} if cid is None else {"index": i, "type": 7, "cardId": cid}
        for i, cid in enumerate(option_card_ids)
    ]
    base["logs"] = []
    return to_observation_class(base)


def test_low_deck_draw_brake_disabled_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し/false では KO探索にすら到達しない(本番挙動不変の回帰)。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when low_deck_draw_brake is disabled")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    assert ml_policy_agent._try_low_deck_draw_brake(
        obs, [0], config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_low_deck_draw_brake(
        obs, [0], config={"low_deck_draw_brake": {"enabled": False}}) is None


def test_low_deck_draw_brake_vetoes_lillie_at_low_deck(ml_policy_agent, encoder_observations, monkeypatch):
    """93526443 T21 と同型: 山5・手札3 のリーリエ(山-3)は次善手へ差し替える。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _LILLIE, _BOSS, None], deck_count=5, hand_size=3)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    # 対象のドローサポートは全て除外されるので、2枚目のリーリエ(idx1)ではなくボス(idx2)。
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_BRAKE_ON) == [2]
    assert ml_policy_agent.get_guard_stats()["low_deck_draw_brake_fired"] == 1


def test_low_deck_draw_brake_threshold_boundary(ml_policy_agent, encoder_observations, monkeypatch):
    """deck_threshold は「以下」で発火。6=発火 / 7=不発火。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))

    fires = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=6, hand_size=3)
    assert ml_policy_agent._try_low_deck_draw_brake(fires, [0], config=_BRAKE_ON) == [1]
    keeps = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=7, hand_size=3)
    assert ml_policy_agent._try_low_deck_draw_brake(keeps, [0], config=_BRAKE_ON) is None


def test_low_deck_draw_brake_never_vetoes_judge(ml_policy_agent, encoder_observations, monkeypatch):
    """ジャッジマン(1213)は対象外。手札を山に戻して4枚引く=手札5枚以上なら山札が増える。

    config で誤って card_ids に入れられてもジャッジは止めない(明示ガード)。
    """
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_JUDGE, _BOSS, None], deck_count=2, hand_size=8)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run for Judge")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_BRAKE_ON) is None
    misconfigured = {"lethal_search": {"enabled": False},
                     "low_deck_draw_brake": {"enabled": True, "deck_threshold": 6,
                                             "card_ids": [_JUDGE, _LILLIE]}}
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=misconfigured) is None


def test_low_deck_draw_brake_allows_draw_when_ko_is_available(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """このターンKOが取れるなら引いて勝ちに行ってよい(ブレーキは掛けない)。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=3, hand_size=3)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(True))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_BRAKE_ON) is None
    assert ml_policy_agent.get_guard_stats()["low_deck_draw_brake_skipped_ko"] == 1
    assert ml_policy_agent.get_guard_stats()["low_deck_draw_brake_fired"] == 0


def test_low_deck_draw_brake_skips_when_deck_would_grow(ml_policy_agent, encoder_observations, monkeypatch):
    """手札 > 引く枚数 なら山札は**増える**ので止めない(93517227 row104: 山10・手札9=山13)。"""
    from ptcg_ai.search import ko_search

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("ko_search must not run when the play grows the deck")

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _must_not_be_called)
    # 手札9 → 山に戻るのは8枚、引くのは6枚 = 山札 +2。
    grows = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=5, hand_size=9)
    assert ml_policy_agent._try_low_deck_draw_brake(grows, [0], config=_BRAKE_ON) is None
    # 手札7 → 戻り6・引き6 = ±0(減らないので対象外、境界)。
    flat = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=5, hand_size=7)
    assert ml_policy_agent._try_low_deck_draw_brake(flat, [0], config=_BRAKE_ON) is None


def test_low_deck_draw_brake_literal_mode_ignores_net_delta(ml_policy_agent, encoder_observations, monkeypatch):
    """`require_net_deck_loss: false`(ablation用)なら実増減を見ず literal に止める。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=5, hand_size=9)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    config = {"lethal_search": {"enabled": False},
              "low_deck_draw_brake": {"enabled": True, "deck_threshold": 6,
                                      "require_net_deck_loss": False}}
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=config) == [1]


def test_low_deck_draw_brake_harlequin_uses_five_card_draw(ml_policy_agent, encoder_observations, monkeypatch):
    """クラウン(1223)はコイン依存(5 or 3)なので山札に厳しい側=5枚引きで評価する。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    # 手札3 → 戻り2・引き5 = 山札 -3 で発火。
    fires = _brake_obs(encoder_observations, [_HARLEQUIN, _BOSS, None], deck_count=4, hand_size=3)
    assert ml_policy_agent._try_low_deck_draw_brake(fires, [0], config=_BRAKE_ON) == [1]
    # 手札6 → 戻り5・引き5 = ±0 で不発火。
    keeps = _brake_obs(encoder_observations, [_HARLEQUIN, _BOSS, None], deck_count=4, hand_size=6)
    assert ml_policy_agent._try_low_deck_draw_brake(keeps, [0], config=_BRAKE_ON) is None


def test_low_deck_draw_brake_lillie_draws_eight_at_six_prizes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """リーリエは自サイド6枚なら8枚引く(`hand_damage_guard` と同じ表・同じ補正を共有)。"""
    from ptcg_ai.search import ko_search

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    # 手札7・サイド6 → 戻り6・引き8 = 山札 -2 で発火。
    fires = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None],
                       deck_count=5, hand_size=7, my_prizes=6)
    assert ml_policy_agent._try_low_deck_draw_brake(fires, [0], config=_BRAKE_ON) == [1]
    # 同じ手札でサイド5なら引きは6枚 = ±0 で不発火(境界)。
    keeps = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None],
                       deck_count=5, hand_size=7, my_prizes=5)
    assert ml_policy_agent._try_low_deck_draw_brake(keeps, [0], config=_BRAKE_ON) is None


def test_low_deck_draw_brake_inconclusive_ko_search_does_not_intervene(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """KO探索が完走しなかった(予算切れ/隠れ状態不能)ら介入しない(過度な抑制を避ける)。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=3, hand_size=3)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, aborted=True))
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_BRAKE_ON) is None
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False, searched=False))
    assert ml_policy_agent._try_low_deck_draw_brake(obs, [0], config=_BRAKE_ON) is None
    assert ml_policy_agent.get_guard_stats()["low_deck_draw_brake_inconclusive"] == 2


def test_low_deck_draw_brake_wired_into_apply_action_vetoes(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """`_apply_action_vetoes` 経由でも差し替わる(キー無しでは素通り)。"""
    from ptcg_ai.search import ko_search

    obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=5, hand_size=3)
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))

    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_BRAKE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(
        obs, [0], config={"lethal_search": {"enabled": False}}) == [0]


# ---------------------------------------------------------------------------
# r9 Fix-E: _try_boss_lethal_gate / _try_boss_target_redirect(ボスのKOゲート)
#
# 93503044 T13: メガガルーラex(HP330)を 8エネ=270ダメージでKOできないのに引きずり出し、
# 60HP残しで逃げられて3サイド分をベンチに放置した。成功例(93528233 T6)は
# 「倒せる」か「最大エネ源を引き剥がす」ときだけだった。
# ---------------------------------------------------------------------------

_BOSS_GATE_ON = {"lethal_search": {"enabled": False},
                 "boss_lethal_gate": {"enabled": True, "allow_energy_denial": True}}


def _boss_play_obs(encoder_observations, option_card_ids, *, bench_energies=(0, 0)):
    """ボスの指令を PLAY しようとしている MAIN 局面(相手ベンチのエネ数を明示)。"""
    import copy

    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    opp = base["current"]["players"][1]
    bench = list(opp["bench"])
    while len(bench) < len(bench_energies):
        bench.append(copy.deepcopy(bench[-1]))
    bench = bench[:len(bench_energies)]
    for slot, n in zip(bench, bench_energies):
        slot["energies"] = [0] * n
        slot["energyCards"] = []
    opp["bench"] = bench
    base["select"]["option"] = [
        {"type": 14} if cid is None else {"index": i, "type": 7, "cardId": cid}
        for i, cid in enumerate(option_card_ids)
    ]
    base["logs"] = []
    return to_observation_class(base)


def _boss_target_obs(encoder_observations, bench_count=3, *, effect_id=_BOSS):
    """ボス直後の対象選択(実測スキーマ: SelectType.CARD / SelectContext.SWITCH / effect=ボス)。"""
    import copy

    from cg.api import to_observation_class

    base = copy.deepcopy(encoder_observations["mid_game"])
    base["select"] = {
        "type": 1,       # SelectType.CARD
        "context": 3,    # SelectContext.SWITCH
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "deck": None,
        "contextCard": None,
        "effect": None if effect_id is None else {"id": effect_id, "playerIndex": 0, "serial": 1},
        "option": [
            {"type": 3, "area": 5, "index": i, "playerIndex": 1}  # CARD / BENCH / 相手
            for i in range(bench_count)
        ],
    }
    base["logs"] = []
    return to_observation_class(base)


def _targets_stub(can_ko_flags, *, any_aborted=False, player_index=1):
    """`boss_target_eval.evaluate_targets` のスタブ(相手ベンチ index 順)。"""

    def _f(obs, factory, play_index, config=None, deadline=None):
        return {
            "targets": [
                {"option_index": i, "area": 5, "index": i, "player_index": player_index,
                 "can_ko": bool(flag), "aborted": False}
                for i, flag in enumerate(can_ko_flags)
            ],
            "any_aborted": any_aborted,
        }

    return _f


def test_boss_lethal_gate_disabled_is_inert(ml_policy_agent, encoder_observations, monkeypatch):
    """キー無し/false では対象評価にすら到達しない(本番挙動不変の回帰)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("boss_target_eval must not run when boss_lethal_gate is disabled")

    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _must_not_be_called)
    assert ml_policy_agent._try_boss_lethal_gate(
        obs, [0], config={"lethal_search": {"enabled": False}}) is None
    assert ml_policy_agent._try_boss_lethal_gate(
        obs, [0], config={"boss_lethal_gate": {"enabled": False}}) is None


def test_boss_lethal_gate_vetoes_when_no_target_is_koable(ml_policy_agent, encoder_observations, monkeypatch):
    """93503044 T13 と同型: 倒せる相手が1体も居ない(かつエネ0)ならボスを次善手へ。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, False]))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) == [1]
    assert ml_policy_agent.get_guard_stats()["boss_lethal_gate_fired"] == 1
    assert ml_policy_agent._boss_target_cache is None


def test_boss_lethal_gate_keeps_boss_when_a_target_is_koable(
    ml_policy_agent, encoder_observations, monkeypatch
):
    """倒せる相手が居るならボス自体は正当=差し替えない(どれを出すかは対象選択で直す)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, True, False]))
    ml_policy_agent.reset_guard_stats()

    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) is None
    assert ml_policy_agent.get_guard_stats()["boss_lethal_gate_fired"] == 0
    assert ml_policy_agent._boss_target_cache["allowed"] == {(5, 1, 1)}


def test_boss_lethal_gate_energy_denial_exception(ml_policy_agent, encoder_observations, monkeypatch):
    """KO不能でも「相手ベンチで最多エネ(1個以上)」の相手なら許可(93528233 T6 の成功例)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 8, 1))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, False]))

    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) is None
    cache = ml_policy_agent._boss_target_cache
    assert cache["allowed"] == {(5, 1, 1)} and cache["denial"] == {(5, 1, 1)}


def test_boss_lethal_gate_energy_denial_can_be_disabled(ml_policy_agent, encoder_observations, monkeypatch):
    """`allow_energy_denial: false` なら例外を使わず veto する(独立ablation)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 8, 1))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, False]))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))
    config = {"lethal_search": {"enabled": False},
              "boss_lethal_gate": {"enabled": True, "allow_energy_denial": False}}

    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=config) == [1]


def test_boss_lethal_gate_energy_denial_needs_energy(ml_policy_agent, encoder_observations, monkeypatch):
    """全員エネ0なら「最多エネ」は例外にならない(=veto)。0 が最多でも許可しない境界。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False]))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))

    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) == [1]


def test_boss_lethal_gate_inconclusive_does_not_intervene(ml_policy_agent, encoder_observations, monkeypatch):
    """対象評価が完走しなかった/判定不能なら介入しない(安全側)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0))
    ml_policy_agent.reset_guard_stats()

    monkeypatch.setattr(boss_target_eval, "evaluate_targets",
                        _targets_stub([False, False], any_aborted=True))
    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) is None
    monkeypatch.setattr(boss_target_eval, "evaluate_targets",
                        lambda *a, **k: None)
    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) is None
    assert ml_policy_agent.get_guard_stats()["boss_lethal_gate_inconclusive"] == 2


def test_boss_lethal_gate_ignores_other_cards(ml_policy_agent, encoder_observations, monkeypatch):
    """ボス以外を選んでいるときは対象評価を走らせない(コストゼロ)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_JUDGE, _BOSS, None])

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("boss_target_eval must not run for non-Boss choices")

    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _must_not_be_called)
    assert ml_policy_agent._try_boss_lethal_gate(obs, [0], config=_BOSS_GATE_ON) is None


def test_boss_target_redirect_switches_to_koable_target(ml_policy_agent, encoder_observations, monkeypatch):
    """PLAY 決定で作ったキャッシュを使い、対象を「倒せる相手」へ振り替える。"""
    from ptcg_ai.search import boss_target_eval

    play_obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, True]))
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 10)
    ml_policy_agent.reset_guard_stats()
    assert ml_policy_agent._try_boss_lethal_gate(play_obs, [0], config=_BOSS_GATE_ON) is None

    # 次の decision = 対象選択。実戦では倒せない bench0 を選んでいる。
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 11)
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.9, 0.2]))
    target_obs = _boss_target_obs(encoder_observations, bench_count=3)
    assert ml_policy_agent._try_boss_target_redirect(target_obs, [0], config=_BOSS_GATE_ON) == [2]
    assert ml_policy_agent.get_guard_stats()["boss_lethal_gate_redirected"] == 1


def test_boss_target_redirect_keeps_allowed_target(ml_policy_agent, encoder_observations, monkeypatch):
    """既に許可対象(=倒せる相手)を選んでいるなら介入しない。"""
    from ptcg_ai.search import boss_target_eval

    play_obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, True, False]))
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 4)
    assert ml_policy_agent._try_boss_lethal_gate(play_obs, [0], config=_BOSS_GATE_ON) is None

    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 5)
    target_obs = _boss_target_obs(encoder_observations, bench_count=3)
    assert ml_policy_agent._try_boss_target_redirect(target_obs, [1], config=_BOSS_GATE_ON) is None


def test_boss_target_redirect_requires_fresh_cache(ml_policy_agent, encoder_observations, monkeypatch):
    """キャッシュが「1つ前の decision」でなければ信用しない(古い判定で振り替えない)。"""
    from ptcg_ai.search import boss_target_eval

    play_obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, True]))
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 10)
    assert ml_policy_agent._try_boss_lethal_gate(play_obs, [0], config=_BOSS_GATE_ON) is None

    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 13)  # 3 decision 後 = 古い
    target_obs = _boss_target_obs(encoder_observations, bench_count=3)
    assert ml_policy_agent._try_boss_target_redirect(target_obs, [0], config=_BOSS_GATE_ON) is None


def test_boss_target_redirect_ignores_non_boss_select(ml_policy_agent, encoder_observations, monkeypatch):
    """ボス以外の効果による CARD/SWITCH 選択には触らない。"""
    from ptcg_ai.search import boss_target_eval

    play_obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False, True]))
    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 20)
    assert ml_policy_agent._try_boss_lethal_gate(play_obs, [0], config=_BOSS_GATE_ON) is None

    monkeypatch.setattr(ml_policy_agent, "_selects_seen", 21)
    other = _boss_target_obs(encoder_observations, bench_count=3, effect_id=1120)
    assert ml_policy_agent._try_boss_target_redirect(other, [0], config=_BOSS_GATE_ON) is None


def test_boss_lethal_gate_wired_into_apply_action_vetoes(ml_policy_agent, encoder_observations, monkeypatch):
    """`_apply_action_vetoes` 経由でも差し替わる(キー無しでは素通り)。"""
    from ptcg_ai.search import boss_target_eval

    obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False]))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))

    assert ml_policy_agent._apply_action_vetoes(obs, [0], config=_BOSS_GATE_ON) == [1]
    assert ml_policy_agent._apply_action_vetoes(
        obs, [0], config={"lethal_search": {"enabled": False}}) == [0]


def test_r9_gates_are_independent(ml_policy_agent, encoder_observations, monkeypatch):
    """2つの修正は独立キー: 片方だけONにしてももう片方は一切発火しない。"""
    from ptcg_ai.search import boss_target_eval, ko_search

    brake_obs = _brake_obs(encoder_observations, [_LILLIE, _BOSS, None], deck_count=5, hand_size=3)
    boss_obs = _boss_play_obs(encoder_observations, [_BOSS, _JUDGE, None], bench_energies=(0, 0))
    monkeypatch.setattr(ko_search, "can_ko_this_turn", _ko_stub(False))
    monkeypatch.setattr(boss_target_eval, "evaluate_targets", _targets_stub([False, False]))
    monkeypatch.setattr(ml_policy_agent, "_get_model",
                        lambda config=None: _StubScoreModel([1.0, 0.5, 0.1]))

    brake_only = {"lethal_search": {"enabled": False},
                  "low_deck_draw_brake": {"enabled": True, "deck_threshold": 6}}
    boss_only = {"lethal_search": {"enabled": False},
                 "boss_lethal_gate": {"enabled": True}}
    assert ml_policy_agent._apply_action_vetoes(brake_obs, [0], config=brake_only) == [1]
    assert ml_policy_agent._apply_action_vetoes(boss_obs, [0], config=brake_only) == [0]
    assert ml_policy_agent._apply_action_vetoes(boss_obs, [0], config=boss_only) == [1]
    assert ml_policy_agent._apply_action_vetoes(brake_obs, [0], config=boss_only) == [0]


def test_r9_configs_are_config_gated_and_independent(ml_policy_agent):
    """3つの config ファイルがキー構成として独立ablationになっていること。"""
    import json
    from pathlib import Path

    configs = Path(ml_policy_agent.__file__).resolve().parents[2] / "configs"
    r8 = json.loads((configs / "abl_5_full_og_r8.json").read_text(encoding="utf-8"))
    full = json.loads((configs / "abl_5_full_og_r9.json").read_text(encoding="utf-8"))
    brake = json.loads((configs / "abl_5_full_og_r9_deckbrake.json").read_text(encoding="utf-8"))
    boss = json.loads((configs / "abl_5_full_og_r9_bossgate.json").read_text(encoding="utf-8"))

    assert "low_deck_draw_brake" not in r8 and "boss_lethal_gate" not in r8
    assert full["low_deck_draw_brake"]["enabled"] and full["boss_lethal_gate"]["enabled"]
    assert brake["low_deck_draw_brake"]["enabled"] and "boss_lethal_gate" not in brake
    assert boss["boss_lethal_gate"]["enabled"] and "low_deck_draw_brake" not in boss
    # og_r8 の全キーを継承していること(isolated_rng を含む)。
    for cfg in (full, brake, boss):
        for key, value in r8.items():
            if key == "name":
                continue
            assert cfg[key] == value, key
        assert cfg["lethal_search"]["isolated_rng"] is True
