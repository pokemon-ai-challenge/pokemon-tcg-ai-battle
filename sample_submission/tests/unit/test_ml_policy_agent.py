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
