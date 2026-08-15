"""anytime 決定化スケジューリング(pipeline)と 時間予算v2(ml_policy_agent)のユニットテスト。

いずれも **config-gated** な追加であり、新しいキー(`pipeline.anytime` /
`time_budget.mode`)が無ければ従来挙動と完全に同じであることを併せて検証する。

test_pipeline.py と同じく、実 cg エンジンで先読みを回すと重く非決定的なので cg_api は
軽量なフェイクに差し替える。ラウンド打切りの検証は壁時計に依存しないよう、`pipeline.time` を
擬似クロックへ差し替えて「1候補の評価ごとに固定コストだけ時間が進む」形で決定的に再現する。
"""

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SAMPLE_SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
if str(SAMPLE_SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_SUBMISSION_ROOT))

_ENCODER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "encoder_observations.json"


@pytest.fixture(scope="module")
def pipeline():
    try:
        from ptcg_ai.search import pipeline as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pipeline / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def ml_policy_agent():
    try:
        from ptcg_ai.ml_policy import ml_policy_agent as mod
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ml_policy_agent / cg engine unavailable: {exc}")
    return mod


@pytest.fixture(scope="module")
def fixtures() -> dict:
    with _ENCODER_FIXTURE.open(encoding="utf-8") as f:
        return json.load(f)


def _obs(fixtures: dict, key: str):
    from cg.api import to_observation_class

    return to_observation_class({**fixtures[key], "logs": []})


class _StubPolicy:
    """score_options / score_options_from_state を固定スコアで返すスタブ。"""

    def __init__(self, scores):
        self._scores = scores

    def score_options(self, obs, factory=None, deadline=None):
        return list(self._scores)

    def score_options_from_state(self, state, select):
        return list(self._scores)


class _Clock:
    """擬似クロック。`t` を明示的に進めた分だけ perf_counter が進む。"""

    def __init__(self, start: float = 0.0):
        self.t = start

    def perf_counter(self) -> float:
        return self.t


class _FakeCg:
    """search_begin/step/release/end のフェイク(ラウンド方式の検証用)。

    `search_begin` は決定化 world を1本作る = ラウンド番号を進める。`search_step` は
    「そのラウンドでどの候補が評価されたか」を `steps` に記録し、末端ノードにラウンド番号と
    候補インデックスを埋め込む(末端評価スタブがそれを見てスコアを返す)。
    """

    def __init__(self, me: int, fail_at: set | None = None):
        self.me = me
        self.round = -1
        self.begins = 0
        self.steps: list[tuple[int, int]] = []      # (round, candidate index)
        self.ends = 0
        self._fail_at = fail_at or set()            # {(round, idx)} で ValueError を投げる

    def search_begin(self, obs, *args, **kwargs):
        self.round += 1
        self.begins += 1
        return SimpleNamespace(searchId=f"root{self.round}", observation=obs)

    def search_step(self, search_id, selection):
        idx = selection[0]
        if (self.round, idx) in self._fail_at:
            raise ValueError("illegal in this determinization")
        self.steps.append((self.round, idx))
        leaf_state = SimpleNamespace(
            result=-1, yourIndex=self.me, players=[None, None],
            round_index=self.round, candidate_index=idx,
        )
        leaf_obs = SimpleNamespace(current=leaf_state, select=None)
        return SimpleNamespace(searchId=f"n{self.round}_{idx}", observation=leaf_obs)

    def search_release(self, search_id):
        pass

    def search_end(self):
        self.ends += 1

    def rounds_order(self) -> list[list[int]]:
        """ラウンドごとの候補評価順。"""
        order: dict[int, list[int]] = {}
        for r, idx in self.steps:
            order.setdefault(r, []).append(idx)
        return [order[r] for r in sorted(order)]


class _ScoreEvaluator:
    """末端評価スタブ。評価1回ごとに擬似クロックを `cost` 進める(クロック未指定なら進めない)。"""

    def __init__(self, score_fn, clock: _Clock | None = None, cost: float = 0.0):
        self._score_fn = score_fn
        self._clock = clock
        self._cost = cost

    def evaluate(self, state, me):
        if self._clock is not None:
            self._clock.t += self._cost
        return self._score_fn(state.round_index, state.candidate_index)


_ANYTIME_ON = {
    "enabled": True,
    "max_rounds": 16,
    "min_rounds_for_early_stop": 3,
    "early_stop_margin": 0.04,
    "rotate_candidates": True,
}


def _config(**overrides):
    """4候補(top_k=4)・top1 集中なしで探索に入る最小 config。"""
    cfg = {"enabled": True, "top_k": 4, "top1_shortcut_prob": 0.9,
           "num_determinizations": 3, "time_limit_ms": 10_000, "tie_eps": 0.02}
    cfg.update(overrides)
    return cfg


def _context(obs, fake, evaluator, config):
    return {
        "observation": obs,
        "config": config,
        # softmax top1 ≒ 0.64 < 0.9 → shortcut しない。ranked=[0,1,2,3]。
        "policy_model": _StubPolicy([4.0, 3.0, 2.0, 1.0]),
        "leaf_evaluator": evaluator,
        "hidden_state_factory": lambda: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                         "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
    }


# ---------------------------------------------------------------------------
# 1. anytime キー無し = 従来挙動(固定 N 決定化・候補順は毎回同じ)
# ---------------------------------------------------------------------------


def test_no_anytime_key_keeps_legacy_loop(pipeline, fixtures, monkeypatch):
    """`anytime` キーが無ければ従来の固定 N 決定化ループ(回転なし・N 世界ぴったり)。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 2 else 0.0)

    got = pipeline.search(obs.current, obs.select.option,
                          _context(obs, fake, evaluator, _config(num_determinizations=3)))
    assert got == [2]
    assert fake.begins == 3
    assert fake.rounds_order() == [[0, 1, 2, 3]] * 3      # 回転しない = 従来どおり


def test_anytime_disabled_flag_keeps_legacy_loop(pipeline, fixtures, monkeypatch):
    """`anytime.enabled=false` も従来ループへ落とす(ablation で OFF に戻せること)。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 2 else 0.0)

    cfg = _config(num_determinizations=2, anytime={**_ANYTIME_ON, "enabled": False})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) == [2]
    assert fake.begins == 2
    assert fake.rounds_order() == [[0, 1, 2, 3]] * 2


# ---------------------------------------------------------------------------
# 2. ラウンド回転
# ---------------------------------------------------------------------------


def test_rounds_rotate_candidate_order(pipeline, fixtures, monkeypatch):
    """rotate_candidates=true のとき、ラウンド r の候補順は r だけ回転する。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    # 早期打切りに掛からないよう全候補同点にする。
    evaluator = _ScoreEvaluator(lambda r, i: 0.0)

    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 3})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))
    assert fake.rounds_order() == [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1]]


def test_rotation_can_be_disabled(pipeline, fixtures, monkeypatch):
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 0.0)

    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 3, "rotate_candidates": False})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))
    assert fake.rounds_order() == [[0, 1, 2, 3]] * 3


# ---------------------------------------------------------------------------
# 3. 完全ラウンドのみ集計
# ---------------------------------------------------------------------------


def test_incomplete_round_is_discarded(pipeline, fixtures, monkeypatch):
    """2ラウンド目の途中で予算切れ → そのラウンドは捨て、1ラウンド目だけで平均する。

    2ラウンド目は(捨てられなければ勝者を覆すほど)高いスコアを返すので、結果が
    1ラウンド目の勝者のままであることが「不完全ラウンドを集計していない」証拠になる。
    """
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    clock = _Clock()
    monkeypatch.setattr(pipeline, "cg_api", fake)
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=clock.perf_counter))

    def _score(r, i):
        if r == 0:
            return 1.0 if i == 3 else 0.0       # ラウンド0の勝者 = 候補3
        return 50.0 if i != 3 else 0.0          # ラウンド1(不完全)は候補3以外が圧勝

    evaluator = _ScoreEvaluator(_score, clock=clock, cost=0.1)
    # deadline=0.65s。1ラウンド=4候補×0.1s。ラウンド1は3候補目までで予算切れ。
    cfg = _config(time_limit_ms=650, anytime={**_ANYTIME_ON, "max_rounds": 8})
    got = pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))

    assert got == [3]
    order = fake.rounds_order()
    assert order[0] == [0, 1, 2, 3]
    assert order[1] == [1, 2, 3]                # 途中まで評価された(=不完全)
    assert fake.begins == 2                     # 予算切れで3ラウンド目は開始しない


def test_simulation_failure_discards_round_but_continues(pipeline, fixtures, monkeypatch):
    """予算切れ以外の理由(この world で違法)で候補が評価できないラウンドは捨て、次へ進む。"""
    obs = _obs(fixtures, "mid_game")
    # ラウンド0の候補2で ValueError → ラウンド0は不完全。ラウンド1・2は完走する。
    fake = _FakeCg(me=obs.current.yourIndex, fail_at={(0, 2)})
    monkeypatch.setattr(pipeline, "cg_api", fake)

    def _score(r, i):
        if r == 0:
            return 9.0 if i == 0 else 0.0       # 捨てられるので結果に効かないはず
        return 1.0 if i == 1 else 0.0

    evaluator = _ScoreEvaluator(_score)
    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 3})
    got = pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))

    assert got == [1]                           # 完全ラウンド(1,2)のみで決まる
    assert fake.begins == 3                     # break せず最後まで回る


# ---------------------------------------------------------------------------
# 4. 完全ラウンド0 → None(呼び出し側が Policy top1 へフォールバック)
# ---------------------------------------------------------------------------


def test_no_complete_round_returns_none_by_deadline(pipeline, fixtures, monkeypatch):
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    clock = _Clock()
    monkeypatch.setattr(pipeline, "cg_api", fake)
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=clock.perf_counter))

    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 0 else 0.0, clock=clock, cost=0.1)
    # deadline=0.25s → 最初のラウンドが3候補目で切れる = 完全ラウンド0本。
    cfg = _config(time_limit_ms=250, anytime={**_ANYTIME_ON, "max_rounds": 8})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) is None


def test_no_complete_round_returns_none_by_failures(pipeline, fixtures, monkeypatch):
    """毎ラウンド違法候補が出る(=完全ラウンドが作れない)なら None。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex, fail_at={(r, 1) for r in range(4)})
    monkeypatch.setattr(pipeline, "cg_api", fake)

    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 0 else 0.0)
    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 3})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) is None
    assert fake.begins == 3


def test_none_hidden_state_round_is_skipped(pipeline, fixtures, monkeypatch):
    """factory が None(決定化に失敗)を返したラウンドは飛ばして次へ進む。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 2 else 0.0)

    calls = {"n": 0}

    def _factory():
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return {"your_deck": [], "your_prize": [], "opponent_deck": [],
                "opponent_prize": [], "opponent_hand": [], "opponent_active": []}

    ctx = _context(obs, fake, evaluator, _config(anytime={**_ANYTIME_ON, "max_rounds": 3}))
    ctx["hidden_state_factory"] = _factory
    assert pipeline.search(obs.current, obs.select.option, ctx) == [2]
    assert calls["n"] == 3
    assert fake.begins == 2


# ---------------------------------------------------------------------------
# 5. 早期打切り
# ---------------------------------------------------------------------------


def test_early_stop_when_leader_is_stable_and_margin_clear(pipeline, fixtures, monkeypatch):
    """完全3ラウンドで1位不変・差 > margin なら打ち切る(max_rounds まで回さない)。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 1 else 0.0)

    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 16, "min_rounds_for_early_stop": 3,
                           "early_stop_margin": 0.04})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) == [1]
    assert fake.begins == 3


def test_no_early_stop_when_margin_is_small(pipeline, fixtures, monkeypatch):
    """1位不変でも差が margin 以下なら打ち切らない(max_rounds まで回す)。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 0.02 if i == 1 else 0.0)   # 差 0.02 <= 0.04

    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 5, "min_rounds_for_early_stop": 3,
                           "early_stop_margin": 0.04})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))
    assert fake.begins == 5


def test_no_early_stop_before_min_rounds(pipeline, fixtures, monkeypatch):
    """差が大きくても min_rounds_for_early_stop 未満では打ち切らない。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 1 else 0.0)

    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 8, "min_rounds_for_early_stop": 5,
                           "early_stop_margin": 0.04})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))
    assert fake.begins == 5


def test_no_early_stop_when_leader_changes(pipeline, fixtures, monkeypatch):
    """直近2ラウンドで1位が入れ替わっている間は打ち切らない。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)

    def _score(r, i):
        # ラウンドごとに勝者を入れ替える(累積平均の1位も交互に振れる)。
        winner = 0 if r % 2 == 0 else 1
        return (2.0 if r % 2 == 0 else 3.0) if i == winner else 0.0

    evaluator = _ScoreEvaluator(_score)
    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 4, "min_rounds_for_early_stop": 2,
                           "early_stop_margin": 0.04})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))
    assert fake.begins == 4


def test_should_early_stop_unit(pipeline):
    """判定関数そのものの境界(単体)。"""
    means = {0: 1.0, 1: 0.9, 2: 0.0}
    # 完全ラウンド数不足
    assert pipeline._should_early_stop(2, [0, 0], means, 3, 0.04) is False
    # 1位が直近2ラウンドで不一致
    assert pipeline._should_early_stop(3, [1, 0], means, 3, 0.04) is False
    # 差 0.1 > 0.04 → 打切り
    assert pipeline._should_early_stop(3, [0, 0], means, 3, 0.04) is True
    # 差ちょうど margin(= 超えていない)→ 続行(浮動小数の誤差を避けて 0.5 で見る)
    assert pipeline._should_early_stop(3, [0, 0], {0: 1.0, 1: 0.5}, 3, 0.5) is False


# ---------------------------------------------------------------------------
# config: abl_5_full_anytime は abl_5_full の差分のみ
# ---------------------------------------------------------------------------


def test_anytime_config_differs_from_abl_5_full_only_in_new_keys():
    """`abl_5_full_anytime` は本番 `abl_5_full` に anytime / 予算v2 キーを足しただけ。"""
    cfgdir = SAMPLE_SUBMISSION_ROOT / "configs"
    base = json.loads((cfgdir / "abl_5_full.json").read_text(encoding="utf-8"))
    cand = json.loads((cfgdir / "abl_5_full_anytime.json").read_text(encoding="utf-8"))
    base.pop("name", None)
    cand.pop("name", None)

    # min_rounds_for_early_stop = max_rounds = 16: 初回 H2H では早期打切りを実質無効化する
    # (累積平均同士の一致は「同じ world 集合を共有した推定」なので統計的根拠にならない。
    #  打切り基準の妥当性は、まず打切り無しの分布を測ってから決める)。
    assert cand["pipeline"].pop("anytime") == {
        "enabled": True, "max_rounds": 16, "min_rounds_for_early_stop": 16,
        "early_stop_margin": 0.04, "rotate_candidates": True,
    }
    budget = cand["pipeline"]["time_budget"]
    assert budget.pop("mode") == "expensive_roots"
    assert budget.pop("assumed_expensive_roots") == 60
    assert budget.pop("soft_stop_ms") == 500000
    assert budget.pop("overshoot_breaker_ms") == 250
    assert budget["max_ms"] == 2000            # 第一弾は cap 据え置き
    assert cand == base
    # 本番 config 側には新キーが混入していないこと(既定 OFF の保証)。
    assert "anytime" not in base["pipeline"]
    assert "mode" not in base["pipeline"]["time_budget"]


def test_budgetv2_config_is_abl_5_full_plus_budget_keys_only():
    """`abl_5_full_budgetv2` は「予算v2だけ ON」= anytime 無し(効果の切り分け用)。"""
    cfgdir = SAMPLE_SUBMISSION_ROOT / "configs"
    base = json.loads((cfgdir / "abl_5_full.json").read_text(encoding="utf-8"))
    cand = json.loads((cfgdir / "abl_5_full_budgetv2.json").read_text(encoding="utf-8"))
    base.pop("name", None)
    cand.pop("name", None)

    assert "anytime" not in cand["pipeline"]        # スケジューリングは触らない
    budget = cand["pipeline"]["time_budget"]
    assert budget.pop("mode") == "expensive_roots"
    assert budget.pop("assumed_expensive_roots") == 60
    assert budget.pop("soft_stop_ms") == 500000
    assert budget.pop("overshoot_breaker_ms") == 250
    assert cand == base


# ---------------------------------------------------------------------------
# 6. 時間予算v2(ml_policy_agent._dynamic_pipeline_time_limit_ms)
# ---------------------------------------------------------------------------


_BUDGET_V1 = {"total_ms": 540000, "assumed_total_selects": 400, "min_ms": 50, "max_ms": 2000}
_BUDGET_V2 = {**_BUDGET_V1, "mode": "expensive_roots", "assumed_expensive_roots": 60,
              "soft_stop_ms": 500000, "overshoot_breaker_ms": 250}


def _set_elapsed(mod, monkeypatch, elapsed_ms: float, selects=0, roots=0):
    monkeypatch.setattr(mod, "_match_start_perf", time.perf_counter() - elapsed_ms / 1000.0)
    monkeypatch.setattr(mod, "_selects_seen", selects)
    monkeypatch.setattr(mod, "_expensive_roots_seen", roots)


def test_budget_without_mode_is_unchanged(ml_policy_agent, monkeypatch):
    """mode キー無し = 従来式(残り時間 ÷ 残り全select数)。"""
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V1)})
    assert got == pytest.approx(540000 / 400, rel=0.02)     # 1350ms

    # 消化済み select 分だけ1手あたりが増える。
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0, selects=200)
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V1)})
    assert got == pytest.approx(2000)                       # 540000/200=2700 → max_ms へクランプ


def test_budget_without_time_budget_key_returns_fixed(ml_policy_agent, monkeypatch):
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    assert ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_limit_ms": 400}) == 400


def test_budget_v2_divides_by_expensive_roots(ml_policy_agent, monkeypatch):
    """mode=expensive_roots は「(soft_stop - 経過) ÷ 残り探索回数」。"""
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    budget = {**_BUDGET_V2, "max_ms": 100000}               # クランプを外して式そのものを見る
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": budget})
    assert got == pytest.approx(500000 / 60, rel=0.02)      # ≒8333ms

    # 実行済み探索が増えると分母が減る。
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0, roots=30)
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": budget})
    assert got == pytest.approx(500000 / 30, rel=0.02)


def test_budget_v2_is_clamped_to_max_ms(ml_policy_agent, monkeypatch):
    """第一弾は max_ms=2000 据え置き = 1手上限は 2.0 秒。"""
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V2)})
    assert got == 2000


def test_budget_v2_boundaries(ml_policy_agent, monkeypatch):
    """境界: soft_stop 到達 / hard line(total_ms)到達 / 想定回数超過。"""
    # soft_stop を超えたら min_ms(=これ以上は探索に時間を割かない)。
    _set_elapsed(ml_policy_agent, monkeypatch, 500_500.0)
    assert ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V2)}) == 50

    # ハードライン(total_ms)超過も従来どおり min_ms。
    _set_elapsed(ml_policy_agent, monkeypatch, 540_500.0)
    assert ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V2)}) == 50

    # 想定回数を使い切っても分母は 1 でクランプされ、正の値を返す。
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0, roots=999)
    assert ml_policy_agent._dynamic_pipeline_time_limit_ms({"time_budget": dict(_BUDGET_V2)}) == 2000


def test_budget_v2_without_match_start_returns_base(ml_policy_agent, monkeypatch):
    """試合開始が未初期化なら固定値(従来と同じ安全側)。"""
    monkeypatch.setattr(ml_policy_agent, "_match_start_perf", None)
    got = ml_policy_agent._dynamic_pipeline_time_limit_ms(
        {"time_limit_ms": 400, "time_budget": dict(_BUDGET_V2)})
    assert got == 400


# ---------------------------------------------------------------------------
# 7. soft stop / circuit breaker(_try_pipeline)
# ---------------------------------------------------------------------------


def _pipeline_config(**budget_overrides):
    return {
        "lethal_search": {"enabled": False},
        "pipeline": {
            "enabled": True, "hidden_state_source": "dummy",
            "time_budget": {**_BUDGET_V2, **budget_overrides},
        },
    }


def test_expensive_roots_counter_increments_via_callback(ml_policy_agent, fixtures, monkeypatch):
    """カウンタは `context["on_expensive_root"]` を pipeline 側が呼んだときだけ進む。"""
    obs = _obs(fixtures, "mid_game")

    def _stub(state, options, context):
        context["on_expensive_root"]()       # 決定化評価へ入った = 重い探索
        return [1]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _stub)
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)

    assert ml_policy_agent._try_pipeline(obs, config=_pipeline_config()) == [1]
    assert ml_policy_agent._try_pipeline(obs, config=_pipeline_config()) == [1]
    assert ml_policy_agent._expensive_roots_seen == 2


def test_expensive_roots_counter_not_incremented_without_callback(ml_policy_agent, fixtures,
                                                                  monkeypatch):
    """pipeline が通知しない(=安い即 return)呼び出しは分母に入らない。"""
    obs = _obs(fixtures, "mid_game")
    monkeypatch.setattr(ml_policy_agent.pipeline, "search", lambda s, o, c: None)
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)

    assert ml_policy_agent._try_pipeline(obs, config=_pipeline_config()) is None
    assert ml_policy_agent._expensive_roots_seen == 0


def test_callback_exception_does_not_break_pipeline(ml_policy_agent, fixtures, monkeypatch):
    """通知コールバックが例外を投げても探索は続行する(計測は意思決定を止めない)。"""
    obs = _obs(fixtures, "mid_game")

    def _stub(state, options, context):
        ml_policy_agent.pipeline._notify_expensive_root(context)   # 実装の握り潰し経路を通す
        return [1]

    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _stub)
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    # 壊れたコールバックを直接渡しても例外は外へ出ない。
    ml_policy_agent.pipeline._notify_expensive_root({"on_expensive_root": _boom})
    # キー自体が無ければ何もしない(既存挙動不変)。
    ml_policy_agent.pipeline._notify_expensive_root({})
    assert ml_policy_agent._try_pipeline(obs, config=_pipeline_config()) == [1]


def test_soft_stop_skips_search(ml_policy_agent, fixtures, monkeypatch):
    """経過が soft_stop_ms を超えたら pipeline.search を呼ばず None(既存経路へ)。"""
    obs = _obs(fixtures, "mid_game")
    called = {"n": 0}

    def _stub(state, options, context):
        called["n"] += 1
        return [1]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _stub)
    _set_elapsed(ml_policy_agent, monkeypatch, 500_500.0)
    assert ml_policy_agent._try_pipeline(obs, config=_pipeline_config()) is None
    assert called["n"] == 0


def test_breaker_trips_on_overshoot_and_resets_next_match(ml_policy_agent, fixtures, monkeypatch):
    """割当+overshoot を超える1手が出たら、その試合は以降 pipeline を使わない。

    新しい試合の開始(obs.select is None)でブレーカとカウンタがリセットされること
    (league の worker はプロセスを跨いで再利用されるため必須)も確認する。
    """
    obs = _obs(fixtures, "mid_game")
    called = {"n": 0}

    def _slow(state, options, context):
        called["n"] += 1
        time.sleep(0.02)                # 割当(1ms)+overshoot(0ms)を確実に超える
        return [1]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _slow)
    # 割当 = clamp(min_ms=1, max_ms=1) = 1ms。overshoot 0ms。
    config = _pipeline_config(min_ms=1, max_ms=1, overshoot_breaker_ms=0)
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    monkeypatch.setattr(ml_policy_agent, "_pipeline_breaker_tripped", False)

    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert ml_policy_agent._pipeline_breaker_tripped is True
    # 以後は探索せず None(既存経路へフォールバック)。
    assert ml_policy_agent._try_pipeline(obs, config=config) is None
    assert called["n"] == 1

    # 新しい試合の開始でリセットされ、再び探索する。
    from cg.api import to_observation_class

    ml_policy_agent.agent(to_observation_class({"current": None, "logs": [], "select": None}))
    assert ml_policy_agent._pipeline_breaker_tripped is False
    assert ml_policy_agent._expensive_roots_seen == 0
    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert called["n"] == 2


def test_breaker_does_not_apply_without_mode(ml_policy_agent, fixtures, monkeypatch):
    """mode キー無し(従来 config)ではブレーカもカウンタも動かない = 挙動不変。"""
    obs = _obs(fixtures, "mid_game")

    def _slow(state, options, context):
        time.sleep(0.02)
        return [1]

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _slow)
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    monkeypatch.setattr(ml_policy_agent, "_pipeline_breaker_tripped", False)
    config = {"lethal_search": {"enabled": False},
              "pipeline": {"enabled": True, "hidden_state_source": "dummy",
                           "time_budget": dict(_BUDGET_V1), "min_ms": 1, "max_ms": 1}}

    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert ml_policy_agent._pipeline_breaker_tripped is False
    assert ml_policy_agent._expensive_roots_seen == 0


# ---------------------------------------------------------------------------
# 8. expensive_roots の意味論(E2E: 実 pipeline.search × フェイク cg)
#
# 「重い探索(Step3 決定化評価)へ実際に入った回数」だけを数えること。適用外 select や
# top1 shortcut のような**安い即 return** を数えると v2 の分母が水増しされ、1手予算が
# 過小になる(実測 94回/試合 vs 実際の探索 29回)。
# ---------------------------------------------------------------------------


def _multi_select_obs(fixtures: dict):
    """mid_game を maxCount=2(複数選択)に書き換えた観測(pipeline の適用範囲外)。"""
    from cg.api import to_observation_class

    raw = copy.deepcopy(fixtures["mid_game"])
    raw["select"]["maxCount"] = 2
    return to_observation_class({**raw, "logs": []})


def _e2e_config(anytime: dict | None) -> dict:
    cfg = {
        "lethal_search": {"enabled": False},
        "pipeline": {
            "enabled": True, "hidden_state_source": "dummy",
            "top_k": 4, "top1_shortcut_prob": 0.9, "num_determinizations": 3,
            "tie_eps": 0.02, "time_budget": dict(_BUDGET_V2),
        },
    }
    if anytime is not None:
        cfg["pipeline"]["anytime"] = dict(anytime)
    return cfg


def _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, scores):
    """`_try_pipeline` → 実 `pipeline.search` を、フェイク cg + スタブ Policy で動かす配線。"""
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 1 else 0.0)
    monkeypatch.setattr(pipeline, "leaf_eval_module",
                        SimpleNamespace(build_evaluator=lambda cfg: evaluator))
    monkeypatch.setattr(ml_policy_agent, "_get_model", lambda config=None: _StubPolicy(scores))
    monkeypatch.setattr(ml_policy_agent, "build_dummy_search_state",
                        lambda o, deck: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                         "opponent_prize": [], "opponent_hand": [],
                                         "opponent_active": []})
    _set_elapsed(ml_policy_agent, monkeypatch, 0.0)
    monkeypatch.setattr(ml_policy_agent, "_pipeline_breaker_tripped", False)
    return fake


_E2E_PATHS = [pytest.param(None, id="legacy_loop"),
              pytest.param(_ANYTIME_ON, id="anytime")]


@pytest.mark.parametrize("anytime", _E2E_PATHS)
def test_expensive_root_not_counted_for_non_main_select(ml_policy_agent, pipeline, fixtures,
                                                        monkeypatch, anytime):
    """(a-1) 非 MAIN の select は適用外 → 探索していないので数えない。"""
    obs = _obs(fixtures, "early_active_none")
    fake = _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, [1.0, 0.0])

    assert ml_policy_agent._try_pipeline(obs, config=_e2e_config(anytime)) is None
    assert ml_policy_agent._expensive_roots_seen == 0
    assert fake.begins == 0


@pytest.mark.parametrize("anytime", _E2E_PATHS)
def test_expensive_root_not_counted_for_multi_select(ml_policy_agent, pipeline, fixtures,
                                                     monkeypatch, anytime):
    """(a-2) 複数選択(maxCount>1)も適用外 → 数えない。"""
    obs = _multi_select_obs(fixtures)
    fake = _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, [4.0, 3.0, 2.0, 1.0])

    assert ml_policy_agent._try_pipeline(obs, config=_e2e_config(anytime)) is None
    assert ml_policy_agent._expensive_roots_seen == 0
    assert fake.begins == 0


@pytest.mark.parametrize("anytime", _E2E_PATHS)
def test_expensive_root_not_counted_for_top1_shortcut(ml_policy_agent, pipeline, fixtures,
                                                      monkeypatch, anytime):
    """(b) top1 集中(>= top1_shortcut_prob)は探索せず即 return → 数えない。"""
    obs = _obs(fixtures, "mid_game")
    fake = _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, [10.0, 0.0, 0.0, 0.0])

    assert ml_policy_agent._try_pipeline(obs, config=_e2e_config(anytime)) == [0]
    assert ml_policy_agent._expensive_roots_seen == 0
    assert fake.begins == 0


@pytest.mark.parametrize("anytime", _E2E_PATHS)
def test_expensive_root_counted_once_per_search(ml_policy_agent, pipeline, fixtures,
                                                monkeypatch, anytime):
    """(c) 決定化評価へ入った探索は、世界を何本回しても **ちょうど +1**。"""
    obs = _obs(fixtures, "mid_game")
    fake = _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, [4.0, 3.0, 2.0, 1.0])
    config = _e2e_config(anytime)

    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert ml_policy_agent._expensive_roots_seen == 1
    assert fake.begins >= 1                     # 世界は複数本回っている(+1 は探索1回ぶん)
    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert ml_policy_agent._expensive_roots_seen == 2


@pytest.mark.parametrize("anytime", _E2E_PATHS)
def test_no_callback_without_expensive_mode(ml_policy_agent, pipeline, fixtures,
                                            monkeypatch, anytime):
    """mode キー無し(従来 config)ではコールバックを渡さない = 完全に従来挙動。"""
    obs = _obs(fixtures, "mid_game")
    _wire_e2e(ml_policy_agent, pipeline, monkeypatch, obs, [4.0, 3.0, 2.0, 1.0])
    seen = {}
    orig_search = pipeline.search

    def _spy(state, options, context):
        seen["has_callback"] = "on_expensive_root" in context
        return orig_search(state, options, context)

    monkeypatch.setattr(ml_policy_agent.pipeline, "search", _spy)
    config = _e2e_config(anytime)
    config["pipeline"]["time_budget"] = dict(_BUDGET_V1)         # mode 無し

    assert ml_policy_agent._try_pipeline(obs, config=config) == [1]
    assert seen["has_callback"] is False
    assert ml_policy_agent._expensive_roots_seen == 0


# ---------------------------------------------------------------------------
# 9. anytime 計装(ANYTIME_STATS)
# ---------------------------------------------------------------------------


def test_anytime_stats_records_rounds(pipeline, fixtures, monkeypatch):
    """完全ラウンド数・決定数・早期打切りを記録する(意思決定には不影響)。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 1 else 0.0)

    pipeline.reset_anytime_stats()
    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 16, "min_rounds_for_early_stop": 3})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) == [1]

    stats = pipeline.ANYTIME_STATS
    assert stats["decisions"] == 1
    assert stats["complete_rounds_total"] == 3          # 早期打切りまでの完全ラウンド
    assert stats["rounds_per_decision"] == [3]
    assert stats["early_stops"] == 1
    assert stats["deadline_discard"] == 0
    assert stats["simulation_discard"] == 0
    assert stats["zero_complete_fallback"] == 0
    pipeline.reset_anytime_stats()
    assert pipeline.ANYTIME_STATS["decisions"] == 0
    assert pipeline.ANYTIME_STATS["rounds_per_decision"] == []


def test_anytime_stats_counts_deadline_and_zero_complete(pipeline, fixtures, monkeypatch):
    """予算切れ由来の破棄 / 完全0本フォールバックを分けて数える。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    clock = _Clock()
    monkeypatch.setattr(pipeline, "cg_api", fake)
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 0 else 0.0, clock=clock, cost=0.1)

    pipeline.reset_anytime_stats()
    # deadline=0.25s → 最初のラウンドが3候補目で切れる = 完全ラウンド0本。
    cfg = _config(time_limit_ms=250, anytime={**_ANYTIME_ON, "max_rounds": 8})
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, cfg)) is None

    stats = pipeline.ANYTIME_STATS
    assert stats["decisions"] == 1
    assert stats["complete_rounds_total"] == 0
    assert stats["rounds_per_decision"] == [0]
    assert stats["deadline_discard"] == 1
    assert stats["zero_complete_fallback"] == 1
    assert stats["early_stops"] == 0


def test_anytime_stats_counts_simulation_discard(pipeline, fixtures, monkeypatch):
    """予算切れ以外(この world で違法 / 決定化失敗)の破棄は simulation_discard。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex, fail_at={(0, 2)})
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 1 else 0.0)

    pipeline.reset_anytime_stats()
    cfg = _config(anytime={**_ANYTIME_ON, "max_rounds": 3, "min_rounds_for_early_stop": 99})
    pipeline.search(obs.current, obs.select.option, _context(obs, fake, evaluator, cfg))

    stats = pipeline.ANYTIME_STATS
    assert stats["simulation_discard"] == 1        # ラウンド0だけ捨てられる
    assert stats["deadline_discard"] == 0
    assert stats["complete_rounds_total"] == 2
    assert stats["rounds_per_decision"] == [2]


def test_anytime_stats_untouched_by_legacy_loop(pipeline, fixtures, monkeypatch):
    """anytime 無効(従来ループ)では計装を一切触らない。"""
    obs = _obs(fixtures, "mid_game")
    fake = _FakeCg(me=obs.current.yourIndex)
    monkeypatch.setattr(pipeline, "cg_api", fake)
    evaluator = _ScoreEvaluator(lambda r, i: 1.0 if i == 2 else 0.0)

    pipeline.reset_anytime_stats()
    before = copy.deepcopy(pipeline.ANYTIME_STATS)
    assert pipeline.search(obs.current, obs.select.option,
                           _context(obs, fake, evaluator, _config(num_determinizations=3))) == [2]
    assert pipeline.ANYTIME_STATS == before
