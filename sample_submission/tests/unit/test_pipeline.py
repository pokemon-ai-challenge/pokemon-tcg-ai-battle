"""ptcg_ai.search.pipeline のユニットテスト。

実 cg エンジンで完全な先読みを回すと重く非決定的なので、Step3〜5(決定化ループ・
末端評価・平均集約)は cg_api を軽量なフェイクに差し替えて決定的に検証する。Step2
(top-k 絞り込み・top1 集中の即返し・適用範囲ゲート)は実 obs フィクスチャで検証する。
"""

import json
import sys
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


def _base_config(**overrides):
    cfg = {"enabled": True, "num_determinizations": 2, "top_k": 4, "top1_shortcut_prob": 0.9}
    cfg.update(overrides)
    return cfg


def test_disabled_returns_none(pipeline, fixtures):
    obs = _obs(fixtures, "mid_game")
    ctx = {"observation": obs, "config": {"enabled": False}, "policy_model": _StubPolicy([1, 0, 0, 0])}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None


def test_out_of_scope_select_returns_none(pipeline, fixtures):
    """MAIN 以外の選択(early_active_none: type=8)は enabled でも None を返す。"""
    obs = _obs(fixtures, "early_active_none")
    ctx = {"observation": obs, "config": _base_config(), "policy_model": _StubPolicy([1, 0])}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None


def test_top1_shortcut_skips_search(pipeline, fixtures):
    """top1 が top1_shortcut_prob 以上に集中していれば、決定化(factory)を呼ばずに top1 を返す。"""
    obs = _obs(fixtures, "mid_game")

    def _factory_must_not_be_called():
        raise AssertionError("hidden_state_factory must not be called on the top1 shortcut path")

    ctx = {
        "observation": obs,
        "config": _base_config(),
        "policy_model": _StubPolicy([10.0, 0.0, 0.0, 0.0]),  # softmax top1 ~ 1.0
        "hidden_state_factory": _factory_must_not_be_called,
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [0]


def _fake_cg(result_by_first_move, me, your_index_leaf=None):
    """search_begin/step/release/end のフェイク。search_step の最初の呼び出し(root からの
    first move)で、その選択インデックスに応じた result を持つ末端ノードを返す。
    """
    def search_begin(obs, *args, **kwargs):
        return SimpleNamespace(searchId="root", observation=obs)

    def search_step(search_id, selection):
        idx = selection[0]
        result = result_by_first_move.get(idx, -1)
        yi = your_index_leaf if your_index_leaf is not None else me
        leaf_state = SimpleNamespace(
            result=result, yourIndex=yi, players=[None, None],
        )
        leaf_obs = SimpleNamespace(current=leaf_state, select=None)
        return SimpleNamespace(searchId=f"n{idx}", observation=leaf_obs)

    def search_release(search_id):
        pass

    def search_end():
        pass

    return SimpleNamespace(
        search_begin=search_begin, search_step=search_step,
        search_release=search_release, search_end=search_end,
    )


def test_averaging_picks_highest_eval_candidate(pipeline, fixtures, monkeypatch):
    """候補1が全決定化で自分の勝ち(1.0)、他が相手勝ち(0.0)なら候補1を選ぶ。

    末端評価は handcrafted(result==me→1.0, result==other→0.0)。search_step が即決着
    ノードを返すので rollout は1手で終わる。
    """
    obs = _obs(fixtures, "mid_game")
    me = obs.current.yourIndex
    opp = 1 - me
    # first-move index 1 -> 自分の勝ち、他 -> 相手の勝ち。
    fake = _fake_cg({0: opp, 1: me, 2: opp, 3: opp}, me=me)
    monkeypatch.setattr(pipeline, "cg_api", fake)

    ctx = {
        "observation": obs,
        "config": _base_config(num_determinizations=3),
        "policy_model": _StubPolicy([1.0, 0.9, 0.8, 0.1]),  # 集中していない(top1<0.9)
        "hidden_state_factory": lambda: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                          "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [1]


def test_tie_break_prefers_higher_policy(pipeline, fixtures, monkeypatch):
    """全候補が同じ末端評価(引き分け近傍)なら Policy スコア上位を採用する。"""
    obs = _obs(fixtures, "mid_game")
    me = obs.current.yourIndex
    # どの first move も result 未決(-1)→ leaf は select=None の中立盤面 → 全候補同スコア。
    fake = _fake_cg({}, me=me)
    monkeypatch.setattr(pipeline, "cg_api", fake)

    ctx = {
        "observation": obs,
        "config": _base_config(num_determinizations=2, tie_eps=1.0),
        "policy_model": _StubPolicy([0.1, 5.0, 0.2, 0.3]),  # index1 が最高
        "hidden_state_factory": lambda: {"your_deck": [], "your_prize": [], "opponent_deck": [],
                                          "opponent_prize": [], "opponent_hand": [], "opponent_active": []},
    }
    assert pipeline.search(obs.current, obs.select.option, ctx) == [1]


def test_missing_policy_model_returns_none(pipeline, fixtures):
    obs = _obs(fixtures, "mid_game")
    ctx = {"observation": obs, "config": _base_config()}
    assert pipeline.search(obs.current, obs.select.option, ctx) is None


def test_extra_candidate_types_included_regardless_of_policy_rank(pipeline):
    """extra_candidate_types の手(ABILITY/ATTACH)は Policy 低ランクでも候補に入る。

    模倣が低評価する特性使用・エネ付与(例: リッチエネルギー付与→+4ドロー、
    にげあしドロー特性)を探索対象から外さないための仕組みの検証。
    """
    from cg.api import OptionType

    def _opt(t):
        return SimpleNamespace(type=t)

    # options: PLAY(0), ATTACH(1), ABILITY(2), PLAY(3), PLAY(4)
    select = SimpleNamespace(option=[
        _opt(OptionType.PLAY), _opt(OptionType.ATTACH), _opt(OptionType.ABILITY),
        _opt(OptionType.PLAY), _opt(OptionType.PLAY),
    ])
    # policy スコア順(降順): PLAYたちが上位、ATTACH/ABILITY は最下位。
    ranked = [0, 3, 4, 1, 2]

    # extra 無し: top_k=2 のみ。
    base = pipeline._select_candidate_indices(select, ranked, {"top_k": 2, "extra_candidate_types": []})
    assert set(base) == {0, 3}

    # extra 有り: top-2 に加え ATTACH(1)/ABILITY(2) も低ランクだが含まれる。
    got = pipeline._select_candidate_indices(
        select, ranked, {"top_k": 2, "extra_candidate_types": ["ABILITY", "ATTACH"], "max_candidates": 8}
    )
    assert {0, 3, 1, 2} <= set(got)

    # max_candidates で頭打ち。
    capped = pipeline._select_candidate_indices(
        select, ranked, {"top_k": 2, "extra_candidate_types": ["ABILITY", "ATTACH"], "max_candidates": 3}
    )
    assert len(capped) == 3


def test_extra_candidate_types_default_off_matches_top_k(pipeline, fixtures):
    """既定(extra_candidate_types 未指定)では従来どおり top-k のみ(挙動不変)。"""
    obs = _obs(fixtures, "mid_game")
    n = len(obs.select.option)
    ranked = list(range(n))
    idx = pipeline._select_candidate_indices(obs.select, ranked, {"top_k": 2})
    assert idx == ranked[:2]


# --- dynamic top-k(確信度で候補数を変える。既定OFF=固定 top_k のまま) ---


def _ranked_probs(top1_prob, n=20):
    """top1 が ``top1_prob``、残りを均等に割った確率分布と降順ランク。"""
    rest = (1.0 - top1_prob) / (n - 1)
    probs = [top1_prob] + [rest] * (n - 1)
    ranked = list(range(n))
    return ranked, probs


def test_resolve_top_k_defaults_to_fixed(pipeline):
    """dynamic_top_k キーが無ければ従来どおり固定 top_k(本番 abl_5_full は不変)。"""
    ranked, probs = _ranked_probs(0.4)
    assert pipeline._resolve_top_k({"top_k": 4}, probs, ranked) == 4
    # enabled=False も同様。
    assert pipeline._resolve_top_k(
        {"top_k": 4, "dynamic_top_k": {"enabled": False, "mode": "confidence",
                                       "min_k": 1, "max_k": 9}},
        probs, ranked) == 4


def test_resolve_top_k_narrows_when_confident(pipeline):
    cfg = {"top_k": 4, "dynamic_top_k": {
        "enabled": True, "mode": "confidence", "min_k": 3, "max_k": 12,
        "confident_prob": 0.6, "uncertain_prob": 0.3}}
    ranked, probs = _ranked_probs(0.85)     # 自明手
    assert pipeline._resolve_top_k(cfg, probs, ranked) == 3


def test_resolve_top_k_widens_when_uncertain(pipeline):
    cfg = {"top_k": 4, "dynamic_top_k": {
        "enabled": True, "mode": "confidence", "min_k": 3, "max_k": 12,
        "confident_prob": 0.6, "uncertain_prob": 0.3}}
    ranked, probs = _ranked_probs(0.12)     # 迷っている
    assert pipeline._resolve_top_k(cfg, probs, ranked) == 12


def test_resolve_top_k_is_monotonic_in_confidence(pipeline):
    """確信度が上がるほど候補数は増えない(単調非増加)。"""
    cfg = {"top_k": 4, "dynamic_top_k": {
        "enabled": True, "mode": "confidence", "min_k": 2, "max_k": 10,
        "confident_prob": 0.7, "uncertain_prob": 0.2}}
    ks = []
    for p in (0.10, 0.25, 0.40, 0.55, 0.70, 0.90):
        ranked, probs = _ranked_probs(p)
        ks.append(pipeline._resolve_top_k(cfg, probs, ranked))
    assert ks == sorted(ks, reverse=True)
    assert min(ks) >= 2 and max(ks) <= 10


def test_resolve_top_k_bad_thresholds_fall_back_to_fixed(pipeline):
    """confident<=uncertain のような設定不正は従来固定へ倒す(探索を壊さない)。"""
    cfg = {"top_k": 5, "dynamic_top_k": {
        "enabled": True, "mode": "confidence", "confident_prob": 0.2, "uncertain_prob": 0.8}}
    ranked, probs = _ranked_probs(0.5)
    assert pipeline._resolve_top_k(cfg, probs, ranked) == 5


def test_resolve_top_k_without_probs_falls_back(pipeline):
    """probs が無ければ固定 top_k へ。ただし候補数を超えないようクランプする
    (Phase5 で `1 <= k <= n_options` を契約にしたため、選択肢2件なら 2 が正)。"""
    cfg = {"top_k": 4, "dynamic_top_k": {"enabled": True, "mode": "confidence"}}
    assert pipeline._resolve_top_k(cfg, None, list(range(10))) == 4
    assert pipeline._resolve_top_k(cfg, None, [0, 1]) == 2


def test_climb_baseline_config_matches_production_abl_5_full():
    """`climb_baseline` は本番 `abl_5_full` の名前付きアンカー。name 以外は完全一致であること。

    ここがズレると「baseline と比較した」という主張自体が無効になる。
    """
    import json
    from pathlib import Path

    cfgdir = Path(__file__).resolve().parents[2] / "configs"
    prod = json.loads((cfgdir / "abl_5_full.json").read_text(encoding="utf-8"))
    base = json.loads((cfgdir / "climb_baseline.json").read_text(encoding="utf-8"))
    prod.pop("name", None)
    base.pop("name", None)
    assert base == prod


def test_leaf_a05_differs_from_baseline_only_in_leaf_eval():
    """`climb_v15_leaf_a05` の差分は leaf_eval **だけ**(top_k 等を巻き込んでいない)。

    「Valueブレンド単体の効果」を測る前提条件。
    """
    import json
    from pathlib import Path

    cfgdir = Path(__file__).resolve().parents[2] / "configs"
    base = json.loads((cfgdir / "climb_baseline.json").read_text(encoding="utf-8"))
    cand = json.loads((cfgdir / "climb_v15_leaf_a05.json").read_text(encoding="utf-8"))
    base.pop("name", None)
    cand.pop("name", None)
    assert cand["pipeline"].pop("leaf_eval") == {"kind": "blend", "alpha": 0.5}
    assert base["pipeline"].pop("leaf_eval") == {"kind": "handcrafted"}
    assert cand == base
    # dynamic_top_k を混ぜていないこと(単一要因比較の保証)。
    assert "dynamic_top_k" not in cand["pipeline"]


# --- n_options-gated dynamic top-k(Phase5 本命)---
#
# 診断(_diag_topk_miss.py n=46)で「探索最良手が top-4 外に出る率」が選択肢数に対して
# 単調(5-7:12.5% / 8-11:37.5% / 12+:50.0%)だったのを受けた設計。
# 確信度ゲートは同じ診断で予測力が無かったため、今回の A/B では使わない。

NOPTK = {"enabled": True, "mode": "n_options"}


def _k(pipeline, n_options, cfg_extra=None):
    cfg = {"top_k": 4, "dynamic_top_k": {**NOPTK, **(cfg_extra or {})}}
    ranked = list(range(n_options))
    probs = [1.0 / n_options] * n_options if n_options else []
    return pipeline._resolve_top_k(cfg, probs, ranked)


@pytest.mark.parametrize("n_options,expected", [
    (1, 1), (4, 4), (5, 4), (7, 4), (8, 8), (11, 8), (12, 12), (20, 12),
])
def test_noptions_gate_boundaries(pipeline, n_options, expected):
    """指示の境界表どおりに解決されること(top_k は候補数を超えない)。"""
    assert _k(pipeline, n_options) == expected


def test_noptions_gate_never_exceeds_n_options(pipeline):
    for n in range(1, 25):
        k = _k(pipeline, n)
        assert 1 <= k <= n


def test_noptions_gate_is_monotonic(pipeline):
    ks = [_k(pipeline, n) for n in range(1, 25)]
    assert ks == sorted(ks)


def test_noptions_thresholds_are_config_driven(pipeline):
    """しきい値・top-k をハードコードせず config から変えられること。"""
    cfg_extra = {"thresholds": [
        {"max_options": 3, "top_k": 2},
        {"max_options": None, "top_k": 6},
    ]}
    assert _k(pipeline, 3, cfg_extra) == 2
    assert _k(pipeline, 10, cfg_extra) == 6
    assert _k(pipeline, 5, cfg_extra) == 5   # top_k=6 > n_options=5 -> クランプ


def test_max_top_k_cap_is_respected(pipeline):
    assert _k(pipeline, 20, {"max_top_k": 6}) == 6


def test_off_is_exactly_fixed_top_k(pipeline):
    """OFF(既定)は固定 top_k=4 と完全一致 = 本番挙動不変。"""
    for n in range(1, 25):
        cfg = {"top_k": 4}
        ranked = list(range(n))
        probs = [1.0 / n] * n
        assert pipeline._resolve_top_k(cfg, probs, ranked) == min(4, n)
        cfg_off = {"top_k": 4, "dynamic_top_k": {"enabled": False, "mode": "n_options"}}
        assert pipeline._resolve_top_k(cfg_off, probs, ranked) == min(4, n)


def test_unknown_mode_falls_back_to_fixed(pipeline):
    assert _k(pipeline, 20, {"mode": "something_new"}) == 4


def test_zero_options_is_safe(pipeline):
    assert pipeline._resolve_top_k({"top_k": 4, "dynamic_top_k": NOPTK}, [], []) == 4


def test_confidence_mode_not_used_by_noptk_config():
    """A/B で使う config が confidence ゲートを持たないこと(取り違え防止)。"""
    import json
    from pathlib import Path

    cfgdir = Path(__file__).resolve().parents[2] / "configs"
    c = json.loads((cfgdir / "climb_v15_leaf_a05_noptk.json").read_text(encoding="utf-8"))
    dt = c["pipeline"]["dynamic_top_k"]
    assert dt["enabled"] is True
    assert dt["mode"] == "n_options"
    for key in ("confident_prob", "uncertain_prob", "min_k", "max_k"):
        assert key not in dt, f"confidence ゲートの設定 {key} が混入している"


def test_noptk_config_differs_from_a05_only_in_dynamic_top_k():
    """B は A に dynamic_top_k を足しただけ(leaf/α/det/rollout/budget は同一)。"""
    import json
    from pathlib import Path

    cfgdir = Path(__file__).resolve().parents[2] / "configs"
    a = json.loads((cfgdir / "climb_v15_leaf_a05.json").read_text(encoding="utf-8"))
    b = json.loads((cfgdir / "climb_v15_leaf_a05_noptk.json").read_text(encoding="utf-8"))
    a.pop("name", None)
    b.pop("name", None)
    assert b["pipeline"].pop("dynamic_top_k")["mode"] == "n_options"
    assert "dynamic_top_k" not in a["pipeline"]
    assert b == a
    # leaf は両方 α=0.5 のまま(αを同時に動かしていない)
    assert a["pipeline"]["leaf_eval"] == {"kind": "blend", "alpha": 0.5}
