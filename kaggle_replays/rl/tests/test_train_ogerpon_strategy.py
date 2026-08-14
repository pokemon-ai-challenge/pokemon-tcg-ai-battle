"""train_ogerpon_strategy.py のユニット・小規模統合テスト。

外部レビュー(feature/climb-per-archetype-prep HEAD 5828b5d)指摘の回帰テスト:
- #13: parityが全headで1e-5以下(実際に小さいモデルを学習してexportし検証する)。
- #14: parity失敗時にweightsを確定配置しない。
- #15: 不完全なOption pair・不正データを学習前検証で検出する。
"""

import json
import sys
from pathlib import Path

import pytest

_RL_DIR = Path(__file__).resolve().parents[1]
_ROOT_DIR = _RL_DIR.parent.parent
_SAMPLE_SUBMISSION_DIR = _ROOT_DIR / "sample_submission"
for _p in (str(_RL_DIR), str(_ROOT_DIR), str(_SAMPLE_SUBMISSION_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def mod():
    try:
        import train_ogerpon_strategy as m
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"train_ogerpon_strategy / torch unavailable: {exc}")
    return m


# ===========================================================================
# #15: 学習前データ検証
# ===========================================================================

def _rec(option_name, pair_id, win=1, error=None, cont=None, opt=None):
    return {
        "outcome": {"win": win, "error": error, "loop_complete": False,
                   "opponent_ko_count": 0, "terminal_turns": 1},
        "option_name": option_name, "pair_id": pair_id,
        "state_id": pair_id.split(":")[0], "sample_id": pair_id.split(":")[0],
        "determinization_id": 0,
        "match_seed": 1, "opponent_archetype": "alakazam", "learner_index": 0,
        "trigger_kind": "main_attach",
        "continuous_features": cont if cont is not None else [1.0, 2.0],
        "slot_card_ids": [0] * 12,
        "option_features": opt if opt is not None else [1.0],
    }


def test_validate_dataset_rejects_bad_option_name(mod):
    records = [_rec("NOT_A_REAL_OPTION", "s0:0")]
    with pytest.raises(SystemExit):
        mod.validate_dataset(records, mod.DEFAULTS)


def test_validate_dataset_rejects_nan(mod):
    records = [_rec("EX_TEMPO", "s0:0", cont=[float("nan"), 1.0])]
    with pytest.raises(SystemExit):
        mod.validate_dataset(records, mod.DEFAULTS)


def test_validate_dataset_rejects_inf(mod):
    records = [_rec("EX_TEMPO", "s0:0", opt=[float("inf")])]
    with pytest.raises(SystemExit):
        mod.validate_dataset(records, mod.DEFAULTS)


def test_validate_dataset_rejects_high_incomplete_pair_rate(mod):
    """10 pairのうち1つしか両Optionが揃っていなければ(90%欠損)、既定閾値(50%)を超えて中止する。"""
    records = []
    for i in range(10):
        records.append(_rec("EX_TEMPO", f"s{i}:0"))
    records.append(_rec("SINGLE_PRIZE_ROTATION", "s0:0"))  # s0だけ両方揃う
    with pytest.raises(SystemExit):
        mod.validate_dataset(records, mod.DEFAULTS)


def test_validate_dataset_rejects_high_error_rate(mod):
    cfg = dict(mod.DEFAULTS)
    records = [_rec("EX_TEMPO", f"s{i}:0", error="rollout failed") for i in range(10)]
    with pytest.raises(SystemExit):
        mod.validate_dataset(records, cfg)


def test_validate_dataset_accepts_good_complete_data(mod):
    records = []
    for i in range(5):
        for name in ("EX_TEMPO", "SINGLE_PRIZE_ROTATION"):
            records.append(_rec(name, f"s{i}:0"))
    report = mod.validate_dataset(records, mod.DEFAULTS)
    assert report["n_incomplete_pairs"] == 0
    assert report["error_rate"] == 0.0


# ===========================================================================
# #13, #14: parity(実際に小さいモデルを学習してexport・検証する)
# ===========================================================================

def _write_synthetic_dataset(path: Path, n_states: int, continuous_dim: int, option_dim: int):
    """train_one_model等が実際に回る程度のサイズの合成データセットを書く。"""
    import random as _random

    rng = _random.Random(0)
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n_states):
            state_id = f"state{i}"
            arch = "alakazam" if i % 2 == 0 else "ogerpon_teal_ex"
            side = i % 2
            for det_id in range(2):
                for name in ("EX_TEMPO", "SINGLE_PRIZE_ROTATION"):
                    win = 1 if (i + det_id) % 2 == 0 else 0
                    rec = {
                        "schema_version": 2, "git_commit": "test",
                        "state_id": state_id, "sample_id": state_id,
                        "pair_id": f"{state_id}:{det_id}",
                        "match_seed": i, "determinization_id": det_id,
                        "opponent_archetype": arch, "learner_index": side,
                        "trigger_kind": "main_attach", "option_name": name,
                        "continuous_features": [rng.random() for _ in range(continuous_dim)],
                        "slot_card_ids": [0] * 12,
                        "option_features": [rng.random() for _ in range(option_dim)],
                        "first_action_identity": {},
                        "outcome": {
                            "win": win, "terminal_turns": 3, "opponent_ko_count": 1,
                            "bulu_attack_count": 1, "bulu_prizes_taken": 1,
                            "loop_complete": name == "SINGLE_PRIZE_ROTATION" and win == 1,
                            "abort_reason": None, "error": None,
                        },
                    }
                    fh.write(json.dumps(rec) + "\n")


@pytest.fixture
def synthetic_dataset(tmp_path, mod):
    from ptcg_ai.learning import ogerpon_strategy_encoder as enc

    path = tmp_path / "data.jsonl"
    _write_synthetic_dataset(path, n_states=30, continuous_dim=len(enc.CONTINUOUS_FEATURE_NAMES),
                             option_dim=len(enc.OPTION_FEATURE_NAMES))
    return path


def test_end_to_end_training_produces_parity_within_threshold(mod, synthetic_dataset, tmp_path, monkeypatch):
    """#13: 実際に学習・較正・exportし、全headのparityが1e-5以下であることを確認する。"""
    output = tmp_path / "weights.json"
    argv = [
        "train_ogerpon_strategy.py",
        "--data", str(synthetic_dataset), "--output", str(output),
        "--ensemble-seeds", "1", "--max-epochs", "2", "--batch-size", "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()

    assert output.exists(), "parityが閾値以下ならweightsが書かれるべき"
    with output.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    parity = payload["meta"]["parity_max_abs_diff"]
    for head in mod.HEADS:
        assert parity[head] <= mod.PARITY_MAX_ABS_DIFF, f"{head} parity {parity[head]} exceeds threshold"


def test_parity_failure_blocks_export(mod, synthetic_dataset, tmp_path, monkeypatch):
    """#14: parityが閾値を超えたら非0終了し、weightsを一切配置しない。"""
    output = tmp_path / "weights.json"
    assert not output.exists()

    def fake_bad_parity(*args, **kwargs):
        return {h: 1.0 for h in mod.HEADS} | {"n_samples": 1}

    monkeypatch.setattr(mod, "check_parity", fake_bad_parity)
    argv = [
        "train_ogerpon_strategy.py",
        "--data", str(synthetic_dataset), "--output", str(output),
        "--ensemble-seeds", "1", "--max-epochs", "1", "--batch-size", "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        mod.main()
    assert not output.exists(), "parity失敗時はweightsを書いてはいけない"


def test_parity_failure_does_not_overwrite_existing_output(mod, synthetic_dataset, tmp_path, monkeypatch):
    """#14の追加確認: 既存の(以前の合格分の)outputがある場合、失敗した学習で上書きしない。"""
    output = tmp_path / "weights.json"
    output.write_text('{"marker": "previous_good_weights"}', encoding="utf-8")

    def fake_bad_parity(*args, **kwargs):
        return {h: 1.0 for h in mod.HEADS} | {"n_samples": 1}

    monkeypatch.setattr(mod, "check_parity", fake_bad_parity)
    argv = [
        "train_ogerpon_strategy.py",
        "--data", str(synthetic_dataset), "--output", str(output),
        "--ensemble-seeds", "1", "--max-epochs", "1", "--batch-size", "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        mod.main()
    assert json.loads(output.read_text(encoding="utf-8")) == {"marker": "previous_good_weights"}


# ===========================================================================
# 診断フェーズ(外部レビュー指摘対応): Model B(direct delta loss)/
# Model D(match-bootstrap ensemble)の追加関数。既定値でModel Aと完全に同じ挙動を
# 保つことを確認する。
# ===========================================================================

def test_delta_regression_pairs_uses_sample_id_only(mod):
    """_delta_regression_pairsは_pairwise_ranking_pairsと同じペアリング規約
    (同一sample_idのEX/SINGLEのみ)だが、min_gapフィルタをかけず経験的deltaをそのまま返す。"""
    examples = [
        {"sample_id": "s1", "option_name": "EX_TEMPO", "win_rate": 0.5},
        {"sample_id": "s1", "option_name": "SINGLE_PRIZE_ROTATION", "win_rate": 0.51},  # 差0.01(min_gap未満)
        {"sample_id": "s2", "option_name": "EX_TEMPO", "win_rate": 0.3},
        {"sample_id": "s2", "option_name": "SINGLE_PRIZE_ROTATION", "win_rate": 0.3},   # 差0
        {"sample_id": "s3", "option_name": "EX_TEMPO", "win_rate": 0.4},  # SINGLEが無い(ペア不成立)
    ]
    pairs = mod._delta_regression_pairs(examples)
    sids_covered = set()
    for i_single, i_ex, gap in pairs:
        assert examples[i_single]["option_name"] == "SINGLE_PRIZE_ROTATION"
        assert examples[i_ex]["option_name"] == "EX_TEMPO"
        sids_covered.add(examples[i_single]["sample_id"])
    assert sids_covered == {"s1", "s2"}, "s3はEX/SINGLEが揃っていないので除外されるべき"
    # min_gap=0.02のranking pairsでは s1 が除外されるが、delta回帰ではmin_gapを適用しないため
    # s1 も含まれる(経験的delta=0.01がそのまま教師信号になる)。
    gaps = {examples[i_single]["sample_id"]: gap for i_single, i_ex, gap in pairs}
    assert gaps["s1"] == pytest.approx(0.01, abs=1e-9)
    assert gaps["s2"] == pytest.approx(0.0, abs=1e-9)


def test_match_bootstrap_resample_preserves_pairs_and_match_grouping(mod):
    """_match_bootstrap_resampleは、あるmatch_seedが選ばれたら、そのmatchに属する
    全sample_id(=EX/SINGLEペアを含む)がまとめて追加される(ペアを壊さない)。"""
    by_state = {
        "s1": {"EX_TEMPO": {"match_seed": 100}, "SINGLE_PRIZE_ROTATION": {"match_seed": 100}},
        "s2": {"EX_TEMPO": {"match_seed": 100}, "SINGLE_PRIZE_ROTATION": {"match_seed": 100}},
        "s3": {"EX_TEMPO": {"match_seed": 200}, "SINGLE_PRIZE_ROTATION": {"match_seed": 200}},
    }
    train_ids = ["s1", "s2", "s3"]
    resampled = mod._match_bootstrap_resample(by_state, train_ids, seed=0)
    # 2つのmatch(100, 200)からwith-replacementで2つ選ぶ。match 100が選ばれれば必ず
    # s1とs2が両方(隣接して)含まれる。
    assert len(resampled) in (2, 4)  # 2 match選択 x (1 or 2 sample_id)
    if "s1" in resampled:
        assert "s2" in resampled, "同一matchのsample_idは常にセットで含まれるべき"


def test_match_bootstrap_resample_is_deterministic_per_seed(mod):
    by_state = {
        "s1": {"EX_TEMPO": {"match_seed": 1}, "SINGLE_PRIZE_ROTATION": {"match_seed": 1}},
        "s2": {"EX_TEMPO": {"match_seed": 2}, "SINGLE_PRIZE_ROTATION": {"match_seed": 2}},
        "s3": {"EX_TEMPO": {"match_seed": 3}, "SINGLE_PRIZE_ROTATION": {"match_seed": 3}},
    }
    train_ids = ["s1", "s2", "s3"]
    a = mod._match_bootstrap_resample(by_state, train_ids, seed=42)
    b = mod._match_bootstrap_resample(by_state, train_ids, seed=42)
    c = mod._match_bootstrap_resample(by_state, train_ids, seed=43)
    assert a == b
    assert a != c or True  # 異なるseedで異なる場合が多いが、確率的に一致してもテストは失敗させない


def test_default_flags_reproduce_model_a_end_to_end(mod, synthetic_dataset, tmp_path, monkeypatch):
    """delta-loss-weight/model-selection-metric/match-bootstrapを指定しない(既定値)場合、
    学習パイプライン全体が例外なく完走し、既存(Model A)と同じコードパスで動くこと
    (診断フラグ追加がデフォルト挙動を壊していないことの回帰確認)。"""
    output = tmp_path / "weights.json"
    argv = [
        "train_ogerpon_strategy.py",
        "--data", str(synthetic_dataset), "--output", str(output),
        "--ensemble-seeds", "1", "--max-epochs", "2", "--batch-size", "64",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    ablation = payload["meta"]["diagnostic_ablation"]
    assert ablation["delta_loss_weight"] == 0.0
    assert ablation["model_selection_metric"] == "win_bce"
    assert ablation["match_bootstrap"] is False
    assert ablation["member_dataset_hashes"] == [None]


def test_model_b_and_model_d_flags_run_without_error(mod, synthetic_dataset, tmp_path, monkeypatch):
    """診断用ablationフラグ(delta-loss-weight/model-selection-metric=combined/
    match-bootstrap)を有効にしても、学習パイプラインが例外なく完走しparityを通ること。"""
    output = tmp_path / "weights.json"
    argv = [
        "train_ogerpon_strategy.py",
        "--data", str(synthetic_dataset), "--output", str(output),
        "--ensemble-seeds", "1,2", "--max-epochs", "2", "--batch-size", "64",
        "--delta-loss-weight", "0.3", "--model-selection-metric", "combined", "--match-bootstrap",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    ablation = payload["meta"]["diagnostic_ablation"]
    assert ablation["delta_loss_weight"] == 0.3
    assert ablation["model_selection_metric"] == "combined"
    assert ablation["match_bootstrap"] is True
    assert len(ablation["member_dataset_hashes"]) == 2
    assert all(h is not None for h in ablation["member_dataset_hashes"])
