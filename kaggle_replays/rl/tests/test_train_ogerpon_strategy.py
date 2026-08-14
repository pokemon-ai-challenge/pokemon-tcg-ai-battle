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
