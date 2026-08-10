"""train_distill_t1.py の単体テスト(D2/D2.1)。実際のcgエンジンで小規模データを収集し、
複数shard読み込み・dataset manifest・train/val分割の固定(split-dir共有)・学習1周・
resume・ログ・checkpoint保存を確認する。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


def _full_meta(pm, vocab, profile_name):
    from ptcg_ai.learning import legacy_feature_manifest as manifest
    import common as C
    weights = pm._weights_path if hasattr(pm, "_weights_path") else None
    return {
        "run_id": "distill_test", "generation": 0,
        "teacher_weights_path": str(weights), "teacher_model_sha256": C.sha256_file(weights),
        "extended_features_profile": profile_name,
        "vocabulary_version": vocab.version, "vocabulary_hash": vocab.hash,
        "card_vocab_size": vocab.size,
        "global_feature_manifest_hash": manifest.manifest_hash(profile_name),
        "games_requested": 3, "temperature": 1.0,
        "action_selection_mode": "softmax_temperature",
    }


@pytest.fixture(scope="module")
def shards(torch):
    """production既定の教師(base166、torch不要な軽い収集)で2本のshardを作る。"""
    try:
        import collect_tokens as ct
        import token_shard as ts
        from ptcg_ai.learning import card_vocab
        from ptcg_ai.learning.policy_model import PolicyModel
        from run_league import read_deck_csv_file
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")

    deck_path = (_ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks"
                / "alakazam" / "01.csv").resolve()
    weights = (_ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json").resolve()
    if not deck_path.exists() or not weights.exists():
        pytest.skip("デッキ/教師重みが無い")
    deck = read_deck_csv_file(str(deck_path))
    pm = PolicyModel(str(weights))
    vocab = card_vocab.load_vocab()
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name

    tmp = Path(tempfile.mkdtemp())
    paths = []
    for i, seed in enumerate((10, 20)):
        trajs, wins, valid, errors = ct.parallel_collect_tokens(
            str(weights), deck, deck, n_games=3, seed0=seed, temperature=1.0, workers=2)
        decisions = [step for tr in trajs for step in tr["steps"]]
        if not decisions:
            pytest.skip("決定点が収集できなかった")
        from ptcg_ai.learning import legacy_feature_manifest as manifest
        for d in decisions:
            d.setdefault("legacy_global_feat",
                        manifest.extract_global_features(d["legacy_state_feat"], profile_name).tolist())
        p = tmp / f"shard{i}.npz"
        meta = _full_meta(pm, vocab, profile_name)
        meta["teacher_weights_path"] = str(weights)
        ts.write_token_shard(p, decisions, trajs, meta)
        paths.append(str(p))
    return paths, str(weights)


def _run_trainer(shards, weights, out_dir, split_dir=None, extra_args=()):
    cmd = [sys.executable, str(_HERE / "train_distill_t1.py"),
          "--shards", *shards, "--teacher-weights", weights, "--out-dir", str(out_dir),
          "--steps", "6", "--eval-every", "2", "--batch-size", "8"]
    if split_dir is not None:
        cmd += ["--split-dir", str(split_dir)]
    cmd += list(extra_args)
    r = subprocess.run(cmd, cwd=str(_HERE), capture_output=True, text=True)
    return r


def test_trainer_runs_end_to_end(shards):
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_dir = tmp / "run1"
        split_dir = tmp / "splits"
        r = _run_trainer(paths, weights, out_dir, split_dir)
        assert r.returncode == 0, r.stdout + r.stderr

        assert (out_dir / "dataset_manifest.json").exists()
        assert list(split_dir.glob("split_*.json")), "split-dirにsplitファイルが無い"
        assert (out_dir / "best.pt").exists()
        assert (out_dir / "trainer_state.pt").exists()
        log_lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(log_lines) >= 1
        record = json.loads(log_lines[-1])
        for key in ("train_objective_kl", "deployment_kl_t1", "top1_agreement",
                    "top1_agreement_high_teacher_conf", "student_prob_under_teacher_mean",
                    "top1_agreement_entropy_low", "top1_agreement_entropy_mid",
                    "top1_agreement_entropy_high"):
            assert key in record, f"{key}がログに無い"


def test_split_is_fixed_across_reruns(shards):
    """同じsplit-dirに2回学習を投げても、splitの分割が変わらない。"""
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_dir = tmp / "splits"
        _run_trainer(paths, weights, tmp / "run2a", split_dir)
        split_files1 = sorted(split_dir.glob("split_*.json"))
        split1 = json.loads(split_files1[0].read_text(encoding="utf-8"))
        _run_trainer(paths, weights, tmp / "run2b", split_dir)  # 別out-dir
        split_files2 = sorted(split_dir.glob("split_*.json"))
        split2 = json.loads(split_files2[0].read_text(encoding="utf-8"))
        assert len(split_files1) == len(split_files2) == 1  # 同じsplitファイルが再利用された
        assert split1["train_game_ids"] == split2["train_game_ids"]
        assert split1["val_game_ids"] == split2["val_game_ids"]


def test_split_rejected_if_dataset_manifest_changed(shards):
    """同じsplitファイルを、違うdataset(shard集合)に対して使おうとすると拒否される。"""
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_dir = tmp / "splits"
        r1 = _run_trainer(paths, weights, tmp / "run3a", split_dir)
        assert r1.returncode == 0, r1.stdout + r1.stderr
        split_file = next(split_dir.glob("split_*.json"))
        split = json.loads(split_file.read_text(encoding="utf-8"))
        split["dataset_manifest_hash"] = "tampered_hash"
        split_file.write_text(json.dumps(split), encoding="utf-8")

        r2 = _run_trainer(paths, weights, tmp / "run3b", split_dir)
        assert r2.returncode != 0
        assert "dataset_manifest_hash" in (r2.stdout + r2.stderr)


def test_resume_continues_from_saved_step(shards):
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_dir = tmp / "run4"
        split_dir = tmp / "splits"
        r1 = _run_trainer(paths, weights, out_dir, split_dir, extra_args=["--steps", "4"])
        assert r1.returncode == 0, r1.stdout + r1.stderr
        state1 = json.loads((out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
        assert state1["step"] == 4

        r2 = subprocess.run(
            [sys.executable, str(_HERE / "train_distill_t1.py"),
             "--shards", *paths, "--teacher-weights", weights, "--out-dir", str(out_dir),
             "--split-dir", str(split_dir),
             "--steps", "8", "--eval-every", "2", "--batch-size", "8", "--resume"],
            cwd=str(_HERE), capture_output=True, text=True)
        assert r2.returncode == 0, r2.stdout + r2.stderr
        assert "resume: step=4" in r2.stdout
        lines = (out_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[-1])["step"] == 8


def test_resume_rejects_changed_hyperparameters(shards):
    """resume時、前回と違うlr等でrunを継続しようとすると明示的に拒否される。"""
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out_dir = tmp / "run5"
        split_dir = tmp / "splits"
        r1 = _run_trainer(paths, weights, out_dir, split_dir, extra_args=["--steps", "4"])
        assert r1.returncode == 0, r1.stdout + r1.stderr

        r2 = subprocess.run(
            [sys.executable, str(_HERE / "train_distill_t1.py"),
             "--shards", *paths, "--teacher-weights", weights, "--out-dir", str(out_dir),
             "--split-dir", str(split_dir),
             "--steps", "8", "--eval-every", "2", "--batch-size", "8", "--resume",
             "--lr", "9e-2"],  # 前回と違う学習率
            cwd=str(_HERE), capture_output=True, text=True)
        assert r2.returncode != 0
        assert "resume条件が前回の実行と一致しない" in (r2.stdout + r2.stderr)


def test_resume_bitwise_matches_straight_through_on_cpu(shards, torch):
    """中断なし学習(8 step)と、4 stepで中断してresumeした学習(4+4 step)が、
    最終的に完全に同じモデル重みになることを確認する(RNG state・シャッフル順の
    復元が正しいことの直接証拠)。GPU側のcuDNN非決定性を避けるため``--device cpu``。
    """
    paths, weights = shards
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        split_dir = tmp / "splits"

        straight_dir = tmp / "straight"
        r_straight = _run_trainer(paths, weights, straight_dir, split_dir,
                                  extra_args=["--steps", "8", "--device", "cpu"])
        assert r_straight.returncode == 0, r_straight.stdout + r_straight.stderr

        resumed_dir = tmp / "resumed"
        r1 = _run_trainer(paths, weights, resumed_dir, split_dir,
                          extra_args=["--steps", "4", "--device", "cpu"])
        assert r1.returncode == 0, r1.stdout + r1.stderr
        r2 = subprocess.run(
            [sys.executable, str(_HERE / "train_distill_t1.py"),
             "--shards", *paths, "--teacher-weights", weights, "--out-dir", str(resumed_dir),
             "--split-dir", str(split_dir), "--steps", "8", "--eval-every", "2",
             "--batch-size", "8", "--device", "cpu", "--resume"],
            cwd=str(_HERE), capture_output=True, text=True)
        assert r2.returncode == 0, r2.stdout + r2.stderr

        state_straight = torch.load(straight_dir / "trainer_state.pt", weights_only=False)
        state_resumed = torch.load(resumed_dir / "trainer_state.pt", weights_only=False)
        sd1 = state_straight["model_state_dict"]
        sd2 = state_resumed["model_state_dict"]
        assert sd1.keys() == sd2.keys()
        for k in sd1:
            assert torch.equal(sd1[k], sd2[k]), f"パラメータ{k}がresume前後で一致しない"
        assert state_straight["best_deployment_kl"] == state_resumed["best_deployment_kl"]


def test_mismatched_shard_metadata_rejected(shards, torch):
    """teacher_model_sha256が違うshardを混ぜようとするとShardMetaMismatchErrorになる。"""
    import token_shard as ts
    paths, _ = shards
    arrays0, meta0 = ts.read_token_shard(paths[0])
    arrays1, meta1 = ts.read_token_shard(paths[1])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        meta1_bad = dict(meta1)
        meta1_bad["teacher_model_sha256"] = "different_sha"
        import numpy as np
        p0 = tmp / "a.npz"
        p1 = tmp / "b.npz"
        np.savez_compressed(p0, meta=np.array(json.dumps(dict(meta0, teacher_model_sha256="abc"))),
                            **arrays0)
        np.savez_compressed(p1, meta=np.array(json.dumps(meta1_bad)), **arrays1)
        with pytest.raises(ts.ShardMetaMismatchError):
            ts.concatenate_shards([p0, p1])
