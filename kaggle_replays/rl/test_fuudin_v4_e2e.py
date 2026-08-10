"""fuudin_v4(389次元profile)実データでのend-to-endテスト(T1.1)。

pool_v1(``kaggle_replays/rl/runs/pool_v1/models/model_v47.json``、
``extended_features_profile=fuudin_v4``)を教師に、実際のcgエンジン自己対戦で
State -> token化 -> shard保存 -> 再読込 -> token_batch -> T1OptionPolicy.forward ->
KL loss -> backward -> optimizer.step -> checkpoint保存 -> 再読込後の出力一致、を
一続きで確認する。base166の合成データでは代替しない。

train/validationは試合(game)単位で分割し、決定点単位では分けない(情報漏洩を防ぐ、
``token_batch.train_val_decision_indices``)。

正式教師checkpointの選定(v24/v32/v40/v47の比較評価)とは無関係。ここでは実装の
動作確認のためにv47をそのまま使う。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_HERE / "distributed"), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TEACHER = _ROOT / "kaggle_replays" / "rl" / "runs" / "pool_v1" / "models" / "model_v47.json"
_DECK = _ROOT / "kaggle_replays" / "meta_analysis" / "archetype_decks" / "alakazam_morioka" / "01.csv"


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


@pytest.fixture(scope="module")
def env(torch):
    if not _TEACHER.exists() or not _DECK.exists():
        pytest.skip(f"fuudin_v4教師/デッキが無い: {_TEACHER} / {_DECK}")
    try:
        import collect_tokens as ct
        import token_shard as ts
        import token_batch as tb
        import token_policy_t1 as t1
        import train_distill_t1 as td
        from ptcg_ai.learning import card_vocab
        from ptcg_ai.learning import legacy_feature_manifest as manifest
        from ptcg_ai.learning.policy_model import PolicyModel
        from run_league import read_deck_csv_file
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dependencies unavailable: {exc}")

    pm = PolicyModel(str(_TEACHER))
    if not pm.is_ready:
        pytest.skip("pool_v1 model_v47.json を読み込めない")
    profile_name = getattr(pm, "_extended_profile", None) and pm._extended_profile.name
    assert profile_name == "fuudin_v4", f"想定外のprofile: {profile_name}"

    return {"ct": ct, "ts": ts, "tb": tb, "t1": t1, "td": td, "card_vocab": card_vocab,
           "manifest": manifest, "pm": pm, "profile": profile_name,
           "deck": read_deck_csv_file(str(_DECK))}


@pytest.fixture(scope="module")
def shard_arrays(env):
    """実際にcgエンジンでfuudin_v4教師の自己対戦を8試合ぶん収集し、shardに書き出して
    読み直したもの(モジュールスコープでキャッシュし、複数テストで使い回す)。"""
    ct, ts = env["ct"], env["ts"]
    trajs, wins, valid, errors = ct.parallel_collect_tokens(
        str(_TEACHER), env["deck"], env["deck"], n_games=8, seed0=123,
        temperature=1.0, workers=2, debug_pointers=False)
    assert valid > 0, f"収集できた試合が0(errors={errors})"
    decisions = [step for tr in trajs for step in tr["steps"]]
    assert decisions, "決定点が1件も記録されなかった"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuudin_v4_e2e.npz"
        ts.write_token_shard(path, decisions, trajs, {"run_id": "fuudin_v4_e2e", "generation": 0})
        arrays, meta = ts.read_token_shard(path)
    return arrays, meta


def test_decision_count_and_dims(shard_arrays):
    """2. fuudin_v4実データの決定点数・次元の確認(報告用の実測値もここで得る)。"""
    arrays, meta = shard_arrays
    assert meta["n_decisions"] > 0
    assert meta["legacy_state_dim"] == 389
    assert meta["legacy_global_dim"] == 145
    assert arrays["legacy_global_features"].shape == (meta["n_decisions"], 145)
    print(f"\n[fuudin_v4 E2E] n_decisions={meta['n_decisions']} n_trajectories={meta['n_trajectories']}")


def _to_tensors(torch, batch):
    b = batch["board"]
    o = batch["option"]
    return dict(
        board_numeric=torch.tensor(b["numeric"], dtype=torch.float32),
        board_card_idx=torch.tensor(b["card_idx"], dtype=torch.long),
        board_zone_ids=torch.tensor(b["zone_ids"], dtype=torch.long),
        board_owner_ids=torch.tensor(b["owner_ids"], dtype=torch.long),
        board_mask=torch.tensor(b["mask"], dtype=torch.bool),
        global_features=torch.tensor(batch["legacy_global_features"], dtype=torch.float32),
        option_features=torch.tensor(o["legacy_option_features"], dtype=torch.float32),
        option_card_idx=torch.tensor(o["card_idx"], dtype=torch.long),
        option_mask=torch.tensor(o["mask"], dtype=torch.bool),
    ), torch.tensor(o["teacher_logits"], dtype=torch.float32)


def _train_and_eval(t1, td, torch, model_config, stats, mode, train_inputs, train_teacher_logits,
                    val_inputs, val_teacher_logits, steps=30, lr=1e-3, seed=0):
    """``train_distill_t1.masked_kl_loss``をそのまま使う(独自実装を重複させない。
    T1.1補修でpaddingを事前に-infで潰す修正が入っているため、そちらを使う方が安全)。"""
    torch.manual_seed(seed)
    model = t1.T1OptionPolicy(**model_config)
    with torch.no_grad():
        model.board_numeric_mean.copy_(torch.tensor(stats["board_numeric_mean"]))
        model.board_numeric_std.copy_(torch.tensor(stats["board_numeric_std"]))
        model.global_mean.copy_(torch.tensor(stats["global_mean"]))
        model.global_std.copy_(torch.tensor(stats["global_std"]))
        model.option_mean.copy_(torch.tensor(stats["option_mean"]))
        model.option_std.copy_(torch.tensor(stats["option_std"]))

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _kl(inputs, teacher_logits):
        scores = model(**inputs)
        return td.masked_kl_loss(scores, teacher_logits, inputs["option_mask"], 1.0)

    model.train()
    first = _kl(train_inputs, train_teacher_logits).item()
    for _ in range(steps):
        opt.zero_grad()
        loss = _kl(train_inputs, train_teacher_logits)
        loss.backward()
        opt.step()
    last_train = _kl(train_inputs, train_teacher_logits).item()
    model.eval()
    with torch.no_grad():
        val = _kl(val_inputs, val_teacher_logits).item()
    return model, first, last_train, val


def test_full_pipeline_state_to_checkpoint(env, shard_arrays, torch):
    """State -> token化 -> shard保存 -> 再読込 -> token_batch -> forward -> KL loss ->
    backward -> optimizer.step -> checkpoint保存 -> 再読込後の出力一致、を一続きで確認する。

    正規化方式2つ(teacher_stats / student_train_stats)を同条件で比較し、
    validation KLで良い方をcheckpointに使う(§正規化方式の比較可能化)。
    """
    arrays, meta = shard_arrays
    tb, t1, td, card_vocab, manifest = env["tb"], env["t1"], env["td"], env["card_vocab"], env["manifest"]
    profile = env["profile"]
    vocab = card_vocab.load_vocab()

    # 試合単位でtrain/val分割(情報漏洩防止)。
    train_idx, val_idx = tb.train_val_decision_indices(arrays, val_fraction=0.25, seed=0)
    assert len(train_idx) > 0 and len(val_idx) > 0

    teacher_std = json.loads(Path(_TEACHER).read_text(encoding="utf-8"))["standardization"]

    train_arrays = tb.slice_arrays_by_decisions(arrays, train_idx)
    val_arrays = tb.slice_arrays_by_decisions(arrays, val_idx)
    train_batch = tb.build_batch(train_arrays, vocab)
    val_batch = tb.build_batch(val_arrays, vocab)
    train_inputs, train_teacher_logits = _to_tensors(torch, train_batch)
    val_inputs, val_teacher_logits = _to_tensors(torch, val_batch)

    model_config = dict(global_dim=manifest.global_dim(profile), option_dim=65,
                        board_vocab_size=vocab.size, num_zones=9)

    results = {}
    for mode in tb.NORMALIZATION_MODES:
        stats = tb.build_normalization_stats(mode, arrays, train_idx, profile,
                                             teacher_standardization=teacher_std)
        model, first, last_train, val = _train_and_eval(
            t1, td, torch, model_config, stats, mode, train_inputs, train_teacher_logits,
            val_inputs, val_teacher_logits)
        results[mode] = {"model": model, "stats": stats, "first_train": first,
                         "last_train": last_train, "val": val}
        assert torch.isfinite(torch.tensor(first))
        assert torch.isfinite(torch.tensor(last_train))
        assert torch.isfinite(torch.tensor(val))
        assert last_train < first, (
            f"[{mode}] trainのKL lossが下がっていない: {first:.4f} -> {last_train:.4f}")
        print(f"\n[fuudin_v4 E2E][{mode}] train {first:.4f} -> {last_train:.4f}, val {val:.4f}")

    best_mode = min(results, key=lambda m: results[m]["val"])
    print(f"[fuudin_v4 E2E] best_mode={best_mode} "
          f"(teacher_stats val={results['teacher_stats']['val']:.4f}, "
          f"student_train_stats val={results['student_train_stats']['val']:.4f})")

    model = results[best_mode]["model"]
    stats = results[best_mode]["stats"]
    model.eval()
    with torch.no_grad():
        out_before = model(**val_inputs)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "t1_fuudin_v4.pt"
        t1.save_checkpoint(
            model, ckpt_path, model_config,
            token_schema_version=env["ts"].TOKEN_SHARD_FORMAT, vocabulary_version=vocab.version,
            vocabulary_hash=vocab.hash, feature_profile=profile,
            global_feature_manifest_hash=manifest.manifest_hash(profile),
            normalization_mode=best_mode, normalization_stats=stats, card_vocab_size=vocab.size)

        reloaded, payload = t1.load_checkpoint(
            ckpt_path, token_schema_version=env["ts"].TOKEN_SHARD_FORMAT,
            vocabulary_version=vocab.version, vocabulary_hash=vocab.hash,
            card_vocab_size=vocab.size, feature_profile=profile,
            global_feature_manifest_hash=manifest.manifest_hash(profile))
        reloaded.eval()
        with torch.no_grad():
            out_after = reloaded(**val_inputs)
        assert torch.allclose(out_before, out_after, atol=1e-6)
        assert payload["feature_profile"] == "fuudin_v4"
        assert payload["normalization_mode"] == best_mode

    print(f"[fuudin_v4 E2E] checkpoint round-trip OK ({ckpt_path.name}, mode={best_mode})")
