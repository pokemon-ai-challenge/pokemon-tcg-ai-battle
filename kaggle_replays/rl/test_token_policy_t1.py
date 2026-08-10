"""T1(BoardTokenEmbedding/BoardTransformer/T1OptionPolicy)の単体テスト。

torchが無い環境ではスキップする(kaggle_replays/rl は元々torch依存、
worker/collect_parallel側はtorch非依存だがT1はlearner側なのでtorch前提でよい)。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


@pytest.fixture(scope="module")
def t1(torch):
    import token_policy_t1
    return token_policy_t1


GLOBAL_DIM = 34
OPTION_DIM = 65
BOARD_VOCAB = 1269  # card_vocab.pyの実測size(1267カード+PAD+UNK)
NUM_ZONES = 9


def _make_model(t1, torch, seed=0, **kwargs):
    torch.manual_seed(seed)
    return t1.T1OptionPolicy(GLOBAL_DIM, OPTION_DIM, BOARD_VOCAB, NUM_ZONES, **kwargs)


def _random_batch(torch, batch_size=3, n_board=4, n_opt=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    board_numeric = torch.randn(batch_size, n_board, 11, generator=g)
    board_card_idx = torch.randint(2, BOARD_VOCAB, (batch_size, n_board), generator=g)
    board_zone_ids = torch.randint(0, 4, (batch_size, n_board), generator=g)
    board_owner_ids = (board_zone_ids >= 2).long()
    board_mask = torch.ones(batch_size, n_board, dtype=torch.bool)
    global_features = torch.randn(batch_size, GLOBAL_DIM, generator=g)
    option_features = torch.randn(batch_size, n_opt, OPTION_DIM, generator=g)
    option_card_idx = torch.randint(0, BOARD_VOCAB, (batch_size, n_opt), generator=g)
    option_mask = torch.ones(batch_size, n_opt, dtype=torch.bool)
    return dict(board_numeric=board_numeric, board_card_idx=board_card_idx,
               board_zone_ids=board_zone_ids, board_owner_ids=board_owner_ids,
               board_mask=board_mask, global_features=global_features,
               option_features=option_features, option_card_idx=option_card_idx,
               option_mask=option_mask)


def test_total_params_computed(t1, torch):
    """実装後にコードで計算した総パラメータ数と一致する。
    [critic追加で更新] value head(value_fc1/value_fc2、d_model+global_dim -> 32 -> 1)を
    追加したぶん、policy-onlyの旧値(base166: 198,537 / fuudin_v4: 202,089)から増えている
    (base166: +3,201=201,738 / fuudin_v4: +6,753=208,842)。"""
    m34 = t1.T1OptionPolicy(34, OPTION_DIM, BOARD_VOCAB, NUM_ZONES)
    m145 = t1.T1OptionPolicy(145, OPTION_DIM, BOARD_VOCAB, NUM_ZONES)
    assert m34.total_params() == 201_738
    assert m145.total_params() == 208_842


def test_no_nan_or_inf_in_unmasked_output(t1, torch):
    model = _make_model(t1, torch)
    batch = _random_batch(torch)
    model.eval()
    with torch.no_grad():
        scores = model(**batch)
    assert torch.isfinite(scores).all()


def test_padding_does_not_affect_output(t1, torch):
    """paddingされた盤面トークン・選択肢の中身を変えても、実データの出力は変わらない。"""
    model = _make_model(t1, torch)
    model.eval()
    batch = _random_batch(torch, n_board=3, n_opt=4)
    # 3個の実トークン+2個のpadding分を追加(mask=False)。
    pad_board = 2
    numeric_padded = torch.cat([batch["board_numeric"],
                                torch.randn(3, pad_board, 11) * 1000], dim=1)
    card_idx_padded = torch.cat([batch["board_card_idx"],
                                 torch.randint(2, BOARD_VOCAB, (3, pad_board))], dim=1)
    zone_padded = torch.cat([batch["board_zone_ids"], torch.zeros(3, pad_board, dtype=torch.long)], dim=1)
    owner_padded = torch.cat([batch["board_owner_ids"], torch.zeros(3, pad_board, dtype=torch.long)], dim=1)
    mask_padded = torch.cat([batch["board_mask"], torch.zeros(3, pad_board, dtype=torch.bool)], dim=1)

    with torch.no_grad():
        out_a = model(numeric_padded, card_idx_padded, zone_padded, owner_padded, mask_padded,
                     batch["global_features"], batch["option_features"], batch["option_card_idx"],
                     batch["option_mask"])

    # paddingの中身をでたらめに変える(mask=Falseのままなので影響してはいけない)。
    numeric_padded2 = numeric_padded.clone()
    numeric_padded2[:, 3:] = torch.randn(3, pad_board, 11) * 9999
    card_idx_padded2 = card_idx_padded.clone()
    card_idx_padded2[:, 3:] = torch.randint(2, BOARD_VOCAB, (3, pad_board))

    with torch.no_grad():
        out_b = model(numeric_padded2, card_idx_padded2, zone_padded, owner_padded, mask_padded,
                     batch["global_features"], batch["option_features"], batch["option_card_idx"],
                     batch["option_mask"])

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_option_padding_produces_neg_inf(t1, torch):
    model = _make_model(t1, torch)
    model.eval()
    batch = _random_batch(torch, n_opt=4)
    mask = batch["option_mask"].clone()
    mask[:, 2:] = False  # 後ろ2個をpaddingにする
    with torch.no_grad():
        scores = model(**{**batch, "option_mask": mask})
    assert torch.isneginf(scores[:, 2:]).all()
    assert torch.isfinite(scores[:, :2]).all()


def test_bench_token_reorder_gives_same_output_within_tolerance(t1, torch):
    """同じ盤面のベンチトークンを並べ替えても出力が許容誤差内で一致する
    (self-attentionが位置埋め込みを持たないため、置換同変になるはず)。"""
    model = _make_model(t1, torch)
    model.eval()
    batch = _random_batch(torch, batch_size=1, n_board=4, n_opt=3)
    with torch.no_grad():
        out_a = model(**batch)

    perm = torch.tensor([2, 0, 3, 1])
    batch_perm = dict(batch)
    batch_perm["board_numeric"] = batch["board_numeric"][:, perm]
    batch_perm["board_card_idx"] = batch["board_card_idx"][:, perm]
    batch_perm["board_zone_ids"] = batch["board_zone_ids"][:, perm]
    batch_perm["board_owner_ids"] = batch["board_owner_ids"][:, perm]
    batch_perm["board_mask"] = batch["board_mask"][:, perm]
    with torch.no_grad():
        out_b = model(**batch_perm)

    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_zero_board_tokens_uses_cls_only(t1, torch):
    model = _make_model(t1, torch)
    model.eval()
    batch = _random_batch(torch, n_board=0, n_opt=3)
    with torch.no_grad():
        scores = model(**batch)
    assert torch.isfinite(scores).all()
    assert scores.shape == (3, 3)


def test_backward_succeeds(t1, torch):
    model = _make_model(t1, torch)
    batch = _random_batch(torch)
    scores = model(**batch)
    loss = scores.masked_select(batch["option_mask"]).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_overfits_tiny_dataset(t1, torch):
    """固定小データセット(3決定点)にhard-label CEで過学習できる(学習経路が機能する確認)。"""
    torch.manual_seed(0)
    model = _make_model(t1, torch, seed=1)
    batch = _random_batch(torch, batch_size=3, n_board=3, n_opt=4, seed=2)
    target = torch.tensor([0, 2, 1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        scores = model(**batch)
        loss = torch.nn.functional.cross_entropy(scores, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.05, f"過学習できていない: {losses[0]:.4f} -> {losses[-1]:.4f}"
    with torch.no_grad():
        pred = model(**batch).argmax(dim=1)
    assert (pred == target).all()


def test_distillation_loss_decreases(t1, torch):
    """教師logits(固定・ダミー)に対するKL蒸留損失が学習で下がる。"""
    torch.manual_seed(0)
    model = _make_model(t1, torch, seed=3)
    batch = _random_batch(torch, batch_size=4, n_board=3, n_opt=5, seed=4)
    teacher_logits = torch.randn(4, 5) * 2
    teacher_logits[~batch["option_mask"]] = float("-inf")
    teacher_probs = torch.softmax(teacher_logits, dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def kl_loss():
        scores = model(**batch)
        log_probs = torch.log_softmax(scores, dim=1)
        return torch.nn.functional.kl_div(log_probs, teacher_probs, reduction="batchmean")

    first = kl_loss().item()
    for _ in range(150):
        opt.zero_grad()
        loss = kl_loss()
        loss.backward()
        opt.step()
    last = kl_loss().item()
    assert last < first * 0.5, f"蒸留lossが下がっていない: {first:.4f} -> {last:.4f}"


def test_checkpoint_round_trip(t1, torch):
    model = _make_model(t1, torch, seed=5)
    batch = _random_batch(torch, seed=6)
    model.eval()
    with torch.no_grad():
        out_before = model(**batch)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t1.pt"
        torch.save(model.state_dict(), path)

        reloaded = _make_model(t1, torch, seed=999)  # 違うseedで初期化(値が違うことを保証)
        with torch.no_grad():
            out_diff = reloaded(**batch)
        assert not torch.allclose(out_before, out_diff, atol=1e-3)

        reloaded.load_state_dict(torch.load(path, weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            out_after = reloaded(**batch)
        assert torch.allclose(out_before, out_after, atol=1e-6)


def test_use_board_card_id_false_ignores_card_identity(t1, torch):
    """use_board_card_id=Falseならboard_card_idxの中身を変えても出力が変わらない
    (アブレーション用フラグが機能している確認)。"""
    model = _make_model(t1, torch, use_board_card_id=False)
    model.eval()
    batch = _random_batch(torch)
    with torch.no_grad():
        out_a = model(**batch)
    batch2 = dict(batch)
    batch2["board_card_idx"] = torch.randint(2, BOARD_VOCAB, batch["board_card_idx"].shape)
    with torch.no_grad():
        out_b = model(**batch2)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def _default_model_config():
    return dict(global_dim=GLOBAL_DIM, option_dim=OPTION_DIM, board_vocab_size=BOARD_VOCAB,
               num_zones=NUM_ZONES)


_CURRENT_ENV = dict(
    token_schema_version="ptcg-rl-token-shard/2", vocabulary_version=1,
    vocabulary_hash="abc", card_vocab_size=BOARD_VOCAB, feature_profile=None,
    global_feature_manifest_hash="def",
)


def _save_default_checkpoint(t1, model, path, **overrides):
    stats = {"board_numeric_mean": [0.0] * 11, "board_numeric_std": [1.0] * 11,
            "global_mean": [0.0] * GLOBAL_DIM, "global_std": [1.0] * GLOBAL_DIM,
            "option_mean": [0.0] * OPTION_DIM, "option_std": [1.0] * OPTION_DIM}
    kwargs = dict(
        token_schema_version="ptcg-rl-token-shard/2", vocabulary_version=1,
        vocabulary_hash="abc", feature_profile=None, global_feature_manifest_hash="def",
        normalization_mode="student_train_stats", normalization_stats=stats,
        card_vocab_size=BOARD_VOCAB)
    kwargs.update(overrides)
    t1.save_checkpoint(model, path, _default_model_config(), **kwargs)


def test_checkpoint_contract_round_trip(t1, torch):
    model = _make_model(t1, torch, seed=7)
    batch = _random_batch(torch, seed=8)
    model.eval()
    with torch.no_grad():
        out_before = model(**batch)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        _save_default_checkpoint(t1, model, path)

        reloaded, payload = t1.load_checkpoint(path, **_CURRENT_ENV)
        reloaded.eval()
        with torch.no_grad():
            out_after = reloaded(**batch)
        assert torch.allclose(out_before, out_after, atol=1e-6)
        assert payload["use_board_card_id"] is True
        assert payload["card_vocab_size"] == BOARD_VOCAB
        assert payload["normalization_mode"] == "student_train_stats"


def test_checkpoint_mismatch_raises(t1, torch):
    """すべての必須検査項目それぞれで、単独の不一致が検出されること。"""
    model = _make_model(t1, torch, seed=9)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        _save_default_checkpoint(t1, model, path)

        for field, bad_value in [
            ("vocabulary_hash", "different_hash"),
            ("token_schema_version", "ptcg-rl-token-shard/1"),
            ("feature_profile", "fuudin_v4"),
            ("vocabulary_version", 999),
            ("card_vocab_size", BOARD_VOCAB + 1),
            ("global_feature_manifest_hash", "other_hash"),
        ]:
            env = dict(_CURRENT_ENV)
            env[field] = bad_value
            with pytest.raises(t1.CheckpointMismatchError):
                t1.load_checkpoint(path, **env)


def test_checkpoint_load_requires_all_env_args(t1, torch):
    """検査引数を1つでも省略するとTypeErrorになる(黙って安全性を回避できない)。
    T1.1補修: 旧APIは``expect_*``が省略可能で、省略すると検査自体がスキップされる
    設計だった。新APIは全項目が位置/キーワード必須引数で、省略はPython自体が拒否する。"""
    model = _make_model(t1, torch, seed=10)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        _save_default_checkpoint(t1, model, path)

        for missing in _CURRENT_ENV:
            env = {k: v for k, v in _CURRENT_ENV.items() if k != missing}
            with pytest.raises(TypeError):
                t1.load_checkpoint(path, **env)


def test_checkpoint_allow_unsafe_mismatch_requires_explicit_flag(t1, torch):
    model = _make_model(t1, torch, seed=11)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        _save_default_checkpoint(t1, model, path)

        env = dict(_CURRENT_ENV)
        env["vocabulary_hash"] = "different_hash"
        with pytest.raises(t1.CheckpointMismatchError):
            t1.load_checkpoint(path, **env)  # allow_unsafe_mismatch省略(既定False)なら拒否
        reloaded, _ = t1.load_checkpoint(path, allow_unsafe_mismatch=True, **env)  # 明示指定なら通る
        assert reloaded is not None


def test_checkpoint_missing_normalization_mode_raises(t1, torch):
    """normalization_mode/statsが無いcheckpoint(旧形式相当)はKeyErrorではなく
    CheckpointMismatchErrorとして明示的に拒否される。"""
    model = _make_model(t1, torch, seed=12)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        _save_default_checkpoint(t1, model, path)
        payload = torch.load(path, weights_only=False)
        del payload["normalization_mode"]
        torch.save(payload, path)
        with pytest.raises(t1.CheckpointMismatchError):
            t1.load_checkpoint(path, **_CURRENT_ENV)


def test_migrate_option_embedding_from_teacher(t1, torch):
    from ptcg_ai.learning import card_vocab
    vocab = card_vocab.CardVocab([10, 20, 30], version=1, vocab_hash=card_vocab._compute_hash([10, 20, 30]))
    # 教師テーブル: card_id_max=30相当、31行×2次元。index=raw card id。
    teacher_table = [[0.0, 0.0]] * 31
    teacher_table[0] = [9.0, 9.0]   # 教師の「識別なし」行
    teacher_table[10] = [1.0, 1.0]
    teacher_table[20] = [2.0, 2.0]
    teacher_table[30] = [3.0, 3.0]

    migrated = t1.migrate_option_embedding_from_teacher(teacher_table, vocab)
    assert migrated.shape == (vocab.size, 2)
    assert torch.equal(migrated[card_vocab.PAD_INDEX], torch.zeros(2))
    assert torch.equal(migrated[card_vocab.UNK_INDEX], torch.tensor([9.0, 9.0]))
    assert torch.equal(migrated[vocab.index_of(10)], torch.tensor([1.0, 1.0]))
    assert torch.equal(migrated[vocab.index_of(20)], torch.tensor([2.0, 2.0]))
    assert torch.equal(migrated[vocab.index_of(30)], torch.tensor([3.0, 3.0]))
