"""masked_kl_loss(train_distill_t1.py)の数式・reductionの単体テスト(D2.1 item3)。
cgエンジン不要、純粋にtorchの計算のみ。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@pytest.fixture(scope="module")
def torch():
    try:
        import torch as _torch
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch unavailable: {exc}")
    return _torch


@pytest.fixture(scope="module")
def td(torch):
    import train_distill_t1 as _td
    return _td


def test_t_squared_scaling_applied(td, torch):
    """T=2でのlossは、T=1相当の値に対してT^2=4倍のスケーリングを含む
    (Hinton et al. 2015の慣行)。手計算で直接確認する。"""
    torch.manual_seed(0)
    scores = torch.randn(4, 5)
    teacher_logits = torch.randn(4, 5)
    mask = torch.ones(4, 5, dtype=torch.bool)

    # T=1でのKL(手計算、T^2=1なのでmasked_kl_lossの結果と一致するはず)
    tp1 = torch.softmax(teacher_logits, dim=1)
    lp1 = torch.log_softmax(scores, dim=1)
    manual_t1 = (tp1 * (torch.log(tp1.clamp_min(1e-12)) - lp1)).sum(dim=1).mean()
    assert torch.allclose(td.masked_kl_loss(scores, teacher_logits, mask, 1.0), manual_t1, atol=1e-5)

    # T=2: KL(softmax(teacher/2)||softmax(scores/2)) を計算し、手動で4倍する
    tp2 = torch.softmax(teacher_logits / 2.0, dim=1)
    lp2 = torch.log_softmax(scores / 2.0, dim=1)
    manual_t2_raw = (tp2 * (torch.log(tp2.clamp_min(1e-12)) - lp2)).sum(dim=1).mean()
    manual_t2_scaled = manual_t2_raw * (2.0 ** 2)
    assert torch.allclose(td.masked_kl_loss(scores, teacher_logits, mask, 2.0), manual_t2_scaled, atol=1e-5)


def test_reduction_weighting_independent_of_option_count(td, torch):
    """決定点ごとの合法手数(paddingを除いた実数)が違っても、各決定点はバッチ平均に
    等しい重みで寄与する(多い選択肢を持つ決定点がlossを支配しない)。"""
    torch.manual_seed(1)
    # 決定点0: 2択、決定点1: 2択だが同じKLになるよう仕組む。まず「同じ内容の決定点を
    # 選択肢数だけ変えて重複させたら、バッチ平均のKLは変わらない」ことを確認する。
    scores_single = torch.tensor([[1.0, 0.0]])
    teacher_single = torch.tensor([[2.0, 0.0]])
    mask_single = torch.tensor([[True, True]])
    kl_single = td.masked_kl_loss(scores_single, teacher_single, mask_single, 1.0)

    # 同じ決定点を(padding込みで)3個ぶんバッチに複製しても、バッチ平均は変わらないはず
    # (各決定点が均等に効いていることの確認)。
    scores_batch = scores_single.repeat(3, 1)
    teacher_batch = teacher_single.repeat(3, 1)
    mask_batch = mask_single.repeat(3, 1)
    kl_batch = td.masked_kl_loss(scores_batch, teacher_batch, mask_batch, 1.0)
    assert torch.allclose(kl_single, kl_batch, atol=1e-5)

    # 決定点ごとに合法手数が違うケース: 決定点Aは2択、決定点Bは5択(3個はpadding)。
    # Aの寄与とBの寄与がバッチ平均で「決定点1個ぶん」として等しく扱われることを、
    # 「Bのpadding部分をどう変えても結果が変わらない」ことで間接的に確認する
    # (test_masked_kl_lossのpadding不変性と同じ発想)。
    scores_mixed = torch.zeros(2, 5)
    scores_mixed[0, :2] = torch.tensor([1.0, 0.0])
    scores_mixed[1, :5] = torch.tensor([0.5, -0.5, 0.1, 0.2, 0.3])
    teacher_mixed = torch.zeros(2, 5)
    teacher_mixed[0, :2] = torch.tensor([2.0, 0.0])
    teacher_mixed[1, :5] = torch.tensor([1.0, -1.0, 0.0, 0.5, 0.2])
    mask_mixed = torch.zeros(2, 5, dtype=torch.bool)
    mask_mixed[0, :2] = True
    mask_mixed[1, :5] = True
    kl_mixed = td.masked_kl_loss(scores_mixed, teacher_mixed, mask_mixed, 1.0)

    scores_mixed2 = scores_mixed.clone()
    scores_mixed2[0, 2:] = 999.0  # 決定点Aのpadding部分をでたらめに変える
    teacher_mixed2 = teacher_mixed.clone()
    teacher_mixed2[0, 2:] = -999.0
    kl_mixed2 = td.masked_kl_loss(scores_mixed2, teacher_mixed2, mask_mixed, 1.0)
    assert torch.allclose(kl_mixed, kl_mixed2, atol=1e-5)

    # 決定点Aだけを取り出したときのKLと、mixedバッチの中でのAの寄与が一致するはず
    # (=Bの合法手数(5)がAの重みを希薄化しない)。
    kl_a_alone = td.masked_kl_loss(scores_mixed[0:1, :2], teacher_mixed[0:1, :2],
                                   mask_mixed[0:1, :2], 1.0)
    kl_b_alone = td.masked_kl_loss(scores_mixed[1:2, :5], teacher_mixed[1:2, :5],
                                   mask_mixed[1:2, :5], 1.0)
    assert torch.allclose(kl_mixed, (kl_a_alone + kl_b_alone) / 2, atol=1e-5)


def test_padding_does_not_cause_nan(td, torch):
    scores = torch.randn(3, 6)
    teacher_logits = torch.full((3, 6), float("-inf"))
    teacher_logits[:, :2] = torch.randn(3, 2)
    mask = torch.zeros(3, 6, dtype=torch.bool)
    mask[:, :2] = True
    loss = td.masked_kl_loss(scores, teacher_logits, mask, 1.0)
    assert torch.isfinite(loss)


def test_deployment_metrics_use_t1_regardless_of_input_scale(td, torch):
    """compute_distill_metricsは常にteacher_logitsをそのまま(T=1)で使う
    (呼び出し側が学習温度で割ったteacher_logitsを渡していないことの確認)。"""
    torch.manual_seed(2)
    scores = torch.randn(5, 4)
    teacher_logits = torch.randn(5, 4) * 3  # 大きめのスケール
    mask = torch.ones(5, 4, dtype=torch.bool)
    m = td.compute_distill_metrics(scores, teacher_logits, mask)
    # T=1で計算した場合と一致するはず(関数自体がT=1固定であることの直接確認)。
    expected_probs = torch.softmax(teacher_logits, dim=1)
    expected_maxprob = expected_probs.max(dim=1).values
    assert abs(m["high_conf_fraction"] - (expected_maxprob >= 0.8).float().mean().item()) < 1e-6
