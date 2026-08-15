"""defect#5(報酬シェーピング)の検証。

train_v3.compute_gae に ``state_rows`` + ``shaping_c`` を渡すと、非終端の遷移は
potential-based shaping Φ=c*(opp_prize_remaining-self_prize_remaining) で報酬を置き換える。
ここで最も壊れやすいのは「終端の Φ」の扱いなので、そこを直接検証する:

検証1: Φ(terminal) が厳密に 0 として扱われていることを、gamma!=1 の数値例で手計算と突き合わせて
       確認する(gamma=1 だと「terminal の Φ を s_{L-1} の Φ で代用してしまう」バグが見た目上
       消えてしまい検出できないため、意図的に gamma=0.9 を使う)。
検証2: gamma=1・軌跡の始状態がサイド差0(ゲーム開始そのもの、自分残6・相手残6)のとき、
       シェーピング項の合計が厳密にゼロへ telescope し、GAE の advantage が素の終局報酬 R と
       完全に一致することを確認する(potential-based shaping の核心的性質。plan doc/タスクが
       「γ=1でシェーピング項の和がゼロへtelescopeする」と要求している具体的な形)。
検証3: 検証2 を Φ(s_0)!=0(ゲーム開始状態ではない任意の状態)に一般化し、
       adv[0] == R - Φ(s_0) という代数的恒等式そのものを検証する(検証2はこの特殊ケース c=0の
       始状態)。
検証4: shaping_c=0(既定オフ)のとき、shaping 有りのコードパスを一切通らず、
       shaping 追加前と完全に同じ値を返すこと(後方互換の回帰テスト)。
検証5: prize_margin_potential を「終端状態っぽい特徴」に対して直接呼ぶと(呼ぶべきではない、
       という文書化のため)一般には0にならないことを示す。もし compute_gae が終端に対して
       この関数を呼んでいたら検証1が失敗するはずなので、検証1・検証5は対になっている。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_v3 import (  # noqa: E402
    OPP_PRIZE_REMAINING_IDX,
    SELF_PRIZE_REMAINING_IDX,
    compute_gae,
    prize_margin_potential,
)
from ptcg_ai.learning import encoder  # noqa: E402

STATE_DIM = len(encoder.FEATURE_NAMES)
C = 0.1


def _row(self_prize, opp_prize):
    r = [0.0] * STATE_DIM
    r[SELF_PRIZE_REMAINING_IDX] = float(self_prize)
    r[OPP_PRIZE_REMAINING_IDX] = float(opp_prize)
    return r


def main():
    # ------------------------------------------------------------------
    # 検証1: gamma!=1 での手計算との突き合わせ(terminal Φ が厳密に0であることのピン留め)
    # ------------------------------------------------------------------
    print("--- 検証1: Φ(terminal)≡0 の数値検証(gamma=0.9) ---")
    gamma, lam = 0.9, 1.0
    s0 = _row(6, 6)          # ゲーム開始(サイド差0 -> Φ=0)
    s1 = _row(4, 5)          # 自分が1枚多く取られている -> Φ = c*(5-4) = 0.1
    state_rows = torch.tensor([s0, s1], dtype=torch.float32)
    values = torch.zeros(2)
    R = 1.0  # 勝ち
    adv, vt = compute_gae([2], [R], values, gamma, lam, "cpu",
                          state_rows=state_rows, shaping_c=C)

    phi0 = prize_margin_potential(s0, C)
    phi1 = prize_margin_potential(s1, C)
    assert abs(phi0 - 0.0) < 1e-9, phi0
    assert abs(phi1 - 0.1) < 1e-9, phi1

    # 正しい実装(Φ(terminal)=定数0.0)での期待値:
    #   r0 = 0 + gamma*phi1 - phi0        = 0.9*0.1 - 0     = 0.09
    #   r1 = R + gamma*Φ(terminal=0) - phi1 = 1 + 0 - 0.1    = 0.9
    #   (values=0, lam=1 なので GAE は素の割引和)
    #   adv[1] = r1                        = 0.9
    #   adv[0] = r0 + gamma*lam*adv[1]     = 0.09 + 0.9*0.9  = 0.9
    expected_adv1 = 0.9
    expected_adv0 = 0.09 + gamma * lam * expected_adv1
    print(f"  adv0={adv[0].item():.6f} (期待 {expected_adv0:.6f})  "
          f"adv1={adv[1].item():.6f} (期待 {expected_adv1:.6f})")
    assert abs(adv[0].item() - expected_adv0) < 1e-6, (adv[0].item(), expected_adv0)
    assert abs(adv[1].item() - expected_adv1) < 1e-6, (adv[1].item(), expected_adv1)

    # もし「terminal の Φ を Φ(s_{L-1}) で代用する」バグがあったら r1 = R + gamma*phi1 - phi1
    # = 1 + 0.9*0.1 - 0.1 = 0.99 になり、adv0 は 0.099 + 0.9*0.99 = 0.99 になるはず。
    # 実測(0.9)とはっきり異なることを確認しておく(このテストが本当に検出力を持つことの確認)。
    buggy_r1 = R + gamma * phi1 - phi1
    assert abs(buggy_r1 - expected_adv1) > 0.05, "gamma!=1 でバグ版と正しい版が区別できていない"
    print("検証1 PASS: Φ(terminal)≡0 が手計算と厳密一致。誤実装(Φ(terminal)=Φ(s_last))とは"
          f"検出可能な差({buggy_r1:.4f} vs {expected_adv1:.4f})")

    # ------------------------------------------------------------------
    # 検証2: gamma=1・Φ(s_0)=0(ゲーム開始状態)でシェーピング項の和が厳密にゼロへ telescope
    # ------------------------------------------------------------------
    print("\n--- 検証2: gamma=1・Φ(s_0)=0 で adv[0] == R (シェーピング項の和=0) ---")
    import random
    rng = random.Random(0)
    for trial in range(10):
        L = rng.randint(2, 6)
        rows = [_row(6, 6)]  # s_0 = ゲーム開始、Φ=0
        for _ in range(L - 1):
            sp = rng.randint(0, 6)
            op = rng.randint(0, 6)
            rows.append(_row(sp, op))
        state_rows = torch.tensor(rows, dtype=torch.float32)
        values = torch.zeros(L)
        R = float(rng.choice([0.0, 1.0]))
        adv, _ = compute_gae([L], [R], values, 1.0, 1.0, "cpu",
                             state_rows=state_rows, shaping_c=C)
        print(f"  trial{trial}: L={L} R={R} adv[0]={adv[0].item():.9f}")
        assert abs(adv[0].item() - R) < 1e-5, (
            f"trial{trial}: adv[0]={adv[0].item()} != R={R} "
            "(shaping の和がゼロへtelescopeしていない)"
        )
    print("検証2 PASS: 全trialで adv[0] == R (シェーピングは全体の収益を変えない)")

    # ------------------------------------------------------------------
    # 検証3: 一般の Φ(s_0)!=0 でも adv[0] == R - Φ(s_0)(telescopingの代数的恒等式そのもの)
    # ------------------------------------------------------------------
    print("\n--- 検証3: 一般の Φ(s_0) で adv[0] == R - Φ(s_0) ---")
    for trial in range(10):
        L = rng.randint(2, 6)
        rows = []
        for _ in range(L):
            sp = rng.randint(0, 6)
            op = rng.randint(0, 6)
            rows.append(_row(sp, op))
        state_rows = torch.tensor(rows, dtype=torch.float32)
        values = torch.zeros(L)
        R = float(rng.choice([0.0, 1.0]))
        adv, _ = compute_gae([L], [R], values, 1.0, 1.0, "cpu",
                             state_rows=state_rows, shaping_c=C)
        phi_s0 = prize_margin_potential(rows[0], C)
        expected = R - phi_s0
        assert abs(adv[0].item() - expected) < 1e-5, (
            f"trial{trial}: adv[0]={adv[0].item()} != R-Φ(s0)={expected}"
        )
    print("検証3 PASS: adv[0] == R - Φ(s_0) が恒等的に成立(telescopingの代数)")

    # ------------------------------------------------------------------
    # 検証4: shaping_c=0(既定)は旧実装と完全に同じ
    # ------------------------------------------------------------------
    print("\n--- 検証4: shaping_c=0 は旧実装(終局のみ報酬)と完全一致 ---")
    L = 4
    rows = [_row(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(L)]
    state_rows = torch.tensor(rows, dtype=torch.float32)
    values = torch.rand(L)
    R = 1.0
    adv_shaped_off, vt_shaped_off = compute_gae([L], [R], values, 0.999, 0.95, "cpu",
                                                state_rows=state_rows, shaping_c=0.0)
    adv_no_kw, vt_no_kw = compute_gae([L], [R], values, 0.999, 0.95, "cpu")  # 新引数を渡さない
    assert torch.allclose(adv_shaped_off, adv_no_kw), "shaping_c=0 なのに旧来の呼び方と結果が違う"
    assert torch.allclose(vt_shaped_off, vt_no_kw)
    print("検証4 PASS: shaping_c=0 は新引数を渡さない旧来の呼び方とビット単位で一致")

    # ------------------------------------------------------------------
    # 検証5: prize_margin_potential を「終端状態っぽい特徴」に直接呼ぶと一般に0でない
    # ------------------------------------------------------------------
    print("\n--- 検証5: 終端状態に prize_margin_potential を素朴に適用すると0にならない例 ---")
    prize_out_win = _row(0, 3)     # サイド取り切り勝ち(自分残0・相手残3)
    prize_out_loss = _row(3, 0)    # サイド取り切り負け
    deckout_loss = _row(2, 4)      # 山札切れ負け(サイドは半端に残る)
    for name, row in [("prize_out_win", prize_out_win), ("prize_out_loss", prize_out_loss),
                      ("deckout_loss", deckout_loss)]:
        phi = prize_margin_potential(row, C)
        print(f"  {name}: naive Φ = {phi:+.3f} (0ではない)")
    assert abs(prize_margin_potential(prize_out_win, C)) > 1e-9
    assert abs(prize_margin_potential(prize_out_loss, C)) > 1e-9
    print("検証5 PASS: これらの値を compute_gae は一切参照しない(terminal は常に定数0.0)。"
          "だからこそ検証1〜3が意味を持つ。")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
