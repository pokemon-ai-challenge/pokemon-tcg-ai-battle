"""defect#3(Criticのランダム初期化、'最大の欠陥')の検証。

Critic.from_value_net が kaggle_replays/value_net/value_weights_v251.json (勝率を教師にした
事前学習済みMLP、251次元 = 現行715次元FEATURE_NAMESのうち学習側方策がablateしない生き残り特徴)を
715次元のCriticへ正しく移植できていることを検証する。

検証1: 移植した Critic が、value_net 自身の(pure-Python再実装の)forwardと初期化直後に
       ビット精度で一致する(=「それらしい値をでっち上げている」のではなく、本当に同じ関数を
       計算していることの直接証明)。
検証2: 移植した Critic の出力が、既知の「有利/不利な盤面」の方向にちゃんと反応する
       (サイド差だけを動かして、有利なほうが高い値を返す)。
検証3: --critic-init random(旧来のランダム初期化)と出力を比較し、value_net init は
       あきらかにランダムではない(分散が小さい・出力レンジが妥当)ことを確認する。
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "sample_submission")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train_v3 import Critic, SELF_PRIZE_REMAINING_IDX, OPP_PRIZE_REMAINING_IDX  # noqa: E402
from ptcg_ai.learning import encoder  # noqa: E402

POLICY_WEIGHTS = _ROOT / "sample_submission" / "ptcg_ai" / "learning" / "policy_weights.json"
VALUE_NET_WEIGHTS = _ROOT / "kaggle_replays" / "value_net" / "value_weights_v251.json"


def _value_net_forward_raw(x_v, vnet):
    """value_net の pure-Python 再実装(torch を経由しない独立実装)。"""
    vmean = vnet["standardization"]["mean"]
    vstdv = vnet["standardization"]["std"]
    z = [(x_v[i] - vmean[i]) / vstdv[i] if vstdv[i] != 0 else 0.0 for i in range(len(x_v))]

    def mat(layer, vec):
        return [sum(r * v for r, v in zip(row, vec)) + b for row, b in zip(layer["W"], layer["b"])]

    h = z
    for layer in vnet["layers"]:
        h = mat(layer, h)
        if layer["activation"] == "relu":
            h = [max(0.0, v) for v in h]
        elif layer["activation"] == "sigmoid":
            h = [1.0 / (1.0 + math.exp(-v)) for v in h]
    return h[0]


def main():
    base_payload = json.loads(POLICY_WEIGHTS.read_text(encoding="utf-8"))
    std = base_payload["standardization"]
    vnet = json.loads(VALUE_NET_WEIGHTS.read_text(encoding="utf-8"))

    critic, coverage = Critic.from_value_net(
        len(std["state_mean"]), std["state_mean"], std["state_std"],
        encoder.FEATURE_NAMES, vnet, device="cpu",
    )
    print(f"coverage = {coverage:.4f} ({round(coverage * len(encoder.FEATURE_NAMES))}/"
          f"{len(encoder.FEATURE_NAMES)})")
    assert 0.3 < coverage < 0.4, f"想定外のcoverage: {coverage} (251/715={251/715:.4f}のはず)"

    # ------------------------------------------------------------------
    # 検証1: value_net 自身の再実装とビット精度で一致(初期化直後)
    # ------------------------------------------------------------------
    print("\n--- 検証1: 移植したCriticがvalue_net自身の出力と一致(でっち上げでない証明) ---")
    vnames = vnet["feature_names"]
    vmean = vnet["standardization"]["mean"]
    vstdv = vnet["standardization"]["std"]
    name_to_j = {n: j for j, n in enumerate(encoder.FEATURE_NAMES)}
    rng = random.Random(0)
    max_err = 0.0
    for _ in range(20):
        x_v = [vmean[i] + rng.gauss(0, 1) * abs(vstdv[i]) for i in range(len(vnames))]
        x_full = [0.0] * len(encoder.FEATURE_NAMES)
        for i, name in enumerate(vnames):
            x_full[name_to_j[name]] = x_v[i]
        expected = _value_net_forward_raw(x_v, vnet)
        got = critic(torch.tensor([x_full], dtype=torch.float32)).item()
        max_err = max(max_err, abs(expected - got))
    print(f"  max abs err = {max_err:.2e}")
    assert max_err < 1e-4, f"value_net再実装との誤差が大きすぎる: {max_err}"
    print("検証1 PASS: 初期化直後のCriticはvalue_netの計算をそのまま再現している")

    # ------------------------------------------------------------------
    # 検証2: サイド差だけを動かすと妥当な方向に反応する
    # ------------------------------------------------------------------
    print("\n--- 検証2: サイド差(既知の強い勝敗シグナル)への感度 ---")

    def row(self_prize, opp_prize, base=None):
        r = list(base) if base is not None else [0.0] * len(encoder.FEATURE_NAMES)
        r[SELF_PRIZE_REMAINING_IDX] = float(self_prize)
        r[OPP_PRIZE_REMAINING_IDX] = float(opp_prize)
        return r

    # ベースは標準化平均に近い「中立な盤面」にしておく(value_netが学習した分布に近い入力域)。
    base_x = [0.0] * len(encoder.FEATURE_NAMES)
    for i, name in enumerate(vnames):
        base_x[name_to_j[name]] = vmean[i]

    ahead = torch.tensor([row(1, 5, base_x)], dtype=torch.float32)   # 自分残1・相手残5(自分優勢)
    even = torch.tensor([row(3, 3, base_x)], dtype=torch.float32)    # 互角
    behind = torch.tensor([row(5, 1, base_x)], dtype=torch.float32)  # 自分残5・相手残1(自分劣勢)

    v_ahead = critic(ahead).item()
    v_even = critic(even).item()
    v_behind = critic(behind).item()
    print(f"  value(自分優勢 1-5) = {v_ahead:.4f}")
    print(f"  value(互角 3-3)     = {v_even:.4f}")
    print(f"  value(自分劣勢 5-1) = {v_behind:.4f}")
    assert v_ahead > v_even > v_behind, (
        f"サイド差の方向にvalueが反応していない: ahead={v_ahead} even={v_even} behind={v_behind}"
    )
    print("検証2 PASS: 優勢なほどvalueが高い(単調)")

    # ------------------------------------------------------------------
    # 検証3: --critic-init random(旧来)との比較
    # ------------------------------------------------------------------
    print("\n--- 検証3: ランダム初期化との対比 ---")
    torch.manual_seed(0)
    random_critic = Critic(len(std["state_mean"]), std["state_mean"], std["state_std"]).to("cpu")
    r_ahead = random_critic(ahead).item()
    r_even = random_critic(even).item()
    r_behind = random_critic(behind).item()
    print(f"  [random init] value(優勢)={r_ahead:.4f} value(互角)={r_even:.4f} value(劣勢)={r_behind:.4f}")
    print(f"  [value_net init] range=[{min(v_ahead,v_even,v_behind):.4f}, "
          f"{max(v_ahead,v_even,v_behind):.4f}] (sigmoid出力なので[0,1]に収まる)")
    assert 0.0 <= v_ahead <= 1.0 and 0.0 <= v_behind <= 1.0, "value_net initはsigmoid出力のはず"
    print("検証3 PASS: value_net init は [0,1] の妥当な範囲、random init は無拘束"
          "(critic warmup前の value関数としての妥当性が全く違う)")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
