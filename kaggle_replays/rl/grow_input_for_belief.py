"""既存 policy_weights JSON の入力次元を belief 特徴ぶん成長させる(belief-conditioned RL の起点作り)。

state 特徴は encode_state_from_state で base の**直後**に extra_features を連結する(= state末尾)。
なので入力レイアウトは [state_base, belief, option, embed]。第1層 weight の index=base_state に
belief_dim 列を **0 で挿入**する(=belief の初期寄与0 → 既存 climb の方策を厳密に保持、RL が使い方を学ぶ)。
standardization は belief を mean0/std1(生の確率)で扱う。meta に extra_features=opp_belief を立てる。

使い方: python kaggle_replays/rl/grow_input_for_belief.py <in.json> <out.json> [belief_dim=21]
"""

from __future__ import annotations

import json
import sys


def grow(in_path: str, out_path: str, belief_dim: int = 21, flag: str = "opp_belief") -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    std = d["standardization"]
    base_state = len(std["state_mean"])  # climb = 166
    assert base_state == len(std["state_std"]), "state_mean/std 長不一致"

    # standardization を belief ぶんパッド(生の確率をそのまま使う)。
    std["state_mean"] = list(std["state_mean"]) + [0.0] * belief_dim
    std["state_std"] = list(std["state_std"]) + [1.0] * belief_dim

    # 第1層 weight [hidden][in_dim] の index=base_state に belief_dim 列を 0 挿入。
    W0 = d["layers"][0]["weight"]
    old_in = len(W0[0])
    for row in W0:
        for _ in range(belief_dim):
            row.insert(base_state, 0.0)
    new_in = len(W0[0])
    assert new_in == old_in + belief_dim, "第1層 in_dim 成長不一致"

    d["meta"]["extra_features"] = flag
    d["meta"]["state_feature_count"] = base_state + belief_dim
    d["meta"]["belief_dim"] = belief_dim
    d["meta"]["grown_from"] = in_path.replace("\\", "/").split("/")[-1]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f)
    print(f"grown: state {base_state} -> {base_state + belief_dim}, layer0 in_dim {old_in} -> {new_in}, "
          f"belief 列を index {base_state} に0挿入。extra_features=opp_belief。-> {out_path}")


if __name__ == "__main__":
    a = sys.argv
    if len(a) < 3:
        print("usage: grow_input_for_belief.py <in.json> <out.json> [dim] [flag]"); sys.exit(1)
    grow(a[1], a[2], int(a[3]) if len(a) > 3 else 21, a[4] if len(a) > 4 else "opp_belief")
