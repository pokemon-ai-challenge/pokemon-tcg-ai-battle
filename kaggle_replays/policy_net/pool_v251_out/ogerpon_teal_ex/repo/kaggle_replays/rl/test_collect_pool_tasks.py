"""collect_pool.build_tasks の単体テスト(cg 非依存、試合は1回も回さない)。

検証1: 端数の丸め(per, dropped_games)が期待通り
検証2: per が必ず偶数
検証3: 相手ごとの席(learner_index)が均等
検証4: 相手と席が同期していないこと(同時分布の回帰テスト。過去に踏んだバグ)
検証5: シードが全タスクで一意
検証6: シードが相手間で衝突しない
検証7: per >= SEED_STRIDE で ValueError
検証8: k=0 で ValueError
検証9: n_games < k で per=0, tasks==[], dropped=n_games
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from collect_pool import SEED_STRIDE, build_tasks  # noqa: E402


def main():
    # ------------------------------------------------------------------
    # 検証1: 端数の丸め(期待値は自分で計算し直したもの)
    # ------------------------------------------------------------------
    print("--- 検証1: 端数の丸め ---")
    cases = [
        # (k, n_games, expected_per, expected_dropped)
        (4, 50, 12, 2),   # 50//4=12(偶数のまま), dropped=50-48=2
        (4, 44, 10, 4),   # 44//4=11(奇数)->10, dropped=44-40=4
        (3, 20, 6, 2),    # 20//3=6(偶数のまま), dropped=20-18=2
        (3, 22, 6, 4),    # 22//3=7(奇数)->6, dropped=22-18=4
    ]
    for k, n_games, exp_per, exp_dropped in cases:
        tasks, per, dropped = build_tasks(k, n_games, seed0=1000)
        print(f"  k={k} n_games={n_games} -> per={per} dropped={dropped} "
              f"(expected per={exp_per} dropped={exp_dropped})")
        assert per == exp_per, f"k={k} n_games={n_games}: per={per} != {exp_per}"
        assert dropped == exp_dropped, f"k={k} n_games={n_games}: dropped={dropped} != {exp_dropped}"
        assert len(tasks) == per * k, f"len(tasks)={len(tasks)} != per*k={per * k}"
    print("検証1 PASS")

    # ------------------------------------------------------------------
    # 検証2: per が必ず偶数
    # ------------------------------------------------------------------
    print("\n--- 検証2: per が必ず偶数 ---")
    for k, n_games, _, _ in cases:
        _, per, _ = build_tasks(k, n_games, seed0=1000)
        assert per % 2 == 0, f"k={k} n_games={n_games}: per={per} が奇数"
    print("検証2 PASS")

    # ------------------------------------------------------------------
    # 検証3: 相手ごとの席均等
    # ------------------------------------------------------------------
    print("\n--- 検証3: 相手ごとの席均等 ---")
    k, n_games = 4, 50
    tasks, per, dropped = build_tasks(k, n_games, seed0=2000)
    for opp_idx in range(k):
        seats = [learner_index for (i, learner_index, seed) in tasks if i == opp_idx]
        n0 = seats.count(0)
        n1 = seats.count(1)
        print(f"  opp_idx={opp_idx}: seat0={n0} seat1={n1}")
        assert n0 == n1, f"opp_idx={opp_idx}: seat0={n0} != seat1={n1}"
        assert n0 + n1 == per
    print("検証3 PASS")

    # ------------------------------------------------------------------
    # 検証4: 相手と席が同期していないこと(同時分布の回帰テスト)
    # ------------------------------------------------------------------
    print("\n--- 検証4: (opp_idx, learner_index) の同時分布 ---")
    joint = Counter((i, learner_index) for (i, learner_index, seed) in tasks)
    print(f"  joint distribution: {dict(sorted(joint.items()))}")
    values = set(joint.values())
    assert len(values) == 1, (
        f"(opp_idx, learner_index) の同時分布が全組み合わせで一致していない: {joint} "
        "(相手と席が同期しているバグの疑い)"
    )
    # 全組み合わせ(k * 2 種類)が存在すること
    assert len(joint) == k * 2, f"組み合わせ数={len(joint)} != k*2={k * 2}: {joint}"
    print("検証4 PASS: 全 (opp_idx, learner_index) 組み合わせの件数が一致")

    # ------------------------------------------------------------------
    # 検証5: シードが全タスクで一意
    # ------------------------------------------------------------------
    print("\n--- 検証5: シードの一意性 ---")
    seeds = [seed for (i, learner_index, seed) in tasks]
    print(f"  len(tasks)={len(tasks)} len(set(seeds))={len(set(seeds))}")
    assert len(set(seeds)) == len(tasks), "シードに重複がある"
    print("検証5 PASS")

    # ------------------------------------------------------------------
    # 検証6: シードが相手間で衝突しない
    # ------------------------------------------------------------------
    print("\n--- 検証6: シードが相手間で衝突しない ---")
    seed_sets = [set(seed for (i, learner_index, seed) in tasks if i == opp_idx)
                 for opp_idx in range(k)]
    for a in range(k):
        for b in range(a + 1, k):
            common = seed_sets[a] & seed_sets[b]
            assert not common, f"opp_idx={a} と opp_idx={b} のシードが衝突: {common}"
    print("検証6 PASS: 相手間でシード集合が互いに素")

    # ------------------------------------------------------------------
    # 検証7: per >= SEED_STRIDE で ValueError
    # ------------------------------------------------------------------
    print("\n--- 検証7: per >= SEED_STRIDE で ValueError ---")
    try:
        build_tasks(1, 200_000, seed0=0)
        raise AssertionError("ValueError が発生しなかった(k=1, n_games=200000)")
    except ValueError as exc:
        print(f"  期待通り ValueError: {exc}")
    print("検証7 PASS")

    # ------------------------------------------------------------------
    # 検証8: k=0 で ValueError
    # ------------------------------------------------------------------
    print("\n--- 検証8: k=0 で ValueError ---")
    try:
        build_tasks(0, 10, seed0=0)
        raise AssertionError("ValueError が発生しなかった(k=0)")
    except ValueError as exc:
        print(f"  期待通り ValueError: {exc}")
    print("検証8 PASS")

    # ------------------------------------------------------------------
    # 検証9: n_games < k で per=0, tasks==[], dropped=n_games
    # ------------------------------------------------------------------
    print("\n--- 検証9: n_games < k ---")
    tasks9, per9, dropped9 = build_tasks(4, 2, seed0=0)
    print(f"  k=4 n_games=2 -> tasks={tasks9} per={per9} dropped={dropped9}")
    assert per9 == 0, f"per={per9} != 0"
    assert tasks9 == [], f"tasks={tasks9} != []"
    assert dropped9 == 2, f"dropped={dropped9} != 2"
    print("検証9 PASS")

    print("\n全検証 PASS")


if __name__ == "__main__":
    main()
